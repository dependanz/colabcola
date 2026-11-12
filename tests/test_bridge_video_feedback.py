"""v24 deterministic feedback keyframe recovery + virtual playout tests.

Tests without codec dependencies cover timing boundaries and frame accounting.
Optional PyAV tests validate REAL H.264 IDRs and decoded reference chains.
"""
from __future__ import annotations

import pytest

from colabcola.bridge_video_stream import VideoAccessUnit
from colabcola.bridge_video_impairment import (
    LinkConfig, LinkTrace, ScheduledUnit, simulate_link, decode_delivered_h264,
)
from colabcola.bridge_video_feedback import (
    _decoded_indices, _parse_buffer_list, compare_recovery_playout,
    encode_with_forced_idr, feedback_idr_plan, first_gap_detection,
    fixed_deadline_playout,
)


def _fake_trace(*, fps=10, n=6, gap_at=None, lateness=0.0):
    items = []
    for s in range(n):
        if s == gap_at:
            continue
        source = s * 1000.0 / fps
        packet = VideoAccessUnit(s, s * 100_000_000, s * 100_000_000,
                                 b"not-a-codec-payload", s == 0)
        items.append(ScheduledUnit(packet, source, source + lateness, 50))
    return LinkTrace(n, tuple(items),
                     ((gap_at, "erased"),) if gap_at is not None else (),
                     50 * len(items), len(items))


def test_gap_detection_waits_for_newer_arrival_not_knowledge_of_loss():
    trace = _fake_trace(fps=10, n=6, gap_at=2, lateness=20)
    missing, detection = first_gap_detection(trace)
    assert missing == 2
    assert detection == pytest.approx(320)  # frame 3 reaches receiver
    plan = feedback_idr_plan(trace, fps=10, frames=6, feedback_ms=10)
    assert plan["missing_seq_detected"] == 2
    assert plan["detected_at_ms"] == pytest.approx(320)
    assert plan["feedback_reaches_source_ms"] == pytest.approx(330)
    assert plan["forced_idr_seq"] == 4

    no_loss = feedback_idr_plan(
        _fake_trace(fps=10, n=6, gap_at=None), fps=10, frames=6,
        feedback_ms=10)
    assert no_loss["forced_idr_seq"] is None


def test_first_gap_at_zero_or_undetectable_last_frame():
    no_initial = _fake_trace(fps=10, n=6, gap_at=0, lateness=40)
    assert first_gap_detection(no_initial) == (0, 140)
    lost_last = _fake_trace(fps=10, n=6, gap_at=5, lateness=40)
    assert first_gap_detection(lost_last) is None


def test_feedback_must_wait_until_next_source_frame_boundary():
    trace = _fake_trace(fps=10, n=6, gap_at=1, lateness=0)
    # Detection at t=200; a zero-latency request still cannot
    # retroactively alter frame 2, already scheduled at t=200.
    result = feedback_idr_plan(trace, fps=10, frames=6, feedback_ms=0)
    assert result["forced_idr_seq"] == 3
    # Feedback that arrives after the last source frame cannot recover.
    late = feedback_idr_plan(trace, fps=10, frames=6, feedback_ms=1000)
    assert late["forced_idr_seq"] is None


def test_feedback_and_buffer_bounds_reject_invalid_config():
    trace = _fake_trace()
    for bad in (-1, 1001, float("nan"), True):
        with pytest.raises(ValueError):
            feedback_idr_plan(trace, fps=10, frames=6, feedback_ms=bad)
    with pytest.raises(ValueError):
        feedback_idr_plan(trace, fps=0, frames=6, feedback_ms=0)
    for bad in ("", "80,40", "40,40", "x", "10,1001", "1,2,3,4,5,6,7,8,9"):
        with pytest.raises(ValueError):
            _parse_buffer_list(bad)
    assert _parse_buffer_list("0,40,80,120") == (0, 40, 80, 120)


def test_virtual_playout_trades_freshness_for_intentional_delay():
    trace = _fake_trace(fps=10, n=6, gap_at=None, lateness=65)
    observed = {"decoded_runs": [[0, 5]], "decoded_frames": 6}
    early = fixed_deadline_playout(trace, observed, fps=10, buffer_ms=40)
    late = fixed_deadline_playout(trace, observed, fps=10, buffer_ms=80)
    assert early["on_time_frames"] == 0
    assert early["late_decoded_frames"] == 6
    assert early["initial_blank_slots"] == 6
    assert early["longest_unfresh_run_ms"] == pytest.approx(600)
    assert late["on_time_frames"] == 6
    assert late["late_decoded_frames"] == 0
    assert late["presented_sequence_runs"] == [[0, 5]]
    assert late["longest_unfresh_run_ms"] == 0


def test_virtual_playout_never_presents_missing_reference_dependents():
    trace = _fake_trace(fps=10, n=6, gap_at=2, lateness=20)
    observed = {"decoded_runs": [[0, 1], [4, 5]], "decoded_frames": 4}
    report = fixed_deadline_playout(trace, observed, fps=10, buffer_ms=40)
    assert report["on_time_frames"] == 4
    assert report["undecodable_frames"] == 2
    assert report["held_slots"] == 2
    assert report["longest_unfresh_run_slots"] == 2
    assert report["presented_sequence_runs"] == [[0, 1], [4, 5]]


def test_playout_rejects_decoder_run_mismatch_or_future_trace():
    trace = _fake_trace()
    with pytest.raises(ValueError):
        fixed_deadline_playout(trace,
                               {"decoded_runs": [[0, 3]], "decoded_frames": 5},
                               fps=10, buffer_ms=40)
    with pytest.raises(ValueError):
        fixed_deadline_playout(trace, {"decoded_runs": [[0, 5]],
                                       "decoded_frames": 6},
                               fps=10, buffer_ms=-2)
    for bad in ([40, 80], (80, 40), (40, 40), (-1,)):
        if isinstance(bad, list):
            with pytest.raises(ValueError):
                compare_recovery_playout(width=64, height=48, fps=10,
                                         frames=6, gop=4, buffers_ms=bad)
        else:
            # Validate the buffer list without loading a codec.
            from colabcola.bridge_video_feedback import _parse_buffer_list
            with pytest.raises(ValueError):
                _parse_buffer_list(",".join(str(x) for x in bad))


def _need_codec():
    av = pytest.importorskip("av", reason="optional PyAV not installed")
    pytest.importorskip("numpy")
    try:
        av.CodecContext.create("libx264", "w")
    except Exception:
        pytest.skip("PyAV build does not expose libx264")


def test_forced_encoder_restart_emits_real_idr_and_decodes_all_intact_frames():
    _need_codec()
    from colabcola.bridge_video_stream import is_keyframe
    packets = encode_with_forced_idr(
        width=96, height=64, fps=30, frames=36, gop=12, forced_seq=8)
    assert len(packets) == 36
    assert packets[0].keyframe
    assert packets[8].keyframe
    assert is_keyframe(packets[8].data)
    config = LinkConfig(delay_ms=30, jitter_ms=0,
                        bandwidth_kbps=0, max_age_ms=0, max_inflight=16)
    trace = simulate_link(packets, 30, config)
    decoded = decode_delivered_h264(trace, 96, 64)
    assert decoded["decoded_frames"] == 36
    assert decoded["decoded_runs"] == [[0, 35]]


def test_real_h264_feedback_restores_reference_chain_before_periodic_idr():
    _need_codec()
    config = LinkConfig(delay_ms=40, jitter_ms=0,
                        bandwidth_kbps=0, max_age_ms=0,
                        max_inflight=16, drop_seqs=(5,))
    report = compare_recovery_playout(
        width=96, height=64, fps=30, frames=36,
        gop=12, feedback_ms=20, link=config,
        buffers_ms=(40, 80, 120),
    )
    assert report["feedback"]["missing_seq_detected"] == 5
    assert report["feedback"]["forced_idr_seq"] == 8
    base, fb = report["periodic"], report["feedback_idr"]
    assert base["decoded_runs"] == [[0, 4], [12, 35]]
    assert fb["decoded_runs"] == [[0, 4], [8, 35]]
    assert base["actually_decoded_frames"] == 29
    assert fb["actually_decoded_frames"] == 33
    assert fb["decoder_idr_resynchronizations"] == 1
    assert fb["encoded_idrs"] >= base["encoded_idrs"]
    assert fb["wire_bytes"] > 0
    assert fb["playout"][-1]["on_time_frames"] == 33


def test_no_gap_yields_identical_real_h264_stream_in_both_policies():
    _need_codec()
    report = compare_recovery_playout(
        width=96, height=64, fps=24, frames=24,
        gop=12, feedback_ms=20,
        link=LinkConfig(delay_ms=40, jitter_ms=0,
                        bandwidth_kbps=0, max_age_ms=0),
        buffers_ms=(40, 80),
    )
    assert report["feedback"]["forced_idr_seq"] is None
    assert report["periodic"] == report["feedback_idr"]


def test_bandwidth_jitter_playout_comparison_is_monotonic_for_fixed_trace():
    _need_codec()
    config = LinkConfig(
        delay_ms=50, jitter_ms=20,
        bandwidth_steps=((0, 2500), (300, 130), (850, 2000)),
        max_inflight=8, max_age_ms=250, seed=7)
    report = compare_recovery_playout(
        width=128, height=72, fps=30, frames=48,
        gop=12, feedback_ms=20, link=config,
        buffers_ms=(40, 80, 120, 180, 250),
    )
    play = report["periodic"]["playout"]
    on_time = [r["on_time_frames"] for r in play]
    assert on_time == sorted(on_time)
    assert any(x < y for x, y in zip(on_time, on_time[1:]))
    assert all(r["late_decoded_frames"] + r["on_time_frames"]
               + r["undecodable_frames"] == 48 for r in play)
