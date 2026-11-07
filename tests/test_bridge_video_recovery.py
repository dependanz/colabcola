"""v28 independent decoder recovery, using real codecs and virtual time."""
from __future__ import annotations

from dataclasses import replace
import json

import pytest

from colabcola import bridge_video_rate as rate_module
from colabcola.bridge_video_controller import ControllerConfig
from colabcola.bridge_video_impairment import LinkConfig, decode_delivered_h264
from colabcola.bridge_video_rate import RateConfig, stream_with_rate_control
from colabcola.bridge_video_recovery import compare_rate_recovery


ARMS = ("fixed_high_rate", "watchdog_only", "watchdog_with_recovery",
        "matched_restart_high_rate")


@pytest.fixture
def codec():
    av = pytest.importorskip("av", reason="optional PyAV not installed")
    pytest.importorskip("numpy", reason="NumPy not installed")
    try:
        av.CodecContext.create("libx264", "w")
    except Exception:
        pytest.skip("PyAV build does not include libx264")


def _link(**changes):
    return replace(LinkConfig(delay_ms=40, jitter_ms=0, bandwidth_kbps=0,
                              max_age_ms=0, max_inflight=16), **changes)


def _rates(**changes):
    return replace(RateConfig(control_delay_ms=20, recovered_age_ms=0,
                              min_switch_ms=0), **changes)


def _stream(*, link=None, rates=None, **changes):
    args = dict(width=96, height=64, fps=30, frames=60, gop=24,
                link=_link() if link is None else link,
                rates=_rates() if rates is None else rates, adapt=True)
    args.update(changes)
    return stream_with_rate_control(**args)


def _decoded_seqs(arm):
    return {seq for start, end in arm["decoded_runs"]
            for seq in range(start, end + 1)}


def test_configuration_rejects_invalid_recovery_and_combined_budget():
    for invalid in (False, 1, {}, "recovery"):
        with pytest.raises(ValueError, match="recovery"):
            _stream(recovery=invalid)
        with pytest.raises(ValueError, match="configuration"):
            compare_rate_recovery(recovery=invalid)
    with pytest.raises(ValueError, match="budgets"):
        _stream(rates=_rates(max_switches=13),
                recovery=ControllerConfig(max_requests=8))
    with pytest.raises(ValueError, match="budgets"):
        compare_rate_recovery(rates=_rates(max_switches=13),
                              recovery=ControllerConfig(max_requests=8))
    with pytest.raises(ValueError, match="budgets"):
        _stream(adapt=False, recovery=ControllerConfig(max_requests=21))


def test_offline_restart_schedule_cannot_be_combined_with_live_recovery():
    for adapt in (True, False):
        with pytest.raises(ValueError, match="restart_seqs"):
            _stream(adapt=adapt, restart_seqs=(8,),
                    recovery=ControllerConfig(max_requests=0))


def test_disabled_recovery_preserves_rate_stream_and_full_trace(codec):
    args = dict(link=_link(drop_seqs=(5, 8, 27)),
                rates=_rates(silence_timeout_ms=150))
    before_packets, before_trace, before_info = _stream(**args)
    after_packets, after_trace, after_info = _stream(
        **args, recovery=ControllerConfig(max_requests=0))
    assert after_packets == before_packets
    assert after_trace == before_trace
    assert {k: v for k, v in after_info.items()
            if k not in ("recovery", "encoder_restarts")} == before_info
    assert after_info["recovery"]["requests_sent"] == 0
    assert [x["source_seq"] for x in after_info["encoder_restarts"]] == [
        x["source_seq"] for x in before_info["applied"]]


def test_clean_link_needs_no_recovery_and_all_four_arms_match(codec):
    report = compare_rate_recovery(
        width=96, height=64, frames=36, link=_link(delay_ms=15))
    controller = report["controllers"]["watchdog_with_recovery"]
    assert controller["encoder_restarts"] == []
    assert controller["recovery"]["requests_sent"] == 0
    assert report["first_policy_difference_seq"] is None
    assert all(report[name] == report[ARMS[0]] for name in ARMS)
    assert report["watchdog_with_recovery"]["actually_decoded_frames"] == 36


def test_lost_recovery_idr_retries_at_low_and_coalesces_the_first_change(codec, monkeypatch):
    created_targets = []
    original_encoder = rate_module._encoder

    def counted_encoder(*args):
        created_targets.append(args[-1])
        return original_encoder(*args)

    monkeypatch.setattr(rate_module, "_encoder", counted_encoder)
    packets, trace, info = _stream(
        link=_link(drop_seqs=(5, 8, 27)),
        recovery=ControllerConfig(feedback_ms=20, retry_ms=80, max_requests=5))
    assert created_targets == [650, 95, 95, 95]
    assert info["switches_applied"] == 1
    assert [r["source_seq"] for r in info["encoder_restarts"]] == [8, 11, 30]
    assert info["encoder_restarts"][0]["rate_change"] is True
    assert info["encoder_restarts"][0]["recovery_request_ids"] == [0]
    assert all(not r["rate_change"] and r["previous_kbps"] == r["target_kbps"] == 95
               for r in info["encoder_restarts"][1:])
    assert [r["source_idr_seq"] for r in info["recovery"]["applied"]] == [8, 11, 30]
    assert [p.seq for p in packets if p.keyframe] == [0, 8, 11, 30, 54]
    assert dict(trace.dropped)[8] == "erased"
    decoded = decode_delivered_h264(trace, 96, 64)
    assert decoded["decoded_frames"] == 51
    assert decoded["decoded_runs"] == [[0, 4], [11, 26], [30, 59]]


@pytest.mark.parametrize("feedback_ms, expected_seq", [(0, 4), (40, 5)])
def test_exact_feedback_ties_wait_for_a_later_source_frame(codec, feedback_ms, expected_seq):
    packets, trace, info = _stream(
        fps=25, frames=12, link=_link(drop_seqs=(1,)),
        rates=_rates(max_switches=0),
        recovery=ControllerConfig(feedback_ms=feedback_ms, max_requests=1))
    request = info["recovery"]["requests"][0]
    assert request["detected_at_ms"] == 120
    assert request["reaches_source_ms"] == 120 + feedback_ms
    assert info["encoder_restarts"][0]["source_seq"] == expected_seq
    assert info["encoder_restarts"][0]["source_ms"] > request["reaches_source_ms"]
    assert [p.seq for p in packets if p.keyframe] == [0, expected_seq]
    baseline, baseline_trace, _ = _stream(
        fps=25, frames=12, link=_link(drop_seqs=(1,)), rates=_rates(max_switches=0))
    assert packets[:expected_seq] == baseline[:expected_seq]
    assert tuple(x for x in trace.delivered if x.packet.seq < expected_seq) == tuple(
        x for x in baseline_trace.delivered if x.packet.seq < expected_seq)


def test_feedback_after_source_end_is_recorded_but_cannot_restart(codec):
    args = dict(frames=12, link=_link(drop_seqs=(5,)), rates=_rates(max_switches=0))
    baseline, baseline_trace, _ = _stream(**args)
    packets, trace, info = _stream(
        **args, recovery=ControllerConfig(feedback_ms=2000, max_requests=1))
    assert packets == baseline
    assert trace == baseline_trace
    assert info["encoder_restarts"] == []
    assert info["recovery"]["requests_sent"] == 1
    assert info["recovery"]["requests_not_applied_before_source_end"] == 1


def test_request_budget_stops_retries_after_the_first_recovery_idr_is_lost(codec):
    args = dict(link=_link(drop_seqs=(5, 8, 27)))
    baseline, baseline_trace, _ = _stream(**args)
    packets, trace, info = _stream(
        **args, recovery=ControllerConfig(feedback_ms=20, retry_ms=80, max_requests=1))
    assert info["recovery"]["requests_sent"] == 1
    assert [x["source_idr_seq"] for x in info["recovery"]["applied"]] == [8]
    # The sole recovery request shares the already-required rate change.
    assert packets == baseline
    assert trace == baseline_trace


@pytest.mark.parametrize("keep_initial", [False, True])
def test_complete_media_silence_does_not_invent_recovery_requests(codec, keep_initial):
    report = compare_rate_recovery(
        width=96, height=64, frames=24,
        link=_link(drop_seqs=tuple(range(1 if keep_initial else 0, 24))),
        rates=_rates(silence_timeout_ms=100))
    controller = report["controllers"]["watchdog_with_recovery"]
    assert controller["recovery"]["requests_sent"] == 0
    assert controller["switches_applied"] == int(keep_initial)
    for name in ARMS:
        assert report[name]["actually_decoded_frames"] == int(keep_initial)
        assert report[name]["matched_frame_count"] == int(keep_initial)
    if not keep_initial:
        assert all(report[name]["matched_frame_rgb_psnr_db"] is None for name in ARMS)
    json.dumps(report, allow_nan=False)


def test_collapse_comparison_replays_all_restarts_and_matches_quality_frame_ids(codec):
    report = compare_rate_recovery(link=LinkConfig(
        delay_ms=40, jitter_ms=0, bandwidth_kbps=0,
        bandwidth_steps=((0, 1200), (600, 90), (1750, 1200)),
        max_inflight=10, max_age_ms=350, seed=7))
    controller = report["controllers"]["watchdog_with_recovery"]
    combined = report["watchdog_with_recovery"]
    replay = report["matched_restart_high_rate"]
    assert any(not x["rate_change"] and x["target_kbps"] == report["rates"]["low_kbps"]
               for x in controller["encoder_restarts"])
    assert report["restart_control"]["restart_seqs"] == [
        x["source_seq"] for x in controller["encoder_restarts"]]
    assert combined["encoded_idr_seqs"] == replay["encoded_idr_seqs"]
    assert combined["actually_decoded_frames"] > report["watchdog_only"]["actually_decoded_frames"]
    common = set.intersection(*(_decoded_seqs(report[name]) for name in ARMS))
    assert report["quality_comparison"]["common_frame_seqs"] == sorted(common)
    for name in ARMS:
        arm = report[name]
        assert arm["delivered_frames"] + arm["link_drops"] == 90
        assert arm["quality_decoded_frames"] == arm["actually_decoded_frames"]
        assert arm["matched_frame_count"] == len(common)
    json.dumps(report, allow_nan=False)
