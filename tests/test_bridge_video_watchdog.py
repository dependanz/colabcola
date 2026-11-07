"""v27 receiver-silence timing, causal H.264 changes, and restart ablation.

All link/feedback times are virtual. Codec integration uses optional PyAV;
no network connection, GUI, or cloud compute is needed.
"""
from __future__ import annotations

from dataclasses import replace
import math

import pytest

from colabcola.bridge_video_impairment import LinkConfig, ScheduledUnit
from colabcola.bridge_video_rate import RateConfig, RateReceiver, stream_with_rate_control
from colabcola.bridge_video_stream import VideoAccessUnit
from colabcola.bridge_video_watchdog import compare_rate_watchdog


ARMS = (
    "fixed_high_rate", "arrival_feedback", "silence_feedback",
    "matched_restart_high_rate",
)


def _arrival(seq, arrival_ms, *, source_ms=None, fps=30):
    source_ms = seq * 1000.0 / fps if source_ms is None else source_ms
    stamp = round(source_ms * 1_000_000)
    return ScheduledUnit(
        VideoAccessUnit(seq, stamp, stamp, b"test", seq == 0),
        source_ms, arrival_ms, 40,
    )


def _need_pyav():
    av = pytest.importorskip("av", reason="optional PyAV not installed")
    pytest.importorskip("numpy", reason="NumPy not installed")
    try:
        av.CodecContext.create("libx264", "w")
    except Exception:
        pytest.skip("PyAV build does not include libx264")


def _clean_link(**options):
    return LinkConfig(delay_ms=15, jitter_ms=0, bandwidth_kbps=0,
                      max_age_ms=0, max_inflight=16, **options)


def test_silence_timeout_requires_a_finite_bounded_number():
    for timeout in (-1, 5000.01, math.nan, math.inf, True, "100"):
        with pytest.raises(ValueError):
            RateConfig(silence_timeout_ms=timeout)
    assert RateConfig().silence_timeout_ms == 0
    assert RateConfig(silence_timeout_ms=5000).silence_timeout_ms == 5000


def test_watchdog_is_disabled_by_default_and_arms_only_after_first_arrival():
    disabled = RateReceiver(RateConfig())
    disabled.observe(_arrival(0, 40))
    disabled.advance(10_000)
    assert disabled.requests == []
    assert disabled.report()["silence_timeouts"] == []

    receiver = RateReceiver(RateConfig(
        silence_timeout_ms=100, congested_age_ms=2000, min_switch_ms=0))
    receiver.advance(1000)
    assert receiver.requests == []  # No invented startup observation.
    receiver.observe(_arrival(0, 1000))
    receiver.advance(1099)
    assert receiver.requests == []
    receiver.advance(1100)
    assert receiver.requests[0]["generated_at_ms"] == 1100
    assert receiver.requests[0]["observed_seq"] == 0


def test_arrival_wins_timeout_tie_and_feedback_waits_for_a_later_source_frame():
    cfg = RateConfig(silence_timeout_ms=100, control_delay_ms=20,
                     congested_age_ms=2000, min_switch_ms=0)
    receiver = RateReceiver(cfg)
    receiver.observe(_arrival(0, 40))
    receiver.advance(139.999)
    assert receiver.requests == []
    receiver.observe(_arrival(1, 140))  # Exactly the old timeout deadline.
    receiver.advance(239.999)
    assert receiver.requests == []
    receiver.advance(240)
    request = receiver.requests[0]
    assert request["trigger"] == "receiver_silence"
    assert request["generated_at_ms"] == 240
    assert request["reaches_source_ms"] == 260
    assert request["observed_seq"] == 1
    assert request["observed_age_ms"] is None
    assert request["silence_ms"] == 100
    assert receiver.apply_due(7, 260, cfg.high_kbps) == cfg.high_kbps
    assert receiver.apply_due(8, 8000 / 30, cfg.high_kbps) == cfg.low_kbps
    assert receiver.applied[0]["source_seq"] == 8


def _receiver_with_recent_recovery():
    receiver = RateReceiver(RateConfig(
        silence_timeout_ms=150, min_switch_ms=200, control_delay_ms=10,
        congested_age_ms=100, recovered_age_ms=50,
        congestion_samples=1, recovery_samples=2))
    # Packet age initiates LOW; sustained fresh arrivals later request HIGH.
    for seq, at in enumerate((100, 140, 240, 340)):
        receiver.observe(_arrival(seq, at, fps=10))
    assert [r["target_kbps"] for r in receiver.requests] == [95, 650]
    assert receiver.requests[-1]["generated_at_ms"] == 340
    return receiver


def test_silence_request_respects_dwell_without_waiting_for_another_packet():
    receiver = _receiver_with_recent_recovery()
    receiver.advance(490)  # Silence expires, but the last request was at 340.
    assert len(receiver.requests) == 2
    receiver.advance(539.999)
    assert len(receiver.requests) == 2
    receiver.advance(540)
    assert len(receiver.requests) == 3
    assert receiver.requests[-1]["trigger"] == "receiver_silence"
    assert receiver.requests[-1]["generated_at_ms"] == 540
    assert receiver.requests[-1]["observed_seq"] == 3


def test_arrival_cancels_a_silence_request_still_waiting_for_dwell():
    receiver = _receiver_with_recent_recovery()
    receiver.advance(500)
    assert len(receiver.requests) == 2
    # This arrival is too old for recovery but still precedes request eligibility.
    receiver.observe(_arrival(4, 530, fps=10))
    receiver.advance(540)
    assert len(receiver.requests) == 2


def test_receiver_rejects_time_travel_after_advancing_without_arrivals():
    receiver = RateReceiver(RateConfig(silence_timeout_ms=100))
    receiver.observe(_arrival(0, 40))
    receiver.advance(80)
    for at in (-1, 79, math.nan, math.inf, True, "90"):
        with pytest.raises(ValueError):
            receiver.advance(at)
    with pytest.raises(ValueError):
        receiver.observe(_arrival(1, 70))  # Newer than last packet, older than clock.
    assert receiver.requests == []
    receiver.observe(_arrival(1, 90))
    assert receiver.report()["packets_observed"] == 2


def test_silence_breaks_a_partial_recovery_streak_even_while_already_low():
    receiver = RateReceiver(RateConfig(
        silence_timeout_ms=100, min_switch_ms=0, recovery_samples=3,
        congestion_samples=1, congested_age_ms=100, recovered_age_ms=50))
    receiver.observe(_arrival(0, 120, source_ms=0))  # Request LOW.
    receiver.observe(_arrival(1, 140, source_ms=100))
    receiver.observe(_arrival(2, 170, source_ms=130))
    assert len(receiver.requests) == 1
    receiver.advance(270)
    # Two fresh observations after silence cannot combine with pre-silence samples.
    receiver.observe(_arrival(3, 340, source_ms=300))
    receiver.observe(_arrival(4, 370, source_ms=330))
    assert len(receiver.requests) == 1
    receiver.observe(_arrival(5, 400, source_ms=360))
    assert [r["target_kbps"] for r in receiver.requests] == [95, 650]
    assert receiver.requests[-1]["generated_at_ms"] == 400


def test_silence_has_bounded_requests_and_stops_with_the_source():
    receiver = RateReceiver(RateConfig(
        silence_timeout_ms=100, min_switch_ms=0, max_switches=1))
    receiver.observe(_arrival(0, 40))
    receiver.advance(140)
    receiver.advance(10_000)
    assert len(receiver.requests) == 1
    assert len(receiver.report()["silence_timeouts"]) == 1

    capped = RateReceiver(RateConfig(silence_timeout_ms=100, max_switches=0))
    capped.observe(_arrival(0, 40))
    capped.advance(10_000)
    assert capped.requests == []

    stopped = RateReceiver(RateConfig(silence_timeout_ms=100))
    stopped.observe(_arrival(0, 40))
    stopped.stop_watchdog()
    stopped.advance(1000)
    stopped.observe(_arrival(1, 1015, source_ms=1000))
    assert stopped.requests == []
    assert stopped.report()["silence_timeouts"] == []


def test_restart_ablation_rejects_noncausal_or_unbounded_schedules():
    args = dict(width=96, height=64, fps=30, frames=36, gop=12,
                link=_clean_link(), rates=RateConfig())
    for schedule in ([3], (0,), (36,), (3, 3), (5, 3), (True,), tuple(range(1, 22))):
        with pytest.raises(ValueError):
            stream_with_rate_control(**args, adapt=False, restart_seqs=schedule)
    with pytest.raises(ValueError):
        stream_with_rate_control(**args, adapt=True, restart_seqs=(3,))


def test_real_h264_watchdog_downshifts_during_blackout_and_replay_matches_idrs():
    _need_pyav()
    rates = RateConfig(
        silence_timeout_ms=120, min_switch_ms=0, control_delay_ms=20,
        congested_age_ms=2000, recovered_age_ms=0, max_switches=2)
    args = dict(width=96, height=64, fps=30, frames=36, gop=24,
                link=_clean_link(drop_seqs=tuple(range(6, 19))))
    packets, trace, controller = stream_with_rate_control(**args, rates=rates, adapt=True)
    arrival_packets, _, arrival_controller = stream_with_rate_control(
        **args, rates=replace(rates, silence_timeout_ms=0), adapt=True)
    request = controller["requests"][0]
    assert request["trigger"] == "receiver_silence"
    assert request["observed_seq"] == 5
    last_arrival = next(x.arrival_ms for x in trace.delivered if x.packet.seq == 5)
    assert request["generated_at_ms"] == pytest.approx(last_arrival + 120, abs=0.001)
    assert controller["applied"][0]["source_seq"] == 10
    assert arrival_controller["applied"][0]["source_seq"] == 21
    resumed = next(x.arrival_ms for x in trace.delivered if x.packet.seq == 19)
    assert controller["applied"][0]["source_ms"] < resumed
    assert packets[:10] == arrival_packets[:10]
    assert packets[10].keyframe
    assert all(r["generated_at_ms"] <= 35_000 / 30
               for r in controller["requests"] if r["trigger"] == "receiver_silence")

    restarts = tuple(a["source_seq"] for a in controller["applied"])
    replay, _, _ = stream_with_rate_control(
        **args, rates=rates, adapt=False, restart_seqs=restarts)
    assert [p.seq for p in replay if p.keyframe] == [p.seq for p in packets if p.keyframe]
    assert replay[:10] == packets[:10]
    assert replay[10].data != packets[10].data  # Same restart, genuinely different encoder rate.


def test_initial_and_both_rate_switch_directions_require_verified_idrs(monkeypatch):
    _need_pyav()
    import colabcola.bridge_video_rate as rate_module

    args = dict(
        width=96, height=64, fps=30, frames=12, gop=12,
        link=_clean_link(drop_seqs=(1,)), adapt=True,
        rates=RateConfig(silence_timeout_ms=150, min_switch_ms=0,
                         control_delay_ms=20, congested_age_ms=2000,
                         recovered_age_ms=50, recovery_samples=2))
    packets, _, controller = stream_with_rate_control(**args)
    assert {a["target_kbps"] for a in controller["applied"]} == {95, 650}
    segment_starts = [0] + [a["source_seq"] for a in controller["applied"]]
    assert all(packets[seq].keyframe for seq in segment_starts)
    original = rate_module.is_keyframe
    # Reject each real segment header in turn. Initial, downshift, and upshift
    # must all fail closed, rather than accepting a falsely advertised IDR.
    for seq in segment_starts:
        rejected = packets[seq].data
        with monkeypatch.context() as patch:
            patch.setattr(rate_module, "is_keyframe",
                          lambda data, rejected=rejected: data != rejected and original(data))
            with pytest.raises(RuntimeError, match="IDR"):
                stream_with_rate_control(**args)


def test_clean_link_four_arm_comparison_preserves_source_and_quality():
    _need_pyav()
    result = compare_rate_watchdog(
        width=96, height=64, fps=24, frames=24, gop=12,
        link=_clean_link(), rates=RateConfig(silence_timeout_ms=150),
        buffers_ms=(40, 80))
    for name in ("arrival_feedback", "silence_feedback"):
        assert result["controllers"][name]["requests_sent"] == 0
    assert result["restart_control"]["restart_seqs"] == []
    assert result["quality_comparison"]["common_frame_seqs"] == list(range(24))
    for name in ARMS:
        assert result[name] == result["fixed_high_rate"]
        assert result[name]["matched_frame_count"] == 24
        assert result[name]["quality_decoded_frames"] == 24
        assert result[name]["matched_frame_rgb_psnr_db"] is not None


def test_bandwidth_collapse_comparison_is_causal_repeatable_and_quality_matched():
    _need_pyav()
    args = dict(
        width=128, height=72, fps=30, frames=90, gop=24,
        link=LinkConfig(delay_ms=40, jitter_ms=0,
                        bandwidth_steps=((0, 1200), (600, 90), (1750, 1200)),
                        max_inflight=10, max_age_ms=350, seed=7),
        rates=RateConfig(silence_timeout_ms=150), buffers_ms=(80, 160, 350))
    result = compare_rate_watchdog(**args)
    assert compare_rate_watchdog(**args) == result
    controller = result["controllers"]["silence_feedback"]
    timeout_requests = [r for r in controller["requests"] if r["trigger"] == "receiver_silence"]
    assert timeout_requests
    assert any(600 < r["generated_at_ms"] < 1750 for r in timeout_requests)
    assert result["matched_restart_high_rate"]["encoded_idr_seqs"] == (
        result["silence_feedback"]["encoded_idr_seqs"])
    assert result["restart_control"]["restart_seqs"] == [
        a["source_seq"] for a in controller["applied"]]
    common = result["quality_comparison"]["common_frame_seqs"]
    assert common == sorted(set(common))
    assert len(common) > 0
    for name in ARMS:
        arm = result[name]
        assert arm["matched_frame_count"] == len(common)
        assert arm["matched_frame_rgb_psnr_db"] is not None
        assert arm["quality_decoded_frames"] == arm["actually_decoded_frames"]
        assert len(common) <= arm["actually_decoded_frames"] <= arm["delivered_frames"] <= 90
        assert arm["delivered_frames"] + arm["link_drops"] == 90
        assert all(p["on_time_frames"] <= arm["actually_decoded_frames"] for p in arm["playout"])
