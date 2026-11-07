"""v26 tests: causal rate feedback and real H264 bitrate/quality changes.

No external network or cloud compute. PyAV is optional outside bridge CI.
"""
from __future__ import annotations

import math

import pytest

from colabcola.bridge_video_stream import VideoAccessUnit
from colabcola.bridge_video_impairment import LinkConfig, ScheduledUnit
from colabcola.bridge_video_rate import (
    RateConfig, RateReceiver, compare_rate_control,
    reconstruction_quality, stream_with_rate_control,
)


def _arrival(seq, arrival_ms, fps=30):
    packet = VideoAccessUnit(seq, seq * 1_000_000, seq * 1_000_000,
                             b"test", seq == 0)
    return ScheduledUnit(packet, seq * 1000.0 / fps, arrival_ms, 40)


def test_rate_policy_bounds_and_hysteresis():
    invalid = (
        {"low_kbps": 700}, {"low_kbps": 650},
        {"high_kbps": -1}, {"high_kbps": True}, {"low_kbps": True},
        {"recovered_age_ms": 150}, {"congested_age_ms": -1},
        {"control_delay_ms": math.nan}, {"control_delay_ms": -1},
        {"min_switch_ms": float("inf")}, {"max_switches": 21},
        {"recovery_samples": 0}, {"max_switches": True},
    )
    for options in invalid:
        with pytest.raises(ValueError):
            RateConfig(**options)

    cfg = RateConfig(congested_age_ms=100, recovered_age_ms=50,
                     recovery_samples=2, control_delay_ms=20,
                     min_switch_ms=60)
    policy = RateReceiver(cfg)
    policy.observe(_arrival(0, 110))
    assert policy.report()["requests_sent"] == 1
    assert policy.requests[0]["target_kbps"] == cfg.low_kbps
    assert policy.apply_due(4, 130, cfg.high_kbps) == cfg.high_kbps
    assert policy.apply_due(5, 160, cfg.high_kbps) == cfg.low_kbps

    # The same stream can return to high rate only after sustained
    # low-age observations and the minimum controller switch interval.
    policy.observe(_arrival(1, 160))
    policy.observe(_arrival(2, 180))
    policy.observe(_arrival(3, 205))
    policy.observe(_arrival(4, 215))
    policy.observe(_arrival(5, 221))  # age ~=54.3ms, not low enough
    policy.observe(_arrival(6, 240))  # age 40ms
    policy.observe(_arrival(7, 270))  # age 36.7ms; second sample
    assert policy.report()["requests_sent"] == 2
    assert policy.requests[1]["target_kbps"] == cfg.high_kbps
    assert policy.apply_due(9, 285, cfg.low_kbps) == cfg.low_kbps
    assert policy.apply_due(10, 305, cfg.low_kbps) == cfg.high_kbps


def test_rate_receiver_only_sees_arrived_packets_and_gap_triggers():
    cfg = RateConfig(congested_age_ms=200, recovered_age_ms=80)
    receiver = RateReceiver(cfg)
    receiver.observe(_arrival(0, 40))
    assert not receiver.requests
    receiver.observe(_arrival(2, 100))
    assert receiver.gaps == 1
    assert receiver.requests[0]["gap_detected"] is True
    with pytest.raises(ValueError):
        receiver.observe(_arrival(1, 120))
    with pytest.raises(ValueError):
        receiver.observe(_arrival(3, float("nan")))
    with pytest.raises(ValueError):
        receiver.observe(_arrival(3, 90))


def _need_pyav():
    av = pytest.importorskip("av", reason="optional PyAV not installed")
    pytest.importorskip("numpy", reason="NumPy not installed")
    try:
        av.CodecContext.create("libx264", "w")
    except Exception:
        pytest.skip("PyAV build does not include libx264")


def _clean_link():
    return LinkConfig(delay_ms=15, jitter_ms=0,
                      bandwidth_kbps=0, max_age_ms=0, max_inflight=16)


def test_no_congestion_does_not_restart_real_h264_encoder():
    _need_pyav()
    result = compare_rate_control(
        width=96, height=64, fps=24, frames=24, gop=12,
        link=_clean_link(), rates=RateConfig(),
        buffers_ms=(40, 80),
    )
    assert result["controller"]["requests_sent"] == 0
    assert result["controller"]["switches_applied"] == 0
    assert result["fixed_high_rate"] == result["adaptive_rate"]
    assert result["fixed_high_rate"]["actually_decoded_frames"] == 24
    assert result["fixed_high_rate"]["quality_decoded_frames"] == 24


def test_real_encoder_target_bitrate_changes_encoded_bytes_and_rgb_quality():
    _need_pyav()
    clean = _clean_link()
    hi, trace_hi, _ = stream_with_rate_control(
        width=128, height=72, fps=30, frames=36, gop=12,
        link=clean, rates=RateConfig(high_kbps=700, low_kbps=90),
        adapt=False,
    )
    lo, trace_lo, _ = stream_with_rate_control(
        width=128, height=72, fps=30, frames=36, gop=12,
        link=clean, rates=RateConfig(high_kbps=100, low_kbps=50),
        adapt=False,
    )
    assert len(hi) == len(lo) == 36
    assert trace_hi.wire_sent_bytes > trace_lo.wire_sent_bytes
    metric_hi = reconstruction_quality(trace_hi, width=128, height=72, fps=30)
    metric_lo = reconstruction_quality(trace_lo, width=128, height=72, fps=30)
    assert metric_hi["quality_decoded_frames"] == 36
    assert metric_lo["quality_decoded_frames"] == 36
    assert metric_hi["rgb_psnr_on_decoded_frames_db"] > metric_lo["rgb_psnr_on_decoded_frames_db"]


def test_delayed_feedback_generates_actual_idr_at_future_frame():
    _need_pyav()
    cfg = RateConfig(
        high_kbps=650, low_kbps=95,
        congested_age_ms=45, recovered_age_ms=0,
        recovery_samples=30, control_delay_ms=35,
        min_switch_ms=100, max_switches=2,
    )
    link = LinkConfig(delay_ms=65, jitter_ms=0,
                      bandwidth_kbps=0, max_inflight=16, max_age_ms=0)
    result = compare_rate_control(
        width=96, height=64, fps=30, frames=36, gop=12,
        link=link, rates=cfg, buffers_ms=(40, 80, 120),
    )
    assert result["controller"]["requests_sent"] >= 1
    actions = result["controller"]["applied"]
    assert len(actions) == 1
    assert actions[0]["previous_kbps"] == 650
    assert actions[0]["target_kbps"] == 95
    assert actions[0]["source_seq"] >= 4
    assert result["adaptive_rate"]["encoded_idrs"] >= result["fixed_high_rate"]["encoded_idrs"]
    assert result["adaptive_rate"]["actually_decoded_frames"] == 36


def test_seeded_bandwidth_collapse_is_deterministic_and_bounded():
    _need_pyav()
    link = LinkConfig(
        delay_ms=40, jitter_ms=8,
        bandwidth_steps=((0, 1200), (600, 90), (1750, 1200)),
        max_inflight=10, max_age_ms=350, seed=7,
    )
    settings = RateConfig()
    args = dict(width=128, height=72, fps=30, frames=90, gop=24,
                link=link, rates=settings,
                buffers_ms=(80, 160, 240, 350))
    a = compare_rate_control(**args)
    b = compare_rate_control(**args)
    assert a["controller"] == b["controller"]
    for key in ("wire_bytes", "encoded_idrs", "delivered_frames",
                "link_drops", "actually_decoded_frames", "decoded_runs",
                "rgb_psnr_on_decoded_frames_db"):
        assert a["adaptive_rate"][key] == b["adaptive_rate"][key]
    for policy in ("adaptive_rate", "fixed_high_rate"):
        assert a[policy]["actually_decoded_frames"] <= a[policy]["delivered_frames"] <= 90
        assert a[policy]["quality_decoded_frames"] == a[policy]["actually_decoded_frames"]
    assert a["controller"]["requests_sent"] <= 8
    assert a["adaptive_rate"]["wire_bytes"] > 0
