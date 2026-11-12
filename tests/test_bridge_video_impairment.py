"""v23 local virtual-link scheduling and genuine H.264 IDR recovery tests.

No real sockets, remote providers, GPU allocations or Colab runtime.
The deterministic core is testable without optional video dependencies.
"""
from __future__ import annotations

import math

import pytest

from colabcola.bridge_video_stream import VideoAccessUnit
from colabcola.bridge_video_impairment import (
    MAX_FRAMES, LinkConfig, _serialize_ms, compare_impairments,
    make_h264_packets, parse_bandwidth_steps, parse_drop_seqs,
    parse_virtual_packet, simulate_link,
)


def fake_packets(count=20, payload_size=128):
    return [
        VideoAccessUnit(i, i * 50_000_000, i * 50_000_000,
                        bytes([i % 256]) * payload_size, keyframe=(i % 12 == 0))
        for i in range(count)
    ]


def test_config_rejects_unsafe_limits_and_invalid_step_profiles():
    for overrides in (
        {"delay_ms": -1}, {"delay_ms": float("nan")},
        {"jitter_ms": float("inf")}, {"jitter_ms": 1001},
        {"loss_percent": -0.1}, {"loss_percent": 100.1},
        {"bandwidth_kbps": -1}, {"bandwidth_kbps": True},
        {"max_inflight": 0}, {"max_inflight": 129},
        {"max_age_ms": -1}, {"seed": -1}, {"seed": True},
        {"bandwidth_steps": ((200, 300),)},
        {"bandwidth_steps": ((0, 300), (0, 100))},
        {"bandwidth_steps": ((0, 300), (500, 0))},
        {"drop_seqs": (5, 5)}, {"drop_seqs": (6, 5)},
        {"drop_seqs": (MAX_FRAMES,)},
        {"drop_seqs": [5]},
    ):
        with pytest.raises(ValueError):
            LinkConfig(**overrides)
    for invalid in ("1:100", "0:100,0:200", "0:100,300:-1", "x:y", "0:100:3"):
        with pytest.raises(ValueError):
            parse_bandwidth_steps(invalid)
    for invalid in ("5,5", "5,4", "-1", "1,x", str(MAX_FRAMES)):
        with pytest.raises(ValueError):
            parse_drop_seqs(invalid)


def test_rate_schedule_exact_piecewise_bit_integration():
    config = LinkConfig(
        delay_ms=0, jitter_ms=0,
        bandwidth_steps=((0, 1000), (5, 100), (40, 1000)),
    )
    # 5000 bits by 5ms; 3000 more bits at 100 bits/ms => 35ms.
    assert _serialize_ms(0, 8000, config) == pytest.approx(35)
    assert _serialize_ms(10, 2000, config) == pytest.approx(30)
    assert _serialize_ms(45, 2000, config) == pytest.approx(47)
    assert _serialize_ms(5, 0, config) == pytest.approx(5)
    assert _serialize_ms(1, 16000, LinkConfig(bandwidth_kbps=0)) == 1


def test_fifo_delay_and_seeded_jitter_are_deterministic():
    frames = fake_packets()
    cfg = LinkConfig(delay_ms=40, jitter_ms=20, bandwidth_kbps=0,
                     max_inflight=16, max_age_ms=0, seed=11)
    a = simulate_link(frames, 20, cfg)
    b = simulate_link(frames, 20, cfg)
    assert a == b
    assert len(a.delivered) == len(frames)
    assert a.loss_reasons == {"erased": 0, "queue_full": 0, "stale": 0}
    assert all(a.delivered[i].arrival_ms <= a.delivered[i + 1].arrival_ms
               for i in range(len(a.delivered) - 1))
    assert all(unit.arrival_ms >= unit.source_ms for unit in a.delivered)
    assert all(parse_virtual_packet(unit.packet) == unit.packet
               for unit in a.delivered)
    assert a != simulate_link(frames, 20, LinkConfig(
        delay_ms=40, jitter_ms=20, bandwidth_kbps=0, max_inflight=16,
        max_age_ms=0, seed=12,
    ))


def test_queue_limit_and_stale_drop_are_distinguished():
    frames = fake_packets()
    cfg = LinkConfig(delay_ms=400, jitter_ms=0, bandwidth_kbps=0,
                     max_inflight=2, max_age_ms=0)
    trace = simulate_link(frames, 20, cfg)
    assert trace.max_queue_occupancy == 2
    assert 0 < trace.loss_reasons["queue_full"] < len(frames)
    assert not trace.loss_reasons["erased"]
    assert len(trace.delivered) + len(trace.dropped) == len(frames)

    stale = simulate_link(fake_packets(8), 20, LinkConfig(
        delay_ms=70, jitter_ms=0, bandwidth_kbps=0,
        max_inflight=8, max_age_ms=30))
    assert stale.loss_reasons["stale"] == 8
    assert stale.wire_sent_bytes > 0
    assert stale.delivered == ()


def test_whole_access_unit_loss_consumes_bandwidth_but_is_not_delivered():
    frames = fake_packets()
    cfg = LinkConfig(delay_ms=15, jitter_ms=0, bandwidth_kbps=1000,
                     drop_seqs=(5,), loss_percent=0,
                     max_inflight=20, max_age_ms=0)
    trace = simulate_link(frames, 20, cfg)
    assert trace.dropped == ((5, "erased"),)
    assert len(trace.delivered) == 19
    assert trace.wire_sent_bytes == sum(len(frame.data) + 30 for frame in frames)
    assert trace.max_queue_occupancy <= cfg.max_inflight


def test_link_rejects_invalid_order_and_source_window():
    with pytest.raises(ValueError, match="consecutive"):
        simulate_link([fake_packets()[2]], 20, LinkConfig())
    with pytest.raises(ValueError, match="source frame count"):
        simulate_link([], 20, LinkConfig())
    with pytest.raises(ValueError, match="source fps"):
        simulate_link(fake_packets(), 0, LinkConfig())


def _need_pyav():
    av = pytest.importorskip("av", reason="optional PyAV not installed")
    pytest.importorskip("numpy")
    try:
        av.CodecContext.create("libx264", "w")
    except Exception:
        pytest.skip("PyAV build does not include libx264")


def test_real_h264_unimpaired_link_decodes_all_frames_and_keyframes():
    _need_pyav()
    report = compare_impairments(
        width=96, height=64, fps=24, frames=24, gop=12,
        config=LinkConfig(delay_ms=40, jitter_ms=5,
                          bandwidth_kbps=0, max_inflight=16, max_age_ms=0),
    )
    s, outcome = report["source"], report["outcome"]
    assert s["encoded_keyframes"] >= 2
    assert outcome["decoded_frames"] == 24
    assert outcome["decoded_runs"] == [[0, 23]]
    assert outcome["link_dropped_units"] == 0
    assert outcome["idr_resynchronizations"] == 0
    assert outcome["simulated_capture_to_arrival_p50_ms"] >= 35
    assert "no real network" in report["scope"]


def test_actual_h264_decoding_suppresses_dependencies_until_next_idr():
    _need_pyav()
    report = compare_impairments(
        width=96, height=64, fps=24, frames=36, gop=12,
        config=LinkConfig(delay_ms=20, jitter_ms=0,
                          bandwidth_kbps=0, max_inflight=32,
                          max_age_ms=0, drop_seqs=(5,)),
    )
    o = report["outcome"]
    assert o["link_drop_seqs"] == [5]
    assert o["link_drop_reasons"]["erased"] == 1
    # Frames 6..11 depend on missing frame 5; the next IDR at 12
    # starts a new decoder, avoiding false "success" from corrupt frames.
    assert o["decoded_runs"] == [[0, 4], [12, 35]]
    assert o["decoded_frames"] == 29
    assert o["dependency_skipped"] == 6
    assert o["idr_resynchronizations"] == 1


def test_missing_initial_idr_waits_for_next_closed_gop_idr():
    _need_pyav()
    report = compare_impairments(
        width=96, height=64, fps=24, frames=24, gop=12,
        config=LinkConfig(delay_ms=0, jitter_ms=0,
                          bandwidth_kbps=0, max_age_ms=0,
                          drop_seqs=(0,)),
    )
    assert report["outcome"]["decoded_runs"] == [[12, 23]]
    assert report["outcome"]["decoded_frames"] == 12


def test_bandwidth_fluctuation_and_seeded_loss_stay_bounded_and_repeatable():
    _need_pyav()
    cfg = LinkConfig(
        delay_ms=50, jitter_ms=25, bandwidth_steps=(
            (0, 2500), (300, 130), (850, 2000)),
        loss_percent=10, max_inflight=6, max_age_ms=250, seed=29,
    )
    a = compare_impairments(width=128, height=72, fps=30,
                            frames=48, gop=12, config=cfg)
    b = compare_impairments(width=128, height=72, fps=30,
                            frames=48, gop=12, config=cfg)
    # Codec binary sizes and virtual-link arrival traces are deterministic
    # for this fixed test source and encoder configuration.
    for key in ("link_drop_seqs", "link_drop_reasons", "decoded_runs",
                "decoded_frames", "dependency_skipped", "idr_resynchronizations"):
        assert a["outcome"][key] == b["outcome"][key]
    out = a["outcome"]
    assert out["peak_inflight_units"] <= cfg.max_inflight
    assert out["decoded_frames"] <= out["link_delivered_units"] <= 48
    assert out["decoded_frames"] + out["dependency_skipped"] + out["link_dropped_units"] == 48
    assert out["admitted_wire_bytes"] > 0
    assert out["link_dropped_units"] > 0
