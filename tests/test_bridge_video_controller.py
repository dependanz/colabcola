"""v25 causal multi-gap IDR controller and incremental link regression tests.

No real network / Colab / GPU. The PyAV tests use real H.264 decoding.
"""
from __future__ import annotations

import math

import pytest

from colabcola.bridge_video_impairment import LinkConfig, simulate_link
from colabcola.bridge_video_stream import VideoAccessUnit
from colabcola.bridge_video_controller import (
    ControllerConfig, MultiGapFeedbackController, VirtualAccessUnitLink,
    compare_multi_feedback, simulate_multi_feedback,
)


def fake_frames(n=24):
    return [
        VideoAccessUnit(
            seq=i, pts_ns=i * 20_000_000,
            capture_ns=i * 20_000_000,
            data=bytes([i % 256]) * (100 + i * 10),
            keyframe=(i == 0),
        )
        for i in range(n)
    ]


def _fake_idr(seq):
    return VideoAccessUnit(
        seq, seq * 1_000_000, seq * 1_000_000,
        b"\x00\x00\x00\x01\x65\x00", keyframe=True,
    )


def _fake_predicted(seq):
    return VideoAccessUnit(
        seq, seq * 1_000_000, seq * 1_000_000,
        b"\x00\x00\x00\x01\x41\x00", keyframe=False,
    )


def test_incremental_link_matches_v23_batch_scheduler_under_seeded_impairments():
    packets = fake_frames()
    link = LinkConfig(
        delay_ms=70, jitter_ms=12,
        bandwidth_steps=((0, 600), (320, 120), (700, 1000)),
        loss_percent=12, max_inflight=3, max_age_ms=200,
        seed=9, drop_seqs=(7,),
    )
    expected = simulate_link(packets, 30, link)
    scheduler = VirtualAccessUnitLink(link)
    for p in packets:
        scheduler.arrive_by(p.seq * 1000 / 30)
        scheduler.send(p, fps=30)
    scheduler.arrive_by(math.inf)
    assert scheduler.snapshot(len(packets)) == expected


def test_receiver_cannot_request_missing_frame_before_later_arrival():
    control = MultiGapFeedbackController(
        ControllerConfig(feedback_ms=20, retry_ms=100))
    control.receive(_fake_idr(0), 40)
    for seq in range(1, 5):
        control.receive(_fake_predicted(seq), 40 + seq * 33.33)
    assert not control.requests
    control.receive(_fake_predicted(6), 240)
    assert len(control.requests) == 1
    assert control.requests[0]["reaches_source_ms"] == 260
    assert control.requests_due_before(260) == []
    due = control.requests_due_before(266.667)
    control.record_applied(due, 8)
    assert control.report()["applied"][0]["source_idr_seq"] == 8
    assert control.report()["gaps_detected"][0]["first_missing_seq"] == 5
    control.receive(_fake_predicted(7), 273.33)
    assert len(control.requests) == 1
    control.receive(_fake_idr(8), 306.667)
    assert len(control.requests) == 1


def test_closed_loop_bounds_and_impossible_receiver_order():
    for override in (
        {"feedback_ms": -1}, {"feedback_ms": math.nan},
        {"retry_ms": -1}, {"retry_ms": math.inf},
        {"max_requests": -1}, {"max_requests": 33},
        {"max_requests": True},
    ):
        with pytest.raises(ValueError):
            ControllerConfig(**override)
    controller = MultiGapFeedbackController(ControllerConfig())
    controller.receive(_fake_idr(0), 20)
    with pytest.raises(ValueError):
        controller.receive(_fake_predicted(0), 30)
    with pytest.raises(ValueError):
        controller.receive(_fake_predicted(1), 10)
    with pytest.raises(ValueError):
        controller.receive(_fake_predicted(1), float("nan"))
    with pytest.raises(ValueError):
        compare_multi_feedback(width=96, height=64, frames=60, gop=24,
                               controller=ControllerConfig(), buffers_ms=(80, 40))


def _need_pyav():
    av = pytest.importorskip("av", reason="optional PyAV not installed")
    pytest.importorskip("numpy", reason="NumPy not installed")
    try:
        av.CodecContext.create("libx264", "w")
    except Exception:
        pytest.skip("PyAV build lacks libx264")


def _loss_link(*sequences):
    return LinkConfig(delay_ms=40, jitter_ms=0,
                      bandwidth_kbps=0, max_inflight=16,
                      max_age_ms=0, drop_seqs=tuple(sequences))


def test_real_h264_two_distinct_losses_trigger_two_causal_idr_resets():
    _need_pyav()
    result = compare_multi_feedback(
        width=96, height=64, fps=30, frames=60, gop=24,
        link=_loss_link(5, 27),
        controller=ControllerConfig(feedback_ms=20, retry_ms=100),
        buffers_ms=(40, 80, 120),
    )
    applied = result["controller"]["applied"]
    assert [x["source_idr_seq"] for x in applied] == [8, 30]
    assert result["controller"]["requests_sent"] == 2
    assert result["controller"]["idr_restarts_applied"] == 2
    assert result["controller"]["requests_not_applied_before_source_end"] == 0
    base, recovery = result["periodic"], result["multi_feedback"]
    assert recovery["actually_decoded_frames"] > base["actually_decoded_frames"]
    assert recovery["decoded_runs"] == [[0, 4], [8, 26], [30, 59]]
    assert recovery["actually_decoded_frames"] == 54
    assert recovery["decoder_idr_resynchronizations"] == 2
    assert recovery["link_drops"] == 2
    assert recovery["playout"][0]["on_time_frames"] == 54


def test_absent_loss_does_not_change_real_h264_bitstream_or_schedule():
    _need_pyav()
    result = compare_multi_feedback(
        width=96, height=64, fps=24, frames=24, gop=12,
        link=LinkConfig(delay_ms=30, jitter_ms=7, bandwidth_kbps=0,
                        max_age_ms=0, max_inflight=16, seed=19),
        controller=ControllerConfig(feedback_ms=20),
        buffers_ms=(40, 80),
    )
    assert result["controller"]["requests_sent"] == 0
    assert result["controller"]["idr_restarts_applied"] == 0
    assert result["periodic"] == result["multi_feedback"]
    assert result["periodic"]["actually_decoded_frames"] == 24


def test_disabling_control_matches_baseline_even_with_multiple_losses():
    _need_pyav()
    result = compare_multi_feedback(
        width=96, height=64, fps=30, frames=60, gop=24,
        link=_loss_link(5, 27),
        controller=ControllerConfig(max_requests=0),
        buffers_ms=(40, 80),
    )
    assert result["controller"]["requests_sent"] == 0
    assert result["periodic"] == result["multi_feedback"]


def test_lost_feedback_generated_idr_can_be_requested_again_after_cooldown():
    _need_pyav()
    result = compare_multi_feedback(
        width=96, height=64, fps=30, frames=36, gop=24,
        link=_loss_link(5, 8),
        controller=ControllerConfig(feedback_ms=20, retry_ms=80),
        buffers_ms=(40,),
    )
    requests = result["controller"]["applied"]
    assert requests[0]["source_idr_seq"] == 8
    assert len(requests) >= 2
    assert requests[1]["source_idr_seq"] > 8
    assert result["multi_feedback"]["actually_decoded_frames"] > 5


def test_seeded_bandwidth_outage_closed_loop_remains_bounded():
    _need_pyav()
    link = LinkConfig(
        delay_ms=50, jitter_ms=15,
        bandwidth_steps=((0, 2500), (400, 130), (1100, 2200)),
        max_inflight=6, max_age_ms=200, seed=17,
    )
    kwargs = dict(width=128, height=72, fps=30, frames=60, gop=24,
                  link=link, controller=ControllerConfig(feedback_ms=20,
                                                        retry_ms=150,
                                                        max_requests=5),
                  buffers_ms=(40, 80, 160, 240))
    a = compare_multi_feedback(**kwargs)
    b = compare_multi_feedback(**kwargs)
    assert a["controller"] == b["controller"]
    for key in ("wire_bytes", "delivered_frames", "link_drops",
                "decoded_runs", "actually_decoded_frames"):
        assert a["multi_feedback"][key] == b["multi_feedback"][key]
    assert a["multi_feedback"]["actually_decoded_frames"] <= 60
    assert a["controller"]["requests_sent"] <= 5
    assert a["multi_feedback"]["playout"][0]["on_time_frames"] <= 60
