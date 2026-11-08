"""v31 integration: unchanged source policy, causal dispatch and full horizons."""
from dataclasses import replace
import json

import pytest

from colabcola import bridge_video_pacing as pacing
from colabcola.bridge_video_controller import ControllerConfig, MultiGapFeedbackController
from colabcola.bridge_video_feedback import fixed_deadline_playout
from colabcola.bridge_video_impairment import LinkConfig, LinkTrace, ScheduledUnit, decode_delivered_h264, simulate_link
from colabcola.bridge_video_load import LoadControlConfig, LoadRateReceiver
from colabcola.bridge_video_rate import EncoderBurstConfig, RateConfig, stream_with_rate_control
from colabcola.bridge_video_sender import SenderQueueConfig
from colabcola.bridge_video_stream import VideoAccessUnit, packet_bytes


ARMS = ("immediate_live", "paced_fifo_live", "bounded_paced_live", "immediate_replay_bounded_source")


@pytest.fixture
def codec():
    av = pytest.importorskip("av", reason="optional PyAV not installed")
    pytest.importorskip("numpy", reason="NumPy not installed")
    try:
        av.CodecContext.create("libx264", "w")
    except Exception:
        pytest.skip("PyAV build does not include libx264")


def _args(**changes):
    result = dict(width=128, height=72, fps=30, frames=36, gop=24,
                  link=LinkConfig(delay_ms=40, jitter_ms=0, bandwidth_kbps=0,
                                  max_inflight=16, max_age_ms=0),
                  rates=RateConfig(silence_timeout_ms=150, max_switches=12),
                  load=LoadControlConfig(), recovery=ControllerConfig(feedback_ms=35, retry_ms=150, max_requests=8),
                  burst=EncoderBurstConfig(vbv_ms=100), sender=SenderQueueConfig())
    return {**result, **changes}


def test_sender_comparison_rejects_invalid_configs_and_unbounded_budgets():
    for values in ({"sender": False}, {"link": {}}, {"rates": 1}, {"load": "load"},
                   {"recovery": False}, {"burst": True}):
        with pytest.raises(ValueError, match="configuration"):
            pacing.compare_sender_queues(**values)
    with pytest.raises(ValueError, match="requires pacing"):
        pacing.compare_sender_queues(sender=SenderQueueConfig(pacing=False))
    with pytest.raises(ValueError, match="budgets"):
        pacing.compare_sender_queues(rates=RateConfig(max_switches=13))
    for buffers in ((), [], (80, 40), (40, 40), (True,), (1001,)):
        with pytest.raises(ValueError, match="buffer depths"):
            pacing.compare_sender_queues(buffers_ms=buffers)


def test_dispatch_peak_windows_exclude_fractional_frame_period_endpoint_ties():
    packets = [VideoAccessUnit(seq, round(seq * 1e9 / 30), round(seq * 1e9 / 30),
                               b"x" * 100, keyframe=seq == 0) for seq in range(12)]
    wire_size = len(packet_bytes(packets[0]))
    trace = LinkTrace(12, tuple(ScheduledUnit(p, p.seq * 1000.0 / 30,
                                             p.seq * 1000.0 / 30 + 10, wire_size)
                                for p in packets), (), wire_size * 12, 1)
    result = pacing._sender_accounting(
        packets, trace, pacing._immediate_sender(packets, fps=30), fps=30)
    assert [window["peak"]["wire_bytes"] for window in result["peak_dispatch_windows"]] == [
        3 * wire_size, 9 * wire_size]


def test_unpaced_driver_reproduces_existing_packets_trace_and_controller(codec):
    args = _args(frames=60, link=LinkConfig(delay_ms=40, jitter_ms=0, bandwidth_kbps=0,
                                           max_inflight=16, max_age_ms=0, drop_seqs=(5, 8, 27)),
                 sender=SenderQueueConfig(pacing=False, max_pending=120, max_age_ms=0, max_repair_requests=0))
    after = pacing.stream_with_sender_queue(**args)
    before_args = {k: v for k, v in args.items() if k not in ("sender", "load")}
    before = stream_with_rate_control(
        **before_args, adapt=True,
        rate_policy=LoadRateReceiver(args["rates"], args["load"], fps=args["fps"]))
    assert after[0] == before[0]
    assert after[1] == before[1]
    normalized = {k: v for k, v in after[2].items() if k != "sender"}
    normalized["encoder_restarts"] = [
        {k: v for k, v in r.items() if k != "local_repair_request_ids"}
        for r in normalized["encoder_restarts"]]
    assert normalized == before[2]
    assert after[2]["sender"]["drops"] == []


def test_rate_remote_and_local_requests_coalesce_into_one_real_idr(codec, monkeypatch):
    # Inject two controlled due causes at the source frame where the real
    # one-slot queue's overflow repair is consumed. This checks composition,
    # not the receiver's separately tested congestion/gap detection policy.
    class TimedRate(LoadRateReceiver):
        def apply_due(self, source_seq, source_ms, current_kbps):
            return 20 if source_seq == 3 else current_kbps

    class TimedRecovery(MultiGapFeedbackController):
        def requests_due_before(self, source_ms):
            if source_ms == 100 and not self.requests:
                request = {"request_id": 0, "receiver_observed_seq": 0,
                           "detected_at_ms": 79, "reaches_source_ms": 99}
                self.requests.append(request)
                return [request]
            return []

    targets = []
    original_encoder = pacing._encoder

    def counted_encoder(*args, **kwargs):
        targets.append(args[5])
        return original_encoder(*args, **kwargs)

    monkeypatch.setattr(pacing, "LoadRateReceiver", TimedRate)
    monkeypatch.setattr(pacing, "MultiGapFeedbackController", TimedRecovery)
    monkeypatch.setattr(pacing, "_encoder", counted_encoder)
    packets, trace, info = pacing.stream_with_sender_queue(**_args(
        frames=4, rates=RateConfig(high_kbps=50, low_kbps=20, max_switches=0),
        recovery=ControllerConfig(feedback_ms=20, max_requests=1),
        sender=SenderQueueConfig(max_pending=1, max_age_ms=0, max_repair_requests=1)))
    assert targets == [50, 20]
    assert len(info["encoder_restarts"]) == 1
    restart = info["encoder_restarts"][0]
    assert restart["source_seq"] == 3 and restart["rate_change"]
    assert restart["recovery_request_ids"] == [0]
    assert restart["local_repair_request_ids"] == [0]
    assert packets[3].keyframe
    assert info["sender"]["applied"][0]["source_idr_seq"] == 3
    assert decode_delivered_h264(trace, 128, 72)["decoded_runs"] == [[0, 0], [3, 3]]


def test_future_capacity_cannot_change_prior_source_or_sender_events(codec):
    args = _args(frames=36, link=LinkConfig(delay_ms=40, jitter_ms=0,
                                           bandwidth_steps=((0, 1200),),
                                           max_inflight=16, max_age_ms=350))
    before = pacing.stream_with_sender_queue(**args)
    after = pacing.stream_with_sender_queue(**{**args, "link": replace(
        args["link"], bandwidth_steps=((0, 1200), (900, 90)))})
    assert before[0][:27] == after[0][:27]
    for key, time_key in (("dispatches", "dispatch_ms"), ("drops", "at_ms"),
                          ("requests", "created_at_ms")):
        assert [x for x in before[2]["sender"][key] if x[time_key] < 900] == [
            x for x in after[2]["sender"][key] if x[time_key] < 900]
    assert [x for x in before[1].delivered if x.arrival_ms < 900] == [
        x for x in after[1].delivered if x.arrival_ms < 900]


def test_downstream_admission_rejection_does_not_refund_or_repair_sender(codec):
    args = _args(frames=12, rates=RateConfig(max_switches=0, silence_timeout_ms=0),
                 recovery=ControllerConfig(max_requests=0),
                 sender=SenderQueueConfig(max_pending=120, max_age_ms=0, max_repair_requests=0),
                 link=LinkConfig(delay_ms=200, jitter_ms=0, bandwidth_kbps=0,
                                 max_inflight=1, max_age_ms=0))
    rejected = pacing.stream_with_sender_queue(**args)
    admitted = pacing.stream_with_sender_queue(**{**args, "link": replace(args["link"], max_inflight=16)})
    assert rejected[0] == admitted[0]
    assert rejected[2]["sender"]["dispatches"] == admitted[2]["sender"]["dispatches"]
    assert rejected[2]["sender"]["drops"] == admitted[2]["sender"]["drops"] == []
    assert rejected[2]["sender"]["requests"] == admitted[2]["sender"]["requests"] == []
    assert rejected[1].loss_reasons["queue_full"] > 0
    assert admitted[1].loss_reasons["queue_full"] == 0


def test_post_source_fifo_tail_retains_original_capture_deadlines(codec):
    args = _args(fps=60, frames=4,
                 rates=RateConfig(high_kbps=50, low_kbps=20, max_switches=0, silence_timeout_ms=0),
                 recovery=ControllerConfig(max_requests=0),
                 sender=SenderQueueConfig(max_pending=120, max_age_ms=0, max_repair_requests=0))
    packets, trace, info = pacing.stream_with_sender_queue(**args)
    observed = decode_delivered_h264(trace, 128, 72)
    immediate = simulate_link(packets, 60, args["link"])
    immediate_observed = decode_delivered_h264(immediate, 128, 72)
    assert trace.source_count == 4 and observed["decoded_frames"] == 4
    assert info["sender"]["post_source_dispatches"] > 0
    assert info["sender"]["last_dispatch_ms"] > info["sender"]["last_capture_ms"]
    assert info["encoder_restarts"] == []
    assert info["sender"]["pending_frames"] == 0
    delayed_playout = fixed_deadline_playout(trace, observed, fps=60, buffer_ms=80)
    immediate_playout = fixed_deadline_playout(immediate, immediate_observed, fps=60, buffer_ms=80)
    assert immediate_playout["on_time_frames"] == 4
    assert delayed_playout["on_time_frames"] < 4
    assert delayed_playout["on_time_frames"] + delayed_playout["late_decoded_frames"] == 4
    assert all(item.source_ms == item.packet.seq * 1000.0 / 60 for item in trace.delivered)


def test_disabled_local_repair_keeps_suppressing_dependents_and_full_horizon(codec):
    packets, trace, info = pacing.stream_with_sender_queue(**_args(
        fps=60, frames=12,
        rates=RateConfig(high_kbps=50, low_kbps=20, max_switches=0, silence_timeout_ms=0),
        recovery=ControllerConfig(max_requests=0),
        sender=SenderQueueConfig(max_pending=1, max_age_ms=0, max_repair_requests=0)))
    assert len(packets) == trace.source_count == 12
    assert info["sender"]["requests"] == info["encoder_restarts"] == []
    assert info["sender"]["need_idr"] and info["sender"]["repair_budget_exhausted"]
    assert info["sender"]["dispatches"][-1]["source_seq"] == 0
    assert len(info["sender"]["drops"]) == 11
    decoded = decode_delivered_h264(trace, 128, 72)
    playout = fixed_deadline_playout(trace, decoded, fps=60, buffer_ms=80)
    assert decoded["decoded_runs"] == [[0, 0]]
    assert playout["on_time_frames"] == 1 and playout["undecodable_frames"] == 11


def test_four_arm_replay_accounts_for_all_frames_bytes_idrs_and_quality(codec):
    report = pacing.compare_sender_queues(
        frames=120, link=LinkConfig(delay_ms=40, jitter_ms=0, bandwidth_kbps=0,
                                    bandwidth_steps=((0, 1200), (600, 300), (2200, 1200)),
                                    max_inflight=32, max_age_ms=1000))
    common = set.intersection(*(
        {seq for start, end in report[name]["decoded_runs"] for seq in range(start, end + 1)}
        for name in ARMS))
    assert report["quality_comparison"]["common_frame_seqs"] == sorted(common)
    for name in ARMS:
        arm = report[name]
        assert arm["sender_discarded_frames"] + arm["dispatch_attempts"] == 120
        assert arm["sender_discarded_frames"] + arm["delivered_frames"] + arm["link_drops"] == 120
        assert arm["encoded_wire_bytes"] == arm["sender_discarded_wire_bytes"] + arm["dispatch_attempted_wire_bytes"]
        assert arm["dispatch_attempted_wire_bytes"] == arm["wire_bytes"] + arm["link_admission_rejected_wire_bytes"]
        assert arm["matched_frame_count"] == len(common)
        for idr in arm["idr_delivery"]:
            assert (idr["arrival_ms"] is not None) != (idr["drop_reason"] is not None)
        for playout in arm["playout"]:
            assert playout["on_time_frames"] + playout["late_decoded_frames"] + playout["undecodable_frames"] == 120
    a, b = report["bounded_paced_live"], report["immediate_replay_bounded_source"]
    assert a["encoded_source_sha256"] == b["encoded_source_sha256"]
    assert a["encoded_idr_seqs"] == b["encoded_idr_seqs"]
    assert a["matched_frame_rgb_psnr_db"] == b["matched_frame_rgb_psnr_db"]
    assert report["paced_fifo_live"]["sender_discarded_frames"] == 0
    assert not report["bounded_queue_ever_dropped"]
    assert report["paced_fifo_live"] == report["bounded_paced_live"]
    local = report["senders"]["bounded_paced_live"]
    assert local["peak_pending_frames"] <= 3 and local["pending_frames"] == 0
    assert len(local["requests"]) <= 8
    for applied in local["applied"]:
        assert applied["source_ms"] > applied["requested_at_ms"]
    assert len(report["controllers"]["bounded_paced_live"]["encoder_restarts"]) <= 28
    json.dumps(report, allow_nan=False)
