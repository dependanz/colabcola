"""v32 integration: independent service, causal observations and full accounting."""
from dataclasses import replace
import json

import pytest

from colabcola import bridge_video_pacing as pacing
from colabcola import bridge_video_service as service_module
from colabcola.bridge_video_controller import ControllerConfig, MultiGapFeedbackController
from colabcola.bridge_video_feedback import fixed_deadline_playout
from colabcola.bridge_video_impairment import LinkConfig, decode_delivered_h264, simulate_link
from colabcola.bridge_video_load import LoadControlConfig, LoadRateReceiver
from colabcola.bridge_video_rate import EncoderBurstConfig, RateConfig
from colabcola.bridge_video_sender import SenderQueueConfig
from colabcola.bridge_video_sender_service import SenderServiceConfig


ARMS = ("immediate_live", "target_paced_live", "fixed_service_live",
        "queue_service_live", "immediate_replay_queue_source")
LIVE_ARMS = ARMS[:-1]


@pytest.fixture
def codec():
    av = pytest.importorskip("av", reason="optional PyAV not installed")
    pytest.importorskip("numpy", reason="NumPy not installed")
    try:
        av.CodecContext.create("libx264", "w")
    except Exception:
        pytest.skip("PyAV build does not include libx264")


def _args(**changes):
    values = dict(
        width=128, height=72, fps=30, frames=36, gop=24,
        link=LinkConfig(delay_ms=40, jitter_ms=0, bandwidth_kbps=0,
                        max_inflight=16, max_age_ms=0),
        rates=RateConfig(silence_timeout_ms=150, max_switches=12),
        load=LoadControlConfig(),
        recovery=ControllerConfig(feedback_ms=35, retry_ms=150, max_requests=8),
        burst=EncoderBurstConfig(vbv_ms=100), sender=SenderQueueConfig(),
    )
    return {**values, **changes}


def test_invalid_service_driver_and_comparison_settings_fail_before_encoding(monkeypatch):
    def unexpected_encoder(*args, **kwargs):
        raise AssertionError("invalid configuration reached the encoder")

    monkeypatch.setattr(pacing, "_encoder", unexpected_encoder)
    monkeypatch.setattr(service_module, "stream_with_rate_control", unexpected_encoder)
    for invalid in (False, {}, "service"):
        with pytest.raises(ValueError, match="service configuration"):
            pacing.stream_with_sender_queue(**_args(), service=invalid)
        with pytest.raises(ValueError, match="configuration"):
            service_module.compare_sender_service(service=invalid)
    with pytest.raises(ValueError, match="requires pacing"):
        pacing.stream_with_sender_queue(
            **_args(sender=SenderQueueConfig(pacing=False)), service=SenderServiceConfig())
    with pytest.raises(ValueError, match="requires pacing"):
        service_module.compare_sender_service(sender=SenderQueueConfig(pacing=False))
    with pytest.raises(ValueError, match="budgets"):
        service_module.compare_sender_service(rates=RateConfig(max_switches=13))
    for buffers in ((), [], (80, 40), (40, 40), (True,), (1001,)):
        with pytest.raises(ValueError, match="buffer depths"):
            service_module.compare_sender_service(buffers_ms=buffers)


def test_retained_target_control_matches_v31_source_delivery_and_controller(codec):
    args = _args(link=LinkConfig(delay_ms=40, jitter_ms=0, bandwidth_kbps=0,
                                 max_inflight=16, max_age_ms=0, drop_seqs=(5, 8, 27)))
    previous = pacing.compare_sender_queues(**args)
    current = service_module.compare_sender_service(**args)
    # The common quality cohort can change when a new comparison arm is added.
    # Every other source, delivery, playout and accounting field must agree.
    cohort_fields = {"matched_frame_count", "matched_frame_rgb_psnr_db"}
    for old_name, new_name in (("immediate_live", "immediate_live"),
                              ("bounded_paced_live", "target_paced_live")):
        assert {k: v for k, v in previous[old_name].items() if k not in cohort_fields} == {
            k: v for k, v in current[new_name].items() if k not in cohort_fields}
        assert previous["controllers"][old_name] == current["controllers"][new_name]
        assert previous["senders"][old_name] == current["senders"][new_name]
    assert "service" not in current["senders"]["target_paced_live"]


def test_disabled_boost_reproduces_fixed_service_in_all_report_layers(codec):
    report = service_module.compare_sender_service(
        **_args(), service=SenderServiceConfig(queue_boost=False))
    assert not report["service_budget"]["queue_boost_ever_active"]
    assert report["fixed_service_live"] == report["queue_service_live"]
    assert report["senders"]["fixed_service_live"] == report["senders"]["queue_service_live"]
    assert report["controllers"]["fixed_service_live"] == report["controllers"]["queue_service_live"]


def test_fixed_service_matches_target_pacing_through_same_rate_recovery(codec):
    args = _args(
        frames=60, rates=RateConfig(max_switches=0, silence_timeout_ms=0),
        recovery=ControllerConfig(feedback_ms=20, retry_ms=80, max_requests=8),
        link=LinkConfig(delay_ms=40, jitter_ms=0, bandwidth_kbps=0,
                        max_inflight=16, max_age_ms=0, drop_seqs=(5, 8, 27)))
    target = pacing.stream_with_sender_queue(**args)
    fixed = pacing.stream_with_sender_queue(**args, service=SenderServiceConfig(queue_boost=False))
    assert target[0] == fixed[0] and target[1] == fixed[1]
    assert {k: v for k, v in target[2].items() if k != "sender"} == {
        k: v for k, v in fixed[2].items() if k != "sender"}
    assert target[2]["recovery"]["applied"]
    assert all(not row["rate_change"] for row in target[2]["encoder_restarts"])
    assert target[2]["sender"] == {k: v for k, v in fixed[2]["sender"].items() if k != "service"}


def test_service_changes_dispatch_but_not_a_fixed_encoder_action_schedule(codec, monkeypatch):
    class TimedRate(LoadRateReceiver):
        def apply_due(self, source_seq, source_ms, current_kbps):
            return 20 if source_seq == 3 else current_kbps

    monkeypatch.setattr(pacing, "LoadRateReceiver", TimedRate)
    args = _args(
        fps=60, frames=12,
        rates=RateConfig(high_kbps=50, low_kbps=20, max_switches=0, silence_timeout_ms=0),
        recovery=ControllerConfig(max_requests=0),
        sender=SenderQueueConfig(max_pending=120, max_age_ms=0, max_repair_requests=0))
    fixed = pacing.stream_with_sender_queue(**args, service=SenderServiceConfig(queue_boost=False))
    boosted = pacing.stream_with_sender_queue(**args, service=SenderServiceConfig())
    # Hold every encoder action fixed. Changing sender service must preserve
    # the real encoded bytes, including the downshift's independently decodable IDR.
    assert fixed[0] == boosted[0]
    assert fixed[2]["encoder_restarts"] == boosted[2]["encoder_restarts"]
    assert [row["source_seq"] for row in fixed[2]["encoder_restarts"]] == [3]
    assert fixed[0][3].keyframe
    assert fixed[2]["sender"]["drops"] == boosted[2]["sender"]["drops"] == []
    assert fixed[2]["sender"]["dispatches"] != boosted[2]["sender"]["dispatches"]
    fixed_service = fixed[2]["sender"]["service"]
    assert fixed_service["base_kbps"] == pytest.approx(64.4)
    assert {row["pacing_kbps"] for row in fixed_service["decisions"]} == {fixed_service["base_kbps"]}
    assert fixed[2]["sender"]["rate_changes"][-1]["target_kbps"] == 20
    assert any(row["boosted"] for row in boosted[2]["sender"]["service"]["decisions"])


def test_future_link_capacity_cannot_change_prior_service_or_source_events(codec):
    args = _args(link=LinkConfig(delay_ms=40, jitter_ms=0,
                                 bandwidth_steps=((0, 1200),),
                                 max_inflight=16, max_age_ms=350))
    before = pacing.stream_with_sender_queue(**args, service=SenderServiceConfig())
    after = pacing.stream_with_sender_queue(
        **{**args, "link": replace(args["link"], bandwidth_steps=((0, 1200), (900, 90)))},
        service=SenderServiceConfig())
    assert before[0][:27] == after[0][:27]
    for key, time_key in (("dispatches", "dispatch_ms"), ("drops", "at_ms"),
                          ("requests", "created_at_ms")):
        assert [row for row in before[2]["sender"][key] if row[time_key] < 900] == [
            row for row in after[2]["sender"][key] if row[time_key] < 900]
    assert [row for row in before[2]["sender"]["service"]["decisions"] if row["at_ms"] < 900] == [
        row for row in after[2]["sender"]["service"]["decisions"] if row["at_ms"] < 900]
    assert [item for item in before[1].delivered if item.arrival_ms < 900] == [
        item for item in after[1].delivered if item.arrival_ms < 900]


def test_link_admission_rejection_cannot_refund_debt_or_change_service(codec):
    args = _args(
        frames=12, rates=RateConfig(max_switches=0, silence_timeout_ms=0),
        recovery=ControllerConfig(max_requests=0),
        sender=SenderQueueConfig(max_pending=120, max_age_ms=0, max_repair_requests=0),
        link=LinkConfig(delay_ms=200, jitter_ms=0, bandwidth_kbps=0,
                        max_inflight=1, max_age_ms=0))
    rejected = pacing.stream_with_sender_queue(**args, service=SenderServiceConfig())
    admitted = pacing.stream_with_sender_queue(
        **{**args, "link": replace(args["link"], max_inflight=16)}, service=SenderServiceConfig())
    assert rejected[0] == admitted[0]
    for key in ("dispatches", "drops", "requests", "service", "remaining_debt_bits"):
        assert rejected[2]["sender"][key] == admitted[2]["sender"][key]
    assert rejected[2]["sender"]["drops"] == rejected[2]["sender"]["requests"] == []
    assert rejected[1].loss_reasons["queue_full"] > 0
    assert admitted[1].loss_reasons["queue_full"] == 0


def test_capped_service_expiry_repair_coalesces_with_rate_and_remote_idr(codec, monkeypatch):
    class TimedRate(LoadRateReceiver):
        def apply_due(self, source_seq, source_ms, current_kbps):
            return 20 if source_seq == 2 else current_kbps

    class TimedRecovery(MultiGapFeedbackController):
        def requests_due_before(self, source_ms):
            if source_ms == 2 * 1000.0 / 30 and not self.requests:
                request = {"request_id": 0, "receiver_observed_seq": 0,
                           "detected_at_ms": 60, "reaches_source_ms": 65}
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
    packets, trace, info = pacing.stream_with_sender_queue(
        **_args(frames=4, rates=RateConfig(high_kbps=50, low_kbps=20, max_switches=0),
                recovery=ControllerConfig(max_requests=1),
                sender=SenderQueueConfig(max_pending=3, max_age_ms=20, max_repair_requests=1)),
        service=SenderServiceConfig(drain_ms=1, max_rate_factor=1))
    assert targets == [50, 20]
    assert len(info["encoder_restarts"]) == 1
    restart = info["encoder_restarts"][0]
    assert restart["source_seq"] == 2 and restart["rate_change"]
    assert restart["recovery_request_ids"] == restart["local_repair_request_ids"] == [0]
    assert packets[2].keyframe
    local = info["sender"]
    assert any(row["source_seq"] == 1 and row["reason"] == "sender_age" for row in local["drops"])
    assert len(local["requests"]) == len(local["applied"]) == 1
    assert local["applied"][0]["source_idr_seq"] == 2
    assert local["applied"][0]["requested_at_ms"] < 2 * 1000.0 / 30
    assert any(row["cap_limited"] for row in local["service"]["decisions"])
    assert all(row["pacing_kbps"] <= local["service"]["cap_kbps"]
               for row in local["service"]["decisions"])
    # The generated recovery IDR is real, but its subsequent sender/link fate
    # remains subject to debt and expiry; this test does not assume recovery.
    assert decode_delivered_h264(trace, 128, 72)["decoded_frames"] >= 1


def test_service_tail_keeps_capture_deadlines_and_stops_encoder_actions(codec):
    args = _args(
        fps=60, frames=4,
        rates=RateConfig(high_kbps=50, low_kbps=20, max_switches=0, silence_timeout_ms=0),
        recovery=ControllerConfig(max_requests=0),
        sender=SenderQueueConfig(max_pending=120, max_age_ms=0, max_repair_requests=0))
    packets, trace, info = pacing.stream_with_sender_queue(
        **args, service=SenderServiceConfig(drain_ms=1, max_rate_factor=1))
    decoded = decode_delivered_h264(trace, 128, 72)
    immediate = simulate_link(packets, args["fps"], args["link"])
    immediate_decoded = decode_delivered_h264(immediate, 128, 72)
    local = info["sender"]
    assert len(packets) == trace.source_count == decoded["decoded_frames"] == 4
    assert info["encoder_restarts"] == [] and info["source_frames_encoded"] == 4
    assert local["pending_frames"] == 0 and local["post_source_dispatches"] > 0
    assert local["last_dispatch_ms"] > local["last_capture_ms"]
    tail_decisions = [row for row in local["service"]["decisions"]
                      if row["at_ms"] > local["last_capture_ms"]]
    assert tail_decisions and {row["cause"] for row in tail_decisions} == {"dispatch"}
    assert all(item.source_ms == item.packet.seq * 1000.0 / args["fps"] for item in trace.delivered)
    delayed = fixed_deadline_playout(trace, decoded, fps=args["fps"], buffer_ms=80)
    unqueued = fixed_deadline_playout(immediate, immediate_decoded, fps=args["fps"], buffer_ms=80)
    assert unqueued["on_time_frames"] == 4
    assert delayed["on_time_frames"] < 4
    assert delayed["on_time_frames"] + delayed["late_decoded_frames"] == 4


def test_five_arm_report_conserves_source_link_and_replay_with_both_loss_layers(codec):
    frames = 18
    report = service_module.compare_sender_service(
        **_args(fps=60, frames=frames, gop=6,
                rates=RateConfig(high_kbps=50, low_kbps=20, max_switches=0, silence_timeout_ms=0),
                recovery=ControllerConfig(max_requests=0),
                sender=SenderQueueConfig(max_pending=1, max_age_ms=80, max_repair_requests=1),
                link=LinkConfig(delay_ms=200, jitter_ms=0, bandwidth_kbps=0,
                                max_inflight=1, max_age_ms=0)),
        buffers_ms=(40, 80, 160, 350))
    assert report["mode"] == "causal-h264-sender-service-comparison"
    common = set.intersection(*(
        {seq for first, last in report[name]["decoded_runs"] for seq in range(first, last + 1)}
        for name in ARMS))
    assert report["quality_comparison"]["common_frame_seqs"] == sorted(common)
    assert any(report[name]["sender_discarded_frames"] > 0 for name in LIVE_ARMS)
    assert any(report[name]["link_admission_rejected_frames"] > 0 for name in ARMS)
    for name in ARMS:
        arm = report[name]
        assert arm["sender_discarded_frames"] + arm["dispatch_attempts"] == frames
        assert arm["sender_discarded_frames"] + arm["delivered_frames"] + arm["link_drops"] == frames
        assert arm["encoded_wire_bytes"] == arm["sender_discarded_wire_bytes"] + arm["dispatch_attempted_wire_bytes"]
        assert arm["dispatch_attempted_wire_bytes"] == arm["wire_bytes"] + arm["link_admission_rejected_wire_bytes"]
        assert arm["matched_frame_count"] == len(common)
        for idr in arm["idr_delivery"]:
            assert (idr["arrival_ms"] is not None) != (idr["drop_reason"] is not None)
        for playout in arm["playout"]:
            assert playout["on_time_frames"] + playout["late_decoded_frames"] + playout["undecodable_frames"] == frames
        if name in LIVE_ARMS:
            controller = report["controllers"][name]
            assert controller["source_frames_encoded"] == frames
            assert controller["source_offered_wire_bytes"] == arm["encoded_wire_bytes"]
    original, replay = report["queue_service_live"], report["immediate_replay_queue_source"]
    for field in ("encoded_source_sha256", "encoded_idr_seqs", "encoded_wire_bytes",
                  "encoded_idr_wire_bytes", "matched_frame_rgb_psnr_db"):
        assert original[field] == replay[field]
    assert report["immediate_replay_queue_source"]["sender_discarded_frames"] == 0
    for name in ("target_paced_live", "fixed_service_live", "queue_service_live"):
        local = report["senders"][name]
        assert local["peak_pending_frames"] <= 1 and local["pending_frames"] == 0
        assert len(local["requests"]) <= 1
        assert len(local["requests"]) == len(local["applied"]) + len(local["cancelled"]) + local["requests_pending"]
        assert all(row["source_ms"] > row["requested_at_ms"] for row in local["applied"])
        assert len(report["controllers"][name]["encoder_restarts"]) <= 1
    json.dumps(report, allow_nan=False)
