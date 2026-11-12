"""v29 causal delivery feedback and sender-load accounting, without a codec."""
from __future__ import annotations

from dataclasses import replace
import json
import math

import pytest

from colabcola.bridge_video_controller import VirtualAccessUnitLink
from colabcola.bridge_video_impairment import LinkConfig, ScheduledUnit
from colabcola.bridge_video_load import LoadControlConfig, LoadRateReceiver, MAX_LOAD_REPORTS
from colabcola.bridge_video_rate import RateConfig, stream_with_rate_control
from colabcola.bridge_video_stream import VideoAccessUnit, packet_bytes


def _rates(**changes):
    return replace(RateConfig(
        congested_age_ms=100, recovered_age_ms=50, congestion_samples=3,
        recovery_samples=3, control_delay_ms=20, min_switch_ms=0,
        max_switches=12, silence_timeout_ms=0), **changes)


def _policy(*, rates=None, load=None, fps=10, wire_aware=True):
    return LoadRateReceiver(_rates() if rates is None else rates,
                            LoadControlConfig() if load is None else load,
                            fps=fps, wire_aware=wire_aware)


def _packet(seq, size=1000, *, fps=10, keyframe=False):
    stamp = round(seq * 1_000_000_000 / fps)
    return VideoAccessUnit(seq, stamp, stamp, b"x" * size, keyframe)


def _arrival(packet, at_ms, *, fps=10):
    return ScheduledUnit(packet, packet.seq * 1000.0 / fps, at_ms,
                         len(packet_bytes(packet)))


def _offer(policy, seq, target=650, size=1000, *, keyframe=False):
    packet = _packet(seq, size, fps=policy.fps, keyframe=keyframe)
    policy.observe_encoded(packet, seq * 1000.0 / policy.fps, target)
    return packet


def _congested_sample(*, anchor_size=50_000, later_sizes=(5000, 5000, 5000),
                      keyframe_seq=None, wire_aware=True, load=None):
    policy = _policy(wire_aware=wire_aware, load=load)
    packets = []
    sizes = (anchor_size, 1000, 1000, *later_sizes)
    for seq, size in enumerate(sizes):
        now = seq * 100.0
        if seq >= 3:
            policy.observe(_arrival(packets[seq - 3], now))
        policy.advance(now)
        packets.append(_offer(policy, seq, size=size, keyframe=seq == keyframe_seq))
    return policy


def _silence_run(*, max_switches=12, frames=14, control_delay_ms=20):
    policy = _policy(rates=_rates(
        silence_timeout_ms=100, min_switch_ms=200, max_switches=max_switches,
        control_delay_ms=control_delay_ms))
    packet = _offer(policy, 0)
    policy.observe(_arrival(packet, 40))
    target = 650
    for seq in range(1, frames):
        now = seq * 100.0
        policy.advance(now)
        target = policy.apply_due(seq, now, target)
        _offer(policy, seq, target)
    return policy, target


def test_load_configuration_and_source_contract_are_bounded():
    for change in (
        {"min_kbps": 19}, {"min_kbps": True}, {"source_window_ms": 0},
        {"receiver_window_ms": math.inf}, {"min_delivery_span_ms": 301},
        {"feedback_freshness_ms": -1}, {"safety_fraction": math.nan},
        {"safety_fraction": 0}, {"safety_fraction": 1.01},
        {"decrease_factor": 1}, {"decrease_factor": True},
        {"increase_factor": 1}, {"increase_factor": 4.01},
    ):
        with pytest.raises(ValueError):
            LoadControlConfig(**change)
    for change in ({"fps": True}, {"fps": 0}, {"wire_aware": 1}):
        with pytest.raises(ValueError):
            _policy(**change)
    with pytest.raises(ValueError):
        _policy(load=LoadControlConfig(min_kbps=650))


def test_delivery_rate_excludes_the_anchor_packet_and_wire_cap_changes_target():
    big_anchor = _congested_sample(anchor_size=50_000)
    small_anchor = _congested_sample(anchor_size=1000)
    for policy in (big_anchor, small_anchor):
        request = policy.requests[-1]
        assert request["delivery_packet_count"] == 3
        assert request["delivery_span_ms"] == 200
        assert request["delivery_wire_bytes_after_anchor"] == 2060
        assert request["observed_delivery_kbps"] == pytest.approx(82.4)
        assert policy.apply_due(6, 600, 650) == 103
        decision = policy.report()["decisions"][-1]
        assert decision["measured_payload_kbps"] == pytest.approx(400)
        assert decision["measured_framing_kbps"] == pytest.approx(2.4)
        assert decision["cut_base_kbps"] == 325
        assert decision["wire_candidate_kbps"] == pytest.approx(103.22)
        assert decision["wire_cap_used"] is True
        assert decision["wire_cap_binding"] is True


@pytest.mark.parametrize("seqs, arrivals", [
    ((0, 1), (300, 500)),
    ((0, 1, 2), (300, 300, 300)),
    ((0, 1, 2), (300, 310, 320)),
    ((0, 2, 3), (300, 500, 600)),
])
def test_sparse_compressed_or_gapped_delivery_cannot_supply_a_wire_cap(seqs, arrivals):
    policy = _policy(rates=_rates(congestion_samples=len(seqs)))
    for seq, at_ms in zip(seqs, arrivals):
        policy.observe(_arrival(_packet(seq), at_ms))
    request = policy.requests[-1]
    assert request["observed_delivery_kbps"] is None
    assert request["delivery_unavailable_reason"]


def test_same_rate_idr_cost_stays_in_the_sender_sample():
    policy = _congested_sample(later_sizes=(1000, 10_000, 1000), keyframe_seq=4)
    policy.apply_due(6, 600, 650)
    decision = policy.report()["decisions"][-1]
    assert decision["target_epoch_seq"] == 0
    assert decision["source_sample_first_seq"] == 3
    assert decision["source_sample_frames"] == 3
    assert decision["measured_payload_kbps"] == pytest.approx(320)
    assert decision["measured_framing_kbps"] == pytest.approx(2.4)


def test_returning_to_an_old_numeric_target_does_not_reuse_its_old_source_epoch():
    policy = _policy(load=LoadControlConfig(source_window_ms=2000))
    packets = []
    for seq in range(12):
        now = seq * 100.0
        if seq >= 9:
            policy.observe(_arrival(packets[seq - 3], now))
        policy.advance(now)
        target = 325 if 3 <= seq < 6 else 650
        size = 1000 if seq >= 6 else 10_000
        packets.append(_offer(policy, seq, target, size))
    policy.apply_due(12, 1200, 650)
    decision = policy.report()["decisions"][-1]
    assert decision["target_epoch_seq"] == 6
    assert decision["source_sample_first_seq"] == 6
    assert decision["source_sample_frames"] == 6
    assert decision["measured_payload_kbps"] == pytest.approx(80)


def test_feedback_arriving_exactly_at_a_source_frame_waits_and_is_consumed_once():
    policy = _policy(fps=25, rates=_rates(congestion_samples=1), wire_aware=False)
    packets = [_offer(policy, seq) for seq in range(3)]
    policy.observe(_arrival(packets[0], 100, fps=25))
    assert policy.requests[0]["reaches_source_ms"] == 120
    policy.advance(120)
    assert policy.apply_due(3, 120, 650) == 650
    _offer(policy, 3)
    policy.advance(160)
    assert policy.apply_due(4, 160, 650) == 325
    _offer(policy, 4, 325)
    policy.advance(200)
    assert policy.apply_due(5, 200, 325) == 325
    assert len(policy.applied) == 1
    assert len(policy.report()["decisions"]) == 1


def test_feedback_from_the_previous_target_epoch_cannot_cut_again():
    policy = _policy(fps=25, rates=_rates(
        congestion_samples=1, control_delay_ms=100), wire_aware=False)
    target = 650
    packets = []
    for seq in range(8):
        now = seq * 40.0
        if seq in (3, 4):
            policy.observe(_arrival(packets[seq - 3], now - 20, fps=25))
        policy.advance(now)
        target = policy.apply_due(seq, now, target)
        packets.append(_offer(policy, seq, target))
    assert target == 325
    assert len(policy.applied) == 1
    decisions = policy.report()["decisions"]
    assert [x["source_seq"] for x in decisions] == [6, 7]
    assert decisions[-1]["fallback_reason"] == "previous_target_epoch"


def test_stale_feedback_is_consumed_without_spending_a_change():
    policy = _policy(fps=25, rates=_rates(
        congestion_samples=1, control_delay_ms=500), wire_aware=False)
    packets = []
    target = 650
    for seq in range(17):
        now = seq * 40.0
        if seq == 3:
            policy.observe(_arrival(packets[0], 100, fps=25))
        policy.advance(now)
        target = policy.apply_due(seq, now, target)
        packets.append(_offer(policy, seq, target))
    assert target == 650
    assert policy.applied == []
    assert policy.report()["decisions"][-1]["fallback_reason"] == "stale_feedback"
    assert policy.report()["unapplied_commands"] == 0


def test_gap_alone_does_not_request_a_rate_cut():
    policy = _policy()
    for seq in (0, 2, 3):
        policy.observe(_arrival(_packet(seq), seq * 100 + 40))
    assert policy.gaps == 1
    assert policy.requests == []


def test_healthy_probe_can_exceed_application_limited_delivery_rate():
    policy = _policy(fps=25, rates=_rates(
        congestion_samples=1, recovery_samples=6))
    target = 650
    packets = []
    for seq in range(13):
        now = seq * 40.0
        if seq == 3:
            policy.observe(_arrival(packets[0], 100, fps=25))
        elif seq >= 5:
            policy.observe(_arrival(packets[seq - 1], now, fps=25))
        policy.advance(now)
        target = policy.apply_due(seq, now, target)
        packets.append(_offer(policy, seq, target, size=100))
    assert [x["target_kbps"] for x in policy.applied] == [325, 407]
    decision = policy.report()["decisions"][-1]
    assert decision["trigger"] == "healthy_probe"
    assert decision["observed_delivery_kbps"] == pytest.approx(26)
    assert decision["chosen_target_kbps"] > decision["observed_delivery_kbps"]
    assert decision["wire_cap_used"] is False


def test_repeated_silence_reduces_to_floor_without_inventing_a_delivery_estimate():
    unarmed = _policy(rates=_rates(silence_timeout_ms=100))
    unarmed.advance(10_000)
    assert unarmed.requests == []
    policy, target = _silence_run()
    assert target == 20
    assert [x["target_kbps"] for x in policy.applied] == [325, 162, 81, 40, 20]
    assert all(x["trigger"] == "receiver_silence" for x in policy.requests)
    assert all(x["observed_delivery_kbps"] is None for x in policy.requests)
    assert all(not x["wire_cap_used"] for x in policy.report()["decisions"])
    assert policy.report()["decisions"][-1]["min_target_saturated"] is True
    policy.stop_watchdog()
    previous = len(policy.requests)
    policy.advance(20_000)
    assert len(policy.requests) == previous


def test_rate_change_budget_is_separate_from_the_feedback_report_budget():
    policy, target = _silence_run(max_switches=3)
    assert target == 81
    assert policy.report()["switches_applied"] == 3
    assert len(policy.requests) > 3
    assert policy.report()["decisions"][-1]["fallback_reason"] == "rate_change_budget_exhausted"
    policy.advance(1_000_000)
    assert len(policy.requests) == MAX_LOAD_REPORTS
    json.dumps(policy.report(), allow_nan=False)


def test_stale_silence_reports_cannot_keep_cutting_a_later_source():
    policy, target = _silence_run(control_delay_ms=500)
    assert target == 650
    assert policy.applied == []
    decisions = policy.report()["decisions"]
    assert decisions
    assert all(x["trigger"] == "receiver_silence" for x in decisions)
    assert all(x["fallback_reason"] == "stale_feedback" for x in decisions)
    assert all(x["observed_delivery_kbps"] is None for x in decisions)


def test_arrival_wins_initial_and_repeated_silence_deadline_ties():
    policy = _policy(rates=_rates(
        silence_timeout_ms=100, congested_age_ms=2000, recovery_samples=30))
    policy.observe(_arrival(_packet(0), 40))
    policy.advance(139.999)
    policy.observe(_arrival(_packet(1), 140))
    assert policy.requests == []
    policy.advance(240)
    assert [x["generated_at_ms"] for x in policy.requests] == [240]
    policy.advance(339.999)
    policy.observe(_arrival(_packet(2), 340))
    assert [x["generated_at_ms"] for x in policy.requests] == [240]
    policy.advance(440)
    assert [x["generated_at_ms"] for x in policy.requests] == [240, 440]


def test_overhead_budget_failure_is_visible_when_clipping_to_the_minimum():
    policy = _policy(load=LoadControlConfig(receiver_window_ms=500))
    packets = []
    for seq in range(8):
        now = seq * 100.0
        if seq in (3, 5, 7):
            policy.observe(_arrival(packets[(seq - 3) // 2], now))
        policy.advance(now)
        packets.append(_offer(policy, seq, size=1 if seq < 3 else 1000))
    assert policy.apply_due(8, 800, 650) == 20
    decision = policy.report()["decisions"][-1]
    assert decision["observed_delivery_kbps"] == pytest.approx(1.24)
    assert decision["wire_budget_kbps"] == pytest.approx(0.992)
    assert decision["measured_framing_kbps"] == pytest.approx(2.4)
    assert decision["wire_candidate_kbps"] == 0
    assert decision["min_target_saturated"] is True
    assert decision["overhead_budget_failure"] is True


def test_sender_sampling_requires_the_declared_frame_clock_without_mutating_on_failure():
    policy = _policy()
    _offer(policy, 0)
    with pytest.raises(ValueError):
        policy.observe_encoded(_packet(1), 110, 650)
    assert policy.source_frames_encoded == 1
    _offer(policy, 1)
    assert policy.source_frames_encoded == 2


def test_source_accounting_includes_admission_rejections_and_all_frame_headers():
    policy = _policy(fps=30, rates=_rates(max_switches=0))
    link = VirtualAccessUnitLink(LinkConfig(
        delay_ms=500, jitter_ms=0, bandwidth_kbps=0, max_inflight=1, max_age_ms=0))
    packets = []
    for seq in range(12):
        now = seq * 1000 / 30
        for arrived in link.arrive_by(now):
            policy.observe(arrived)
        policy.advance(now)
        packet = _offer(policy, seq, size=9000 if seq == 6 else 1000, keyframe=seq in (0, 6))
        packets.append(packet)
        link.send(packet, fps=30)
    policy.stop_watchdog()
    for arrived in link.arrive_by(math.inf):
        policy.observe(arrived)
    trace = link.snapshot(12)
    report = policy.report()
    assert trace.loss_reasons["queue_full"] == 11
    assert report["source_frames_encoded"] == 12
    assert report["source_offered_payload_bytes"] == sum(len(p.data) for p in packets)
    assert report["source_offered_wire_bytes"] == sum(len(packet_bytes(p)) for p in packets)
    assert report["source_offered_wire_bytes"] - report["source_offered_payload_bytes"] == 12 * 30
    assert report["source_offered_wire_bytes"] > trace.wire_sent_bytes


def test_stream_injection_rejects_a_used_or_mismatched_policy_before_encoding():
    rates = _rates()
    args = dict(width=96, height=64, fps=10, frames=12, gop=12,
                link=LinkConfig(), rates=rates, adapt=True)
    used = _policy(rates=rates)
    _offer(used, 0)
    for policy in (used, _policy(rates=replace(rates, control_delay_ms=21)), object()):
        with pytest.raises(ValueError, match="rate_policy"):
            stream_with_rate_control(**args, rate_policy=policy)
    with pytest.raises(ValueError, match="rate_policy"):
        stream_with_rate_control(**{**args, "adapt": False}, rate_policy=_policy(rates=rates))
