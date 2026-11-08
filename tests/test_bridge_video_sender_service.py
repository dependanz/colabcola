"""Independent service budgets, mutation clocks, and complete framed work."""
from __future__ import annotations

import json
import math

import pytest

from colabcola.bridge_video_sender import LocalSenderQueue, SenderQueueConfig
from colabcola.bridge_video_sender_service import SenderServiceConfig, ServiceSenderQueue
from colabcola.bridge_video_stream import PACKET_HEADER, VideoAccessUnit


FPS = 50  # Initial 188 payload kbps plus framing gives a 200 kbps floor.


def _packet(seq, *, wire_size=1000, fps=FPS, idr=None):
    idr = seq == 0 if idr is None else idr
    prefix = b"\x00\x00\x00\x01" + (b"\x65" if idr else b"\x41")
    size = wire_size - PACKET_HEADER.size
    assert size >= len(prefix)
    stamp = round(seq * 1_000_000_000 / fps)
    return VideoAccessUnit(seq, stamp, stamp, prefix + b"x" * (size - len(prefix)), idr)


def _sender(*, service=None, fps=FPS, target=188, **queue_options):
    config = SenderQueueConfig(**{
        "max_pending": 120, "max_age_ms": 0, "max_repair_requests": 0,
        **queue_options,
    })
    return ServiceSenderQueue(
        config, fps=fps, initial_target_kbps=target,
        service=SenderServiceConfig() if service is None else service)


def _capture(sender, seq, **packet_options):
    now = seq * 1000.0 / sender.fps
    events = sender.advance(now)
    sender.enqueue(_packet(seq, fps=sender.fps, **packet_options), at_ms=now)
    return events + sender.advance(now)


def _accounting(sender):
    report = sender.report()
    assert report["offered_wire_bytes"] == (
        report["dispatched_wire_bytes"] + report["dropped_wire_bytes"]
        + report["pending_wire_bytes"])
    assert report["frames_observed"] == (
        report["attempted_dispatches"] + len(report["drops"]) + report["pending_frames"])
    attempts = {row["source_seq"] for row in report["dispatches"]}
    drops = {row["source_seq"] for row in report["drops"]}
    assert not attempts & drops
    if sender.pending_count == 0:
        assert attempts | drops == set(range(report["frames_observed"]))
    json.dumps(report, allow_nan=False)
    return report


def test_service_configuration_rejects_invalid_values_and_unpaced_use():
    for changes in (
        {"queue_boost": 0}, {"queue_boost": "yes"},
        {"drain_ms": 0}, {"drain_ms": 2001}, {"drain_ms": True},
        {"drain_ms": math.inf}, {"drain_ms": math.nan},
        {"max_rate_factor": 0.99}, {"max_rate_factor": 8.01},
        {"max_rate_factor": True}, {"max_rate_factor": math.inf},
        {"max_rate_factor": math.nan},
    ):
        with pytest.raises(ValueError):
            SenderServiceConfig(**changes)
    with pytest.raises(ValueError, match="service configuration"):
        ServiceSenderQueue(SenderQueueConfig(), fps=FPS,
                           initial_target_kbps=188, service={})
    with pytest.raises(ValueError, match="requires pacing"):
        _sender(pacing=False)
    # Validation boundaries are accepted, without weakening the base sender.
    SenderServiceConfig(drain_ms=1, max_rate_factor=1)
    SenderServiceConfig(drain_ms=2000, max_rate_factor=8)
    with pytest.raises(ValueError, match="sender source rate"):
        _sender(fps=True)


def test_fixed_service_retains_initial_floor_across_encoder_target_changes():
    sender = _sender(service=SenderServiceConfig(queue_boost=False))
    assert sender.base_kbps == 200 and sender.cap_kbps == 400
    assert sender.report()["service"]["decisions"] == []
    assert _capture(sender, 0)[0].pacing_kbps == 200
    sender.advance(10)
    assert sender.debt_bits == 6000
    before = sender.report()["service"]["decisions"]
    sender.set_target(20, at_ms=10)
    assert sender.report()["service"]["decisions"] == before
    assert sender.pacing_kbps == 200 and sender.cap_kbps == 400
    assert _capture(sender, 1) == []
    before = sender.report()["service"]["decisions"]
    sender.set_target(388, at_ms=20)
    assert sender.report()["service"]["decisions"] == before
    assert sender.advance(39.999) == []
    dispatched = sender.advance(40)
    assert [(x.packet.seq, x.dispatch_ms, x.pacing_kbps) for x in dispatched] == [(1, 40, 200)]
    report = _accounting(sender)
    assert [x["target_kbps"] for x in report["rate_changes"]] == [20, 388]
    assert report["service"]["current_kbps"] == 200
    assert not any(x["boosted"] or x["cap_limited"] for x in report["service"]["decisions"])


def test_queue_arrival_pays_old_rate_then_changes_service_without_repricing_debt():
    sender = _sender()
    assert _capture(sender, 0, wire_size=1250)[0].dispatch_ms == 0
    # 10,000 initial bits, with 4,000 paid at 200 kbps by the next capture.
    sender.advance(20)
    assert sender.debt_bits == 6000
    sender.enqueue(_packet(1, wire_size=1250), at_ms=20)
    assert sender.debt_bits == 6000
    assert sender.pacing_kbps == 320  # (6,000 debt + 10,000 pending) / 50.
    decisions = sender.report()["service"]["decisions"]
    assert sender.advance(30) == []
    assert sender.debt_bits == 2800  # Elapsed 10 ms paid at the old 320 kbps.
    sender.set_target(20, at_ms=30)
    assert sender.pacing_kbps == 320
    assert sender.report()["service"]["decisions"] == decisions
    assert sender.advance(38.749) == []
    dispatched = sender.advance(38.75)
    assert [(x.packet.seq, x.dispatch_ms, x.pacing_kbps) for x in dispatched] == [(1, 38.75, 320)]
    # Pop+charge moves the complete pending unit into debt. It cannot lower
    # the service request using an artificial empty-queue, zero-debt instant.
    last = _accounting(sender)["service"]["decisions"][-1]
    assert last["cause"] == "dispatch"
    assert last["debt_bits"] == last["work_bits"] == 10_000
    assert last["pending_wire_bytes"] == 0
    assert sender.pacing_kbps == 200


def test_service_work_includes_debt_without_pending_units_and_the_last_queued_unit():
    sender = _sender()
    first = _capture(sender, 0, wire_size=1875)
    assert [(x.dispatch_ms, x.pacing_kbps) for x in first] == [(0, 300)]
    assert sender.pending_count == 0 and sender.debt_bits == 15_000
    report = sender.report()
    enqueue, dispatch = report["service"]["decisions"]
    assert enqueue["work_bits"] == dispatch["work_bits"] == 15_000
    assert dispatch["boosted"] and dispatch["pending_wire_bytes"] == 0
    assert dispatch["pacing_kbps"] == 300

    sender.advance(20)
    assert sender.debt_bits == 9000
    sender.enqueue(_packet(1, wire_size=750), at_ms=20)
    last = sender.report()["service"]["decisions"][-1]
    assert last["work_bits"] == 15_000  # Last queued unit contributes 6,000.
    assert last["pacing_kbps"] == 300
    dispatched = sender.advance(50)
    assert [(x.packet.seq, x.dispatch_ms) for x in dispatched] == [(1, 50)]
    assert sender.debt_bits == 6000 and sender.pacing_kbps == 200
    _accounting(sender)


def test_cap_applies_to_debt_even_when_the_queue_is_empty():
    sender = _sender()
    first = _capture(sender, 0, wire_size=10_000)
    assert first[0].pacing_kbps == 400
    assert sender.pending_count == 0 and sender.debt_bits == 80_000
    after_dispatch = sender.report()["service"]["decisions"][-1]
    assert after_dispatch["requested_kbps"] == 1600
    assert after_dispatch["pacing_kbps"] == 400 and after_dispatch["cap_limited"]
    sender.advance(10)
    assert sender.debt_bits == 76_000
    assert sender.pacing_kbps == 400
    assert sender.report()["service"]["decisions"][-1] == after_dispatch
    _accounting(sender)


def test_expiring_a_whole_chain_removes_pending_work_without_refunding_debt():
    sender = _sender(max_age_ms=30, max_repair_requests=1)
    _capture(sender, 0, wire_size=10_000)
    _capture(sender, 1, wire_size=100)
    _capture(sender, 2, wire_size=100)
    assert sender.advance(50) == []
    assert sender.pending_count == 0
    assert sender.debt_bits == 60_000  # 80,000 - 50 * 400; no drop refund.
    report = _accounting(sender)
    assert [(x["source_seq"], x["reason"], x["at_ms"]) for x in report["drops"]] == [
        (1, "sender_age", 50), (2, "sender_dependency", 50)]
    expiry = report["service"]["decisions"][-1]
    assert expiry["cause"] == "expiry"
    assert expiry["pending_wire_bytes"] == 0
    assert expiry["debt_bits"] == expiry["work_bits"] == 60_000
    assert expiry["cap_limited"] and expiry["pacing_kbps"] == 400
    assert len(report["requests"]) == 1
    # Suppressed descendants still produce one final enqueue observation;
    # their discarded bytes cannot inflate the pending service request.
    assert _capture(sender, 3, wire_size=100) == []
    suppressed = _accounting(sender)["service"]["decisions"][-1]
    assert suppressed["cause"] == "enqueue" and suppressed["at_ms"] == 60
    assert suppressed["pending_wire_bytes"] == 0
    assert suppressed["work_bits"] == 56_000


def test_overflow_refresh_occurs_after_chain_discard_and_retained_idr_enqueue():
    sender = _sender(service=SenderServiceConfig(drain_ms=2000),
                     max_pending=3, max_repair_requests=1)
    _capture(sender, 0, wire_size=10_000)  # Busy at the 200 kbps floor until t=400.
    _capture(sender, 1, wire_size=1000)
    _capture(sender, 2, wire_size=1000)
    _capture(sender, 3, wire_size=2000, idr=True)
    before = len(sender.report()["service"]["decisions"])
    assert _capture(sender, 4, wire_size=500) == []
    report = _accounting(sender)
    assert len(report["service"]["decisions"]) == before + 1
    final_enqueue = report["service"]["decisions"][-1]
    assert final_enqueue["cause"] == "enqueue" and final_enqueue["at_ms"] == 80
    assert final_enqueue["debt_bits"] == 64_000
    assert final_enqueue["pending_wire_bytes"] == 2500
    assert final_enqueue["work_bits"] == 84_000
    assert [(x["source_seq"], x["reason"]) for x in report["drops"]] == [
        (1, "sender_queue_full"), (2, "sender_dependency")]
    assert report["requests"] == [] and not report["need_idr"]
    assert [(x.packet.seq, x.dispatch_ms) for x in sender.advance(480)] == [(3, 400), (4, 480)]
    _accounting(sender)


def test_idle_time_cannot_bank_credit_and_observations_cannot_lower_service():
    sender = _sender()
    _capture(sender, 0, wire_size=40)
    before = sender.report()["service"]["decisions"]
    sender.advance(20)
    assert sender.debt_bits == 0
    assert sender.report()["service"]["decisions"] == before
    sender.enqueue(_packet(1, wire_size=1875), at_ms=20)
    assert sender.advance(20)[0].dispatch_ms == 20
    assert sender.debt_bits == 15_000 and sender.pacing_kbps == 300
    before = sender.report()["service"]["decisions"]
    sender.advance(30)
    assert sender.debt_bits == 12_000 and sender.pacing_kbps == 300
    assert sender.report()["service"]["decisions"] == before
    assert _capture(sender, 2, wire_size=40) == []
    assert sender.debt_bits == 9000 and sender.pacing_kbps == 200
    assert sender.advance(84.999) == []
    dispatched = sender.advance(85)
    assert [(x.packet.seq, x.dispatch_ms) for x in dispatched] == [(2, 85)]
    _accounting(sender)


@pytest.mark.parametrize("max_age_ms", [0, 45])
def test_fractional_trajectories_do_not_depend_on_extra_advance_or_report_reads(max_age_ms):
    def run(split):
        sender = _sender(fps=30, target=30, max_age_ms=max_age_ms,
                         max_pending=3, max_repair_requests=1)
        events = []

        def advance(end):
            start = sender.clock_ms
            if split:
                for fraction in (0.19, 0.43, 0.77):
                    events.extend(sender.advance(start + (end - start) * fraction))
                    snapshot = sender.report()
                    assert sender.report() == snapshot
            events.extend(sender.advance(end))

        for seq, size in enumerate((620, 100, 105, 700, 110, 115, 800, 120)):
            now = seq * 1000.0 / 30
            advance(now)
            if seq == 4:
                sender.set_target(20, at_ms=now)
            sender.enqueue(_packet(seq, fps=30, wire_size=size, idr=seq in (0, 3, 6)), at_ms=now)
            events.extend(sender.advance(now))
        advance(1000)
        return events, _accounting(sender)

    direct_events, direct = run(False)
    split_events, split = run(True)
    assert [x.packet for x in direct_events] == [x.packet for x in split_events]
    assert [x.dispatch_ms for x in direct_events] == pytest.approx(
        [x.dispatch_ms for x in split_events], abs=1e-9, rel=0)
    assert [x.pacing_kbps for x in direct_events] == pytest.approx(
        [x.pacing_kbps for x in split_events], abs=1e-9, rel=0)
    assert direct["drops"] == split["drops"]
    assert direct["requests"] == split["requests"]
    assert direct["dispatched_wire_bytes"] == split["dispatched_wire_bytes"]
    assert direct["dropped_wire_bytes"] == split["dropped_wire_bytes"]
    a, b = direct["service"]["decisions"], split["service"]["decisions"]
    assert len(a) == len(b)
    for one, two in zip(a, b):
        assert {k: v for k, v in one.items() if type(v) is not float} == {
            k: v for k, v in two.items() if type(v) is not float}
        assert {k: v for k, v in one.items() if type(v) is float} == pytest.approx(
            {k: v for k, v in two.items() if type(v) is float}, abs=1e-8, rel=0)
    if max_age_ms:
        assert any(x["reason"] == "sender_age" for x in direct["drops"])


def test_fractional_service_expiry_tie_keeps_actual_endpoint_and_never_dispatches_early():
    sender = _sender(fps=30, target=30, max_age_ms=100,
                     max_pending=3, max_repair_requests=1)
    # Doubled v31 fractional tie size, with the queue-aware service at its
    # doubled 74.4 kbps cap. The mathematical readiness is still 400/3 ms.
    first = _capture(sender, 0, wire_size=1240)
    assert first[0].pacing_kbps == 74.4
    _capture(sender, 1, wire_size=100)
    _capture(sender, 2, wire_size=100)
    early = 1240 * 8 / 74.4
    expiry = 1000.0 / 30 + 100
    assert early < expiry and expiry - early < 1e-9
    assert sender.next_event_ms == expiry
    before = sender.report()["service"]["decisions"]
    assert sender.advance(early) == []
    assert sender.pending_count == 2 and sender.drops == []
    assert sender.report()["service"]["decisions"] == before
    assert sender.next_event_ms == expiry > sender.clock_ms
    assert sender.advance(early) == []
    assert sender.advance(expiry) == []
    assert [(x["source_seq"], x["reason"], x["at_ms"]) for x in sender.drops] == [
        (1, "sender_age", expiry), (2, "sender_dependency", expiry)]
    assert sender.debt_bits == 0 and sender.pacing_kbps == 37.2
    assert sender.next_event_ms is None
    assert _accounting(sender)["service"]["decisions"][-1]["cause"] == "expiry"


def test_tail_is_finite_and_only_actual_tail_mutations_refresh_service():
    sender = _sender()
    events = []
    for seq, size in enumerate((2500, 1000, 1000, 1000)):
        events.extend(_capture(sender, seq, wire_size=size))
    assert sender.clock_ms == 60 and sender.pending_count == 2
    sender.set_target(20, at_ms=60)  # The source target is independent of service.
    steps = 0
    while sender.next_event_ms is not None:
        at = sender.next_event_ms
        assert math.isfinite(at) and at >= sender.clock_ms
        events.extend(sender.advance(at))
        steps += 1
        assert steps <= 4
    assert [(x.packet.seq, x.dispatch_ms) for x in events] == [(0, 0), (1, 50), (2, 72), (3, 97)]
    report = _accounting(sender)
    assert report["pending_wire_bytes"] == 0
    assert report["remaining_debt_bits"] == 8000
    assert report["service"]["current_kbps"] == 200
    tail = [x for x in report["service"]["decisions"] if x["at_ms"] > 60]
    assert [(x["cause"], x["at_ms"], x["pacing_kbps"]) for x in tail] == [
        ("dispatch", 72, 320), ("dispatch", 97, 200)]
    assert len(report["rate_changes"]) == 1


@pytest.mark.parametrize("max_age_ms", [0, 35])
def test_fixed_service_with_unchanged_target_preserves_complete_legacy_sender_report(max_age_ms):
    config = SenderQueueConfig(max_pending=3, max_age_ms=max_age_ms, max_repair_requests=1)
    legacy = LocalSenderQueue(config, fps=FPS, initial_target_kbps=188)
    fixed = ServiceSenderQueue(config, fps=FPS, initial_target_kbps=188,
                               service=SenderServiceConfig(queue_boost=False))
    for seq, size in enumerate((1000, 1250, 100, 2000, 500, 750)):
        kwargs = dict(wire_size=size, idr=seq in (0, 3))
        assert _capture(legacy, seq, **kwargs) == _capture(fixed, seq, **kwargs)
    assert legacy.advance(1000) == fixed.advance(1000)
    legacy_report = legacy.report()
    fixed_report = _accounting(fixed)
    service_report = fixed_report.pop("service")
    assert fixed_report == legacy_report
    assert not any(x["boosted"] for x in service_report["decisions"])
