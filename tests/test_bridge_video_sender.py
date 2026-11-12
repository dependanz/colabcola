"""Causal sender clocks, framed-bit pacing and unsent dependency handling."""
from __future__ import annotations

import json
import math

import pytest

from colabcola.bridge_video_controller import VirtualAccessUnitLink
from colabcola.bridge_video_impairment import LinkConfig
from colabcola.bridge_video_sender import LocalSenderQueue, SenderQueueConfig
from colabcola.bridge_video_stream import PACKET_HEADER, VideoAccessUnit, is_keyframe


FPS = 50  # Framing contributes 12 kbps, so target 188 gives 200 wire kbps.


def _packet(seq, *, wire_size=1000, idr=None, annex_idr=None, fps=FPS):
    """Synthetic Annex-B units for scheduler tests; no fake decoder result."""
    idr = seq == 0 if idr is None else idr
    annex_idr = idr if annex_idr is None else annex_idr
    prefix = b"\x00\x00\x00\x01" + (b"\x65" if annex_idr else b"\x41")
    payload_size = wire_size - PACKET_HEADER.size
    assert payload_size >= len(prefix)
    timestamp = round(seq * 1_000_000_000 / fps)
    return VideoAccessUnit(seq, timestamp, timestamp,
                           prefix + b"x" * (payload_size - len(prefix)), idr)


def _sender(**changes):
    options = dict(max_pending=120, max_age_ms=0, max_repair_requests=0)
    options.update(changes)
    return LocalSenderQueue(SenderQueueConfig(**options), fps=FPS,
                            initial_target_kbps=188)


def _capture(sender, seq, **packet_options):
    now = seq * 1000.0 / FPS
    dispatched = sender.advance(now)
    sender.enqueue(_packet(seq, **packet_options), at_ms=now)
    return dispatched + sender.advance(now)


def _assert_accounting(sender):
    report = sender.report()
    assert report["offered_wire_bytes"] == (
        report["dispatched_wire_bytes"] + report["dropped_wire_bytes"]
        + report["pending_wire_bytes"])
    assert report["frames_observed"] == (
        report["attempted_dispatches"] + len(report["drops"])
        + report["pending_frames"])
    dispatched = {row["source_seq"] for row in report["dispatches"]}
    dropped = {row["source_seq"] for row in report["drops"]}
    assert not dispatched & dropped
    if not sender.pending_count:
        assert dispatched | dropped == set(range(report["frames_observed"]))
    json.dumps(report, allow_nan=False)
    return report


def test_sender_configuration_and_source_bounds_are_strict():
    for changes in (
        {"pacing": 0}, {"pacing": "yes"},
        {"max_pending": 0}, {"max_pending": 121}, {"max_pending": True},
        {"max_age_ms": -1}, {"max_age_ms": math.inf},
        {"max_age_ms": math.nan}, {"max_age_ms": True}, {"max_age_ms": 2001},
        {"max_repair_requests": -1}, {"max_repair_requests": 9},
        {"max_repair_requests": True},
    ):
        with pytest.raises(ValueError):
            SenderQueueConfig(**changes)
    config = SenderQueueConfig()
    for fps, target in ((0, 188), (61, 188), (True, 188),
                        (50, 19), (50, 20_001), (50, True)):
        with pytest.raises(ValueError):
            LocalSenderQueue(config, fps=fps, initial_target_kbps=target)
    with pytest.raises(ValueError):
        LocalSenderQueue({}, fps=FPS, initial_target_kbps=188)


def test_actions_require_the_current_clock_and_consecutive_capture_ids():
    sender = _sender()
    for operation in (
        lambda: sender.enqueue(_packet(1), at_ms=0),
        lambda: sender.enqueue(_packet(0), at_ms=20),
        lambda: sender.set_target(88, at_ms=10),
        lambda: sender.take_repair_before(10, 0),
    ):
        with pytest.raises(ValueError):
            operation()
    assert sender.report()["frames_observed"] == 0
    assert sender.report()["offered_wire_bytes"] == 0
    _capture(sender, 0)
    for invalid in (-1, math.inf, math.nan, True):
        with pytest.raises(ValueError):
            sender.advance(invalid)
    sender.advance(20)
    with pytest.raises(ValueError):
        sender.enqueue(_packet(1), at_ms=19)
    sender.enqueue(_packet(1), at_ms=20)
    with pytest.raises(ValueError):
        sender.enqueue(_packet(1), at_ms=20)
    assert _assert_accounting(sender)["frames_observed"] == 2


@pytest.mark.parametrize("target, expected_dispatch", [(88, 70.0), (388, 25.0)])
def test_target_change_preserves_debt_and_integrates_the_old_rate_first(target, expected_dispatch):
    sender = _sender()
    first = _capture(sender, 0)
    assert first[0].dispatch_ms == 0
    assert first[0].pacing_kbps == 200
    # Of the first 8,000 bits, 2,000 have been paid at the old rate by t=10.
    assert sender.advance(10) == []
    sender.set_target(target, at_ms=10)
    assert _capture(sender, 1) == []
    assert sender.advance(expected_dispatch - 0.001) == []
    released = sender.advance(expected_dispatch)
    assert [x.packet.seq for x in released] == [1]
    assert released[0].dispatch_ms == pytest.approx(expected_dispatch)
    assert released[0].queue_delay_ms == pytest.approx(expected_dispatch - 20)
    _assert_accounting(sender)


def test_idle_time_does_not_bank_credit_for_the_next_large_unit():
    sender = _sender()
    assert _capture(sender, 0, wire_size=40)[0].dispatch_ms == 0
    # First debt ends at 1.6 ms, leaving 18.4 ms idle before the next capture.
    second = _capture(sender, 1, wire_size=1000)
    assert second[0].dispatch_ms == 20
    assert _capture(sender, 2, wire_size=40) == []
    assert sender.advance(59.999) == []
    assert sender.advance(60)[0].packet.seq == 2
    _assert_accounting(sender)


def test_expiry_wins_an_exact_pacer_dispatch_tie():
    sender = _sender(max_age_ms=20)
    _capture(sender, 0, wire_size=1000)  # Debt clears at 40 ms.
    assert _capture(sender, 1, wire_size=100) == []  # Expires at 40 ms too.
    assert sender.next_event_ms == 40
    assert sender.advance(40) == []
    report = _assert_accounting(sender)
    assert [(x["source_seq"], x["reason"], x["at_ms"]) for x in report["drops"]] == [
        (1, "sender_age", 40)]
    assert sender.pending_count == 0
    assert sender.next_event_ms is None


def test_fractional_expiry_tie_waits_for_the_actual_deadline_without_dispatching_early():
    fps = 30
    sender = LocalSenderQueue(
        SenderQueueConfig(max_pending=3, max_age_ms=100, max_repair_requests=8),
        fps=fps, initial_target_kbps=30)
    sender.enqueue(_packet(0, wire_size=620, fps=fps), at_ms=0)
    first = sender.advance(0)
    assert [(x.packet.seq, x.dispatch_ms, x.pacing_kbps) for x in first] == [(0, 0, 37.2)]
    for seq in (1, 2):
        capture = seq * 1000.0 / fps
        assert sender.advance(capture) == []
        sender.enqueue(_packet(seq, wire_size=100, fps=fps), at_ms=capture)
        assert sender.advance(capture) == []

    # Mathematically both events occur at 400/3 ms. Binary arithmetic makes
    # readiness about 3e-14 ms earlier than the oldest unit's expiry.
    early_ready = 620 * 8 / 37.2
    expiry = 1000.0 / fps + 100
    assert early_ready < expiry
    assert expiry - early_ready < 1e-9
    assert sender.next_event_ms == expiry
    assert sender.advance(early_ready) == []
    assert sender.clock_ms == early_ready
    assert sender.pending_count == 2
    assert sender.drops == sender.requests == []
    assert sender.next_event_ms == expiry > sender.clock_ms
    # Reaching readiness alone must not create a zero-time dispatch loop or
    # process an expiry beyond advance()'s requested observation frontier.
    assert sender.advance(early_ready) == []
    assert sender.next_event_ms == expiry

    assert sender.advance(expiry) == []
    assert [(x["source_seq"], x["reason"], x["at_ms"]) for x in sender.drops] == [
        (1, "sender_age", expiry), (2, "sender_dependency", expiry)]
    assert len(sender.requests) == 1
    assert sender.requests[0]["created_at_ms"] == expiry
    assert sender.debt_bits == 0
    assert sender.next_event_ms is None
    report = _assert_accounting(sender)
    assert report["attempted_dispatches"] == 1
    assert report["dispatched_wire_bytes"] == 620
    assert report["dropped_wire_bytes"] == 200
    assert report["requests_pending"] == 1


def test_expiring_a_chain_does_not_refund_debt_and_a_new_idr_cancels_repair():
    sender = _sender(max_age_ms=20, max_repair_requests=8)
    _capture(sender, 0, wire_size=1250)  # 10,000 bits require 50 ms.
    _capture(sender, 1, wire_size=100)
    assert sender.advance(40) == []
    assert sender.take_repair_before(40, 2) is None
    sender.enqueue(_packet(2, wire_size=100, idr=True), at_ms=40)
    assert sender.advance(40) == []
    assert sender.advance(49.999) == []
    released = sender.advance(50)
    assert [(x.packet.seq, x.dispatch_ms) for x in released] == [(2, 50)]
    sender.advance(60)
    assert sender.take_repair_before(60, 3) is None
    report = _assert_accounting(sender)
    assert len(report["requests"]) == len(report["cancelled"]) == 1
    assert report["applied"] == []
    assert report["cancelled"][0]["reason"] == "new_idr_available"


@pytest.mark.parametrize("verified", [True, False])
def test_overflow_discards_descendants_only_until_a_verified_queued_idr(verified):
    sender = _sender(max_pending=3, max_repair_requests=8)
    _capture(sender, 0, wire_size=5000)  # Busy until 200 ms.
    _capture(sender, 1, wire_size=100)
    _capture(sender, 2, wire_size=100)
    candidate = _packet(3, wire_size=100, idr=True, annex_idr=verified)
    assert is_keyframe(candidate.data) is verified
    sender.advance(60)
    sender.enqueue(candidate, at_ms=60)
    sender.advance(60)
    assert _capture(sender, 4, wire_size=100) == []
    report = _assert_accounting(sender)
    assert report["peak_pending_frames"] <= 3
    if verified:
        assert [(x["source_seq"], x["reason"]) for x in report["drops"]] == [
            (1, "sender_queue_full"), (2, "sender_dependency")]
        assert report["requests"] == []
        assert [(x.packet.seq, x.dispatch_ms) for x in sender.advance(204)] == [
            (3, 200), (4, 204)]
    else:
        assert [(x["source_seq"], x["reason"]) for x in report["drops"]] == [
            (1, "sender_queue_full"), (2, "sender_dependency"),
            (3, "sender_dependency"), (4, "sender_dependency")]
        assert sender.pending_count == 0
        assert len(report["requests"]) == 1
        assert sender.advance(204) == []
    _assert_accounting(sender)


def test_local_repair_waits_for_a_later_capture_is_consumed_once_and_has_a_budget():
    sender = _sender(max_pending=1, max_age_ms=20, max_repair_requests=1)
    _capture(sender, 0, wire_size=1250)
    _capture(sender, 1, wire_size=100)
    sender.advance(40)  # Unit 1 expires exactly at the next capture.
    assert sender.take_repair_before(40, 2) is None
    sender.enqueue(_packet(2, wire_size=100), at_ms=40)
    assert sender.advance(40) == []  # A descendant is suppressed, not sent.
    assert len(sender.report()["requests"]) == 1
    sender.advance(60)
    request = sender.take_repair_before(60, 3)
    assert request["created_at_ms"] == 40
    assert sender.take_repair_before(60, 3) is None
    sender.enqueue(_packet(3, wire_size=5000, idr=True), at_ms=60)
    assert sender.advance(60)[0].packet.seq == 3
    _capture(sender, 4, wire_size=100)
    sender.advance(100)  # A second local chain fails, after the budget is spent.
    assert sender.take_repair_before(100, 5) is None
    sender.enqueue(_packet(5, wire_size=100), at_ms=100)
    assert sender.advance(100) == []
    report = _assert_accounting(sender)
    assert len(report["requests"]) == len(report["applied"]) == 1
    assert report["applied"][0]["source_idr_seq"] == 3
    assert report["requests_pending"] == 0
    assert [(x["source_seq"], x["reason"]) for x in report["drops"]] == [
        (1, "sender_age"), (2, "sender_dependency"),
        (4, "sender_age"), (5, "sender_dependency")]


def test_disabled_local_repair_still_suppresses_the_broken_chain_until_periodic_idr():
    sender = _sender(max_pending=1, max_repair_requests=0)
    _capture(sender, 0, wire_size=5000)
    _capture(sender, 1, wire_size=100)
    _capture(sender, 2, wire_size=100)
    _capture(sender, 3, wire_size=100)
    _capture(sender, 4, wire_size=100, idr=True)
    assert [x.packet.seq for x in sender.advance(200)] == [4]
    report = _assert_accounting(sender)
    assert report["requests"] == report["applied"] == []
    assert [x["source_seq"] for x in report["drops"]] == [1, 2, 3]


@pytest.mark.parametrize("max_age, expected_ids, finish", [
    (0, [0, 1, 2, 3], 120), (50, [0, 1, 2], 110),
])
def test_source_end_tail_dispatch_or_expiry_is_finite_and_accounts_for_every_unit(max_age, expected_ids, finish):
    sender = _sender(max_age_ms=max_age, max_repair_requests=8)
    dispatched = []
    for seq in range(4):
        dispatched.extend(_capture(sender, seq))
    assert sender.pending_count == 2
    events = 0
    while sender.pending_count:
        at = sender.next_event_ms
        assert at is not None and math.isfinite(at) and at >= sender.clock_ms
        dispatched.extend(sender.advance(at))
        events += 1
        assert events <= 4
    assert [x.packet.seq for x in dispatched] == expected_ids
    assert sender.clock_ms == finish
    assert sender.next_event_ms is None
    report = _assert_accounting(sender)
    assert report["pending_wire_bytes"] == 0
    assert report["offered_wire_bytes"] == 4000
    if max_age:
        assert report["requests_not_applied_before_source_end"] == 1
        assert report["applied"] == []
    else:
        assert [x.dispatch_ms for x in dispatched] == [0, 40, 80, 120]


def test_downstream_admission_rejection_does_not_refund_attempted_dispatch_debt():
    sender = _sender()
    media = VirtualAccessUnitLink(LinkConfig(
        delay_ms=0, jitter_ms=0, bandwidth_kbps=20, max_inflight=1, max_age_ms=0))
    dispatched = []
    for seq in range(3):
        for unit in _capture(sender, seq):
            dispatched.append(unit)
            media.arrive_by(unit.dispatch_ms)
            media.send(unit.packet, fps=FPS, dispatch_ms=unit.dispatch_ms)
    for unit in sender.advance(80):
        dispatched.append(unit)
        media.arrive_by(unit.dispatch_ms)
        media.send(unit.packet, fps=FPS, dispatch_ms=unit.dispatch_ms)
    assert [x.dispatch_ms for x in dispatched] == [0, 40, 80]
    assert media.dropped == [(1, "queue_full"), (2, "queue_full")]
    assert _assert_accounting(sender)["dispatched_wire_bytes"] == 3000
    assert media.wire_sent_bytes == 1000
    media.arrive_by(math.inf)
    assert media.snapshot(3).source_count == 3


def test_unpaced_sender_dispatches_at_capture_without_rewriting_metadata():
    sender = _sender(pacing=False)
    for seq in range(4):
        packet = _packet(seq, wire_size=1000 + seq * 100)
        now = seq * 1000.0 / FPS
        assert sender.advance(now) == []
        sender.enqueue(packet, at_ms=now)
        dispatched = sender.advance(now)
        assert len(dispatched) == 1
        assert dispatched[0].packet is packet
        assert dispatched[0].dispatch_ms == now
        assert dispatched[0].queue_delay_ms == 0
    assert _assert_accounting(sender)["drops"] == []


def test_explicit_capture_dispatch_preserves_the_legacy_seeded_link_trace():
    config = LinkConfig(delay_ms=20, jitter_ms=7,
                        bandwidth_steps=((0, 800), (50, 50), (200, 1000)),
                        max_inflight=3, max_age_ms=200, drop_seqs=(2,))
    legacy, explicit = VirtualAccessUnitLink(config), VirtualAccessUnitLink(config)
    for seq in range(8):
        packet = _packet(seq, wire_size=1000 + seq * 100)
        now = seq * 1000.0 / FPS
        legacy.arrive_by(now)
        explicit.arrive_by(now)
        legacy.send(packet, fps=FPS)
        explicit.send(packet, fps=FPS, dispatch_ms=now)
    legacy.arrive_by(math.inf)
    explicit.arrive_by(math.inf)
    assert legacy.snapshot(8) == explicit.snapshot(8)


def test_link_staleness_includes_waiting_before_dispatch():
    media = VirtualAccessUnitLink(LinkConfig(
        delay_ms=10, jitter_ms=0, bandwidth_kbps=80,
        max_inflight=16, max_age_ms=150))
    packet = _packet(0)
    media.arrive_by(100)
    media.send(packet, fps=FPS, dispatch_ms=100)
    scheduled, reason = media.inflight[0]
    assert scheduled.packet is packet
    assert scheduled.source_ms == 0
    assert scheduled.arrival_ms == 210  # 100 queue + 100 serialization + 10 propagation.
    assert reason == "stale"  # Dispatch-to-arrival alone would incorrectly pass the cutoff.
    media.arrive_by(math.inf)
    trace = media.snapshot(1)
    assert trace.delivered == ()
    assert trace.dropped == ((0, "stale"),)
    assert trace.wire_sent_bytes == 1000


def test_link_dispatch_validation_and_rejection_preserve_already_admitted_serialization():
    media = VirtualAccessUnitLink(LinkConfig(
        delay_ms=10, jitter_ms=0, bandwidth_kbps=80,
        max_inflight=1, max_age_ms=0))
    for invalid in (-1, True, math.nan, math.inf):
        with pytest.raises(ValueError):
            media.send(_packet(0), fps=FPS, dispatch_ms=invalid)
    assert media.wire_sent_bytes == 0
    media.send(_packet(0), fps=FPS, dispatch_ms=0)
    original = tuple(media.inflight)
    media.arrive_by(20)
    media.send(_packet(1), fps=FPS, dispatch_ms=20)
    assert media.dropped == [(1, "queue_full")]
    assert tuple(media.inflight) == original
    assert media.link_free_ms == 100
    assert media.wire_sent_bytes == 1000
    with pytest.raises(ValueError):
        media.send(_packet(1), fps=FPS, dispatch_ms=19)
    # At dispatch=110 an existing arrival must be drained even though this
    # queued packet's original capture time was only 40 ms.
    with pytest.raises(RuntimeError, match="drained"):
        media.send(_packet(2), fps=FPS, dispatch_ms=110)
    media.arrive_by(110)
    media.send(_packet(2), fps=FPS, dispatch_ms=110)
    assert media.inflight[0][0].source_ms == 40
    assert media.inflight[0][0].arrival_ms == 220
    media.arrive_by(math.inf)
    trace = media.snapshot(3)
    assert [x.packet.seq for x in trace.delivered] == [0, 2]
    assert trace.dropped == ((1, "queue_full"),)
    assert trace.wire_sent_bytes == 2000
