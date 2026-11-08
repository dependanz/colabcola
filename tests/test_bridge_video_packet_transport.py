"""Exact-byte grouping, causal pacing, partial local drops and byte admission."""
from __future__ import annotations

from dataclasses import replace
import json
import math

import pytest

from colabcola import bridge_video_packet_transport as transport
from colabcola.bridge_video_datagram import DatagramConfig, datagram_bytes, fragment_unit
from colabcola.bridge_video_impairment import LinkConfig
from colabcola.bridge_video_packet_transport import (
    BytePacketLink, PacketSenderConfig, PacketSenderQueue,
)
from colabcola.bridge_video_sender_service import SenderServiceConfig
from colabcola.bridge_video_stream import PACKET_HEADER, VideoAccessUnit, packet_bytes


FPS = 50  # 188 payload kbps + nominal 30-byte frame framing = 200 kbps.


def _unit(seq, *, frame_bytes=1000, fps=FPS, idr=None):
    idr = seq == 0 if idr is None else idr
    prefix = b"\x00\x00\x00\x01" + (b"\x65" if idr else b"\x41")
    length = frame_bytes - PACKET_HEADER.size
    assert length >= len(prefix)
    stamp = round(seq * 1_000_000_000 / fps)
    return VideoAccessUnit(seq, stamp, stamp, prefix + b"x" * (length - len(prefix)), idr)


def _sender(*, datagram=None, fps=FPS, target=188, service=None, **options):
    config = PacketSenderConfig(**{"max_pending": 120, "max_age_ms": 0, **options})
    return PacketSenderQueue(config, datagram or DatagramConfig(), fps=fps,
                             initial_target_kbps=target,
                             service=service or SenderServiceConfig())


def _capture(sender, seq, **options):
    at = seq * 1000.0 / sender.fps
    output = sender.advance(at)
    unit = _unit(seq, fps=sender.fps, **options)
    sender.enqueue(unit, fragment_unit(unit, sender.datagram_config), at_ms=at)
    return output + sender.advance(at)


def _drain_sender(sender):
    output = []
    steps = 0
    while sender.next_event_ms is not None:
        at = sender.next_event_ms
        assert math.isfinite(at) and at >= sender.clock_ms
        output.extend(sender.advance(at))
        steps += 1
        assert steps <= sender.report()["offered_packets"] + sender.frames_observed
    return output


def _account(sender):
    report = sender.report()
    assert report["offered_packets"] == (
        report["dispatched_packets"] + report["dropped_packets"] + report["pending_packets"])
    assert report["offered_wire_bytes"] == (
        report["dispatched_wire_bytes"] + report["dropped_wire_bytes"] + report["pending_wire_bytes"])
    attempted = [tuple(identity) for row in report["dispatches"] for identity in row["fragment_ids"]]
    dropped = [tuple(row["fragment_id"]) for row in report["drops"]]
    assert len(set(attempted)) == len(attempted)
    assert len(set(dropped)) == len(dropped)
    assert not set(attempted) & set(dropped)
    if sender.pending_count == 0:
        expected = {(row["source_seq"], index) for row in report["frames"]
                    for index in range(row["offered_packets"])}
        assert set(attempted) | set(dropped) == expected
    assert sum(report["frame_states"].values()) == report["frames_observed"]
    json.dumps(report, allow_nan=False)
    return report


def _send(media, datagrams, dispatch_ms, source_ms):
    observed = media.arrive_by(dispatch_ms)
    media.send(datagrams, dispatch_ms=dispatch_ms, source_ms=source_ms)
    return observed


def _link_account(media):
    report = media.report()
    assert report["offered_packets"] == report["admitted_packets"] + report["rejected_packets"]
    assert report["admitted_packets"] == report["delivered_packets"] + report["erased_packets"]
    assert report["offered_wire_bytes"] == report["admitted_wire_bytes"] + report["rejected_wire_bytes"]
    assert report["admitted_wire_bytes"] == report["delivered_wire_bytes"] + report["erased_wire_bytes"]
    assert report["attempted_groups"] == report["admitted_groups"] + report["rejected_groups"]
    assert report["max_inflight_bytes"] <= report["max_inflight_bytes_limit"]
    assert len({tuple(row["fragment_id"]) for row in report["packets"]}) == report["offered_packets"]
    json.dumps(report, allow_nan=False)
    return report


@pytest.mark.parametrize("options", [
    {"grouping": "frame"}, {"grouping": True}, {"pacing": True}, {"pacing": "adaptive"},
    {"max_pending": 0}, {"max_pending": 121}, {"max_pending": True},
    {"max_age_ms": -1}, {"max_age_ms": 2001}, {"max_age_ms": math.nan},
    {"max_age_ms": math.inf}, {"max_age_ms": True}, {"max_age_ms": 10**1000},
])
def test_sender_configuration_bounds(options):
    with pytest.raises(ValueError):
        PacketSenderConfig(**options)


def test_sender_requires_valid_frozen_source_and_atomic_complete_framing():
    sender = _sender(pacing="fixed")
    unit = _unit(0, frame_bytes=2500)
    fragments = fragment_unit(unit, sender.datagram_config)
    before = sender.report()
    for invalid in (fragments[:-1], fragments[::-1], list(fragments),
                    (replace(fragments[0], body=b"z" + fragments[0].body[1:]),) + fragments[1:]):
        with pytest.raises(ValueError, match="exactly frame"):
            sender.enqueue(unit, invalid, at_ms=0)
        assert sender.report() == before
    with pytest.raises(ValueError, match="original source time"):
        other = _unit(1)
        sender.enqueue(other, fragment_unit(other, sender.datagram_config), at_ms=0)
    with pytest.raises(ValueError, match="advance sender"):
        sender.enqueue(unit, fragments, at_ms=1)
    sender.enqueue(unit, fragments, at_ms=0)
    assert sender.report()["offered_wire_bytes"] == len(packet_bytes(unit)) + 16 * len(fragments)
    with pytest.raises(ValueError, match="consecutive IDs"):
        sender.enqueue(unit, fragments, at_ms=0)
    _account(sender)


@pytest.mark.parametrize("grouping", ["batch", "packet"])
@pytest.mark.parametrize("pacing", ["none", "fixed", "queue"])
def test_all_modes_offer_and_attempt_identical_exact_datagram_bytes(grouping, pacing):
    sender = _sender(grouping=grouping, pacing=pacing)
    events = []
    source_fragments = []
    for seq, size in enumerate((4000, 100, 2500, 6000)):
        source_fragments.extend(fragment_unit(_unit(seq, frame_bytes=size), sender.datagram_config))
        events.extend(_capture(sender, seq, frame_bytes=size))
    events.extend(_drain_sender(sender))
    actual = [fragment for event in events for fragment in event.datagrams]
    assert actual == source_fragments
    assert all(event.wire_size == sum(len(datagram_bytes(d, sender.datagram_config))
                                      for d in event.datagrams) for event in events)
    if grouping == "batch":
        assert len(events) == 4 and all(len({d.frame_seq for d in e.datagrams}) == 1 for e in events)
    else:
        assert all(len(e.datagrams) == 1 for e in events)
    report = _account(sender)
    assert report["dropped_packets"] == 0
    assert report["frame_states"]["fully_dispatched"] == 4
    assert report["service"]["base_kbps"] == 200
    assert report["service"]["max_kbps"] <= 400


@pytest.mark.parametrize("bandwidth,steps,jitter", [
    (0, (), 0), (500, (), 12), (0, ((0, 500), (20, 75), (160, 1000)), 12),
])
def test_unconstrained_batch_and_packet_burst_have_same_completion_and_exact_bytes(bandwidth, steps, jitter):
    config = DatagramConfig()
    link = LinkConfig(delay_ms=40, jitter_ms=jitter, bandwidth_kbps=bandwidth,
                      bandwidth_steps=steps, max_inflight=1, max_age_ms=1)
    sources = [_unit(seq, frame_bytes=size) for seq, size in enumerate((5000, 2000, 10000))]

    def run(batch):
        media = BytePacketLink(link, config, max_inflight_bytes=100_000)
        for unit in sources:
            fragments = fragment_unit(unit, config)
            groups = [fragments] if batch else [(d,) for d in fragments]
            for group in groups:
                _send(media, group, unit.seq * 20, unit.seq * 20)
        media.arrive_by(math.inf)
        return _link_account(media)

    batch, burst = run(True), run(False)
    assert batch["offered_packets"] == burst["offered_packets"]
    assert batch["admitted_wire_bytes"] == burst["admitted_wire_bytes"]
    assert batch["delivered_packets"] == burst["delivered_packets"] == batch["offered_packets"]
    for seq in range(3):
        one = [r for r in batch["packets"] if r["source_seq"] == seq]
        two = [r for r in burst["packets"] if r["source_seq"] == seq]
        assert {r["jitter_ms"] for r in one + two} == {one[0]["jitter_ms"]}
        assert max(r["arrival_ms"] for r in one) == pytest.approx(max(r["arrival_ms"] for r in two), abs=1e-9)
    assert all(r["serialization_finish_ms"] <= r["arrival_ms"] for r in burst["packets"])


def test_partial_packet_admission_does_not_abort_later_fragments_and_batch_is_atomic():
    config = DatagramConfig()
    fragments = fragment_unit(_unit(0, frame_bytes=2500), config)
    assert [len(datagram_bytes(d, config)) for d in fragments] == [1200, 1200, 148]
    link = LinkConfig(delay_ms=40, jitter_ms=0, bandwidth_kbps=0)
    batch = BytePacketLink(link, config, max_inflight_bytes=1400)
    _send(batch, fragments, 0, 0)
    batch.arrive_by(math.inf)
    batch_report = _link_account(batch)
    assert batch_report["admitted_packets"] == 0 and batch_report["rejected_packets"] == 3
    assert batch_report["rejected_groups"] == 1 and batch_report["admitted_wire_bytes"] == 0

    packet = BytePacketLink(link, config, max_inflight_bytes=1400)
    for fragment in fragments:
        _send(packet, (fragment,), 0, 0)
    observed = packet.arrive_by(math.inf)
    assert [x.fragment.index for x in observed] == [0, 2]
    packet_report = _link_account(packet)
    assert [r["status"] for r in packet_report["packets"]] == ["delivered", "queue_full", "delivered"]
    assert packet_report["admitted_wire_bytes"] == 1348
    assert packet_report["offered_wire_bytes"] == batch_report["offered_wire_bytes"] == 2548


def test_erased_fragments_consume_bytes_until_arrival_even_inside_an_atomic_batch():
    config = DatagramConfig(max_datagram_bytes=256)
    media = BytePacketLink(LinkConfig(delay_ms=50, jitter_ms=0, bandwidth_kbps=0,
                                     max_inflight=1, max_age_ms=1), config,
                           max_inflight_bytes=512, erase_first_seqs=(0,))
    first = fragment_unit(_unit(0, frame_bytes=480), config)
    _send(media, first, 0, 0)
    assert media.inflight_bytes == 512
    assert media.arrive_by(20) == []
    media.send(fragment_unit(_unit(1, frame_bytes=100), config), dispatch_ms=20, source_ms=20)
    assert media.inflight_bytes == 512
    assert media.arrive_by(49) == [] and media.inflight_bytes == 512
    with pytest.raises(RuntimeError, match="undrained"):
        media.report()
    observed = media.arrive_by(50)
    assert [r.fragment.index for r in observed] == [1]
    assert media.inflight_bytes == 0
    report = _link_account(media)
    assert (report["admitted_packets"], report["erased_packets"], report["delivered_packets"]) == (2, 1, 1)
    assert report["erased_wire_bytes"] == report["delivered_wire_bytes"] == 256
    assert report["rejected_wire_bytes"] == 116


def test_jitter_and_random_erasure_are_paired_independently_of_prior_admission():
    config = DatagramConfig(max_datagram_bytes=256)
    link = LinkConfig(delay_ms=5, jitter_ms=10, bandwidth_kbps=0, loss_percent=50, seed=17)
    first = fragment_unit(_unit(0, frame_bytes=6000), config)
    second = fragment_unit(_unit(1, frame_bytes=2400), config)

    def run(budget):
        media = BytePacketLink(link, config, max_inflight_bytes=budget)
        _send(media, first, 0, 0)
        for fragment in second:
            _send(media, (fragment,), 20, 20)
        media.arrive_by(math.inf)
        return _link_account(media)

    small, large = run(5000), run(20000)
    assert small["groups"][0]["admitted"] is False
    assert large["groups"][0]["admitted"] is True
    a = [r for r in small["packets"] if r["source_seq"] == 1]
    b = [r for r in large["packets"] if r["source_seq"] == 1]
    assert [r["status"] for r in a] == [r["status"] for r in b]
    assert {r["status"] for r in a} == {"erased", "delivered"}
    assert [r["jitter_ms"] for r in a] == [r["jitter_ms"] for r in b]
    assert len({r["jitter_ms"] for r in a}) == 1


def test_piecewise_link_serialization_integrates_exact_group_wire_work():
    config = DatagramConfig()
    media = BytePacketLink(LinkConfig(delay_ms=10, jitter_ms=0,
                                     bandwidth_steps=((0, 100), (20, 50), (60, 200))), config)
    first = fragment_unit(_unit(0, frame_bytes=1000), config)
    second = fragment_unit(_unit(1, frame_bytes=1000), config)
    _send(media, first, 0, 0)
    # 8,128 bits: 2,000 before t=20, 2,000 by t=60, then 4,128/200.
    assert media.next_arrival_ms == pytest.approx(90.64)
    _send(media, second, 20, 20)
    assert media.next_arrival_ms == pytest.approx(90.64)  # Existing schedule is immutable.
    assert media.arrive_by(90.639) == []
    assert len(media.arrive_by(90.64)) == 1
    media.arrive_by(math.inf)
    report = _link_account(media)
    assert [r["serialization_start_ms"] for r in report["groups"]] == pytest.approx([0, 80.64])
    assert [r["serialization_finish_ms"] for r in report["groups"]] == pytest.approx([80.64, 121.28])
    assert [r["arrival_ms"] for r in report["groups"]] == pytest.approx([90.64, 131.28])


def test_future_capacity_suffix_cannot_change_earlier_completed_serialization():
    config = DatagramConfig()
    fragments = fragment_unit(_unit(0, frame_bytes=1000), config)
    reports = []
    for future in (10, 10000):
        media = BytePacketLink(LinkConfig(delay_ms=40, jitter_ms=0,
                                         bandwidth_steps=((0, 1000), (1000, future))), config)
        _send(media, fragments, 0, 0)
        media.arrive_by(math.inf)
        reports.append(_link_account(media))
    assert reports[0]["groups"] == reports[1]["groups"]
    assert reports[0]["packets"] == reports[1]["packets"]


def test_duplicate_malformed_and_noncausal_link_inputs_do_not_change_accounting():
    config = DatagramConfig()
    media = BytePacketLink(LinkConfig(delay_ms=10, jitter_ms=0, bandwidth_kbps=0), config)
    first = fragment_unit(_unit(0, frame_bytes=2500), config)
    second = fragment_unit(_unit(1), config)
    for invalid in ((), [first[0]], (first[0], first[0]), (first[0], second[0]),
                    (replace(first[0], body=b"short"),)):
        with pytest.raises(ValueError):
            media.send(invalid, dispatch_ms=0, source_ms=0)
        assert media.report()["offered_packets"] == 0
    media.send((first[0],), dispatch_ms=0, source_ms=0)
    with pytest.raises(ValueError, match="only once"):
        media.send((first[0],), dispatch_ms=0, source_ms=0)
    with pytest.raises(ValueError, match="metadata changed"):
        media.send((first[1],), dispatch_ms=1, source_ms=1)
    with pytest.raises(RuntimeError, match="drained before"):
        media.send((first[1],), dispatch_ms=10, source_ms=0)
    media.arrive_by(20)
    with pytest.raises(ValueError, match="frontier"):
        media.send((first[1],), dispatch_ms=10, source_ms=0)
    for invalid in (-1, math.nan, -math.inf, True, "30", 10**1000):
        with pytest.raises(ValueError):
            media.arrive_by(invalid)
        with pytest.raises(ValueError):
            media.send((first[1],), dispatch_ms=invalid, source_ms=0)
    media.send((first[1],), dispatch_ms=20, source_ms=0)
    media.arrive_by(math.inf)
    assert _link_account(media)["offered_packets"] == 2


def test_byte_link_rejects_ambiguous_legacy_drops_and_invalid_limits():
    config = DatagramConfig()
    with pytest.raises(ValueError, match="erase_first_seqs"):
        BytePacketLink(LinkConfig(drop_seqs=(0,)), config)
    for budget in (0, True, 24 * 1024 * 1024 + 1):
        with pytest.raises(ValueError, match="max_inflight_bytes"):
            BytePacketLink(LinkConfig(), config, max_inflight_bytes=budget)
    for sequences in ([0], (1, 0), (0, 0), (120,), (True,)):
        with pytest.raises(ValueError, match="erase_first_seqs"):
            BytePacketLink(LinkConfig(), config, erase_first_seqs=sequences)


def test_every_rejected_attempt_incurred_full_pacing_debt_without_sender_feedback():
    config = DatagramConfig(max_datagram_bytes=1000)

    def run(budget, future_rate):
        sender = _sender(datagram=config, pacing="fixed")
        media = BytePacketLink(LinkConfig(delay_ms=40, jitter_ms=0,
                                         bandwidth_steps=((0, 1000), (30, future_rate))), config,
                               max_inflight_bytes=budget)
        events = _capture(sender, 0, frame_bytes=1968)
        events.extend(_capture(sender, 1, frame_bytes=100))
        events.extend(_drain_sender(sender))
        for event in events:
            _send(media, event.datagrams, event.dispatch_ms, event.source_ms)
        media.arrive_by(math.inf)
        return _account(sender), _link_account(media)

    rejected, bad_link = run(999, 10)
    admitted, good_link = run(100_000, 10000)
    assert rejected == admitted  # Neither admission nor future capacity can enter the sender.
    assert [r["dispatch_ms"] for r in rejected["dispatches"]] == [0, 40, 80]
    assert [r["charged_bits"] for r in rejected["dispatches"]] == [8000, 8000, 928]
    assert bad_link["rejected_packets"] == 2 and good_link["rejected_packets"] == 0
    assert rejected["remaining_debt_bits"] == 928


def test_queue_service_counts_all_remaining_datagrams_and_charges_before_refresh():
    sender = _sender(datagram=DatagramConfig(max_datagram_bytes=1000))
    first = _capture(sender, 0, frame_bytes=1968)
    assert len(first) == 1 and first[0].pacing_kbps == 320
    enqueue, dispatch = sender.report()["service"]["decisions"]
    assert enqueue["work_bits"] == dispatch["work_bits"] == 16000
    assert dispatch["debt_bits"] == 8000 and dispatch["pending_wire_bytes"] == 1000
    sender.advance(10)
    before = sender.report()["service"]["decisions"]
    assert sender.debt_bits == 4800 and sender.pacing_kbps == 320
    assert sender.report()["service"]["decisions"] == before
    assert _capture(sender, 1, frame_bytes=984) == []
    assert sender.debt_bits == 1600 and sender.pacing_kbps == 352
    ready = 20 + 1600 / 352
    events = sender.advance(ready)
    assert len(events) == 1 and events[0].datagrams[0].index == 1
    decision = _account(sender)["service"]["decisions"][-1]
    assert decision["debt_bits"] == 8000
    assert decision["pending_wire_bytes"] == 1000 and decision["work_bits"] == 16000
    assert decision["pacing_kbps"] == 320


def test_partial_expiry_drops_unsent_suffix_and_dependents_but_retains_queued_idr():
    sender = _sender(datagram=DatagramConfig(max_datagram_bytes=1000), pacing="fixed",
                     max_pending=3, max_age_ms=50)
    events = _capture(sender, 0, frame_bytes=10000)
    events.extend(_capture(sender, 1, frame_bytes=100))
    events.extend(_capture(sender, 2, frame_bytes=100, idr=True))
    assert [(e.datagrams[0].frame_seq, e.datagrams[0].index, e.dispatch_ms)
            for e in events] == [(0, 0, 0), (0, 1, 40)]
    assert sender.advance(50) == []
    assert sender.debt_bits == 6000 and sender.pending_count == 1 and not sender.need_idr
    report = _account(sender)
    assert [(r["source_seq"], r["reason"]) for r in report["frame_drops"]] == [
        (0, "sender_age"), (1, "sender_dependency")]
    assert report["frame_drops"][0]["partial_frame"]
    assert all(r["fragment_index"] >= 2 for r in report["drops"] if r["source_seq"] == 0)
    assert report["service"]["decisions"][-1]["pending_wire_bytes"] == 116
    assert report["service"]["decisions"][-1]["debt_bits"] == 6000
    assert report["local_repair_requests"] == 0
    events.extend(_capture(sender, 3, frame_bytes=100))
    events.extend(sender.advance(100))
    assert [(e.datagrams[0].frame_seq, e.dispatch_ms) for e in events[2:]] == [(2, 80), (3, 84.64)]
    final = _account(sender)
    assert final["frame_states"] == {"fully_dispatched": 2, "fully_dropped": 1,
                                     "partly_dispatched_then_dropped": 1, "pending": 0}


def test_after_a_partial_drop_predictions_stay_suppressed_until_a_verified_new_idr():
    sender = _sender(datagram=DatagramConfig(max_datagram_bytes=1000), pacing="fixed",
                     max_age_ms=10)
    _capture(sender, 0, frame_bytes=1968)
    assert sender.advance(10) == [] and sender.need_idr
    assert _capture(sender, 1, frame_bytes=100) == []
    sender.advance(40)
    forged = replace(_unit(2, frame_bytes=100, idr=False), keyframe=True)
    sender.enqueue(forged, fragment_unit(forged, sender.datagram_config), at_ms=40)
    assert sender.advance(40) == [] and sender.need_idr
    final = _capture(sender, 3, frame_bytes=100, idr=True)
    assert len(final) == 1 and final[0].datagrams[0].frame_seq == 3
    assert not sender.need_idr
    report = _account(sender)
    assert [r["source_seq"] for r in report["frame_drops"]] == [0, 1, 2]
    assert report["local_repair_requests"] == 0


def test_overflow_counts_partially_unsent_frames_and_refreshes_after_whole_chain_drop():
    sender = _sender(datagram=DatagramConfig(max_datagram_bytes=1000), target=20,
                     pacing="fixed", max_pending=3)
    events = _capture(sender, 0, frame_bytes=1968)
    events.extend(_capture(sender, 1, frame_bytes=100))
    events.extend(_capture(sender, 2, frame_bytes=100, idr=True))
    before = len(sender.report()["service"]["decisions"])
    assert _capture(sender, 3, frame_bytes=100) == []
    report = _account(sender)
    assert len(report["service"]["decisions"]) == before + 1
    assert report["pending_frames"] == 2 and report["peak_pending_frames"] == 3
    assert [(r["source_seq"], r["reason"]) for r in report["frame_drops"]] == [
        (0, "sender_queue_full"), (1, "sender_dependency")]
    final = report["service"]["decisions"][-1]
    assert final["cause"] == "enqueue" and final["pending_wire_bytes"] == 232
    assert final["debt_bits"] == 6080
    events.extend(_drain_sender(sender))
    assert [(e.datagrams[0].frame_seq, e.dispatch_ms) for e in events] == [(0, 0), (2, 250), (3, 279)]
    _account(sender)


def test_fractional_readiness_tie_expires_at_actual_timestamp_never_at_earlier_endpoint():
    sender = _sender(fps=30, target=30, grouping="batch", pacing="fixed", max_age_ms=100,
                     max_pending=3)
    first = _capture(sender, 0, frame_bytes=604)
    assert first[0].wire_size == 620 and first[0].pacing_kbps == 37.2
    _capture(sender, 1, frame_bytes=84)
    _capture(sender, 2, frame_bytes=84)
    early = 620 * 8 / 37.2
    expiry = 1000.0 / 30 + 100
    assert early < expiry and expiry - early < 1e-9
    assert sender.next_event_ms == expiry
    before = sender.report()["service"]["decisions"]
    assert sender.advance(early) == [] and sender.drops == []
    assert sender.report()["service"]["decisions"] == before
    assert sender.next_event_ms == expiry > sender.clock_ms
    assert sender.advance(expiry) == []
    assert [(r["source_seq"], r["at_ms"]) for r in sender.frame_drops] == [(1, expiry), (2, expiry)]
    assert sender.debt_bits == 0 and sender.next_event_ms is None
    _account(sender)


@pytest.mark.parametrize("grouping", ["batch", "packet"])
@pytest.mark.parametrize("max_age_ms", [0, 45])
def test_observation_endpoints_and_report_reads_cannot_refresh_service(grouping, max_age_ms):
    def run(split):
        sender = _sender(datagram=DatagramConfig(max_datagram_bytes=256), fps=30,
                         target=30, grouping=grouping, max_pending=3, max_age_ms=max_age_ms)
        events = []

        def advance(end):
            start = sender.clock_ms
            if split:
                for fraction in (0.19, 0.43, 0.77):
                    events.extend(sender.advance(start + (end - start) * fraction))
                    report = sender.report()
                    assert sender.report() == report
            events.extend(sender.advance(end))

        for seq, size in enumerate((620, 100, 105, 700, 110, 115, 800, 120)):
            at = seq * 1000.0 / sender.fps
            advance(at)
            unit = _unit(seq, frame_bytes=size, fps=sender.fps, idr=seq in (0, 3, 6))
            sender.enqueue(unit, fragment_unit(unit, sender.datagram_config), at_ms=at)
            events.extend(sender.advance(at))
        advance(1000)
        return events, _account(sender)

    direct_events, direct = run(False)
    split_events, split = run(True)
    assert [e.datagrams for e in direct_events] == [e.datagrams for e in split_events]
    assert [e.dispatch_ms for e in direct_events] == pytest.approx(
        [e.dispatch_ms for e in split_events], abs=1e-8, rel=0)
    assert [e.pacing_kbps for e in direct_events] == pytest.approx(
        [e.pacing_kbps for e in split_events], abs=1e-8, rel=0)
    assert direct["drops"] == split["drops"]
    a, b = direct["service"]["decisions"], split["service"]["decisions"]
    assert len(a) == len(b)
    for one, two in zip(a, b):
        assert {k: v for k, v in one.items() if type(v) is not float} == {
            k: v for k, v in two.items() if type(v) is not float}
        assert {k: v for k, v in one.items() if type(v) is float} == pytest.approx(
            {k: v for k, v in two.items() if type(v) is float}, abs=1e-8, rel=0)


def test_tail_is_finite_and_final_packet_debt_remains_until_actually_paid():
    sender = _sender(datagram=DatagramConfig(max_datagram_bytes=1000))
    events = []
    for seq, size in enumerate((6000, 1000, 1000, 1000)):
        events.extend(_capture(sender, seq, frame_bytes=size))
    events.extend(_drain_sender(sender))
    assert sender.pending_count == 0 and sender.debt_bits == events[-1].wire_size * 8 > 0
    report = _account(sender)
    assert report["tail"]["post_source_dispatches"] > 0
    assert report["tail"]["last_dispatch_ms"] > report["tail"]["last_capture_ms"]
    before = report["service"]["decisions"]
    assert sender.advance(report["tail"]["debt_clear_at_current_service_ms"] + 1) == []
    assert sender.debt_bits == 0
    assert sender.report()["service"]["decisions"] == before
    assert report["service"]["cap_limited_decisions"] > 0
    assert report["service"]["cap_binding_ms"] > 0
    assert report["service"]["max_kbps"] == sender.cap_kbps == 400


def test_dispatch_peaks_include_entire_simultaneous_burst_and_exclude_endpoints():
    sender = _sender(datagram=DatagramConfig(max_datagram_bytes=256), pacing="none")
    _capture(sender, 0, frame_bytes=1000)
    _capture(sender, 1, frame_bytes=1000)
    report = _account(sender)
    peaks = {row["window_ms"]: row["peak"] for row in report["peak_dispatch_windows"]}
    assert peaks[5]["wire_bytes"] == peaks[20]["wire_bytes"] == 1080
    assert peaks[100]["wire_bytes"] == peaks[300]["wire_bytes"] == 2160
    assert peaks[5]["attempted_wire_kbps"] == 1728


def test_idle_time_does_not_bank_credit_and_invalid_times_leave_sender_unchanged():
    sender = _sender(datagram=DatagramConfig(max_datagram_bytes=1000), pacing="fixed")
    _capture(sender, 0, frame_bytes=100)
    sender.advance(20)
    assert sender.debt_bits == 0
    large = _unit(1, frame_bytes=1968)
    sender.enqueue(large, fragment_unit(large, sender.datagram_config), at_ms=20)
    events = sender.advance(20)
    assert len(events) == 1 and events[0].dispatch_ms == 20
    assert sender.next_event_ms == 60 and sender.debt_bits == 8000
    before = sender.report()
    for at in (-1, 19, True, math.inf, math.nan, 10**1000):
        with pytest.raises(ValueError):
            sender.advance(at)
        assert sender.report() == before


def test_offered_byte_limits_are_checked_before_local_or_link_mutation(monkeypatch):
    monkeypatch.setattr(transport, "MAX_WIRE_BYTES", 2000)
    config = DatagramConfig()
    sender = _sender(datagram=config, pacing="none")
    _capture(sender, 0, frame_bytes=1000)
    sender.advance(20)
    before = sender.report()
    with pytest.raises(ValueError, match="offered wire budget"):
        _capture(sender, 1, frame_bytes=1000)
    assert sender.report() == before
    media = BytePacketLink(LinkConfig(delay_ms=40, jitter_ms=0, bandwidth_kbps=0), config,
                           max_inflight_bytes=2000)
    _send(media, fragment_unit(_unit(0), config), 0, 0)
    media.arrive_by(20)
    with pytest.raises(ValueError, match="offered wire budget"):
        media.send(fragment_unit(_unit(1), config), dispatch_ms=20, source_ms=20)
    media.arrive_by(math.inf)
    assert _link_account(media)["offered_packets"] == 1


def test_source_capture_limit_is_bounded_and_unpaced_dispatch_has_no_debt():
    sender = _sender(pacing="none", fps=60)
    for seq in range(120):
        assert len(_capture(sender, seq, frame_bytes=100)) == 1
    assert sender.frames_observed == 120 and sender.debt_bits == 0
    at = 120 * 1000.0 / sender.fps
    sender.advance(at)
    with pytest.raises(ValueError, match="consecutive IDs"):
        sender.enqueue(_unit(120, fps=60), (), at_ms=at)
    assert _account(sender)["frames_observed"] == 120
