"""Custom datagram bounds, complete-only reconstruction and receiver accounting."""
from __future__ import annotations

from dataclasses import replace
import json
import math

import pytest

from colabcola.bridge_video_datagram import (
    AccessUnitReassembler, DATAGRAM_HEADER, DATAGRAM_MAGIC, DatagramConfig,
    MAX_FRAGMENTS, MAX_UNIT_BYTES, VideoDatagram, datagram_bytes, fragment_unit,
    parse_datagram,
)
from colabcola.bridge_video_impairment import MAX_WIRE_BYTES
from colabcola.bridge_video_stream import MAX_PACKET, PACKET_HEADER, VideoAccessUnit, packet_bytes


def _unit(seq=0, size=600, fps=30, keyframe=True):
    stamp = round(seq * 1_000_000_000 / fps)
    prefix = b"\x00\x00\x00\x01" + (b"\x65" if keyframe else b"\x41")
    return VideoAccessUnit(seq, stamp, stamp, (prefix + bytes(range(256)) * ((size + 255) // 256))[:size], keyframe)


def _receiver(config=None, *, fps=30, max_age_ms=350, **kwargs):
    return AccessUnitReassembler(config or DatagramConfig(256), fps=fps,
                                max_age_ms=max_age_ms, **kwargs)


def _accounting(receiver):
    report = receiver.report()
    for field in ("datagrams", "payload_bytes", "wire_bytes"):
        assert report[f"received_{field}"] == (
            report[f"completed_{field}"] + report[f"dropped_{field}"] + report[f"buffered_{field}"])
    assert 0 <= report["buffered_payload_bytes"] <= report["peak_buffer_bytes"] <= report["max_buffer_bytes"]
    assert 0 <= report["pending_frames"] <= report["peak_pending_frames"] <= report["max_pending_frames"]
    for row in report["terminal_frames"]:
        for kind in ("payload", "wire"):
            assert row[f"received_{kind}_bytes"] == row[f"usable_{kind}_bytes"] + row[f"unusable_{kind}_bytes"]
    if report["finished"]:
        assert report["buffered_datagrams"] == report["pending_frames"] == 0
        assert len(report["terminal_frames"]) == report["source_count"]
        for field in ("datagrams", "payload_bytes", "wire_bytes"):
            assert sum(row[f"received_{field}"] for row in report["terminal_frames"]) == report[f"received_{field}"]
    json.dumps(report, allow_nan=False)
    return report


@pytest.mark.parametrize("value", [True, False, 0, -1, 255, 9001, 1200.0, "1200", math.inf, math.nan, None])
def test_datagram_configuration_requires_bounded_integer(value):
    with pytest.raises(ValueError):
        DatagramConfig(value)


@pytest.mark.parametrize("limit,size,fps,seq", [
    (256, 1, 30, 0), (256, 210, 30, 0), (256, 211, 30, 0),
    (1200, 1154, 30, 0), (1200, 1155, 30, 0),
    (1200, 12000, 7, 119), (9000, 9001, 60, 3),
    (256, MAX_PACKET, 30, 0),
])
def test_exact_roundtrip_including_boundary_sizes(limit, size, fps, seq):
    config = DatagramConfig(limit)
    unit = _unit(seq, size=size, fps=fps)
    fragments = fragment_unit(unit, config)
    assert config.body_capacity == limit - 16
    assert DATAGRAM_HEADER.size == 16
    assert len(fragments) <= MAX_FRAGMENTS
    assert b"".join(fragment.body for fragment in fragments) == packet_bytes(unit)
    assert all(len(datagram_bytes(fragment, config)) <= limit for fragment in fragments)
    receiver = _receiver(config, fps=fps, max_buffer_bytes=MAX_UNIT_BYTES)
    arrival = seq * 1000.0 / fps + 40
    for fragment in reversed(fragments[1:]):
        assert receiver.receive(parse_datagram(datagram_bytes(fragment, config), config), arrival) is None
    assert receiver.receive(parse_datagram(datagram_bytes(fragments[0], config), config), arrival) == unit
    report = _accounting(receiver)
    assert report["completed_payload_bytes"] == len(packet_bytes(unit))
    assert report["completed_wire_bytes"] == len(packet_bytes(unit)) + 16 * len(fragments)
    assert report["completed_media_bytes"] == len(unit.data)
    assert report["completed_seqs"] == [seq]
    assert receiver.receive(fragments[0], arrival) is None
    assert _accounting(receiver)["complete_frames"] == 1


@pytest.mark.parametrize("changes", [
    {"frame_seq": -1}, {"frame_seq": 120}, {"frame_seq": True}, {"frame_seq": 0.0},
    {"index": -1}, {"index": 3}, {"index": True}, {"index": 0.0},
    {"count": 0}, {"count": 2}, {"count": 4}, {"count": MAX_FRAGMENTS + 1}, {"count": True},
    {"total_size": 30}, {"total_size": MAX_UNIT_BYTES + 1}, {"total_size": True},
    {"body": b""}, {"body": b"x" * 239}, {"body": b"x" * 241}, {"body": bytearray(240)},
])
def test_datagram_fields_are_validated_before_serializing_or_buffering(changes):
    config = DatagramConfig(256)
    fragment = replace(fragment_unit(_unit(), config)[0], **changes)
    receiver = _receiver(config)
    with pytest.raises(ValueError):
        datagram_bytes(fragment, config)
    with pytest.raises(ValueError):
        receiver.receive(fragment, 100)
    assert _accounting(receiver)["received_datagrams"] == 0


@pytest.mark.parametrize("wire", [
    b"", b"x" * 16, bytearray(47), "not bytes", b"x" * 257,
    DATAGRAM_HEADER.pack(b"RTP!", 0, 0, 1, 31) + b"x" * 31,
    DATAGRAM_HEADER.pack(DATAGRAM_MAGIC, 120, 0, 1, 31) + b"x" * 31,
    DATAGRAM_HEADER.pack(DATAGRAM_MAGIC, 0, 0, 0, 31) + b"x" * 31,
    DATAGRAM_HEADER.pack(DATAGRAM_MAGIC, 0, 1, 1, 31) + b"x" * 31,
    DATAGRAM_HEADER.pack(DATAGRAM_MAGIC, 0, 0, 2, 31) + b"x" * 31,
    DATAGRAM_HEADER.pack(DATAGRAM_MAGIC, 0, 0, 1, 30) + b"x" * 30,
    DATAGRAM_HEADER.pack(DATAGRAM_MAGIC, 0, 0, 1, MAX_UNIT_BYTES + 1) + b"x" * 31,
    DATAGRAM_HEADER.pack(DATAGRAM_MAGIC, 0, 0, 1, 31) + b"x" * 30,
    DATAGRAM_HEADER.pack(DATAGRAM_MAGIC, 0, 0, 1, 31) + b"x" * 32,
    DATAGRAM_HEADER.pack(DATAGRAM_MAGIC, 0, 0, 65535, 31) + b"x" * 31,
])
def test_wire_parser_rejects_truncation_extra_bytes_and_invalid_headers(wire):
    with pytest.raises(ValueError):
        parse_datagram(wire, DatagramConfig(256))


def test_final_fragment_has_exact_residual_length_and_requires_matching_configuration():
    config = DatagramConfig(256)
    fragment = fragment_unit(_unit(), config)[-1]
    assert len(fragment.body) == 150
    for body in (fragment.body[:-1], fragment.body + b"x"):
        with pytest.raises(ValueError):
            datagram_bytes(replace(fragment, body=body), config)
    with pytest.raises(ValueError):
        parse_datagram(datagram_bytes(fragment, config), DatagramConfig(1200))


def test_invalid_source_and_configuration_inputs_are_rejected():
    for unit in (None, replace(_unit(), seq=120), replace(_unit(), seq=True),
                 replace(_unit(), keyframe=1), replace(_unit(), data=b""),
                 replace(_unit(), data=b"x" * (MAX_PACKET + 1)),
                 replace(_unit(), pts_ns=-1), replace(_unit(), capture_ns=10**19)):
        with pytest.raises(ValueError):
            fragment_unit(unit, DatagramConfig())
    for operation in (lambda: datagram_bytes(fragment_unit(_unit(), DatagramConfig())[0], {}),
                      lambda: parse_datagram(b"x" * 47, {}),
                      lambda: fragment_unit(_unit(), {}),
                      lambda: AccessUnitReassembler({}, fps=30, max_age_ms=350),
                      lambda: datagram_bytes(None, DatagramConfig())):
        with pytest.raises(ValueError):
            operation()


@pytest.mark.parametrize("changes", [
    {"fps": True}, {"fps": 0}, {"fps": 61}, {"fps": 30.0},
    {"max_age_ms": -1}, {"max_age_ms": True}, {"max_age_ms": math.nan},
    {"max_age_ms": math.inf}, {"max_age_ms": 10001},
    {"max_buffer_bytes": True}, {"max_buffer_bytes": 0}, {"max_buffer_bytes": MAX_WIRE_BYTES + 1},
    {"max_pending_frames": True}, {"max_pending_frames": 0}, {"max_pending_frames": 121},
])
def test_receiver_configuration_is_bounded(changes):
    with pytest.raises(ValueError):
        _receiver(**changes)
    _receiver(fps=1, max_age_ms=0, max_buffer_bytes=1, max_pending_frames=1)
    _receiver(fps=60, max_age_ms=10000, max_buffer_bytes=MAX_WIRE_BYTES, max_pending_frames=120)


@pytest.mark.parametrize("missing", [0, 1, 2])
def test_missing_first_middle_or_final_never_emits_partial_unit(missing):
    receiver = _receiver()
    fragments = fragment_unit(_unit(), receiver.config)
    for fragment in fragments:
        if fragment.index != missing:
            assert receiver.receive(fragment, 100) is None
    receiver.advance(350)
    assert _accounting(receiver)["pending_frames"] == 1
    receiver.advance(350.000001)
    report = _accounting(receiver)
    assert report["expired_frames"] == 1
    assert report["terminal_frames"][0]["terminal_ms"] == 350
    assert report["terminal_frames"][0]["missing_fragments"] == 1
    assert receiver.receive(fragments[missing], 351) is None
    assert _accounting(receiver)["completed_payload_bytes"] == 0


def test_reverse_order_and_duplicate_active_fragment_preserve_exact_bytes_once():
    receiver = _receiver()
    unit = _unit(keyframe=False)
    fragments = fragment_unit(unit, receiver.config)
    assert receiver.receive(fragments[2], 100) is None
    assert receiver.receive(fragments[2], 101) is None
    assert receiver.pending_bytes == len(fragments[2].body)
    assert receiver.receive(fragments[1], 102) is None
    assert receiver.receive(fragments[0], 103) == unit
    assert receiver.receive(fragments[1], 104) is None
    report = receiver.finish(1, 104)
    _accounting(receiver)
    assert report["duplicate_datagrams"] == report["terminal_datagrams"] == 1
    row = report["terminal_frames"][0]
    assert row["received_datagrams"] == 5
    assert row["accepted_fragments"] == 3
    assert row["usable_wire_bytes"] == len(packet_bytes(unit)) + 3 * 16
    assert row["unusable_wire_bytes"] == len(fragments[2].body) + len(fragments[1].body) + 2 * 16
    assert row["last_received_ms"] == 104


@pytest.mark.parametrize("conflict", ["body", "metadata"])
def test_conflicts_invalidate_active_frame_and_prevent_resurrection(conflict):
    receiver = _receiver()
    first = fragment_unit(_unit(), receiver.config)[0]
    receiver.receive(first, 100)
    bad = (replace(first, body=first.body[:-1] + b"!") if conflict == "body"
           else replace(first, total_size=first.total_size + 1))
    with pytest.raises(ValueError, match="conflicting"):
        receiver.receive(bad, 101)
    report = _accounting(receiver)
    assert report["invalid_frames"] == report["conflicting_datagrams"] == 1
    assert report["buffered_payload_bytes"] == 0
    assert receiver.receive(first, 102) is None
    assert _accounting(receiver)["received_datagrams"] == 3


def test_exact_deadline_completion_is_accepted_but_later_completion_is_expired():
    unit = _unit()
    config = DatagramConfig(256)
    fragments = fragment_unit(unit, config)
    exact, late = _receiver(config), _receiver(config)
    for receiver in (exact, late):
        for fragment in fragments[:-1]:
            receiver.receive(fragment, 100)
    assert exact.receive(fragments[-1], 350) == unit
    assert late.receive(fragments[-1], 350.000001) is None
    assert _accounting(exact)["complete_frames"] == 1
    assert _accounting(late)["expired_frames"] == 1
    assert _accounting(late)["late_datagrams"] == 1


def test_first_fragment_after_deadline_cannot_create_partial_state():
    receiver = _receiver()
    fragment = fragment_unit(_unit(size=10), receiver.config)[0]
    assert receiver.receive(fragment, 351) is None
    report = _accounting(receiver)
    assert report["peak_buffer_bytes"] == report["peak_pending_frames"] == 0
    assert report["expired_frames"] == report["late_datagrams"] == 1
    row = report["terminal_frames"][0]
    assert row["detail"] == "first_fragment_late"
    assert row["terminal_ms"] == 350
    assert row["received_wire_bytes"] == len(datagram_bytes(fragment, receiver.config))


def test_byte_pressure_evicts_oldest_arrival_and_late_fragments_do_not_resurrect():
    receiver = _receiver(max_buffer_bytes=480)
    fragments = {seq: fragment_unit(_unit(seq), receiver.config) for seq in range(3)}
    receiver.receive(fragments[1][0], 100)
    receiver.receive(fragments[0][0], 101)
    receiver.receive(fragments[2][0], 102)
    report = _accounting(receiver)
    assert report["budget_frames"] == 1
    assert report["terminal_frames"][0]["frame_seq"] == 1
    assert report["terminal_frames"][0]["detail"] == "byte_budget"
    assert receiver.receive(fragments[1][1], 103) is None
    assert receiver.pending_count == 2
    assert receiver.receive(fragments[0][1], 104) is None
    report = _accounting(receiver)
    assert report["budget_frames"] == 2  # The oldest frame can be the incoming frame itself.
    assert receiver.pending_count == 1 and receiver.pending_bytes == 240
    receiver.finish(3, math.inf)
    _accounting(receiver)


def test_tied_oldest_frames_use_source_sequence_and_pending_count_has_own_limit():
    config = DatagramConfig(256)
    by_bytes = _receiver(config, max_buffer_bytes=480)
    for seq in (1, 0, 2):
        by_bytes.receive(fragment_unit(_unit(seq), config)[0], 100)
    assert _accounting(by_bytes)["terminal_frames"][0]["frame_seq"] == 0
    by_count = _receiver(config, max_pending_frames=1)
    for seq in (0, 1):
        by_count.receive(fragment_unit(_unit(seq), config)[0], 100)
    row = _accounting(by_count)["terminal_frames"][0]
    assert row["frame_seq"] == 0 and row["detail"] == "pending_frame_limit"


def test_impossible_incoming_fragment_does_not_evict_other_useful_partial_state():
    receiver = _receiver(max_buffer_bytes=100)
    small_tail = fragment_unit(_unit(0, size=500), receiver.config)[-1]
    assert len(small_tail.body) == 50
    receiver.receive(small_tail, 100)
    receiver.receive(fragment_unit(_unit(1), receiver.config)[0], 101)
    report = _accounting(receiver)
    assert report["pending_frames"] == 1 and report["buffered_payload_bytes"] == 50
    assert report["terminal_frames"][0]["frame_seq"] == 1
    assert report["terminal_frames"][0]["detail"] == "fragment_exceeds_byte_budget"


@pytest.mark.parametrize("field,value", [
    (0, b"BAD!"), (1, 2), (2, 2), (3, 1), (4, 1), (5, 1),
    (6, 0), (6, 21), (6, MAX_PACKET + 1),
])
def test_completed_datagram_bytes_must_contain_valid_matching_au_header(field, value):
    receiver = _receiver(DatagramConfig(1200))
    framed = packet_bytes(_unit(size=20))
    values = list(PACKET_HEADER.unpack_from(framed))
    values[field] = value
    forged = PACKET_HEADER.pack(*values) + framed[PACKET_HEADER.size:]
    fragment = VideoDatagram(0, 0, 1, len(forged), forged)
    with pytest.raises(ValueError, match="reconstructed"):
        receiver.receive(fragment, 100)
    report = _accounting(receiver)
    assert report["invalid_frames"] == 1 and report["complete_frames"] == 0
    assert report["buffered_payload_bytes"] == 0
    assert receiver.receive(fragment, 101) is None


def test_age_disabled_retains_partial_until_finish_and_unseen_frames_stay_distinct():
    receiver = _receiver(max_age_ms=0)
    receiver.receive(fragment_unit(_unit(), receiver.config)[1], 100)
    receiver.advance(math.inf)
    assert _accounting(receiver)["pending_frames"] == 1
    report = receiver.finish(3, math.inf)
    _accounting(receiver)
    assert report["missing_frames"] == 3 and report["unseen_frames"] == 2
    assert report["expired_frames"] == 0
    assert report["terminal_frames"][0]["detail"] == "incomplete_at_finish"
    assert report["terminal_frames"][1]["detail"] == "no_datagrams_received"
    assert all(row["terminal_ms"] == 100 for row in report["terminal_frames"])
    assert receiver.finish(3, math.inf) == report


def test_infinite_drain_uses_finite_deadlines_and_finite_finish_can_close_early():
    receiver = _receiver()
    receiver.receive(fragment_unit(_unit(), receiver.config)[0], 100)
    report = receiver.finish(3, math.inf)
    _accounting(receiver)
    assert report["expired_frames"] == 1 and report["missing_frames"] == report["unseen_frames"] == 2
    assert [row["terminal_ms"] for row in report["terminal_frames"]] == pytest.approx([350, 350 + 1000 / 30, 350 + 2000 / 30])
    early = _receiver()
    early.receive(fragment_unit(_unit(), early.config)[0], 100)
    assert early.finish(1, 120)["terminal_frames"][0]["terminal_ms"] == 120
    assert _accounting(early)["missing_frames"] == 1


@pytest.mark.parametrize("time", [True, -1, math.nan, math.inf, -math.inf, "100", 10**1000])
def test_receive_rejects_invalid_times_without_consuming_bytes(time):
    receiver = _receiver()
    with pytest.raises(ValueError):
        receiver.receive(fragment_unit(_unit(), receiver.config)[0], time)
    assert _accounting(receiver)["received_datagrams"] == 0


def test_clock_order_source_clock_and_finish_bounds_are_enforced():
    receiver = _receiver()
    first = fragment_unit(_unit(), receiver.config)[0]
    second = fragment_unit(_unit(1), receiver.config)[0]
    with pytest.raises(ValueError, match="before its source"):
        receiver.receive(second, 20)
    receiver.receive(first, 100)
    for operation in (lambda: receiver.advance(99), lambda: receiver.receive(first, 99),
                      lambda: receiver.finish(1, 99), lambda: receiver.finish(0, 100),
                      lambda: receiver.finish(120, 100)):
        with pytest.raises(ValueError):
            operation()
    for value in (-1, 121, True, 3.0):
        with pytest.raises(ValueError):
            receiver.finish(value, math.inf)
    receiver.finish(1, 100)
    with pytest.raises(ValueError, match="already finished"):
        receiver.receive(first, 100)
    with pytest.raises(ValueError, match="another source count"):
        receiver.finish(2, 100)
    _accounting(receiver)
    assert _receiver().finish(0, math.inf)["source_count"] == 0


def test_report_is_observational_and_nested_rows_are_copied():
    receiver = _receiver()
    receiver.receive(fragment_unit(_unit(size=20), receiver.config)[0], 100)
    before = receiver.report()
    before["terminal_frames"][0]["reason"] = "changed"
    before["completed_seqs"].clear()
    report = _accounting(receiver)
    assert report["terminal_frames"][0]["reason"] == "complete"
    assert report["completed_seqs"] == [0]
