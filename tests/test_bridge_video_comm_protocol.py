"""Offline v33a Comm H.264 contract tests: no kernel, GPU, or public socket."""
from __future__ import annotations

import base64
from copy import deepcopy

import pytest

from colabcola.bridge_video_stream import PACKET_HEADER, VideoAccessUnit
from colabcola.bridge_video_comm_protocol import (
    MAX_VIDEO_BASE64, VideoCommProtocolError, VideoSequenceGate,
    decode_video_reply, encode_video_reply,
)

RUN = "d" * 32
IDR = b"\x00\x00\x00\x01\x65\x88\x84"  # IDR marker for contract checks, not decodable video
P = b"\x00\x00\x00\x01\x41\x88\x84"  # non-IDR marker


def unit(seq: int, *, idr: bool = False) -> VideoAccessUnit:
    return VideoAccessUnit(seq=seq, pts_ns=seq * 33333333,
                           capture_ns=1000 + seq, data=IDR if idr else P,
                           keyframe=idr)


def reply() -> dict:
    return encode_video_reply(run_id=RUN, pull_seq=5, input_ack=2, unit=unit(0, idr=True))


def test_video_reply_round_trip_is_explicitly_versioned_and_correlated():
    payload = reply()
    assert payload["kind"] == "video_access_unit"
    assert payload["codec"] == "h264-annexb"
    decoded, ack = decode_video_reply(payload, run_id=RUN, expected_pull_seq=5)
    assert decoded == unit(0, idr=True)
    assert ack == 2
    assert len(payload["data"]) <= MAX_VIDEO_BASE64


@pytest.mark.parametrize("field,bad", [
    ("v", True), ("v", 2), ("kind", "frame"), ("codec", "ppm"),
    ("run_id", "a" * 32), ("pull_seq", 4), ("pull_seq", True),
    ("input_ack", -1), ("input_ack", True), ("input_ack", 10000000),
    ("data", "***"), ("data", "A" * (MAX_VIDEO_BASE64 + 1)),
])
def test_invalid_reply_fields_fail_closed(field, bad):
    payload = reply()
    payload[field] = bad
    with pytest.raises(VideoCommProtocolError):
        decode_video_reply(payload, run_id=RUN, expected_pull_seq=5)


def test_mismatched_expectations_and_run_ids_fail_closed():
    with pytest.raises(VideoCommProtocolError):
        decode_video_reply(reply(), run_id="a" * 32, expected_pull_seq=5)
    with pytest.raises(VideoCommProtocolError):
        decode_video_reply(reply(), run_id=RUN, expected_pull_seq=6)
    for bad in (True, 0, 100_000_001):
        with pytest.raises(VideoCommProtocolError):
            encode_video_reply(run_id=RUN, pull_seq=bad, input_ack=0, unit=unit(0, idr=True))
    with pytest.raises(VideoCommProtocolError):
        encode_video_reply(run_id="bad", pull_seq=5, input_ack=0, unit=unit(0, idr=True))


def test_truncated_extended_and_forged_ccv1_packets_are_rejected():
    original = reply()
    wire = bytearray(base64.b64decode(original["data"]))
    variants = [wire[:-1], wire + b"x"]
    wrong_magic = bytearray(wire)
    wrong_magic[0] = ord("Z")
    variants.append(wrong_magic)
    wrong_version = bytearray(wire)
    wrong_version[4] = 99
    variants.append(wrong_version)
    bad_flags = bytearray(wire)
    bad_flags[5] = 2
    variants.append(bad_flags)
    wrong_length = bytearray(wire)
    wrong_length[PACKET_HEADER.size - 1] = 99
    variants.append(wrong_length)
    for broken in variants:
        payload = deepcopy(original)
        payload["data"] = base64.b64encode(broken).decode("ascii")
        with pytest.raises(VideoCommProtocolError):
            decode_video_reply(payload, run_id=RUN, expected_pull_seq=5)


def test_metadata_cannot_falsely_advertise_decoder_resync():
    bad = VideoAccessUnit(seq=0, pts_ns=0, capture_ns=1, data=P, keyframe=True)
    with pytest.raises(VideoCommProtocolError, match="IDR"):
        encode_video_reply(run_id=RUN, pull_seq=1, input_ack=0, unit=bad)
    altered = reply()
    wire = bytearray(base64.b64decode(altered["data"]))
    wire[PACKET_HEADER.size:] = P
    altered["data"] = base64.b64encode(wire).decode("ascii")
    with pytest.raises(VideoCommProtocolError, match="IDR"):
        decode_video_reply(altered, run_id=RUN, expected_pull_seq=5)


def test_non_annexb_or_nonboolean_keyframe_declarations_are_rejected():
    for bad in (
        VideoAccessUnit(seq=0, pts_ns=0, capture_ns=1, data=b"random", keyframe=False),
        VideoAccessUnit(seq=0, pts_ns=0, capture_ns=1, data=IDR, keyframe=1),
    ):
        with pytest.raises(VideoCommProtocolError):
            encode_video_reply(run_id=RUN, pull_seq=1, input_ack=0, unit=bad)


def test_sequence_gate_requires_initial_idr_and_resync_after_gap():
    gate = VideoSequenceGate()
    assert gate.accept(unit(0)) is False
    assert gate.accept(unit(1)) is False
    assert gate.accept(unit(2, idr=True)) is True
    assert gate.accept(unit(3)) is True
    assert gate.accept(unit(6)) is False
    assert gate.accept(unit(7)) is False
    assert gate.accept(unit(8, idr=True)) is True
    assert gate.accept(unit(9)) is True


def test_sequence_gate_rejects_reorder_and_duplicates():
    gate = VideoSequenceGate()
    assert gate.accept(unit(0, idr=True))
    assert gate.accept(unit(1))
    for seq in (1, 0):
        with pytest.raises(VideoCommProtocolError, match="duplicate or reordered"):
            gate.accept(unit(seq))
    assert gate.accept(unit(2))


def test_sequence_gate_idr_at_nonzero_start_is_safe():
    gate = VideoSequenceGate()
    assert gate.accept(unit(20, idr=True))
    assert gate.accept(unit(21))
