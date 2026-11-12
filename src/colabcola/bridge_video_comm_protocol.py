"""Optional H.264 access-unit messages for a future authenticated Jupyter Comm mode.

Contract only: this module does not attach a Comm, allocate a GPU, or change
existing PPM frames. The sender/receiver must already be bound to an owned
kernel and must correlate each reply with exactly one outstanding pull.

CCV1.capture_ns belongs to the producer's monotonic clock; do not subtract it
from local host timestamps to estimate network or end-to-end latency.
"""
from __future__ import annotations

import base64
import binascii
import re

from .bridge_video_stream import (
    KEYFRAME, MAGIC, MAX_PACKET, PACKET_HEADER, VERSION,
    VideoAccessUnit, is_keyframe, packet_bytes,
)

VIDEO_KIND = "video_access_unit"
VIDEO_CODEC = "h264-annexb"
MAX_PULL_SEQ = 100_000_000
MAX_INPUT_ACK = 9_999_999
MAX_VIDEO_BASE64 = 4 * ((PACKET_HEADER.size + MAX_PACKET + 2) // 3)
_RUN_ID = re.compile(r"[0-9a-f]{32}\Z")


class VideoCommProtocolError(ValueError):
    """Invalid or uncorrelated optional video message (no payload in errors)."""


def _check_identity(run_id: str, pull_seq: int) -> None:
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise VideoCommProtocolError("invalid owned run identifier")
    if type(pull_seq) is not int or not 1 <= pull_seq <= MAX_PULL_SEQ:
        raise VideoCommProtocolError("invalid video pull identifier")


def _check_ack(input_ack: int) -> None:
    if type(input_ack) is not int or not 0 <= input_ack <= MAX_INPUT_ACK:
        raise VideoCommProtocolError("invalid video input acknowledgment")


def _check_idr_flag(unit: VideoAccessUnit) -> None:
    # Never accept metadata falsely advertising a decoder recovery point.
    # Actual H.264 decodability still needs a real decoder in the next phase.
    if type(unit.keyframe) is not bool or not unit.data.startswith(
            (b"\x00\x00\x01", b"\x00\x00\x00\x01")):
        raise VideoCommProtocolError("invalid Annex B access-unit declaration")
    if unit.keyframe != is_keyframe(unit.data):
        raise VideoCommProtocolError("video keyframe flag and IDR NAL disagree")


def encode_video_reply(*, run_id: str, pull_seq: int,
                       input_ack: int, unit: VideoAccessUnit) -> dict:
    """Encode one bounded whole access unit as a versioned Comm data object."""
    _check_identity(run_id, pull_seq)
    _check_ack(input_ack)
    if not isinstance(unit, VideoAccessUnit):
        raise VideoCommProtocolError("invalid video access unit")
    try:
        binary = packet_bytes(unit)
    except ValueError:
        raise VideoCommProtocolError("invalid CCV1 video access unit") from None
    _check_idr_flag(unit)
    return {
        "v": 1, "kind": VIDEO_KIND, "run_id": run_id,
        "pull_seq": pull_seq, "input_ack": input_ack,
        "codec": VIDEO_CODEC,
        "data": base64.b64encode(binary).decode("ascii"),
    }


def decode_video_reply(payload: object, *, run_id: str,
                       expected_pull_seq: int) -> tuple[VideoAccessUnit, int]:
    """Validate identity, request correlation, payload and IDR declaration.

    Caller is responsible for checking that expected_pull_seq was outstanding
    and removing it only once. Use VideoSequenceGate before feeding a decoder.
    """
    _check_identity(run_id, expected_pull_seq)
    if not isinstance(payload, dict):
        raise VideoCommProtocolError("video Comm reply is not an object")
    if (type(payload.get("v")) is not int or payload["v"] != 1
            or payload.get("kind") != VIDEO_KIND
            or payload.get("run_id") != run_id
            or type(payload.get("pull_seq")) is not int
            or payload["pull_seq"] != expected_pull_seq
            or payload.get("codec") != VIDEO_CODEC):
        raise VideoCommProtocolError("video Comm reply contract mismatch")
    ack = payload.get("input_ack")
    _check_ack(ack)
    encoded = payload.get("data")
    if (not isinstance(encoded, str)
            or not 1 <= len(encoded) <= MAX_VIDEO_BASE64):
        raise VideoCommProtocolError("video Comm Base64 exceeds bounds")
    try:
        wire = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise VideoCommProtocolError("video Comm Base64 malformed") from None
    if (base64.b64encode(wire).decode("ascii") != encoded
            or not PACKET_HEADER.size < len(wire) <= PACKET_HEADER.size + MAX_PACKET):
        raise VideoCommProtocolError("video Comm access-unit encoding invalid")
    magic, version, flags, seq, pts, captured, length = PACKET_HEADER.unpack_from(wire)
    if (magic != MAGIC or version != VERSION or flags & ~KEYFRAME
            or seq >= 1_000_000 or pts >= 10**14 or captured >= 10**19
            or not 0 < length <= MAX_PACKET
            or length != len(wire) - PACKET_HEADER.size):
        raise VideoCommProtocolError("CCV1 video access-unit header invalid")
    unit = VideoAccessUnit(
        seq=seq, pts_ns=pts, capture_ns=captured,
        data=wire[PACKET_HEADER.size:], keyframe=bool(flags & KEYFRAME),
    )
    _check_idr_flag(unit)
    return unit, ack


class VideoSequenceGate:
    """Conservatively suppress dependent frames after startup or sequence loss.

    Strictly increasing source sequences only. A gap invalidates all dependent
    non-IDR frames until a new IDR arrives; it does not claim to repair loss.
    Per-session gate; create a new instance for each new decoder/session.
    """

    def __init__(self) -> None:
        self.last_seq = -1
        self.awaiting_idr = True

    def accept(self, unit: VideoAccessUnit) -> bool:
        if not isinstance(unit, VideoAccessUnit):
            raise VideoCommProtocolError("invalid video sequence candidate")
        try:
            packet_bytes(unit)
        except ValueError:
            raise VideoCommProtocolError("invalid CCV1 sequence candidate") from None
        _check_idr_flag(unit)
        if unit.seq <= self.last_seq:
            raise VideoCommProtocolError("duplicate or reordered video access unit")
        if unit.seq != self.last_seq + 1:
            self.awaiting_idr = True
        self.last_seq = unit.seq
        if unit.keyframe:
            self.awaiting_idr = False
        return not self.awaiting_idr
