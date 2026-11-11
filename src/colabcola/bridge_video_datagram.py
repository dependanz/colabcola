"""Bounded custom research datagrams and receiver-only AU reassembly.

This is an offline experiment format, not RTP or an interoperable UDP
transport. Each datagram carries a slice of the entire existing CCV1 access
unit packet. The receiver uses only these bytes, arrival times and the
declared synchronized fixed-frame-rate source clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import struct

from .bridge_video_impairment import MAX_FRAMES, MAX_WIRE_BYTES
from .bridge_video_stream import (
    KEYFRAME, MAGIC as PACKET_MAGIC, MAX_PACKET, PACKET_HEADER, VERSION,
    VideoAccessUnit, packet_bytes,
)


DATAGRAM_MAGIC = b"CCP1"
DATAGRAM_HEADER = struct.Struct("!4sIHHI")
MAX_FRAGMENTS = 16384
MAX_UNIT_BYTES = MAX_PACKET + PACKET_HEADER.size


@dataclass(frozen=True)
class DatagramConfig:
    max_datagram_bytes: int = 1200

    def __post_init__(self) -> None:
        if (type(self.max_datagram_bytes) is not int
                or not 256 <= self.max_datagram_bytes <= 9000):
            raise ValueError("max_datagram_bytes must be an integer in 256..9000")

    @property
    def body_capacity(self) -> int:
        return self.max_datagram_bytes - DATAGRAM_HEADER.size


@dataclass(frozen=True)
class VideoDatagram:
    frame_seq: int
    index: int
    count: int
    total_size: int
    body: bytes


def _validate_fragment(fragment: VideoDatagram, config: DatagramConfig) -> None:
    if not isinstance(config, DatagramConfig):
        raise ValueError("invalid datagram configuration")
    if not isinstance(fragment, VideoDatagram):
        raise ValueError("invalid video datagram")
    if (type(fragment.frame_seq) is not int or not 0 <= fragment.frame_seq < MAX_FRAMES
            or type(fragment.total_size) is not int
            or not PACKET_HEADER.size < fragment.total_size <= MAX_UNIT_BYTES
            or type(fragment.count) is not int or not 1 <= fragment.count <= MAX_FRAGMENTS
            or type(fragment.index) is not int or not 0 <= fragment.index < fragment.count
            or not isinstance(fragment.body, bytes)):
        raise ValueError("invalid video datagram fields")
    expected_count = (fragment.total_size + config.body_capacity - 1) // config.body_capacity
    expected_size = min(config.body_capacity,
                        fragment.total_size - fragment.index * config.body_capacity)
    if fragment.count != expected_count or len(fragment.body) != expected_size:
        raise ValueError("invalid video datagram count or body length")


def datagram_bytes(fragment: VideoDatagram, config: DatagramConfig) -> bytes:
    _validate_fragment(fragment, config)
    return DATAGRAM_HEADER.pack(
        DATAGRAM_MAGIC, fragment.frame_seq, fragment.index,
        fragment.count, fragment.total_size,
    ) + fragment.body


def parse_datagram(wire: bytes, config: DatagramConfig) -> VideoDatagram:
    if not isinstance(config, DatagramConfig):
        raise ValueError("invalid datagram configuration")
    if (not isinstance(wire, bytes)
            or not DATAGRAM_HEADER.size < len(wire) <= config.max_datagram_bytes):
        raise ValueError("invalid datagram wire length")
    magic, seq, index, count, total = DATAGRAM_HEADER.unpack_from(wire)
    if magic != DATAGRAM_MAGIC:
        raise ValueError("invalid datagram magic")
    fragment = VideoDatagram(seq, index, count, total, wire[DATAGRAM_HEADER.size:])
    _validate_fragment(fragment, config)
    return fragment


def fragment_unit(unit: VideoAccessUnit, config: DatagramConfig) -> tuple[VideoDatagram, ...]:
    if not isinstance(config, DatagramConfig):
        raise ValueError("invalid datagram configuration")
    if (not isinstance(unit, VideoAccessUnit) or type(unit.seq) is not int
            or not 0 <= unit.seq < MAX_FRAMES or type(unit.keyframe) is not bool):
        raise ValueError("invalid source access unit")
    framed = packet_bytes(unit)
    count = (len(framed) + config.body_capacity - 1) // config.body_capacity
    if count > MAX_FRAGMENTS:
        raise ValueError("access unit requires too many datagrams")
    return tuple(
        VideoDatagram(unit.seq, index, count, len(framed),
                      framed[index * config.body_capacity:(index + 1) * config.body_capacity])
        for index in range(count)
    )


def _time_ms(value: float, *, allow_infinity: bool = False) -> float:
    if type(value) not in (int, float) or value < 0:
        raise ValueError("time must be a nonnegative number")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError("time must be finite") from None
    if not math.isfinite(result) and not (allow_infinity and result == math.inf):
        raise ValueError("time must be finite")
    return result


@dataclass
class _PartialFrame:
    count: int
    total_size: int
    first_arrival_ms: float
    last_arrival_ms: float
    parts: dict[int, bytes] = field(default_factory=dict)
    payload_bytes: int = 0

    @property
    def wire_bytes(self) -> int:
        return self.payload_bytes + DATAGRAM_HEADER.size * len(self.parts)


class AccessUnitReassembler:
    """Reconstruct exactly once within explicit body-byte and frame bounds.

    The byte budget covers stored datagram bodies; Python metadata, the
    returned completed unit and temporary reconstruction storage are not a
    process-memory claim. Metadata is separately bounded by frame/fragment
    limits. Oldest means first fragment arrival, then source sequence.

    Conflicting active-frame metadata/body bytes invalidate that frame and
    raise ValueError. Later datagrams for any terminal frame are discarded
    without retaining their bodies. No retransmission or repair is inferred.
    """

    def __init__(self, config: DatagramConfig, *, fps: int, max_age_ms: float,
                 max_buffer_bytes: int = 65536, max_pending_frames: int = MAX_FRAMES):
        if not isinstance(config, DatagramConfig):
            raise ValueError("invalid datagram configuration")
        if type(fps) is not int or not 1 <= fps <= 60:
            raise ValueError("fps must be an integer in 1..60")
        age = _time_ms(max_age_ms)
        if age > 10000:
            raise ValueError("max_age_ms must be in 0..10000")
        if (type(max_buffer_bytes) is not int
                or not 1 <= max_buffer_bytes <= MAX_WIRE_BYTES):
            raise ValueError("invalid receiver byte budget")
        if (type(max_pending_frames) is not int
                or not 1 <= max_pending_frames <= MAX_FRAMES):
            raise ValueError("invalid pending frame limit")
        self.config = config
        self.fps = fps
        self.max_age_ms = age
        self.max_buffer_bytes = max_buffer_bytes
        self.max_pending_frames = max_pending_frames
        self._pending: dict[int, _PartialFrame] = {}
        self._terminal: dict[int, dict] = {}
        self._observed: set[int] = set()
        self._receipts: dict[int, dict] = {}
        self._clock_ms = 0.0
        self._last_finite_ms = 0.0
        self._source_count: int | None = None
        self._finished = False
        self._buffered_payload_bytes = 0
        self._buffered_datagrams = 0
        self._peak_buffer_bytes = 0
        self._peak_pending_frames = 0
        self._counts = {
            f"{stage}_{quantity}": 0
            for stage in ("received", "completed", "dropped")
            for quantity in ("datagrams", "payload_bytes", "wire_bytes")
        }
        self._completed_media_bytes = 0
        self._duplicate_datagrams = 0
        self._conflicting_datagrams = 0
        self._terminal_datagrams = 0
        self._late_datagrams = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def pending_bytes(self) -> int:
        return self._buffered_payload_bytes

    def _deadline(self, seq: int) -> float:
        return seq * 1000.0 / self.fps + self.max_age_ms

    def _count(self, stage: str, count: int, payload_bytes: int) -> None:
        self._counts[f"{stage}_datagrams"] += count
        self._counts[f"{stage}_payload_bytes"] += payload_bytes
        self._counts[f"{stage}_wire_bytes"] += payload_bytes + DATAGRAM_HEADER.size * count

    def _terminalize(self, seq: int, reason: str, at_ms: float, *, detail: str,
                     fragment: VideoDatagram | None = None,
                     arrival_ms: float | None = None) -> None:
        if seq in self._terminal:
            return
        state = self._pending.pop(seq, None)
        count = len(state.parts) if state else 0
        payload = state.payload_bytes if state else 0
        expected = state.count if state else (fragment.count if fragment else None)
        self._buffered_payload_bytes -= payload
        self._buffered_datagrams -= count
        self._count("completed" if reason == "complete" else "dropped", count, payload)
        self._terminal[seq] = {
            "frame_seq": seq, "reason": reason, "detail": detail,
            "terminal_ms": at_ms, "observed": seq in self._observed,
            "first_arrival_ms": state.first_arrival_ms if state else arrival_ms,
            "last_arrival_ms": state.last_arrival_ms if state else arrival_ms,
            "expected_fragments": expected, "accepted_fragments": count,
            "missing_fragments": expected - count if expected is not None else None,
            "accepted_payload_bytes": payload,
            "accepted_wire_bytes": payload + DATAGRAM_HEADER.size * count,
        }

    def advance(self, now_ms: float) -> None:
        now = _time_ms(now_ms, allow_infinity=True)
        if now < self._clock_ms:
            raise ValueError("receiver time cannot move backwards")
        self._clock_ms = now
        if math.isfinite(now):
            self._last_finite_ms = now
        if self.max_age_ms:
            expired = sorted((self._deadline(seq), seq) for seq in self._pending
                             if self._deadline(seq) < now)
            for deadline, seq in expired:
                self._terminalize(seq, "expired", deadline, detail="age_bound")

    def _oldest(self) -> int:
        return min(self._pending, key=lambda seq: (self._pending[seq].first_arrival_ms, seq))

    def receive(self, fragment: VideoDatagram, arrival_ms: float) -> VideoAccessUnit | None:
        _validate_fragment(fragment, self.config)
        arrival = _time_ms(arrival_ms)
        if self._finished:
            raise ValueError("receiver is already finished")
        if arrival < fragment.frame_seq * 1000.0 / self.fps:
            raise ValueError("datagram arrived before its source capture")
        self.advance(arrival)
        seq = fragment.frame_seq
        self._observed.add(seq)
        self._count("received", 1, len(fragment.body))
        receipt = self._receipts.setdefault(seq, {
            "received_datagrams": 0, "received_payload_bytes": 0, "received_wire_bytes": 0,
            "first_received_ms": arrival, "last_received_ms": arrival,
        })
        receipt["received_datagrams"] += 1
        receipt["received_payload_bytes"] += len(fragment.body)
        receipt["received_wire_bytes"] += len(fragment.body) + DATAGRAM_HEADER.size
        receipt["last_received_ms"] = arrival
        late = bool(self.max_age_ms and self._deadline(seq) < arrival)
        self._late_datagrams += int(late)
        if seq in self._terminal:
            self._terminal_datagrams += 1
            self._count("dropped", 1, len(fragment.body))
            return None
        if late:
            self._terminalize(seq, "expired", self._deadline(seq), detail="first_fragment_late",
                              fragment=fragment, arrival_ms=arrival)
            self._count("dropped", 1, len(fragment.body))
            return None
        state = self._pending.get(seq)
        if state and (state.count != fragment.count or state.total_size != fragment.total_size
                      or (fragment.index in state.parts
                          and state.parts[fragment.index] != fragment.body)):
            self._conflicting_datagrams += 1
            self._terminalize(seq, "invalid", arrival, detail="conflicting_datagram")
            self._count("dropped", 1, len(fragment.body))
            raise ValueError("conflicting datagram for active frame")
        if state and fragment.index in state.parts:
            self._duplicate_datagrams += 1
            self._count("dropped", 1, len(fragment.body))
            return None
        if len(fragment.body) > self.max_buffer_bytes:
            self._terminalize(seq, "budget", arrival, detail="fragment_exceeds_byte_budget",
                              fragment=fragment, arrival_ms=arrival)
            self._count("dropped", 1, len(fragment.body))
            return None
        while (self._buffered_payload_bytes + len(fragment.body) > self.max_buffer_bytes
               or (seq not in self._pending and len(self._pending) >= self.max_pending_frames)):
            oldest = self._oldest()
            detail = ("byte_budget" if self._buffered_payload_bytes + len(fragment.body)
                      > self.max_buffer_bytes else "pending_frame_limit")
            self._terminalize(oldest, "budget", arrival, detail=detail)
            if oldest == seq:
                self._count("dropped", 1, len(fragment.body))
                return None
        if state is None:
            state = _PartialFrame(fragment.count, fragment.total_size, arrival, arrival)
            self._pending[seq] = state
        state.parts[fragment.index] = fragment.body
        state.payload_bytes += len(fragment.body)
        state.last_arrival_ms = arrival
        self._buffered_payload_bytes += len(fragment.body)
        self._buffered_datagrams += 1
        self._peak_buffer_bytes = max(self._peak_buffer_bytes, self._buffered_payload_bytes)
        self._peak_pending_frames = max(self._peak_pending_frames, len(self._pending))
        if len(state.parts) != state.count:
            return None
        framed = b"".join(state.parts[index] for index in range(state.count))
        try:
            unit = self._reconstruct(seq, framed, state.total_size)
        except ValueError:
            self._terminalize(seq, "invalid", arrival, detail="invalid_access_unit_header")
            raise
        self._completed_media_bytes += len(unit.data)
        self._terminalize(seq, "complete", arrival, detail="all_fragments")
        return unit

    def _reconstruct(self, seq: int, framed: bytes, total_size: int) -> VideoAccessUnit:
        magic, version, flags, actual_seq, pts, capture, size = PACKET_HEADER.unpack_from(framed)
        stamp = round(seq * 1_000_000_000 / self.fps)
        if (len(framed) != total_size or magic != PACKET_MAGIC or version != VERSION
                or flags & ~KEYFRAME or actual_seq != seq or pts != stamp or capture != stamp
                or not 0 < size <= MAX_PACKET or size + PACKET_HEADER.size != total_size):
            raise ValueError("invalid reconstructed access unit header or source clock")
        return VideoAccessUnit(seq, pts, capture, framed[PACKET_HEADER.size:], bool(flags & KEYFRAME))

    def finish(self, source_count: int, now_ms: float) -> dict:
        """Close after all intended arrivals, accounting for entirely unseen frames.

        Infinity is a virtual drain marker only. Expired states retain their
        finite source-clock deadline; other terminal rows use the finite
        supplied time, or the later of source end and last finite observation.
        """
        if type(source_count) is not int or not 0 <= source_count <= MAX_FRAMES:
            raise ValueError("invalid receiver source count")
        now = _time_ms(now_ms, allow_infinity=True)
        if any(seq >= source_count for seq in self._observed):
            raise ValueError("source count excludes observed frames")
        if self._finished:
            if source_count != self._source_count:
                raise ValueError("receiver already finished with another source count")
            return self.report()
        if now < max(0, (source_count - 1) * 1000.0 / self.fps):
            raise ValueError("finish precedes the final source capture")
        self.advance(now)
        fallback = (now if math.isfinite(now)
                    else max(self._last_finite_ms, source_count * 1000.0 / self.fps))
        for seq in range(source_count):
            if seq not in self._terminal:
                at_ms = self._deadline(seq) if not math.isfinite(now) and self.max_age_ms else fallback
                self._terminalize(seq, "missing", at_ms,
                                  detail="incomplete_at_finish" if seq in self._pending
                                  else "no_datagrams_received")
        self._source_count = source_count
        self._finished = True
        return self.report()

    def report(self) -> dict:
        rows = [dict(self._terminal[seq]) for seq in sorted(self._terminal)]
        for row in rows:
            receipt = self._receipts.get(row["frame_seq"], {
                "received_datagrams": 0, "received_payload_bytes": 0,
                "received_wire_bytes": 0, "first_received_ms": None, "last_received_ms": None,
            })
            row.update(receipt)
            for kind in ("payload", "wire"):
                useful = row[f"accepted_{kind}_bytes"] if row["reason"] == "complete" else 0
                row[f"usable_{kind}_bytes"] = useful
                row[f"unusable_{kind}_bytes"] = row[f"received_{kind}_bytes"] - useful
        return {
            "source_count": self._source_count, "frames_observed": len(self._observed),
            "finished": self._finished,
            **{f"{reason}_frames": sum(row["reason"] == reason for row in rows)
               for reason in ("complete", "expired", "missing", "budget", "invalid")},
            "unseen_frames": sum(not row["observed"] for row in rows),
            "completed_seqs": [row["frame_seq"] for row in rows if row["reason"] == "complete"],
            "pending_frames": self.pending_count,
            **self._counts, "completed_media_bytes": self._completed_media_bytes,
            "buffered_datagrams": self._buffered_datagrams,
            "buffered_payload_bytes": self._buffered_payload_bytes,
            "buffered_wire_bytes": self._buffered_payload_bytes + DATAGRAM_HEADER.size * self._buffered_datagrams,
            "duplicate_datagrams": self._duplicate_datagrams,
            "conflicting_datagrams": self._conflicting_datagrams,
            "terminal_datagrams": self._terminal_datagrams, "late_datagrams": self._late_datagrams,
            "peak_buffer_bytes": self._peak_buffer_bytes,
            "peak_pending_frames": self._peak_pending_frames,
            "max_buffer_bytes": self.max_buffer_bytes,
            "max_pending_frames": self.max_pending_frames,
            "max_age_ms": self.max_age_ms, "terminal_frames": rows,
            "payload_note": "datagram body bytes include the original CCV1 access-unit header; "
                            "wire bytes additionally include the custom CCP1 datagram header",
            "buffer_note": "byte budget and peak count stored datagram bodies; frame metadata, "
                           "temporary reconstruction and returned units are outside that byte count",
        }
