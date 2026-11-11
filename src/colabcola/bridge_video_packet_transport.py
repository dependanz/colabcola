"""v33: frozen-source datagram scheduling and a byte-bounded virtual link.

Every mode offers the same framed bytes. A batch is one atomic admission
and serialization of all of a frame's datagrams, including every datagram
header. Packet mode makes separate attempts. This is a custom offline
model, not UDP, RTP, a socket implementation, or a capacity estimator.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import math
import random

from .bridge_video_datagram import (
    DatagramConfig, VideoDatagram, datagram_bytes, fragment_unit,
)
from .bridge_video_impairment import LinkConfig, MAX_FRAMES, MAX_WIRE_BYTES, _serialize_ms
from .bridge_video_sender_service import SenderServiceConfig
from .bridge_video_stream import PACKET_HEADER, VideoAccessUnit, is_keyframe


_EVENT_TIE_MS = 1e-9
_PEAK_WINDOWS_MS = (5, 20, 100, 300)


def _finite_time(value: float) -> bool:
    if type(value) not in (int, float) or value < 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _fragment_id(fragment: VideoDatagram) -> list[int]:
    return [fragment.frame_seq, fragment.index]


@dataclass(frozen=True)
class PacketSenderConfig:
    grouping: str = "packet"
    pacing: str = "queue"
    max_pending: int = 3
    max_age_ms: float = 100.0

    def __post_init__(self) -> None:
        if type(self.grouping) is not str or self.grouping not in ("batch", "packet"):
            raise ValueError("grouping must be batch or packet")
        if type(self.pacing) is not str or self.pacing not in ("none", "fixed", "queue"):
            raise ValueError("pacing must be none, fixed or queue")
        if type(self.max_pending) is not int or not 1 <= self.max_pending <= MAX_FRAMES:
            raise ValueError("max_pending must be an integer in 1..120")
        if not _finite_time(self.max_age_ms) or self.max_age_ms > 2000:
            raise ValueError("max_age_ms must be finite and in 0..2000")


@dataclass(frozen=True)
class PacketDispatch:
    datagrams: tuple[VideoDatagram, ...]
    source_ms: float
    dispatch_ms: float
    wire_size: int
    pacing_kbps: float
    queue_delay_ms: float


@dataclass
class _QueuedFrame:
    packet: VideoAccessUnit
    source_ms: float
    datagrams: tuple[VideoDatagram, ...]
    sizes: tuple[int, ...]
    verified_idr: bool
    next_index: int = 0

    @property
    def remaining_bytes(self) -> int:
        return sum(self.sizes[self.next_index:])


def _dispatch_peaks(records: list[dict]) -> list[dict]:
    # Aggregate equal timestamps first: every packet in a simultaneous burst
    # belongs to the same half-open window, regardless of its list position.
    times: list[float] = []
    sizes: list[int] = []
    for row in records:
        if times and times[-1] == row["dispatch_ms"]:
            sizes[-1] += row["wire_size"]
        else:
            times.append(row["dispatch_ms"])
            sizes.append(row["wire_size"])
    peaks = []
    for duration in _PEAK_WINDOWS_MS:
        end = 0
        total = 0
        best = None
        for start, at_ms in enumerate(times):
            while end < len(times) and times[end] < at_ms + duration - _EVENT_TIE_MS:
                total += sizes[end]
                end += 1
            if best is None or total > best["wire_bytes"]:
                best = {"start_ms": at_ms, "wire_bytes": total,
                        "attempted_wire_kbps": total * 8 / duration}
            total -= sizes[start]
        peaks.append({"window_ms": duration, "peak": best})
    return peaks


class PacketSenderQueue:
    """A local queue with no link, admission, feedback, or future-rate input.

    Advance to a source capture, enqueue all its frozen datagrams atomically,
    then advance again to release immediately ready attempts. Bounds count
    frames with unsent data. Expiry/overflow discard only unsent fragments
    and dependent queued frames, retaining the next verified IDR. A missing
    IDR suppresses future predicted frames; this frozen source cannot repair.

    Pacing debt is charged for every attempted byte, even when downstream
    admission fails. No credit accumulates while idle. Service changes only
    after complete enqueue, expiry-chain removal, or pop-plus-debt-charge.
    """

    def __init__(self, config: PacketSenderConfig, datagram: DatagramConfig, *,
                 fps: int, initial_target_kbps: int,
                 service: SenderServiceConfig = SenderServiceConfig()):
        if not isinstance(config, PacketSenderConfig):
            raise ValueError("invalid packet sender configuration")
        if not isinstance(datagram, DatagramConfig):
            raise ValueError("invalid datagram configuration")
        if not isinstance(service, SenderServiceConfig):
            raise ValueError("invalid sender service configuration")
        if type(fps) is not int or not 1 <= fps <= 60:
            raise ValueError("invalid sender source rate")
        if type(initial_target_kbps) is not int or not 20 <= initial_target_kbps <= 20_000:
            raise ValueError("invalid sender target bitrate")
        if config.pacing == "queue" and not service.queue_boost:
            raise ValueError("queue pacing requires queue_boost enabled")
        self.config = config
        self.datagram_config = datagram
        self.service_config = service
        self.fps = fps
        self.initial_target_kbps = initial_target_kbps
        self._base_kbps = initial_target_kbps + PACKET_HEADER.size * 8 * fps / 1000.0
        self._cap_kbps = self._base_kbps * service.max_rate_factor
        self._service_kbps = self._base_kbps if config.pacing != "none" else 0.0
        self.clock_ms = 0.0
        self.debt_bits = 0.0
        self.frames_observed = 0
        self.need_idr = False
        self._queue: deque[_QueuedFrame] = deque()
        self._pending_wire_bytes = 0
        self._pending_packets = 0
        self._offered_wire_bytes = 0
        self._offered_packets = 0
        self._dispatched_wire_bytes = 0
        self._dispatched_packets = 0
        self._dropped_wire_bytes = 0
        self._peak_pending_frames = 0
        self._peak_pending_packets = 0
        self._peak_pending_bytes = 0
        self._max_queue_delay_ms = 0.0
        self._dispatches: list[dict] = []
        self._service_decisions: list[dict] = []
        self._frame_records: list[dict] = []
        self.drops: list[dict] = []
        self.frame_drops: list[dict] = []

    @property
    def base_kbps(self) -> float:
        return self._base_kbps

    @property
    def cap_kbps(self) -> float:
        return self._cap_kbps

    @property
    def pacing_kbps(self) -> float:
        return self._service_kbps

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    @property
    def pending_packets(self) -> int:
        return self._pending_packets

    def _validate_time(self, now_ms: float) -> None:
        if not _finite_time(now_ms) or now_ms < self.clock_ms:
            raise ValueError("sender time must be finite and monotonic")

    def _refresh_service(self, cause: str) -> None:
        work_bits = self.debt_bits + 8 * self._pending_wire_bytes
        if self.config.pacing == "none":
            requested = 0.0
        elif self.config.pacing == "fixed":
            requested = self._base_kbps
        else:
            requested = max(self._base_kbps, work_bits / self.service_config.drain_ms)
        self._service_kbps = min(self._cap_kbps, requested)
        self._service_decisions.append({
            "at_ms": self.clock_ms, "cause": cause,
            "debt_bits": self.debt_bits, "pending_wire_bytes": self._pending_wire_bytes,
            "pending_packets": self._pending_packets, "pending_frames": len(self._queue),
            "work_bits": work_bits, "requested_kbps": requested,
            "pacing_kbps": self._service_kbps,
            "boosted": self._service_kbps > self._base_kbps,
            "cap_limited": requested > self._cap_kbps,
        })

    def _settle_debt(self, at_ms: float) -> None:
        self.debt_bits = max(0.0, self.debt_bits - (at_ms - self.clock_ms) * self.pacing_kbps)
        self.clock_ms = at_ms

    def _next_event(self) -> tuple[float, bool]:
        ready_ms = (self.clock_ms + self.debt_bits / self.pacing_kbps
                    if self.config.pacing != "none" else self.clock_ms)
        expiry_ms = (self._queue[0].source_ms + self.config.max_age_ms
                     if self.config.max_age_ms else math.inf)
        expires = expiry_ms <= ready_ms + _EVENT_TIE_MS
        # Expiry wins a near tie at its actual timestamp, not the slightly
        # earlier rounded readiness. An earlier advance endpoint must wait.
        at_ms = expiry_ms if expires else ready_ms
        return max(self.clock_ms, at_ms), expires

    @property
    def next_event_ms(self) -> float | None:
        return self._next_event()[0] if self._queue else None

    def _drop_frame(self, frame: _QueuedFrame, reason: str) -> None:
        first = frame.next_index
        remaining = frame.datagrams[first:]
        wire_size = frame.remaining_bytes
        delay = self.clock_ms - frame.source_ms
        self._dropped_wire_bytes += wire_size
        self._max_queue_delay_ms = max(self._max_queue_delay_ms, delay)
        record = self._frame_records[frame.packet.seq]
        record["dropped_packets"] += len(remaining)
        record["dropped_wire_bytes"] += wire_size
        self.frame_drops.append({
            "source_seq": frame.packet.seq, "source_ms": frame.source_ms,
            "at_ms": self.clock_ms, "reason": reason, "wire_size": wire_size,
            "fragment_ids": [_fragment_id(d) for d in remaining],
            "dropped_packets": len(remaining), "already_dispatched_packets": first,
            "partial_frame": first > 0, "queue_delay_ms": delay,
            "verified_idr": frame.verified_idr,
        })
        self.drops.extend({
            "source_seq": d.frame_seq, "fragment_index": d.index,
            "fragment_id": _fragment_id(d), "source_ms": frame.source_ms,
            "at_ms": self.clock_ms, "reason": reason,
            "wire_size": size, "partial_frame": first > 0,
        } for d, size in zip(remaining, frame.sizes[first:]))

    def _pop_frame(self) -> _QueuedFrame:
        frame = self._queue.popleft()
        self._pending_wire_bytes -= frame.remaining_bytes
        self._pending_packets -= len(frame.datagrams) - frame.next_index
        return frame

    def _discard_chain(self, reason: str) -> None:
        self._drop_frame(self._pop_frame(), reason)
        while self._queue and not self._queue[0].verified_idr:
            self._drop_frame(self._pop_frame(), "sender_dependency")
        self.need_idr = not self._queue

    def enqueue(self, unit: VideoAccessUnit, datagrams: tuple[VideoDatagram, ...],
                at_ms: float) -> None:
        self._validate_time(at_ms)
        if at_ms != self.clock_ms:
            raise ValueError("advance sender before applying a source action")
        if (not isinstance(unit, VideoAccessUnit) or unit.seq != self.frames_observed
                or unit.seq >= MAX_FRAMES or at_ms < unit.seq * 1000.0 / self.fps
                or not math.isclose(at_ms, unit.seq * 1000.0 / self.fps,
                                    rel_tol=0, abs_tol=1e-6)):
            raise ValueError("sender captures require consecutive IDs and original source time")
        # Compare the full ordered framing, not just fragment counts or IDs.
        # No caller can silently change source bytes between comparison arms.
        if type(datagrams) is not tuple or datagrams != fragment_unit(unit, self.datagram_config):
            raise ValueError("sender datagrams must exactly frame the supplied source unit")
        sizes = tuple(len(datagram_bytes(d, self.datagram_config)) for d in datagrams)
        wire_size = sum(sizes)
        if self._offered_wire_bytes + wire_size > MAX_WIRE_BYTES:
            raise ValueError("sender offered wire budget exceeded")
        frame = _QueuedFrame(unit, unit.seq * 1000.0 / self.fps, datagrams, sizes,
                             bool(unit.keyframe and is_keyframe(unit.data)))
        self.frames_observed += 1
        self._offered_wire_bytes += wire_size
        self._offered_packets += len(datagrams)
        self._frame_records.append({
            "source_seq": unit.seq, "source_ms": frame.source_ms,
            "offered_packets": len(datagrams), "offered_wire_bytes": wire_size,
            "dispatched_packets": 0, "dispatched_wire_bytes": 0,
            "dropped_packets": 0, "dropped_wire_bytes": 0,
            "first_dispatch_ms": None, "last_dispatch_ms": None,
        })
        if len(self._queue) >= self.config.max_pending:
            self._discard_chain("sender_queue_full")
        if self.need_idr and not frame.verified_idr:
            self._drop_frame(frame, "sender_dependency")
            self._refresh_service("enqueue")
            return
        if frame.verified_idr:
            self.need_idr = False
        self._queue.append(frame)
        self._pending_wire_bytes += wire_size
        self._pending_packets += len(datagrams)
        self._peak_pending_frames = max(self._peak_pending_frames, len(self._queue))
        self._peak_pending_packets = max(self._peak_pending_packets, self._pending_packets)
        self._peak_pending_bytes = max(self._peak_pending_bytes, self._pending_wire_bytes)
        self._refresh_service("enqueue")

    def advance(self, now_ms: float) -> list[PacketDispatch]:
        self._validate_time(now_ms)
        output = []
        while self._queue:
            at_ms, expires = self._next_event()
            if at_ms > now_ms:
                break
            self._settle_debt(at_ms)
            if expires:
                self._discard_chain("sender_age")
                self._refresh_service("expiry")
                continue
            frame = self._queue[0]
            first = frame.next_index
            end = len(frame.datagrams) if self.config.grouping == "batch" else first + 1
            datagrams = frame.datagrams[first:end]
            wire_size = sum(frame.sizes[first:end])
            delay = at_ms - frame.source_ms
            dispatch = PacketDispatch(datagrams, frame.source_ms, at_ms, wire_size,
                                      self.pacing_kbps, delay)
            output.append(dispatch)
            frame.next_index = end
            self._pending_packets -= len(datagrams)
            self._pending_wire_bytes -= wire_size
            if end == len(frame.datagrams):
                self._queue.popleft()
            self._dispatched_packets += len(datagrams)
            self._dispatched_wire_bytes += wire_size
            self._max_queue_delay_ms = max(self._max_queue_delay_ms, delay)
            record = self._frame_records[frame.packet.seq]
            record["dispatched_packets"] += len(datagrams)
            record["dispatched_wire_bytes"] += wire_size
            if record["first_dispatch_ms"] is None:
                record["first_dispatch_ms"] = at_ms
            record["last_dispatch_ms"] = at_ms
            # Readiness paid the previous debt in full; floating residues at
            # that exact event do not create a refund or reusable credit.
            self.debt_bits = wire_size * 8.0 if self.config.pacing != "none" else 0.0
            self._refresh_service("dispatch")
            self._dispatches.append({
                "source_seq": frame.packet.seq, "source_ms": frame.source_ms,
                "fragment_ids": [_fragment_id(d) for d in datagrams],
                "packet_count": len(datagrams), "dispatch_ms": at_ms,
                "wire_size": wire_size, "pacing_kbps": dispatch.pacing_kbps,
                "queue_delay_ms": delay, "charged_bits": self.debt_bits,
                "remaining_debt_bits": self.debt_bits,
                "service_after_kbps": self.pacing_kbps,
            })
        self._settle_debt(now_ms)
        return output

    def report(self) -> dict:
        if (self._offered_wire_bytes != self._dispatched_wire_bytes
                + self._dropped_wire_bytes + self._pending_wire_bytes
                or self._offered_packets != self._dispatched_packets
                + len(self.drops) + self._pending_packets):
            raise RuntimeError("packet sender byte/packet accounting mismatch")
        frames = []
        states = {"fully_dispatched": 0, "fully_dropped": 0,
                  "partly_dispatched_then_dropped": 0, "pending": 0}
        for original in self._frame_records:
            row = dict(original)
            row["pending_packets"] = (row["offered_packets"] - row["dispatched_packets"]
                                      - row["dropped_packets"])
            row["pending_wire_bytes"] = (row["offered_wire_bytes"] - row["dispatched_wire_bytes"]
                                         - row["dropped_wire_bytes"])
            state = ("pending" if row["pending_packets"] else
                     "fully_dispatched" if not row["dropped_packets"] else
                     "fully_dropped" if not row["dispatched_packets"] else
                     "partly_dispatched_then_dropped")
            states[state] += 1
            row["state"] = state
            row["complete_dispatch_ms"] = (row["last_dispatch_ms"]
                                           if state == "fully_dispatched" else None)
            frames.append(row)
        if states["pending"] != len(self._queue) or sum(states.values()) != self.frames_observed:
            raise RuntimeError("packet sender frame accounting mismatch")
        last_capture = ((self.frames_observed - 1) * 1000.0 / self.fps
                        if self.frames_observed else None)
        decisions = self._service_decisions
        cap_binding_ms = sum(
            ((decisions[i + 1]["at_ms"] if i + 1 < len(decisions) else self.clock_ms)
             - row["at_ms"])
            for i, row in enumerate(decisions) if row["cap_limited"])
        return {
            "config": asdict(self.config), "datagram_config": asdict(self.datagram_config),
            "fps": self.fps, "initial_target_kbps": self.initial_target_kbps,
            "frames_observed": self.frames_observed, "frame_states": states,
            "pending_frames": len(self._queue), "pending_packets": self._pending_packets,
            "need_idr": self.need_idr, "local_repair_requests": 0,
            "offered_packets": self._offered_packets,
            "dispatched_packets": self._dispatched_packets, "dropped_packets": len(self.drops),
            "offered_wire_bytes": self._offered_wire_bytes,
            "dispatched_wire_bytes": self._dispatched_wire_bytes,
            "dropped_wire_bytes": self._dropped_wire_bytes,
            "pending_wire_bytes": self._pending_wire_bytes,
            "attempted_dispatches": len(self._dispatches),
            "peak_pending_frames": self._peak_pending_frames,
            "peak_pending_packets": self._peak_pending_packets,
            "peak_pending_bytes": self._peak_pending_bytes,
            "max_queue_delay_ms": self._max_queue_delay_ms,
            "dispatches": [{**r, "fragment_ids": [list(x) for x in r["fragment_ids"]]}
                           for r in self._dispatches],
            "drops": [{**r, "fragment_id": list(r["fragment_id"])} for r in self.drops],
            "frame_drops": [{**r, "fragment_ids": [list(x) for x in r["fragment_ids"]]}
                            for r in self.frame_drops],
            "frames": frames, "clock_ms": self.clock_ms,
            "remaining_debt_bits": self.debt_bits,
            "event_tie_tolerance_ms": _EVENT_TIE_MS,
            "peak_dispatch_windows": _dispatch_peaks(self._dispatches),
            "dispatch_window_note": "half-open dispatch-time windows, including all bytes at each "
                                    "timestamp; endpoint ties within 1e-9 ms are excluded",
            "service": {
                "config": asdict(self.service_config), "base_kbps": self.base_kbps,
                "cap_kbps": self.cap_kbps, "current_kbps": self.pacing_kbps,
                "max_kbps": max((r["pacing_kbps"] for r in decisions), default=self.pacing_kbps),
                "boosted_decisions": sum(r["boosted"] for r in decisions),
                "cap_limited_decisions": sum(r["cap_limited"] for r in decisions),
                "cap_binding_ms": cap_binding_ms,
                "decisions": [dict(r) for r in decisions],
                "formula": "base = initial_target_kbps + 30 * 8 * fps / 1000; "
                           "work_bits = debt_bits + 8 * pending_datagram_wire_bytes; "
                           "queue service = min(base * max_rate_factor, max(base, work_bits / drain_ms)); "
                           "fixed service = base; unpaced service = 0",
            },
            "tail": {
                "last_capture_ms": last_capture,
                "last_dispatch_ms": self._dispatches[-1]["dispatch_ms"] if self._dispatches else None,
                "post_source_dispatches": sum(r["dispatch_ms"] > last_capture
                                              for r in self._dispatches) if last_capture is not None else 0,
                "debt_clear_at_current_service_ms": (self.clock_ms + self.debt_bits / self.pacing_kbps
                                                      if self.pacing_kbps else self.clock_ms),
            },
            "note": "all attempted datagram wire bytes incur pacing debt regardless of downstream "
                    "admission; local drops affect only unsent fragments and cannot refund debt; "
                    "the frozen source receives no repair or encoder requests; service refreshes only "
                    "after complete enqueue, expiry-chain removal and pop-plus-charge; "
                    "original capture age is retained and idle credit is never accumulated",
        }


@dataclass(frozen=True)
class ScheduledDatagram:
    fragment: VideoDatagram
    source_ms: float
    arrival_ms: float
    wire_size: int


class BytePacketLink:
    """FIFO atomic-group admission with a shared in-flight byte budget.

    Bytes occupy the budget from admission through serialization and
    propagation, including erased fragments. Each group's serialization
    integrates the configured capacity schedule once. Receiver age is not
    applied here. Frame jitter and fragment loss draws are paired across
    arms independently of earlier admissions or local sender decisions.

    Drain arrive_by(dispatch_ms) before every send. report/snapshot is only
    available once every admitted fragment has reached its arrival event.
    """

    def __init__(self, link: LinkConfig, datagram: DatagramConfig, *,
                 max_inflight_bytes: int = 16_384, erase_first_seqs: tuple[int, ...] = ()):
        if not isinstance(link, LinkConfig):
            raise ValueError("invalid link configuration")
        if not isinstance(datagram, DatagramConfig):
            raise ValueError("invalid datagram configuration")
        if type(max_inflight_bytes) is not int or not 1 <= max_inflight_bytes <= MAX_WIRE_BYTES:
            raise ValueError("max_inflight_bytes must be an integer within the wire budget")
        if (type(erase_first_seqs) is not tuple or len(erase_first_seqs) > MAX_FRAMES
                or any(type(seq) is not int or not 0 <= seq < MAX_FRAMES for seq in erase_first_seqs)
                or tuple(sorted(set(erase_first_seqs))) != erase_first_seqs):
            raise ValueError("erase_first_seqs must be sorted unique source IDs in 0..119")
        if link.drop_seqs:
            raise ValueError("legacy drop_seqs is unsupported; use erase_first_seqs")
        self.config = link
        self.datagram_config = datagram
        self.max_inflight_bytes = max_inflight_bytes
        self.erase_first_seqs = erase_first_seqs
        rng = random.Random(link.seed)
        self._frame_jitter = tuple(rng.uniform(-link.jitter_ms, link.jitter_ms)
                                   if link.jitter_ms else 0.0 for _ in range(MAX_FRAMES))
        self._inflight: deque[tuple[ScheduledDatagram, str, int]] = deque()
        self._inflight_bytes = 0
        self._inflight_group_counts: dict[int, int] = {}
        self._max_inflight_bytes = 0
        self._max_inflight_packets = 0
        self._max_inflight_groups = 0
        self._last_dispatch_ms = 0.0
        self._frontier_ms = 0.0
        self.link_free_ms = 0.0
        self.last_arrival_ms = 0.0
        self.wire_sent_bytes = 0
        self._offered_wire_bytes = 0
        self._offered_ids: set[tuple[int, int]] = set()
        self._frame_shapes: dict[int, tuple[int, int, float]] = {}
        self._packets: list[dict] = []
        self._groups: list[dict] = []
        self.delivered: list[ScheduledDatagram] = []

    @property
    def inflight_bytes(self) -> int:
        return self._inflight_bytes

    @property
    def next_arrival_ms(self) -> float | None:
        return self._inflight[0][0].arrival_ms if self._inflight else None

    def _erased(self, fragment: VideoDatagram) -> bool:
        if fragment.index == 0 and fragment.frame_seq in self.erase_first_seqs:
            return True
        if not self.config.loss_percent:
            return False
        identity = (self.config.seed.to_bytes(4, "big")
                    + fragment.frame_seq.to_bytes(4, "big") + fragment.index.to_bytes(4, "big"))
        draw = int.from_bytes(hashlib.sha256(b"CCV33-loss" + identity).digest()[:8], "big") / 2**64
        return draw * 100 < self.config.loss_percent

    def send(self, datagrams: tuple[VideoDatagram, ...], *, dispatch_ms: float,
             source_ms: float) -> None:
        if (not _finite_time(source_ms) or not _finite_time(dispatch_ms)
                or dispatch_ms < source_ms or dispatch_ms < self._last_dispatch_ms
                or dispatch_ms < self._frontier_ms):
            raise ValueError("dispatch time must be finite, monotonic and after source/frontier")
        if type(datagrams) is not tuple or not datagrams:
            raise ValueError("send requires a nonempty datagram tuple")
        sizes = tuple(len(datagram_bytes(d, self.datagram_config)) for d in datagrams)
        first = datagrams[0]
        shape = (first.count, first.total_size, source_ms)
        if any((d.frame_seq != first.frame_seq or (d.count, d.total_size, source_ms) != shape)
               for d in datagrams):
            raise ValueError("an atomic admission group must contain one consistent source frame")
        if first.frame_seq in self._frame_shapes and self._frame_shapes[first.frame_seq] != shape:
            raise ValueError("source frame metadata changed between fragment attempts")
        identities = {(d.frame_seq, d.index) for d in datagrams}
        if len(identities) != len(datagrams) or identities & self._offered_ids:
            raise ValueError("a source fragment can be attempted only once")
        wire_size = sum(sizes)
        if self._offered_wire_bytes + wire_size > MAX_WIRE_BYTES:
            raise ValueError("virtual link offered wire budget exceeded")
        if self._inflight and self._inflight[0][0].arrival_ms <= dispatch_ms:
            raise RuntimeError("receiver arrivals must be drained before send")
        admitted = self._inflight_bytes + wire_size <= self.max_inflight_bytes
        if admitted and self.wire_sent_bytes + wire_size > MAX_WIRE_BYTES:
            raise ValueError("virtual link admitted wire budget exceeded")
        # All input and aggregate bounds are checked before mutating state.
        group_id = len(self._groups)
        self._last_dispatch_ms = dispatch_ms
        self._offered_wire_bytes += wire_size
        self._offered_ids.update(identities)
        self._frame_shapes[first.frame_seq] = shape
        started = finished = arrival = None
        jitter = self._frame_jitter[first.frame_seq]
        before = self._inflight_bytes
        if admitted:
            started = max(dispatch_ms, self.link_free_ms)
            finished = _serialize_ms(started, wire_size * 8, self.config)
            arrival = max(finished, finished + self.config.delay_ms + jitter, self.last_arrival_ms)
            self.link_free_ms = finished
            self.last_arrival_ms = arrival
            self.wire_sent_bytes += wire_size
            self._inflight_bytes += wire_size
            self._inflight_group_counts[group_id] = len(datagrams)
        self._groups.append({
            "group_id": group_id, "source_seq": first.frame_seq, "source_ms": source_ms,
            "fragment_ids": [_fragment_id(d) for d in datagrams], "packet_count": len(datagrams),
            "dispatch_ms": dispatch_ms, "wire_size": wire_size, "admitted": admitted,
            "status": "admitted" if admitted else "queue_full",
            "serialization_start_ms": started, "serialization_finish_ms": finished,
            "arrival_ms": arrival, "jitter_ms": jitter,
            "inflight_bytes_before": before, "inflight_bytes_after": self._inflight_bytes,
        })
        for fragment, size in zip(datagrams, sizes):
            status = ("erased" if self._erased(fragment) else "delivered") if admitted else "queue_full"
            self._packets.append({
                "group_id": group_id, "source_seq": fragment.frame_seq,
                "fragment_index": fragment.index, "fragment_count": fragment.count,
                "fragment_id": _fragment_id(fragment), "source_ms": source_ms,
                "dispatch_ms": dispatch_ms, "wire_size": size, "status": status,
                "serialization_start_ms": started, "serialization_finish_ms": finished,
                "arrival_ms": arrival, "jitter_ms": jitter,
            })
            if admitted:
                self._inflight.append((ScheduledDatagram(fragment, source_ms, arrival, size),
                                       status, group_id))
        self._max_inflight_bytes = max(self._max_inflight_bytes, self._inflight_bytes)
        self._max_inflight_packets = max(self._max_inflight_packets, len(self._inflight))
        self._max_inflight_groups = max(self._max_inflight_groups, len(self._inflight_group_counts))

    def arrive_by(self, time_ms: float) -> list[ScheduledDatagram]:
        if ((not _finite_time(time_ms) and not (type(time_ms) is float and time_ms == math.inf))
                or time_ms < self._frontier_ms or time_ms < self._last_dispatch_ms):
            raise ValueError("arrival frontier must be monotonic and nonnegative")
        self._frontier_ms = time_ms
        observed = []
        while self._inflight and self._inflight[0][0].arrival_ms <= time_ms:
            item, status, group_id = self._inflight.popleft()
            self._inflight_bytes -= item.wire_size
            self._inflight_group_counts[group_id] -= 1
            if not self._inflight_group_counts[group_id]:
                del self._inflight_group_counts[group_id]
            if status == "delivered":
                self.delivered.append(item)
                observed.append(item)
        return observed

    def report(self) -> dict:
        if self._inflight:
            raise RuntimeError("cannot finalize an undrained byte packet link")
        counts = {status: 0 for status in ("delivered", "erased", "queue_full")}
        sizes = dict.fromkeys(counts, 0)
        for row in self._packets:
            counts[row["status"]] += 1
            sizes[row["status"]] += row["wire_size"]
        admitted_packets = counts["delivered"] + counts["erased"]
        if (self._offered_wire_bytes != sum(sizes.values())
                or self.wire_sent_bytes != sizes["delivered"] + sizes["erased"]
                or len(self._offered_ids) != sum(counts.values())
                or len(self.delivered) != counts["delivered"]
                or self._inflight_bytes != 0 or self._inflight_group_counts):
            raise RuntimeError("byte packet link accounting mismatch")
        accepted_groups = sum(row["admitted"] for row in self._groups)
        return {
            "config": asdict(self.config), "datagram_config": asdict(self.datagram_config),
            "max_inflight_bytes_limit": self.max_inflight_bytes,
            "erase_first_seqs": list(self.erase_first_seqs),
            "source_frames_offered": len(self._frame_shapes),
            "offered_packets": len(self._packets), "offered_wire_bytes": self._offered_wire_bytes,
            "admitted_packets": admitted_packets, "admitted_wire_bytes": self.wire_sent_bytes,
            "wire_sent_bytes": self.wire_sent_bytes,
            "delivered_packets": counts["delivered"], "delivered_wire_bytes": sizes["delivered"],
            "erased_packets": counts["erased"], "erased_wire_bytes": sizes["erased"],
            "rejected_packets": counts["queue_full"], "rejected_wire_bytes": sizes["queue_full"],
            "queue_full_packets": counts["queue_full"], "queue_full_wire_bytes": sizes["queue_full"],
            "attempted_groups": len(self._groups), "admitted_groups": accepted_groups,
            "rejected_groups": len(self._groups) - accepted_groups,
            "max_inflight_bytes": self._max_inflight_bytes,
            "max_inflight_packets": self._max_inflight_packets,
            "max_inflight_groups": self._max_inflight_groups,
            "last_serialization_finish_ms": self.link_free_ms,
            "last_arrival_ms": self.last_arrival_ms,
            "packets": [{**r, "fragment_id": list(r["fragment_id"])} for r in self._packets],
            "groups": [{**r, "fragment_ids": [list(x) for x in r["fragment_ids"]]}
                       for r in self._groups],
            "jitter_coupling": "one seeded jitter draw per source sequence, precomputed independently "
                               "of admission and shared by every fragment in that frame",
            "loss_coupling": "first-fragment erasure by source sequence; random loss keyed by "
                             "seed, source sequence and fragment index, independently of admission",
            "note": "one atomic admission and serialization per group, charged the sum of exact "
                    "datagram bytes; admitted bytes remain in flight until arrival including erasures; "
                    "legacy frame-slot and frame-age settings are inactive; the receiver applies age",
        }

    def snapshot(self) -> dict:
        return self.report()
