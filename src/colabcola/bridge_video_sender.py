"""v31: bounded local access-unit queuing and causal debt-based pacing.

This sender knows its own encoded units, capture clock and configured rate.
It receives no virtual-link object, admission result or future capacity.
Dispatches are whole access units: each attempt incurs its full framed-bit
debt, even if a downstream link subsequently rejects that attempt.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math

from .bridge_video_impairment import MAX_FRAMES
from .bridge_video_stream import PACKET_HEADER, VideoAccessUnit, is_keyframe, packet_bytes


_EVENT_TIE_MS = 1e-9


@dataclass(frozen=True)
class SenderQueueConfig:
    pacing: bool = True
    max_pending: int = 3
    max_age_ms: float = 100.0
    max_repair_requests: int = 8

    def __post_init__(self) -> None:
        if type(self.pacing) is not bool:
            raise ValueError("pacing must be a boolean")
        if type(self.max_pending) is not int or not 1 <= self.max_pending <= MAX_FRAMES:
            raise ValueError("max_pending must be an integer in 1..120")
        if (type(self.max_age_ms) not in (int, float)
                or not math.isfinite(self.max_age_ms) or not 0 <= self.max_age_ms <= 2000):
            raise ValueError("max_age_ms must be finite and in 0..2000")
        if (type(self.max_repair_requests) is not int
                or not 0 <= self.max_repair_requests <= 8):
            raise ValueError("max_repair_requests must be an integer in 0..8")


@dataclass(frozen=True)
class Dispatch:
    packet: VideoAccessUnit
    dispatch_ms: float
    wire_size: int
    pacing_kbps: float
    queue_delay_ms: float


@dataclass(frozen=True)
class _QueuedUnit:
    packet: VideoAccessUnit
    source_ms: float
    wire_size: int
    verified_idr: bool


class LocalSenderQueue:
    """FIFO unsent units, bounded expiry, and one outstanding local IDR request.

    Call advance(T) before each capture, then set_target/repair/enqueue at T,
    followed by advance(T) for an immediately ready new unit. advance drains
    local events through its endpoint. Expiry wins dispatch ties within
    1e-9 ms, at the actual expiry time; a chosen event beyond the advance
    endpoint waits. A request created exactly at capture T can affect only
    a later frame.

    Pacing allows one complete access-unit burst when debt is zero. The
    debt then drains at target payload kbps plus nominal per-frame framing.
    It survives target changes, local drops and IDRs. Idle time earns no
    extra credit. This is not byte-level serialization or a network oracle.
    """

    def __init__(self, config: SenderQueueConfig, *, fps: int,
                 initial_target_kbps: int):
        if not isinstance(config, SenderQueueConfig):
            raise ValueError("invalid sender configuration")
        if type(fps) is not int or not 1 <= fps <= 60:
            raise ValueError("invalid sender source rate")
        self._validate_target(initial_target_kbps)
        self.config = config
        self.fps = fps
        self.target_kbps = initial_target_kbps
        self.clock_ms = 0.0
        self.debt_bits = 0.0
        self.frames_observed = 0
        self.need_idr = False
        self._queue: deque[_QueuedUnit] = deque()
        self._pending_wire_bytes = 0
        self._offered_wire_bytes = 0
        self._dispatched_wire_bytes = 0
        self._dropped_wire_bytes = 0
        self._peak_pending_frames = 0
        self._peak_pending_bytes = 0
        self._max_queue_delay_ms = 0.0
        self._dispatches: list[Dispatch] = []
        self.drops: list[dict] = []
        self.requests: list[dict] = []
        self.applied: list[dict] = []
        self.cancelled: list[dict] = []
        self._pending_repair: dict | None = None
        self.rate_changes: list[dict] = []

    @staticmethod
    def _validate_target(target_kbps: int) -> None:
        if type(target_kbps) is not int or not 20 <= target_kbps <= 20_000:
            raise ValueError("invalid sender target bitrate")

    def _validate_time(self, now_ms: float) -> None:
        if (type(now_ms) not in (int, float) or not math.isfinite(now_ms)
                or now_ms < self.clock_ms):
            raise ValueError("sender time must be finite and monotonic")

    def _require_clock(self, at_ms: float) -> None:
        self._validate_time(at_ms)
        if at_ms != self.clock_ms:
            raise ValueError("advance sender to this time before applying a source action")

    @property
    def pacing_kbps(self) -> float:
        return self.target_kbps + PACKET_HEADER.size * 8 * self.fps / 1000.0

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    def _after_queue_mutation(self, cause: str) -> None:
        """Extension point after an atomic queue change at the settled clock.

        The v31 sender keeps its target-based rate unchanged. A separate
        service policy can update its rate here without making it depend
        on arbitrary advance() observation endpoints.
        """

    def _event_times(self) -> tuple[float, float]:
        ready_ms = (self.clock_ms + self.debt_bits / self.pacing_kbps
                    if self.config.pacing else self.clock_ms)
        expiry_ms = (self._queue[0].source_ms + self.config.max_age_ms
                     if self.config.max_age_ms else math.inf)
        return ready_ms, expiry_ms

    def _next_event(self) -> tuple[float, bool]:
        ready_ms, expiry_ms = self._event_times()
        expires = expiry_ms <= ready_ms + _EVENT_TIE_MS
        # Fractional frame periods can put a mathematically tied readiness
        # a few ulps before expiry. Choose expiry's actual timestamp, so an
        # advance ending at the earlier rounded readiness cannot expire it.
        at_ms = expiry_ms if expires else ready_ms
        return max(self.clock_ms, at_ms), expires

    @property
    def next_event_ms(self) -> float | None:
        if not self._queue:
            return None
        return self._next_event()[0]

    def _settle_debt(self, at_ms: float) -> None:
        # All intervals use the previously configured rate. A switch never
        # recalculates the cost of bytes already dispatched.
        self.debt_bits = max(0.0, self.debt_bits - (at_ms - self.clock_ms) * self.pacing_kbps)
        self.clock_ms = at_ms

    def set_target(self, target_kbps: int, at_ms: float) -> None:
        self._validate_target(target_kbps)
        self._require_clock(at_ms)
        if target_kbps != self.target_kbps:
            self.rate_changes.append({
                "at_ms": at_ms, "previous_target_kbps": self.target_kbps,
                "target_kbps": target_kbps, "remaining_debt_bits": self.debt_bits,
            })
            self.target_kbps = target_kbps

    def _pop(self) -> _QueuedUnit:
        unit = self._queue.popleft()
        self._pending_wire_bytes -= unit.wire_size
        return unit

    def _drop(self, unit: _QueuedUnit, reason: str) -> None:
        delay = self.clock_ms - unit.source_ms
        self._dropped_wire_bytes += unit.wire_size
        self._max_queue_delay_ms = max(self._max_queue_delay_ms, delay)
        self.drops.append({
            "source_seq": unit.packet.seq, "source_ms": unit.source_ms,
            "at_ms": self.clock_ms, "reason": reason, "wire_size": unit.wire_size,
            "queue_delay_ms": delay, "verified_idr": unit.verified_idr,
        })

    def _cancel_repair(self, source_idr_seq: int, reason: str) -> None:
        if self._pending_repair is None:
            return
        self.cancelled.append({
            "request_id": self._pending_repair["request_id"],
            "at_ms": self.clock_ms, "source_idr_seq": source_idr_seq, "reason": reason,
        })
        self._pending_repair = None

    def _discard_chain(self, reason: str) -> int:
        first = self._pop()
        self._drop(first, reason)
        # Every following predicted unit may depend on the discarded unit.
        # A verified IDR starts an independent chain which can be retained.
        while self._queue and not self._queue[0].verified_idr:
            self._drop(self._pop(), "sender_dependency")
        self.need_idr = not self._queue
        if self._queue:
            self._cancel_repair(self._queue[0].packet.seq, "queued_idr_available")
        return first.packet.seq

    def _request_repair(self, trigger_seq: int, reason: str) -> None:
        if (not self.need_idr or self._pending_repair is not None
                or len(self.requests) >= self.config.max_repair_requests):
            return
        request = {
            "request_id": len(self.requests), "created_at_ms": self.clock_ms,
            "trigger_source_seq": trigger_seq, "trigger": reason,
        }
        self.requests.append(request)
        self._pending_repair = request

    def enqueue(self, packet: VideoAccessUnit, at_ms: float) -> None:
        self._require_clock(at_ms)
        if (not isinstance(packet, VideoAccessUnit)
                or packet.seq != self.frames_observed or packet.seq >= MAX_FRAMES
                or at_ms < packet.seq * 1000.0 / self.fps
                or not math.isclose(at_ms, packet.seq * 1000.0 / self.fps,
                                    rel_tol=0, abs_tol=1e-6)):
            raise ValueError("sender captures must have consecutive IDs and their original source time")
        wire_size = len(packet_bytes(packet))
        verified_idr = bool(packet.keyframe and is_keyframe(packet.data))
        unit = _QueuedUnit(packet, packet.seq * 1000.0 / self.fps, wire_size, verified_idr)
        self.frames_observed += 1
        self._offered_wire_bytes += wire_size

        trigger_seq = packet.seq
        trigger = "sender_dependency"
        if len(self._queue) >= self.config.max_pending:
            trigger = "sender_queue_full"
            trigger_seq = self._discard_chain(trigger)
        if self.need_idr and not verified_idr:
            self._drop(unit, "sender_dependency")
            self._request_repair(trigger_seq, trigger)
            self._after_queue_mutation("enqueue")
            return
        if verified_idr:
            self.need_idr = False
            self._cancel_repair(packet.seq, "new_idr_available")
        self._queue.append(unit)
        self._pending_wire_bytes += wire_size
        self._peak_pending_frames = max(self._peak_pending_frames, len(self._queue))
        self._peak_pending_bytes = max(self._peak_pending_bytes, self._pending_wire_bytes)
        self._after_queue_mutation("enqueue")

    def take_repair_before(self, now_ms: float, source_seq: int) -> dict | None:
        """Consume a local request only before a strictly later source frame."""
        self._require_clock(now_ms)
        if (type(source_seq) is not int or source_seq != self.frames_observed
                or not 0 <= source_seq < MAX_FRAMES
                or now_ms < source_seq * 1000.0 / self.fps
                or not math.isclose(now_ms, source_seq * 1000.0 / self.fps,
                                    rel_tol=0, abs_tol=1e-6)):
            raise ValueError("invalid local-repair source frame")
        request = self._pending_repair
        if (request is None or not self.need_idr
                or request["created_at_ms"] >= now_ms - 1e-9):
            return None
        self._pending_repair = None
        self.applied.append({
            "request_id": request["request_id"], "source_idr_seq": source_seq,
            "source_ms": now_ms, "requested_at_ms": request["created_at_ms"],
        })
        return dict(request)

    def advance(self, now_ms: float) -> list[Dispatch]:
        """Process only local events through now and return irrevocable attempts."""
        self._validate_time(now_ms)
        output = []
        while self._queue:
            at_ms, expires = self._next_event()
            if at_ms > now_ms:
                break
            self._settle_debt(at_ms)
            if expires:
                first_seq = self._discard_chain("sender_age")
                self._request_repair(first_seq, "sender_age")
                self._after_queue_mutation("expiry")
                continue
            unit = self._pop()
            delay = at_ms - unit.source_ms
            dispatch = Dispatch(unit.packet, at_ms, unit.wire_size, self.pacing_kbps, delay)
            output.append(dispatch)
            self._dispatches.append(dispatch)
            self._dispatched_wire_bytes += unit.wire_size
            self._max_queue_delay_ms = max(self._max_queue_delay_ms, delay)
            # Reaching ready_ms settles the old debt. Floating point
            # subtraction can leave a tiny residue; no old bytes are refunded.
            self.debt_bits = unit.wire_size * 8.0 if self.config.pacing else 0.0
            self._after_queue_mutation("dispatch")
        self._settle_debt(now_ms)
        return output

    def report(self) -> dict:
        if (self._offered_wire_bytes != self._dispatched_wire_bytes
                + self._dropped_wire_bytes + self._pending_wire_bytes
                or self.frames_observed != len(self._dispatches) + len(self.drops) + len(self._queue)):
            raise RuntimeError("sender queue accounting mismatch")
        if len(self.requests) != len(self.applied) + len(self.cancelled) + int(self._pending_repair is not None):
            raise RuntimeError("sender repair-request accounting mismatch")

        def rounded_records(records: list[dict]) -> list[dict]:
            return [{key: round(value, 6) if type(value) is float else value
                     for key, value in record.items()} for record in records]

        return {
            "config": asdict(self.config), "frames_observed": self.frames_observed,
            "pending_frames": len(self._queue), "need_idr": self.need_idr,
            "peak_pending_frames": self._peak_pending_frames,
            "peak_pending_bytes": self._peak_pending_bytes,
            "max_queue_delay_ms": round(self._max_queue_delay_ms, 6),
            "attempted_dispatches": len(self._dispatches),
            "offered_wire_bytes": self._offered_wire_bytes,
            "dispatched_wire_bytes": self._dispatched_wire_bytes,
            "dropped_wire_bytes": self._dropped_wire_bytes,
            "pending_wire_bytes": self._pending_wire_bytes,
            "dispatches": [{
                "source_seq": x.packet.seq, "dispatch_ms": round(x.dispatch_ms, 6),
                "wire_size": x.wire_size, "pacing_kbps": x.pacing_kbps,
                "queue_delay_ms": round(x.queue_delay_ms, 6),
            } for x in self._dispatches],
            "drops": rounded_records(self.drops),
            "requests": rounded_records(self.requests),
            "applied": rounded_records(self.applied),
            "cancelled": rounded_records(self.cancelled),
            "requests_pending": int(self._pending_repair is not None),
            "requests_not_applied_before_source_end": int(self._pending_repair is not None),
            "repair_budget_exhausted": (self.need_idr and self._pending_repair is None
                                        and len(self.requests) >= self.config.max_repair_requests),
            "rate_changes": rounded_records(self.rate_changes),
            "clock_ms": round(self.clock_ms, 6),
            "remaining_debt_bits": round(self.debt_bits, 6),
            "event_tie_tolerance_ms": _EVENT_TIE_MS,
            "note": "whole access-unit dispatch attempts incur framed-bit debt regardless of downstream "
                    "admission; expiry and dependency suppression affect only unsent units; "
                    "all age is measured from original capture time; idle credit is not accumulated; "
                    "expiry wins readiness ties within 1e-9 ms at its actual timestamp, never early",
        }
