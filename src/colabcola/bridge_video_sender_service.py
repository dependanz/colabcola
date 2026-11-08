"""v32: local pacing service independent of later encoder target changes.

The service floor and cap are allowances chosen from the configured initial
encoder ceiling, not estimates of physical link capacity. Queue-work boost
can drain local work faster while moving congestion into the downstream
link. This primitive has no link, admission-result or feedback input.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .bridge_video_sender import LocalSenderQueue, SenderQueueConfig
from .bridge_video_stream import PACKET_HEADER


@dataclass(frozen=True)
class SenderServiceConfig:
    """Bounded research policy; milliseconds and a dimensionless cap factor.

    drain_ms is in 1..2000 and max_rate_factor in 1..8. The default 50 ms
    horizon is half the declared 100 ms local age bound, and the service
    cap is twice the initial wire budget. Neither is a capacity guarantee.
    """

    queue_boost: bool = True
    drain_ms: float = 50.0
    max_rate_factor: float = 2.0

    def __post_init__(self) -> None:
        if type(self.queue_boost) is not bool:
            raise ValueError("queue_boost must be a boolean")
        if (type(self.drain_ms) not in (int, float)
                or not math.isfinite(self.drain_ms) or not 1 <= self.drain_ms <= 2000):
            raise ValueError("drain_ms must be finite and in 1..2000")
        if (type(self.max_rate_factor) not in (int, float)
                or not math.isfinite(self.max_rate_factor)
                or not 1 <= self.max_rate_factor <= 8):
            raise ValueError("max_rate_factor must be finite and in 1..8")


class ServiceSenderQueue(LocalSenderQueue):
    """Fixed service floor, optionally raised by debt plus queued framed bits.

    At enqueue, refresh only after local count/dependency handling is done.
    At expiry, refresh after the entire affected chain has been discarded.
    At dispatch, refresh after moving the unit's bits from the queue into
    debt, preserving total work at that instant. Between these mutations,
    service is constant and elapsed debt is paid at that previous rate.

    Target-only changes, report reads and arbitrary advance() endpoints do
    not refresh service. Tail dispatches and expiry still refresh service;
    stopping source captures does not turn existing debt into free credit.
    """

    def __init__(self, config: SenderQueueConfig, *, fps: int,
                 initial_target_kbps: int, service: SenderServiceConfig):
        if not isinstance(service, SenderServiceConfig):
            raise ValueError("invalid sender service configuration")
        super().__init__(config, fps=fps, initial_target_kbps=initial_target_kbps)
        if not config.pacing:
            raise ValueError("sender service requires pacing")
        self.service_config = service
        self._base_kbps = initial_target_kbps + PACKET_HEADER.size * 8 * fps / 1000.0
        self._cap_kbps = self._base_kbps * service.max_rate_factor
        self._service_kbps = self._base_kbps
        self._service_decisions: list[dict] = []

    @property
    def base_kbps(self) -> float:
        return self._base_kbps

    @property
    def cap_kbps(self) -> float:
        return self._cap_kbps

    @property
    def pacing_kbps(self) -> float:
        return self._service_kbps

    def _after_queue_mutation(self, cause: str) -> None:
        # Debt includes already attempted bytes, even with no pending units.
        # The last pending unit is included too: this is a rolling whole-work
        # clearance objective, including its post-dispatch pacing debt.
        work_bits = self.debt_bits + 8 * self._pending_wire_bytes
        requested = (max(self._base_kbps, work_bits / self.service_config.drain_ms)
                     if self.service_config.queue_boost else self._base_kbps)
        self._service_kbps = min(self._cap_kbps, requested)
        self._service_decisions.append({
            "at_ms": self.clock_ms, "cause": cause,
            "debt_bits": self.debt_bits,
            "pending_wire_bytes": self._pending_wire_bytes,
            "work_bits": work_bits,
            "requested_kbps": requested, "pacing_kbps": self._service_kbps,
            "boosted": self._service_kbps > self._base_kbps,
            "cap_limited": requested > self._cap_kbps,
        })

    def report(self) -> dict:
        report = super().report()
        report["service"] = {
            "config": asdict(self.service_config),
            "base_kbps": self._base_kbps, "cap_kbps": self._cap_kbps,
            "current_kbps": self._service_kbps,
            "decisions": [dict(row) for row in self._service_decisions],
            "formula": "work_bits = debt_bits + 8 * pending_wire_bytes; "
                       "service_kbps = min(cap_kbps, max(base_kbps, work_bits / drain_ms)); "
                       "with queue_boost disabled service_kbps = base_kbps",
            "note": "base and cap are fixed from initial target plus nominal framing, not capacity "
                    "estimates; rate refreshes only after complete enqueue, expiry or dispatch "
                    "mutations; no refresh at observation endpoints, reports or target-only changes; "
                    "the rolling work horizon includes the last unit's post-dispatch debt and is not a "
                    "dispatch-wait or end-to-end latency guarantee; tail mutations still refresh service",
        }
        return report
