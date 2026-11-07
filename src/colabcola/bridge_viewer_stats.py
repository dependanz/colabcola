"""Local Tk presentation-side telemetry for the experimental interactive bridge.

The transport counts new frames received. This meter counts how often the GUI
actually decodes a new image and configures the Tk label. It does *not*
measure OS-composited screen presentation or input-to-photon latency.
"""
from __future__ import annotations

from statistics import median


class ViewerPresentationMeter:
    """Window-local GUI frame counter and decode/configuration timing."""

    def __init__(self, *, started_at: float):
        self.started_at = started_at
        self.count = 0
        self._times_ms: list[float] = []

    def presented(self, cost_ms: float) -> None:
        if not 0 <= cost_ms < 60000:
            raise ValueError("invalid GUI frame processing duration")
        self.count += 1
        self._times_ms.append(cost_ms)

    def report(self, now: float, *, every_seconds: float = 2.0) -> str | None:
        elapsed = now - self.started_at
        if not 0 < every_seconds <= 30:
            raise ValueError("invalid GUI reporting interval")
        if elapsed < every_seconds:
            return None
        fps = self.count / max(elapsed, .001)
        decode = f"{median(self._times_ms):.1f}" if self._times_ms else "n/a"
        message = f"GUI configured {fps:.1f} FPS · Tk image+label p50 {decode} ms"
        self.started_at = now
        self.count = 0
        self._times_ms.clear()
        return message
