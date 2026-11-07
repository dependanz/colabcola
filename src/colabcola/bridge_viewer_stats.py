"""Local Tk presentation-side telemetry for the experimental interactive bridge.

The transport counts new frames received. This meter counts how often the GUI
actually decodes a new image and configures the Tk label. It does *not*
measure OS-composited screen presentation or input-to-photon latency.
"""
from __future__ import annotations

from statistics import median
from dataclasses import dataclass


@dataclass(frozen=True)
class LocalFrameTiming:
    """Only local-process performance timestamps; never put on remote wire."""

    captured_ns: int
    decoded_ns: int
    input_callback_ns: int | None = None



class ViewerPresentationMeter:
    """Window-local GUI frame counter and decode/configuration timing."""

    def __init__(self, *, started_at: float):
        self.started_at = started_at
        self.count = 0
        self._times_ms: list[float] = []
        self._capture_to_gui_ms: list[float] = []
        self._decoded_to_gui_ms: list[float] = []
        self._input_to_gui_ms: list[float] = []
        self._last_reported_input_ns = 0

    def presented(self, cost_ms: float) -> None:
        if not 0 <= cost_ms < 60000:
            raise ValueError("invalid GUI frame processing duration")
        self.count += 1
        self._times_ms.append(cost_ms)

    def configured_timed_frame(self, timing: LocalFrameTiming, now_ns: int) -> None:
        """Observe one actual Tk label configuration, including queue wait.

        An input is counted once, on the first configured frame showing its
        applied state. Coalesced inputs are intentionally not counted.
        """
        if (not isinstance(timing, LocalFrameTiming)
                or type(now_ns) is not int
                or type(timing.captured_ns) is not int
                or type(timing.decoded_ns) is not int
                or not 0 < timing.captured_ns <= timing.decoded_ns <= now_ns
                or now_ns - timing.captured_ns > 180 * 1_000_000_000):
            raise ValueError("invalid local frame timing")
        self._capture_to_gui_ms.append((now_ns - timing.captured_ns) / 1e6)
        self._decoded_to_gui_ms.append((now_ns - timing.decoded_ns) / 1e6)
        input_ns = timing.input_callback_ns
        if input_ns is not None:
            if (type(input_ns) is not int or not 0 < input_ns <= now_ns
                    or now_ns - input_ns > 180 * 1_000_000_000):
                raise ValueError("invalid local input callback timestamp")
            if input_ns > self._last_reported_input_ns:
                self._input_to_gui_ms.append((now_ns - input_ns) / 1e6)
                self._last_reported_input_ns = input_ns

    def report(self, now: float, *, every_seconds: float = 2.0) -> str | None:
        elapsed = now - self.started_at
        if not 0 < every_seconds <= 30:
            raise ValueError("invalid GUI reporting interval")
        if elapsed < every_seconds:
            return None
        fps = self.count / max(elapsed, .001)
        decode = f"{median(self._times_ms):.1f}" if self._times_ms else "n/a"
        message = f"GUI configured {fps:.1f} FPS · Tk image+label p50 {decode} ms"
        if self._capture_to_gui_ms:
            message += f" · capture→Tk p50 {median(self._capture_to_gui_ms):.1f} ms"
            message += f" · decode→Tk p50 {median(self._decoded_to_gui_ms):.1f} ms"
        if self._input_to_gui_ms:
            message += f" · input callback→Tk p50 {median(self._input_to_gui_ms):.1f} ms"
            message += f" ({len(self._input_to_gui_ms)} changed frames)"
        self.started_at = now
        self.count = 0
        self._times_ms.clear()
        self._capture_to_gui_ms.clear()
        self._decoded_to_gui_ms.clear()
        self._input_to_gui_ms.clear()
        return message
