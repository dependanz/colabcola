"""Local Tk presentation-side telemetry for the experimental interactive bridge.

The transport counts new frames received. This meter counts how often the GUI
actually decodes a new image and configures the Tk label. It does *not*
measure OS-composited screen presentation or input-to-photon latency.
"""
from __future__ import annotations

from statistics import median
from dataclasses import dataclass
import math


def console_safe_viewer_summary(summary: str) -> str:
    """Render diagnostic-only Unicode text safely in legacy Windows consoles.

    The Tk update callback must never be interrupted by cp1252 encoding.
    The original report remains Unicode for tests, logs, and GUI consumers.
    """
    if not isinstance(summary, str):
        raise TypeError("viewer summary must be text")
    return (summary.replace("→", "->").replace("·", "|")
            .encode("ascii", errors="backslashreplace").decode("ascii"))


def pooled_percentiles(values) -> dict[str, float | None]:
    """Linearly interpolated p50/p95/p99 over *individual* observations.

    A reporting-window median is never itself treated as one observation.
    Empty collections have null quantiles, and nonfinite/bool inputs fail.
    """
    observations = list(values)
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0
           for x in observations):
        raise ValueError("invalid latency or spacing samples")
    if not observations:
        return {"p50": None, "p95": None, "p99": None}
    ordered = sorted(observations)
    result = {}
    for label, fraction in (("p50", .50), ("p95", .95), ("p99", .99)):
        position = (len(ordered) - 1) * fraction
        left = int(position)
        weight = position - left
        value = ordered[left] + weight * (
            ordered[min(left + 1, len(ordered) - 1)] - ordered[left])
        result[label] = round(value, 2)
    return result


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
        self.session_started_at = started_at
        self.count = 0
        self.session_count = 0
        self._session_configured_ns: list[int] = []
        self._session_capture_to_gui_ms: list[float] = []
        self._session_decode_to_gui_ms: list[float] = []
        self._session_input_to_gui_ms: list[float] = []
        self._times_ms: list[float] = []
        self._capture_to_gui_ms: list[float] = []
        self._decoded_to_gui_ms: list[float] = []
        self._input_to_gui_ms: list[float] = []
        self._last_reported_input_ns = 0

    def presented(self, cost_ms: float, configured_ns: int | None = None) -> None:
        if not 0 <= cost_ms < 60000:
            raise ValueError("invalid GUI frame processing duration")
        if configured_ns is not None and (
                type(configured_ns) is not int or configured_ns <= 0 or
                (self._session_configured_ns and
                 configured_ns <= self._session_configured_ns[-1])):
            raise ValueError("invalid GUI configuration timestamp")
        self.count += 1
        self.session_count += 1
        self._times_ms.append(cost_ms)
        if configured_ns is not None:
            self._session_configured_ns.append(configured_ns)

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
        input_ns = timing.input_callback_ns
        if input_ns is not None and (
                type(input_ns) is not int
                or not 0 < input_ns <= timing.captured_ns
                or now_ns - input_ns > 180 * 1_000_000_000):
            raise ValueError("invalid local input callback timestamp")
        # Validate everything before recording any measurements.
        capture_to_gui = (now_ns - timing.captured_ns) / 1e6
        decode_to_gui = (now_ns - timing.decoded_ns) / 1e6
        self._capture_to_gui_ms.append(capture_to_gui)
        self._decoded_to_gui_ms.append(decode_to_gui)
        self._session_capture_to_gui_ms.append(capture_to_gui)
        self._session_decode_to_gui_ms.append(decode_to_gui)
        if input_ns is not None and input_ns > self._last_reported_input_ns:
            input_to_gui = (now_ns - input_ns) / 1e6
            self._input_to_gui_ms.append(input_to_gui)
            self._session_input_to_gui_ms.append(input_to_gui)
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


    def session_summary(self, now: float) -> dict:
        """Session-pooled Tk metrics, not OS scanout or input-to-photon."""
        if not isinstance(now, (int, float)) or not math.isfinite(now):
            raise ValueError("invalid session end time")
        elapsed = now - self.session_started_at
        if not 0 < elapsed <= 7200:
            raise ValueError("invalid session elapsed time")
        gaps = [(right - left) / 1e6 for left, right in zip(
            self._session_configured_ns, self._session_configured_ns[1:])]
        return {
            "schema": 1, "mode": "tk-configured-session",
            "gui_configured_frames": self.session_count,
            "elapsed_seconds": round(elapsed, 3),
            "gui_configured_fps": round(self.session_count / elapsed, 3),
            "gui_frame_spacing_observations": len(gaps),
            **{f"gui_frame_spacing_{key}_ms": value
               for key, value in pooled_percentiles(gaps).items()},
            "host_receipt_to_tk_observations": len(self._session_capture_to_gui_ms),
            **{f"host_receipt_to_tk_{key}_ms": value
               for key, value in pooled_percentiles(
                   self._session_capture_to_gui_ms).items()},
            "decode_to_tk_observations": len(self._session_decode_to_gui_ms),
            **{f"decode_to_tk_{key}_ms": value
               for key, value in pooled_percentiles(
                   self._session_decode_to_gui_ms).items()},
            "input_callback_to_tk_observations": len(
                self._session_input_to_gui_ms),
            **{f"input_callback_to_tk_{key}_ms": value
               for key, value in pooled_percentiles(
                   self._session_input_to_gui_ms).items()},
            "timing_scope": (
                "same-host perf_counter_ns callback to Tk label configuration; "
                "not physical display scanout or input-to-photon"),
        }
