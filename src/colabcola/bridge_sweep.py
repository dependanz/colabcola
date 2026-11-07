"""Deterministic in-session rate sweep for the experimental Jupyter Comm bridge.

This is a fixed 2×2 comparison plus repeated baseline, not an adaptive GPU
allocator. One owned runtime and one authenticated kernel connection are used.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median


SWEEP_PHASES = ((6, 6), (12, 6), (6, 12), (12, 12), (6, 6))
MIN_PHASE_SECONDS = 10
MAX_PHASE_SECONDS = 40
WARMUP_SECONDS = 2


@dataclass
class PhaseMeasurement:
    renderer_fps: int
    pull_hz: int
    phase: int
    phase_seconds: int
    warmup_seconds: int = WARMUP_SECONDS

    def __post_init__(self):
        self.start = None
        self.pulls = 0
        self.fresh = 0
        self.bytes_received = 0
        self.rtts: list[float] = []
        self.renderer_confirmed = 0
        self.render_metrics_seq = -1
        self.renderer_fps_seen = None

    def begin(self, now: float) -> None:
        self.start = now

    def collect(self, *, now: float, rtt_ms: float, size: int, fresh: bool,
                metrics: dict | None, required_rate_seq: int) -> None:
        if self.start is None or now - self.start < self.warmup_seconds:
            return
        self.pulls += 1
        self.rtts.append(rtt_ms)
        if fresh:
            self.fresh += 1
            self.bytes_received += size
        if (isinstance(metrics, dict)
                and metrics.get("target_fps") == self.renderer_fps
                and metrics.get("rate_seq", -1) >= required_rate_seq):
            self.renderer_confirmed += 1
            self.renderer_fps_seen = metrics["renderer_fps"]
            self.render_metrics_seq = max(self.render_metrics_seq, metrics["seq"])

    def summary(self, now: float) -> str:
        if self.start is None:
            raise RuntimeError("sweep phase never started")
        duration = max(0.001, now - self.start - self.warmup_seconds)
        fps = self.fresh / duration
        rtt = f"{median(self.rtts):.0f}" if self.rtts else "none"
        confirmed = f"{self.renderer_confirmed}/{self.pulls}"
        renderer = (f"{self.renderer_fps_seen:.1f}" if self.renderer_fps_seen is not None
                    else "unconfirmed")
        return (f"phase {self.phase}/5 · render {self.renderer_fps} · pull {self.pull_hz} Hz "
                f"· receive {fps:.2f} FPS · RTT p50 {rtt} ms · "
                f"fresh {self.fresh}/{self.pulls} · rate-confirmed {confirmed} "
                f"· actual-render {renderer} FPS")


def sweep_phases(phase_seconds: int):
    if type(phase_seconds) is not int or not MIN_PHASE_SECONDS <= phase_seconds <= MAX_PHASE_SECONDS:
        raise ValueError("invalid benchmark phase duration")
    return [
        PhaseMeasurement(renderer, pull, i + 1, phase_seconds)
        for i, (renderer, pull) in enumerate(SWEEP_PHASES)
    ]
