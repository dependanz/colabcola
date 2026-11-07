"""Deterministic in-session rate sweep for the experimental Jupyter Comm bridge.

This is a fixed 2×2 comparison plus repeated baseline, not an adaptive GPU
allocator. One owned runtime and one authenticated kernel connection are used.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
import queue
import time

from .bridge_colab import offer_update
from .bridge_telemetry import BridgeTransferMeter
from .lifecycle_io import LifecycleError


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


def run_rate_sweep(peer, gpu: str, outgoing: queue.Queue, frames: queue.Queue,
                   stopped, deadline: float, phase_seconds: int = 12) -> None:
    """Compare four rate combinations + repeat baseline on one owned kernel.

    One frame request remains in flight at a time. A control ACK means only
    the kernel accepted the change; renderer-side rate_seq confirms actual
    application. A 2s warmup is excluded from each phase's sample window.
    """
    phases = sweep_phases(phase_seconds)
    last_remote_seq = -1
    input_seq = 0
    control_seq = 0
    previous_renderer_rate = phases[0].renderer_fps
    ack_meter = BridgeTransferMeter()
    finished = 0
    print("[colabcola] Starting in-session rate sweep (single owned T4)", flush=True)
    try:
        for phase in phases:
            if stopped.is_set() or time.monotonic() >= deadline:
                break
            phase.begin(time.monotonic())
            if phase.renderer_fps != previous_renderer_rate:
                control_seq += 1
                peer.set_render_fps(control_seq, phase.renderer_fps)
                previous_renderer_rate = phase.renderer_fps
            heading = (f"phase {phase.phase}/{len(phases)}: "
                       f"renderer {phase.renderer_fps} FPS, pulls {phase.pull_hz} Hz")
            print("[colabcola] " + heading, flush=True)
            offer_update(frames, ("status", heading))
            while not stopped.is_set() and time.monotonic() - phase.start < phase_seconds:
                begin = time.monotonic()
                if begin >= deadline:
                    break
                latest = None
                while True:
                    try:
                        latest = outgoing.get_nowait()
                    except queue.Empty:
                        break
                if latest:
                    input_seq += 1
                    ack_meter.sent(input_seq, time.monotonic())
                    peer.send_input(input_seq, latest)
                requested = time.monotonic()
                peer.request_frame()
                seq, ack, pixels, size = peer.next_frame(
                    min(8.0, max(0.1, deadline - time.monotonic())))
                received = time.monotonic()
                fresh = seq > last_remote_seq
                if fresh:
                    last_remote_seq = seq
                    offer_update(frames, ("frame", pixels))
                ack_meter.received_ack(ack, received)
                phase.collect(
                    now=received, rtt_ms=(received - requested) * 1000,
                    size=size, fresh=fresh,
                    metrics=getattr(peer, "latest_renderer_metrics", None),
                    required_rate_seq=control_seq)
                interval = 1 / phase.pull_hz
                pause = max(0.0, interval - (time.monotonic() - begin))
                if pause:
                    time.sleep(pause)
            now = time.monotonic()
            if stopped.is_set() or now >= deadline and now - phase.start < phase_seconds:
                print("[colabcola] Sweep interrupted; preserving partial measurements", flush=True)
                break
            finished += 1
            report = phase.summary(now)
            print("[colabcola] SWEEP " + report, flush=True)
            offer_update(frames, ("status", report))
        if finished == len(phases):
            print("[colabcola] Sweep completed; ending the owned interactive session", flush=True)
        else:
            print(f"[colabcola] Sweep completed {finished}/{len(phases)} phases", flush=True)
    except Exception as exc:
        print("[colabcola] Sweep stopped: " + type(exc).__name__, flush=True)
        raise LifecycleError("in-session rate sweep interrupted") from None
    finally:
        stopped.set()
