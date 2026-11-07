"""Controlled frame-credit sweep over ONE authenticated, owned Colab kernel.

This is deliberately separate from the ordinary single-credit and bounded
pipeline streams. No provider allocation, kernel creation, or remote worker
restart happens here. Each phase drains the previous phase's outstanding
requests before changing the credit limit.
"""
from __future__ import annotations

import queue
from statistics import median
import time

from .bridge_colab import offer_update
from .bridge_telemetry import BridgeTransferMeter
from .lifecycle_io import LifecycleError

CREDITS = (1, 2, 3, 4, 1)
MIN_PHASE_SECONDS = 10
MAX_PHASE_SECONDS = 40
WARMUP_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 8
STALE_SECONDS = 30


def phase_summary(*, phase: int, credits: int, seconds: float, fresh: int,
                  responses: int, rtts: list[float], ack_samples: list[float],
                  remote_fps: float | None) -> str:
    """Safe, deterministic summary from post-warmup request observations."""
    measured_seconds = max(.001, seconds - WARMUP_SECONDS)
    frame_rate = fresh / measured_seconds
    rtt = f"{median(rtts):.0f}" if rtts else "none"
    ack = f"{median(ack_samples):.0f}" if ack_samples else "none"
    remote = f"{remote_fps:.1f}" if remote_fps is not None else "unknown"
    return (f"CREDIT SWEEP phase {phase}/{len(CREDITS)} · credits {credits}"
            f" · received {frame_rate:.2f} FPS · RTT p50 {rtt} ms"
            f" · fresh {fresh}/{responses} · input ACK p50 {ack} ms"
            f" · remote render {remote} FPS")


def run_credit_sweep(peer, gpu: str, outgoing: queue.Queue, frames: queue.Queue,
                     stopped, deadline: float, phase_seconds: int = 12) -> None:
    """Run a 1/2/3/4/1 credit sweep with fixed 12Hz request pacing.

    Wait for all outstanding responses at each phase boundary and exclude
    warmup, avoiding attribution of a previous phase's frames to the next.
    Request IDs increase monotonically for the entire kernel session.
    """
    if type(phase_seconds) is not int or not MIN_PHASE_SECONDS <= phase_seconds <= MAX_PHASE_SECONDS:
        raise ValueError("invalid credit sweep phase duration")
    next_id = 0
    input_seq = 0
    last_frame_seq = -1
    last_fresh_at = time.monotonic()
    pending: dict[int, float] = {}
    meter = BridgeTransferMeter()
    completed = 0

    print("[colabcola] Starting single-session credit sweep: 1/2/3/4/1", flush=True)
    try:
        for index, credits in enumerate(CREDITS, 1):
            if stopped.is_set():
                raise LifecycleError("credit sweep interrupted")
            if pending:
                raise LifecycleError("credit sweep phase started with outstanding requests")
            phase_start = time.monotonic()
            phase_end = phase_start + phase_seconds
            next_due = phase_start
            responses = fresh = 0
            rtts = []
            acks = []
            remote_fps = None
            print(f"[colabcola] CREDIT phase {index}/{len(CREDITS)} · credits {credits}", flush=True)
            offer_update(frames, ("status", f"Credit sweep · phase {index}/{len(CREDITS)} · credits {credits}"))

            while not stopped.is_set():
                now = time.monotonic()
                if now >= deadline:
                    raise LifecycleError("credit sweep time budget reached")
                if now >= phase_end and not pending:
                    break

                event = None
                while True:
                    try:
                        event = outgoing.get_nowait()
                    except queue.Empty:
                        break
                if event is not None:
                    input_seq += 1
                    meter.sent(input_seq, time.monotonic())
                    peer.send_input(input_seq, event)

                # The sending window closes at phase_end, then drains.
                now = time.monotonic()
                if now < phase_end and now >= next_due and len(pending) < credits:
                    next_id += 1
                    sent = time.monotonic()
                    peer.request_frame(next_id)
                    pending[next_id] = sent
                    next_due = sent + 1 / 12

                if pending:
                    reply = peer.poll_pipeline_frame(.03)
                    if reply is not None:
                        pull_id, renderer_seq, ack, pixels, size = reply
                        sent = pending.pop(pull_id, None)
                        if sent is None:
                            raise LifecycleError("unmatched credit sweep reply")
                        now = time.monotonic()
                        is_fresh = renderer_seq > last_frame_seq
                        if is_fresh:
                            last_frame_seq = renderer_seq
                            last_fresh_at = now
                            offer_update(frames, ("frame", pixels))
                        ack_ms = meter.received_ack(ack, now)
                        if now - phase_start >= WARMUP_SECONDS and sent >= phase_start + WARMUP_SECONDS:
                            responses += 1
                            fresh += int(is_fresh)
                            rtts.append((now - sent) * 1000)
                            if ack_ms is not None:
                                acks.append(ack_ms)
                            remote_metrics = getattr(peer, "latest_renderer_metrics", None)
                            if (isinstance(remote_metrics, dict)
                                    and type(remote_metrics.get("renderer_fps")) in (int, float)
                                    and 0 <= remote_metrics["renderer_fps"] <= 120):
                                remote_fps = remote_metrics["renderer_fps"]
                else:
                    time.sleep(min(.02, max(0, min(next_due, phase_end) - time.monotonic())))

                now = time.monotonic()
                if pending and now - min(pending.values()) > REQUEST_TIMEOUT_SECONDS:
                    raise LifecycleError("credit sweep request timeout")
                if now - last_fresh_at > STALE_SECONDS:
                    raise LifecycleError("credit sweep renderer became stale")

            if stopped.is_set():
                raise LifecycleError("credit sweep stopped before phase completion")
            report = phase_summary(
                phase=index, credits=credits, seconds=time.monotonic() - phase_start,
                fresh=fresh, responses=responses, rtts=rtts,
                ack_samples=acks, remote_fps=remote_fps)
            print("[colabcola] " + report, flush=True)
            offer_update(frames, ("status", report))
            completed += 1
        print("[colabcola] Credit sweep completed in one owned T4 session", flush=True)
    except Exception as exc:
        print("[colabcola] Credit sweep stopped: " + type(exc).__name__, flush=True)
        raise LifecycleError("bounded credit sweep interrupted") from None
    finally:
        stopped.set()
