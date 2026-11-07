"""Bounded, opt-in in-flight frame requests over an existing Jupyter Comm.

One authenticated kernel WebSocket and owned T4. Requests are numbered, paced,
and capped to four outstanding replies. This is strictly an experiment: it
must never alter the default single-credit interactive transport.
"""
from __future__ import annotations

from statistics import median
import queue
import time

from .bridge_colab import offer_update
from .bridge_telemetry import BridgeTransferMeter
from .lifecycle_io import LifecycleError


MAX_FRAME_CREDITS = 4
MAX_PULL_WAIT_SECONDS = 8
STALE_FRAME_SECONDS = 30


def stream_comm_pipeline(peer, gpu: str, outgoing: queue.Queue,
                         frames: queue.Queue, stopped, deadline: float,
                         poll_hz: int, in_flight: int) -> None:
    """Pace pull requests independently of replies, subject to strict credits.

    Only frames with increasing renderer sequence numbers reach the GUI.
    Timed-out, untagged, or unsolicited replies fail closed. The per-frame
    request ID is distinct from the remote renderer's frame sequence.
    """
    if (type(in_flight) is not int or not 2 <= in_flight <= MAX_FRAME_CREDITS
            or type(poll_hz) is not int or not 1 <= poll_hz <= 12):
        raise ValueError("invalid bounded pipeline parameters")
    interval = 1 / poll_hz
    next_due = time.monotonic()
    pending: dict[int, float] = {}
    request_id = 0
    input_seq = 0
    last_remote_seq = -1
    last_fresh = next_due
    last_report = next_due
    meter = BridgeTransferMeter()
    received = 0
    fresh = 0
    transferred = 0
    rtts: list[float] = []
    latest_remote_metrics = None
    last_metric_seq = -1

    print(f"[colabcola] Comm pipelining enabled: up to {in_flight} outstanding frames", flush=True)
    offer_update(frames, ("status", f"Kernel Comm · {gpu} · pipeline {in_flight}"))
    try:
        while not stopped.is_set() and time.monotonic() < deadline:
            now = time.monotonic()
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

            # At most one new request per iteration. No catch-up bursts:
            # missed send opportunities are discarded rather than queued.
            now = time.monotonic()
            if now >= next_due and len(pending) < in_flight:
                request_id += 1
                sent_at = time.monotonic()
                peer.request_frame(request_id)
                pending[request_id] = sent_at
                next_due = sent_at + interval

            if pending:
                reply = peer.poll_pipeline_frame(.03)
                if reply is not None:
                    pull_id, remote_seq, ack, ppm, byte_count = reply
                    sent_at = pending.pop(pull_id, None)
                    if sent_at is None:
                        raise LifecycleError("unsolicited or repeated pipeline frame response")
                    now = time.monotonic()
                    rtts.append((now - sent_at) * 1000)
                    received += 1
                    if remote_seq > last_remote_seq:
                        last_remote_seq = remote_seq
                        last_fresh = now
                        fresh += 1
                        transferred += byte_count
                        offer_update(frames, ("frame", ppm))
                    meter.received_ack(ack, now)
                    remote_metrics = getattr(peer, "latest_renderer_metrics", None)
                    if (isinstance(remote_metrics, dict)
                            and remote_metrics["seq"] > last_metric_seq):
                        latest_remote_metrics = remote_metrics
                        last_metric_seq = remote_metrics["seq"]
            else:
                time.sleep(min(.02, max(0.0, next_due - time.monotonic())))

            now = time.monotonic()
            if pending and now - min(pending.values()) > MAX_PULL_WAIT_SECONDS:
                raise LifecycleError("pipeline frame response exceeded request timeout")
            if now - last_fresh > STALE_FRAME_SECONDS:
                raise LifecycleError("pipeline produced no fresh frame within timeout")
            if now - last_report >= 2:
                elapsed = max(.001, now - last_report)
                report = (f"{gpu} · kernel Comm {fresh / elapsed:.1f} FPS · "
                          f"{transferred / elapsed / 1024:.0f} KiB/s")
                if rtts:
                    report += f" · pull RTT p50 {median(rtts):.0f} ms"
                report += f" · fresh {fresh}/{received} · in-flight {len(pending)}/{in_flight}"
                samples = meter.flush()
                if samples["ack"] is not None:
                    report += f" · input ACK observed p50 {samples['ack']} ms"
                if latest_remote_metrics is not None:
                    report += (
                        f" · remote render {latest_remote_metrics['renderer_fps']:.1f} FPS"
                        f" · render+copy {latest_remote_metrics['render_copy_ms']:.0f} ms"
                        f" · compress {latest_remote_metrics['compress_ms']:.0f} ms"
                    )
                print("[colabcola] " + report, flush=True)
                offer_update(frames, ("status", report))
                received = fresh = transferred = 0
                rtts.clear()
                latest_remote_metrics = None
                last_report = now
        if time.monotonic() >= deadline:
            print("[colabcola] Comm pipeline time budget reached", flush=True)
    except Exception as exc:
        print("[colabcola] Comm pipeline stopped: " + type(exc).__name__, flush=True)
        raise LifecycleError("bounded Comm frame pipeline interrupted") from None
    finally:
        stopped.set()
