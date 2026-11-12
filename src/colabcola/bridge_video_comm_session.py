"""v33e: opt-in H.264 over the OWNED Jupyter Comm, single-credit only.

The remote CCV1 producer's capture timestamp has a DIFFERENT clock domain.
Only host-side receive/decode durations are measured here. No new socket,
Jupyter session, GPU allocation, or provider operations are initiated here.
"""
from __future__ import annotations

import queue
from statistics import median
import time

from .bridge_colab import offer_update
from .bridge_telemetry import BridgeTransferMeter
from .bridge_video_comm_wire import KernelVideoCommReceiver
from .lifecycle_io import LifecycleError
from .bridge_video_stream import PACKET_HEADER

WIDTH, HEIGHT = 640, 360
MAX_NO_FRESH_SECONDS = 30


def application_video_rates(*, h264_bytes: int, ccv1_bytes: int,
                            base64_chars: int, elapsed: float) -> str:
    """Report only application bytes; Jupyter JSON/WebSocket overhead is excluded."""
    if (not all(type(value) is int and value >= 0
                for value in (h264_bytes, ccv1_bytes, base64_chars))
            or not isinstance(elapsed, (int, float)) or elapsed <= 0):
        raise ValueError("invalid application video accounting")
    return (f"H.264 {h264_bytes / elapsed / 1024:.1f} KiB/s"
            f" · CCV1 {ccv1_bytes / elapsed / 1024:.1f} KiB/s"
            f" · Base64 {base64_chars / elapsed / 1024:.1f} KiB/s")


def local_elapsed_us(start_ns: int, end_ns: int) -> float:
    """Elapsed host-only microseconds; never mix remote and host clock readings."""
    if (type(start_ns) is not int or type(end_ns) is not int
            or start_ns < 0 or end_ns < start_ns):
        raise ValueError("invalid local timing samples")
    return (end_ns - start_ns) / 1000


def make_host_decoder():
    """Fail before GPU allocation when the local optional decoder is absent."""
    try:
        import av
        import numpy as np  # noqa: F401 - required by frame.to_ndarray
        decoder = av.CodecContext.create("h264", "r")
        decoder.thread_count = 1
        decoder.open()
    except Exception:
        raise LifecycleError("optional local PyAV H.264 decoder unavailable") from None
    return av, decoder


def stream_comm_h264(peer, gpu: str, outgoing: queue.Queue,
                     frames: queue.Queue, stopped, deadline: float,
                     poll_hz: int) -> None:
    """Receive ordered CCV1 access units on the existing kernel Comm.

    One request at a time, never replayed after an ambiguous error. A sequence
    gap suppresses dependent frames until IDR; the viewer holds its last image.
    Decoded RGB is published through the existing bounded Tk frame queue.
    """
    if type(poll_hz) is not int or not 1 <= poll_hz <= 12:
        raise ValueError("H.264 video poll rate outside 1..12 Hz")
    if getattr(peer, "video_codec", None) != "h264":
        raise LifecycleError("H.264 stream requires an opted-in kernel bridge")
    av, decoder = make_host_decoder()
    receiver = KernelVideoCommReceiver(peer)
    meter = BridgeTransferMeter()
    prefix = b"P6\n640 360\n255\n"
    last_fresh = time.monotonic()
    last_report = last_fresh
    next_pull = 1
    input_seq = 0
    replies = decoded_count = h264_bytes = ccv1_bytes = base64_chars = 0
    rtts_ms = []
    decode_us = []
    print("[colabcola] Opt-in H.264 kernel Comm connected (single credit)", flush=True)
    offer_update(frames, ("status", f"Kernel Comm H.264 · {gpu}"))
    try:
        while not stopped.is_set() and time.monotonic() < deadline:
            started = time.monotonic()
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

            sent_at = time.monotonic()
            receiver.request_video(next_pull)
            unit, ack = receiver.next_video(
                min(8, max(.1, deadline - time.monotonic())))
            now = time.monotonic()
            rtts_ms.append((now - sent_at) * 1000)
            replies += 1
            next_pull += 1
            meter.received_ack(ack, now)
            # Count accepted replies even when dependency gating suppresses decode.
            sizes = getattr(receiver, "last_reply_sizes", None)
            if sizes is None and unit is not None:
                # Compatibility with simple fake peers used by offline tests.
                framed = PACKET_HEADER.size + len(unit.data)
                sizes = (len(unit.data), framed, 4 * ((framed + 2) // 3))
            if sizes is not None:
                raw, framed, encoded = sizes
                h264_bytes += raw
                ccv1_bytes += framed
                base64_chars += encoded
            if unit is not None:
                decode_started_ns = time.perf_counter_ns()
                try:
                    packet = av.Packet(unit.data)
                    packet.pts = unit.seq
                    images = decoder.decode(packet)
                    if len(images) > 1:
                        raise LifecycleError("H.264 decoder emitted multiple frames per unit")
                    for image in images:
                        if image.width != WIDTH or image.height != HEIGHT:
                            raise LifecycleError("H.264 decoded dimensions invalid")
                        pixels = image.to_ndarray(format="rgb24").tobytes()
                        if len(pixels) != WIDTH * HEIGHT * 3:
                            raise LifecycleError("H.264 decoded RGB size invalid")
                        offer_update(frames, ("frame", prefix + pixels))
                        decoded_count += 1
                        last_fresh = time.monotonic()
                    decode_us.append(local_elapsed_us(
                        decode_started_ns, time.perf_counter_ns()))
                except LifecycleError:
                    raise
                except Exception:
                    raise LifecycleError("kernel H.264 decoding failed") from None
            if time.monotonic() - last_fresh > MAX_NO_FRESH_SECONDS:
                raise LifecycleError("kernel H.264 published no fresh decoded image")
            now = time.monotonic()
            if now - last_report >= 2:
                elapsed = max(.001, now - last_report)
                summary = (f"{gpu} · H.264 Comm {decoded_count / elapsed:.1f} decoded FPS"
                           f" · {application_video_rates(h264_bytes=h264_bytes, ccv1_bytes=ccv1_bytes, base64_chars=base64_chars, elapsed=elapsed)}"
                           f" · replies {replies}"
                           f" · skipped dependent {receiver.suppressed_dependent_units}")
                if rtts_ms:
                    summary += f" · video pull RTT p50 {median(rtts_ms):.0f} ms"
                if decode_us:
                    summary += f" · local decode+RGB p50 {median(decode_us):.1f} µs"
                ack_stats = meter.flush()
                if ack_stats["ack"] is not None:
                    summary += f" · input ACK observed p50 {ack_stats['ack']} ms"
                print("[colabcola] " + summary, flush=True)
                offer_update(frames, ("status", summary))
                decoded_count = replies = h264_bytes = ccv1_bytes = base64_chars = 0
                rtts_ms.clear()
                decode_us.clear()
                last_report = now
            sleep = max(0, 1 / poll_hz - (time.monotonic() - started))
            if sleep:
                time.sleep(min(sleep, max(0, deadline - time.monotonic())))
        if time.monotonic() >= deadline:
            print("[colabcola] H.264 Comm session time budget reached", flush=True)
    except Exception as exc:
        print("[colabcola] H.264 Comm interrupted: " + type(exc).__name__, flush=True)
        raise LifecycleError("kernel H.264 Comm transport interrupted") from None
    finally:
        stopped.set()
