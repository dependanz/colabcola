"""Live, input-responsive H.264 research scene over an in-process socketpair.

Unlike the earlier precomputed-frame benchmarks, this source renders a new
RGB image after applying the latest mouse/arrow input. A persistent PyAV
encoder and independent decoder thread exchange bounded H.264 packets over
a local socketpair. It has no Colab, GPU, internet socket or public server.

The GUI uses the existing Tk viewer (only with --show). This is not a
remote-media transport or input-to-photon measurement. Codec libraries are
experiment-only dependencies, not requirements of the batch runner.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from statistics import median
import math
import queue
import socket
import threading
import time

from .bridge_colab import offer_update
from .bridge_viewer_stats import LocalFrameTiming
from .bridge_video_stream import VideoAccessUnit, packet_bytes, receive_packet

MAX_LIVE_SECONDS = 180
MAX_LIVE_FRAMES = 60 * MAX_LIVE_SECONDS


@dataclass
class LiveScene:
    x: float = 0.5
    y: float = 0.5
    phase: float = 0.0

    def apply(self, event) -> bool:
        """Validate and apply an event from the local bounded input queue."""
        if not isinstance(event, dict):
            return False
        if event.get("event") == "mouse":
            x, y = event.get("x"), event.get("y")
            if (type(x) not in (int, float) or type(y) not in (int, float)
                    or not math.isfinite(x) or not math.isfinite(y)
                    or not 0 <= x <= 1 or not 0 <= y <= 1):
                return False
            self.x, self.y = float(x), float(y)
            return True
        if event.get("event") == "key":
            key = event.get("key")
            if key == "Left":
                self.x = max(0.0, self.x - .08)
            elif key == "Right":
                self.x = min(1.0, self.x + .08)
            elif key == "Up":
                self.y = max(0.0, self.y - .08)
            elif key == "Down":
                self.y = min(1.0, self.y + .08)
            else:
                return False
            return True
        return False


def validate_live(width: int, height: int, fps: int, max_frames: int) -> None:
    if (type(width) is not int or type(height) is not int
            or width % 2 or height % 2
            or not 64 <= width <= 640 or not 48 <= height <= 360
            or type(fps) is not int or not 1 <= fps <= 60
            or type(max_frames) is not int or not 4 <= max_frames <= MAX_LIVE_FRAMES
            or max_frames > fps * MAX_LIVE_SECONDS):
        raise ValueError("live video dimensions, FPS, or frame limit outside bounds")


def live_background(width, height, np):
    """Precompute a gradient; the moving marker is generated every frame."""
    x = np.arange(width, dtype=np.uint16)[None, :]
    y = np.arange(height, dtype=np.uint16)[:, None]
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = 32 + (x * 95 // width)
    rgb[:, :, 1] = 44 + (y * 105 // height)
    rgb[:, :, 2] = 105 + (x * 70 // width)
    return rgb


def render_live_rgb(scene: LiveScene, base, seq: int):
    """Make a fresh image that visibly changes with pointer and time."""
    rgb = base.copy()
    height, width, _ = rgb.shape
    side = max(8, min(42, width // 7, height // 5))
    cx = min(width - side, max(0, int(scene.x * (width - side))))
    cy = min(height - side, max(0, int(scene.y * (height - side))))
    rgb[cy:cy + side, cx:cx + side] = (250, 190, 42)
    # Moving cyan stripe proves the source continues rendering without input.
    stripe = (seq * 7) % width
    rgb[:, stripe:stripe + 2] = (18, 225, 226)
    return rgb


def make_live_codecs(width: int, height: int, fps: int):
    try:
        import av
        import numpy as np
    except ImportError:
        raise RuntimeError("local H.264 demo needs optional PyAV and NumPy") from None
    encoder = av.CodecContext.create("libx264", "w")
    encoder.width, encoder.height = width, height
    encoder.pix_fmt = "yuv420p"
    encoder.time_base = Fraction(1, fps)
    encoder.framerate = Fraction(fps, 1)
    encoder.thread_count = 1
    encoder.options = {
        "preset": "ultrafast", "tune": "zerolatency", "crf": "23",
        "x264-params": f"bframes=0:rc-lookahead=0:sync-lookahead=0:keyint={fps}:repeat-headers=1:aud=1",
    }
    encoder.open()
    decoder = av.CodecContext.create("h264", "r")
    decoder.thread_count = 1
    decoder.open()
    return av, np, encoder, decoder


def run_live_stream(outgoing: queue.Queue, frames: queue.Queue, stopped,
                    *, width: int = 640, height: int = 360,
                    fps: int = 30, max_frames: int = 120,
                    timed_frames: bool = False) -> dict:
    """Run one bounded live interactive source and independently decode it.

    Returned metrics use the same local high-resolution clock. Received frame
    acknowledgments refer to input observed by the source thread, not to the
    physical monitor presenting the result.
    """
    validate_live(width, height, fps, max_frames)
    if type(timed_frames) is not bool:
        raise ValueError("timed frames must be a boolean")
    av, np, encoder, decoder = make_live_codecs(width, height, fps)
    base = live_background(width, height, np)
    writer, reader = socket.socketpair()
    writer.settimeout(3)
    reader.settimeout(3)
    meta = queue.Queue(maxsize=16)
    failures = queue.Queue(maxsize=2)
    sent_frames = []
    decoded_ages_ms = []
    encode_costs_ms = []
    decode_costs_ms = []
    input_acks_ms = []
    input_callback_to_decode_ms = []
    delivered = 0
    input_changes = 0
    first_decode = None
    last_decode = None
    last_input_latency_seq = 0

    def produce():
        stage = "init"
        try:
            scene = LiveScene()
            latest_input_seq = 0
            latest_input_at = 0
            latest_input_callback_ns = None
            start = time.perf_counter()
            seq = 0
            while seq < max_frames and not stopped.is_set():
                delay = start + seq / fps - time.perf_counter()
                if delay > 0:
                    time.sleep(min(delay, .05))
                    continue
                event = None
                while True:
                    try:
                        event = outgoing.get_nowait()
                    except queue.Empty:
                        break
                if event is not None and scene.apply(event):
                    latest_input_seq += 1
                    latest_input_at = time.perf_counter_ns()
                    callback_ns = event.get("_ui_callback_ns")
                    # Keep callback timestamps local. They are never sent
                    # over the socket or accepted as a provider input field.
                    if (type(callback_ns) is int
                            and 0 < callback_ns <= latest_input_at
                            and latest_input_at - callback_ns <= 180 * 1_000_000_000):
                        latest_input_callback_ns = callback_ns
                    else:
                        latest_input_callback_ns = None
                captured_at = time.perf_counter_ns()
                stage = "render"
                rgb = render_live_rgb(scene, base, seq)
                stage = "frame-conversion"
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                frame.pts = seq
                stage = "encode"
                payloads = encoder.encode(frame)
                if len(payloads) != 1:
                    raise RuntimeError("live H.264 expected one packet per input frame")
                encoded = payloads[0]
                # A PyAV Packet exposes .size rather than a guaranteed
                # __len__. Validate the actual encoded byte payload.
                payload = bytes(encoded)
                stage = "packet-validate"
                if not 0 < len(payload) <= 2 * 1024 * 1024:
                    raise RuntimeError("live H.264 packet exceeds bounded size")
                now = time.perf_counter_ns()
                metadata = (seq, captured_at, latest_input_seq, latest_input_at,
                            latest_input_callback_ns)
                meta.put(metadata, timeout=3)
                sent_frames.append(seq)
                encode_costs_ms.append((now - captured_at) / 1e6)
                stage = "packet-send"
                writer.sendall(packet_bytes(VideoAccessUnit(
                    seq=seq, pts_ns=round(seq * 1e9 / fps),
                    capture_ns=captured_at, data=payload,
                    keyframe=bool(encoded.is_keyframe),
                )))
                seq += 1
            # Only useful on normal completion. B-frames disabled, so this
            # should not introduce a frame delay.
            stage = "final-flush"
            for tail in encoder.encode(None):
                raise RuntimeError("live H.264 encoder emitted unexpected delayed packet")
        except BaseException as exc:
            # Only static, allowlisted local stage names and exception types;
            # never copy arbitrary exception content into logs.
            failures.put(stage + ":" + type(exc).__name__)
        finally:
            try:
                writer.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    thread = threading.Thread(target=produce, daemon=True, name="h264-live-producer")
    thread.start()
    ppm_header = f"P6\n{width} {height}\n255\n".encode("ascii")
    print("[colabcola] Local H.264 live source connected (no Colab/GPU)", flush=True)
    offer_update(frames, ("status", "Local H.264 · interactive test"))
    try:
        previous = -1
        pending = deque()
        last_report = time.perf_counter()
        window_count = 0
        while True:
            packet = receive_packet(reader)
            if packet is None:
                break
            if packet.seq != previous + 1:
                raise RuntimeError("live H.264 source sequence discontinuity")
            previous = packet.seq
            metadata = meta.get(timeout=3)
            if metadata[0] != packet.seq or metadata[1] != packet.capture_ns:
                raise RuntimeError("live H.264 input/frame metadata mismatch")
            pending.append(metadata)
            start_decode = time.perf_counter_ns()
            decoded = decoder.decode(av.Packet(packet.data))
            decode_costs_ms.append((time.perf_counter_ns() - start_decode) / 1e6)
            for image in decoded:
                if not pending:
                    raise RuntimeError("live H.264 decoder produced unsolicited frame")
                seq, captured, ack_seq, input_at, callback_ns = pending.popleft()
                pixels = image.to_ndarray(format="rgb24").tobytes()
                if len(pixels) != width * height * 3:
                    raise RuntimeError("live H.264 RGB shape mismatch")
                now = time.perf_counter_ns()
                decoded_ages_ms.append((now - captured) / 1e6)
                delivered += 1
                if first_decode is None:
                    first_decode = now
                last_decode = now
                if ack_seq > last_input_latency_seq and input_at:
                    input_acks_ms.append((now - input_at) / 1e6)
                    if callback_ns is not None:
                        input_callback_to_decode_ms.append((now - callback_ns) / 1e6)
                    last_input_latency_seq = ack_seq
                    input_changes += 1
                if timed_frames:
                    timing = LocalFrameTiming(
                        captured_ns=captured, decoded_ns=now,
                        input_callback_ns=callback_ns)
                    offer_update(frames, ("frame-timed", (ppm_header + pixels, timing)))
                else:
                    offer_update(frames, ("frame", ppm_header + pixels))
                window_count += 1
            now_sec = time.perf_counter()
            if now_sec - last_report >= 2:
                rate = window_count / (now_sec - last_report)
                status = (
                    f"Local H.264 · decoded {rate:.1f} FPS · "
                    f"capture→decoded p50 {median(decoded_ages_ms):.2f} ms"
                )
                if input_acks_ms:
                    status += f" · input→decoded p50 {median(input_acks_ms):.1f} ms"
                print("[colabcola] " + status, flush=True)
                offer_update(frames, ("status", status))
                last_report, window_count = now_sec, 0

        for image in decoder.decode(None):
            if not pending:
                raise RuntimeError("live H.264 flush emitted unsolicited frame")
            seq, captured, ack_seq, input_at, callback_ns = pending.popleft()
            pixels = image.to_ndarray(format="rgb24").tobytes()
            now = time.perf_counter_ns()
            decoded_ages_ms.append((now - captured) / 1e6)
            delivered += 1
            last_decode = now
            if ack_seq > last_input_latency_seq and input_at:
                input_acks_ms.append((now - input_at) / 1e6)
                if callback_ns is not None:
                    input_callback_to_decode_ms.append((now - callback_ns) / 1e6)
                last_input_latency_seq = ack_seq
                input_changes += 1
            if timed_frames:
                timing = LocalFrameTiming(
                    captured_ns=captured, decoded_ns=now,
                    input_callback_ns=callback_ns)
                offer_update(frames, ("frame-timed", (ppm_header + pixels, timing)))
            else:
                offer_update(frames, ("frame", ppm_header + pixels))
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("live H.264 source shutdown timeout")
        if not failures.empty():
            raise RuntimeError("live H.264 source failed: " + failures.get())
        if not stopped.is_set() and (delivered != len(sent_frames) or pending):
            raise RuntimeError("live H.264 decode did not match encoded frames")
        rate = ((delivered - 1) * 1e9 / (last_decode - first_decode)
                if delivered >= 2 and last_decode > first_decode else 0)
        result = {
            "schema": 1, "mode": "live-interactive-h264-loopback",
            "decoded_frames": delivered,
            "produced_frames": len(sent_frames),
            "live_decoded_fps": round(rate, 2),
            "capture_to_decoded_p50_ms": round(median(decoded_ages_ms), 3)
            if decoded_ages_ms else None,
            "encode_p50_ms": round(median(encode_costs_ms), 3)
            if encode_costs_ms else None,
            "decode_p50_ms": round(median(decode_costs_ms), 3)
            if decode_costs_ms else None,
            "input_events_observed_in_decoded_frames": input_changes,
            "input_applied_to_decoded_p50_ms": round(median(input_acks_ms), 3)
            if input_acks_ms else None,
            "input_callback_to_decoded_p50_ms": round(
                median(input_callback_to_decode_ms), 3
            ) if input_callback_to_decode_ms else None,
            "scope": "live local generated frames and input; no Colab, GPU, remote network or compositor",
        }
        print("[colabcola] Live local H.264: " + str(result), flush=True)
        return result
    finally:
        stopped.set()
        writer.close()
        reader.close()
        thread.join(timeout=5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-seconds", type=int, default=30,
                        help="bounded local demo duration, between 1 and 180 seconds")
    parser.add_argument("--frames", type=int, default=None,
                        help="optional exact frame budget, capped by --max-seconds")
    parser.add_argument("--show", action="store_true",
                        help="display the live source in the local Tk viewer")
    args = parser.parse_args(argv)
    if not 1 <= args.max_seconds <= MAX_LIVE_SECONDS:
        parser.error("--max-seconds must be 1..180")
    frame_budget = args.frames if args.frames is not None else args.fps * args.max_seconds
    if frame_budget > args.fps * args.max_seconds:
        parser.error("--frames cannot exceed the explicit time budget")
    validate_live(args.width, args.height, args.fps, frame_budget)
    if args.show:
        from .bridge_demo import run_viewer
        run_viewer("127.0.0.1", 0, "", width=args.width, height=args.height,
                   title="Colab Cola · Live H.264 local proof",
                   worker=lambda outgoing, frames, stopped: run_live_stream(
                       outgoing, frames, stopped, width=args.width,
                       height=args.height, fps=args.fps, max_frames=frame_budget,
                       timed_frames=True),
                   viewer_stats=True, timestamp_local_input=True)
    else:
        stopped = threading.Event()
        run_live_stream(queue.Queue(maxsize=64), queue.Queue(maxsize=2),
                        stopped, width=args.width, height=args.height,
                        fps=args.fps, max_frames=frame_budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
