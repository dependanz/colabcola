"""Optional in-process H.264 loopback benchmark using PyAV.

Unlike the FFmpeg subprocess experiment, PyAV returns complete encoder
packets and accepts decoder packets directly. This isolates subprocess
stream parsing and buffering from the H.264 codec itself. No Colab,
external socket, GPU, or PyAV dependency is added to the released runner.

Usage (in an experimental Python environment):
    python -m pip install av numpy
    python -m colabcola.bridge_video_inprocess --width 640 --height 360 \
        --frames 24 --fps 12 --pattern scene

NOTE: PyAV wheels may bundle GPL-licensed x264/FFmpeg components. Review
distribution licensing before adding any of them as package dependencies.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import socket
import time
import zlib
from statistics import median

from .bridge_video_benchmark import (
    _validate_dimensions, scene_like_rgb_frames, synthetic_rgb_frames,
)
from .bridge_video_stream import VideoAccessUnit, packet_bytes, receive_packet


def compare_inprocess(*, width: int = 192, height: int = 120,
                      frames: int = 24, fps: int = 12,
                      pattern: str = "square") -> dict:
    """Pace precomputed RGB frames through the same bounded local wire format.

    The encoder and decoder run in the calling process. There is no extra
    access-unit parser, OS subprocess pipe, or remote transport. A local
    monotonic clock measures capture through output RGB conversion.
    """
    _validate_dimensions(width, height, frames)
    if type(fps) is not int or not 1 <= fps <= 60 or pattern not in ("scene", "square"):
        raise ValueError("invalid in-process benchmark configuration")
    try:
        import av
        import numpy as np
    except ImportError:
        raise RuntimeError("optional in-process benchmark requires PyAV and NumPy") from None

    source = (scene_like_rgb_frames(width, height, frames, fps)
              if pattern == "scene" else synthetic_rgb_frames(width, height, frames))
    zlib_bytes = sum(len(zlib.compress(frame, 1)) for frame in source)

    encoder = av.CodecContext.create("libx264", "w")
    encoder.width = width
    encoder.height = height
    encoder.pix_fmt = "yuv420p"
    encoder.time_base = Fraction(1, fps)
    encoder.framerate = Fraction(fps, 1)
    encoder.thread_count = 1
    encoder.options = {
        "preset": "ultrafast",
        "tune": "zerolatency",
        "crf": "23",
        "x264-params": f"bframes=0:rc-lookahead=0:sync-lookahead=0:keyint={fps}:repeat-headers=1:aud=1",
    }
    encoder.open()
    decoder = av.CodecContext.create("h264", "r")
    decoder.thread_count = 1
    decoder.open()

    left, right = socket.socketpair()
    left.settimeout(5)
    right.settimeout(5)
    captured_ns: list[int] = []
    packet_ready_ns: list[int] = []
    socket_recv_ns: list[int] = []
    decoded_ns: list[int] = []
    decoded_pixels = []
    keyframes = 0
    encoded_bytes = 0
    decoded_before_end = 0
    next_packet = 0
    started = time.monotonic()

    def decode_packet(packet_bytes_: bytes, seq: int, source_active: bool):
        nonlocal decoded_before_end
        packet = av.Packet(packet_bytes_)
        # Input frames and encoded packets are in presentation order (no B-frames).
        packet.pts = seq
        for frame in decoder.decode(packet):
            pixels = frame.to_ndarray(format="rgb24").tobytes()
            if len(pixels) != width * height * 3:
                raise RuntimeError("in-process H.264 decoded frame has incorrect dimensions")
            decoded_pixels.append(pixels)
            decoded_ns.append(time.monotonic_ns())
            if source_active:
                decoded_before_end += 1

    def deliver(encoded, *, source_active: bool):
        nonlocal next_packet, keyframes, encoded_bytes
        payload = bytes(encoded)
        if not 0 < len(payload) <= 2 * 1024 * 1024:
            raise RuntimeError("encoded H.264 packet outside bounded size")
        if next_packet >= frames:
            raise RuntimeError("unexpected extra H.264 packet")
        seq = next_packet
        next_packet += 1
        ready = time.monotonic_ns()
        packet_ready_ns.append(ready)
        wire = packet_bytes(VideoAccessUnit(
            seq=seq, pts_ns=round(seq * 1_000_000_000 / fps),
            capture_ns=captured_ns[seq], data=payload,
            keyframe=bool(encoded.is_keyframe),
        ))
        left.sendall(wire)
        received = receive_packet(right)
        if received is None or received.seq != seq:
            raise RuntimeError("local H.264 sequence ID mismatch")
        socket_recv_ns.append(time.monotonic_ns())
        encoded_bytes += len(payload)
        keyframes += int(received.keyframe)
        decode_packet(received.data, seq, source_active)

    try:
        for seq, raw in enumerate(source):
            remaining = started + seq / fps - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            captured_ns.append(time.monotonic_ns())
            array = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            source_frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            source_frame.pts = seq
            for encoded in encoder.encode(source_frame):
                deliver(encoded, source_active=seq < frames - 1)
        for encoded in encoder.encode(None):
            deliver(encoded, source_active=False)
        for frame in decoder.decode(None):
            pixels = frame.to_ndarray(format="rgb24").tobytes()
            decoded_pixels.append(pixels)
            decoded_ns.append(time.monotonic_ns())

        if (next_packet != frames or len(decoded_pixels) != frames
                or len(decoded_ns) != frames or len(packet_ready_ns) != frames):
            raise RuntimeError("in-process codec lost or duplicated frames")
        if decoded_pixels[0] == decoded_pixels[-1]:
            raise RuntimeError("in-process decoded video did not change")

        ages = [(decode_at - capture_at) / 1e6
                for capture_at, decode_at in zip(captured_ns, decoded_ns)]
        to_packet = [(ready - capture_at) / 1e6
                     for capture_at, ready in zip(captured_ns, packet_ready_ns)]
        to_socket = [(recv - ready) / 1e6
                     for ready, recv in zip(packet_ready_ns, socket_recv_ns)]
        from_socket = [(done - recv) / 1e6
                       for recv, done in zip(socket_recv_ns, decoded_ns)]
        if any(ms < 0 for stage in (ages, to_packet, to_socket, from_socket)
               for ms in stage):
            raise RuntimeError("unexpected in-process codec timestamp ordering")
        active = decoded_ns[:decoded_before_end]
        active_fps = ((len(active) - 1) * 1e9 / (active[-1] - active[0])
                      if len(active) > 1 else 0.0)
        return {
            "schema": 1, "mode": "inprocess-h264-loopback",
            "width": width, "height": height, "target_fps": fps, "frames": frames,
            "decoded_frames": len(decoded_pixels),
            "decoded_before_source_finished": decoded_before_end,
            "live_decode_fps": round(active_fps, 2),
            "capture_to_decoded_p50_ms": round(median(ages), 1),
            "capture_to_packet_p50_ms": round(median(to_packet), 1),
            "packet_to_socket_p50_ms": round(median(to_socket), 1),
            "socket_to_decoded_p50_ms": round(median(from_socket), 1),
            "capture_to_decoded_frame_periods": round(median(ages) * fps / 1000, 2),
            "h264_bytes": encoded_bytes, "zlib_bytes": zlib_bytes,
            "h264_relative_size": round(encoded_bytes / zlib_bytes, 4),
            "keyframes": keyframes,
            "scope": "in-process optional PyAV local socketpair; no Colab, GPU or media endpoint",
        }
    finally:
        left.close()
        right.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--pattern", choices=("square", "scene"), default="square")
    args = parser.parse_args(argv)
    print(json.dumps(compare_inprocess(
        width=args.width, height=args.height, frames=args.frames,
        fps=args.fps, pattern=args.pattern,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
