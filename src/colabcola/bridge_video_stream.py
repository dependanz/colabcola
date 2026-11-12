"""Offline paced H.264 streaming over a local socket pair.

This experimental module runs a persistent encoder and decoder concurrently
through bounded, versioned packets on a local socketpair. It makes no Colab
API requests, allocates no GPU and creates no public listening endpoint.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import queue
import re
import shutil
import socket
import struct
import subprocess
import threading
import time
import zlib
from statistics import median

from .bridge_video_benchmark import (
    _validate_dimensions, scene_like_rgb_frames, synthetic_rgb_frames,
)

MAGIC = b"CCV1"
VERSION = 1
KEYFRAME = 1
MAX_PACKET = 2 * 1024 * 1024
PACKET_HEADER = struct.Struct("!4sBBIQQI")
AUD = re.compile(b"\x00\x00(?:\x00)?\x01\x09")
NAL = re.compile(b"\x00\x00(?:\x00)?\x01")


@dataclass(frozen=True)
class VideoAccessUnit:
    seq: int
    pts_ns: int
    capture_ns: int
    data: bytes
    keyframe: bool = False


def packet_bytes(packet: VideoAccessUnit) -> bytes:
    if (type(packet.seq) is not int or not 0 <= packet.seq < 1000000
            or type(packet.pts_ns) is not int or not 0 <= packet.pts_ns < 10**14
            or type(packet.capture_ns) is not int or not 0 <= packet.capture_ns < 10**19
            or not isinstance(packet.data, bytes)
            or not 0 < len(packet.data) <= MAX_PACKET):
        raise ValueError("invalid encoded video packet")
    return PACKET_HEADER.pack(
        MAGIC, VERSION, KEYFRAME if packet.keyframe else 0,
        packet.seq, packet.pts_ns, packet.capture_ns, len(packet.data),
    ) + packet.data


def _exactly(sock: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        part = sock.recv(size - len(result))
        if not part:
            raise EOFError("truncated local video packet")
        result += part
    return bytes(result)


def receive_packet(sock: socket.socket) -> VideoAccessUnit | None:
    first = sock.recv(1)
    if not first:
        return None
    magic, version, flags, seq, pts, captured, size = PACKET_HEADER.unpack(
        first + _exactly(sock, PACKET_HEADER.size - 1)
    )
    if (magic != MAGIC or version != VERSION or flags & ~KEYFRAME
            or seq >= 1000000 or pts >= 10**14 or captured >= 10**19
            or not 0 < size <= MAX_PACKET):
        raise ValueError("invalid or oversized video packet header")
    return VideoAccessUnit(
        seq, pts, captured, _exactly(sock, size), bool(flags & KEYFRAME),
    )


def is_keyframe(unit: bytes) -> bool:
    for marker in NAL.finditer(unit):
        offset = marker.end()
        if offset < len(unit) and (unit[offset] & 0x1f) == 5:
            return True
    return False


class AccessUnitParser:
    """Split Annex B by x264's access-unit delimiter NALs (AUD enabled).

    The next AUD completes the previous frame: only one access-unit
    lookahead is necessary. The last access unit is flushed at encoder EOF.
    """

    def __init__(self):
        self.buffer = bytearray()

    def feed(self, chunk: bytes) -> list[bytes]:
        self.buffer.extend(chunk)
        if len(self.buffer) > MAX_PACKET * 2:
            raise ValueError("H.264 access-unit buffer exceeded limit")
        markers = list(AUD.finditer(self.buffer))
        if not markers:
            return []
        if markers[0].start() != 0:
            raise ValueError("H.264 stream missing initial access-unit delimiter")
        output = []
        for left, right in zip(markers, markers[1:]):
            unit = bytes(self.buffer[left.start():right.start()])
            if not 0 < len(unit) <= MAX_PACKET:
                raise ValueError("H.264 encoded access unit exceeded limit")
            output.append(unit)
        if len(markers) > 1:
            del self.buffer[:markers[-1].start()]
        return output

    def finish(self) -> list[bytes]:
        if not self.buffer:
            return []
        if not AUD.match(self.buffer) or len(self.buffer) > MAX_PACKET:
            raise ValueError("last H.264 access unit invalid")
        result = [bytes(self.buffer)]
        self.buffer.clear()
        return result


def _launch(ffmpeg: str, arguments: list[str]) -> subprocess.Popen:
    if not shutil.which(ffmpeg):
        raise RuntimeError("FFmpeg executable not available for optional streaming test")
    try:
        return subprocess.Popen(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", *arguments],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, bufsize=0,
        )
    except OSError as exc:
        raise RuntimeError("local FFmpeg launch failed: " + type(exc).__name__) from None


def frame_stage_times_ms(captured_ns: int, packetized_ns: int,
                         received_ns: int, decoded_ns: int) -> tuple[float, float, float]:
    """Decompose one frame's local age into consecutive monotonic intervals.

    Encoder output is observed after access-unit parsing; the first component
    includes the packetizer's lookahead. The second includes local socket and
    thread scheduling. The last includes decoder input buffering/conversion.
    """
    if not (0 <= captured_ns <= packetized_ns <= received_ns <= decoded_ns):
        raise ValueError("invalid local frame stage timestamps")
    return ((packetized_ns - captured_ns) / 1e6,
            (received_ns - packetized_ns) / 1e6,
            (decoded_ns - received_ns) / 1e6)


def _encoder(ffmpeg: str, width: int, height: int, fps: int):
    return _launch(ffmpeg, [
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}",
        "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "ultrafast", "-tune", "zerolatency",
        "-x264-params", "aud=1:repeat-headers=1",
        "-g", str(fps), "-bf", "0", "-pix_fmt", "yuv420p",
        "-f", "h264", "pipe:1",
    ])


def _decoder(ffmpeg: str, *, low_delay: bool = False):
    decoder_flags = ["-flags", "+low_delay"] if low_delay else []
    return _launch(ffmpeg, [
        "-f", "h264", "-probesize", "32", "-analyzeduration", "0",
        # This is an explicit A/B experiment rather than a new default:
        # some H.264 streams and decoder versions may handle this differently.
        *decoder_flags,
        # Without this, software decoder frame-threading adds several
        # frames of latency on the same test input.
        "-threads", "1", "-i", "pipe:0", "-an",
        "-fps_mode", "passthrough", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "pipe:1",
    ])


def compare_stream(*, width: int = 192, height: int = 120, frames: int = 24,
                   fps: int = 12, pattern: str = "square",
                   ffmpeg: str = "ffmpeg", decoder_low_delay: bool = False) -> dict:
    """Measure live decode progress and capture-to-decode age on loopback.

    A synthetic frame sequence is precomputed, then written to a persistent
    encoder at wall-clock pacing. Access units are transmitted as bounded
    binary packets and fed into a persistent decoder as they arrive.

    Capture-to-decode timestamps share one local monotonic clock. They
    cannot be used as remote one-way latency measurements.
    """
    _validate_dimensions(width, height, frames)
    if (type(fps) is not int or not 1 <= fps <= 60
            or pattern not in ("square", "scene")
            or type(decoder_low_delay) is not bool):
        raise ValueError("invalid paced video benchmark arguments")
    originals = (synthetic_rgb_frames(width, height, frames)
                 if pattern == "square"
                 else scene_like_rgb_frames(width, height, frames, fps))
    zlib_bytes = sum(len(zlib.compress(rgb, 1)) for rgb in originals)
    encoder = _encoder(ffmpeg, width, height, fps)
    try:
        decoder = _decoder(ffmpeg, low_delay=decoder_low_delay)
    except Exception:
        encoder.terminate()
        encoder.wait(timeout=5)
        raise

    sender, receiver = socket.socketpair()
    sender.settimeout(10)
    receiver.settimeout(10)
    failures = queue.Queue()
    source_done = threading.Event()
    captured_ns: list[int] = []
    packetized_ns: list[int] = []
    received_ns: list[int] = []
    decoded_results: list[tuple[int, int, bool, bytes]] = []
    sent_packets: list[VideoAccessUnit] = []
    metadata: queue.Queue[VideoAccessUnit] = queue.Queue(maxsize=48)
    frame_size = width * height * 3

    def guarded(fn):
        def run():
            try:
                fn()
            except BaseException as exc:
                failures.put(type(exc).__name__)
        return run

    def source_writer():
        beginning = time.monotonic()
        for index, raw in enumerate(originals):
            remaining = beginning + index / fps - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            captured_ns.append(time.monotonic_ns())
            encoder.stdin.write(raw)
        source_done.set()
        encoder.stdin.close()

    def send_encoded():
        parser = AccessUnitParser()
        index = 0

        def transmit(unit: bytes):
            nonlocal index
            if index >= frames:
                raise ValueError("encoded more H.264 units than source frames")
            while len(captured_ns) <= index:
                time.sleep(.002)
            packet = VideoAccessUnit(
                index, round(index * 1_000_000_000 / fps),
                captured_ns[index], unit, is_keyframe(unit),
            )
            # Record before sendall so a concurrent receiver cannot observe
            # the packet before its local enqueue timestamp is recorded.
            packetized_ns.append(time.monotonic_ns())
            sender.sendall(packet_bytes(packet))
            sent_packets.append(packet)
            index += 1

        while True:
            data = os.read(encoder.stdout.fileno(), 65536)
            if not data:
                break
            for unit in parser.feed(data):
                transmit(unit)
        for unit in parser.finish():
            transmit(unit)
        if index != frames:
            raise ValueError("unexpected encoder output access-unit count")
        sender.shutdown(socket.SHUT_WR)

    def receive_encoded():
        previous = -1
        while True:
            packet = receive_packet(receiver)
            if packet is None:
                break
            if packet.seq != previous + 1:
                raise ValueError("encoded video packet arrived out of order")
            previous = packet.seq
            received_ns.append(time.monotonic_ns())
            metadata.put(packet, timeout=5)
            decoder.stdin.write(packet.data)
        decoder.stdin.close()

    def read_decoded():
        while True:
            output = bytearray()
            while len(output) < frame_size:
                chunk = os.read(decoder.stdout.fileno(), frame_size - len(output))
                if not chunk:
                    if output:
                        raise ValueError("truncated decoded RGB frame")
                    return
                output.extend(chunk)
            packet = metadata.get(timeout=5)
            decoded_results.append(
                (packet.seq, time.monotonic_ns(),
                 not source_done.is_set(), bytes(output))
            )

    threads = [
        threading.Thread(target=guarded(fn), daemon=True, name=name)
        for name, fn in (
            ("source", source_writer), ("encoder", send_encoded),
            ("receiver", receive_encoded), ("decoder", read_decoded),
        )
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        if any(thread.is_alive() for thread in threads):
            raise RuntimeError("local video streaming thread timed out")
        if not failures.empty():
            raise RuntimeError("local video streaming worker failed: " + failures.get())
        if encoder.wait(timeout=5) or decoder.wait(timeout=5):
            raise RuntimeError("local FFmpeg encoder or decoder failed")
        if (len(decoded_results) != frames or len(sent_packets) != frames
                or len(packetized_ns) != frames or len(received_ns) != frames):
            raise RuntimeError("local video stream did not decode all frames")
        if decoded_results[0][3] == decoded_results[-1][3]:
            raise RuntimeError("decoded stream had no visual changes")

        stage_ms = [
            frame_stage_times_ms(captured_ns[index], packetized_ns[index],
                                 received_ns[index], when)
            for index, when, preclose, pixels in decoded_results
        ]
        ages_ms = [(when - captured_ns[index]) / 1e6
                   for index, when, preclose, pixels in decoded_results]
        during = [(index, when) for index, when, preclose, pixels
                  in decoded_results if preclose]
        live_fps = ((len(during) - 1) * 1e9 / (during[-1][1] - during[0][1])
                    if len(during) >= 2 else 0.0)
        encoded_bytes = sum(len(packet.data) for packet in sent_packets)
        return {
            "schema": 1, "mode": "persistent-h264-loopback",
            "decoder_low_delay": decoder_low_delay,
            "width": width, "height": height, "frames": frames,
            "target_fps": fps, "decoded_frames": len(decoded_results),
            "decoded_before_source_finished": len(during),
            "live_decode_fps": round(live_fps, 2),
            "capture_to_decoded_p50_ms": round(median(ages_ms), 1),
            "capture_to_decoded_max_ms": round(max(ages_ms), 1),
            "capture_to_packet_p50_ms": round(median(s[0] for s in stage_ms), 1),
            "packet_to_socket_p50_ms": round(median(s[1] for s in stage_ms), 1),
            "socket_to_decoded_p50_ms": round(median(s[2] for s in stage_ms), 1),
            "capture_to_decoded_frame_periods": round(median(ages_ms) * fps / 1000, 2),
            "h264_bytes": encoded_bytes, "zlib_bytes": zlib_bytes,
            "h264_relative_size": round(encoded_bytes / zlib_bytes, 4),
            "keyframes": sum(int(packet.keyframe) for packet in sent_packets),
            "scope": "paced offline local socketpair; no Colab, GPU or media endpoint",
        }
    finally:
        for sock in (sender, receiver):
            try:
                sock.close()
            except OSError:
                pass
        for process in (encoder, decoder):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--pattern", choices=("square", "scene"), default="square")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--decoder-low-delay", action="store_true",
                        help="experimental FFmpeg decoder low-delay flag; A/B against default")
    args = parser.parse_args(argv)
    print(json.dumps(compare_stream(
        width=args.width, height=args.height, frames=args.frames,
        fps=args.fps, pattern=args.pattern, ffmpeg=args.ffmpeg,
        decoder_low_delay=args.decoder_low_delay,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
