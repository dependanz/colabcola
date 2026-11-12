"""Offline H.264 vs zlib experiment for the Colab Cola interactive bridge.

Uses synthetic RGB frames and an optional local FFmpeg executable. No Colab
connection, GPU allocation, sockets, or additions to the released runner.
This is a CODEC benchmark, not an end-to-end streaming performance claim.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
import zlib


def _validate_dimensions(width: int, height: int, count: int) -> None:
    if (type(width) is not int or type(height) is not int or type(count) is not int
            or not 32 <= width <= 640 or not 32 <= height <= 360
            or width % 2 or height % 2 or not 4 <= count <= 48
            or width * height * 3 * count > 40 * 1024 * 1024):
        raise ValueError("video benchmark dimensions or frame count outside bounds")


def synthetic_rgb_frames(width: int, height: int, count: int) -> list[bytes]:
    """Deterministic animated color field with one moving square per frame."""
    _validate_dimensions(width, height, count)
    background = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            offset = 3 * (y * width + x)
            background[offset] = (40 + 175 * x // width) % 256
            background[offset + 1] = (25 + 155 * y // height) % 256
            background[offset + 2] = (95 + x // 7 + y // 9) % 256
    side = min(52, width // 4, height // 3)
    result = []
    for seq in range(count):
        frame = background.copy()
        px = (seq * 11) % max(1, width - side)
        py = (seq * 7) % max(1, height - side)
        square = bytes((220, (seq * 9) % 180 + 40, 40)) * side
        for row in range(py, py + side):
            start = 3 * (row * width + px)
            frame[start:start + 3 * side] = square
        result.append(bytes(frame))
    return result


def scene_like_rgb_frames(width: int, height: int, count: int, fps: int) -> list[bytes]:
    """Approximate the bridge's moving CUDA glow/wave scene using CPU math."""
    _validate_dimensions(width, height, count)
    if type(fps) is not int or not 1 <= fps <= 60:
        raise ValueError("invalid benchmark frame rate")
    xs = [x / (width - 1) for x in range(width)]
    ys = [y / (height - 1) for y in range(height)]
    result = []
    for seq in range(count):
        seconds = seq / fps
        cx = .5 + .15 * math.sin(seconds * .8)
        cy = .5 + .12 * math.cos(seconds * .6)
        frame = bytearray(width * height * 3)
        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                glow = math.exp(-42 * ((x - cx) ** 2 + (y - cy) ** 2))
                wave = math.sin(x * 10 + seconds * 2) * .5 + .5
                offset = 3 * (yi * width + xi)
                frame[offset] = min(255, int(45 + 170 * glow + 35 * wave))
                frame[offset + 1] = min(255, int(30 + 100 * y + 65 * glow))
                frame[offset + 2] = min(255, int(90 + 95 * x + 50 * (1 - wave)))
        result.append(bytes(frame))
    return result


def _ffmpeg(data: bytes, arguments: list[str], *, binary: str, timeout: int = 45) -> bytes:
    if not binary or shutil.which(binary) is None:
        raise RuntimeError("FFmpeg unavailable; install FFmpeg for this optional benchmark")
    try:
        process = subprocess.run(
            [binary, "-hide_banner", "-loglevel", "error", "-nostdin", *arguments],
            input=data, capture_output=True, check=False, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("local FFmpeg benchmark failed: " + type(exc).__name__) from None
    if process.returncode or not process.stdout:
        # Do not copy subprocess stderr or environment paths into public output.
        raise RuntimeError("local FFmpeg codec operation failed")
    return process.stdout


def encode_h264(frames: list[bytes], width: int, height: int, fps: int,
                *, binary: str = "ffmpeg", crf: int = 23) -> bytes:
    _validate_dimensions(width, height, len(frames))
    if (type(fps) is not int or not 1 <= fps <= 60
            or type(crf) is not int or not 0 <= crf <= 45
            or any(len(frame) != width * height * 3 for frame in frames)):
        raise ValueError("invalid H.264 benchmark frame input")
    return _ffmpeg(b"".join(frames), [
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}",
        "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "ultrafast", "-tune", "zerolatency", "-bf", "0",
        "-g", str(fps), "-pix_fmt", "yuv420p", "-crf", str(crf),
        "-f", "h264", "pipe:1",
    ], binary=binary)


def decode_h264(stream: bytes, width: int, height: int,
                *, binary: str = "ffmpeg") -> list[bytes]:
    decoded = _ffmpeg(stream, [
        "-f", "h264", "-i", "pipe:0", "-an", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-fps_mode", "passthrough", "pipe:1",
    ], binary=binary)
    frame_bytes = width * height * 3
    if (not 32 <= width <= 640 or not 32 <= height <= 360
            or width % 2 or height % 2 or len(decoded) % frame_bytes):
        raise RuntimeError("decoded H.264 frame shape is invalid")
    return [decoded[i:i + frame_bytes] for i in range(0, len(decoded), frame_bytes)]


def compare_codecs(*, width: int = 640, height: int = 360, frames: int = 24,
                   fps: int = 12, ffmpeg: str = "ffmpeg",
                   pattern: str = "scene") -> dict:
    if pattern not in ("scene", "square"):
        raise ValueError("unsupported video benchmark pattern")
    _validate_dimensions(width, height, frames)
    if type(fps) is not int or not 1 <= fps <= 60:
        raise ValueError("invalid benchmark FPS")
    original = (scene_like_rgb_frames(width, height, frames, fps)
                if pattern == "scene" else synthetic_rgb_frames(width, height, frames))
    zlib_bytes = sum(len(zlib.compress(rgb, 1)) for rgb in original)
    started = time.perf_counter()
    stream = encode_h264(original, width, height, fps, binary=ffmpeg)
    encode_time = time.perf_counter() - started
    started = time.perf_counter()
    decoded = decode_h264(stream, width, height, binary=ffmpeg)
    decode_time = time.perf_counter() - started
    if len(decoded) != frames:
        raise RuntimeError("H.264 did not decode to the expected number of frames")
    total_error = sum((u - v) ** 2 for expected, actual in zip(original, decoded)
                      for u, v in zip(expected, actual))
    mse = total_error / (frames * width * height * 3)
    psnr = 10 * math.log10(255 * 255 / mse) if mse else None
    return {
        "schema": 1, "pattern": pattern, "codec": "H.264 libx264 zerolatency",
        "width": width, "height": height, "frames": frames, "fps": fps,
        "zlib_bytes": zlib_bytes, "h264_bytes": len(stream),
        "h264_relative_size": round(len(stream) / zlib_bytes, 4),
        "h264_bytes_per_frame": round(len(stream) / frames),
        "psnr_db": round(psnr, 2) if psnr is not None else None,
        "encode_batch_seconds": round(encode_time, 3),
        "decode_batch_seconds": round(decode_time, 3),
        "scope": "offline synthetic frames; no network, GPU or live streaming",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--pattern", choices=("scene", "square"), default="scene")
    args = parser.parse_args(argv)
    print(json.dumps(compare_codecs(width=args.width, height=args.height,
                                    frames=args.frames, fps=args.fps,
                                    ffmpeg=args.ffmpeg, pattern=args.pattern), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
