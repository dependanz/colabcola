"""Standalone worker uploaded into an owned Colab kernel by bridge_colab.

It uses PyTorch CUDA for every rendered frame, without requiring Colab Cola to
be installed inside the runtime. A single background thread atomically publishes
the latest compressed frame and applies bounded input updates.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import struct
import threading
from statistics import median
import time
import zlib

WIDTH = 640
HEIGHT = 360
MAGIC = b"CCB1"
HEADER = struct.Struct("!4sIIII")
RING_SLOTS = 8  # must match bridge_video_colab_source.RING_SLOTS


def atomic_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def publish_status(directory: Path, **details) -> None:
    atomic_bytes(directory / "status.json", (json.dumps(details, sort_keys=True) + "\n").encode())


def make_producer_window(*, run_id: str, last_seq: int, window_seq: int,
                         published_count: int, elapsed_s: float,
                         target_fps: int, rate_seq: int,
                         render_samples: list[float],
                         compress_samples: list[float],
                         publish_samples: list[float]) -> dict:
    """Build a remote-only, one-second CUDA producer sampling window.

    Not derived from the frequency of remote H.264 pull requests.
    """
    return {
        "schema": 1, "run_id": run_id, "seq": last_seq,
        "window_seq": window_seq, "published_count": published_count,
        "window_elapsed_ms": round(elapsed_s * 1000, 3),
        "target_fps": target_fps, "rate_seq": rate_seq,
        "renderer_fps": round(published_count / elapsed_s, 2),
        "render_copy_ms": round(median(render_samples), 2),
        "compress_ms": round(median(compress_samples), 2),
        "publish_ms": round(median(publish_samples), 2),
    }


def start(directory: str, fps: int, run_id: str, *, sequence_buffered: bool = False):
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not 1 <= fps <= 60:
        raise ValueError("fps outside accepted range")
    if type(sequence_buffered) is not bool:
        raise ValueError("invalid sequence-buffered selector")

    def worker():
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable on assigned runtime")
            name = torch.cuda.get_device_name(0)
            memory = int(torch.cuda.get_device_properties(0).total_memory)
            if "T4" not in name or memory < 12 * 1024**3:
                raise RuntimeError("assigned GPU is not a Tesla T4 with >=12 GiB")
            # Allocate and keep all coordinate grids on the remote CUDA device.
            x = torch.linspace(0, 1, WIDTH, device="cuda").view(1, WIDTH).expand(HEIGHT, WIDTH)
            y = torch.linspace(0, 1, HEIGHT, device="cuda").view(HEIGHT, 1).expand(HEIGHT, WIDTH)
            cx, cy, phase = 0.5, 0.5, 0.0
            seq, applied_seq = 0, 0
            current_fps = fps
            applied_rate_seq = 0
            started = time.monotonic()
            metrics_since = started
            render_samples = []
            compress_samples = []
            publish_samples = []
            publish_count = 0
            producer_window_seq = 0
            publish_status(directory, state="ready", device="cuda", gpu=name,
                           memory_bytes=memory, run_id=run_id, width=WIDTH, height=HEIGHT)
            while not (directory / "stop").exists():
                tick = time.monotonic()
                # An authenticated Comm handler writes this local control file.
                # Validate identity and monotonically increasing control sequence.
                rate_file = directory / "rate-control.json"
                if rate_file.is_file():
                    try:
                        rate = json.loads(rate_file.read_text(encoding="utf-8"))
                        requested_seq, requested_fps = rate.get("seq"), rate.get("fps")
                        if (rate.get("run_id") == run_id
                                and type(requested_seq) is int
                                and applied_rate_seq < requested_seq < 10000
                                and type(requested_fps) is int and 1 <= requested_fps <= 12):
                            current_fps = requested_fps
                            applied_rate_seq = requested_seq
                    except (OSError, UnicodeError, ValueError, AttributeError):
                        pass
                path = directory / "input.json"
                if path.is_file():
                    try:
                        event = json.loads(path.read_text(encoding="utf-8"))
                        number = event.get("seq")
                        if type(number) is int and applied_seq < number <= applied_seq + 100000:
                            if event.get("event") == "mouse":
                                nx, ny = event.get("x"), event.get("y")
                                if (type(nx) in (float, int) and type(ny) in (float, int)
                                        and 0 <= nx <= 1 and 0 <= ny <= 1):
                                    cx, cy = float(nx), float(ny)
                                    applied_seq = number
                            elif event.get("event") == "key" and isinstance(event.get("key"), str):
                                key = event["key"].lower()
                                if key == "left":
                                    phase -= 0.3
                                elif key == "right":
                                    phase += 0.3
                                elif key == "up":
                                    cy = max(0, cy - 0.05)
                                elif key == "down":
                                    cy = min(1, cy + 0.05)
                                applied_seq = number
                    except (OSError, UnicodeError, ValueError, KeyError):
                        pass  # A concurrently replaced event is retried next frame.
                seconds = time.monotonic() - started
                render_start = time.monotonic()
                glow = torch.exp(-42 * ((x - cx) ** 2 + (y - cy) ** 2))
                wave = torch.sin(x * 10 + seconds * 2 + phase) * 0.5 + 0.5
                r = (45 + 170 * glow + 35 * wave).clamp(0, 255)
                g = (30 + 100 * y + 65 * glow).clamp(0, 255)
                b = (90 + 95 * x + 50 * (1 - wave)).clamp(0, 255)
                pixels = torch.stack((r, g, b), dim=-1).to(torch.uint8).cpu().numpy().tobytes()
                # Includes CUDA kernels, synchronization, D2H transfer and CPU array conversion.
                render_copy_ms = (time.monotonic() - render_start) * 1000
                ppm = f"P6\n{WIDTH} {HEIGHT}\n255\n".encode() + pixels
                compress_start = time.monotonic()
                compressed = zlib.compress(ppm, level=1)
                compress_ms = (time.monotonic() - compress_start) * 1000
                if len(compressed) > 2 * 1024 * 1024:
                    raise RuntimeError("frame exceeds bridge transport bound")
                packed = HEADER.pack(MAGIC, seq, applied_seq, WIDTH, HEIGHT) + compressed
                publish_start = time.monotonic()
                # Opt-in fixed-size, monotonically numbered producer history.
                # Atomic replacement prevents torn reads; fixed slots cap
                # disk usage without an ever-growing frame queue.
                if sequence_buffered:
                    atomic_bytes(directory / f"frame-ring-{seq % RING_SLOTS:02d}.bin", packed)
                atomic_bytes(directory / "frame.bin", packed)
                publish_ms = (time.monotonic() - publish_start) * 1000
                render_samples.append(render_copy_ms)
                compress_samples.append(compress_ms)
                publish_samples.append(publish_ms)
                publish_count += 1
                now = time.monotonic()
                if now - metrics_since >= 1:
                    elapsed = now - metrics_since
                    # A unique producer measurement window is distinct from
                    # any host video_pull and from the renderer's frame seq.
                    producer_window_seq += 1
                    metrics = make_producer_window(
                        run_id=run_id, last_seq=seq, window_seq=producer_window_seq,
                        published_count=publish_count, elapsed_s=elapsed,
                        target_fps=current_fps, rate_seq=applied_rate_seq,
                        render_samples=render_samples, compress_samples=compress_samples,
                        publish_samples=publish_samples)
                    atomic_bytes(directory / "render-metrics.json",
                                 (json.dumps(metrics, separators=(",", ":"), allow_nan=False) + "\n").encode())
                    metrics_since = now
                    render_samples.clear()
                    compress_samples.clear()
                    publish_samples.clear()
                    publish_count = 0
                seq += 1
                time.sleep(max(0, 1 / current_fps - (time.monotonic() - tick)))
            publish_status(directory, state="stopped", device="cuda", gpu=name, run_id=run_id)
        except Exception as exc:
            # Keep diagnostics bounded; never publish secrets or raw stack traces.
            publish_status(directory, state="failed", error_type=type(exc).__name__,
                           message=str(exc)[:160], run_id=run_id)

    thread = threading.Thread(target=worker, name="colabcola-interactive", daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    # BRIDGE_DIRECTORY, BRIDGE_FPS and BRIDGE_RUN_ID are provided in init_globals.
    start(BRIDGE_DIRECTORY, BRIDGE_FPS, BRIDGE_RUN_ID)
