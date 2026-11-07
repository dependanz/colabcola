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
import time
import zlib

WIDTH = 640
HEIGHT = 360
MAGIC = b"CCB1"
HEADER = struct.Struct("!4sIIII")


def atomic_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def publish_status(directory: Path, **details) -> None:
    atomic_bytes(directory / "status.json", (json.dumps(details, sort_keys=True) + "\n").encode())


def start(directory: str, fps: int, run_id: str):
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not 1 <= fps <= 12:
        raise ValueError("fps outside accepted range")

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
            started = time.monotonic()
            publish_status(directory, state="ready", device="cuda", gpu=name,
                           memory_bytes=memory, run_id=run_id, width=WIDTH, height=HEIGHT)
            while not (directory / "stop").exists():
                tick = time.monotonic()
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
                glow = torch.exp(-42 * ((x - cx) ** 2 + (y - cy) ** 2))
                wave = torch.sin(x * 10 + seconds * 2 + phase) * 0.5 + 0.5
                r = (45 + 170 * glow + 35 * wave).clamp(0, 255)
                g = (30 + 100 * y + 65 * glow).clamp(0, 255)
                b = (90 + 95 * x + 50 * (1 - wave)).clamp(0, 255)
                pixels = torch.stack((r, g, b), dim=-1).to(torch.uint8).cpu().numpy().tobytes()
                ppm = f"P6\n{WIDTH} {HEIGHT}\n255\n".encode() + pixels
                compressed = zlib.compress(ppm, level=1)
                if len(compressed) > 2 * 1024 * 1024:
                    raise RuntimeError("frame exceeds bridge transport bound")
                packed = HEADER.pack(MAGIC, seq, applied_seq, WIDTH, HEIGHT) + compressed
                atomic_bytes(directory / "frame.bin", packed)
                seq += 1
                time.sleep(max(0, 1 / fps - (time.monotonic() - tick)))
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
