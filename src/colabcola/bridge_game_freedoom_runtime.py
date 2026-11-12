"""v36A: isolated, opt-in Freedoom/X11 frame source for the authenticated bridge.

Executed only after explicit attachment to one user-owned Colab kernel.
Requires pre-installed local binaries and freely redistributable IWAD.
No package manager, shell, remote listener, or generic command execution.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import zlib
from statistics import median

WIDTH, HEIGHT, FPS = 640, 360, 30
FRAME_BYTES = WIDTH * HEIGHT * 3
FRAME_PREFIX = f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii")
FRAME_HEADER = struct.Struct("!4sIIII")
RING_SLOTS = 8
DISPLAY = ":98"
IWAD_CANDIDATES = (
    "/usr/share/games/doom/freedoom1.wad",
    "/usr/share/games/doom/freedoom2.wad",
)
BINARY_NAMES = ("Xvfb", "chocolate-doom", "ffmpeg", "xdotool")
# Ubuntu's chocolate-doom package installs the game under /usr/games,
# which minimal Colab/Jupyter PATH values often omit.
PACKAGED_CHOCOLATE_DOOM = "/usr/games/chocolate-doom"
KEY_MAP = {
    "Up": "Up", "Down": "Down", "Left": "Left", "Right": "Right",
    "Control_L": "Control_L", "Control_R": "Control_R",
    "space": "space", "Return": "Return", "Escape": "Escape",
    "Shift_L": "Shift_L", "Shift_R": "Shift_R",
    "Tab": "Tab",
}
BUTTONS = (1, 3)
MAX_EVENTS = 64
_CONTROLLER = None


def atomic_bytes(path: Path, body: bytes) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(body)
    os.replace(temp, path)


def _status(root, *, run_id, state, **fields) -> None:
    atomic_bytes(root / "status.json", (json.dumps(
        {"run_id": run_id, "state": state, **fields}, sort_keys=True) + "\n").encode())


def _p95(values: list[float]) -> float:
    values = sorted(values)
    index = (len(values) - 1) * 0.95
    low = int(index)
    return round(values[low] + (index - low) *
                 (values[min(low + 1, len(values) - 1)] - values[low]), 2)


def validate_event(event: object) -> dict:
    """Bounded input contract independent of shell/X11 and transport state."""
    if not isinstance(event, dict) or type(event.get("seq")) is not int or not (
            1 <= event["seq"] < 10_000_000):
        raise ValueError("game input sequence invalid")
    kind = event.get("event")
    if kind in ("key_down", "key_up"):
        key = event.get("key")
        if type(key) is not str or key not in KEY_MAP or set(event) != {"event", "key", "seq"}:
            raise ValueError("game key event invalid")
        return {"seq": event["seq"], "event": kind, "key": key}
    if kind in ("mouse_button_down", "mouse_button_up"):
        button = event.get("button")
        if type(button) is not int or button not in BUTTONS or set(event) != {
                "event", "button", "seq"}:
            raise ValueError("game mouse button invalid")
        return {"seq": event["seq"], "event": kind, "button": button}
    if kind == "mouse_delta":
        dx, dy = event.get("dx"), event.get("dy")
        if (type(dx) is not int or type(dy) is not int
                or not -64 <= dx <= 64 or not -64 <= dy <= 64
                or set(event) != {"event", "dx", "dy", "seq"}):
            raise ValueError("game mouse delta invalid")
        return {"seq": event["seq"], "event": kind, "dx": dx, "dy": dy}
    raise ValueError("unsupported game input")


def binary_inventory(which=shutil.which) -> dict[str, str]:
    """Only approved pre-installed absolute executable paths are used."""
    result = {}
    for name in BINARY_NAMES:
        executable = which(name)
        if executable is None and name == "chocolate-doom":
            # Standard Ubuntu package location, not a dynamic executable path.
            executable = PACKAGED_CHOCOLATE_DOOM
        if (not isinstance(executable, str) or not os.path.isabs(executable)
                or not os.path.isfile(executable) or not os.access(executable, os.X_OK)):
            raise RuntimeError("freedoom dependency missing: " + name)
        result[name] = executable
    return result


def select_freedoom_iwad(candidates=IWAD_CANDIDATES) -> str:
    """Only fixed package-distributed Freedoom IWADs; never a user command."""
    for filename in candidates:
        path = Path(filename)
        try:
            if (not path.is_file() or path.is_symlink()
                    or not 1_000_000 <= path.stat().st_size <= 100_000_000):
                continue
            with path.open("rb") as f:
                header = f.read(12)
            if len(header) == 12 and header[:4] == b"IWAD":
                return str(path)
        except OSError:
            pass
    raise RuntimeError("freedoom IWAD unavailable; install package explicitly")


def doom_warp_arguments(iwad: str) -> tuple[str, ...]:
    """Chocolate Doom's Doom I and Doom II warp syntax differs."""
    if iwad == IWAD_CANDIDATES[0]:
        return ("-warp", "1", "1")
    if iwad == IWAD_CANDIDATES[1]:
        return ("-warp", "1")
    raise ValueError("game IWAD is not allowlisted")


def _read_exact(pipe, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        more = pipe.read(size - len(chunks))
        if not more:
            raise RuntimeError("game capture pipe ended")
        chunks.extend(more)
    return bytes(chunks)


def _stop_proc(proc) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
    if getattr(proc, "stdout", None):
        proc.stdout.close()


class FreedoomSource:
    """Owns only the three processes started for this run, never the T4."""

    def __init__(self, directory, run_id, fps=30, sequence_buffered=True):
        if (not isinstance(run_id, str) or re.fullmatch(r"[0-9a-f]{32}", run_id) is None
                or type(fps) is not int or fps != FPS
                or sequence_buffered is not True):
            raise ValueError("Freedoom requires 30Hz buffered source")
        self.root = Path(directory)
        self.run_id = run_id
        self.events = queue.Queue(maxsize=MAX_EVENTS)
        self._latest_queued = 0
        self._applied_seq = 0
        self._lock = threading.Lock()
        self._pressed_keys = set()
        self._pressed_buttons = set()
        self._stop = threading.Event()
        self._thread = None
        self._processes = []
        self._xdotool = None
        self._env = None

    def submit_input(self, payload: object) -> bool:
        record = validate_event(payload)
        with self._lock:
            if self._stop.is_set() or record["seq"] <= self._latest_queued:
                return False
            try:
                self.events.put_nowait(record)
            except queue.Full:
                return False
            self._latest_queued = record["seq"]
        return True

    def _xdotool_command(self, *args) -> None:
        completed = subprocess.run(
            [self._xdotool, *args], env=self._env,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, check=False, timeout=1)
        if completed.returncode != 0:
            raise RuntimeError("virtual game input failed")

    def _apply(self, item: dict) -> None:
        kind = item["event"]
        if kind in ("key_down", "key_up"):
            key = KEY_MAP[item["key"]]
            if kind == "key_down" and key not in self._pressed_keys:
                self._xdotool_command("keydown", key)
                self._pressed_keys.add(key)
            elif kind == "key_up" and key in self._pressed_keys:
                self._xdotool_command("keyup", key)
                self._pressed_keys.remove(key)
        elif kind in ("mouse_button_down", "mouse_button_up"):
            button = item["button"]
            if kind == "mouse_button_down" and button not in self._pressed_buttons:
                self._xdotool_command("mousedown", str(button))
                self._pressed_buttons.add(button)
            elif kind == "mouse_button_up" and button in self._pressed_buttons:
                self._xdotool_command("mouseup", str(button))
                self._pressed_buttons.remove(button)
        elif item["dx"] or item["dy"]:
            self._xdotool_command(
                "mousemove_relative", "--", str(item["dx"]), str(item["dy"]))
        self._applied_seq = item["seq"]

    def _release_all(self) -> None:
        for key in sorted(self._pressed_keys):
            try:
                self._xdotool_command("keyup", key)
            except Exception:
                pass
        self._pressed_keys.clear()
        for button in sorted(self._pressed_buttons):
            try:
                self._xdotool_command("mouseup", str(button))
            except Exception:
                pass
        self._pressed_buttons.clear()

    def _launch(self, args, *, capture=False):
        proc = subprocess.Popen(
            args, cwd=str(self.root), env=self._env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, bufsize=0,
            start_new_session=True,
        )
        self._processes.append(proc)
        return proc

    def _worker(self):
        try:
            if not sys.platform.startswith("linux"):
                raise RuntimeError("Freedoom source requires Linux")
            binaries = binary_inventory()
            iwad = select_freedoom_iwad()
            # Verify the same user-owned T4 as the existing external bridge.
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("external T4 GPU unavailable")
            gpu = torch.cuda.get_device_name(0)
            if "T4" not in gpu:
                raise RuntimeError("external GPU is not T4")

            if (Path("/tmp/.X11-unix/X98").exists() or
                    Path("/tmp/.X98-lock").exists()):
                raise RuntimeError("game virtual display already occupied")
            self._env = dict(os.environ, DISPLAY=DISPLAY,
                             SDL_VIDEODRIVER="x11",
                             SDL_AUDIODRIVER="dummy")
            self._xdotool = binaries["xdotool"]
            self._launch([binaries["Xvfb"], DISPLAY, "-screen", "0",
                          f"{WIDTH}x{HEIGHT}x24", "-nolisten", "tcp"])
            for _ in range(60):
                if self._stop.is_set():
                    return
                if self._processes[0].poll() is not None:
                    raise RuntimeError("virtual display exited")
                if Path("/tmp/.X11-unix/X98").exists():
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError("virtual display startup timeout")

            if self._stop.is_set():
                return
            doom = self._launch([
                binaries["chocolate-doom"], "-iwad", iwad, "-window",
                "-geometry", f"{WIDTH}x{HEIGHT}", "-nograbmouse",
                "-nosound", "-skill", "3",
                *doom_warp_arguments(iwad),
            ])
            window_id = None
            for _ in range(100):
                if self._stop.is_set():
                    return
                if doom.poll() is not None:
                    raise RuntimeError("game process exited")
                found = subprocess.run(
                    [self._xdotool, "search", "--pid", str(doom.pid)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, env=self._env,
                    check=False, timeout=1)
                if found.returncode == 0:
                    matches = found.stdout.decode("ascii", errors="ignore").split()
                    if matches and matches[0].isdigit():
                        window_id = matches[0]
                        break
                time.sleep(0.1)
            if window_id is None:
                raise RuntimeError("game window not found")
            if self._stop.is_set():
                return
            self._xdotool_command("windowfocus", window_id)

            capture = self._launch([
                binaries["ffmpeg"], "-nostdin", "-hide_banner",
                "-loglevel", "error", "-f", "x11grab",
                "-draw_mouse", "0", "-video_size", f"{WIDTH}x{HEIGHT}",
                "-framerate", str(FPS), "-i", DISPLAY + ".0+0,0",
                "-pix_fmt", "rgb24", "-f", "rawvideo",
                "-an", "-sn", "pipe:1",
            ], capture=True)
            seq = 0
            started = time.monotonic()
            metrics_since = started
            render_samples, compress_samples, publish_samples = [], [], []
            frame_count = window_seq = 0
            while not self._stop.is_set() and not (self.root / "stop").exists():
                if doom.poll() is not None or capture.poll() is not None:
                    raise RuntimeError("game/capture process exited")
                for _ in range(16):
                    try:
                        event = self.events.get_nowait()
                    except queue.Empty:
                        break
                    self._apply(event)
                render_start = time.monotonic()
                pixels = _read_exact(capture.stdout, FRAME_BYTES)
                render_ms = (time.monotonic() - render_start) * 1000
                if len(pixels) != FRAME_BYTES:
                    raise RuntimeError("game RGB frame size invalid")
                with self._lock:
                    applied = self._applied_seq
                ppm = FRAME_PREFIX + pixels
                compress_start = time.monotonic()
                packed_zlib = zlib.compress(ppm, level=1)
                compress_ms = (time.monotonic() - compress_start) * 1000
                if not 0 < len(packed_zlib) <= 2 * 1024 * 1024:
                    raise RuntimeError("game frame exceeds bound")
                frame = FRAME_HEADER.pack(b"CCB1", seq, applied, WIDTH, HEIGHT) + packed_zlib
                publish_start = time.monotonic()
                atomic_bytes(self.root / f"frame-ring-{seq % RING_SLOTS:02d}.bin", frame)
                atomic_bytes(self.root / "frame.bin", frame)
                publish_ms = (time.monotonic() - publish_start) * 1000
                if seq == 0:
                    _status(self.root, run_id=self.run_id, state="ready",
                            device="cuda", gpu=gpu,
                            width=WIDTH, height=HEIGHT, application="freedoom")
                render_samples.append(render_ms)
                compress_samples.append(compress_ms)
                publish_samples.append(publish_ms)
                frame_count += 1
                now = time.monotonic()
                if now - metrics_since >= 1.0:
                    elapsed = now - metrics_since
                    window_seq += 1
                    record = {
                        "schema": 1, "run_id": self.run_id,
                        "seq": seq, "window_seq": window_seq,
                        "published_count": frame_count,
                        "window_elapsed_ms": round(elapsed * 1000, 3),
                        "target_fps": FPS, "rate_seq": 0,
                        "renderer_fps": round(frame_count / elapsed, 2),
                        "render_copy_ms": round(median(render_samples), 2),
                        "render_copy_p95_ms": _p95(render_samples),
                        "compress_ms": round(median(compress_samples), 2),
                        "compress_p95_ms": _p95(compress_samples),
                        "publish_ms": round(median(publish_samples), 2),
                        "publish_p95_ms": _p95(publish_samples),
                    }
                    atomic_bytes(
                        self.root / "render-metrics.json",
                        (json.dumps(record, separators=(",", ":")) + "\n").encode())
                    metrics_since = now
                    frame_count = 0
                    render_samples.clear()
                    compress_samples.clear()
                    publish_samples.clear()
                seq += 1
            _status(self.root, run_id=self.run_id, state="stopped",
                    device="cuda", gpu=gpu, application="freedoom")
        except Exception as exc:
            # Never include arbitrary exception strings, paths, or credentials.
            # A shutdown intentionally interrupting the pipe is not a crash.
            if not self._stop.is_set():
                _status(self.root, run_id=self.run_id, state="failed",
                        error_type=type(exc).__name__, application="freedoom")
        finally:
            self._release_all()
            for proc in reversed(self._processes):
                try:
                    _stop_proc(proc)
                except Exception:
                    pass
            self._stop.set()

    def start(self):
        if self._thread is not None:
            raise ValueError("game already started")
        self.root.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._worker, name="colabcola-freedoom", daemon=True)
        self._thread.start()
        return self._thread

    def close(self):
        self._stop.set()
        for proc in reversed(self._processes):
            try:
                _stop_proc(proc)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2)


def start(directory, fps, run_id, *, sequence_buffered=False):
    global _CONTROLLER
    if _CONTROLLER is not None:
        raise ValueError("Freedoom worker already active")
    instance = FreedoomSource(directory, run_id, fps, sequence_buffered)
    _CONTROLLER = instance
    return instance.start()


def submit_input(event):
    if _CONTROLLER is None:
        raise ValueError("Freedoom worker not initialized")
    return _CONTROLLER.submit_input(event)


def stop():
    if _CONTROLLER is not None:
        _CONTROLLER.close()
