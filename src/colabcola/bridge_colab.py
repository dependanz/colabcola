"""Experimental managed Colab T4 bridge over authenticated Jupyter Contents API.

This is a bounded interactive research proof, not a general remote desktop
or low-latency video transport. It deliberately does not modify managed jobs.
"""
from __future__ import annotations

import base64
import importlib.resources
import json
from pathlib import Path
import queue
import re
import struct
import threading
import time
import uuid
import zlib

from .bridge_demo import run_viewer
from .lifecycle_io import LifecycleError, atomic
from .lifecycle_provider import GoogleProvider
from .lifecycle import run_lock

_FRAME_HEADER = struct.Struct("!4sIIII")
_MAX_FRAME_BYTES = 2 * 1024 * 1024
_MAX_UNCOMPRESSED = 3 * 1024 * 768 + 64


def unpack_frame(body: bytes) -> tuple[int, int, int, int, bytes]:
    """Bounded decoding: never trust a remotely supplied decompression size."""
    if not isinstance(body, bytes) or not _FRAME_HEADER.size < len(body) <= _MAX_FRAME_BYTES + _FRAME_HEADER.size:
        raise LifecycleError("invalid remote frame size")
    magic, seq, ack, width, height = _FRAME_HEADER.unpack(body[:_FRAME_HEADER.size])
    if magic != b"CCB1" or not 1 <= width <= 1024 or not 1 <= height <= 768:
        raise LifecycleError("invalid remote frame header")
    expected_header = f"P6\n{width} {height}\n255\n".encode("ascii")
    expected_size = len(expected_header) + width * height * 3
    decoder = zlib.decompressobj()
    pixels = decoder.decompress(body[_FRAME_HEADER.size:], expected_size + 1)
    if (len(pixels) != expected_size or not decoder.eof or decoder.unconsumed_tail
            or decoder.unused_data or not pixels.startswith(expected_header)):
        raise LifecycleError("remote frame decompression failed integrity checks")
    return seq, ack, width, height, pixels


def remote_file(provider, session, location: str) -> bytes:
    record = provider._contents(session, "GET", location)
    if record.get("type") != "file" or record.get("format") != "base64":
        raise LifecycleError("bridge transport expected a binary remote file")
    try:
        return base64.b64decode(record["content"], validate=True)
    except (ValueError, KeyError) as exc:
        raise LifecycleError("remote frame encoding invalid") from exc


def read_journal(directory: Path) -> dict:
    path = directory / "state.json"
    if path.is_symlink() or not path.is_file():
        raise LifecycleError("bridge run has no regular state.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (data.get("schema") != 1 or data.get("backend") != "google-colab"
                or data.get("gpu") != "T4" or not re.fullmatch(r"[0-9a-f]{32}", data.get("run_id", ""))
                or not isinstance(data.get("allocation_id"), str)
                or not isinstance(data.get("profile"), str)):
            raise ValueError()
        uuid.UUID(data["allocation_id"])
        return data
    except (OSError, ValueError, TypeError, KeyError):
        raise LifecycleError("bridge journal invalid; refusing to release unknown resources") from None


def save(directory: Path, state: dict, phase: str, **fields) -> None:
    state.update(phase=phase, **fields)
    atomic(directory / "state.json", state)


def release_owned(provider: GoogleProvider, directory: Path, state: dict) -> bool:
    """Release only the allocation matching the saved immutable allocation id."""
    if state["phase"] == "released":
        return True
    try:
        endpoint = provider.reconcile(state["allocation_id"], "T4")
        recorded = state.get("endpoint")
        if endpoint and recorded and endpoint != recorded:
            raise LifecycleError("allocation mismatch; refusing to release unrelated runtime")
        if not endpoint and recorded and provider.lookup(recorded):
            raise LifecycleError("saved runtime remains assigned but allocation reconciliation failed")
        if endpoint:
            save(directory, state, "releasing", endpoint=endpoint)
            provider.release(endpoint)
        save(directory, state, "released", released_at=time.time())
        return True
    except Exception as exc:
        save(directory, state, "needs_cleanup", cleanup_error=type(exc).__name__)
        print("[colabcola] Release not confirmed; use 'colabcola bridge cleanup --run-dir ...'", flush=True)
        return False


def verify_preflight(provider: GoogleProvider) -> None:
    assignments = provider.assignments()
    if assignments:
        raise LifecycleError(
            "active Colab assignments exist; refusing to allocate another T4 for the experimental bridge"
        )


def wait_for_session(provider: GoogleProvider, endpoint: str, deadline: float):
    while time.monotonic() < deadline:
        try:
            return provider._session(endpoint)
        except Exception:
            time.sleep(2)
    raise LifecycleError("assigned Colab runtime proxy did not become available")


def start_worker(provider: GoogleProvider, session, run_id: str, fps: int) -> tuple[str, dict]:
    remote = f"content/colabcola-bridge/{run_id}"
    root = "/" + remote
    # Ensure directory exists before REST file upload. The executable payload is
    # packaged in this distribution and uploaded only to our allocated runtime.
    provider._execute(session, f"from pathlib import Path\nPath({root!r}).mkdir(parents=True, exist_ok=True)")
    source = importlib.resources.files("colabcola").joinpath("bridge_colab_runtime.py").read_bytes()
    provider._upload(session, remote + "/bridge_colab_runtime.py", source)
    code = ("import runpy\n"
            f"runpy.run_path({root + '/bridge_colab_runtime.py'!r}, run_name='__main__', "
            f"init_globals={{'BRIDGE_DIRECTORY':{root!r},'BRIDGE_FPS':{fps},"
            f"'BRIDGE_RUN_ID':{run_id!r}}})\n")
    # An uncertain kernel execution reply must not trigger resubmission.
    provider._execute(session, code)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        status = provider._json_file(session, remote + "/status.json")
        if status and status.get("run_id") == run_id:
            if status.get("state") == "failed":
                raise LifecycleError("remote GPU worker failed: " + str(status.get("message", "unknown"))[:160])
            if (status.get("state") == "ready" and status.get("device") == "cuda"
                    and "T4" in status.get("gpu", "") and status.get("width") == 640
                    and status.get("height") == 360):
                return remote, status
        time.sleep(1)
    raise LifecycleError("remote GPU worker did not become ready")


def offer_update(items: queue.Queue, item: tuple[str, object]) -> None:
    """Do not block transport if a UI thread is temporarily too slow."""
    try:
        items.put_nowait(item)
    except queue.Full:
        try:
            items.get_nowait()
        except queue.Empty:
            pass
        try:
            items.put_nowait(item)
        except queue.Full:
            pass


def stream_remote(provider, session, remote: str, run_id: str, gpu: str,
                  outgoing: queue.Queue, frames: queue.Queue,
                  stopped: threading.Event, deadline: float, poll_hz: int) -> None:
    """Conservative single-threaded request loop for the first real-GPU proof."""
    seq = 0
    last_seq = -1
    acked = 0
    pending = {}
    start = time.monotonic()
    last_stats = start
    seen = 0
    transferred = 0
    print("[colabcola] Interactive T4 bridge connected; close the viewer to release compute", flush=True)
    try:
        offer_update(frames, ("status", f"Connected · {gpu} · remote rendering"))
        while not stopped.is_set() and time.monotonic() < deadline:
            tick = time.monotonic()
            event = None
            while True:
                try:
                    event = outgoing.get_nowait()
                except queue.Empty:
                    break
            if event is not None:
                seq += 1
                record = {"run_id": run_id, "seq": seq, **event}
                provider._upload(session, remote + "/input.json",
                                 (json.dumps(record, separators=(",", ":")) + "\n").encode())
                pending[seq] = time.monotonic()
            try:
                packed = remote_file(provider, session, remote + "/frame.bin")
            except FileNotFoundError:
                time.sleep(0.15)
                continue
            frame_seq, ack, width, height, ppm = unpack_frame(packed)
            if width != 640 or height != 360:
                raise LifecycleError("unexpected remote frame dimensions")
            if frame_seq > last_seq:
                last_seq = frame_seq
                seen += 1
                transferred += len(packed)
                # Only keep the newest frame; never allow a slow window to build latency.
                offer_update(frames, ("frame", ppm))
            now = time.monotonic()
            latency = None
            if ack > acked and ack in pending:
                latency = round(1000 * (now - pending[ack]))
                acked = ack
                pending = {n: sent for n, sent in pending.items() if n > acked}
            if now - last_stats >= 2:
                elapsed = max(now - last_stats, .001)
                text = f"{gpu} · {seen / elapsed:.1f} FPS · {transferred / elapsed / 1024:.0f} KiB/s"
                if latency is not None:
                    text += f" · input-to-frame {latency} ms"
                offer_update(frames, ("status", text))
                print("[colabcola] " + text, flush=True)
                seen = 0
                transferred = 0
                last_stats = now
            time.sleep(max(0, 1 / poll_hz - (time.monotonic() - tick)))
        if time.monotonic() >= deadline:
            print("[colabcola] Interactive session time budget reached", flush=True)
    except Exception as exc:
        print("[colabcola] Interactive bridge stopped: " + type(exc).__name__ + ": " + str(exc)[:180],
              flush=True)
    finally:
        stopped.set()


def run_colab(args) -> int:
    from .profiles import selected
    with selected(args.profile):
        provider = GoogleProvider()
        try:
            verify_preflight(provider)
            if args.preflight_only:
                print("[colabcola] Bridge preflight passed; no runtime allocated", flush=True)
                return 0
            if not args.confirm_allocation:
                raise LifecycleError("this command allocates a paid/limited T4; pass --confirm-allocation")
            if args.run_dir.exists():
                raise LifecycleError("bridge run directory already exists; use a new one or bridge cleanup")
            args.run_dir.mkdir(parents=True, exist_ok=False)
            with run_lock(args.run_dir):
                state = {
                    "schema": 1, "backend": "google-colab", "gpu": "T4",
                    "profile": args.profile or "", "run_id": uuid.uuid4().hex,
                    "allocation_id": str(uuid.uuid4()), "created_at": time.time(),
                }
                save(args.run_dir, state, "allocating")
                deadline = time.monotonic() + args.max_seconds
                try:
                    print("[colabcola] Allocating managed Tesla T4...", flush=True)
                    endpoint = provider.allocate(state["allocation_id"], "T4")
                    save(args.run_dir, state, "allocated", endpoint=endpoint)
                    print("[colabcola] T4 allocated; starting remote CUDA renderer", flush=True)
                    session = wait_for_session(provider, endpoint, min(deadline, time.monotonic() + 90))
                    remote, details = start_worker(provider, session, state["run_id"], args.fps)
                    save(args.run_dir, state, "running", gpu_name=details["gpu"])
                    print("[colabcola] Remote GPU verified: " + details["gpu"], flush=True)
                    run_viewer("127.0.0.1", 0, "", width=640, height=360,
                               title="Colab Cola · Google Colab T4",
                               worker=lambda outgoing, frames, stopped:
                                   stream_remote(provider, session, remote, state["run_id"],
                                                 details["gpu"], outgoing, frames, stopped,
                                                 deadline, args.poll_hz))
                finally:
                    print("[colabcola] Releasing only the owned T4 runtime...", flush=True)
                    if not release_owned(provider, args.run_dir, state):
                        raise LifecycleError("owned T4 release not confirmed; run bridge cleanup")
                print("[colabcola] T4 release confirmed", flush=True)
                return 0
        finally:
            provider.close()


def cleanup_colab(args) -> int:
    from .profiles import selected
    with run_lock(args.run_dir):
        state = read_journal(args.run_dir)
        if state.get("profile", "") != (args.profile or ""):
            raise LifecycleError("cleanup profile differs from journal; pass the same --profile")
        with selected(args.profile):
            provider = GoogleProvider()
            try:
                if not release_owned(provider, args.run_dir, state):
                    raise LifecycleError("owned runtime release unconfirmed")
                print("[colabcola] Owned runtime confirmed absent", flush=True)
                return 0
            finally:
                provider.close()
