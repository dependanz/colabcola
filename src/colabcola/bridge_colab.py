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
from .bridge_telemetry import BridgeTransferMeter
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


def decode_remote_file_record(record: object) -> bytes:
    """Decode a Jupyter Contents binary file with bounded, safe diagnostics.

    Some Contents implementations fold base64 into multiple lines. Allow only
    ASCII base64 whitespace, then run strict validation on the normalized data.
    Never log content, endpoints, headers, or authentication values.
    """
    if not isinstance(record, dict):
        raise LifecycleError("remote frame response invalid: not a JSON object")
    if record.get("type") != "file" or record.get("format") != "base64":
        raise LifecycleError("remote frame response invalid: expected file/base64")
    data = record.get("content")
    if not isinstance(data, str):
        raise LifecycleError("remote frame response invalid: missing or non-string content")
    max_encoded = ((_MAX_FRAME_BYTES + _FRAME_HEADER.size + 2) // 3) * 4
    if len(data) > max_encoded + 8192:
        raise LifecycleError("remote frame response exceeds encoded size limit")
    clean = "".join(data.split())
    whitespace = len(data) - len(clean)
    if len(clean) > max_encoded:
        raise LifecycleError("remote frame response exceeds encoded size limit")
    try:
        content = base64.b64decode(clean, validate=True)
    except (ValueError, TypeError) as exc:
        raise LifecycleError(
            "remote frame Base64 invalid "
            + f"(characters={len(data)}, whitespace={whitespace}, "
            + f"error_type={type(exc).__name__})"
        ) from None
    if not 0 < len(content) <= _MAX_FRAME_BYTES + _FRAME_HEADER.size:
        raise LifecycleError("remote frame decoded size invalid")
    return content


def remote_file(provider, session, location: str) -> bytes:
    return decode_remote_file_record(provider._contents(session, "GET", location))


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
        if not endpoint and not recorded and provider.assignments():
            raise LifecycleError("allocation outcome uncertain; active assignment exists")
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
    meter = BridgeTransferMeter()
    start = time.monotonic()
    last_stats = start
    seen = 0
    transferred = 0
    last_frame_at = start
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
                upload_started = time.monotonic()
                meter.sent(seq, upload_started)
                provider._upload(session, remote + "/input.json",
                                 (json.dumps(record, separators=(",", ":")) + "\n").encode())
                meter.record("put", time.monotonic() - upload_started)
            fetch_started = time.monotonic()
            try:
                packed = remote_file(provider, session, remote + "/frame.bin")
            except FileNotFoundError:
                if time.monotonic() - last_frame_at >= 30:
                    status = provider._json_file(session, remote + "/status.json")
                    if status and status.get("state") == "failed":
                        raise LifecycleError("remote renderer failed: " + str(status.get("message", "unknown"))[:160])
                    raise LifecycleError("remote renderer published no frame within 30 seconds")
                time.sleep(0.15)
                continue
            meter.record("get", time.monotonic() - fetch_started)
            decode_started = time.monotonic()
            frame_seq, ack, width, height, ppm = unpack_frame(packed)
            meter.record("decode", time.monotonic() - decode_started)
            if width != 640 or height != 360:
                raise LifecycleError("unexpected remote frame dimensions")
            if frame_seq > last_seq:
                last_seq = frame_seq
                last_frame_at = time.monotonic()
                seen += 1
                transferred += len(packed)
                # Only keep the newest frame; never allow a slow window to build latency.
                offer_update(frames, ("frame", ppm))
            elif time.monotonic() - last_frame_at >= 30:
                status = provider._json_file(session, remote + "/status.json")
                if status and status.get("state") == "failed":
                    raise LifecycleError("remote renderer failed: " + str(status.get("message", "unknown"))[:160])
                raise LifecycleError("remote renderer frame is stale for 30 seconds")
            now = time.monotonic()
            meter.received_ack(ack, now)
            if now - last_stats >= 2:
                elapsed = max(now - last_stats, .001)
                timings = meter.flush()
                text = f"{gpu} · {seen / elapsed:.1f} FPS · {transferred / elapsed / 1024:.0f} KiB/s"
                for stage in ("get", "put", "decode"):
                    value = timings[stage]
                    if value is not None:
                        text += f" · {stage.upper()} p50 {value} ms"
                if timings["ack"] is not None:
                    text += f" · input ACK observed p50 {timings['ack']} ms"
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
        raise LifecycleError("remote interactive transport stopped: " + type(exc).__name__) from None
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
                    transport = getattr(args, "transport", "files")
                    if transport == "comm":
                        # Opt-in persistent Jupyter kernel websocket.
                        # The proven Contents-API polling route remains the default.
                        from .bridge_comm import KernelCommBridge, stream_comm
                        peer = KernelCommBridge(provider, session, state["run_id"], args.fps)
                        try:
                            details = peer.open()
                            save(args.run_dir, state, "running", gpu_name=details["gpu"],
                                 transport="comm")
                            print("[colabcola] Remote GPU verified: " + details["gpu"], flush=True)
                            if getattr(args, "benchmark_sweep", False):
                                from .bridge_sweep import run_rate_sweep
                                def work(outgoing, frames, stopped):
                                    run_rate_sweep(
                                        peer, details["gpu"], outgoing, frames, stopped,
                                        deadline, args.benchmark_phase_seconds)
                            else:
                                def work(outgoing, frames, stopped):
                                    stream_comm(
                                        peer, details["gpu"], outgoing, frames,
                                        stopped, deadline, args.poll_hz,
                                        probe_link=getattr(args, "probe_link", False),
                                        in_flight=getattr(args, "inflight", 1))
                            run_viewer("127.0.0.1", 0, "", width=640, height=360,
                                       title="Colab Cola · T4 (kernel Comm)",
                                       worker=work)
                        finally:
                            peer.close()
                    else:
                        remote, details = start_worker(provider, session, state["run_id"], args.fps)
                        save(args.run_dir, state, "running", gpu_name=details["gpu"],
                             transport="files")
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
                    # Report verified release even if the workload itself failed.
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
