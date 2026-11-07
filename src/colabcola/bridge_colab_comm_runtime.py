"""Kernel-side Comm adapter for the experimental Colab Cola bridge.

Runs inside the Colab-owned Jupyter kernel. The existing CUDA renderer writes
a latest-frame file in the runtime; Jupyter Comm messages carry frames and
inputs over a single authenticated kernel WebSocket instead of HTTP polling.
There is no public listening socket, tunnel, or remote proxy.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
import runpy
import time
import os

BOOTSTRAP_STAGE = "imports"


def _create_kernel_comm(*, target_name, data):
    """Use the Colab-documented ipykernel Comm API, with a modern fallback."""
    try:
        from ipykernel import comm as kernel_comm
        comm_class = getattr(kernel_comm, "Comm", None)
        if comm_class is not None:
            return comm_class(target_name=target_name, data=data)
    except ImportError:
        pass
    from comm import create_comm
    return create_comm(target_name=target_name, data=data)


def _record_bootstrap_failure(directory, run_id, error):
    """Persist only a known stage and exception type, never arbitrary traceback."""
    try:
        safe_stage = BOOTSTRAP_STAGE if BOOTSTRAP_STAGE in {
            "imports", "loading_renderer", "starting_renderer", "waiting_cuda",
            "creating_comm", "registering_handler", "ready",
        } else "unknown"
        name = type(error).__name__
        if not isinstance(name, str) or not name.isidentifier() or len(name) > 80:
            name = "Exception"
        payload = {"schema": 1, "run_id": run_id, "state": "failed",
                   "stage": safe_stage, "error_type": name}
        destination = Path(directory) / "bridge-bootstrap-diagnostic.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    except Exception:
        pass

MAX_FRAME_BYTES = 2 * 1024 * 1024 + 20
TARGET = "colabcola.bridge.v1"


def start(directory: str, run_id: str, fps: int, renderer_path: str):
    global BOOTSTRAP_STAGE
    root = Path(directory)
    BOOTSTRAP_STAGE = "loading_renderer"
    renderer = runpy.run_path(renderer_path)
    BOOTSTRAP_STAGE = "starting_renderer"
    renderer["start"](directory, fps, run_id)
    BOOTSTRAP_STAGE = "waiting_cuda"
    deadline = time.monotonic() + 45
    details = None
    while time.monotonic() < deadline:
        try:
            details = json.loads((root / "status.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            time.sleep(.15)
            continue
        if details.get("run_id") != run_id:
            raise RuntimeError("remote bridge identity mismatch")
        if details.get("state") == "failed":
            raise RuntimeError("GPU renderer initialization failed")
        if details.get("state") == "ready":
            break
        time.sleep(.15)
    else:
        raise RuntimeError("CUDA bridge worker did not become ready")
    if details.get("device") != "cuda" or "T4" not in details.get("gpu", ""):
        raise RuntimeError("remote bridge did not verify T4 CUDA compute")

    BOOTSTRAP_STAGE = "creating_comm"
    comm = _create_kernel_comm(target_name=TARGET, data={
        "v": 1, "kind": "ready", "run_id": run_id,
        "gpu": details["gpu"], "device": "cuda", "width": details["width"],
        "height": details["height"],
    })
    state = {"latest_input": 0, "latest_rate": 0, "last_frame": -1, "closed": False}

    def on_message(message):
        event = message.get("content", {}).get("data", {})
        if (not isinstance(event, dict) or event.get("v") != 1
                or event.get("run_id") != run_id or state["closed"]):
            return
        kind = event.get("kind")
        if kind == "input":
            sequence = event.get("seq")
            action = event.get("event")
            if type(sequence) is not int or not state["latest_input"] < sequence < 10000000:
                return
            if action == "mouse":
                x, y = event.get("x"), event.get("y")
                if (type(x) not in (int, float) or type(y) not in (int, float)
                        or not 0 <= x <= 1 or not 0 <= y <= 1):
                    return
                payload = {"event": "mouse", "x": x, "y": y}
            elif action == "key":
                key = event.get("key")
                if key not in ("Left", "Right", "Up", "Down"):
                    return
                payload = {"event": "key", "key": key}
            else:
                return
            payload.update(run_id=run_id, seq=sequence)
            renderer["atomic_bytes"](
                root / "input.json", (json.dumps(payload, separators=(",", ":")) + "\n").encode()
            )
            state["latest_input"] = sequence
        elif kind == "rate":
            request_seq, requested_fps = event.get("seq"), event.get("fps")
            if (type(request_seq) is not int
                    or not state["latest_rate"] < request_seq < 10000
                    or type(requested_fps) is not int or not 1 <= requested_fps <= 12):
                return
            renderer["atomic_bytes"](
                root / "rate-control.json",
                (json.dumps({"run_id": run_id, "seq": request_seq,
                             "fps": requested_fps}, separators=(",", ":")) + "\n").encode()
            )
            state["latest_rate"] = request_seq
            # ACK means command accepted and durably written, not yet applied.
            # The renderer's rate_seq metric confirms later application.
            comm.send({"v": 1, "kind": "rate_ack", "run_id": run_id,
                       "seq": request_seq, "fps": requested_fps})
        elif kind == "ping":
            # Small payload, same authenticated shell→IOPub Comm path as frames.
            # No frame reads, CUDA work, or additional sockets.
            ping_seq = event.get("seq")
            if type(ping_seq) is int and 0 < ping_seq <= 10000:
                comm.send({"v": 1, "kind": "pong", "run_id": run_id,
                           "seq": ping_seq})
        elif kind == "pull":
            try:
                body = (root / "frame.bin").read_bytes()
                if not 20 < len(body) <= MAX_FRAME_BYTES:
                    return
                # A frame is pushed only when the host requests it (bounded credit).
                # Stats are a small, optional sidecar; never block a frame if
                # the worker is updating it concurrently.
                metrics = None
                try:
                    candidate = json.loads((root / "render-metrics.json").read_text(encoding="utf-8"))
                    if (isinstance(candidate, dict) and candidate.get("schema") == 1
                            and candidate.get("run_id") == run_id):
                        metrics = candidate
                except (FileNotFoundError, OSError, UnicodeError, ValueError):
                    pass
                comm.send({"v": 1, "kind": "frame", "run_id": run_id,
                           "data": base64.b64encode(body).decode("ascii"),
                           "metrics": metrics})
            except (OSError, ValueError):
                return
        elif kind == "close":
            state["closed"] = True
            renderer["atomic_bytes"](root / "stop", b"1")
            comm.close()

    BOOTSTRAP_STAGE = "registering_handler"
    comm.on_msg(on_message)
    BOOTSTRAP_STAGE = "ready"
    return comm


if __name__ == "__main__":
    try:
        start(BRIDGE_DIRECTORY, BRIDGE_RUN_ID, BRIDGE_FPS, BRIDGE_RENDERER_PATH)
    except BaseException as exc:
        _record_bootstrap_failure(BRIDGE_DIRECTORY, BRIDGE_RUN_ID, exc)
        raise
