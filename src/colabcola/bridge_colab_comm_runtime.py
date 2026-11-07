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
from comm import create_comm

MAX_FRAME_BYTES = 2 * 1024 * 1024 + 20
TARGET = "colabcola.bridge.v1"


def start(directory: str, run_id: str, fps: int, renderer_path: str):
    root = Path(directory)
    renderer = runpy.run_path(renderer_path)
    renderer["start"](directory, fps, run_id)
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

    comm = create_comm(target_name=TARGET, data={
        "v": 1, "kind": "ready", "run_id": run_id,
        "gpu": details["gpu"], "device": "cuda", "width": details["width"],
        "height": details["height"],
    })
    state = {"latest_input": 0, "last_frame": -1, "closed": False}

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
        elif kind == "pull":
            try:
                body = (root / "frame.bin").read_bytes()
                if not 20 < len(body) <= MAX_FRAME_BYTES:
                    return
                # A frame is pushed only when the host requests it (bounded credit).
                comm.send({"v": 1, "kind": "frame", "run_id": run_id,
                           "data": base64.b64encode(body).decode("ascii")})
            except (OSError, ValueError):
                return
        elif kind == "close":
            state["closed"] = True
            renderer["atomic_bytes"](root / "stop", b"1")
            comm.close()

    comm.on_msg(on_message)
    return comm


if __name__ == "__main__":
    start(BRIDGE_DIRECTORY, BRIDGE_RUN_ID, BRIDGE_FPS, BRIDGE_RENDERER_PATH)
