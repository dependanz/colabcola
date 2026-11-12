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


def start(directory: str, run_id: str, fps: int, renderer_path: str,
          video_source_path: str | None = None, preflight_video: bool = False,
          install_remote_pyav: bool = False,
          sequence_buffered: bool = False,
          max_buffer_lag_frames: int = 2,
          video_push: bool = False, video_push_source_path: str | None = None,
          push_credits: int = 4, producer_zlib_level0: bool = False,
          application: str = "synthetic",
          game_renderer_path: str | None = None):
    global BOOTSTRAP_STAGE
    if (type(video_push) is not bool
            or (video_push and (not sequence_buffered or not preflight_video
                                or video_push_source_path is None))):
        raise ValueError("push video requires sequence-buffered H.264")
    if (type(push_credits) is not int or push_credits not in (4, 6, 8, 10, 12)
            or (not video_push and push_credits != 4)
            or (video_push and
                ((fps == 30 and push_credits not in (4, 6, 8))
                 or (fps == 60 and push_credits not in (8, 10, 12))
                 or fps not in (30, 60)))):
        raise ValueError("invalid sender FPS/credit pairing")
    if (type(producer_zlib_level0) is not bool
            or (producer_zlib_level0 and not (
                fps == 60 and video_push and sequence_buffered and preflight_video))):
        raise ValueError("level0 compression requires 60Hz buffered H.264 push")
    if (application not in ("synthetic", "freedoom", "inputlab")
            or (application in ("freedoom", "inputlab") and not (
                fps == 30 and video_push and sequence_buffered and preflight_video
                and game_renderer_path is not None
                and push_credits in (4, 6, 8) and not producer_zlib_level0))):
        raise ValueError("game/input lab requires opt-in 30Hz external H.264 push")
    if type(sequence_buffered) is not bool or (sequence_buffered and not preflight_video):
        raise ValueError("sequence buffer requires video preflight")
    if (type(max_buffer_lag_frames) is not int
            or not 0 <= max_buffer_lag_frames <= 2
            or (not sequence_buffered and max_buffer_lag_frames != 2)):
        raise ValueError("invalid renderer lag mode")
    if install_remote_pyav and (not preflight_video or video_source_path is None):
        raise ValueError("remote PyAV installation requires video preflight")
    root = Path(directory)
    BOOTSTRAP_STAGE = "loading_renderer"
    renderer = runpy.run_path(
        game_renderer_path if application in ("freedoom", "inputlab") else renderer_path)
    BOOTSTRAP_STAGE = "starting_renderer"
    if sequence_buffered:
        renderer["start"](directory, fps, run_id, sequence_buffered=True,
                          **({"zlib_level": 0} if producer_zlib_level0 else {}))
    else:
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
    if (application in ("freedoom", "inputlab")
            and details.get("application") != application):
        raise RuntimeError("remote game did not become ready")
    if details.get("device") != "cuda" or "T4" not in details.get("gpu", ""):
        raise RuntimeError("remote bridge did not verify T4 CUDA compute")

    BOOTSTRAP_STAGE = "creating_comm"
    video_source = None
    video_preflight_error = None
    if preflight_video:
        if video_source_path is None:
            video_preflight_error = "source_path_missing"
        else:
            try:
                # A video dependency failure should not prevent the owned
                # Jupyter Comm from reporting a sanitized diagnostic.
                source = runpy.run_path(video_source_path)
                if install_remote_pyav:
                    source["provision_remote_pyav"]()
                video_source = source["RemoteH264PullSource"](
                    root, run_id, fps=fps, width=640, height=360,
                    **({"sequence_buffered": True,
                         "max_buffer_lag_frames": max_buffer_lag_frames}
                        if sequence_buffered else {}))
            except Exception as exc:
                # Match ONLY our own fixed messages. Never send arbitrary
                # exception content, paths, tokens or tracebacks over Comm.
                safe_source_errors = {
                    ("remote PyAV unavailable",): "pyav_unavailable",
                    ("remote NumPy unavailable",): "numpy_unavailable",
                    ("remote libx264 encoder unavailable",): "libx264_unavailable",
                    ("remote PyAV installation failed",): "pyav_install_failed",
                    ("remote PyAV installation timed out",): "pyav_install_timeout",
                }
                video_preflight_error = (
                    safe_source_errors.get(exc.args, "video_initialization_failed")
                    if type(exc).__name__ == "RemoteVideoSourceError"
                    and isinstance(exc.args, tuple) and all(isinstance(a, str) for a in exc.args)
                    else "video_initialization_failed")
    ready = {
        "v": 1,
        "kind": "video_preflight_failed" if video_preflight_error else "ready",
        "run_id": run_id,
        "gpu": details["gpu"], "device": "cuda", "width": details["width"],
        "height": details["height"],
    }
    if video_preflight_error:
        ready["failure_code"] = video_preflight_error
    elif video_source_path is not None:
        ready["video_codec"] = "h264-annexb"
        if video_push:
            ready["video_mode"] = "credit-push"
            if application in ("freedoom", "inputlab"):
                ready["application"] = application
            if producer_zlib_level0:
                ready["producer_zlib_level"] = 0
    comm = _create_kernel_comm(target_name=TARGET, data=ready)
    state = {"latest_input": 0, "latest_rate": 0, "last_frame": -1,
             "last_video_pull": 0, "closed": False}
    video_failed = video_preflight_error is not None
    push_sender = None
    if video_push and not video_failed:
        try:
            push_impl = runpy.run_path(video_push_source_path)
            push_sender = push_impl["CreditPushSender"](
                comm=comm, source=video_source, directory=root,
                run_id=run_id, credits=push_credits,
                **({"enable_status_sidecar": True}
                   if application == "inputlab" else {}))
        except Exception:
            video_failed = True

    def on_message(message):
        nonlocal video_source, video_failed
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
            if application in ("freedoom", "inputlab"):
                # Both applications validate a small, explicit input schema.
                # Never forward unrestricted command or filesystem fields.
                record = {"seq": sequence, "event": action}
                if action in ("key_down", "key_up"):
                    record["key"] = event.get("key")
                elif action in ("mouse_button_down", "mouse_button_up"):
                    record["button"] = event.get("button")
                    if application == "inputlab":
                        record["x"], record["y"] = event.get("x"), event.get("y")
                elif action == "mouse_delta":
                    record["dx"], record["dy"] = event.get("dx"), event.get("dy")
                elif application == "inputlab" and action == "mouse_move":
                    record["x"], record["y"] = event.get("x"), event.get("y")
                elif application == "inputlab" and action == "mouse_wheel":
                    record["ticks"] = event.get("ticks")
                elif application == "inputlab" and action == "pointer_lock":
                    record["enabled"] = event.get("enabled")
                else:
                    return
                try:
                    if not renderer["submit_input"](record):
                        raise ValueError("game input queue rejected")
                except (ValueError, TypeError, KeyError):
                    comm.send({"v": 1, "kind": "error", "run_id": run_id,
                               "code": "application_input_rejected"})
                    return
                state["latest_input"] = sequence
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
        elif kind == "rate" and application == "synthetic":
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
            pull_seq = event.get("pull_seq")
            if (pull_seq is not None and
                    (type(pull_seq) is not int or not 1 <= pull_seq <= 100000000)):
                return
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
                reply = {"v": 1, "kind": "frame", "run_id": run_id,
                         "data": base64.b64encode(body).decode("ascii"),
                         "metrics": metrics}
                if pull_seq is not None:
                    reply["pull_seq"] = pull_seq
                comm.send(reply)
            except (OSError, ValueError):
                return
        elif kind == "video_credit" and video_push:
            if push_sender is None or video_failed:
                comm.send({"v": 1, "kind": "error", "run_id": run_id,
                           "code": "push_video_unavailable"})
                return
            try:
                # Credit grants must be strict and bounded. A video sender
                # never issues a request-response pull or resets the codec.
                push_sender.grant(event.get("count"))
                if application == "inputlab":
                    # Only a valid, authenticated credit extends the remote
                    # producer lease. Input events do not extend this lease.
                    renderer["note_host_credit"]()
            except (ValueError, TypeError):
                comm.send({"v": 1, "kind": "error", "run_id": run_id,
                           "code": "push_credit_invalid"})
        elif kind == "video_pull" and video_source_path is not None and not video_push:
            # Only an explicitly bootstrapped video session permits these
            # messages. Existing PPM Comm sessions never instantiate PyAV.
            pull_seq = event.get("pull_seq")
            if (type(pull_seq) is not int
                    or not state["last_video_pull"] < pull_seq <= 100000000):
                return
            state["last_video_pull"] = pull_seq
            try:
                if video_failed:
                    raise RuntimeError("video source already failed")
                if video_source is None:
                    source = runpy.run_path(video_source_path)
                    video_source = source["RemoteH264PullSource"](
                        root, run_id, fps=fps, width=640, height=360,
                    **({"sequence_buffered": True,
                         "max_buffer_lag_frames": max_buffer_lag_frames}
                        if sequence_buffered else {}))
                reply = video_source.pull(pull_seq)
                comm.send(reply)
            except Exception:
                # No content/traceback: never leak kernel/provider secrets.
                # The host's one-credit receiver fails closed on kind=error.
                video_failed = True
                comm.send({"v": 1, "kind": "error", "run_id": run_id,
                           "pull_seq": pull_seq, "code": "video_source_unavailable"})
        elif kind == "close":
            state["closed"] = True
            if push_sender is not None:
                push_sender.close()
            renderer["atomic_bytes"](root / "stop", b"1")
            if application in ("freedoom", "inputlab"):
                renderer["stop"]()
            comm.close()

    BOOTSTRAP_STAGE = "registering_handler"
    comm.on_msg(on_message)
    BOOTSTRAP_STAGE = "ready"
    if push_sender is not None:
        push_sender.start()
    return comm


if __name__ == "__main__":
    try:
        start(BRIDGE_DIRECTORY, BRIDGE_RUN_ID, BRIDGE_FPS, BRIDGE_RENDERER_PATH,
              video_source_path=globals().get("BRIDGE_VIDEO_SOURCE_PATH"),
              preflight_video=globals().get("BRIDGE_VIDEO_PREFLIGHT", False) is True,
              install_remote_pyav=globals().get("BRIDGE_INSTALL_REMOTE_PYAV", False) is True,
              sequence_buffered=globals().get("BRIDGE_SEQUENCE_BUFFERED", False) is True,
              max_buffer_lag_frames=globals().get("BRIDGE_MAX_BUFFER_LAG_FRAMES", 2),
              video_push=globals().get("BRIDGE_VIDEO_PUSH", False) is True,
              video_push_source_path=globals().get("BRIDGE_VIDEO_PUSH_SOURCE_PATH"),
              push_credits=globals().get("BRIDGE_PUSH_CREDITS", 4),
              producer_zlib_level0=globals().get("BRIDGE_PRODUCER_ZLIB_LEVEL0", False),
              application=globals().get("BRIDGE_APPLICATION", "synthetic"),
              game_renderer_path=globals().get("BRIDGE_GAME_RENDERER_PATH"))
    except BaseException as exc:
        _record_bootstrap_failure(BRIDGE_DIRECTORY, BRIDGE_RUN_ID, exc)
        raise
