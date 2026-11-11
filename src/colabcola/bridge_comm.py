"""Persistent Jupyter Comm transport for Colab Cola's experimental T4 bridge.

One authenticated kernel WebSocket carries discrete input events and
credit-limited frame snapshots. It never exposes a server, adds a proxy,
or modifies the established batch execution and checkpoint code paths.
"""
from __future__ import annotations

import base64
import importlib.resources
import json
import queue
from statistics import median
import time
from typing import Any

from .bridge_colab import offer_update, unpack_frame
from .bridge_telemetry import BridgeTransferMeter
from .lifecycle_io import LifecycleError
from .lifecycle_kernel import safe_client_class
from .bridge_ws_probe import KernelWebSocketProbe, safe_error_type

TARGET = "colabcola.bridge.v1"
MAX_ENCODED_FRAME = 3 * 1024 * 1024


def validated_renderer_metrics(raw: object, run_id: str) -> dict | None:
    """Only accept bounded numeric timings tied to this owned remote session."""
    if not isinstance(raw, dict) or raw.get("schema") != 1 or raw.get("run_id") != run_id:
        return None
    seq = raw.get("seq")
    if type(seq) is not int or not 0 <= seq < 100000000:
        return None
    for key in ("renderer_fps", "render_copy_ms", "compress_ms", "publish_ms"):
        v = raw.get(key)
        if type(v) not in (int, float) or not 0 <= v <= 10000:
            return None
    if raw["renderer_fps"] > 120:
        return None
    result = {k: raw[k] for k in (
        "seq", "renderer_fps", "render_copy_ms", "compress_ms", "publish_ms")}
    if "target_fps" in raw or "rate_seq" in raw:
        target, rate_seq = raw.get("target_fps"), raw.get("rate_seq")
        if (type(target) is not int or not 1 <= target <= 12
                or type(rate_seq) is not int or not 0 <= rate_seq < 10000):
            return None
        result.update(target_fps=target, rate_seq=rate_seq)
    return result


_SAFE_BOOTSTRAP_STAGES = frozenset({
    "imports", "loading_renderer", "starting_renderer", "waiting_cuda",
    "creating_comm", "registering_handler", "ready",
})
_SAFE_ERROR_TYPES = frozenset({
    "ModuleNotFoundError", "ImportError", "NameError", "RuntimeError",
    "TypeError", "ValueError", "AttributeError", "FileNotFoundError",
    "OSError", "NotImplementedError", "SyntaxError", "KeyError",
    "PermissionError", "AssertionError", "SystemError",
})


def bootstrap_error_summary(provider, session, remote: str, run_id: str,
                            reply: object) -> str:
    """Return only vetted error classes and pipeline stages; never error values."""
    name = ""
    if isinstance(reply, dict):
        reply_type = reply.get("ename")
        if isinstance(reply_type, str) and reply_type in _SAFE_ERROR_TYPES:
            name = reply_type
    try:
        diagnostic = provider._json_file(
            session, remote + "/bridge-bootstrap-diagnostic.json")
    except Exception:
        diagnostic = None
    if (isinstance(diagnostic, dict) and diagnostic.get("schema") == 1
            and diagnostic.get("run_id") == run_id
            and diagnostic.get("state") == "failed"):
        stage = diagnostic.get("stage")
        if not isinstance(stage, str) or stage not in _SAFE_BOOTSTRAP_STAGES:
            stage = "unknown"
        error_type = diagnostic.get("error_type")
        if not isinstance(error_type, str) or error_type not in _SAFE_ERROR_TYPES:
            error_type = name or "RemoteException"
        return f"stage={stage}, error_type={error_type}"
    return "error_type=" + (name or "RemoteException")


class KernelCommBridge:
    def __init__(self, provider, session, run_id: str, fps: int,
                 video_codec: str = "ppm", *, install_remote_pyav: bool = False,
                 sequence_buffered: bool = False,
                 max_buffer_lag_frames: int = 2,
                 video_push: bool = False, push_credits: int = 4):
        if video_codec not in ("ppm", "h264"):
            raise ValueError("invalid optional kernel video codec")
        if type(install_remote_pyav) is not bool or (install_remote_pyav and video_codec != "h264"):
            raise ValueError("remote PyAV installation requires opt-in H.264")
        self.provider = provider
        self.session = session
        self.run_id = run_id
        self.fps = fps
        self.video_codec = video_codec
        if type(sequence_buffered) is not bool or (sequence_buffered and video_codec != "h264"):
            raise ValueError("sequence buffer requires H.264")
        self.install_remote_pyav = install_remote_pyav
        if (type(max_buffer_lag_frames) is not int
                or not 0 <= max_buffer_lag_frames <= 2
                or (not sequence_buffered and max_buffer_lag_frames != 2)):
            raise ValueError("renderer lag requires buffered H.264 mode")
        self.sequence_buffered = sequence_buffered
        if (type(video_push) is not bool
                or (video_push and not sequence_buffered)):
            raise ValueError("push requires buffered H.264")
        if (type(push_credits) is not int or push_credits not in (4, 6, 8)
                or (not video_push and push_credits != 4)):
            raise ValueError("push credit window only applies to external push")
        self.video_push = video_push
        self.push_credits = push_credits
        self.max_buffer_lag_frames = max_buffer_lag_frames
        self.client = None
        self.wire = None
        self.comm_id = None
        self._closed = False
        self.latest_renderer_metrics = None
        self.latest_rate_ack = 0

    def _send(self, kind: str, **fields: object) -> None:
        if self._closed or not self.comm_id or self.wire is None:
            raise LifecycleError("kernel bridge not attached")
        if not self.wire.connection_ready.is_set():
            raise LifecycleError("kernel websocket disconnected")
        data = {"v": 1, "kind": kind, "run_id": self.run_id, **fields}
        msg = self.wire.session.msg("comm_msg",
                                    {"comm_id": self.comm_id, "data": data})
        self.wire.shell_channel.send(msg)

    def open(self, *, expected_kernel_id: str | None = None) -> dict[str, Any]:
        """Upload runtime code, attach to owned kernel, then submit bootstrap once."""
        from ._vendor import jupyter_kernel_client as jkc

        # External mode checks the expected kernel BEFORE uploading or
        # executing code. Ordinary owned-runtime behavior remains unchanged.
        kernel_id = None
        if expected_kernel_id is not None:
            if (not isinstance(expected_kernel_id, str)
                    or not 1 <= len(expected_kernel_id) <= 128):
                raise LifecycleError("external kernel identity invalid")
            kernel = self.provider._kernel(self.session, expected_kernel_id)
            if kernel.get("id") != expected_kernel_id:
                raise LifecycleError("external kernel changed before bootstrap")
            kernel_id = expected_kernel_id
        remote = f"content/colabcola-bridge/{self.run_id}"
        root = "/" + remote
        self.provider._execute(self.session, 
            f"from pathlib import Path\nPath({root!r}).mkdir(parents=True, exist_ok=True)")
        files = importlib.resources.files("colabcola")
        names = ["bridge_colab_runtime.py", "bridge_colab_comm_runtime.py"]
        if self.video_codec == "h264":
            names.append("bridge_video_colab_source.py")
        if self.video_push:
            names.append("bridge_video_push_runtime.py")
        for name in names:
            self.provider._upload(self.session, remote + "/" + name, files.joinpath(name).read_bytes())
        if kernel_id is None:
            kernel_id = self.provider._kernel(self.session)["id"]
        base = getattr(jkc, "KernelClient", None) or jkc.JupyterKernelClient
        # Only connection setup may be retried. Once an execute_request is
        # submitted, an uncertain response must NEVER trigger replay.
        reports = []
        for connect_attempt in range(2):
            client = safe_client_class(base)(
                server_url=self.session.url, token=self.session.token, kernel_id=kernel_id,
                client_kwargs={
                    "subprotocol": jkc.JupyterSubprotocol.DEFAULT,
                    "extra_params": {"colab-runtime-proxy-token": self.session.token},
                },
                headers={
                    "X-Colab-Client-Agent": "colab-cli",
                    "X-Colab-Runtime-Proxy-Token": self.session.token,
                },
            )
            probe = KernelWebSocketProbe()
            wire = None
            try:
                wire = client._manager.client
                probe.install(wire)
                client.start()
                if not wire.connection_ready.is_set():
                    wire.connection_ready.wait(timeout=2)
                if wire.connection_ready.is_set():
                    self.client = client
                    self.wire = wire
                    break
                reports.append(probe.summary(wire))
            except Exception as exc:
                summary = probe.summary(wire) if wire is not None else "wire=unavailable"
                reports.append(summary + ", start_error=" + safe_error_type(exc))
            finally:
                if self.client is not client:
                    try:
                        client.stop()
                    except Exception as exc:
                        raise LifecycleError(
                            "kernel websocket failed and connection cleanup failed: "
                            + safe_error_type(exc)
                        ) from None
            if connect_attempt == 0:
                time.sleep(2)
        if self.wire is None:
            raise LifecycleError(
                "kernel websocket did not open after 2 pre-execution connections: "
                + " | ".join(reports)
            )
        try:
            video_path = (root + "/bridge_video_colab_source.py"
                          if self.video_codec == "h264" else None)
            push_path = (root + "/bridge_video_push_runtime.py"
                         if self.video_push else None)
            code = ("import runpy\n"
                    f"runpy.run_path({root + '/bridge_colab_comm_runtime.py'!r}, "
                    "run_name='__main__', init_globals={"
                    f"'BRIDGE_DIRECTORY':{root!r}, 'BRIDGE_FPS':{self.fps}, "
                    f"'BRIDGE_RUN_ID':{self.run_id!r},"
                    f"'BRIDGE_RENDERER_PATH':{root + '/bridge_colab_runtime.py'!r},"
                    f"'BRIDGE_VIDEO_SOURCE_PATH':{video_path!r},"
                    f"'BRIDGE_VIDEO_PREFLIGHT':{self.video_codec == 'h264'!r},"
                    f"'BRIDGE_INSTALL_REMOTE_PYAV':{self.install_remote_pyav!r},"
                    f"'BRIDGE_SEQUENCE_BUFFERED':{self.sequence_buffered!r},"
                    f"'BRIDGE_MAX_BUFFER_LAG_FRAMES':{self.max_buffer_lag_frames!r},"
                    f"'BRIDGE_VIDEO_PUSH':{self.video_push!r},"
                    f"'BRIDGE_VIDEO_PUSH_SOURCE_PATH':{push_path!r},"
                    f"'BRIDGE_PUSH_CREDITS':{self.push_credits!r}"
                    "})\n")
            # Never replay an uncertain request. A new run directory is required
            # after any ambiguous submission outcome.
            request_id = self.wire.execute(code, allow_stdin=False, store_history=False)
            deadline = time.monotonic() + (115 if self.install_remote_pyav else 65)
            while time.monotonic() < deadline:
                if not self.wire.connection_ready.is_set():
                    raise LifecycleError("kernel websocket disconnected during bootstrap")
                for channel in (self.wire.iopub_channel, self.wire.shell_channel):
                    try:
                        msg = channel.get_msg(timeout=.1)
                    except queue.Empty:
                        continue
                    if msg.get("parent_header", {}).get("msg_id") != request_id:
                        continue
                    kind = msg.get("header", {}).get("msg_type")
                    content = msg.get("content", {})
                    if kind == "execute_reply" and content.get("status") != "ok":
                        summary = bootstrap_error_summary(
                            self.provider, self.session, remote, self.run_id, content)
                        raise LifecycleError("kernel bridge bootstrap execution failed: " + summary)
                    if kind == "error":
                        # IOPub error often precedes execute_reply, but it may
                        # be lost/reordered by a hosted kernel proxy.
                        summary = bootstrap_error_summary(
                            self.provider, self.session, remote, self.run_id, content)
                        raise LifecycleError("kernel bridge bootstrap execution failed: " + summary)
                    if kind != "comm_open":
                        continue
                    if content.get("target_name") != TARGET:
                        continue
                    details = content.get("data", {})
                    if (not isinstance(details, dict) or details.get("v") != 1
                            or details.get("kind") not in ("ready", "video_preflight_failed")
                            or details.get("run_id") != self.run_id
                            or details.get("device") != "cuda"
                            or not isinstance(details.get("gpu"), str)
                            or "T4" not in details["gpu"]
                            or details.get("width") != 640 or details.get("height") != 360):
                        raise LifecycleError("kernel bridge CUDA handshake invalid")
                    comm_id = content.get("comm_id")
                    if not isinstance(comm_id, str) or not 1 <= len(comm_id) <= 128:
                        raise LifecycleError("kernel bridge communication id invalid")
                    self.comm_id = comm_id
                    if details["kind"] == "video_preflight_failed":
                        # Preserve the Comm ID for best-effort close before
                        # the owned allocation is released by the caller.
                        if self.video_codec != "h264":
                            raise LifecycleError("unexpected video preflight reply")
                        code = details.get("failure_code")
                        if code not in (
                                "source_path_missing", "pyav_unavailable", "numpy_unavailable",
                                "libx264_unavailable", "video_initialization_failed",
                                "pyav_install_failed", "pyav_install_timeout"):
                            code = "unknown"
                        raise LifecycleError("remote H.264 preflight failed: code=" + code)
                    if self.video_codec == "h264" and details.get("video_codec") != "h264-annexb":
                        raise LifecycleError("kernel bridge H.264 handshake invalid")
                    if self.video_push and details.get("video_mode") != "credit-push":
                        raise LifecycleError("kernel push video handshake invalid")
                    return details
            raise LifecycleError("kernel bridge bootstrap timeout")
        except BaseException:
            self.close()
            raise

    def _frame_payload(self, message) -> dict | None:
        """Filter unrelated Jupyter traffic and maintain control ACK state."""
        if not isinstance(message, dict) or message.get("header", {}).get("msg_type") != "comm_msg":
            return None
        content = message.get("content", {})
        if not isinstance(content, dict) or content.get("comm_id") != self.comm_id:
            return None
        payload = content.get("data")
        if (not isinstance(payload, dict) or payload.get("v") != 1
                or payload.get("run_id") != self.run_id):
            return None
        kind = payload.get("kind")
        if kind == "error":
            raise LifecycleError("remote kernel bridge reported an error")
        if kind == "rate_ack":
            rate_seq = payload.get("seq")
            if type(rate_seq) is int and self.latest_rate_ack < rate_seq < 10000:
                self.latest_rate_ack = rate_seq
        return payload if kind == "frame" else None

    def _decode_frame(self, payload: dict, *, require_pull_id: bool = False):
        encoded = payload.get("data")
        if not isinstance(encoded, str) or len(encoded) > MAX_ENCODED_FRAME:
            raise LifecycleError("kernel frame encoding exceeds limit")
        try:
            binary = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            raise LifecycleError("kernel frame Base64 invalid") from None
        sequence, ack, width, height, ppm = unpack_frame(binary)
        if width != 640 or height != 360:
            raise LifecycleError("kernel frame dimensions invalid")
        self.latest_renderer_metrics = validated_renderer_metrics(
            payload.get("metrics"), self.run_id)
        if require_pull_id:
            pull_id = payload.get("pull_seq")
            if type(pull_id) is not int or not 1 <= pull_id <= 100000000:
                raise LifecycleError("kernel frame response has invalid request ID")
            return pull_id, sequence, ack, ppm, len(binary)
        return sequence, ack, ppm, len(binary)

    def _recv_frame(self, seconds: float, *, require_pull_id: bool, optional: bool):
        if self.wire is None or not self.comm_id:
            raise LifecycleError("kernel bridge not started")
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if not self.wire.connection_ready.is_set():
                raise LifecycleError("kernel websocket disconnected")
            try:
                message = self.wire.iopub_channel.get_msg(
                    timeout=min(.25, max(.01, deadline - time.monotonic())))
            except queue.Empty:
                continue
            payload = self._frame_payload(message)
            if payload is not None:
                return self._decode_frame(payload, require_pull_id=require_pull_id)
        if optional:
            return None
        raise LifecycleError("kernel frame timeout; no reply to last pull")

    def next_frame(self, seconds: float) -> tuple[int, int, bytes, int]:
        """Receive one ordinary frame (legacy single-in-flight contract)."""
        return self._recv_frame(seconds, require_pull_id=False, optional=False)

    def poll_pipeline_frame(self, seconds: float):
        """Return (pull ID, frame sequence, ACK, pixels, bytes) or None.

        A short timeout does not imply transport failure; the pipeline
        maintains independent per-request deadlines and a strict credit cap.
        """
        if not 0 < seconds <= .5:
            raise ValueError("pipeline receive timeout must be within (0, 0.5] seconds")
        return self._recv_frame(seconds, require_pull_id=True, optional=True)

    def probe_small_message_rtt(self, samples: int = 8) -> int:
        """Measure tiny shell→IOPub Comm round trip before streaming starts.

        Runs before frame requests, so consuming the IOPub queue cannot
        interfere with outstanding video. Never reports frame performance.
        """
        if type(samples) is not int or not 1 <= samples <= 16:
            raise ValueError("invalid ping sample count")
        if self.wire is None or not self.comm_id:
            raise LifecycleError("kernel bridge not attached")
        durations = []
        for seq in range(1, samples + 1):
            sent_at = time.monotonic()
            self._send("ping", seq=seq)
            expiry = sent_at + 5
            while time.monotonic() < expiry:
                if not self.wire.connection_ready.is_set():
                    raise LifecycleError("kernel websocket disconnected during ping")
                try:
                    msg = self.wire.iopub_channel.get_msg(
                        timeout=min(.25, max(.01, expiry - time.monotonic())))
                except queue.Empty:
                    continue
                if msg.get("header", {}).get("msg_type") != "comm_msg":
                    continue
                contents = msg.get("content", {})
                if contents.get("comm_id") != self.comm_id:
                    continue
                payload = contents.get("data")
                if (not isinstance(payload, dict)
                        or payload.get("v") != 1 or payload.get("run_id") != self.run_id):
                    continue
                if payload.get("kind") == "pong" and type(payload.get("seq")) is int:
                    if payload["seq"] == seq:
                        durations.append((time.monotonic() - sent_at) * 1000)
                        break
            else:
                raise LifecycleError("tiny Comm ping timed out")
        return round(median(durations))

    def request_frame(self, pull_seq: int | None = None):
        if pull_seq is None:
            self._send("pull")
        else:
            if type(pull_seq) is not int or not 1 <= pull_seq <= 100000000:
                raise ValueError("invalid frame request ID")
            self._send("pull", pull_seq=pull_seq)

    def send_input(self, seq: int, event: dict) -> None:
        self._send("input", seq=seq, **event)

    def set_render_fps(self, seq: int, fps: int) -> None:
        if (type(seq) is not int or not 0 < seq < 10000
                or type(fps) is not int or not 1 <= fps <= 12):
            raise ValueError("invalid renderer rate request")
        self._send("rate", seq=seq, fps=fps)

    def close(self):
        if self._closed:
            return
        if self.comm_id and self.wire and self.wire.connection_ready.is_set():
            try:
                self._send("close")
            except Exception:
                pass
        self._closed = True
        if self.client:
            try:
                self.client.stop()
            finally:
                self.client = None
                self.wire = None


def stream_comm(peer: KernelCommBridge, gpu: str, outgoing: queue.Queue,
                frames: queue.Queue, stopped, deadline: float,
                poll_hz: int = 6, probe_link: bool = False,
                in_flight: int = 1) -> None:
    """One-in-flight proven default, or explicitly opted-in bounded pipeline."""
    if type(in_flight) is not int or not 1 <= in_flight <= 4:
        raise ValueError("in-flight requests must be between 1 and 4")
    if in_flight > 1:
        if probe_link:
            raise ValueError("link probe and frame pipeline cannot be combined")
        from .bridge_pipeline import stream_comm_pipeline
        return stream_comm_pipeline(
            peer, gpu, outgoing, frames, stopped, deadline, poll_hz, in_flight)
    meter = BridgeTransferMeter()
    last_seq = -1
    input_seq = 0
    seen = 0
    transferred = 0
    pulls = 0
    pull_rtts_ms: list[float] = []
    recent_remote_metrics = None
    last_remote_metrics_seq = -1
    last_report = time.monotonic()
    last_fresh_frame = last_report
    interval = 1 / poll_hz
    print("[colabcola] Persistent kernel Comm connected", flush=True)
    offer_update(frames, ("status", f"Kernel Comm · {gpu}"))
    try:
        if probe_link:
            baseline = peer.probe_small_message_rtt(samples=8)
            print(f"[colabcola] tiny Comm RTT p50 {baseline} ms (8 samples, no frame payload)", flush=True)
        while not stopped.is_set() and time.monotonic() < deadline:
            start = time.monotonic()
            latest = None
            while True:
                try:
                    latest = outgoing.get_nowait()
                except queue.Empty:
                    break
            if latest:
                input_seq += 1
                meter.sent(input_seq, time.monotonic())
                peer.send_input(input_seq, latest)
            pull_started = time.monotonic()
            peer.request_frame()
            sequence, ack, ppm, size = peer.next_frame(min(8.0, max(0.1, deadline-time.monotonic())))
            now = time.monotonic()
            pulls += 1
            pull_rtts_ms.append((now - pull_started) * 1000)
            if sequence > last_seq:
                last_seq = sequence
                last_fresh_frame = now
                seen += 1
                transferred += size
                offer_update(frames, ("frame", ppm))
            elif now - last_fresh_frame >= 30:
                raise LifecycleError("kernel renderer published no fresh frame for 30 seconds")
            meter.received_ack(ack, now)
            remote_metrics = getattr(peer, "latest_renderer_metrics", None)
            if (isinstance(remote_metrics, dict)
                    and remote_metrics["seq"] > last_remote_metrics_seq):
                last_remote_metrics_seq = remote_metrics["seq"]
                recent_remote_metrics = remote_metrics
            if now-last_report >= 2:
                elapsed = max(now-last_report, .001)
                stats = meter.flush()
                result = (f"{gpu} · kernel Comm {seen/elapsed:.1f} FPS · "
                          f"{transferred/elapsed/1024:.0f} KiB/s")
                if pull_rtts_ms:
                    result += f" · pull RTT p50 {median(pull_rtts_ms):.0f} ms"
                    result += f" · fresh {seen}/{pulls}"
                if recent_remote_metrics is not None:
                    result += (
                        f" · remote render {recent_remote_metrics['renderer_fps']:.1f} FPS"
                        f" · render+copy {recent_remote_metrics['render_copy_ms']:.0f} ms"
                        f" · compress {recent_remote_metrics['compress_ms']:.0f} ms"
                        f" · publish {recent_remote_metrics['publish_ms']:.0f} ms"
                    )
                if stats["ack"] is not None:
                    result += f" · input ACK observed p50 {stats['ack']} ms"
                print("[colabcola] "+result, flush=True)
                offer_update(frames, ("status", result))
                seen = 0
                transferred = 0
                pulls = 0
                pull_rtts_ms.clear()
                recent_remote_metrics = None
                last_report = now
            sleep_for = max(0, interval-(time.monotonic()-start))
            if sleep_for:
                time.sleep(sleep_for)
    except Exception as exc:
        print("[colabcola] Persistent kernel Comm failed: " + type(exc).__name__, flush=True)
        raise LifecycleError("kernel Comm transport interrupted") from None
    finally:
        stopped.set()
