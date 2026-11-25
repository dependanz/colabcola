"""Credential-safe diagnostics for an owned Jupyter kernel WebSocket.

Only collect connection lifecycle flags and coarse, allowlisted error types.
Never retain or display proxy URLs, tokens, response bodies, exception text,
WebSocket close messages or frame contents.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import threading
import time

_ERROR_TYPES = frozenset({
    "WebSocketBadStatusException", "WebSocketTimeoutException",
    "WebSocketConnectionClosedException", "WebSocketException",
    "WebSocketAddressException", "SSLError", "ConnectionError",
    "ConnectionRefusedError", "TimeoutError", "OSError",
    "ValueError", "RuntimeError", "TypeError",
})


def safe_error_type(error: object) -> str:
    name = type(error).__name__
    return name if name in _ERROR_TYPES else "OtherException"


@dataclass
class KernelWebSocketProbe:
    opened: bool = False
    closed: bool = False
    error_type: str = "none"
    close_code: int | None = None
    thread_alive: bool = False

    def install(self, wire) -> None:
        """Wrap per-instance hooks BEFORE start_channels creates WebSocketApp.

        The vendor package remains byte-for-byte unchanged. The hooks record
        only non-sensitive metadata and preserve original callback behavior.
        """
        original_open = wire._on_open
        original_close = wire._on_close
        original_runner = wire._run_websocket

        def on_open(socket):
            self.opened = True
            return original_open(socket)

        def on_close(socket, code, message):
            self.closed = True
            self.close_code = code if type(code) is int and 1000 <= code <= 4999 else None
            return original_close(socket, code, message)

        def on_error(socket, error):
            self.error_type = safe_error_type(error)

        def run_with_diagnostics():
            socket = getattr(wire, "kernel_socket", None)
            if socket is not None:
                socket.on_error = on_error
            return original_runner()

        wire._on_open = on_open
        wire._on_close = on_close
        wire._run_websocket = run_with_diagnostics

    def summary(self, wire) -> str:
        thread = getattr(wire, "connection_thread", None)
        self.thread_alive = bool(thread is not None and thread.is_alive())
        current = bool(wire.connection_ready.is_set())
        code = self.close_code if self.close_code is not None else "none"
        return (
            f"opened={int(self.opened)}, connected={int(current)}, "
            f"closed={int(self.closed)}, thread_alive={int(self.thread_alive)}, "
            f"close_code={code}, error_type={self.error_type}"
        )


def matched_video_sequence(message, run_id, comm_id):
    """Return a candidate video sequence, not protocol acceptance."""
    if not isinstance(message, dict) or not isinstance(comm_id, str) or not comm_id:
        return None
    header, content = message.get("header"), message.get("content")
    if (not isinstance(header, dict) or header.get("msg_type") != "comm_msg"
            or not isinstance(content, dict) or content.get("comm_id") != comm_id):
        return None
    payload = content.get("data")
    if (not isinstance(payload, dict) or payload.get("run_id") != run_id
            or type(payload.get("v")) is not int or payload["v"] != 1
            or payload.get("kind") not in ("video_access_unit", "video_no_new")):
        return None
    seq = payload.get("pull_seq")
    if type(seq) is int and 1 <= seq <= 100_000_000:
        return seq
    return None


def install_video_ingress(wire, *, trace, run_id, comm_id_getter):
    """Observe the pinned vendor callback and its IOPub enqueue on this instance.

    Wrap BEFORE start_channels. Preserve the vendor's single deserialization
    and original queue insertion; no payload mutation or extra network traffic.
    The timestamp is application callback entry, not TCP or hardware arrival.
    """
    original_callback = wire._on_message
    channel_queue = wire._iopub_msg_queue
    original_put = channel_queue.put_nowait
    local = threading.local()

    def on_message(socket, message):
        previous = getattr(local, "callback_ns", None)
        local.callback_ns = time.perf_counter_ns()
        try:
            return original_callback(socket, message)
        finally:
            local.callback_ns = previous

    def put_nowait(message):
        callback_ns = getattr(local, "callback_ns", None)
        seq = (matched_video_sequence(message, run_id, comm_id_getter())
               if callback_ns is not None else None)
        if seq is None:
            return original_put(message)
        depth = channel_queue.qsize()
        enqueue_start_ns = time.perf_counter_ns()
        result = original_put(message)
        trace.add("wire_video_enqueued", video_seq=seq,
                  callback_ns=callback_ns, enqueue_start_ns=enqueue_start_ns,
                  enqueue_return_ns=time.perf_counter_ns(),
                  iopub_depth_before=depth)
        return result

    channel_queue.put_nowait = put_nowait
    wire._on_message = on_message
