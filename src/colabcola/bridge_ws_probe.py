"""Credential-safe diagnostics for an owned Jupyter kernel WebSocket.

Only collect connection lifecycle flags and coarse, allowlisted error types.
Never retain or display proxy URLs, tokens, response bodies, exception text,
WebSocket close messages or frame contents.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

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
