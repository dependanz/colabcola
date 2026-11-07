"""Versioned, bounded bidirectional messages for an experimental interactive bridge.

This is transport-independent. The initial demonstration uses loopback TCP;
a managed Colab adapter must provide an authenticated transport separately.
"""
from __future__ import annotations

import asyncio
import json
import struct
from typing import Any

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 2 * 1024 * 1024
MAX_FRAME_BYTES = 1024 * 1024
HEADER = struct.Struct("!I")
KINDS = frozenset({"hello", "ready", "error", "frame", "input", "close"})


class BridgeProtocolError(ValueError):
    """Malformed or unsupported bridge message."""


def validate_message(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise BridgeProtocolError("message must be an object")
    if message.get("v") != PROTOCOL_VERSION:
        raise BridgeProtocolError("unsupported protocol version")
    if message.get("type") not in KINDS:
        raise BridgeProtocolError("unsupported message type")
    kind = message["type"]
    if kind == "frame":
        if not isinstance(message.get("seq"), int) or isinstance(message["seq"], bool) or message["seq"] < 0:
            raise BridgeProtocolError("invalid frame sequence")
        if not isinstance(message.get("data"), str):
            raise BridgeProtocolError("frame data must be base64 text")
        if message.get("format") != "ppm":
            raise BridgeProtocolError("unsupported frame format")
        if len(message["data"]) > ((MAX_FRAME_BYTES + 2) // 3) * 4:
            raise BridgeProtocolError("frame exceeds byte limit")
    elif kind == "input":
        if message.get("event") not in ("key", "mouse"):
            raise BridgeProtocolError("unsupported input event")
        if kind == "input" and message["event"] == "key":
            key = message.get("key")
            if not isinstance(key, str) or not 1 <= len(key) <= 32 or not key.isprintable():
                raise BridgeProtocolError("invalid key")
        else:
            for field in ("x", "y"):
                value = message.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    raise BridgeProtocolError("mouse coordinates must be normalized")
    elif kind == "hello":
        token = message.get("token")
        if not isinstance(token, str) or len(token) > 512:
            raise BridgeProtocolError("invalid handshake token")
    return message


def encode_message(message: dict[str, Any]) -> bytes:
    validate_message(message)
    payload = json.dumps(message, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")
    if not payload or len(payload) > MAX_MESSAGE_BYTES:
        raise BridgeProtocolError("message exceeds byte limit")
    return HEADER.pack(len(payload)) + payload


async def send_message(writer: asyncio.StreamWriter, message: dict[str, Any]) -> None:
    writer.write(encode_message(message))
    await writer.drain()


async def read_message(reader: asyncio.StreamReader) -> dict[str, Any]:
    prefix = await reader.readexactly(HEADER.size)
    size = HEADER.unpack(prefix)[0]
    if not 0 < size <= MAX_MESSAGE_BYTES:
        raise BridgeProtocolError("invalid message length")
    payload = await reader.readexactly(size)
    try:
        decoded = json.loads(payload)
    except (ValueError, UnicodeDecodeError) as exc:
        raise BridgeProtocolError("invalid JSON message") from exc
    return validate_message(decoded)
