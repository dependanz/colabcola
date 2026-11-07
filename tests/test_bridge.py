"""No-network-to-Google regression tests for the experimental bridge."""
from __future__ import annotations

import asyncio
import base64
import json
import struct

import pytest

from colabcola.bridge_demo import Scene, handle_connection, main, photoimage_from_ppm, render_ppm
from colabcola.bridge_protocol import (
    BridgeProtocolError, MAX_MESSAGE_BYTES, PROTOCOL_VERSION,
    encode_message, read_message, send_message,
)


def message(kind: str, **fields):
    return {"v": PROTOCOL_VERSION, "type": kind, **fields}


def test_length_prefix_and_version_checks():
    raw = encode_message(message("input", event="mouse", x=0.25, y=1.0))
    assert struct.unpack("!I", raw[:4])[0] == len(raw) - 4
    with pytest.raises(BridgeProtocolError, match="version"):
        encode_message({"v": 999, "type": "close"})
    with pytest.raises(BridgeProtocolError, match="coordinates"):
        encode_message(message("input", event="mouse", x=-1, y=0))
    with pytest.raises(BridgeProtocolError, match="key"):
        encode_message(message("input", event="key", key=""))
    with pytest.raises(BridgeProtocolError, match="sequence"):
        encode_message(message("frame", seq=True, format="ppm", data=""))
    with pytest.raises(BridgeProtocolError, match="format"):
        encode_message(message("frame", seq=0, format="jpeg", data=""))


def test_oversized_message_rejected_before_allocation():
    async def check():
        reader = asyncio.StreamReader()
        reader.feed_data(struct.pack("!I", MAX_MESSAGE_BYTES + 1))
        with pytest.raises(BridgeProtocolError, match="length"):
            await read_message(reader)
    asyncio.run(check())


def test_scene_updates_from_input_and_cpu_renders():
    scene = Scene()
    before = render_ppm(scene, 0)
    scene.apply(message("input", event="mouse", x=0.9, y=0.1))
    scene.apply(message("input", event="key", key="Right"))
    after = render_ppm(scene, 0)
    assert (scene.x, scene.y) == (0.9, 0.1)
    assert scene.phase > 0
    assert before.startswith(b"P6\n192 120\n255\n")
    assert len(before) == len(after)
    assert before != after


def test_tk_viewer_accepts_raw_ppm_frame():
    """Catch the image decoding failure that previously crashed on Windows."""
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no graphical display available for Tk: {exc}")
    try:
        root.withdraw()
        frame = render_ppm(Scene(), seconds=0)
        image = photoimage_from_ppm(root, frame)
        assert image.width() == 192
        assert image.height() == 120
    finally:
        root.destroy()


def test_local_session_streams_frames_and_accepts_input():
    async def check():
        async def on_client(reader, writer):
            await handle_connection(reader, writer, token="local-secret", fps=20, device="cpu")

        server = await asyncio.start_server(on_client, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            await send_message(writer, message("hello", token="local-secret"))
            ready = await asyncio.wait_for(read_message(reader), 3)
            assert ready["type"] == "ready" and ready["device"] == "cpu"
            first = await asyncio.wait_for(read_message(reader), 3)
            assert first["type"] == "frame" and first["seq"] == 0
            assert base64.b64decode(first["data"], validate=True).startswith(b"P6\n")
            await send_message(writer, message("input", event="mouse", x=1.0, y=0.0))
            await send_message(writer, message("input", event="key", key="Right"))
            second = await asyncio.wait_for(read_message(reader), 3)
            assert second["type"] == "frame" and second["seq"] > first["seq"]
            await send_message(writer, message("close"))
            writer.close()
            await writer.wait_closed()
        finally:
            server.close()
            await server.wait_closed()
    asyncio.run(check())


def test_wrong_token_is_rejected():
    async def check():
        server = await asyncio.start_server(
            lambda r, w: handle_connection(r, w, token="expected", fps=8, device="cpu"),
            "127.0.0.1", 0,
        )
        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", server.sockets[0].getsockname()[1],
            )
            await send_message(writer, message("hello", token="incorrect"))
            reply = await asyncio.wait_for(read_message(reader), 3)
            assert reply["type"] == "error" and reply["message"] == "access denied"
            writer.close()
            await writer.wait_closed()
        finally:
            server.close()
            await server.wait_closed()
    asyncio.run(check())


def test_nonloopback_address_is_rejected():
    with pytest.raises(SystemExit) as error:
        main(["serve", "--host", "0.0.0.0"])
    assert error.value.code == 2


def test_cli_dispatches_bridge_help():
    from colabcola.cli import main as cli_main
    with pytest.raises(SystemExit) as result:
        cli_main(["bridge", "--help"])
    assert result.value.code == 0
