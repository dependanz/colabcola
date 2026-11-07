"""Exercise the Comm wire protocol against a real, local IPython kernel.

No Google API calls or GPU allocation. The renderer is a small fake that
publishes a static frame and status file; the Jupyter Comm machinery is real.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
import queue
import struct
import time
import zlib

import pytest


def test_local_jupyter_comm_open_pull_input_close(tmp_path):
    pytest.importorskip("ipykernel")
    jupyter_client = pytest.importorskip("jupyter_client")
    run_id = "f" * 32
    work = tmp_path / "runtime"
    work.mkdir()
    pixels = bytes((19, 78, 126)) * (640 * 360)
    ppm = b"P6" + bytes((10,)) + b"640 360" + bytes((10,)) + b"255" + bytes((10,)) + pixels
    binary = struct.pack("!4sIIII", b"CCB1", 0, 0, 640, 360) + zlib.compress(ppm)
    (work / "frame.bin").write_bytes(binary)

    renderer = tmp_path / "fake_renderer.py"
    renderer.write_text(
        "import json\nfrom pathlib import Path\n"
        "def start(directory, fps, run_id):\n"
        " p=Path(directory)\n"
        " (p/'status.json').write_text(json.dumps({'state':'ready','run_id':run_id,"
        "'device':'cuda','gpu':'Tesla T4','width':640,'height':360}))\n"
        "def atomic_bytes(path, content):\n"
        " Path(path).write_bytes(content)\n",
        encoding="utf-8",
    )

    from importlib.resources import files
    adapter = str(files("colabcola").joinpath("bridge_colab_comm_runtime.py"))
    code = (
        "import runpy\n"
        f"runpy.run_path({adapter!r}, run_name='__main__', init_globals={{"
        f"'BRIDGE_DIRECTORY':{str(work)!r}, 'BRIDGE_RUN_ID':{run_id!r},"
        f"'BRIDGE_FPS':6, 'BRIDGE_RENDERER_PATH':{str(renderer)!r}}})\n"
    )
    manager = jupyter_client.KernelManager(kernel_name="python3")
    client = None
    try:
        manager.start_kernel()
        client = manager.client()
        client.start_channels()
        client.wait_for_ready(timeout=15)
        request = client.execute(code, store_history=False, allow_stdin=False)
        comm_id = None
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline and comm_id is None:
            try:
                message = client.get_iopub_msg(timeout=.5)
            except queue.Empty:
                continue
            if message.get("parent_header", {}).get("msg_id") != request:
                continue
            if message.get("header", {}).get("msg_type") == "error":
                pytest.fail("local kernel Comm bootstrap returned a Python error")
            if message.get("header", {}).get("msg_type") != "comm_open":
                continue
            content = message["content"]
            assert content["target_name"] == "colabcola.bridge.v1"
            assert content["data"]["run_id"] == run_id
            assert content["data"]["gpu"] == "Tesla T4"
            comm_id = content["comm_id"]
        assert comm_id is not None

        def send(data):
            message = client.session.msg("comm_msg",
                content={"comm_id": comm_id, "data": {"v":1, "run_id":run_id, **data}})
            client.shell_channel.send(message)

        def poll_frame(timeout=8):
            until = time.monotonic() + timeout
            while time.monotonic() < until:
                try:
                    message = client.get_iopub_msg(timeout=.5)
                except queue.Empty:
                    continue
                if message.get("header", {}).get("msg_type") != "comm_msg":
                    continue
                data = message.get("content", {})
                if data.get("comm_id") != comm_id:
                    continue
                if data.get("data", {}).get("kind") == "frame":
                    return data["data"]
            pytest.fail("local kernel did not send frame over Jupyter Comm")

        # Validate a tiny Comm message on a *real* IPython kernel before
        # measuring the much larger frame response.
        send({"kind": "ping", "seq": 1})
        until = time.monotonic() + 8
        while time.monotonic() < until:
            try:
                ping_reply = client.get_iopub_msg(timeout=.5)
            except queue.Empty:
                continue
            if (ping_reply.get("header", {}).get("msg_type") == "comm_msg"
                    and ping_reply.get("content", {}).get("comm_id") == comm_id
                    and ping_reply.get("content", {}).get("data", {}).get("kind") == "pong"):
                assert ping_reply["content"]["data"]["seq"] == 1
                break
        else:
            pytest.fail("local kernel did not acknowledge tiny Comm ping")

        send({"kind": "pull"})
        result = poll_frame()
        assert base64.b64decode(result["data"], validate=True) == binary
        send({"kind": "pull", "pull_seq": 42})
        tagged = poll_frame()
        assert tagged["pull_seq"] == 42
        assert base64.b64decode(tagged["data"], validate=True) == binary
        send({"kind": "input", "seq": 1, "event": "mouse", "x": .8, "y": .4})
        send({"kind": "pull"})
        assert poll_frame()["kind"] == "frame"
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not (work / "input.json").exists():
            time.sleep(.05)
        assert json.loads((work / "input.json").read_text())["seq"] == 1
        send({"kind": "close"})
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not (work / "stop").exists():
            time.sleep(.05)
        assert (work / "stop").exists()
    finally:
        if client is not None:
            client.stop_channels()
        manager.shutdown_kernel(now=True)
