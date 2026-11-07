"""Offline tests of experimental persistent Jupyter Comm bridge."""
from __future__ import annotations

import base64
from collections import deque
import importlib.resources
import json
from pathlib import Path
import queue
import runpy
import struct
import sys
import threading
from types import SimpleNamespace, ModuleType
import zlib

import pytest

from colabcola.bridge_comm import KernelCommBridge, TARGET, stream_comm
from colabcola.bridge_demo import render_ppm, Scene
from colabcola.lifecycle_io import LifecycleError


def packed_frame(sequence=5, ack=3) -> bytes:
    ppm = render_ppm(Scene(), 0)
    return struct.pack("!4sIIII", b"CCB1", sequence, ack, 192, 120) + zlib.compress(ppm)


class FakeChannel:
    def __init__(self, messages=()):
        self.messages = deque(messages)
        self.sent = []
    def get_msg(self, timeout=None):
        if not self.messages:
            raise queue.Empty()
        return self.messages.popleft()
    def send(self, msg):
        self.sent.append(msg)


class FakeWire:
    def __init__(self, messages=()):
        self.iopub_channel = FakeChannel(messages)
        self.shell_channel = FakeChannel()
        self.session = SimpleNamespace(msg=lambda kind, content: {"msg_type": kind, "content": content})
        self.connection_ready = threading.Event()
        self.connection_ready.set()
        self.commands = []
    def execute(self, code, **options):
        compile(code, "comm_bootstrap.py", "exec")
        self.commands.append(code)
        return "request-1"


def comm_msg(data, *, comm_id="my-comm"):
    return {"header": {"msg_type": "comm_msg"},
            "content": {"comm_id": comm_id, "data": data}}


def test_authenticated_comm_open_and_close(monkeypatch):
    from colabcola import bridge_comm

    run_id = "a" * 32
    ready = {"v": 1, "kind": "ready", "run_id": run_id, "gpu": "Tesla T4",
             "device": "cuda", "width": 640, "height": 360}
    message = {"header": {"msg_type": "comm_open"},
               "parent_header": {"msg_id": "request-1"},
               "content": {"comm_id": "my-comm", "target_name": TARGET, "data": ready}}
    wire = FakeWire((message,))
    created = []
    class FakeClient:
        def __init__(self, **kwargs):
            self._manager = SimpleNamespace(client=wire)
            self.started = False
            self.stopped = False
            created.append(self)
        def start(self):
            self.started = True
        def stop(self):
            self.stopped = True
    monkeypatch.setattr(bridge_comm, "safe_client_class", lambda _: FakeClient)

    class Provider:
        def __init__(self):
            self.commands = []
            self.uploads = []
        def _execute(self, session, code):
            self.commands.append(code)
        def _upload(self, session, name, data):
            self.uploads.append((name, data))
        def _kernel(self, session):
            return {"id": "owned-kernel"}
    provider = Provider()
    session = SimpleNamespace(url="https://example.invalid", token="sensitive")
    peer = KernelCommBridge(provider, session, run_id, 6)
    assert peer.open() == ready
    assert peer.comm_id == "my-comm"
    assert len(provider.uploads) == 2
    assert "bridge_colab_runtime.py" in provider.uploads[0][0]
    assert "bridge_colab_comm_runtime.py" in provider.uploads[1][0]
    assert created[0].started
    assert "runpy.run_path" in wire.commands[0]
    assert "sensitive" not in wire.commands[0]
    peer.request_frame()
    peer.send_input(7, {"event": "mouse", "x": 0.6, "y": 0.4})
    assert wire.shell_channel.sent[0]["content"]["data"]["kind"] == "pull"
    assert wire.shell_channel.sent[1]["content"]["data"]["seq"] == 7
    peer.close()
    assert wire.shell_channel.sent[-1]["content"]["data"]["kind"] == "close"
    assert created[0].stopped


def test_comm_frames_filter_unrelated_iopub_and_decode():
    run_id = "b" * 32
    ppm_expected = b"P6" + bytes((10,)) + b"640 360" + bytes((10,)) + b"255" + bytes((10,)) + bytes((16, 32, 48)) * (640 * 360)
    binary = struct.pack("!4sIIII", b"CCB1", 5, 3, 640, 360) + zlib.compress(ppm_expected)
    good = comm_msg({
        "v": 1, "run_id": run_id, "kind": "frame",
        "data": base64.b64encode(binary).decode("ascii"),
    })
    wire = FakeWire((
        comm_msg(good["content"]["data"], comm_id="different"),
        {"header": {"msg_type": "status"}, "content": {"execution_state": "idle"}},
        good,
    ))
    peer = KernelCommBridge(None, None, run_id, 6)
    peer.wire = wire
    peer.comm_id = "my-comm"
    seq, ack, ppm, compressed_size = peer.next_frame(.5)
    assert seq == 5 and ack == 3
    assert ppm == ppm_expected
    assert compressed_size == len(binary)


def test_comm_frame_rejects_invalid_base64_without_leaking_content():
    run_id = "c" * 32
    wire = FakeWire((comm_msg({
        "v": 1, "run_id": run_id, "kind": "frame", "data": "secret_invalid@@",
    }),))
    peer = KernelCommBridge(None, None, run_id, 6)
    peer.wire = wire
    peer.comm_id = "my-comm"
    with pytest.raises(LifecycleError) as exc:
        peer.next_frame(.5)
    assert "secret_invalid" not in str(exc.value)


def test_remote_comm_handler_uses_frame_credit_and_validates_input(tmp_path, monkeypatch):
    """Exercise remote adapter with fake renderer and fake kernel Comm."""
    root = tmp_path / "runtime"
    root.mkdir()
    rendered = packed_frame()
    (root / "frame.bin").write_bytes(rendered)
    fake_renderer = tmp_path / "fake_renderer.py"
    fake_renderer.write_text(
        "from pathlib import Path\n"
        "import json,os\n"
        "def start(directory,fps,run_id):\n"
        " p=Path(directory)\n"
        " (p/'status.json').write_text(json.dumps(dict(run_id=run_id,state='ready',"
        "device='cuda',gpu='Tesla T4',width=640,height=360)))\n"
        "def atomic_bytes(path,body):\n"
        " Path(path).write_bytes(body)\n"
    )
    channels = []
    class FakeComm:
        def __init__(self, **kwargs):
            self.open_message = kwargs
            self.replies = []
            self.closed = False
            channels.append(self)
        def on_msg(self, callback):
            self.callback = callback
        def send(self, content):
            self.replies.append(content)
        def close(self):
            self.closed = True
    package = ModuleType("comm")
    package.create_comm = lambda **kw: FakeComm(**kw)
    monkeypatch.setitem(sys.modules, "comm", package)
    kernel = ModuleType("ipykernel")
    legacy = ModuleType("ipykernel.comm")
    legacy.Comm = lambda **kw: FakeComm(**kw)
    kernel.comm = legacy
    monkeypatch.setitem(sys.modules, "ipykernel", kernel)
    monkeypatch.setitem(sys.modules, "ipykernel.comm", legacy)
    path = importlib.resources.files("colabcola").joinpath("bridge_colab_comm_runtime.py")
    env = runpy.run_path(str(path), run_name="test_comm")
    run_id = "d" * 32
    env["start"](str(root), run_id, 6, str(fake_renderer))
    channel = channels[0]
    assert channel.open_message["data"]["gpu"] == "Tesla T4"
    channel.callback({"content": {"data": {"v": 1, "kind": "input", "run_id": run_id,
                                           "event": "mouse", "x": .5, "y": .3, "seq": 1}}})
    assert json.loads((root / "input.json").read_text())["seq"] == 1
    channel.callback({"content": {"data": {"v": 1, "kind": "pull", "run_id": run_id}}})
    assert len(channel.replies) == 1
    assert base64.b64decode(channel.replies[0]["data"], validate=True) == rendered
    channel.callback({"content": {"data": {"v": 1, "kind": "close", "run_id": run_id}}})
    assert channel.closed
    assert (root / "stop").is_file()


def test_cli_comm_transport_is_opt_in(monkeypatch):
    from colabcola.cli import main
    from colabcola import bridge_colab
    called = []
    monkeypatch.setattr(bridge_colab, "run_colab", lambda opts: called.append(opts.transport) or 0)
    assert main(["--profile", "example", "bridge", "colab",
                 "--preflight-only", "--transport", "comm"]) == 0
    assert called == ["comm"]


def test_bootstrap_error_summary_reports_only_vetted_stages():
    from colabcola.bridge_comm import bootstrap_error_summary
    class FakeProvider:
        def _json_file(self, session, path):
            assert path.endswith("bridge-bootstrap-diagnostic.json")
            return {
                "schema": 1, "run_id": "f" * 32, "state": "failed",
                "stage": "creating_comm", "error_type": "ModuleNotFoundError",
            }

    assert bootstrap_error_summary(
        FakeProvider(), object(), "content/colabcola-bridge/test", "f" * 32,
        {"ename": "ImportError", "evalue": "DO_NOT_ECHO_REMOTE_SECRETS"}
    ) == "stage=creating_comm, error_type=ModuleNotFoundError"


def test_bootstrap_error_summary_rejects_untrusted_diagnostics():
    from colabcola.bridge_comm import bootstrap_error_summary

    class FakeProvider:
        def _json_file(self, session, path):
            return {"schema": 1, "run_id": "WRONG_ID", "state": "failed",
                    "stage": ["leak"], "error_type": ["leak"]}

    result = bootstrap_error_summary(
        FakeProvider(), object(), "content/colabcola-bridge/test", "a" * 32,
        {"ename": "ValueError", "evalue": "SECRET_VALUE"})
    assert result == "error_type=ValueError"
    assert "SECRET_VALUE" not in result


def test_remote_comm_bootstrap_failure_writes_sanitized_report(tmp_path):
    root = tmp_path / "owned"
    root.mkdir()
    renderer = tmp_path / "failed_renderer.py"
    renderer.write_text(
        "from pathlib import Path\\n"
        "import json\\n"
        "def start(directory, fps, run_id):\\n"
        " Path(directory, 'status.json').write_text(json.dumps("
        "{'state':'failed','run_id':run_id}))\\n",
        encoding="utf-8",
    )
    path = importlib.resources.files("colabcola").joinpath("bridge_colab_comm_runtime.py")
    with pytest.raises(RuntimeError, match="initialization failed"):
        runpy.run_path(str(path), run_name="__main__", init_globals={
            "BRIDGE_DIRECTORY": str(root), "BRIDGE_RUN_ID": "c" * 32,
            "BRIDGE_FPS": 6, "BRIDGE_RENDERER_PATH": str(renderer),
        })
    report = json.loads((root / "bridge-bootstrap-diagnostic.json").read_text())
    assert report == {"schema": 1, "run_id": "c" * 32,
                      "state": "failed", "stage": "waiting_cuda",
                      "error_type": "RuntimeError"}
