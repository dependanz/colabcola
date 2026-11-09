"""v33f: scoped H.264 bootstrap diagnostics and safe owned-Comm shutdown.

All tests are offline. No Colab account or runtime is accessed.
"""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import queue
import runpy
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from colabcola.lifecycle_io import LifecycleError

RUN = "f" * 32


def _fake_renderer(path: Path) -> str:
    path.write_text(
        "from pathlib import Path\nimport json\n"
        "def start(directory, fps, run_id):\n"
        " Path(directory, 'status.json').write_text(json.dumps(dict("
        "run_id=run_id, state='ready', device='cuda', gpu='Tesla T4', "
        "width=640, height=360)))\n"
        "def atomic_bytes(path, body):\n"
        " Path(path).write_bytes(body)\n",
        encoding="utf-8",
    )
    return str(path)


def _install_fake_kernel(monkeypatch):
    comms = []

    class FakeComm:
        def __init__(self, **kwargs):
            self.ready = kwargs["data"]
            self.closed = False
            comms.append(self)

        def on_msg(self, callback):
            self.callback = callback

        def close(self):
            self.closed = True

        def send(self, _):
            raise AssertionError("no streaming expected during preflight")

    ipykernel = ModuleType("ipykernel")
    module = ModuleType("ipykernel.comm")
    module.Comm = lambda **kwargs: FakeComm(**kwargs)
    ipykernel.comm = module
    monkeypatch.setitem(sys.modules, "ipykernel", ipykernel)
    monkeypatch.setitem(sys.modules, "ipykernel.comm", module)
    return comms


@pytest.mark.parametrize("message,expected", [
    ("remote PyAV unavailable", "pyav_unavailable"),
    ("remote NumPy unavailable", "numpy_unavailable"),
    ("remote libx264 encoder unavailable", "libx264_unavailable"),
    ("unexpected private details", "video_initialization_failed"),
])
def test_remote_bootstrap_reports_allowlisted_code_and_stays_closable(
        tmp_path, monkeypatch, message, expected):
    from colabcola import bridge_colab_comm_runtime as remote

    comms = _install_fake_kernel(monkeypatch)
    renderer = _fake_renderer(tmp_path / "renderer.py")
    source = tmp_path / "failing_video_source.py"
    source.write_text(
        "class RemoteVideoSourceError(RuntimeError):\n pass\n"
        "class RemoteH264PullSource:\n"
        " def __init__(self, *args, **kwargs):\n"
        f"  raise RemoteVideoSourceError({message!r})\n",
        encoding="utf-8",
    )
    comm = remote.start(str(tmp_path), RUN, 12, renderer,
                        video_source_path=str(source), preflight_video=True)
    assert comm is comms[0]
    assert comm.ready["kind"] == "video_preflight_failed"
    assert comm.ready["failure_code"] == expected
    assert comm.ready["run_id"] == RUN
    assert comm.ready["gpu"] == "Tesla T4"
    assert "private" not in str(comm.ready)
    assert "video_codec" not in comm.ready
    comm.callback({"content": {"data": {
        "v": 1, "kind": "close", "run_id": RUN,
    }}})
    assert comm.closed
    assert (tmp_path / "stop").read_bytes() == b"1"


class FakeChannel:
    def __init__(self, messages=()):
        self.messages = deque(messages)
        self.sent = []

    def get_msg(self, timeout=None):
        if not self.messages:
            raise queue.Empty()
        return self.messages.popleft()

    def send(self, message):
        self.sent.append(message)


class FakeWire:
    def __init__(self, ready):
        self.connection_ready = threading.Event()
        self.connection_ready.set()
        self.iopub_channel = FakeChannel([{
            "header": {"msg_type": "comm_open"},
            "parent_header": {"msg_id": "request-1"},
            "content": {"target_name": "colabcola.bridge.v1",
                        "comm_id": "owned-comm", "data": ready},
        }])
        self.shell_channel = FakeChannel()
        self.session = SimpleNamespace(
            msg=lambda kind, content: {"type": kind, "content": content})
        self._on_open = lambda _: self.connection_ready.set()
        self._on_close = lambda *_: self.connection_ready.clear()
        self._run_websocket = lambda: None
        self.connection_thread = None

    def execute(self, code, **kwargs):
        compile(code, "remote_bootstrap.py", "exec")
        return "request-1"


def _host_open(monkeypatch, ready, *, video_codec="h264"):
    from colabcola import bridge_comm

    wire = FakeWire(ready)
    clients = []

    class FakeClient:
        def __init__(self, **kwargs):
            self._manager = SimpleNamespace(client=wire)
            self.stopped = False
            clients.append(self)

        def start(self):
            pass

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(bridge_comm, "safe_client_class", lambda _: FakeClient)

    class FakeProvider:
        def _execute(self, session, code):
            pass

        def _upload(self, session, name, body):
            pass

        def _kernel(self, session):
            return {"id": "owned-kernel"}

    peer = bridge_comm.KernelCommBridge(
        FakeProvider(), SimpleNamespace(url="https://example.invalid", token="SECRET"),
        RUN, 12, video_codec=video_codec)
    return peer, wire, clients


@pytest.mark.parametrize("raw_code,expected", [
    ("pyav_unavailable", "pyav_unavailable"),
    ("numpy_unavailable", "numpy_unavailable"),
    ("libx264_unavailable", "libx264_unavailable"),
    ("token:SECRET", "unknown"),
])
def test_host_propagates_only_safe_code_and_closes_comm(
        monkeypatch, raw_code, expected):
    ready = {"v": 1, "kind": "video_preflight_failed", "run_id": RUN,
             "gpu": "Tesla T4", "device": "cuda", "width": 640, "height": 360,
             "failure_code": raw_code}
    peer, wire, clients = _host_open(monkeypatch, ready)
    with pytest.raises(LifecycleError, match="preflight failed") as e:
        peer.open()
    assert f"code={expected}" in str(e.value)
    assert "SECRET" not in str(e.value)
    assert clients and clients[0].stopped
    assert wire.shell_channel.sent[-1]["content"]["data"]["kind"] == "close"


def test_legacy_ppm_cannot_accept_video_failure_reply(monkeypatch):
    ready = {"v": 1, "kind": "video_preflight_failed", "run_id": RUN,
             "gpu": "Tesla T4", "device": "cuda", "width": 640, "height": 360,
             "failure_code": "pyav_unavailable"}
    peer, wire, clients = _host_open(monkeypatch, ready, video_codec="ppm")
    with pytest.raises(LifecycleError, match="unexpected video preflight reply"):
        peer.open()
    assert clients[0].stopped


def test_legacy_remote_ppm_handshake_remains_ready(tmp_path, monkeypatch):
    from colabcola import bridge_colab_comm_runtime as remote

    comms = _install_fake_kernel(monkeypatch)
    renderer = _fake_renderer(tmp_path / "renderer.py")
    result = remote.start(str(tmp_path), RUN, 12, renderer)
    assert result is comms[0]
    assert result.ready["kind"] == "ready"
    assert "failure_code" not in result.ready
    assert "video_codec" not in result.ready
