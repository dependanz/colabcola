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
        self._on_open = lambda socket: self.connection_ready.set()
        self._on_close = lambda socket, code, message: self.connection_ready.clear()
        self._run_websocket = lambda: None
        self.connection_thread = None
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
    run_id = "d" * 32
    renderer_metrics = {
        "schema": 1, "run_id": run_id, "seq": 3,
        "renderer_fps": 5.9, "render_copy_ms": 13.5,
        "compress_ms": 2.0, "publish_ms": 0.5,
    }
    (root / "render-metrics.json").write_text(json.dumps(renderer_metrics))
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
    def modern_api_must_not_be_used(**kw):
        raise AssertionError("Colab-documented ipykernel Comm should take precedence")
    package.create_comm = modern_api_must_not_be_used
    monkeypatch.setitem(sys.modules, "comm", package)
    kernel = ModuleType("ipykernel")
    legacy = ModuleType("ipykernel.comm")
    legacy.Comm = lambda **kw: FakeComm(**kw)
    kernel.comm = legacy
    monkeypatch.setitem(sys.modules, "ipykernel", kernel)
    monkeypatch.setitem(sys.modules, "ipykernel.comm", legacy)
    path = importlib.resources.files("colabcola").joinpath("bridge_colab_comm_runtime.py")
    env = runpy.run_path(str(path), run_name="test_comm")
    env["start"](str(root), run_id, 6, str(fake_renderer))
    channel = channels[0]
    assert channel.open_message["data"]["gpu"] == "Tesla T4"
    channel.callback({"content": {"data": {"v": 1, "kind": "input", "run_id": run_id,
                                           "event": "mouse", "x": .5, "y": .3, "seq": 1}}})
    assert json.loads((root / "input.json").read_text())["seq"] == 1
    channel.callback({"content": {"data": {"v": 1, "kind": "pull", "run_id": run_id}}})
    assert len(channel.replies) == 1
    assert base64.b64decode(channel.replies[0]["data"], validate=True) == rendered
    assert channel.replies[0]["metrics"] == renderer_metrics
    channel.callback({"content": {"data": {"v": 1, "kind": "rate", "run_id": run_id,
                                           "seq": 1, "fps": 12}}})
    assert json.loads((root / "rate-control.json").read_text()) == {
        "run_id": run_id, "seq": 1, "fps": 12,
    }
    assert channel.replies[-1]["kind"] == "rate_ack"
    assert channel.replies[-1]["seq"] == 1
    channel.callback({"content": {"data": {"v": 1, "kind": "rate", "run_id": run_id,
                                           "seq": 2, "fps": 999}}})
    channel.callback({"content": {"data": {"v": 1, "kind": "rate", "run_id": "wrong",
                                           "seq": 3, "fps": 6}}})
    assert json.loads((root / "rate-control.json").read_text())["fps"] == 12
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
        "from pathlib import Path\n"
        "import json\n"
        "def start(directory, fps, run_id):\n"
        " Path(directory, 'status.json').write_text(json.dumps("
        "{'state':'failed','run_id':run_id}))\n",
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


def test_bootstrap_failure_reports_stage_without_exception_value(monkeypatch):
    from colabcola import bridge_comm
    run_id = "e" * 32
    wire = FakeWire()
    wire.shell_channel.messages.append({
        "header": {"msg_type": "execute_reply"},
        "parent_header": {"msg_id": "request-1"},
        "content": {"status": "error", "ename": "ModuleNotFoundError",
                    "evalue": "SENSITIVE_TOKEN_SHOULD_NOT_BE_PRINTED"},
    })

    class FakeClient:
        def __init__(self, **kwargs):
            self._manager = SimpleNamespace(client=wire)
            self.stopped = False
        def start(self):
            pass
        def stop(self):
            self.stopped = True

    clients = []
    def instantiate(_base):
        def build(**kwargs):
            result = FakeClient(**kwargs)
            clients.append(result)
            return result
        return build
    monkeypatch.setattr(bridge_comm, "safe_client_class", instantiate)

    class FakeProvider:
        def _execute(self, session, code):
            pass
        def _upload(self, session, name, body):
            pass
        def _kernel(self, session):
            return {"id": "owned"}
        def _json_file(self, session, path):
            return {"schema": 1, "run_id": run_id, "state": "failed",
                    "stage": "creating_comm", "error_type": "ModuleNotFoundError"}

    session = SimpleNamespace(url="https://example.invalid", token="LOCAL_TEST_TOKEN")
    peer = bridge_comm.KernelCommBridge(FakeProvider(), session, run_id, 6)
    with pytest.raises(LifecycleError, match="stage=creating_comm, error_type=ModuleNotFoundError") as error:
        peer.open()
    assert "SENSITIVE_TOKEN" not in str(error.value)
    assert clients and clients[0].stopped


def test_websocket_probe_filters_sensitive_errors():
    from colabcola.bridge_ws_probe import KernelWebSocketProbe
    wire = FakeWire()
    wire.connection_ready.clear()
    probe = KernelWebSocketProbe()
    probe.install(wire)
    wire._on_open(None)
    wire._on_close(None, 1006, "secret close content")
    assert probe.opened and probe.closed
    assert probe.close_code == 1006
    text = probe.summary(wire)
    assert "opened=1" in text and "connected=0" in text
    assert "secret" not in text
    wire._on_open(None)
    assert wire.connection_ready.is_set()


def test_comm_bridge_failed_websocket_never_submits_and_cleans_both(monkeypatch):
    """A failed connection may be retried, but never execute workload code."""
    from colabcola import bridge_comm

    class NotReady:
        def is_set(self):
            return False
        def wait(self, timeout=None):
            return False

    clients = []
    class FakeClient:
        def __init__(self, **kwargs):
            self._manager = SimpleNamespace(client=FakeWire())
            self._manager.client.connection_ready = NotReady()
            self.stopped = False
            clients.append(self)
        def start(self):
            pass
        def stop(self):
            self.stopped = True

    monkeypatch.setattr(bridge_comm, "safe_client_class", lambda base: FakeClient)
    monkeypatch.setattr(bridge_comm.time, "sleep", lambda duration: None)

    class Provider:
        def _execute(self, session, code):
            pass
        def _upload(self, session, name, content):
            pass
        def _kernel(self, session):
            return {"id": "owned-kernel"}

    peer = bridge_comm.KernelCommBridge(
        Provider(), SimpleNamespace(url="https://example.invalid", token="VERY_SENSITIVE_TOKEN"),
        "e" * 32, 6
    )
    with pytest.raises(LifecycleError, match="did not open after 2 pre-execution connections") as error:
        peer.open()
    assert len(clients) == 2
    assert all(client.stopped for client in clients)
    assert all(not client._manager.client.commands for client in clients)
    assert "VERY_SENSITIVE_TOKEN" not in str(error.value)
    assert "opened=0" in str(error.value)


def test_websocket_probe_records_only_error_class():
    from colabcola.bridge_ws_probe import KernelWebSocketProbe
    wire = FakeWire()
    probe = KernelWebSocketProbe()
    probe.install(wire)
    socket = SimpleNamespace(on_error=None)
    wire.kernel_socket = socket
    wire._run_websocket()
    assert callable(socket.on_error)
    socket.on_error(None, TimeoutError("sensitive-message-never-echoed"))
    summary = probe.summary(wire)
    assert "error_type=TimeoutError" in summary
    assert "sensitive-message" not in summary


def test_comm_bridge_retries_attachment_but_executes_once(monkeypatch):
    from colabcola import bridge_comm
    rid = "9" * 32
    details = {"v": 1, "kind": "ready", "run_id": rid,
               "gpu": "Tesla T4", "device": "cuda", "width": 640, "height": 360}
    opened = {"header": {"msg_type": "comm_open"},
              "parent_header": {"msg_id": "request-1"},
              "content": {"comm_id": "ok-comm", "target_name": TARGET, "data": details}}
    first = FakeWire()
    first.connection_ready.clear()
    second = FakeWire((opened,))

    class NotReady:
        def is_set(self):
            return False
        def wait(self, timeout=None):
            return False

    first.connection_ready = NotReady()
    wires = [first, second]
    created = []
    class FakeClient:
        def __init__(self, **kwargs):
            self._manager = SimpleNamespace(client=wires[len(created)])
            self.stopped = False
            created.append(self)
        def start(self):
            pass
        def stop(self):
            self.stopped = True
    monkeypatch.setattr(bridge_comm, "safe_client_class", lambda _: FakeClient)
    monkeypatch.setattr(bridge_comm.time, "sleep", lambda duration: None)

    class Provider:
        def _execute(self, session, code):
            pass
        def _upload(self, session, path, content):
            pass
        def _kernel(self, session):
            return {"id": "owned"}
    peer = bridge_comm.KernelCommBridge(
        Provider(), SimpleNamespace(url="https://example.invalid", token="hidden"), rid, 6
    )
    assert peer.open() == details
    assert len(created) == 2 and created[0].stopped and not created[1].stopped
    assert first.commands == []
    assert len(second.commands) == 1
    peer.close()
    assert created[1].stopped


def test_cli_accepts_12hz_comm_but_preserves_file_polling_limit(monkeypatch):
    from colabcola.cli import main
    from colabcola import bridge_colab
    selected = []
    monkeypatch.setattr(bridge_colab, "run_colab", lambda opts: selected.append(
        (opts.transport, opts.fps, opts.poll_hz)) or 0)
    assert main(["--profile", "example", "bridge", "colab", "--preflight-only",
                 "--transport", "comm", "--fps", "12", "--poll-hz", "12"]) == 0
    assert selected == [("comm", 12, 12)]
    with pytest.raises(SystemExit) as exc:
        main(["--profile", "example", "bridge", "colab", "--preflight-only",
              "--transport", "files", "--fps", "12", "--poll-hz", "12"])
    assert exc.value.code == 2


def test_kernel_comm_frame_rtt_and_freshness_instrumentation(monkeypatch, capsys):
    """Exercise the reporting loop without a GPU, file API or elapsed wall time."""
    from colabcola import bridge_comm
    from types import SimpleNamespace
    from itertools import count

    ticks = count(0)
    clock = SimpleNamespace(
        monotonic=lambda: next(ticks) / 10,
        sleep=lambda seconds: None,
    )
    monkeypatch.setattr(bridge_comm, "time", clock)
    stopped = threading.Event()
    outgoing = queue.Queue()
    frames = queue.Queue(maxsize=2)

    class FakePeer:
        def __init__(self):
            self.requests = 0
            self.received = 0
        def request_frame(self):
            self.requests += 1
        def send_input(self, seq, event):
            raise AssertionError("unexpected input")
        def next_frame(self, seconds):
            self.received += 1
            if self.received >= 15:
                stopped.set()
            # Duplicate frames should not count as new delivered frames.
            return self.received // 2, 0, b"test", 1400

    peer = FakePeer()
    bridge_comm.stream_comm(peer, "Tesla T4", outgoing, frames,
                            stopped, deadline=100, poll_hz=12)
    output = capsys.readouterr().out
    assert "pull RTT p50" in output
    assert "fresh" in output
    assert "kernel Comm" in output
    assert peer.requests == peer.received == 15


def test_kernel_comm_repeated_stale_frames_trigger_bounded_failure(monkeypatch):
    from colabcola import bridge_comm
    from types import SimpleNamespace
    from itertools import count

    ticks = count(0)
    monkeypatch.setattr(bridge_comm, "time", SimpleNamespace(
        monotonic=lambda: next(ticks) / 2,
        sleep=lambda seconds: None,
    ))
    stopped = threading.Event()
    frames = queue.Queue(maxsize=2)

    class StalePeer:
        def request_frame(self):
            pass
        def next_frame(self, seconds):
            return 0, 0, b"frame", 1400

    with pytest.raises(LifecycleError, match="kernel Comm transport interrupted"):
        bridge_comm.stream_comm(
            StalePeer(), "Tesla T4", queue.Queue(), frames,
            stopped, deadline=300, poll_hz=12)
    assert stopped.is_set()


def test_renderer_metrics_reject_unknown_identity_and_nonnumeric_values():
    from colabcola.bridge_comm import validated_renderer_metrics
    rid = "c" * 32
    valid = {"schema": 1, "run_id": rid, "seq": 14,
             "renderer_fps": 11.7, "render_copy_ms": 21.4,
             "compress_ms": 3.2, "publish_ms": 0.9}
    assert validated_renderer_metrics(valid, rid) == {
        "seq": 14, "renderer_fps": 11.7, "render_copy_ms": 21.4,
        "compress_ms": 3.2, "publish_ms": 0.9,
    }
    assert validated_renderer_metrics({**valid, "run_id": "wrong"}, rid) is None
    assert validated_renderer_metrics({**valid, "seq": True}, rid) is None
    assert validated_renderer_metrics({**valid, "render_copy_ms": "secret"}, rid) is None
    assert validated_renderer_metrics({**valid, "renderer_fps": float("nan")}, rid) is None
    assert validated_renderer_metrics({**valid, "compress_ms": float("inf")}, rid) is None


def test_comm_frame_uses_sanitized_remote_renderer_telemetry():
    from colabcola.bridge_comm import validated_renderer_metrics
    rid = "b" * 32
    metrics = {"schema": 1, "run_id": rid, "seq": 8,
               "renderer_fps": 8.2, "render_copy_ms": 47.1,
               "compress_ms": 2.1, "publish_ms": 0.2}
    ppm = b"P6" + bytes((10,)) + b"640 360" + bytes((10,)) + b"255" + bytes((10,)) + bytes((0, 0, 0)) * (640 * 360)
    binary = struct.pack("!4sIIII", b"CCB1", 9, 0, 640, 360) + zlib.compress(ppm)
    message = comm_msg({
        "v": 1, "run_id": rid, "kind": "frame",
        "data": base64.b64encode(binary).decode("ascii"), "metrics": metrics,
    })
    peer = KernelCommBridge(None, None, rid, 6)
    peer.wire = FakeWire((message,))
    peer.comm_id = "my-comm"
    assert peer.next_frame(.5)[0] == 9
    assert peer.latest_renderer_metrics == validated_renderer_metrics(metrics, rid)


def test_kernel_comm_frame_reader_handles_rate_ack_and_rate_metrics():
    from colabcola.bridge_comm import validated_renderer_metrics
    rid = "b" * 32
    ppm = (b"P6" + bytes((10,)) + b"640 360" + bytes((10,)) +
           b"255" + bytes((10,)) + bytes((20, 30, 40)) * (640 * 360))
    binary = struct.pack("!4sIIII", b"CCB1", 4, 0, 640, 360) + zlib.compress(ppm)
    metrics = {"schema": 1, "run_id": rid, "seq": 4,
               "renderer_fps": 11.9, "render_copy_ms": 1.0,
               "compress_ms": 7.0, "publish_ms": 0.2,
               "target_fps": 12, "rate_seq": 1}
    rate_ack = comm_msg({"v": 1, "kind": "rate_ack", "run_id": rid,
                         "seq": 1, "fps": 12})
    frame = comm_msg({"v": 1, "kind": "frame", "run_id": rid,
                      "data": base64.b64encode(binary).decode(), "metrics": metrics})
    peer = KernelCommBridge(None, None, rid, 6)
    peer.wire = FakeWire((rate_ack, frame))
    peer.comm_id = "my-comm"
    assert peer.next_frame(.5)[0] == 4
    assert peer.latest_rate_ack == 1
    assert peer.latest_renderer_metrics == validated_renderer_metrics(metrics, rid)
    peer.set_render_fps(2, 6)
    assert peer.wire.shell_channel.sent[-1]["content"]["data"] == {
        "v": 1, "kind": "rate", "run_id": rid, "seq": 2, "fps": 6
    }
    with pytest.raises(ValueError):
        peer.set_render_fps(3, 100)


def test_tiny_comm_ping_rtt_and_identity_filtering(monkeypatch):
    from colabcola import bridge_comm
    rid = "b" * 32
    messages = [comm_msg({"v": 1, "kind": "pong", "run_id": "wrong", "seq": 1})]
    for i in range(1, 9):
        messages.append(comm_msg({"v": 1, "kind": "pong", "run_id": rid, "seq": i}))
    peer = KernelCommBridge(None, None, rid, 6)
    peer.wire = FakeWire(messages)
    peer.comm_id = "my-comm"
    assert isinstance(peer.probe_small_message_rtt(8), int)
    outbound = [m["content"]["data"] for m in peer.wire.shell_channel.sent]
    assert [(m["kind"], m["seq"]) for m in outbound] == [("ping", i) for i in range(1, 9)]
    with pytest.raises(ValueError):
        peer.probe_small_message_rtt(17)


def test_remote_comm_handler_replies_to_bounded_tiny_ping(tmp_path, monkeypatch):
    import runpy
    import sys
    from types import ModuleType

    rid = "d" * 32
    root = tmp_path / "owned"
    root.mkdir()
    fake_renderer = tmp_path / "fake_renderer.py"
    fake_renderer.write_text(
        "from pathlib import Path\n"
        "import json\n"
        "def start(directory, fps, run_id):\n"
        " Path(directory, 'status.json').write_text(json.dumps("
        "{'state':'ready','run_id':run_id,'device':'cuda','gpu':'Tesla T4',"
        "'width':640,'height':360}))\n"
        "def atomic_bytes(path, body):\n"
        " Path(path).write_bytes(body)\n"
    )
    created = []
    class FakeComm:
        def __init__(self, **kwargs):
            self.messages = []
            created.append(self)
        def on_msg(self, callback):
            self.callback = callback
        def send(self, message):
            self.messages.append(message)
        def close(self):
            pass
    kernel = ModuleType("ipykernel")
    kernelcomm = ModuleType("ipykernel.comm")
    kernelcomm.Comm = FakeComm
    kernel.comm = kernelcomm
    monkeypatch.setitem(sys.modules, "ipykernel", kernel)
    monkeypatch.setitem(sys.modules, "ipykernel.comm", kernelcomm)
    path = importlib.resources.files("colabcola").joinpath("bridge_colab_comm_runtime.py")
    runpy.run_path(str(path), run_name="__main__", init_globals={
        "BRIDGE_DIRECTORY": str(root), "BRIDGE_RUN_ID": rid,
        "BRIDGE_FPS": 6, "BRIDGE_RENDERER_PATH": str(fake_renderer),
    })
    peer = created[0]
    peer.callback({"content": {"data": {"v": 1, "run_id": rid, "kind": "ping", "seq": 3}}})
    assert peer.messages == [{"v": 1, "run_id": rid, "kind": "pong", "seq": 3}]
    peer.callback({"content": {"data": {"v": 1, "run_id": rid, "kind": "ping", "seq": True}}})
    peer.callback({"content": {"data": {"v": 1, "run_id": "wrong", "kind": "ping", "seq": 4}}})
    assert len(peer.messages) == 1


def test_cli_requires_explicit_comm_probe_opt_in(monkeypatch):
    from colabcola.cli import main
    from colabcola import bridge_colab
    calls = []
    monkeypatch.setattr(bridge_colab, "run_colab", lambda args: calls.append(args.probe_link) or 0)
    base = ["--profile", "example", "bridge", "colab", "--preflight-only"]
    assert main(base + ["--transport", "comm", "--probe-link"]) == 0
    assert calls == [True]
    for args in (["--probe-link"], ["--transport", "comm", "--probe-link", "--benchmark-sweep"]):
        with pytest.raises(SystemExit) as error:
            main(base + args)
        assert error.value.code == 2


def test_comm_tagged_frame_echo_and_strict_pipeline_decoder():
    rid = "b" * 32
    ppm = (b"P6" + bytes((10,)) + b"640 360" + bytes((10,)) +
           b"255" + bytes((10,)) + bytes((14, 22, 35)) * (640 * 360))
    binary = struct.pack("!4sIIII", b"CCB1", 18, 0, 640, 360) + zlib.compress(ppm)
    data = {"v": 1, "kind": "frame", "run_id": rid, "pull_seq": 7,
            "data": base64.b64encode(binary).decode()}
    peer = KernelCommBridge(None, None, rid, 12)
    peer.wire = FakeWire((comm_msg(data),))
    peer.comm_id = "my-comm"
    assert peer.poll_pipeline_frame(.1) == (7, 18, 0, ppm, len(binary))
    peer.request_frame(8)
    assert peer.wire.shell_channel.sent[-1]["content"]["data"]["pull_seq"] == 8

    bad = KernelCommBridge(None, None, rid, 12)
    bad.wire = FakeWire((comm_msg({**data, "pull_seq": True}),))
    bad.comm_id = "my-comm"
    with pytest.raises(LifecycleError, match="invalid request ID"):
        bad.poll_pipeline_frame(.1)


def test_remote_handler_responds_with_request_ID_without_breaking_legacy(tmp_path, monkeypatch):
    from types import ModuleType
    import runpy
    rid = "e" * 32
    folder = tmp_path / "kernel"
    folder.mkdir()
    (folder / "frame.bin").write_bytes(packed_frame())
    renderer = tmp_path / "renderer.py"
    renderer.write_text(
        "from pathlib import Path\n"
        "import json\n"
        "def start(directory, fps, run_id):\n"
        " Path(directory, 'status.json').write_text(json.dumps("
        "{'state':'ready','run_id':run_id,'device':'cuda','gpu':'Tesla T4',"
        "'width':640,'height':360}))\n"
        "def atomic_bytes(path, body):\n"
        " Path(path).write_bytes(body)\n"
    )
    created = []
    class FakeComm:
        def __init__(self, **kwargs):
            self.sent = []
            created.append(self)
        def on_msg(self, callback):
            self.callback = callback
        def send(self, message):
            self.sent.append(message)
        def close(self):
            pass
    kernel = ModuleType("ipykernel")
    kernelcomm = ModuleType("ipykernel.comm")
    kernelcomm.Comm = FakeComm
    kernel.comm = kernelcomm
    monkeypatch.setitem(sys.modules, "ipykernel", kernel)
    monkeypatch.setitem(sys.modules, "ipykernel.comm", kernelcomm)
    path = importlib.resources.files("colabcola").joinpath("bridge_colab_comm_runtime.py")
    runpy.run_path(str(path), run_name="__main__", init_globals={
        "BRIDGE_DIRECTORY": str(folder), "BRIDGE_RUN_ID": rid,
        "BRIDGE_FPS": 6, "BRIDGE_RENDERER_PATH": str(renderer),
    })
    comm = created[0]
    def deliver(request):
        comm.callback({"content": {"data": {"v": 1, "run_id": rid, **request}}})
    deliver({"kind": "pull", "pull_seq": 8})
    assert comm.sent[-1]["pull_seq"] == 8
    deliver({"kind": "pull"})
    assert "pull_seq" not in comm.sent[-1]
    previous = len(comm.sent)
    deliver({"kind": "pull", "pull_seq": True})
    assert len(comm.sent) == previous
