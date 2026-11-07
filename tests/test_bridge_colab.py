"""Offline, no-Google tests for the managed interactive T4 bridge."""
from __future__ import annotations

from contextlib import nullcontext
import base64
import importlib.resources
from pathlib import Path
import argparse
import json
import struct
import threading
import zlib

import pytest

from colabcola import bridge_colab as bridge
from colabcola.bridge_demo import Scene, render_ppm
from colabcola.lifecycle_io import LifecycleError


def make_frame(*, seq=3, ack=1, width=192, height=120, ppm=None):
    payload = ppm or render_ppm(Scene(), seconds=0)
    return struct.pack("!4sIIII", b"CCB1", seq, ack, width, height) + zlib.compress(payload)


def test_bounded_frame_codec_roundtrip():
    raw = render_ppm(Scene(), 0)
    seq, ack, width, height, output = bridge.unpack_frame(make_frame(ppm=raw))
    assert (seq, ack, width, height) == (3, 1, 192, 120)
    assert output == raw


@pytest.mark.parametrize("payload", [
    b"",
    b"x" * 10,
    struct.pack("!4sIIII", b"BAD!", 1, 0, 192, 120) + zlib.compress(b"X"),
    struct.pack("!4sIIII", b"CCB1", 1, 0, 0, 0) + zlib.compress(b"X"),
    struct.pack("!4sIIII", b"CCB1", 1, 0, 192, 120) + zlib.compress(b"X"),
    make_frame() + b"not-a-valid-trailer",
])
def test_malformed_frames_are_rejected(payload):
    with pytest.raises(LifecycleError):
        bridge.unpack_frame(payload)


def test_bad_compression_bomb_is_bounded():
    compressed = zlib.compress(b"A" * (3 * 1024 * 768 + 1000))
    frame = struct.pack("!4sIIII", b"CCB1", 0, 0, 640, 360) + compressed
    with pytest.raises(LifecycleError):
        bridge.unpack_frame(frame)


class FakeProvider:
    instances = []

    def __init__(self):
        self.released = []
        self.endpoint = None
        self.allocations = []
        self.closed = False
        self.instances.append(self)

    def assignments(self):
        return []

    def allocate(self, allocation_id, gpu):
        self.allocations.append((allocation_id, gpu))
        self.endpoint = "owned-t4"
        return self.endpoint

    def reconcile(self, allocation_id, gpu):
        return self.endpoint

    def lookup(self, endpoint):
        return endpoint if self.endpoint == endpoint else None

    def release(self, endpoint):
        assert endpoint == self.endpoint
        self.released.append(endpoint)
        self.endpoint = None
        return {"state": "absent"}

    def close(self):
        self.closed = True


def args(tmp_path, **overrides):
    data = {
        "profile": "example", "preflight_only": False,
        "run_dir": tmp_path / "owned-bridge-run",
        "confirm_allocation": True, "max_seconds": 120,
        "fps": 6, "poll_hz": 3,
    }
    data.update(overrides)
    return argparse.Namespace(**data)


def patch_provider(monkeypatch):
    FakeProvider.instances.clear()
    monkeypatch.setattr(bridge, "GoogleProvider", FakeProvider)
    import colabcola.profiles
    monkeypatch.setattr(colabcola.profiles, "selected", lambda name: nullcontext())


def test_preflight_does_not_allocate_or_create_run_dir(tmp_path, monkeypatch):
    patch_provider(monkeypatch)
    opts = args(tmp_path, preflight_only=True, confirm_allocation=False, run_dir=None)
    assert bridge.run_colab(opts) == 0
    assert not FakeProvider.instances[-1].allocations
    assert FakeProvider.instances[-1].closed


def test_run_allocates_and_releases_only_its_owned_endpoint(tmp_path, monkeypatch):
    patch_provider(monkeypatch)
    monkeypatch.setattr(bridge, "wait_for_session", lambda provider, endpoint, deadline: object())
    monkeypatch.setattr(bridge, "start_worker", lambda provider, session, run_id, fps: (
        "content/colabcola-bridge/" + run_id,
        {"gpu": "Tesla T4", "device": "cuda", "width": 640, "height": 360},
    ))
    monkeypatch.setattr(bridge, "run_viewer", lambda *a, **k: None)
    opts = args(tmp_path)
    assert bridge.run_colab(opts) == 0
    instance = FakeProvider.instances[-1]
    assert instance.released == ["owned-t4"]
    assert instance.closed
    journal = json.loads((opts.run_dir / "state.json").read_text())
    assert journal["phase"] == "released"
    assert journal["endpoint"] == "owned-t4"
    assert journal["gpu_name"] == "Tesla T4"
    assert "access_token" not in journal


def test_bootstrap_failure_releases_owned_runtime(tmp_path, monkeypatch):
    patch_provider(monkeypatch)
    monkeypatch.setattr(bridge, "wait_for_session", lambda *a: object())

    def fail(*a):
        raise LifecycleError("remote GPU not ready")

    monkeypatch.setattr(bridge, "start_worker", fail)
    with pytest.raises(LifecycleError, match="not ready"):
        bridge.run_colab(args(tmp_path))
    instance = FakeProvider.instances[-1]
    assert instance.released == ["owned-t4"]
    assert json.loads((tmp_path / "owned-bridge-run" / "state.json").read_text())["phase"] == "released"


def test_cleanup_refuses_to_release_reconciled_mismatch(tmp_path):
    directory = tmp_path / "journal"
    directory.mkdir()
    state = {
        "schema": 1, "backend": "google-colab", "gpu": "T4",
        "profile": "example", "run_id": "a" * 32,
        "allocation_id": "00000000-0000-4000-8000-000000000001",
        "phase": "allocated", "endpoint": "original-owner",
    }
    bridge.save(directory, state, "allocated")
    provider = FakeProvider()
    provider.endpoint = "different-owner"
    assert not bridge.release_owned(provider, directory, state)
    assert not provider.released
    assert bridge.read_journal(directory)["phase"] == "needs_cleanup"


def test_colab_cli_profile_dispatch_without_live_google(monkeypatch):
    from colabcola.cli import main as cli_main
    called = {}

    def offline(args):
        called["profile"] = args.profile
        called["preflight_only"] = args.preflight_only
        return 0

    monkeypatch.setattr(bridge, "run_colab", offline)
    assert cli_main(["--profile", "example", "bridge", "colab", "--preflight-only"]) == 0
    assert called == {"profile": "example", "preflight_only": True}


def test_packaged_remote_worker_compiles():
    source = importlib.resources.files("colabcola").joinpath(
        "bridge_colab_runtime.py"
    ).read_text(encoding="utf-8")
    compile(source, "bridge_colab_runtime.py", "exec")


def test_authenticated_remote_frame_fetch():
    payload = make_frame()
    class Transport:
        def _contents(self, session, method, path):
            assert method == "GET"
            assert path.endswith("frame.bin")
            return {"type": "file", "format": "base64", "content": base64.b64encode(payload).decode()}
    assert bridge.remote_file(Transport(), object(), "content/bridge/frame.bin") == payload


def test_boot_uploads_worker_to_owned_colab_kernel():
    rid = "a" * 32
    class Transport:
        def __init__(self):
            self.commands = []
            self.files = []
        def _execute(self, session, code):
            self.commands.append(code)
        def _upload(self, session, path, content):
            self.files.append((path, content))
        def _json_file(self, session, path):
            assert path == f"content/colabcola-bridge/{rid}/status.json"
            return {
                "run_id": rid, "state": "ready", "device": "cuda",
                "gpu": "Tesla T4", "width": 640, "height": 360,
            }

    provider = Transport()
    remote, facts = bridge.start_worker(provider, object(), rid, 6)
    assert remote == f"content/colabcola-bridge/{rid}"
    assert facts["device"] == "cuda" and facts["gpu"] == "Tesla T4"
    assert len(provider.commands) == 2
    assert "runpy.run_path" in provider.commands[1]
    assert "BRIDGE_RUN_ID" in provider.commands[1]
    compile(provider.commands[1], "remote_bootstrap.py", "exec")
    assert len(provider.files) == 1
    assert provider.files[0][0] == remote + "/bridge_colab_runtime.py"
    compile(provider.files[0][1], "uploaded_worker.py", "exec")


def test_remote_frame_accepts_wrapped_base64_contents():
    payload = make_frame()
    encoded = base64.b64encode(payload).decode("ascii")
    folded = (chr(13) + chr(10)).join(encoded[i:i + 76] for i in range(0, len(encoded), 76))
    assert bridge.decode_remote_file_record({
        "type": "file", "format": "base64", "content": folded,
    }) == payload


@pytest.mark.parametrize("bad", [
    None,
    {"type": "file", "format": "base64", "content": None},
    {"type": "file", "format": "text", "content": "test"},
    {"type": "directory", "format": "base64", "content": ""},
    {"type": "file", "format": "base64", "content": "not base64!"}
])
def test_remote_frame_encoding_errors_are_sanitized(bad):
    with pytest.raises(LifecycleError) as exc:
        bridge.decode_remote_file_record(bad)
    assert "not base64!" not in str(exc.value)
    assert "remote frame" in str(exc.value)


def test_remote_frame_rejects_oversized_encoded_body():
    with pytest.raises(LifecycleError, match="size limit"):
        bridge.decode_remote_file_record({
            "type": "file", "format": "base64", "content": "A" * (3 * 1024 * 1024)
        })
