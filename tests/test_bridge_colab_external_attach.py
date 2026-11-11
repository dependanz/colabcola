"""v33u: browser-provisioned T4 attach is explicit, non-owning and fail-closed."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from colabcola import bridge_colab, bridge_colab_attach
from colabcola.lifecycle_io import LifecycleError


def opts(tmp_path, **kwargs):
    params = dict(
        profile="example", run_dir=tmp_path / "external-run",
        transport="comm", video_codec="h264", install_remote_pyav=False,
        benchmark_h264_capacity=True, h264_capacity_phase_seconds=10,
        fps=30, poll_hz=30, inflight=1, max_seconds=300,
        confirm_allocation=False, confirm_attach=True,
        attach_existing=True, preflight_only=False, viewer_stats=False,
    )
    params.update(kwargs)
    return argparse.Namespace(**params)


class FakeProvider:
    def __init__(self, *, items=None):
        if items is None:
            items = [SimpleNamespace(endpoint="browser-assignment", accelerator="T4")]
        self.items = items
        self.allocations = self.releases = self.unassigns = 0
        self.session_calls = 0

    def assignments(self):
        return self.items

    def allocate(self, *args, **kwargs):
        self.allocations += 1
        raise AssertionError("external attachment allocated")

    def release(self, *args, **kwargs):
        self.releases += 1
        raise AssertionError("external attachment released a GPU")

    def _session(self, endpoint):
        self.session_calls += 1
        assert endpoint == "browser-assignment"
        return SimpleNamespace(url="https://example.invalid", token="private-test-token")


@pytest.mark.parametrize("items", [
    [], [SimpleNamespace(endpoint="x", accelerator="T4"),
         SimpleNamespace(endpoint="y", accelerator="T4")],
    [SimpleNamespace(endpoint="browser-assignment", accelerator="L4")],
    [SimpleNamespace(endpoint="", accelerator="T4")],
])
def test_assignment_selection_fails_closed(items):
    with pytest.raises(LifecycleError):
        bridge_colab_attach.select_single_t4(FakeProvider(items=items))


def test_existing_kernel_inventory_requires_exactly_one(monkeypatch):
    calls = []
    class Response:
        status_code = 200
        def __init__(self, listing):
            self.listing = listing
        def json(self):
            return self.listing

    import requests
    monkeypatch.setattr(requests, "get", lambda url, **kw: (
        calls.append((url, kw)) or Response([{"id": "existing-kernel"}])
    ))
    p = FakeProvider()
    session = p._session("browser-assignment")
    assert bridge_colab_attach.single_kernel_id(p, session) == "existing-kernel"
    assert calls[0][0].endswith("/api/kernels")
    assert calls[0][1]["timeout"] == (10, 40)
    assert calls[0][1]["headers"]["X-Colab-Runtime-Proxy-Token"] == "private-test-token"
    monkeypatch.setattr(requests, "get", lambda *_a, **_kw: Response([{"id": "a"}, {"id": "b"}]))
    with pytest.raises(LifecycleError, match="exactly one"):
        bridge_colab_attach.single_kernel_id(p, session)


def test_attach_preflight_is_read_only_and_creates_no_journal(tmp_path, monkeypatch):
    p = FakeProvider()
    monkeypatch.setattr(bridge_colab_attach, "single_kernel_id",
                        lambda *_: "existing-kernel")
    assert bridge_colab_attach.run_attached_colab(
        opts(tmp_path, preflight_only=True, confirm_attach=False), p) == 0
    assert not (tmp_path / "external-run").exists()
    assert p.allocations == p.releases == p.unassigns == 0


def test_live_attach_closes_only_comm_and_never_runtime(tmp_path, monkeypatch):
    p = FakeProvider()
    monkeypatch.setattr(bridge_colab_attach, "single_kernel_id",
                        lambda *_: "existing-kernel")
    from colabcola import bridge_comm
    records = []
    class Peer:
        def __init__(self, *args, **kwargs):
            records.append(("init", kwargs))
        def open(self, **kwargs):
            records.append(("open", kwargs))
            return {"gpu": "Tesla T4"}
        def close(self):
            records.append(("close",))
    monkeypatch.setattr(bridge_comm, "KernelCommBridge", Peer)
    monkeypatch.setattr(bridge_colab_attach, "run_viewer",
                        lambda *args, **kwargs: records.append(("viewer", kwargs)))
    assert bridge_colab_attach.run_attached_colab(opts(tmp_path), p) == 0
    assert [x[0] for x in records] == ["init", "open", "viewer", "close"]
    assert records[1][1]["expected_kernel_id"] == "existing-kernel"
    assert records[0][1]["install_remote_pyav"] is False
    state = json.loads((tmp_path / "external-run" / "state.json").read_text())
    assert state["phase"] == "detached"
    assert state["backend"] == "google-colab-external"
    assert state["ownership"] == "external"
    assert state["close_status"] == "host_disconnected"
    assert "allocation_id" not in state and "endpoint" not in state
    assert "private-test-token" not in str(state)
    assert p.allocations == p.releases == p.unassigns == 0
    # The existing bridge cleanup must never release an external runtime.
    with pytest.raises(LifecycleError, match="user-owned"):
        bridge_colab.read_journal(tmp_path / "external-run")


def test_ambiguous_bootstrap_needs_review_not_runtime_release(tmp_path, monkeypatch):
    p = FakeProvider()
    monkeypatch.setattr(bridge_colab_attach, "single_kernel_id",
                        lambda *_: "existing-kernel")
    from colabcola import bridge_comm
    class Peer:
        def __init__(self, *a, **k):
            pass
        def open(self, **k):
            raise LifecycleError("test bootstrap failed")
        def close(self):
            pass
    monkeypatch.setattr(bridge_comm, "KernelCommBridge", Peer)
    with pytest.raises(LifecycleError, match="bootstrap failed"):
        bridge_colab_attach.run_attached_colab(opts(tmp_path), p)
    state = json.loads((tmp_path / "external-run" / "state.json").read_text())
    assert state["phase"] == "needs_manual_review"
    assert p.allocations == p.releases == p.unassigns == 0


def test_kernel_identity_checked_before_remote_side_effects():
    from colabcola.bridge_comm import KernelCommBridge
    class Provider:
        def __init__(self):
            self.executed = False
        def _kernel(self, session, kernel_id=None):
            assert kernel_id == "expected"
            return {"id": "changed"}
        def _execute(self, *args):
            self.executed = True
            raise AssertionError("remote code should not execute")
    p = Provider()
    peer = KernelCommBridge(p, object(), "a" * 32, 30, video_codec="h264")
    with pytest.raises(LifecycleError, match="changed before bootstrap"):
        peer.open(expected_kernel_id="expected")
    assert not p.executed


def test_explicit_attach_consent_and_mode_validation(tmp_path):
    p = FakeProvider()
    for overrides in ({"confirm_attach": False}, {"confirm_allocation": True},
                      {"video_codec": "ppm"}, {"poll_hz": 60},
                      {"benchmark_h264_capacity": False}):
        with pytest.raises(LifecycleError):
            bridge_colab_attach.run_attached_colab(opts(tmp_path, **overrides), p)
    assert p.session_calls == 0


def test_cli_attach_flags_are_required_and_do_not_alias_allocation(monkeypatch):
    from colabcola.cli import main
    calls = []
    monkeypatch.setattr(bridge_colab, "run_colab", lambda args: calls.append(args) or 0)
    base = ["--profile", "example", "bridge", "colab",
            "--attach-existing", "--transport", "comm",
            "--video-codec", "h264", "--fps", "30", "--poll-hz", "30",
            "--benchmark-h264-capacity", "--max-seconds", "300"]
    assert main(base + ["--preflight-only"]) == 0
    assert calls[-1].attach_existing is True
    assert main(base + ["--confirm-attach", "--run-dir", "unused"]) == 0
    assert calls[-1].confirm_allocation is False
    with pytest.raises(SystemExit):
        main(base + ["--confirm-allocation", "--confirm-attach", "--run-dir", "unused"])
    with pytest.raises(SystemExit):
        main(base + ["--run-dir", "unused"])
