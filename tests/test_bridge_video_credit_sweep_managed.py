"""v33l: allocation-free managed H.264 A/B/A credit sweep wiring and ownership."""
from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
import queue
import threading
from types import SimpleNamespace

import pytest


def _command(*args):
    return ["--profile", "example", "bridge", "colab", "--preflight-only", *args]


def test_cli_sweep_is_explicit_and_preserves_standard_video_modes(monkeypatch):
    from colabcola.cli import main
    from colabcola import bridge_colab

    observed = []
    monkeypatch.setattr(bridge_colab, "run_colab", lambda opts: observed.append(opts) or 0)
    common = ["--transport", "comm", "--video-codec", "h264", "--fps", "12",
              "--poll-hz", "12"]
    assert main(_command(*common, "--benchmark-h264-credits")) == 0
    opts = observed[-1]
    assert (opts.inflight, opts.h264_credit_phase_seconds,
            opts.benchmark_h264_credits) == (1, 16, True)
    assert main(_command(*common, "--inflight", "2")) == 0
    assert (observed[-1].inflight, observed[-1].benchmark_h264_credits) == (2, False)
    assert main(_command("--transport", "comm")) == 0
    assert observed[-1].video_codec == "ppm"


@pytest.mark.parametrize("extra", [
    ["--transport", "comm", "--benchmark-h264-credits"],
    ["--transport", "comm", "--video-codec", "h264", "--benchmark-h264-credits"],
    ["--transport", "comm", "--video-codec", "h264", "--benchmark-h264-credits",
     "--fps", "12", "--poll-hz", "11"],
    ["--transport", "comm", "--video-codec", "h264", "--benchmark-h264-credits",
     "--fps", "12", "--poll-hz", "12", "--inflight", "2"],
    ["--transport", "comm", "--video-codec", "h264", "--benchmark-h264-credits",
     "--fps", "12", "--poll-hz", "12", "--probe-link"],
    ["--transport", "comm", "--video-codec", "h264", "--benchmark-h264-credits",
     "--fps", "12", "--poll-hz", "12", "--benchmark-sweep"],
    ["--transport", "comm", "--video-codec", "h264", "--benchmark-h264-credits",
     "--fps", "12", "--poll-hz", "12", "--benchmark-credits"],
    ["--transport", "comm", "--video-codec", "h264", "--benchmark-h264-credits",
     "--fps", "12", "--poll-hz", "12", "--h264-credit-phase-seconds", "5"],
    ["--transport", "comm", "--video-codec", "h264", "--benchmark-h264-credits",
     "--fps", "12", "--poll-hz", "12", "--h264-credit-phase-seconds", "41"],
    ["--transport", "comm", "--video-codec", "h264", "--benchmark-h264-credits",
     "--fps", "12", "--poll-hz", "12", "--max-seconds", "120"],
    ["--h264-credit-phase-seconds", "20"],
])
def test_cli_rejects_unsafe_or_unscoped_sweep_configuration(monkeypatch, extra):
    from colabcola.cli import main
    from colabcola import bridge_colab
    monkeypatch.setattr(bridge_colab, "run_colab",
                        lambda _: pytest.fail("invalid options reached provider"))
    with pytest.raises(SystemExit) as caught:
        main(_command(*extra))
    assert caught.value.code == 2


def _options(run_dir: Path, **changes):
    options = {
        "profile": "example", "preflight_only": False,
        "confirm_allocation": True, "run_dir": run_dir,
        "transport": "comm", "video_codec": "h264", "inflight": 1,
        "benchmark_h264_credits": True, "h264_credit_phase_seconds": 16,
        "benchmark_sweep": False, "benchmark_credits": False,
        "probe_link": False, "install_remote_pyav": False,
        "fps": 12, "poll_hz": 12, "max_seconds": 180,
        "viewer_stats": True,
    }
    options.update(changes)
    return SimpleNamespace(**options)


@pytest.mark.parametrize("field,value", [
    ("transport", "files"), ("video_codec", "ppm"),
    ("inflight", 2), ("fps", 6), ("poll_hz", 6),
    ("h264_credit_phase_seconds", 4),
    ("h264_credit_phase_seconds", 41), ("max_seconds", 120),
    ("benchmark_sweep", True), ("benchmark_credits", True),
    ("probe_link", True),
])
def test_managed_entry_rejects_invalid_sweep_before_preflight_or_allocation(
        tmp_path, monkeypatch, field, value):
    from colabcola import bridge_colab
    import colabcola.profiles

    class Provider:
        def close(self):
            pass
    monkeypatch.setattr(bridge_colab, "GoogleProvider", Provider)
    monkeypatch.setattr(colabcola.profiles, "selected", lambda _: nullcontext())
    monkeypatch.setattr(bridge_colab, "verify_preflight",
                        lambda _: pytest.fail("invalid sweep reached preflight"))
    from colabcola import bridge_video_comm_session
    monkeypatch.setattr(bridge_video_comm_session, "make_host_decoder", lambda: (None, None))
    opts = _options(tmp_path / "not-created", **{field: value})
    from colabcola.lifecycle_io import LifecycleError
    with pytest.raises(LifecycleError, match="credit sweep"):
        bridge_colab.run_colab(opts)
    assert not opts.run_dir.exists()


@pytest.mark.parametrize("outcome", ["success", "bootstrap", "viewer", "worker"])
def test_single_owned_runtime_released_for_each_sweep_outcome(
        tmp_path, monkeypatch, outcome):
    from colabcola import bridge_colab, bridge_comm
    from colabcola import bridge_video_credit_sweep as sweep
    from colabcola import bridge_video_comm_session as video
    from colabcola.lifecycle_io import LifecycleError
    import colabcola.profiles

    class Provider:
        last = None
        def __init__(self):
            self.allocations = []
            self.releases = []
            self.endpoint = None
            self.closed = False
            Provider.last = self
        def allocate(self, allocation_id, gpu):
            assert gpu == "T4"
            self.allocations.append(allocation_id)
            self.endpoint = "owned-aba-t4"
            return self.endpoint
        def reconcile(self, allocation_id, gpu):
            assert allocation_id in self.allocations and gpu == "T4"
            return self.endpoint
        def lookup(self, endpoint):
            return self.endpoint if endpoint == self.endpoint else None
        def release(self, endpoint):
            assert endpoint == self.endpoint == "owned-aba-t4"
            self.releases.append(endpoint)
            self.endpoint = None
            return {"state": "absent"}
        def assignments(self):
            return []
        def close(self):
            self.closed = True

    peers = []
    class Peer:
        def __init__(self, provider, session, run_id, fps, *, video_codec,
                     install_remote_pyav=False):
            assert provider is Provider.last and video_codec == "h264" and fps == 12
            self.closed = False
            peers.append(self)
        def open(self):
            if outcome == "bootstrap":
                raise LifecycleError("simulated H.264 bootstrap error")
            return {"gpu": "Tesla T4", "video_codec": "h264-annexb"}
        def close(self):
            self.closed = True

    calls = []
    def fake_sweep(peer, gpu, outgoing, frames, stopped, deadline, poll_hz, *,
                   phase_seconds, warmup_seconds=2):
        assert peer is peers[-1] and gpu == "Tesla T4"
        calls.append((poll_hz, phase_seconds))
        stopped.set()
        if outcome == "worker":
            raise LifecycleError("simulated sweep worker failure")

    def fake_viewer(*args, **kwargs):
        if outcome == "viewer":
            raise LifecycleError("simulated viewer failure")
        kwargs["worker"](queue.Queue(), queue.Queue(), threading.Event())

    monkeypatch.setattr(bridge_colab, "GoogleProvider", Provider)
    monkeypatch.setattr(bridge_colab, "verify_preflight", lambda _: None)
    monkeypatch.setattr(bridge_colab, "wait_for_session", lambda *args: object())
    monkeypatch.setattr(bridge_colab, "run_viewer", fake_viewer)
    monkeypatch.setattr(bridge_comm, "KernelCommBridge", Peer)
    monkeypatch.setattr(sweep, "run_h264_credit_sweep", fake_sweep)
    monkeypatch.setattr(video, "make_host_decoder", lambda: (None, None))
    monkeypatch.setattr(colabcola.profiles, "selected", lambda _: nullcontext())

    opts = _options(tmp_path / f"sweep-{outcome}")
    if outcome == "success":
        assert bridge_colab.run_colab(opts) == 0
    else:
        with pytest.raises(LifecycleError):
            bridge_colab.run_colab(opts)
    provider = Provider.last
    assert len(provider.allocations) == 1
    assert provider.releases == ["owned-aba-t4"]
    assert provider.closed
    assert peers and peers[-1].closed
    assert calls == ([] if outcome in ("bootstrap", "viewer") else [(12, 16)])
    journal = json.loads((opts.run_dir / "state.json").read_text())
    assert journal["phase"] == "released"
    if outcome != "bootstrap":
        assert journal["video_codec"] == "h264"
