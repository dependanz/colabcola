"""v33e: allocation-free managed-video selection, host decode, fail-closed tests."""
from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

import pytest

from colabcola.lifecycle_io import LifecycleError
from colabcola import bridge_video_comm_session as session
from colabcola.bridge_video_stream import VideoAccessUnit


class FakeFrame:
    width, height = 640, 360

    def to_ndarray(self, *, format):
        assert format == "rgb24"
        return SimpleNamespace(tobytes=lambda: bytes((30, 70, 90)) * (640 * 360))


class FakeDecoder:
    def decode(self, packet):
        return [FakeFrame()]


class FakePacket:
    def __init__(self, data):
        self.data = data
        self.pts = None


class FakeReceiver:
    instances = []

    def __init__(self, peer):
        self.peer = peer
        self.requests = []
        self.suppressed_dependent_units = 0
        self.instances.append(self)

    def request_video(self, n):
        self.requests.append(n)

    def next_video(self, seconds):
        n = self.requests[-1]
        unit = VideoAccessUnit(
            seq=n - 1, pts_ns=(n-1) * 83333333, capture_ns=1000 + n,
            data=b"\x00\x00\x00\x01\x65" if n == 1 else b"\x00\x00\x00\x01\x41",
            keyframe=n == 1)
        return unit, self.peer.last_ack


def make_peer():
    class Peer:
        video_codec = "h264"
        last_ack = 0
        sent = []
        def send_input(self, seq, event):
            self.sent.append((seq, event))
            self.last_ack = seq
    return Peer()


def test_host_stream_sends_frames_and_tracks_input_without_remote_clock_subtraction(monkeypatch):
    FakeReceiver.instances.clear()
    monkeypatch.setattr(session, "KernelVideoCommReceiver", FakeReceiver)
    monkeypatch.setattr(session, "make_host_decoder",
                        lambda: (SimpleNamespace(Packet=FakePacket), FakeDecoder()))
    peer = make_peer()
    outgoing = queue.Queue()
    outgoing.put({"event": "key", "key": "Left"})
    frames = queue.Queue(maxsize=64)
    stopped = threading.Event()
    deadline = time.monotonic() + .31
    session.stream_comm_h264(peer, "Tesla T4", outgoing, frames, stopped, deadline, 12)
    assert stopped.is_set()
    assert peer.sent == [(1, {"event": "key", "key": "Left"})]
    items = list(frames.queue)
    payloads = [data for kind, data in items if kind == "frame"]
    assert 2 <= len(payloads) <= 6
    assert all(p.startswith(b"P6\n640 360\n255\n") for p in payloads)
    assert FakeReceiver.instances[0].requests == list(range(1, len(payloads) + 1))


def test_host_stream_rejects_nonoptin_and_invalid_pacing():
    peer = make_peer()
    peer.video_codec = "ppm"
    with pytest.raises(LifecycleError, match="opted-in"):
        session.stream_comm_h264(peer, "T4", queue.Queue(), queue.Queue(),
                                 threading.Event(), time.monotonic() + 1, 6)
    peer.video_codec = "h264"
    for bad in (0, 13, True, 1.2):
        with pytest.raises(ValueError):
            session.stream_comm_h264(peer, "T4", queue.Queue(), queue.Queue(),
                                     threading.Event(), time.monotonic() + 1, bad)


def test_remote_decode_failure_is_redacted_and_terminal(monkeypatch):
    FakeReceiver.instances.clear()
    monkeypatch.setattr(session, "KernelVideoCommReceiver", FakeReceiver)
    class BadDecoder:
        def decode(self, packet):
            raise RuntimeError("private auth token must not appear")
    monkeypatch.setattr(session, "make_host_decoder",
                        lambda: (SimpleNamespace(Packet=FakePacket), BadDecoder()))
    stopped = threading.Event()
    with pytest.raises(LifecycleError, match="transport interrupted") as exc:
        session.stream_comm_h264(make_peer(), "T4", queue.Queue(), queue.Queue(),
                                 stopped, time.monotonic() + .2, 12)
    assert stopped.is_set()
    assert "private auth token" not in str(exc.value)


def test_cli_video_selection_requires_owned_comm_without_other_experiments(monkeypatch):
    from colabcola.cli import main
    from colabcola import bridge_colab
    calls = []
    monkeypatch.setattr(bridge_colab, "run_colab", lambda args: calls.append(
        (args.transport, args.video_codec, args.inflight)) or 0)
    base = ["--profile", "example", "bridge", "colab", "--preflight-only"]
    assert main(base + ["--transport", "comm", "--video-codec", "h264"]) == 0
    assert calls[-1] == ("comm", "h264", 1)
    assert main(base + ["--transport", "comm"]) == 0
    assert calls[-1] == ("comm", "ppm", 2)
    for extra in (
        ["--video-codec", "h264"],
        ["--transport", "comm", "--video-codec", "h264", "--inflight", "2"],
        ["--transport", "comm", "--video-codec", "h264", "--probe-link"],
        ["--transport", "comm", "--video-codec", "h264", "--benchmark-credits"],
        ["--transport", "comm", "--video-codec", "h264", "--benchmark-sweep"],
    ):
        with pytest.raises(SystemExit) as exc:
            main(base + extra)
        assert exc.value.code == 2


def test_run_colab_checks_local_decoder_before_preflight_or_allocation(monkeypatch, tmp_path):
    from colabcola import bridge_colab
    from contextlib import nullcontext
    import colabcola.profiles
    class FakeProvider:
        def __init__(self):
            self.closed = False
        def close(self):
            self.closed = True
    provider = FakeProvider()
    monkeypatch.setattr(bridge_colab, "GoogleProvider", lambda: provider)
    monkeypatch.setattr(colabcola.profiles, "selected", lambda name: nullcontext())
    monkeypatch.setattr(bridge_colab, "verify_preflight",
                        lambda p: pytest.fail("preflight should not be reached"))
    def unavailable():
        raise LifecycleError("optional local PyAV H.264 decoder unavailable")
    monkeypatch.setattr(session, "make_host_decoder", unavailable)
    options = SimpleNamespace(
        profile="example", video_codec="h264", transport="comm", inflight=1,
        preflight_only=False, confirm_allocation=False, run_dir=tmp_path / "no-run",
        fps=6, poll_hz=6, max_seconds=120)
    with pytest.raises(LifecycleError, match="local PyAV"):
        bridge_colab.run_colab(options)
    assert not options.run_dir.exists()
    assert provider.closed


@pytest.mark.parametrize("failure_stage", [None, "bootstrap", "viewer"])
def test_managed_video_releases_only_owned_t4_on_success_or_failure(
        monkeypatch, tmp_path, failure_stage):
    """No real Google connection; exercise journal and release on every path."""
    from colabcola import bridge_colab
    from contextlib import nullcontext
    import colabcola.profiles
    from colabcola import bridge_video_comm_session as video

    class Provider:
        instances = []
        def __init__(self):
            self.endpoint = None
            self.allocations = []
            self.releases = []
            self.closed = False
            self.instances.append(self)
        def allocate(self, allocation_id, gpu):
            assert gpu == "T4"
            self.allocations.append(allocation_id)
            self.endpoint = "owned-h264-t4"
            return self.endpoint
        def reconcile(self, allocation_id, gpu):
            assert allocation_id in self.allocations and gpu == "T4"
            return self.endpoint
        def lookup(self, endpoint):
            return endpoint if endpoint == self.endpoint else None
        def release(self, endpoint):
            assert endpoint == self.endpoint == "owned-h264-t4"
            self.releases.append(endpoint)
            self.endpoint = None
            return {"state": "absent"}
        def assignments(self):
            return []
        def close(self):
            self.closed = True

    monkeypatch.setattr(bridge_colab, "GoogleProvider", Provider)
    monkeypatch.setattr(bridge_colab, "verify_preflight", lambda p: None)
    monkeypatch.setattr(bridge_colab, "wait_for_session", lambda *a: object())
    monkeypatch.setattr(colabcola.profiles, "selected", lambda name: nullcontext())
    monkeypatch.setattr(video, "make_host_decoder",
                        lambda: (SimpleNamespace(Packet=FakePacket), FakeDecoder()))

    class Peer:
        instances = []
        def __init__(self, provider, kernel, run_id, fps, *, video_codec):
            assert video_codec == "h264"
            self.closed = False
            self.instances.append(self)
        def open(self):
            if failure_stage == "bootstrap":
                raise LifecycleError("fake remote video bootstrap failed")
            return {"gpu": "Tesla T4", "device": "cuda", "width": 640, "height": 360,
                    "video_codec": "h264-annexb"}
        def close(self):
            self.closed = True
    from colabcola import bridge_comm
    monkeypatch.setattr(bridge_comm, "KernelCommBridge", Peer)
    monkeypatch.setattr(video, "stream_comm_h264",
                        lambda peer, gpu, outgoing, frames, stopped, deadline, poll_hz:
                        stopped.set())

    def viewer(*args, **kwargs):
        if failure_stage == "viewer":
            raise LifecycleError("fake viewer interrupted")
        kwargs["worker"](queue.Queue(), queue.Queue(), threading.Event())
    monkeypatch.setattr(bridge_colab, "run_viewer", viewer)

    run_dir = tmp_path / ("video-" + str(failure_stage))
    opts = SimpleNamespace(
        profile="example", video_codec="h264", transport="comm", inflight=1,
        preflight_only=False, confirm_allocation=True, run_dir=run_dir,
        fps=6, poll_hz=6, max_seconds=120, benchmark_sweep=False,
        benchmark_credits=False, probe_link=False, viewer_stats=True)

    if failure_stage is not None:
        with pytest.raises(LifecycleError):
            bridge_colab.run_colab(opts)
    else:
        assert bridge_colab.run_colab(opts) == 0
    provider = Provider.instances[-1]
    assert len(provider.allocations) == 1
    assert provider.releases == ["owned-h264-t4"]
    assert provider.closed
    assert Peer.instances[-1].closed
    import json
    journal = json.loads((run_dir / "state.json").read_text())
    assert journal["phase"] == "released"
    assert journal.get("video_codec") == "h264" or failure_stage == "bootstrap"
