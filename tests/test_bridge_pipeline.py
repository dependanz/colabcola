"""Offline, deterministic tests for bounded Jupyter Comm frame pipelining."""
from __future__ import annotations

import queue
import threading

import pytest

from colabcola import bridge_pipeline
from colabcola.bridge_pipeline import stream_comm_pipeline
from colabcola.lifecycle_io import LifecycleError


class FakeClock:
    def __init__(self):
        self.now = 0.0
    def monotonic(self):
        return self.now
    def sleep(self, duration):
        self.now += max(0.0, duration)


def test_multiple_outstanding_frame_pulls_are_bounded_and_paced(monkeypatch, capsys):
    clock = FakeClock()
    monkeypatch.setattr(bridge_pipeline, "time", clock)
    stopped = threading.Event()
    images = queue.Queue(maxsize=2)

    class Peer:
        def __init__(self):
            self.requests = []
            self.max_pending = 0
            self.received = 0
            self.latest_renderer_metrics = None
        def request_frame(self, pull_seq):
            self.requests.append((pull_seq, clock.now + 0.44))
            self.max_pending = max(self.max_pending, len(self.requests))
        def poll_pipeline_frame(self, seconds):
            clock.now += seconds
            if self.requests and self.requests[0][1] <= clock.now:
                req_id, _ = self.requests.pop(0)
                self.received += 1
                self.latest_renderer_metrics = {
                    "seq": req_id, "renderer_fps": 11.9,
                    "render_copy_ms": 1, "compress_ms": 7,
                    "publish_ms": .2,
                }
                return req_id, req_id, 0, b"ppm", 140000
            return None
        def send_input(self, seq, event):
            raise AssertionError("unexpected input")

    peer = Peer()
    stream_comm_pipeline(peer, "Tesla T4", queue.Queue(), images,
                         stopped, deadline=5, poll_hz=12, in_flight=3)
    output = capsys.readouterr().out
    assert stopped.is_set()
    assert 1 < peer.max_pending <= 3
    assert peer.received >= 9
    assert "in-flight" in output
    assert "pull RTT p50" in output
    assert peer.received > 5  # More than a 1Hz sequential test


def test_unsolicited_pipeline_reply_fails_and_stops(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(bridge_pipeline, "time", clock)
    stopped = threading.Event()

    class Peer:
        def request_frame(self, pull_seq):
            pass
        def poll_pipeline_frame(self, seconds):
            clock.now += .1
            return (987, 12, 0, b"ppm", 1000)
        def send_input(self, seq, event):
            pass

    with pytest.raises(LifecycleError, match="bounded Comm frame pipeline interrupted"):
        stream_comm_pipeline(Peer(), "Tesla T4", queue.Queue(), queue.Queue(maxsize=2),
                             stopped, deadline=10, poll_hz=12, in_flight=2)
    assert stopped.is_set()


def test_pipeline_rejects_unbounded_credit_or_invalid_poll():
    stopped = threading.Event()
    for credits, hz in ((0, 12), (1, 12), (5, 12), (2, 0), (2, 13), (True, 12)):
        with pytest.raises(ValueError):
            stream_comm_pipeline(None, "Tesla T4", queue.Queue(), queue.Queue(),
                                 stopped, deadline=50, poll_hz=hz, in_flight=credits)


def test_cli_pipeline_requires_comm_and_no_other_benchmarks(monkeypatch):
    from colabcola.cli import main
    from colabcola import bridge_colab
    accepted = []
    monkeypatch.setattr(bridge_colab, "run_colab",
                        lambda args: accepted.append(args.inflight) or 0)
    base = ["--profile", "example", "bridge", "colab", "--preflight-only"]
    assert main(base + ["--transport", "comm", "--inflight", "3"]) == 0
    assert accepted == [3]
    for tail in (
        ["--inflight", "3"],
        ["--transport", "comm", "--inflight", "5"],
        ["--transport", "comm", "--inflight", "3", "--probe-link"],
        ["--transport", "comm", "--inflight", "3", "--benchmark-sweep"],
    ):
        with pytest.raises(SystemExit) as error:
            main(base + tail)
        assert error.value.code == 2
