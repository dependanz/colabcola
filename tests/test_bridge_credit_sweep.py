"""Deterministic tests of the opt-in credit sweep (no Google allocation)."""
from __future__ import annotations

import queue
import threading

import pytest

from colabcola import bridge_credit_sweep as sweep
from colabcola.lifecycle_io import LifecycleError


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, delay):
        self.now += max(0.0, delay)


class FakePeer:
    def __init__(self, clock, *, delay=.18, unsolicited=False):
        self.clock = clock
        self.delay = delay
        self.unsolicited = unsolicited
        self.requests = []
        self.all_ids = []
        self.max_outstanding = 0
        self.latest_renderer_metrics = None

    def request_frame(self, request_id):
        self.requests.append((request_id, self.clock.now + self.delay))
        self.all_ids.append(request_id)
        self.max_outstanding = max(self.max_outstanding, len(self.requests))

    def poll_pipeline_frame(self, timeout):
        self.clock.now += timeout
        if self.unsolicited:
            self.unsolicited = False
            return 999, 10, 0, b"ppm", 90000
        if self.requests and self.requests[0][1] <= self.clock.now:
            rid, _ = self.requests.pop(0)
            self.latest_renderer_metrics = {
                "renderer_fps": 12.0, "seq": rid,
                "render_copy_ms": 1, "compress_ms": 8,
            }
            return rid, rid, 0, b"ppm", 120000
        return None

    def send_input(self, seq, event):
        raise AssertionError("unrequested input event")


def test_credit_sweep_five_phases_single_peer_and_bounded_requests(monkeypatch, capsys):
    clock = Clock()
    monkeypatch.setattr(sweep, "time", clock)
    peer = FakePeer(clock)
    stopped = threading.Event()
    sweep.run_credit_sweep(
        peer, "Tesla T4", queue.Queue(), queue.Queue(maxsize=2),
        stopped, deadline=100, phase_seconds=10)
    output = capsys.readouterr().out
    assert output.count("CREDIT SWEEP phase") == 5
    assert [f"credits {c}" in output for c in sweep.CREDITS] == [True] * 5
    assert "Credit sweep completed in one owned T4 session" in output
    assert stopped.is_set()
    assert peer.max_outstanding == 4
    assert peer.requests == []
    assert peer.all_ids == list(range(1, len(peer.all_ids) + 1))
    assert len(peer.all_ids) > 80


def test_credit_phase_summary_has_postwarmup_metrics():
    report = sweep.phase_summary(
        phase=2, credits=2, seconds=12.0, fresh=85, responses=90,
        rtts=[160, 170, 180], ack_samples=[210, 220, 230],
        remote_fps=11.9)
    assert "phase 2/5" in report
    assert "received 8.50 FPS" in report
    assert "RTT p50 170 ms" in report
    assert "fresh 85/90" in report
    assert "input ACK p50 220 ms" in report


def test_credit_sweep_rejects_unmatched_replies_and_stops(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(sweep, "time", clock)
    stopped = threading.Event()
    peer = FakePeer(clock, unsolicited=True)
    with pytest.raises(LifecycleError, match="bounded credit sweep interrupted"):
        sweep.run_credit_sweep(
            peer, "T4", queue.Queue(), queue.Queue(maxsize=2),
            stopped, deadline=100, phase_seconds=10)
    assert stopped.is_set()


def test_credit_sweep_rejects_bad_duration():
    for n in (0, 9, 41, None, 12.5, True):
        with pytest.raises(ValueError, match="duration"):
            sweep.run_credit_sweep(
                None, "T4", queue.Queue(), queue.Queue(), threading.Event(),
                deadline=100, phase_seconds=n)


def test_credit_sweep_cli_validates_no_conflicting_modes(monkeypatch):
    from colabcola.cli import main
    from colabcola import bridge_colab

    used = []
    monkeypatch.setattr(bridge_colab, "run_colab", lambda args: used.append(args) or 0)
    args = ["--profile", "example", "bridge", "colab", "--preflight-only",
            "--transport", "comm", "--fps", "12", "--poll-hz", "12",
            "--benchmark-credits", "--max-seconds", "180"]
    assert main(args) == 0
    assert len(used) == 1 and used[0].benchmark_credits

    for conflict in (
        ["--benchmark-sweep"], ["--probe-link"], ["--inflight", "3"],
        ["--credit-phase-seconds", "9"], ["--credit-phase-seconds", "41"],
        ["--transport", "files"], ["--fps", "6"], ["--poll-hz", "6"],
        ["--max-seconds", "120"],
    ):
        with pytest.raises(SystemExit) as exc:
            main(args + conflict)
        assert exc.value.code == 2
