"""No-Google regression tests for the in-session renderer/pull-rate sweep."""
from __future__ import annotations

import queue
import threading

import pytest

from colabcola import bridge_sweep
from colabcola.bridge_sweep import PhaseMeasurement, SWEEP_PHASES, sweep_phases
from colabcola.lifecycle_io import LifecycleError


def test_sweep_is_fixed_two_by_two_with_baseline_repeat():
    assert SWEEP_PHASES == ((6, 6), (12, 6), (6, 12), (12, 12), (6, 6))
    for bad in (None, 9, 41, True, 10.0):
        with pytest.raises(ValueError):
            sweep_phases(bad)
    phases = sweep_phases(12)
    assert [(p.renderer_fps, p.pull_hz) for p in phases] == list(SWEEP_PHASES)
    assert [p.phase for p in phases] == [1, 2, 3, 4, 5]


def test_sweep_measurement_excludes_warmup_and_rejects_unconfirmed_remote_rate():
    phase = PhaseMeasurement(12, 6, 2, 10)
    phase.begin(100.0)
    good = {"target_fps": 12, "rate_seq": 1, "renderer_fps": 11.9, "seq": 15}
    phase.collect(now=101.0, rtt_ms=400, size=1200, fresh=True,
                  metrics=good, required_rate_seq=1)
    assert phase.pulls == 0  # warmup
    phase.collect(now=103.0, rtt_ms=200, size=1200, fresh=True,
                  metrics={**good, "rate_seq": 0}, required_rate_seq=1)
    phase.collect(now=103.5, rtt_ms=180, size=1200, fresh=False,
                  metrics=good, required_rate_seq=1)
    assert phase.pulls == 2 and phase.fresh == 1
    assert phase.renderer_confirmed == 1
    text = phase.summary(110.0)
    assert "phase 2/5" in text and "RTT p50 190 ms" in text
    assert "rate-confirmed 1/2" in text
    assert "actual-render 11.9 FPS" in text


def test_sweep_completes_in_one_transport_without_extra_allocations(monkeypatch, capsys):
    """A deterministic clock replaces time; no Jupyter, network, or CUDA needed."""
    class FakeClock:
        now = 0.0
        def monotonic(self):
            return self.now
        def sleep(self, seconds):
            self.now += seconds

    clock = FakeClock()
    monkeypatch.setattr(bridge_sweep, "time", clock)

    class FakePeer:
        def __init__(self):
            self.current_fps = 6
            self.control_seq = 0
            self.controls = []
            self.frames = 0
            self.latest_renderer_metrics = None
        def set_render_fps(self, seq, fps):
            self.controls.append((seq, fps))
            self.current_fps = fps
            self.control_seq = seq
        def request_frame(self):
            pass
        def send_input(self, seq, event):
            raise AssertionError("unexpected input")
        def next_frame(self, seconds):
            self.frames += 1
            clock.now += 0.14
            self.latest_renderer_metrics = {
                "seq": self.frames, "target_fps": self.current_fps,
                "rate_seq": self.control_seq, "renderer_fps": float(self.current_fps),
            }
            return self.frames, 0, b"frame", 1000

    peer = FakePeer()
    stopped = threading.Event()
    bridge_sweep.run_rate_sweep(
        peer, "Tesla T4", queue.Queue(), queue.Queue(maxsize=2),
        stopped, deadline=120, phase_seconds=10)
    log = capsys.readouterr().out
    assert stopped.is_set()
    assert peer.controls == [(1, 12), (2, 6), (3, 12), (4, 6)]
    assert log.count("[colabcola] SWEEP phase") == 5
    assert "Sweep completed; ending the owned interactive session" in log
    assert "rate-confirmed" in log


def test_sweep_invalid_transport_or_suspicious_budget_rejected(monkeypatch):
    from colabcola.cli import main
    from colabcola import bridge_colab
    started = []
    monkeypatch.setattr(bridge_colab, "run_colab", lambda args: started.append(args) or 0)
    command = ["--profile", "example", "bridge", "colab", "--preflight-only",
               "--benchmark-sweep", "--transport", "comm", "--fps", "6",
               "--poll-hz", "6", "--max-seconds", "180"]
    assert main(command) == 0
    assert started and started[0].benchmark_phase_seconds == 12
    for tail in (["--transport", "files"], ["--benchmark-phase-seconds", "9"],
                 ["--max-seconds", "120"]):
        with pytest.raises(SystemExit) as error:
            main(command + tail)
        assert error.value.code == 2
