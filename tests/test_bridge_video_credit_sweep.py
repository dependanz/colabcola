"""v33k: allocation-free A/B/A' video-credit sweep contract tests."""
from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import pytest

from colabcola import bridge_video_credit_sweep as sweep
from colabcola.lifecycle_io import LifecycleError


class VirtualClock:
    def __init__(self):
        self.now = 10.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0, seconds)

    def perf_counter_ns(self):
        return int(self.now * 1_000_000_000)


class FakeImage:
    width = 640
    height = 360

    def to_ndarray(self, *, format):
        assert format == "rgb24"
        return SimpleNamespace(tobytes=lambda: bytes((10, 20, 30)) * (640 * 360))


class FakeDecoder:
    def __init__(self):
        self.seqs = []

    def decode(self, packet):
        self.seqs.append(packet.pts)
        return [FakeImage()]


class FakePacket:
    def __init__(self, data):
        self.data = data
        self.pts = None


class FakeWindow:
    instances = []

    def __init__(self, peer, *, credits):
        assert credits == 2
        self.clock = peer.clock
        self.pending = []
        self.requests = []
        self.maximum_outstanding = 0
        self.instances.append(self)
        self.suppressed_dependent_units = 0

    @property
    def outstanding(self):
        return len(self.pending)

    def submit(self, pull_seq):
        assert not self.requests or pull_seq == self.requests[-1] + 1
        self.requests.append(pull_seq)
        self.pending.append((pull_seq, self.clock.now, self.clock.now + .145))
        self.maximum_outstanding = max(self.maximum_outstanding, self.outstanding)

    def poll(self, seconds):
        self.clock.sleep(seconds)
        if self.pending and self.pending[0][2] <= self.clock.now:
            pull, sent_at, _ = self.pending.pop(0)
            packet = SimpleNamespace(seq=pull - 1, data=b"\0\0\0\1\x65")
            return [(packet, self.peer_ack(pull), (512, 542, 728),
                     (self.clock.now - sent_at) * 1000)]
        return []

    def peer_ack(self, pull):
        return 1 if pull > 2 else 0

    def check_timeouts(self, *, now=None):
        return None


def build(monkeypatch):
    FakeWindow.instances.clear()
    clock = VirtualClock()
    monkeypatch.setattr(sweep, "time", clock)
    monkeypatch.setattr(sweep, "KernelH264PullWindow", FakeWindow)
    decoder = FakeDecoder()
    monkeypatch.setattr(sweep, "make_host_decoder",
                        lambda: (SimpleNamespace(Packet=FakePacket), decoder))
    monkeypatch.setattr(sweep, "offer_update", lambda q, item: None)
    class Peer:
        video_codec = "h264"
        def __init__(self):
            self.clock = clock
            self.inputs = []
        def send_input(self, seq, event):
            self.inputs.append((seq, event))
    return clock, decoder, Peer()


def test_controlled_sweep_uses_single_window_and_decoder_without_reset(monkeypatch):
    clock, decoder, peer = build(monkeypatch)
    stopped = threading.Event()
    inputs = queue.Queue()
    inputs.put({"event": "key", "key": "Left"})
    report = sweep.run_h264_credit_sweep(
        peer, "Tesla T4", inputs, queue.Queue(), stopped,
        deadline=clock.now + 40, poll_hz=12, phase_seconds=6, warmup_seconds=1)
    assert stopped.is_set()
    assert report["mode"] == "single-owned-comm-h264-credit-aba"
    assert [x["credits"] for x in report["phases"]] == [1, 2, 1]
    assert [x["name"] for x in report["phases"]] == ["A", "B", "A_repeat"]
    assert [x["max_outstanding"] <= x["credits"] for x in report["phases"]] == [True] * 3
    assert report["phases"][1]["max_outstanding"] == 2
    assert all(p["decoded_frames"] > 0 for p in report["phases"])
    assert all(p["measurement_seconds"] == 5 for p in report["phases"])
    assert all(p["longest_no_new_decoded_frame_ms"] >= 0 for p in report["phases"])
    assert all(p["base64_kib_s"] > p["h264_kib_s"] for p in report["phases"])
    assert len(FakeWindow.instances) == 1
    window = FakeWindow.instances[0]
    assert window.requests == list(range(1, report["request_count"] + 1))
    assert window.outstanding == 0
    assert decoder.seqs == list(range(len(decoder.seqs)))
    assert report["all_phase_boundaries_drained"] is True
    assert peer.inputs == [(1, {"event": "key", "key": "Left"})]
    assert report["maximum_window_outstanding"] == 2


def test_sweep_rejects_invalid_or_nonopted_config_before_source_setup():
    peer = SimpleNamespace(video_codec="ppm")
    for kwargs in ({"poll_hz": 0}, {"poll_hz": 13}, {"poll_hz": True},
                   {"phase_seconds": 2}, {"phase_seconds": 41},
                   {"warmup_seconds": 0}, {"warmup_seconds": 6}):
        conf = {"poll_hz": 12, "phase_seconds": 6, "warmup_seconds": 1}
        conf.update(kwargs)
        with pytest.raises(ValueError):
            sweep.run_h264_credit_sweep(peer, "T4", queue.Queue(), queue.Queue(),
                                        threading.Event(), 200, **conf)
    with pytest.raises(LifecycleError, match="opted-in"):
        sweep.run_h264_credit_sweep(peer, "T4", queue.Queue(), queue.Queue(),
                                    threading.Event(), 200, 12)


def test_deadline_failure_is_terminal_and_never_claims_completed_phases(monkeypatch, capsys):
    clock, decoder, peer = build(monkeypatch)
    stopped = threading.Event()
    with pytest.raises(LifecycleError, match="credit sweep interrupted"):
        sweep.run_h264_credit_sweep(
            peer, "Tesla T4", queue.Queue(), queue.Queue(), stopped,
            deadline=clock.now + 2, poll_hz=12, phase_seconds=6, warmup_seconds=1)
    assert stopped.is_set()
    assert "H.264 sweep result" not in capsys.readouterr().out


def test_report_rate_definitions_are_application_only():
    stats = sweep._empty_phase()
    stats.update(decoded=5, replies=6, h264_bytes=10240,
                 ccv1_bytes=10400, base64_chars=14000,
                 rtts=[100, 120, 140], decode_us=[500, 700, 900],
                 acks_ms=[200, 300], fresh_gaps_ms=[125])
    report = sweep._summarize_phase("A", 1, stats, measured_s=5, end_gap_ms=100)
    assert report["decoded_fps"] == 1
    assert report["pull_rtt_p50_ms"] == 120
    assert report["decode_rgb_p50_us"] == 700
    assert report["input_ack_observed_p50_ms"] == 250
    assert report["longest_no_new_decoded_frame_ms"] == 125
    assert report["h264_kib_s"] == 2
