"""v33r: Windows-coarse-clock reproduction for experimental 30/60 Hz video pacing.

No Colab, GPU, Jupyter websocket, or host clock adjustment is involved.
"""
from __future__ import annotations

from collections import deque
import math
import queue
import threading
from types import SimpleNamespace

import pytest

from colabcola.bridge_video_comm_pipeline import KernelH264PullWindow
from colabcola import bridge_video_credit_sweep as sweep


class Channel:
    def __init__(self):
        self.items = deque()
        self.timeouts = []

    def get_msg(self, timeout=None):
        self.timeouts.append(timeout)
        if not self.items:
            raise queue.Empty()
        return self.items.popleft()


def test_h264_window_supports_immediate_iopub_receive_without_replay():
    ready = threading.Event()
    ready.set()
    channel = Channel()
    sent = []
    peer = SimpleNamespace(run_id="a" * 32, comm_id="owned",
                           wire=SimpleNamespace(connection_ready=ready, iopub_channel=channel),
                           _send=lambda *a, **k: sent.append((a, k)))
    window = KernelH264PullWindow(peer, credits=4, experimental=True)
    window.submit(1)
    assert window.poll(0.0) == []
    assert channel.timeouts == [0.0]
    assert window.outstanding == 1  # empty poll is NOT a lost request
    assert sent == [(('video_pull',), {'pull_seq': 1})]

    from colabcola.bridge_video_comm_protocol import encode_video_reply
    from colabcola.bridge_video_stream import VideoAccessUnit
    nal = b"\x00\x00\x00\x01\x65\x88\x84"
    unit = VideoAccessUnit(seq=0, pts_ns=0, capture_ns=123,
                           data=nal, keyframe=True)
    payload = encode_video_reply(run_id="a" * 32, pull_seq=1,
                                 input_ack=0, unit=unit)
    channel.items.append({"header": {"msg_type": "comm_msg"},
                          "content": {"comm_id": "owned", "data": payload}})
    committed = window.poll(0.0)
    assert len(committed) == 1 and committed[0][0].seq == 0
    assert window.outstanding == 0
    assert channel.timeouts == [0.0, 0.0]
    for invalid in (-0.0001, .0005, .51, True, "0"):
        with pytest.raises(ValueError):
            window.poll(invalid)


class CoarseWindowsClock:
    """Simulate ~15.625 ms monotonic ticks but nanosecond perf-counter pacing."""
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return math.floor(self.now / .015625) * .015625

    def perf_counter_ns(self):
        return round(self.now * 1_000_000_000)

    def sleep(self, seconds):
        self.now += max(0.0, seconds)


class FakeImage:
    width, height = 640, 360
    def to_ndarray(self, *, format):
        assert format == "rgb24"
        return SimpleNamespace(tobytes=lambda: bytes((1, 2, 3)) * (640 * 360))


class FakeDecoder:
    def __init__(self):
        self.seqs = []
    def decode(self, packet):
        self.seqs.append(packet.pts)
        return [FakeImage()]


class FakePacket:
    def __init__(self, raw):
        self.pts = None


class FakeWindow:
    instances = []
    def __init__(self, peer, *, credits, experimental=False):
        assert credits == 32 and experimental is True
        self.clock = peer.clock
        self.pending = deque()
        self.suppressed_dependent_units = 0
        self.maximum_outstanding = 0
        self.sent_ids = []
        self.instances.append(self)

    @property
    def outstanding(self):
        return len(self.pending)

    def submit(self, seq):
        assert self.outstanding < 32
        self.sent_ids.append(seq)
        self.pending.append((seq, self.clock.now + .105))
        self.maximum_outstanding = max(self.maximum_outstanding, self.outstanding)

    def poll(self, seconds):
        assert seconds == 0.0 or seconds == sweep.POLL_WAIT_SECONDS
        if seconds:
            # Drain happens outside scoring. Simulate a coarse blocking read.
            self.clock.sleep(seconds)
        if self.pending and self.pending[0][1] <= self.clock.now:
            seq, _ = self.pending.popleft()
            return [(SimpleNamespace(seq=seq-1, data=b"h264"),
                     0, (100, 130, 176), .105 * 1000)]
        return []

    def check_timeouts(self, *, now=None):
        pass


@pytest.mark.parametrize("hz,credit,interval_upper,frame_floor", [
    (30, 4, 35, 27), (60, 8, 18, 52),
])
def test_highrate_sweep_uses_perf_counter_and_nowait_under_coarse_monotonic(
        monkeypatch, hz, credit, interval_upper, frame_floor):
    clock = CoarseWindowsClock()
    FakeWindow.instances.clear()
    monkeypatch.setattr(sweep, "time", clock)
    monkeypatch.setattr(sweep, "KernelH264PullWindow", FakeWindow)
    decoder = FakeDecoder()
    monkeypatch.setattr(sweep, "make_host_decoder", lambda: (
        SimpleNamespace(Packet=FakePacket), decoder))
    monkeypatch.setattr(sweep, "offer_update", lambda *a: None)
    peer = SimpleNamespace(video_codec="h264", clock=clock,
                           send_input=lambda *a: None)
    stopped = threading.Event()
    result = sweep.run_h264_capacity_sweep(
        peer, "T4", queue.Queue(), queue.Queue(), stopped,
        deadline=clock.monotonic() + 90, poll_hz=hz,
        phase_seconds=6, warmup_seconds=1)
    assert stopped.is_set()
    assert result["nonblocking_iopub_poll"] is True
    assert result["host_phase_clock"] == "perf_counter_ns"
    stage = next(p for p in result["phases"] if p["credits"] == credit)
    assert stage["host_pull_submit_interval_p50_ms"] < interval_upper
    assert stage["decoded_fps"] >= frame_floor
    assert stage["host_receive_poll_elapsed_p50_ms"] == 0
    assert all(p["max_outstanding"] <= p["credits"] for p in result["phases"])
    assert result["all_phase_boundaries_drained"]
    assert len(FakeWindow.instances) == 1
    win = FakeWindow.instances[0]
    assert win.outstanding == 0
    assert win.sent_ids == list(range(1, result["request_count"]+1))
    assert decoder.seqs == list(range(len(decoder.seqs)))
