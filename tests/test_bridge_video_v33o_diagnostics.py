"""v33o: same-clock timing and renderer sequence diagnostics (offline only)."""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import queue
import struct
import threading
import zlib

import pytest

from colabcola.bridge_video_colab_source import RemoteH264PullSource
from colabcola.bridge_video_comm_pipeline import (
    KernelH264PullWindow, validated_remote_video_diagnostics,
)
from colabcola.bridge_video_credit_sweep import _empty_phase, _summarize_phase

RUN = "a" * 32
IDR = b"\x00\x00\x00\x01\x65\x88\x84"


def write_snapshot(path: Path, renderer_seq: int, *, ack: int = 0):
    ppm = b"P6\n640 360\n255\n" + bytes([20, 40, 60]) * (640 * 360)
    path.write_bytes(struct.pack("!4sIIII", b"CCB1", renderer_seq, ack, 640, 360)
                     + zlib.compress(ppm, level=1))


def test_remote_source_reports_renderer_identity_and_independent_durations(tmp_path):
    snapshot = tmp_path / "frame.bin"
    source = RemoteH264PullSource(
        tmp_path, RUN, fps=30,
        encoder_factory=lambda *_: (lambda rgb, seq: (IDR, True)))
    write_snapshot(snapshot, 19)
    a = source.pull(1)
    b = source.pull(2)
    write_snapshot(snapshot, 20)
    c = source.pull(3)
    assert [v["renderer_seq"] for v in (a, b, c)] == [19, 19, 20]
    for payload in (a, b, c):
        details = validated_remote_video_diagnostics(payload)
        assert details is not None
        assert details["renderer_seq"] == payload["renderer_seq"]
        assert details["source_read_us"] >= 0
        assert details["encode_us"] >= 0


@pytest.mark.parametrize("changes", [
    {"renderer_seq": True}, {"renderer_seq": -1},
    {"encode_us": -1}, {"encode_us": float("nan")},
    {"source_read_us": float("inf")}, {"source_read_us": "3.0"},
    {"encode_us": 11_000_000},
])
def test_rejects_invalid_optional_diagnostics_without_crashing_codec(changes):
    item = {"renderer_seq": 1, "source_read_us": 20.0, "encode_us": 30.0}
    item.update(changes)
    assert validated_remote_video_diagnostics(item) is None
    assert validated_remote_video_diagnostics({"renderer_seq": 1}) is None


class Channel:
    def __init__(self):
        self.messages = []

    def get_msg(self, timeout=None):
        if not self.messages:
            raise queue.Empty
        return self.messages.pop(0)


class Peer:
    def __init__(self):
        ready = threading.Event()
        ready.set()
        self.run_id = RUN
        self.comm_id = "owned"
        self.wire = SimpleNamespace(connection_ready=ready, iopub_channel=Channel())
        self.requests = []

    def _send(self, kind, **kwargs):
        self.requests.append((kind, kwargs))


def test_reordered_replies_keep_diagnostics_aligned(tmp_path):
    from colabcola.bridge_video_comm_protocol import encode_video_reply
    from colabcola.bridge_video_stream import VideoAccessUnit

    peer = Peer()
    window = KernelH264PullWindow(peer, credits=2)
    window.submit(1)
    window.submit(2)
    # Deliver pull 2 before pull 1.
    for pull in (2, 1):
        unit = VideoAccessUnit(seq=pull - 1, pts_ns=(pull - 1) * 33333333,
                               capture_ns=10 + pull, data=IDR, keyframe=True)
        payload = encode_video_reply(run_id=RUN, pull_seq=pull, input_ack=0, unit=unit)
        payload.update(renderer_seq=(60 if pull == 2 else 59),
                       source_read_us=12.5 * pull, encode_us=100.0 * pull)
        peer.wire.iopub_channel.messages.append({
            "header": {"msg_type": "comm_msg"},
            "content": {"comm_id": "owned", "data": payload},
        })
    assert window.poll(.01) == []
    assert window.last_committed_diagnostics == []
    ordered = window.poll(.01)
    assert [unit.seq for unit, *_ in ordered] == [0, 1]
    assert [v["renderer_seq"] for v in window.last_committed_diagnostics] == [59, 60]
    assert [v["encode_us"] for v in window.last_committed_diagnostics] == [100, 200]
    assert window.outstanding == 0
    assert window.poll(.01) == []
    assert window.last_committed_diagnostics == []


def test_summary_reports_distinct_snapshots_not_only_encoded_units():
    stats = _empty_phase()
    stats.update(decoded=5, replies=5, renderer_seqs={7, 8},
                 duplicate_renderer_replies=3, source_read_us=[20, 40],
                 encode_us=[1000, 1200], submit_intervals_ms=[30, 60])
    result = _summarize_phase("C4", 4, stats, measured_s=2, end_gap_ms=0)
    assert result["decoded_fps"] == 2.5
    assert result["distinct_renderer_sequences_in_replies"] == 2
    assert result["distinct_renderer_reply_fps"] == 1
    assert result["duplicate_renderer_replies"] == 3
    assert result["remote_snapshot_read_p50_us"] == 30
    assert result["remote_encode_p50_us"] == 1100
    assert result["host_pull_submit_interval_p50_ms"] == 45
    assert _summarize_phase("C4", 4, _empty_phase(), measured_s=2,
                            end_gap_ms=0)["distinct_renderer_reply_fps"] is None
