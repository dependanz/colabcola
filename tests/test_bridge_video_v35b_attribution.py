"""v35B offline attribution: bounded diagnostics; no Colab, CUDA, or network."""
from __future__ import annotations

from collections import deque
import json
import math
import queue
import struct
import threading
import time
from types import SimpleNamespace

import pytest

from colabcola.bridge_colab import offer_update
from colabcola.bridge_colab_runtime import make_producer_window
from colabcola.bridge_demo import MeteredFrameQueue
from colabcola.bridge_video_colab_source import read_producer_window
from colabcola.bridge_video_comm_pipeline import validated_producer_window
from colabcola.bridge_video_comm_protocol import encode_video_reply
from colabcola.bridge_video_push_runtime import CreditPushSender
from colabcola.bridge_video_push_session import (
    producer_stage_summary, run_push_comm_h264, validated_sender_diagnostics,
)
from colabcola.bridge_video_stream import VideoAccessUnit

RUN = "d" * 32
IDR = b"\x00\x00\x00\x01\x65\x88\x84"
P = b"\x00\x00\x00\x01\x41\x88\x84"


def window():
    return make_producer_window(
        run_id=RUN, last_seq=200, window_seq=2,
        published_count=60, elapsed_s=1.0, target_fps=60, rate_seq=0,
        render_samples=[1., 2., 3., 4., 5.],
        compress_samples=[3., 4., 5., 6., 7.],
        publish_samples=[0.1, 0.2, 0.3, 0.4, 0.5],
    )


def test_remote_window_stage_tails_roundtrip_and_weighted_rate(tmp_path):
    item = window()
    assert item["render_copy_ms"] == 3.0
    assert item["render_copy_p95_ms"] == 4.8
    assert item["compress_p95_ms"] == 6.8
    assert item["publish_p95_ms"] == 0.48
    path = tmp_path / "render-metrics.json"
    path.write_text(json.dumps(item), encoding="utf-8")
    assert read_producer_window(path, RUN) == validated_producer_window(item)
    assert validated_producer_window(item)["render_copy_p95_ms"] == 4.8
    summary = producer_stage_summary({2: item})
    assert summary["producer_render_copy_window_median_p50_ms"] == 3.
    assert summary["producer_render_copy_within_window_p95_median_ms"] == 4.8
    assert summary["producer_compress_within_window_p95_median_ms"] == 6.8
    old = {k: v for k, v in item.items() if "_p95_ms" not in k}
    assert validated_producer_window(old) is not None
    assert producer_stage_summary({2: old})[
        "producer_render_copy_within_window_p95_median_ms"] is None


@pytest.mark.parametrize("bad", [True, -1., math.inf, math.nan, "bad", 1.])
def test_invalid_producer_stage_tails_fail_closed_without_breaking_video(bad):
    item = {**window(), "render_copy_p95_ms": bad}
    assert validated_producer_window(item) is None


@pytest.mark.parametrize("bad", [
    {"sender_credit_wait_us_total": True, "sender_credit_wait_iterations": 0},
    {"sender_credit_wait_us_total": math.nan, "sender_credit_wait_iterations": 0},
    {"sender_credit_wait_us_total": 0., "sender_credit_wait_iterations": False},
    {"sender_credit_wait_us_total": 0., "sender_credit_wait_iterations": 0,
     "sender_comm_previous_us": math.inf},
])
def test_optional_sender_diagnostics_reject_bad_values(bad):
    assert validated_sender_diagnostics(bad) is None
    assert validated_sender_diagnostics({"kind": "video_access_unit"}) is None


def test_sender_metrics_report_previous_send_only_at_60hz(tmp_path):
    class Source:
        sequence_buffered = True
        def __init__(self, fps):
            self.fps = fps
            self.last_renderer_seq = None
        def pull(self, seq):
            self.last_renderer_seq = seq - 1
            return {"v": 1, "kind": "video_access_unit", "pull_seq": seq}
    class Comm:
        def __init__(self):
            self.sent = []
        def send(self, packet):
            self.sent.append(packet)

    for fps, credits in ((60, 10), (30, 6)):
        source, comm = Source(fps), Comm()
        sender = CreditPushSender(
            comm=comm, source=source, directory=tmp_path,
            run_id=RUN, credits=credits)
        sender.grant(credits)
        sender._credit_wait_ns = 2_500_000
        sender._credit_wait_iterations = 2
        for seq in range(2):
            (tmp_path / "frame.bin").write_bytes(
                struct.pack("!4sIIII", b"CCB1", seq, 0, 640, 360))
            assert sender.step()
        assert len(comm.sent) == 2
        if fps == 60:
            assert comm.sent[0]["sender_credit_wait_us_total"] == 2500.
            assert "sender_comm_previous_us" not in comm.sent[0]
            assert comm.sent[1]["sender_comm_previous_us"] >= 0
            assert validated_sender_diagnostics(comm.sent[1])[
                "credit_wait_iterations"] == 2
        else:
            assert all(not any(k.startswith("sender_") for k in p)
                       for p in comm.sent)
        sender.close()


def test_metered_gui_queue_counts_evicted_frames_not_statuses():
    q = MeteredFrameQueue(maxsize=2)
    offer_update(q, ("frame-timed", ("frame0", None)))
    offer_update(q, ("status", "running"))
    offer_update(q, ("frame", "frame1"))
    assert q.coalesced_video_frames == 1
    assert q.get_nowait()[0] == "status"
    offer_update(q, ("frame", "frame2"))
    offer_update(q, ("frame", "frame3"))
    assert q.coalesced_video_frames == 2


def test_host_report_attributes_weighted_producer_codec_and_sender(monkeypatch):
    from colabcola import bridge_video_push_session as mod
    prepared = window()
    responses = deque()
    done = threading.Event()
    connected = threading.Event()
    connected.set()

    class Channel:
        def get_msg(self, timeout=0):
            if responses:
                return responses.popleft()
            raise queue.Empty

    class Peer:
        fps = 60
        video_push = True
        video_codec = "h264"
        push_credits = 10
        run_id = RUN
        comm_id = "owned"
        wire = SimpleNamespace(
            connection_ready=connected, iopub_channel=Channel())
        def _send(self, kind, **kwargs):
            if kind != "video_credit" or kwargs != {"count": 10}:
                return
            for idx in range(2):
                unit = VideoAccessUnit(
                    seq=idx, pts_ns=idx * 16_666_667,
                    capture_ns=idx + 1,
                    data=IDR if idx == 0 else P, keyframe=idx == 0)
                payload = encode_video_reply(
                    run_id=RUN, pull_seq=idx + 1, input_ack=0, unit=unit)
                payload.update(
                    renderer_seq=100 + idx,
                    source_read_us=3000., encode_us=4000.,
                    producer_window=prepared,
                    sender_credit_wait_us_total=1500. + 100 * idx,
                    sender_credit_wait_iterations=3 + idx)
                if idx:
                    payload["sender_comm_previous_us"] = 1750.
                responses.append({
                    "header": {"msg_type": "comm_msg"},
                    "content": {"comm_id": "owned", "data": payload},
                })

    class Frame:
        width, height = 640, 360
        def to_ndarray(self, format):
            assert format == "rgb24"
            return SimpleNamespace(tobytes=lambda: b"x" * (640 * 360 * 3))
    class Packet:
        def __init__(self, data):
            self.pts = None
    class Decoder:
        def decode(self, packet):
            return [Frame()]

    monkeypatch.setattr(mod, "make_host_decoder",
                        lambda: (SimpleNamespace(Packet=Packet), Decoder()))
    seen = []
    def publish(items, item):
        seen.append(item)
        if sum(kind == "frame-timed" for kind, _ in seen) == 2:
            done.set()
    monkeypatch.setattr(mod, "offer_update", publish)
    result = run_push_comm_h264(
        Peer(), "Tesla T4", queue.Queue(), queue.Queue(), done,
        time.monotonic() + 1.0, credits=10)
    assert result["decoded_frames"] == 2
    assert result["producer_renderer_fps_weighted_covered"] == 60.
    assert result["producer_frames_covered"] == 60
    assert result["producer_elapsed_ms_covered"] == 1000.
    assert result["producer_render_copy_within_window_p95_median_ms"] == 4.8
    assert result["remote_source_read_p95_ms"] == 3.
    assert result["remote_encode_p99_ms"] == 4.
    assert result["remote_comm_previous_send_p50_ms"] == 1.75
    assert result["remote_sender_credit_wait_ms_total_observed"] == 1.6
    assert result["remote_sender_credit_wait_iterations_observed"] == 4
    assert result["remote_sender_diagnostic_observations"] == 2
