"""v33s: independent CUDA producer windows carried across H.264 pulls.

Allocation-free: no Colab, CUDA, internet, real PyAV, or network resources.
"""
from __future__ import annotations

import json
from pathlib import Path
import queue
import struct
import threading
from types import SimpleNamespace
import zlib

import pytest

from colabcola.bridge_colab_runtime import make_producer_window
from colabcola.bridge_video_colab_source import (
    RemoteH264PullSource, read_producer_window,
)
from colabcola.bridge_video_comm_pipeline import (
    KernelH264PullWindow, validated_producer_window,
    validated_remote_video_diagnostics,
)
from colabcola.bridge_video_credit_sweep import (
    _empty_phase, _record_producer_window, _summarize_phase,
)

RUN = "a" * 32
IDR = b"\x00\x00\x00\x01\x65\x88\x84"


def producer_window(wid=4, *, last_seq=100, published_count=30,
                    elapsed_s=1.0, target_fps=30):
    return make_producer_window(
        run_id=RUN, last_seq=last_seq, window_seq=wid,
        published_count=published_count, elapsed_s=elapsed_s,
        target_fps=target_fps, rate_seq=0,
        render_samples=[2.0, 3.0, 4.0],
        compress_samples=[4.0, 5.0, 6.0],
        publish_samples=[0.1, 0.2, 0.3],
    )


def write_snapshot(path: Path, renderer_seq=123):
    pixels = bytes((20, 40, 60)) * (640 * 360)
    ppm = b"P6\n640 360\n255\n" + pixels
    path.write_bytes(struct.pack("!4sIIII", b"CCB1", renderer_seq, 0, 640, 360)
                     + zlib.compress(ppm, level=1))


def test_remote_window_is_independent_of_pull_sequence():
    window = producer_window(8, last_seq=500, published_count=27, elapsed_s=1.0)
    assert window["window_seq"] == 8
    assert window["seq"] == 500
    assert window["renderer_fps"] == 27
    assert window["render_copy_ms"] == 3.0
    assert window["compress_ms"] == 5.0
    assert window["publish_ms"] == .2
    assert validated_producer_window(window) is not None


def test_remote_sidecar_read_is_bounded_and_owned(tmp_path):
    path = tmp_path / "render-metrics.json"
    assert read_producer_window(path, RUN) is None
    path.write_text(json.dumps(producer_window()), encoding="utf-8")
    assert read_producer_window(path, RUN)["seq"] == 100
    assert read_producer_window(path, "b" * 32) is None
    bad_schema = {**producer_window(), "schema": True}
    path.write_text(json.dumps(bad_schema), encoding="utf-8")
    assert read_producer_window(path, RUN) is None
    path.write_bytes(b"{" + b" " * 4096)
    assert read_producer_window(path, RUN) is None
    path.write_text("invalid json", encoding="utf-8")
    assert read_producer_window(path, RUN) is None


@pytest.mark.parametrize("field,invalid", [
    ("window_seq", True), ("window_seq", -1), ("seq", -1),
    ("published_count", False), ("published_count", 0),
    ("window_elapsed_ms", 0), ("window_elapsed_ms", float("nan")),
    ("renderer_fps", float("inf")), ("renderer_fps", 31.0),
    ("target_fps", 500), ("compress_ms", "5.0"),
    ("render_copy_ms", -1), ("publish_ms", 10001),
])
def test_invalid_producer_metadata_does_not_invalidate_video_diagnostics(field, invalid):
    window = producer_window()
    window[field] = invalid
    assert validated_producer_window(window) is None
    diagnostics = validated_remote_video_diagnostics({
        "renderer_seq": 125, "source_read_us": 3000.0,
        "encode_us": 2000.0, "producer_window": window,
    })
    assert diagnostics is not None
    assert diagnostics["renderer_seq"] == 125
    assert diagnostics["producer_window"] is None


def test_remote_source_attaches_optional_producer_windows_without_changing_video(tmp_path):
    write_snapshot(tmp_path / "frame.bin", renderer_seq=123)
    source = RemoteH264PullSource(
        tmp_path, RUN, fps=30,
        encoder_factory=lambda *_: (lambda pixels, seq: (IDR, True)),
    )
    # The sidecar may not exist before the producer publishes the first second.
    first = source.pull(1)
    assert "producer_window" not in first
    (tmp_path / "render-metrics.json").write_text(
        json.dumps(producer_window()), encoding="utf-8")
    second = source.pull(2)
    third = source.pull(3)
    assert second["renderer_seq"] == third["renderer_seq"] == 123
    assert second["producer_window"]["window_seq"] == 4
    assert third["producer_window"]["window_seq"] == 4
    assert [first["pull_seq"], second["pull_seq"], third["pull_seq"]] == [1, 2, 3]
    assert "producer_window" not in first
    assert second["data"] != ""
    assert not source.failed


class Channel:
    def __init__(self):
        self.messages = []

    def get_msg(self, timeout=None):
        if not self.messages:
            raise queue.Empty
        return self.messages.pop(0)


def test_producer_metadata_preserves_order_under_reordered_iopub_arrival():
    from colabcola.bridge_video_comm_protocol import encode_video_reply
    from colabcola.bridge_video_stream import VideoAccessUnit

    ready = threading.Event()
    ready.set()
    channel = Channel()
    peer = SimpleNamespace(
        run_id=RUN, comm_id="owned",
        wire=SimpleNamespace(connection_ready=ready, iopub_channel=channel),
        _send=lambda *a, **k: None,
    )
    window = KernelH264PullWindow(peer, credits=2)
    window.submit(1)
    window.submit(2)
    for pull, wid in ((2, 8), (1, 7)):
        unit = VideoAccessUnit(seq=pull - 1, pts_ns=(pull - 1) * 33_333_333,
                               capture_ns=10 + pull, data=IDR, keyframe=True)
        payload = encode_video_reply(run_id=RUN, pull_seq=pull, input_ack=0, unit=unit)
        payload.update(renderer_seq=50 + pull, source_read_us=2000.0,
                       encode_us=3000.0, producer_window=producer_window(wid))
        channel.messages.append({"header": {"msg_type": "comm_msg"},
                                 "content": {"comm_id": "owned", "data": payload}})
    assert window.poll(.01) == []
    ordered = window.poll(.01)
    assert [u.seq for u, *_ in ordered] == [0, 1]
    assert [v["producer_window"]["window_seq"]
            for v in window.last_committed_diagnostics] == [7, 8]
    assert window.outstanding == 0


def test_duplicate_producer_window_is_only_sampled_once():
    stats = _empty_phase()
    first = producer_window(11, published_count=25)
    second = producer_window(12, published_count=29)
    for _ in range(24):
        _record_producer_window(stats, {"producer_window": first})
    for _ in range(24):
        _record_producer_window(stats, {"producer_window": second})
    _record_producer_window(stats, {"producer_window": None})
    report = _summarize_phase("C4", 4, stats, measured_s=8, end_gap_ms=0)
    assert report["producer_windows_observed"] == 2
    assert report["producer_renderer_fps_p50"] == 27
    assert report["producer_renderer_fps_min"] == 25
    assert report["producer_renderer_fps_max"] == 29
    assert report["producer_render_copy_p50_ms"] == 3
    assert report["producer_compress_p50_ms"] == 5
    assert report["producer_publish_p50_ms"] == .2
    assert report["producer_window_metrics_missing_replies"] == 1
    assert report["producer_window_conflicts"] == 0


def test_conflicting_producer_windows_fail_closed_as_telemetry_only():
    stats = _empty_phase()
    first = producer_window(12, published_count=25)
    changed = producer_window(12, published_count=26)
    _record_producer_window(stats, {"producer_window": first})
    _record_producer_window(stats, {"producer_window": changed})
    _record_producer_window(stats, {"producer_window": first})
    result = _summarize_phase("C4", 4, stats, measured_s=8, end_gap_ms=0)
    assert result["producer_windows_observed"] == 0
    assert result["producer_renderer_fps_p50"] is None
    assert result["producer_window_conflicts"] == 1


def test_no_producer_metrics_is_not_reported_as_zero_fps():
    stats = _empty_phase()
    _record_producer_window(stats, None)
    report = _summarize_phase("C1", 1, stats, measured_s=8, end_gap_ms=0)
    assert report["producer_windows_observed"] == 0
    assert report["producer_renderer_fps_p50"] is None
    assert report["producer_window_metrics_missing_replies"] == 1
