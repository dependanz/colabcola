"""v33v sequence-buffered H.264 source and ordered Comm receiver tests.

Purely offline. No Colab, CUDA, PyAV, DNS, or GPU allocations.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
import queue
import struct
import threading
from types import SimpleNamespace
import zlib

import pytest

from colabcola.bridge_video_colab_source import (
    RING_SLOTS, RemoteH264PullSource, read_next_buffered_rgb,
)
from colabcola.bridge_video_comm_protocol import (
    VideoSequenceGate, VideoAccessUnit, decode_video_reply, encode_video_reply,
)
from colabcola.bridge_video_comm_pipeline import (
    KernelH264PullWindow, validated_remote_video_diagnostics,
)
from colabcola.bridge_video_credit_sweep import _empty_phase, _summarize_phase
from colabcola.lifecycle_io import LifecycleError

RUN = "f" * 32
IDR = b"\x00\x00\x00\x01\x65\x88\x84"
P = b"\x00\x00\x00\x01\x41\x88\x84"
W, H = 640, 360


def publish(directory: Path, seq: int, ack: int = 0) -> None:
    pixels = bytes((seq % 251, 20, 30)) * (W * H)
    ppm = b"P6\n640 360\n255\n" + pixels
    frame = struct.pack("!4sIIII", b"CCB1", seq, ack, W, H)
    frame += zlib.compress(ppm, 1)
    (directory / f"frame-ring-{seq % RING_SLOTS:02d}.bin").write_bytes(frame)


def source(directory):
    def make_encoder(width, height, fps):
        assert (width, height, fps) == (W, H, 30)
        def encode(pixels, coded_seq):
            assert len(pixels) == 3 * W * H
            return (IDR if coded_seq % 12 == 0 else P, coded_seq % 12 == 0)
        return encode
    return RemoteH264PullSource(directory, RUN, fps=30,
                                encoder_factory=make_encoder,
                                sequence_buffered=True)


def test_no_new_returns_correlated_empty_without_consuming_video_sequence(tmp_path):
    publish(tmp_path, 0, ack=2)
    src = source(tmp_path)
    first = src.pull(1)
    assert first["kind"] == "video_access_unit"
    assert first["renderer_seq"] == 0
    empty = src.pull(2)
    assert empty == {"v": 1, "kind": "video_no_new", "run_id": RUN,
                     "pull_seq": 2, "input_ack": 2}
    assert src.coded_seq == 1
    publish(tmp_path, 1, ack=3)
    third = src.pull(3)
    assert third["renderer_seq"] == 1
    assert third["renderer_frames_skipped"] == 0
    assert decode_video_reply(third, run_id=RUN, expected_pull_seq=3)[0].seq == 1
    assert src.coded_seq == 2
    with pytest.raises(Exception):
        src.pull(3)


def test_ring_reads_distinct_history_after_first_snapshot(tmp_path):
    publish(tmp_path, 0)
    src = source(tmp_path)
    assert src.pull(1)["renderer_seq"] == 0
    for seq in (1, 2, 3):
        publish(tmp_path, seq)
    # Only a genuinely new renderer frame is H.264-encoded per pull.
    assert src.pull(2)["renderer_seq"] == 1
    # Remaining ring entries are consumed oldest-first.
    assert src.pull(3)["renderer_seq"] == 2
    assert src.pull(4)["renderer_seq"] == 3
    assert src.pull(5)["kind"] == "video_no_new"


def test_ring_overrun_reports_skipped_source_not_encoded_gap(tmp_path):
    publish(tmp_path, 0)
    src = source(tmp_path)
    assert src.pull(1)["renderer_seq"] == 0
    for seq in range(1, RING_SLOTS + 5):
        publish(tmp_path, seq)
    result = src.pull(2)
    # Eight retained frames: 5..12; the freshness policy retains 10..12.
    assert result["renderer_seq"] == 10
    assert result["renderer_frames_skipped"] == 9
    assert result["renderer_buffer_lag"] == 2
    assert decode_video_reply(result, run_id=RUN, expected_pull_seq=2)[0].seq == 1
    assert src.pull(3)["renderer_seq"] == 11


def test_corrupt_ring_slot_is_ignored_but_never_repeated(tmp_path):
    publish(tmp_path, 0)
    src = source(tmp_path)
    assert src.pull(1)["renderer_seq"] == 0
    publish(tmp_path, 1)
    (tmp_path / "frame-ring-01.bin").write_bytes(b"not-a-CCB1-frame")
    assert src.pull(2)["kind"] == "video_no_new"
    assert src.coded_seq == 1
    publish(tmp_path, 2)
    assert src.pull(3)["renderer_seq"] == 2


class Channel:
    def __init__(self):
        self.messages = deque()
    def get_msg(self, timeout=None):
        if not self.messages:
            raise queue.Empty
        return self.messages.popleft()


def fake_peer(buffered=True):
    event = threading.Event()
    event.set()
    channel = Channel()
    requests = []
    peer = SimpleNamespace(
        run_id=RUN, comm_id="bridge", sequence_buffered=buffered,
        wire=SimpleNamespace(connection_ready=event, iopub_channel=channel),
        _send=lambda name, **kw: requests.append((name, kw)),
        requests=requests,
    )
    return peer


def queue_reply(peer, payload):
    peer.wire.iopub_channel.messages.append({
        "header": {"msg_type": "comm_msg"},
        "content": {"comm_id": peer.comm_id, "data": payload},
    })


def test_empty_reply_frees_credit_without_advancing_h264_gate():
    peer = fake_peer()
    window = KernelH264PullWindow(peer, credits=2)
    window.submit(1)
    window.submit(2)
    first_unit = VideoAccessUnit(seq=0, pts_ns=0, capture_ns=1,
                                 data=IDR, keyframe=True)
    encoded = encode_video_reply(run_id=RUN, pull_seq=2,
                                 input_ack=3, unit=first_unit)
    empty = {"v": 1, "kind": "video_no_new", "run_id": RUN,
             "pull_seq": 1, "input_ack": 0}
    queue_reply(peer, encoded)
    queue_reply(peer, empty)
    assert window.poll(0.0) == []
    result = window.poll(0.0)
    assert result[0][0] is None
    assert result[0][1] == 0
    assert result[0][2] == (0, 0, 0)
    assert result[1][0].seq == 0
    assert window.outstanding == 0
    assert window.suppressed_dependent_units == 0
    assert window.input_ack == 3
    assert window.last_committed_diagnostics[0] == {"no_new": True}


@pytest.mark.parametrize("bad", [
    {"input_ack": True},
    {"input_ack": -1},
    {"data": "unexpected"},
    {"renderer_seq": 10},
])
def test_corrupted_empty_reply_fails_closed(bad):
    peer = fake_peer()
    window = KernelH264PullWindow(peer, credits=2)
    window.submit(1)
    payload = {"v": 1, "kind": "video_no_new", "run_id": RUN,
               "pull_seq": 1, "input_ack": 0, **bad}
    queue_reply(peer, payload)
    with pytest.raises(LifecycleError, match="invalid correlated"):
        window.poll(0.0)
    assert window.failed


def test_legacy_peer_cannot_inject_empty_reply():
    peer = fake_peer(buffered=False)
    window = KernelH264PullWindow(peer, credits=2)
    window.submit(1)
    queue_reply(peer, {"v": 1, "kind": "video_no_new",
                       "run_id": RUN, "pull_seq": 1, "input_ack": 0})
    with pytest.raises(LifecycleError):
        window.poll(0.0)


def test_buffer_accounting_does_not_conflate_empty_and_suppression():
    stats = _empty_phase()
    stats["replies"] = 30
    stats["no_new_replies"] = 2
    stats["decoded"] = 28
    stats["renderer_seqs"] = set(range(28))
    stats["source_read_us"] = [500.0] * 28
    stats["renderer_frames_skipped_total"] = 1
    stats["renderer_buffer_lag_frames"] = [0, 1, 2]
    report = _summarize_phase("C4", 4, stats, measured_s=1, end_gap_ms=0)
    assert report["decoded_fps"] == 28
    assert report["distinct_renderer_reply_fps"] == 28
    assert report["no_new_renderer_frame_replies"] == 2
    assert report["renderer_frames_skipped_total"] == 1
    assert report["renderer_buffer_lag_p50_frames"] == 1
    assert report["renderer_buffer_lag_max_frames"] == 2
    assert report["suppressed_dependent_units"] == 0


def test_diagnostic_skip_count_is_strictly_bounded():
    valid = {"renderer_seq": 12, "source_read_us": 300.0,
             "encode_us": 100.0, "renderer_frames_skipped": 4}
    assert validated_remote_video_diagnostics(valid)["renderer_frames_skipped"] == 4
    assert validated_remote_video_diagnostics(
        {**valid, "renderer_buffer_lag": 2})["renderer_buffer_lag"] == 2
    assert validated_remote_video_diagnostics(
        {**valid, "renderer_buffer_lag": 3}) is None
    for x in (True, -1, 100_000_001, "4"):
        assert validated_remote_video_diagnostics(
            {**valid, "renderer_frames_skipped": x}) is None


def test_cli_sequence_buffered_is_explicitly_external_only(monkeypatch):
    from colabcola.cli import main
    from colabcola import bridge_colab

    seen = []
    monkeypatch.setattr(bridge_colab, "run_colab",
                        lambda args: seen.append(args) or 0)
    common = ["--profile", "example", "bridge", "colab",
              "--transport", "comm", "--video-codec", "h264",
              "--inflight", "1", "--fps", "30", "--poll-hz", "30",
              "--benchmark-h264-capacity", "--max-seconds", "300",
              "--sequence-buffered", "--preflight-only"]
    assert main(common + ["--attach-existing"]) == 0
    assert seen[-1].sequence_buffered is True
    assert seen[-1].attach_existing is True
    with pytest.raises(SystemExit):
        main(common)
    with pytest.raises(SystemExit):
        main(common + ["--attach-existing", "--confirm-allocation"])


def test_source_rejects_nonboolean_sequence_buffer(tmp_path):
    with pytest.raises(Exception):
        RemoteH264PullSource(
            tmp_path, RUN, fps=30,
            encoder_factory=lambda *_: (lambda pixels, seq: (IDR, True)),
            sequence_buffered=1)
