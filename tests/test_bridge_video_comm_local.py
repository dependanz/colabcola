"""v33b: offline H.264/Comm-envelope integration and fail-closed checks."""
from __future__ import annotations

import queue
import threading
import time

import pytest

from colabcola.bridge_video_comm_protocol import (
    VideoCommProtocolError, encode_video_reply,
)
from colabcola.bridge_video_comm_local import (
    LocalVideoCommInbox, LocalVideoCommProducer, _validated_drops,
    run_local_video_comm,
)
from colabcola.bridge_video_live import LiveScene
from colabcola.bridge_video_stream import VideoAccessUnit

RUN = "b" * 32
IDR = b"\x00\x00\x00\x01\x65\x88\x84"  # only for protocol tests, not real decoding
P = b"\x00\x00\x00\x01\x41\x88\x84"


def envelope(seq, *, idr=False, pull=None, ack=0):
    unit = VideoAccessUnit(
        seq=seq, pts_ns=seq * 33333333,
        capture_ns=1000 + seq, data=IDR if idr else P,
        keyframe=idr)
    return encode_video_reply(
        run_id=RUN, pull_seq=seq + 1 if pull is None else pull,
        input_ack=ack, unit=unit)


def test_inbox_accepts_matching_correlated_idr_and_p_frame():
    receiver = LocalVideoCommInbox(RUN)
    unit, ack = receiver.receive(envelope(0, idr=True, ack=3), expected_pull_seq=1)
    assert unit.seq == 0 and ack == 3
    unit, ack = receiver.receive(envelope(1, ack=3), expected_pull_seq=2)
    assert unit.seq == 1 and ack == 3
    assert receiver.suppressed_units == 0


def test_inbox_gates_p_frames_after_missing_reply_until_idr():
    receiver = LocalVideoCommInbox(RUN)
    assert receiver.receive(envelope(0, idr=True), expected_pull_seq=1)[0].seq == 0
    # Pull #2 / source #1 was sent but its reply was omitted.
    unit, _ = receiver.receive(envelope(2), expected_pull_seq=3)
    assert unit is None
    assert receiver.receive(envelope(3), expected_pull_seq=4)[0] is None
    unit, _ = receiver.receive(envelope(4, idr=True), expected_pull_seq=5)
    assert unit.seq == 4
    assert receiver.receive(envelope(5), expected_pull_seq=6)[0].seq == 5
    assert receiver.suppressed_units == 2


def test_inbox_mismatch_does_not_advance_state_and_repeated_pull_fails():
    receiver = LocalVideoCommInbox(RUN)
    with pytest.raises(VideoCommProtocolError):
        receiver.receive(envelope(0, idr=True, pull=7), expected_pull_seq=1)
    assert receiver.last_received_pull == 0
    assert receiver.receive(envelope(0, idr=True), expected_pull_seq=1)[0]
    with pytest.raises(VideoCommProtocolError, match="stale or repeated"):
        receiver.receive(envelope(0, idr=True), expected_pull_seq=1)
    with pytest.raises(VideoCommProtocolError):
        receiver.receive(envelope(1), expected_pull_seq=3)  # wrong reply pull_id
    assert receiver.last_received_pull == 1


def test_local_input_applies_valid_events_without_propagating_viewer_metadata():
    source = LocalVideoCommProducer.__new__(LocalVideoCommProducer)
    source.scene = LiveScene()
    source.input_ack = 0
    source.latest_callback_ns = None
    callback = time.perf_counter_ns()
    event = {"event": "mouse", "x": .8, "y": .3,
             "_ui_callback_ns": callback, "not_a_provider_field": "secret"}
    assert source.send_input(1, event)
    assert source.input_ack == 1
    assert (source.scene.x, source.scene.y) == (.8, .3)
    assert source.latest_callback_ns == callback
    assert not source.send_input(1, event)
    assert not source.send_input(2, {"event": "key", "key": "Escape"})
    assert source.input_ack == 1
    assert source.send_input(2, {"event": "key", "key": "Left"})
    assert source.scene.x < .8
    assert source.latest_callback_ns is None


@pytest.mark.parametrize("invalid", [(1, 1), (3, 2), (-1,), (1.0,), (True,), (24,), [1]])
def test_loss_injection_requires_sorted_unique_in_range(invalid):
    with pytest.raises(ValueError):
        _validated_drops(invalid, 24)



def test_complete_local_worker_contract_with_fake_codec_and_missing_reply(monkeypatch):
    """Always exercises the full worker, even on hosts without optional PyAV."""
    from types import SimpleNamespace
    from colabcola import bridge_video_comm_local as local

    class FakeFrame:
        def __init__(self, seq):
            self.seq = seq
            self.pts = None

        @staticmethod
        def from_ndarray(array, *, format):
            assert format == "rgb24"
            return FakeFrame(array)

    class FakePacket:
        def __init__(self, data):
            self.data = data
            self.pts = None

    class FakeEncoded:
        def __init__(self, seq):
            self.seq = seq
            self.is_keyframe = seq % 4 == 0

        def __bytes__(self):
            marker = b"\x65" if self.is_keyframe else b"\x41"
            return b"\x00\x00\x00\x01" + marker + bytes((self.seq,))

    class FakeEncoder:
        def encode(self, frame):
            return [FakeEncoded(frame.seq)]

    class FakeArray:
        def tobytes(self):
            return bytes((27, 63, 88)) * (96 * 64)

    class FakeDecoded:
        def to_ndarray(self, *, format):
            assert format == "rgb24"
            return FakeArray()

    class FakeDecoder:
        def decode(self, packet):
            return [] if packet is None else [FakeDecoded()]

    def fake_codecs(width, height, fps):
        assert (width, height, fps) == (96, 64, 60)
        return (SimpleNamespace(VideoFrame=FakeFrame, Packet=FakePacket),
                None, FakeEncoder(), FakeDecoder())

    monkeypatch.setattr(local, "make_live_codecs", fake_codecs)
    monkeypatch.setattr(local, "live_background", lambda *args: None)
    monkeypatch.setattr(local, "render_live_rgb", lambda scene, base, seq: seq)
    outgoing = queue.Queue()
    outgoing.put({"event": "mouse", "x": .9, "y": .2})
    images = queue.Queue(maxsize=32)
    report = local.run_local_video_comm(
        outgoing, images, threading.Event(),
        width=96, height=64, fps=60, max_frames=8,
        drop_source_seqs=(1,))
    assert report["received_envelopes"] == 7
    assert report["synthetic_missing_replies"] == 1
    assert report["decoded_source_seqs"] == [0, 4, 5, 6, 7]
    assert report["suppressed_dependent_units"] == 2
    assert report["accepted_input_events"] == 1
    assert report["last_input_ack_observed_in_decoded_frame"] == 1
    assert len([v for kind, v in list(images.queue) if kind == "frame"]) == 5

def _codec_available():
    av = pytest.importorskip("av", reason="optional PyAV unavailable")
    pytest.importorskip("numpy")
    try:
        av.CodecContext.create("libx264", "w")
    except Exception:
        pytest.skip("PyAV build lacks libx264")


def test_real_h264_comm_envelopes_decode_every_frame_and_ack_input():
    _codec_available()
    outgoing = queue.Queue(maxsize=8)
    outgoing.put_nowait({"event": "mouse", "x": .8, "y": .3,
                         "_ui_callback_ns": time.perf_counter_ns()})
    images = queue.Queue(maxsize=64)
    stopped = threading.Event()
    report = run_local_video_comm(
        outgoing, images, stopped, width=96, height=64,
        fps=12, max_frames=16, timed_frames=True)
    assert stopped.is_set()
    assert report["produced_frames"] == report["received_envelopes"] == 16
    assert report["decoded_frames"] == 16
    assert report["decoded_source_seqs"] == list(range(16))
    assert report["suppressed_dependent_units"] == 0
    assert report["accepted_input_events"] == 1
    assert report["last_input_ack_observed_in_decoded_frame"] == 1
    assert report["input_callback_to_decoded_p50_ms"] is not None
    assert report["capture_to_decoded_p50_ms"] is not None
    from colabcola.bridge_viewer_stats import LocalFrameTiming
    displayed = [item for item in list(images.queue) if item[0] == "frame-timed"]
    assert len(displayed) == 16
    assert isinstance(displayed[0][1][1], LocalFrameTiming)
    assert displayed[0][1][0].startswith(b"P6\n96 64\n255\n")
    assert "not Jupyter Comm" in report["scope"]


def test_real_h264_recovers_after_injected_missing_reply_at_next_idr():
    _codec_available()
    report = run_local_video_comm(
        queue.Queue(), queue.Queue(maxsize=2), threading.Event(),
        width=96, height=64, fps=12, max_frames=16,
        drop_source_seqs=(1,))
    assert report["produced_frames"] == 16
    assert report["received_envelopes"] == 15
    assert report["synthetic_missing_replies"] == 1
    assert report["suppressed_dependent_units"] > 0
    assert 0 in report["decoded_source_seqs"]
    assert 1 not in report["decoded_source_seqs"]
    assert any(seq > 1 for seq in report["decoded_source_seqs"])
    assert report["decoded_frames"] < 16


def test_cli_show_uses_existing_viewer_without_allocating(monkeypatch):
    from colabcola import bridge_video_comm_local as local
    from colabcola import bridge_demo
    captured = []
    monkeypatch.setattr(bridge_demo, "run_viewer",
                        lambda *args, **kwargs: captured.append(kwargs))
    assert local.main(["--width", "96", "--height", "64", "--fps", "12",
                       "--frames", "8", "--max-seconds", "1", "--show"]) == 0
    assert captured[0]["viewer_stats"] is True
    assert captured[0]["timestamp_local_input"] is True
    assert callable(captured[0]["worker"])
    for args in (["--drop-source-seqs", "2,1"],
                 ["--viewer-poll-ms", "4"],
                 ["--frames", "100", "--max-seconds", "1"]):
        with pytest.raises(SystemExit) as exc:
            local.main(["--fps", "12", *args])
        assert exc.value.code == 2
