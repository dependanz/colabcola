"""v34 sender-driven Comm and host UI timing; offline, no Colab/GPU/HTTP."""
from __future__ import annotations

from collections import deque
from pathlib import Path
import queue
import struct
import threading
import time
from types import SimpleNamespace

import pytest

from colabcola.bridge_video_push_runtime import (
    CreditPushSender, MAX_PUSH_CREDITS, newest_frame_seq,
)
from colabcola.bridge_video_push_session import (
    PUSH_CREDITS, decode_push_message, run_push_comm_h264,
    validate_push_message,
)
from colabcola.bridge_video_comm_protocol import (
    VideoSequenceGate, encode_video_reply,
)
from colabcola.bridge_video_stream import VideoAccessUnit
from colabcola.lifecycle_io import LifecycleError

RUN = "e" * 32
IDR = b"\x00\x00\x00\x01\x65\x88\x84"
P = b"\x00\x00\x00\x01\x41\x88\x84"


def publish_header(directory: Path, seq):
    (directory / "frame.bin").write_bytes(
        struct.pack("!4sIIII", b"CCB1", seq, 0, 640, 360))


class Source:
    sequence_buffered = True
    def __init__(self):
        self.last_renderer_seq = None
        self.calls = []
    def pull(self, pull_seq):
        self.calls.append(pull_seq)
        self.last_renderer_seq = pull_seq - 1
        return {"kind": "video_access_unit", "run_id": RUN,
                "pull_seq": pull_seq, "v": 1}


class Comm:
    def __init__(self):
        self.sent = []
    def send(self, payload):
        self.sent.append(payload)


def test_push_sender_never_sends_without_credit_or_new_snapshot(tmp_path):
    source, comm = Source(), Comm()
    sender = CreditPushSender(
        comm=comm, source=source, directory=tmp_path, run_id=RUN)
    assert sender.step() is False
    sender.grant(MAX_PUSH_CREDITS)
    assert sender.step() is False
    publish_header(tmp_path, 0)
    assert newest_frame_seq(tmp_path) == 0
    assert sender.step() is True
    assert len(comm.sent) == 1
    assert sender.available == 3
    assert sender.step() is False
    publish_header(tmp_path, 1)
    assert sender.step() is True
    assert source.calls == [1, 2]
    assert sender.available == 2
    with pytest.raises(ValueError):
        sender.grant(3)  # would exceed the four-credit window
    sender.grant(1)
    assert sender.available == 3
    sender.close()


@pytest.mark.parametrize("bad", [0, -1, 5, True, 1.0, "1"])
def test_push_sender_rejects_bad_credit(tmp_path, bad):
    sender = CreditPushSender(
        comm=Comm(), source=Source(), directory=tmp_path, run_id=RUN)
    with pytest.raises(ValueError):
        sender.grant(bad)


def test_push_sender_is_terminal_on_send_failure(tmp_path):
    class BadComm:
        def send(self, payload):
            raise OSError("private token must not enter error text")
    sender = CreditPushSender(
        comm=BadComm(), source=Source(), directory=tmp_path, run_id=RUN)
    publish_header(tmp_path, 0)
    sender.grant(4)
    assert sender.step() is False
    assert sender.failed and sender.stopped.is_set()
    with pytest.raises(ValueError):
        sender.grant(1)


def payload_message(payload, *, run=RUN, comm="authenticated"):
    return {"header": {"msg_type": "comm_msg"},
            "content": {"comm_id": comm, "data": {**payload, "run_id": run}}}


def h264_reply(seq, *, pull_seq, ack, idr):
    unit = VideoAccessUnit(
        seq=seq, pts_ns=seq * 33333333, capture_ns=100 + seq,
        data=IDR if idr else P, keyframe=idr)
    return encode_video_reply(run_id=RUN, pull_seq=pull_seq,
                              input_ack=ack, unit=unit)


def test_push_receiver_requires_same_run_and_comm():
    good = h264_reply(0, pull_seq=1, ack=0, idr=True)
    assert validate_push_message(payload_message(good),
                                 comm_id="authenticated", run_id=RUN) == good
    assert validate_push_message(payload_message(good, comm="other"),
                                 comm_id="authenticated", run_id=RUN) is None
    assert validate_push_message(payload_message(good, run="0" * 32),
                                 comm_id="authenticated", run_id=RUN) is None


def test_push_gate_decodes_only_strictly_monotone_sequences():
    gate = VideoSequenceGate()
    p0 = h264_reply(0, pull_seq=1, ack=0, idr=True)
    p1 = h264_reply(1, pull_seq=2, ack=3, idr=False)
    unit, ack, renderer = decode_push_message(
        {**p0, "renderer_seq": 14}, expected_seq=1, run_id=RUN,
        last_ack=0, last_renderer_seq=-1, gate=gate)
    assert (unit.seq, ack, renderer) == (0, 0, 14)
    unit, ack, renderer = decode_push_message(
        {**p1, "renderer_seq": 15}, expected_seq=2, run_id=RUN,
        last_ack=0, last_renderer_seq=14, gate=gate)
    assert (unit.seq, ack, renderer) == (1, 3, 15)
    with pytest.raises(LifecycleError):
        decode_push_message(
            {**p1, "renderer_seq": 15}, expected_seq=2,
            run_id=RUN, last_ack=3, last_renderer_seq=15, gate=gate)


@pytest.mark.parametrize("bad", [
    {"pull_seq": True}, {"pull_seq": 2}, {"input_ack": True},
    {"data": "unexpected"}, {"renderer_seq": 3},
])
def test_push_empty_replies_fail_closed_when_malformed(bad):
    gate = VideoSequenceGate()
    empty = {"v": 1, "kind": "video_no_new", "run_id": RUN,
             "pull_seq": 1, "input_ack": 0, **bad}
    with pytest.raises(LifecycleError):
        decode_push_message(empty, expected_seq=1, run_id=RUN,
                            last_ack=0, last_renderer_seq=-1, gate=gate)


def test_valid_empty_push_does_not_advance_h264_gate():
    gate = VideoSequenceGate()
    empty = {"v": 1, "kind": "video_no_new", "run_id": RUN,
             "pull_seq": 1, "input_ack": 0}
    assert decode_push_message(empty, expected_seq=1, run_id=RUN,
                               last_ack=0, last_renderer_seq=-1,
                               gate=gate) == (None, 0, -1)
    assert gate.last_seq == -1


def test_host_push_sends_credit_not_pulls_and_stamps_input_for_tk(monkeypatch):
    from colabcola import bridge_video_push_session as mod
    from colabcola.bridge_colab_runtime import make_producer_window
    window = make_producer_window(
        run_id=RUN, last_seq=102, window_seq=1, published_count=30,
        elapsed_s=1.0, target_fps=30, rate_seq=0,
        render_samples=[1.5], compress_samples=[8.0],
        publish_samples=[0.5])
    ready = threading.Event()
    ready.set()
    messages = deque()
    stopped = threading.Event()
    events = queue.Queue()
    callback_ns = time.perf_counter_ns() - 5_000_000
    events.put({"event": "mouse", "x": 0.7, "y": 0.4,
                "_ui_callback_ns": callback_ns})
    sent = []
    displayed = []

    class Channel:
        def get_msg(self, timeout=0):
            if messages:
                return messages.popleft()
            raise queue.Empty

    class Peer:
        video_push = True
        video_codec = "h264"
        comm_id = "authenticated"
        run_id = RUN
        wire = SimpleNamespace(
            connection_ready=ready, iopub_channel=Channel())
        def _send(self, kind, **kwargs):
            sent.append((kind, kwargs))
            if kind == "video_credit" and kwargs == {"count": 4}:
                for seq, ack in ((0, 1), (1, 1)):
                    p = h264_reply(seq, pull_seq=seq + 1, ack=ack,
                                   idr=seq == 0)
                    p.update(renderer_seq=100 + seq * 2,
                             source_read_us=4200.0, encode_us=2700.0,
                             producer_window=window)
                    messages.append(payload_message(p))
        def send_input(self, seq, event):
            assert seq == 1
            assert event == {"event": "mouse", "x": 0.7, "y": 0.4}
            sent.append(("input", {"seq": seq}))

    class Frame:
        width, height = 640, 360
        def to_ndarray(self, format):
            assert format == "rgb24"
            return SimpleNamespace(
                tobytes=lambda: b"\x05\x06\x07" * (640 * 360))

    class Packet:
        def __init__(self, data):
            self.pts = None

    class Decoder:
        def decode(self, packet):
            return [Frame()]

    monkeypatch.setattr(mod, "make_host_decoder",
                        lambda: (SimpleNamespace(Packet=Packet), Decoder()))
    def publish(queue_out, item):
        displayed.append(item)
        if item[0] == "frame-timed" and sum(
                x[0] == "frame-timed" for x in displayed) >= 2:
            stopped.set()
    monkeypatch.setattr(mod, "offer_update", publish)
    report = run_push_comm_h264(
        Peer(), "Tesla T4", events, queue.Queue(), stopped,
        time.monotonic() + 0.5)
    assert report["decoded_frames"] == 2
    assert report["configured_credits"] == PUSH_CREDITS
    assert report["input_to_decoded_observations"] == 1
    assert report["input_callback_to_receive_p50_ms"] is not None
    assert report["input_callback_to_decode_p50_ms"] is not None
    assert report["input_callback_to_decode_p50_ms"] >= (
        report["input_callback_to_receive_p50_ms"])
    assert report["host_receive_interval_p50_ms"] is not None
    assert report["renderer_frames_skipped_total"] == 1
    assert report["producer_renderer_fps_p50"] == 30
    assert report["producer_windows_observed"] == 1
    assert report["remote_encode_p50_ms"] == 2.7
    assert [name for name, _ in sent].count("video_credit") == 3
    assert not any(name == "video_pull" for name, _ in sent)
    timed = [item[1][1] for item in displayed if item[0] == "frame-timed"]
    assert len(timed) == 2
    assert timed[0].input_callback_ns == callback_ns
    assert timed[0].captured_ns <= timed[0].decoded_ns
    assert displayed[1][1][0].startswith(b"P6\n640 360\n255\n")


def test_cli_rejects_managed_or_pull_push_mode(monkeypatch):
    from colabcola.cli import main
    from colabcola import bridge_colab
    seen = []
    monkeypatch.setattr(bridge_colab, "run_colab", lambda args: seen.append(args) or 0)
    base = ["--profile", "example", "bridge", "colab",
            "--transport", "comm", "--video-codec", "h264",
            "--fps", "30", "--poll-hz", "30", "--push-video",
            "--preflight-only", "--max-seconds", "180"]
    assert main(base + ["--attach-existing", "--sequence-buffered"]) == 0
    assert seen[-1].push_video is True
    for extra in ([], ["--attach-existing"],
                  ["--attach-existing", "--sequence-buffered",
                   "--benchmark-h264-capacity"]):
        with pytest.raises(SystemExit):
            main(base + extra)



def test_push_credit_total_including_inflight_is_bounded(tmp_path):
    sender = CreditPushSender(
        comm=Comm(), source=Source(), directory=tmp_path, run_id=RUN)
    sender.grant(4)
    with pytest.raises(ValueError):
        sender.grant(1)  # no message has been sent or acknowledged
    publish_header(tmp_path, 0)
    assert sender.step()
    sender.grant(1)
    assert sender.available == 4
    with pytest.raises(ValueError):
        sender.grant(1)  # cannot return the same outstanding frame twice
    with pytest.raises(ValueError):
        sender.grant(4)  # only the initial four-credit grant may be batched
    assert sender.available + sender.sent_count - sender.replenished == 4
    sender.close()
