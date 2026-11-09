"""v33h: size accounting on a validated, correlated Comm reply (no provider)."""
from __future__ import annotations

import base64
from types import SimpleNamespace
import queue
import threading

from colabcola.bridge_video_comm_protocol import encode_video_reply
from colabcola.bridge_video_comm_wire import KernelVideoCommReceiver
from colabcola.bridge_video_stream import VideoAccessUnit, PACKET_HEADER

RUN = "9" * 32

class Channel:
    def __init__(self):
        self.items = []
    def get_msg(self, timeout=None):
        if not self.items:
            raise queue.Empty
        return self.items.pop(0)


def test_reply_accounting_includes_suppressed_dependent_units():
    ready = threading.Event()
    ready.set()
    channel = Channel()
    peer = SimpleNamespace(
        run_id=RUN, comm_id="owned", wire=SimpleNamespace(
            connection_ready=ready, iopub_channel=channel),
        _send=lambda *a, **k: None)
    receiver = KernelVideoCommReceiver(peer)
    samples = [(0, True), (2, False), (3, True)]
    for pull, (source_seq, idr) in enumerate(samples, start=1):
        nal = b"\x00\x00\x00\x01" + (b"\x65" if idr else b"\x41") + b"\x88\x84"
        unit = VideoAccessUnit(seq=source_seq, pts_ns=source_seq * 100,
                               capture_ns=123 + source_seq, data=nal, keyframe=idr)
        reply = encode_video_reply(
            run_id=RUN, pull_seq=pull, input_ack=0, unit=unit)
        channel.items.append({"header": {"msg_type": "comm_msg"},
                              "content": {"comm_id": "owned", "data": reply}})
        receiver.request_video(pull)
        decoded, _ = receiver.next_video(.1)
        assert receiver.last_reply_sizes == (
            len(nal), len(nal) + PACKET_HEADER.size, len(reply["data"]))
        assert base64.b64decode(reply["data"], validate=True).endswith(nal)
        if source_seq == 2:
            assert decoded is None  # missing source #1 -> suppress dependent P
        else:
            assert decoded.seq == source_seq
    assert receiver.suppressed_dependent_units == 1
    receiver.request_video(4)
    assert receiver.last_reply_sizes is None  # no stale size reuse
