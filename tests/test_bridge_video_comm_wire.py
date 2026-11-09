"""v33c fake Jupyter IOPub/shell tests; no provider or GPU allocation."""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace
import queue
import threading

import pytest

from colabcola.bridge_video_comm_protocol import encode_video_reply
from colabcola.bridge_video_comm_wire import KernelVideoCommReceiver
from colabcola.bridge_video_stream import VideoAccessUnit
from colabcola.lifecycle_io import LifecycleError

RUN = "9" * 32
IDR = b"\x00\x00\x00\x01\x65\x88\x84"
P = b"\x00\x00\x00\x01\x41\x88\x84"


def video(seq, pull_seq, *, idr=False, ack=0, run=RUN):
    unit = VideoAccessUnit(seq, seq * 83333333, 100 + seq,
                           IDR if idr else P, idr)
    return encode_video_reply(run_id=run, pull_seq=pull_seq, input_ack=ack, unit=unit)


def comm_message(payload, *, comm_id="owned-comm"):
    return {"header": {"msg_type": "comm_msg"},
            "content": {"comm_id": comm_id, "data": payload}}


class FakeChannel:
    def __init__(self):
        self.messages = deque()

    def get_msg(self, timeout=None):
        try:
            return self.messages.popleft()
        except IndexError:
            raise queue.Empty()


class Peer:
    def __init__(self):
        self.run_id = RUN
        self.comm_id = "owned-comm"
        ready = threading.Event()
        ready.set()
        self.wire = SimpleNamespace(iopub_channel=FakeChannel(), connection_ready=ready)
        self.sends = []

    def _send(self, kind, **fields):
        self.sends.append((kind, fields))

    def deliver(self, payload, **kwargs):
        self.wire.iopub_channel.messages.append(comm_message(payload, **kwargs))


def test_single_inflight_uses_existing_authenticated_comm_and_correlated_reply():
    peer = Peer()
    rx = KernelVideoCommReceiver(peer)
    rx.request_video(1)
    assert peer.sends == [("video_pull", {"pull_seq": 1})]
    with pytest.raises(LifecycleError, match="outstanding"):
        rx.request_video(2)
    peer.wire.iopub_channel.messages.append({"header": {"msg_type": "status"}})
    peer.deliver(video(0, 1, idr=True, ack=2), comm_id="other-comm")
    peer.deliver(video(0, 1, idr=True, ack=2))
    unit, ack = rx.next_video(0.1)
    assert (unit.seq, ack, rx.last_input_ack, rx.pending_pull) == (0, 2, 2, None)
    rx.request_video(2)
    peer.deliver(video(1, 2, ack=2))
    assert rx.next_video(.1)[0].seq == 1


def test_other_run_video_message_ignored_then_owned_accepted():
    peer = Peer()
    rx = KernelVideoCommReceiver(peer)
    rx.request_video(1)
    peer.deliver(video(0, 1, idr=True, run="8" * 32))
    peer.deliver(video(0, 1, idr=True))
    assert rx.next_video(.1)[0].seq == 0


def test_gap_suppresses_dependent_and_recovers_at_idr():
    peer = Peer()
    rx = KernelVideoCommReceiver(peer)
    for p, seq, idr in ((1, 0, True), (2, 2, False), (3, 3, False),
                        (4, 4, True), (5, 5, False)):
        rx.request_video(p)
        peer.deliver(video(seq, p, idr=idr))
        unit, _ = rx.next_video(.1)
        assert (unit is not None) == idr or (seq == 5 and unit is not None)
    assert rx.suppressed_dependent_units == 2


@pytest.mark.parametrize("invalid", [
    lambda x: {**x, "pull_seq": 777},
    lambda x: {**x, "input_ack": -1},
    lambda x: {**x, "data": "***"},
])
def test_bad_payload_poisons_receiver_without_leaking_bytes(invalid):
    peer = Peer()
    rx = KernelVideoCommReceiver(peer)
    rx.request_video(1)
    peer.deliver(invalid(video(0, 1, idr=True)))
    with pytest.raises(LifecycleError, match="response invalid"):
        rx.next_video(.1)
    assert rx.failed
    with pytest.raises(LifecycleError, match="unavailable"):
        rx.request_video(2)


def test_ack_regression_fails_closed():
    peer = Peer()
    rx = KernelVideoCommReceiver(peer)
    rx.request_video(1)
    peer.deliver(video(0, 1, idr=True, ack=2))
    assert rx.next_video(.1)[1] == 2
    rx.request_video(2)
    peer.deliver(video(1, 2, ack=1))
    with pytest.raises(LifecycleError, match="response invalid"):
        rx.next_video(.1)


def test_timeout_poison_no_replay(monkeypatch):
    from colabcola import bridge_video_comm_wire as mod
    peer = Peer()
    rx = KernelVideoCommReceiver(peer)
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock.now)
    def fake_get_msg(timeout=None):
        clock.now += timeout
        raise queue.Empty()
    peer.wire.iopub_channel.get_msg = fake_get_msg
    rx.request_video(1)
    with pytest.raises(LifecycleError, match="timed out"):
        rx.next_video(.1)
    assert peer.sends == [("video_pull", {"pull_seq": 1})]
    with pytest.raises(LifecycleError):
        rx.request_video(2)


def test_disconnect_and_send_uncertainty_fail_without_replay():
    peer = Peer()
    rx = KernelVideoCommReceiver(peer)
    peer.wire.connection_ready.clear()
    with pytest.raises(LifecycleError):
        rx.request_video(1)
    assert not peer.sends
    peer2 = Peer()
    peer2._send = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret"))
    rx2 = KernelVideoCommReceiver(peer2)
    with pytest.raises(LifecycleError, match="submission failed") as exc:
        rx2.request_video(1)
    assert "secret" not in str(exc.value)
    with pytest.raises(LifecycleError):
        rx2.request_video(2)


def test_no_reply_without_pull_and_invalid_request_sequence():
    peer = Peer()
    rx = KernelVideoCommReceiver(peer)
    with pytest.raises(LifecycleError, match="without pull"):
        rx.next_video(.1)
    for bad in (0, True, -1, 100_000_001, 1.5):
        with pytest.raises(LifecycleError):
            rx.request_video(bad)
    rx.request_video(1)
    peer.deliver(video(0, 1, idr=True))
    rx.next_video(.1)
    with pytest.raises(LifecycleError):
        rx.request_video(1)


def test_legacy_ppm_comm_traffic_does_not_become_video():
    peer = Peer()
    rx = KernelVideoCommReceiver(peer)
    rx.request_video(1)
    peer.deliver({"v": 1, "run_id": RUN, "kind": "frame", "data": "PPM"})
    peer.deliver({"v": 1, "run_id": RUN, "kind": "rate_ack", "seq": 3})
    peer.deliver(video(0, 1, idr=True))
    assert rx.next_video(.1)[0].seq == 0


def test_malformed_iopub_metadata_ignored_before_correlated_reply():
    peer = Peer()
    rx = KernelVideoCommReceiver(peer)
    rx.request_video(1)
    peer.wire.iopub_channel.messages.extend([
        None, {}, {"header": None},
        {"header": {"msg_type": "comm_msg"}, "content": []},
    ])
    peer.deliver(video(0, 1, idr=True))
    assert rx.next_video(.1)[0].seq == 0


def test_channel_exception_redacted_and_terminal():
    peer = Peer()
    rx = KernelVideoCommReceiver(peer)
    rx.request_video(1)
    def broken_channel(timeout=None):
        raise RuntimeError("secret remote token")
    peer.wire.iopub_channel.get_msg = broken_channel
    with pytest.raises(LifecycleError, match="receive failed") as exc:
        rx.next_video(.1)
    assert "secret" not in str(exc.value)
    assert rx.failed
