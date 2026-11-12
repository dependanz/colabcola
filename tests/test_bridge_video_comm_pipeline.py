"""v33i offline two-credit pipeline regression tests, no Colab allocation."""
from __future__ import annotations

from collections import deque
import queue
import threading
from types import SimpleNamespace

import pytest

from colabcola.bridge_video_comm_protocol import encode_video_reply
from colabcola.bridge_video_comm_pipeline import KernelH264PullWindow
from colabcola.bridge_video_stream import VideoAccessUnit, PACKET_HEADER
from colabcola.lifecycle_io import LifecycleError

RUN = 'e' * 32
IDR = b'\x00\x00\x00\x01\x65\x88\x84'
P = b'\x00\x00\x00\x01\x41\x88\x84'


class Channel:
    def __init__(self):
        self.messages = deque()
    def get_msg(self, timeout=None):
        if not self.messages:
            raise queue.Empty()
        return self.messages.popleft()


class Peer:
    def __init__(self):
        ready = threading.Event()
        ready.set()
        self.run_id = RUN
        self.comm_id = 'owned-comm'
        self.video_codec = 'h264'
        self.wire = SimpleNamespace(connection_ready=ready, iopub_channel=Channel())
        self.requests = []
    def _send(self, kind, **kwargs):
        self.requests.append((kind, kwargs))
    def reply(self, seq, *, pull_seq, idr=False, ack=0, run_id=RUN, comm_id='owned-comm', corrupt=False):
        unit = VideoAccessUnit(seq=seq, pts_ns=seq * 83333333, capture_ns=17 + seq,
                               data=IDR if idr else P, keyframe=idr)
        payload = encode_video_reply(run_id=run_id, pull_seq=pull_seq, input_ack=ack, unit=unit)
        if corrupt:
            payload['data'] = '@@@'
        self.wire.iopub_channel.messages.append({
            'header': {'msg_type': 'comm_msg'},
            'content': {'comm_id': comm_id, 'data': payload},
        })


def test_two_credits_limit_and_monotonic_pulls():
    peer = Peer()
    w = KernelH264PullWindow(peer)
    w.submit(1)
    w.submit(2)
    assert w.outstanding == w.maximum_outstanding == 2
    with pytest.raises(LifecycleError, match='credit'):
        w.submit(3)
    assert peer.requests == [('video_pull', {'pull_seq': 1}),
                             ('video_pull', {'pull_seq': 2})]
    peer.reply(0, pull_seq=1, idr=True)
    assert w.poll(.01)[0][0].seq == 0
    assert w.outstanding == 1
    w.submit(3)
    with pytest.raises(LifecycleError, match='credit'):
        w.submit(3)


def test_out_of_order_wire_replies_commit_in_decoder_order():
    peer = Peer()
    w = KernelH264PullWindow(peer)
    w.submit(1)
    w.submit(2)
    peer.reply(1, pull_seq=2, ack=3)
    peer.reply(0, pull_seq=1, idr=True, ack=1)
    assert w.poll(.01) == []
    assert w.outstanding == 2
    batch = w.poll(.01)
    assert [unit.seq for unit, *_ in batch] == [0, 1]
    assert [ack for _, ack, *_ in batch] == [1, 3]
    assert w.outstanding == 0
    assert w.input_ack == 3
    assert batch[0][2] == (len(IDR), PACKET_HEADER.size + len(IDR),
                            4 * ((PACKET_HEADER.size + len(IDR) + 2) // 3))


def test_arrival_order_ack_regression_permitted_but_committed_regression_not():
    peer = Peer()
    w = KernelH264PullWindow(peer)
    w.submit(1)
    w.submit(2)
    peer.reply(1, pull_seq=2, ack=8)
    peer.reply(0, pull_seq=1, ack=7, idr=True)
    assert w.poll(.01) == []
    assert [ack for _, ack, *_ in w.poll(.01)] == [7, 8]
    w.submit(3)
    peer.reply(2, pull_seq=3, ack=6)
    with pytest.raises(LifecycleError, match='ordered frame invalid'):
        w.poll(.01)
    assert w.failed


def test_missing_access_unit_requires_idr_resynchronization():
    peer = Peer()
    w = KernelH264PullWindow(peer)
    for p, seq, idr in [(1, 0, True), (2, 2, False), (3, 3, False),
                        (4, 4, True), (5, 5, False)]:
        w.submit(p)
        peer.reply(seq, pull_seq=p, idr=idr)
        committed = w.poll(.01)
        assert len(committed) == 1
        assert (committed[0][0] is not None) == (p in (1, 4, 5))
    assert w.suppressed_dependent_units == 2


@pytest.mark.parametrize('kind', ['duplicate', 'uncorrelated', 'invalid', 'error'])
def test_malformed_correlated_or_unsolicited_replies_fail_closed(kind):
    peer = Peer()
    w = KernelH264PullWindow(peer)
    w.submit(1)
    peer.reply(0, pull_seq=1, idr=True)
    assert len(w.poll(.01)) == 1
    w.submit(2)
    if kind == 'duplicate':
        peer.reply(0, pull_seq=1, idr=True)
    elif kind == 'uncorrelated':
        peer.reply(2, pull_seq=3)
    elif kind == 'invalid':
        peer.reply(1, pull_seq=2, corrupt=True)
    else:
        peer.wire.iopub_channel.messages.append({
            'header': {'msg_type': 'comm_msg'},
            'content': {'comm_id': peer.comm_id,
                        'data': {'v': 1, 'run_id': RUN, 'kind': 'error'}},
        })
    with pytest.raises(LifecycleError):
        w.poll(.01)
    assert w.failed
    with pytest.raises(LifecycleError):
        w.submit(3)


def test_other_comm_or_run_is_ignored():
    peer = Peer()
    w = KernelH264PullWindow(peer)
    w.submit(1)
    peer.reply(0, pull_seq=1, idr=True, comm_id='other')
    peer.reply(0, pull_seq=1, idr=True, run_id='a'*32)
    peer.reply(0, pull_seq=1, idr=True)
    assert w.poll(.01) == []
    assert w.poll(.01) == []
    assert w.poll(.01)[0][0].seq == 0


def test_transport_failure_and_disconnect_are_terminal():
    peer = Peer()
    def bad_send(*a, **kw):
        raise RuntimeError('private credentials')
    peer._send = bad_send
    w = KernelH264PullWindow(peer)
    with pytest.raises(LifecycleError, match='submission failed') as exc:
        w.submit(1)
    assert 'credentials' not in str(exc.value)
    assert w.failed
    peer2 = Peer()
    w2 = KernelH264PullWindow(peer2)
    w2.submit(1)
    peer2.wire.connection_ready.clear()
    with pytest.raises(LifecycleError, match='disconnected'):
        w2.poll(.01)
    assert w2.failed


def test_missing_earlier_reply_times_out_without_replay(monkeypatch):
    from colabcola import bridge_video_comm_pipeline as module
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock.now)
    peer = Peer()
    w = KernelH264PullWindow(peer)
    w.submit(1)
    w.submit(2)
    peer.reply(1, pull_seq=2)
    assert w.poll(.01) == []
    clock.now = 8.1
    with pytest.raises(LifecycleError, match='deadline'):
        w.check_timeouts()
    assert w.failed
    assert len(peer.requests) == 2


def test_no_unbounded_credits_or_nonconsecutive_ids():
    peer = Peer()
    for invalid in (0, 1, 3, True, 2.0):
        with pytest.raises(ValueError):
            KernelH264PullWindow(peer, credits=invalid)
    w = KernelH264PullWindow(peer)
    with pytest.raises(LifecycleError):
        w.submit(2)
    w.submit(1)
    with pytest.raises(LifecycleError):
        w.submit(3)


def test_host_worker_decodes_reordered_wire_replies_without_remote_clock_math(monkeypatch):
    """Exercise actual two-credit host worker with fake authenticated kernel and codec."""
    import time
    from colabcola import bridge_video_comm_pipeline as mod

    class Frame:
        width, height = 640, 360
        def to_ndarray(self, *, format):
            assert format == 'rgb24'
            return SimpleNamespace(tobytes=lambda: bytes((8, 18, 28)) * (640 * 360))
    class Decoder:
        def __init__(self):
            self.seqs = []
        def decode(self, packet):
            self.seqs.append(packet.pts)
            return [Frame()]
    class Packet:
        def __init__(self, data):
            self.pts = None
            self.data = data
    decoder = Decoder()
    monkeypatch.setattr(mod, 'make_host_decoder',
                        lambda: (SimpleNamespace(Packet=Packet), decoder))
    peer = Peer()
    peer.sent_inputs = []
    peer.send_input = lambda seq, event: peer.sent_inputs.append((seq, event))
    original_send = peer._send
    pending_pairs = []
    def send_reply(kind, **fields):
        original_send(kind, **fields)
        pull = fields['pull_seq']
        if pull % 2 == 1:
            pending_pairs.append(pull)
        else:
            # Reverse the IOPub delivery order of each pair.
            peer.reply(pull - 1, pull_seq=pull, ack=len(peer.sent_inputs))
            first = pending_pairs.pop(0)
            peer.reply(first - 1, pull_seq=first, idr=(first == 1),
                       ack=len(peer.sent_inputs))
    peer._send = send_reply
    inputs = queue.Queue()
    inputs.put({'event': 'mouse', 'x': .6, 'y': .4})
    frames = queue.Queue(maxsize=32)
    stopped = threading.Event()
    mod.stream_comm_h264_pipeline(peer, 'Tesla T4', inputs, frames, stopped,
                                  time.monotonic() + .42, 12)
    assert stopped.is_set()
    assert peer.sent_inputs == [(1, {'event': 'mouse', 'x': .6, 'y': .4})]
    assert len(decoder.seqs) >= 2
    assert decoder.seqs == list(range(len(decoder.seqs)))
    assert all(kind == 'status' or image.startswith(b'P6\n640 360\n255\n')
               for kind, image in frames.queue)
    assert not any(t == 'frame' and len(frame) != 640*360*3 + len(b'P6\n640 360\n255\n')
                   for t, frame in frames.queue)


def test_host_worker_requires_opt_in_and_fixed_two_credits():
    import time
    from colabcola.bridge_video_comm_pipeline import stream_comm_h264_pipeline
    peer = Peer()
    for invalid in (0, 13, 1.2, True):
        with pytest.raises(ValueError):
            stream_comm_h264_pipeline(peer, 'T4', queue.Queue(), queue.Queue(),
                                      threading.Event(), time.monotonic() + 1, invalid)
    peer.video_codec = 'ppm'
    with pytest.raises(LifecycleError, match='explicitly opted-in'):
        stream_comm_h264_pipeline(peer, 'T4', queue.Queue(), queue.Queue(),
                                  threading.Event(), time.monotonic() + 1, 12)
