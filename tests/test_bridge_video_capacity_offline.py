"""Offline-only protocol checks for experimental 30/60 Hz and 1..32 credits."""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace
import queue
import threading

import pytest

from colabcola.bridge_video_colab_source import RemoteH264PullSource, RemoteVideoSourceError
from colabcola.bridge_video_comm_pipeline import (
    KernelH264PullWindow, MAX_EXPERIMENTAL_VIDEO_CREDITS,
)
from colabcola.bridge_video_comm_protocol import encode_video_reply
from colabcola.bridge_video_stream import VideoAccessUnit
from colabcola.lifecycle_io import LifecycleError

RUN = 'd' * 32


class FakeChannel:
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
        self.comm_id = 'only-owned-comm'
        self.wire = SimpleNamespace(connection_ready=ready, iopub_channel=FakeChannel())
        self.sent = []

    def _send(self, kind, **fields):
        self.sent.append((kind, fields))

    def reply(self, pull_seq, *, idr=False):
        nal = (b'\x00\x00\x00\x01\x65\x88\x84' if idr
               else b'\x00\x00\x00\x01\x41\x88\x84')
        unit = VideoAccessUnit(seq=pull_seq-1, pts_ns=(pull_seq-1)*16666667,
                               capture_ns=100+pull_seq, data=nal, keyframe=idr)
        payload = encode_video_reply(run_id=RUN, pull_seq=pull_seq,
                                     input_ack=0, unit=unit)
        self.wire.iopub_channel.messages.append({
            'header': {'msg_type': 'comm_msg'},
            'content': {'comm_id': self.comm_id, 'data': payload},
        })


def test_experimental_credits_allow_requested_windows_but_legacy_stays_fixed():
    for credits in (1, 2, 3, 4, 8, 16, 32):
        peer = Peer()
        window = KernelH264PullWindow(peer, credits=credits, experimental=True)
        assert window.credits == credits
    for credits in (0, -1, 33, True, 3.0):
        with pytest.raises(ValueError):
            KernelH264PullWindow(Peer(), credits=credits, experimental=True)
    for credits in (1, 3, 4, 8, 16, 32):
        with pytest.raises(ValueError):
            KernelH264PullWindow(Peer(), credits=credits)
    assert KernelH264PullWindow(Peer()).credits == 2
    assert MAX_EXPERIMENTAL_VIDEO_CREDITS == 32


def test_32_credit_window_buffers_reverse_replies_then_commits_in_order():
    peer = Peer()
    window = KernelH264PullWindow(peer, credits=32, experimental=True)
    for seq in range(1, 33):
        window.submit(seq)
    assert window.outstanding == window.maximum_outstanding == 32
    with pytest.raises(LifecycleError):
        window.submit(33)
    for seq in range(32, 0, -1):
        peer.reply(seq, idr=(seq == 1))
    for _ in range(31):
        assert window.poll(.01) == []
        assert window.outstanding == 32
    ordered = window.poll(.01)
    assert [item[0].seq for item in ordered] == list(range(32))
    assert window.outstanding == 0
    assert len(peer.sent) == 32


def test_remote_encoder_configuration_accepts_30_60_without_pyav(tmp_path):
    selected = []
    def factory(w, h, fps):
        selected.append((w,h,fps))
        return lambda rgb, seq: (b'\x00\x00\x00\x01\x65', True)
    for fps in (30, 60):
        source = RemoteH264PullSource(tmp_path, RUN, fps=fps, encoder_factory=factory)
        assert source.fps == fps
    assert selected == [(640,360,30), (640,360,60)]
    for fps in (0, 61, True):
        with pytest.raises(RemoteVideoSourceError):
            RemoteH264PullSource(tmp_path, RUN, fps=fps, encoder_factory=factory)
