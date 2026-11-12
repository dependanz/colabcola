"""Virtual-clock capacity sweep checks, no provider, CUDA, or network."""
from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import pytest
from colabcola import bridge_video_credit_sweep as sweep


class Clock:
    def __init__(self): self.now = 100.0
    def monotonic(self): return self.now
    def sleep(self, t): self.now += max(t, 0)
    def perf_counter_ns(self): return int(self.now * 1e9)


class Image:
    width, height = 640, 360
    def to_ndarray(self, *, format):
        assert format == 'rgb24'
        return SimpleNamespace(tobytes=lambda: bytes((1,2,3))* (640*360))


class Decoder:
    def __init__(self): self.seqs = []
    def decode(self, packet):
        self.seqs.append(packet.pts)
        return [Image()]


class Packet:
    def __init__(self, data): self.pts=None


class Window:
    created=[]
    def __init__(self, peer, *, credits, experimental=False):
        self.peer=peer
        self.clock=peer.clock
        self.credits=credits
        self.pending=[]
        self.suppressed_dependent_units=0
        self.maximum_outstanding=0
        self.sends=[]
        assert (credits,experimental)==(32,True)
        self.created.append(self)
    @property
    def outstanding(self): return len(self.pending)
    def submit(self, pull_seq):
        assert self.outstanding < self.credits
        self.sends.append(pull_seq)
        self.pending.append((pull_seq, self.clock.now, self.clock.now+0.16))
        self.maximum_outstanding=max(self.maximum_outstanding,self.outstanding)
    def poll(self, seconds):
        self.clock.sleep(seconds)
        if self.pending and self.pending[0][2] <= self.clock.now:
            pull,t,_ = self.pending.pop(0)
            return [(SimpleNamespace(seq=pull-1, data=b'h264'),0,(100,130,176),
                     (self.clock.now-t)*1000)]
        return []
    def check_timeouts(self, *, now=None): pass


@pytest.mark.parametrize('hz', [30,60])
def test_seven_phases_have_continuous_sequence_and_bounded_concurrency(monkeypatch,hz):
    clock=Clock()
    monkeypatch.setattr(sweep,'time',clock)
    monkeypatch.setattr(sweep,'KernelH264PullWindow',Window)
    decoder=Decoder()
    monkeypatch.setattr(sweep,'make_host_decoder',lambda:(SimpleNamespace(Packet=Packet),decoder))
    monkeypatch.setattr(sweep,'offer_update',lambda *a:None)
    Window.created.clear()
    peer=SimpleNamespace(video_codec='h264', clock=clock, send_input=lambda *a:None)
    stopped=threading.Event()
    result=sweep.run_h264_capacity_sweep(peer,'T4',queue.Queue(),queue.Queue(),
                                         stopped,deadline=clock.now+100,
                                         poll_hz=hz,phase_seconds=6,warmup_seconds=1)
    assert stopped.is_set()
    assert result['requested_poll_hz']==hz
    assert result['mode']=='single-owned-comm-h264-capacity'
    assert [p['credits'] for p in result['phases']]==[1,2,3,4,8,16,32]
    assert all(p['max_outstanding']<=p['credits'] for p in result['phases'])
    assert all(p['decoded_frames']>0 for p in result['phases'])
    assert len(Window.created)==1
    window=Window.created[0]
    assert window.outstanding==0
    assert window.sends==list(range(1,len(window.sends)+1))
    assert decoder.seqs==list(range(len(decoder.seqs)))
    assert result['all_phase_boundaries_drained']
    assert result['maximum_window_outstanding']<=32


@pytest.mark.parametrize('hz', [0,12,29,31,61,True,30.0])
def test_high_rate_sweep_rejects_wrong_poll_frequency_without_side_effect(hz):
    peer=SimpleNamespace(video_codec='h264')
    with pytest.raises(ValueError):
        sweep.run_h264_capacity_sweep(peer,'T4',queue.Queue(),queue.Queue(),
                                      threading.Event(),deadline=1000,poll_hz=hz)
