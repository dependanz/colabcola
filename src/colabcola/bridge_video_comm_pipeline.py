"""v33i: bounded two-credit H.264 Comm receiver and optional host worker.

No GPU allocation, new WebSocket, or kernel bootstrap. The existing run owner
still creates/releases a single authenticated Jupyter Comm. Replies may arrive
out of order but *only* ordered access units enter a stateful H.264 decoder.
This module is opt-in and does not change the established single-credit mode.
"""
from __future__ import annotations

import math
import queue
from statistics import median
import time

from .bridge_colab import offer_update
from .bridge_telemetry import BridgeTransferMeter
from .bridge_video_comm_protocol import (
    MAX_PULL_SEQ, VideoCommProtocolError, VideoSequenceGate,
    decode_video_reply,
)
from .bridge_video_comm_session import (
    WIDTH, HEIGHT, MAX_NO_FRESH_SECONDS, application_video_rates,
    local_elapsed_us, make_host_decoder,
)
from .bridge_video_stream import PACKET_HEADER
from .lifecycle_io import LifecycleError

MAX_VIDEO_CREDITS = 2
MAX_EXPERIMENTAL_VIDEO_CREDITS = 32
PULL_TIMEOUT_S = 8.0


def validated_remote_video_diagnostics(payload: object) -> dict | None:
    """Optional same-remote-clock timings and CUDA source ID.

    Older test peers may omit the fields. Invalid optional metadata is ignored,
    never treated as frame content, protocol authorization or a host timestamp.
    """
    if not isinstance(payload, dict):
        return None
    seq = payload.get("renderer_seq")
    read_us = payload.get("source_read_us")
    encode_us = payload.get("encode_us")
    if (type(seq) is not int or not 0 <= seq < 100_000_000
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   or not 0 <= value <= 10_000_000
                   for value in (read_us, encode_us))):
        return None
    return {"renderer_seq": seq, "source_read_us": float(read_us),
            "encode_us": float(encode_us)}


class KernelH264PullWindow:
    """Two outstanding requests, ordered commit, never replay after uncertainty.

    `poll` returns committed tuples (unit or None, ack, size tuple, RTT-ms).
    None is a dependency-suppressed frame, not a missing reply. Empty list
    means no newly committable reply; it is not a failure. The ordered gate and
    ACK state advance only at commit, not when an out-of-order reply arrives.
    """

    def __init__(self, peer, *, credits: int = MAX_VIDEO_CREDITS,
                 experimental: bool = False):
        allowed = (type(experimental) is bool and type(credits) is int and
                   ((1 <= credits <= MAX_EXPERIMENTAL_VIDEO_CREDITS)
                    if experimental else credits == MAX_VIDEO_CREDITS))
        if (not allowed
                or not isinstance(getattr(peer, 'run_id', None), str)
                or len(peer.run_id) != 32
                or not isinstance(getattr(peer, 'comm_id', None), str)
                or not peer.comm_id or getattr(peer, 'wire', None) is None):
            raise ValueError('bounded H.264 receiver requires an attached owned Comm')
        self.peer = peer
        self.credits = credits
        self.pending: dict[int, float] = {}
        self.completed: dict[int, tuple] = {}
        self.last_sent = 0
        self.next_commit = 1
        self.input_ack = 0
        self.gate = VideoSequenceGate()
        self.suppressed_dependent_units = 0
        self.failed = False
        self.maximum_outstanding = 0
        # Same order as the last nonempty poll() return; not a wire/codec change.
        self.last_committed_diagnostics: list[dict | None] = []

    def _healthy(self):
        wire = self.peer.wire
        if self.failed or wire is None or not wire.connection_ready.is_set():
            self.failed = True
            raise LifecycleError('kernel H.264 pipeline disconnected or failed')

    @property
    def outstanding(self) -> int:
        """Pending plus decoded but not yet committed replies."""
        return len(self.pending) + len(self.completed)

    def submit(self, pull_seq: int) -> None:
        self._healthy()
        if (type(pull_seq) is not int or not self.last_sent < pull_seq <= MAX_PULL_SEQ
                or pull_seq != self.last_sent + 1
                or self.outstanding >= self.credits):
            raise LifecycleError('video pipeline credit or pull sequence invalid')
        self.last_sent = pull_seq
        # Mark outstanding before any uncertain transport action.
        self.pending[pull_seq] = time.monotonic()
        self.maximum_outstanding = max(self.maximum_outstanding, self.outstanding)
        try:
            self.peer._send('video_pull', pull_seq=pull_seq)
        except Exception:
            self.failed = True
            raise LifecycleError('video pipeline request submission failed') from None

    def check_timeouts(self, *, now: float | None = None):
        self._healthy()
        t = time.monotonic() if now is None else now
        # An out-of-order reply can be buffered while an earlier reply is
        # missing; both use their original send time as the deadline anchor.
        if any(t - started >= PULL_TIMEOUT_S for started in self.pending.values()):
            self.failed = True
            raise LifecycleError('video pipeline reply exceeded bounded deadline')

    def _commit_ready(self) -> list[tuple]:
        result = []
        try:
            while self.next_commit in self.completed:
                unit, ack, sizes, rtt, diagnostics = self.completed.pop(self.next_commit)
                if ack < self.input_ack:
                    raise VideoCommProtocolError('video pipeline applied ACK regressed')
                keep = self.gate.accept(unit)
                self.input_ack = ack
                self.next_commit += 1
                if not keep:
                    self.suppressed_dependent_units += 1
                result.append((unit if keep else None, ack, sizes, rtt))
                self.last_committed_diagnostics.append(diagnostics)
        except VideoCommProtocolError:
            self.failed = True
            raise LifecycleError('video pipeline ordered frame invalid') from None
        return result

    def poll(self, seconds: float = .03) -> list[tuple]:
        if (type(seconds) not in (float, int) or isinstance(seconds, bool)
                or not .001 <= seconds <= .5):
            raise ValueError('video pipeline poll timeout outside bounds')
        self.last_committed_diagnostics = []
        self.check_timeouts()
        try:
            message = self.peer.wire.iopub_channel.get_msg(timeout=seconds)
        except queue.Empty:
            return []
        except Exception:
            self.failed = True
            raise LifecycleError('video pipeline receive failed') from None
        if not isinstance(message, dict):
            return []
        header = message.get('header')
        if not isinstance(header, dict) or header.get('msg_type') != 'comm_msg':
            return []
        content = message.get('content')
        if not isinstance(content, dict) or content.get('comm_id') != self.peer.comm_id:
            return []
        payload = content.get('data')
        if not isinstance(payload, dict) or payload.get('v') != 1 or payload.get('run_id') != self.peer.run_id:
            return []
        if payload.get('kind') == 'error':
            self.failed = True
            raise LifecycleError('remote H.264 pipeline reported an error')
        if payload.get('kind') != 'video_access_unit':
            return []
        pull_seq = payload.get('pull_seq')
        if type(pull_seq) is not int or pull_seq not in self.pending:
            self.failed = True
            raise LifecycleError('unsolicited or duplicate video pipeline reply')
        sent_at = self.pending.pop(pull_seq)
        try:
            unit, ack = decode_video_reply(payload, run_id=self.peer.run_id,
                                          expected_pull_seq=pull_seq)
        except VideoCommProtocolError:
            self.failed = True
            raise LifecycleError('invalid correlated H.264 pipeline reply') from None
        sizes = (len(unit.data), PACKET_HEADER.size + len(unit.data), len(payload['data']))
        diagnostics = validated_remote_video_diagnostics(payload)
        self.completed[pull_seq] = (
            unit, ack, sizes, (time.monotonic() - sent_at) * 1000, diagnostics)
        return self._commit_ready()


def stream_comm_h264_pipeline(peer, gpu: str, outgoing: queue.Queue,
                              frames: queue.Queue, stopped, deadline: float,
                              poll_hz: int, *, credits: int = MAX_VIDEO_CREDITS):
    """Explicit two-credit H.264 host worker; default video worker is untouched.

    Renderer and host timestamps are never subtracted; request RTT and
    decode/GUI durations are measured only on the host clock.
    """
    if (type(poll_hz) is not int or not 1 <= poll_hz <= 12
            or type(credits) is not int or credits != MAX_VIDEO_CREDITS):
        raise ValueError('invalid H.264 pipeline pacing or credit count')
    if getattr(peer, 'video_codec', None) != 'h264':
        raise LifecycleError('video pipeline requires explicitly opted-in H.264 Comm')
    av, decoder = make_host_decoder()
    window = KernelH264PullWindow(peer, credits=credits)
    meter = BridgeTransferMeter()
    prefix = b'P6\n640 360\n255\n'
    next_due = time.monotonic()
    last_fresh = last_report = next_due
    next_pull = 1
    input_seq = 0
    replies = decoded_count = h264_bytes = ccv1_bytes = base64_chars = 0
    rtts_ms, decode_us = [], []
    print('[colabcola] Opt-in H.264 Comm pipeline connected (2 credits)', flush=True)
    offer_update(frames, ('status', f'Kernel Comm H.264 · {gpu} · pipeline 2'))
    try:
        while not stopped.is_set() and time.monotonic() < deadline:
            latest = None
            while True:
                try:
                    latest = outgoing.get_nowait()
                except queue.Empty:
                    break
            if latest is not None:
                input_seq += 1
                meter.sent(input_seq, time.monotonic())
                peer.send_input(input_seq, latest)
            now = time.monotonic()
            if now >= next_due and window.outstanding < credits:
                window.submit(next_pull)
                next_pull += 1
                next_due = now + 1 / poll_hz  # never catch up or burst
            # Do not advance the decoder on IOPub arrival order. Window commits
            # only contiguous requests after checking run/comm/pull identity.
            if window.outstanding:
                ready = window.poll(.03)
            else:
                ready = []
                time.sleep(min(.02, max(0, next_due - time.monotonic())))
            for unit, ack, sizes, rtt in ready:
                replies += 1
                meter.received_ack(ack, time.monotonic())
                raw, framed, b64 = sizes
                h264_bytes += raw
                ccv1_bytes += framed
                base64_chars += b64
                rtts_ms.append(rtt)
                if unit is None:
                    continue
                decode_started_ns = time.perf_counter_ns()
                try:
                    packet = av.Packet(unit.data)
                    packet.pts = unit.seq
                    images = decoder.decode(packet)
                    if len(images) > 1:
                        raise LifecycleError('H.264 pipeline decoder emitted extra frames')
                    for image in images:
                        if image.width != WIDTH or image.height != HEIGHT:
                            raise LifecycleError('H.264 pipeline decoded frame dimensions invalid')
                        pixels = image.to_ndarray(format='rgb24').tobytes()
                        if len(pixels) != WIDTH * HEIGHT * 3:
                            raise LifecycleError('H.264 pipeline decoded RGB size invalid')
                        offer_update(frames, ('frame', prefix + pixels))
                        decoded_count += 1
                        last_fresh = time.monotonic()
                    decode_us.append(local_elapsed_us(decode_started_ns, time.perf_counter_ns()))
                except LifecycleError:
                    raise
                except Exception:
                    raise LifecycleError('H.264 pipeline local decode failed') from None
            now = time.monotonic()
            window.check_timeouts(now=now)
            if now - last_fresh > MAX_NO_FRESH_SECONDS:
                raise LifecycleError('H.264 pipeline produced no fresh decoded image')
            if now - last_report >= 2:
                elapsed = max(.001, now - last_report)
                summary = (f'{gpu} · H.264 pipeline {decoded_count / elapsed:.1f} decoded FPS'
                           f' · {application_video_rates(h264_bytes=h264_bytes, ccv1_bytes=ccv1_bytes, base64_chars=base64_chars, elapsed=elapsed)}'
                           f' · replies {replies} · skipped dependent {window.suppressed_dependent_units}'
                           f' · outstanding {window.outstanding}/2')
                if rtts_ms:
                    summary += f' · video pull RTT p50 {median(rtts_ms):.0f} ms'
                if decode_us:
                    summary += f' · local decode+RGB p50 {median(decode_us):.1f} µs'
                stats = meter.flush()
                if stats['ack'] is not None:
                    summary += f" · input ACK observed p50 {stats['ack']} ms"
                print('[colabcola] ' + summary, flush=True)
                offer_update(frames, ('status', summary))
                decoded_count = replies = h264_bytes = ccv1_bytes = base64_chars = 0
                rtts_ms.clear()
                decode_us.clear()
                last_report = now
        if time.monotonic() >= deadline:
            print('[colabcola] H.264 pipeline time budget reached', flush=True)
    except Exception as exc:
        print('[colabcola] H.264 pipeline stopped: ' + type(exc).__name__, flush=True)
        raise LifecycleError('kernel H.264 pipeline interrupted') from None
    finally:
        stopped.set()
