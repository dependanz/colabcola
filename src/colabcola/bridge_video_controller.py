"""v25: causal multi-gap H.264 IDR feedback over a deterministic virtual link.

An incrementally produced REAL libx264 stream feeds a bounded, virtual-time
FIFO link. A receiver sees only packets that have already arrived, detects
missing sequence IDs, and sends delayed, rate-limited IDR requests. The
encoder restarts only at a SOURCE frame strictly after feedback arrives.

This is a local/offline multi-request state machine, NOT a real UDP/WebRTC
sender, bandwidth controller, packet-level simulator, GUI or Colab service.
It deliberately retains fixed encoder CRF and fixed virtual playout buffers.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
import random

from .bridge_video_impairment import (
    LinkConfig, LinkTrace, ScheduledUnit, MAX_WIRE_BYTES, _serialize_ms,
    decode_delivered_h264, make_h264_packets, parse_bandwidth_steps,
    parse_drop_seqs, simulate_link,
)
from .bridge_video_feedback import (
    _new_encoder, _parse_buffer_list, _summarize_trace, _validate_video,
)
from .bridge_video_stream import VideoAccessUnit, packet_bytes, is_keyframe


@dataclass(frozen=True)
class ControllerConfig:
    feedback_ms: float = 20.0
    retry_ms: float = 100.0
    max_requests: int = 8

    def __post_init__(self):
        for name, value in (
            ("feedback_ms", self.feedback_ms),
            ("retry_ms", self.retry_ms),
        ):
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not 0 <= value <= 2000):
                raise ValueError(f"invalid {name}")
        if type(self.max_requests) is not int or not 0 <= self.max_requests <= 32:
            raise ValueError("max_requests must be an integer in 0..32")


class VirtualAccessUnitLink:
    """Incremental equivalent of v23's FIFO access-unit link scheduler.

    An erased/stale admitted unit still occupies an in-flight slot and
    consumes bandwidth until its simulated arrival. Admission rejection
    doesn't consume bandwidth. Receiver observations are delayed until
    virtual arrival; the controller cannot inspect future drop decisions.
    """

    def __init__(self, config: LinkConfig):
        if not isinstance(config, LinkConfig):
            raise ValueError("invalid link configuration")
        self.config = config
        self.rng = random.Random(config.seed)
        self.inflight: deque[tuple[ScheduledUnit, str | None]] = deque()
        self.delivered: list[ScheduledUnit] = []
        self.dropped: list[tuple[int, str]] = []
        self.link_free_ms = 0.0
        self.last_arrival_ms = 0.0
        self.wire_sent_bytes = 0
        self.peak = 0
        self._last_dispatch_ms = 0.0

    def arrive_by(self, time_ms: float) -> list[ScheduledUnit]:
        """Advance the receiver's observable frontier, in arrival order."""
        observed = []
        while self.inflight and self.inflight[0][0].arrival_ms <= time_ms:
            item, reason = self.inflight.popleft()
            if reason is None:
                self.delivered.append(item)
                observed.append(item)
        return observed

    def send(self, packet: VideoAccessUnit, *, fps: int,
             dispatch_ms: float | None = None) -> None:
        """Attempt admission at dispatch time, retaining original capture age.

        Omitting dispatch_ms preserves the original capture-time scheduler.
        A local sender may delay dispatch, but it cannot move an admitted
        unit or its serialization backward or reset the source timestamp.
        """
        if type(fps) is not int or not 1 <= fps <= 60:
            raise ValueError("invalid source rate")
        if not isinstance(packet, VideoAccessUnit):
            raise ValueError("invalid video packet")
        source_ms = packet.seq * 1000.0 / fps
        at_ms = source_ms if dispatch_ms is None else dispatch_ms
        if (type(at_ms) not in (int, float) or not math.isfinite(at_ms)
                or at_ms < source_ms or at_ms < self._last_dispatch_ms):
            raise ValueError("dispatch time must be finite, monotonic and after capture")
        if self.inflight and self.inflight[0][0].arrival_ms <= at_ms:
            raise RuntimeError("receiver arrivals must be drained before send")
        wire_size = len(packet_bytes(packet))
        if len(self.inflight) >= self.config.max_inflight:
            self._last_dispatch_ms = at_ms
            self.dropped.append((packet.seq, "queue_full"))
            return
        if self.wire_sent_bytes + wire_size > MAX_WIRE_BYTES:
            raise ValueError("virtual link wire budget exceeded")
        self._last_dispatch_ms = at_ms
        started = max(at_ms, self.link_free_ms)
        finished = _serialize_ms(started, wire_size * 8, self.config)
        self.link_free_ms = finished
        jitter = (self.rng.uniform(-self.config.jitter_ms, self.config.jitter_ms)
                  if self.config.jitter_ms else 0.0)
        arrival = max(finished, finished + self.config.delay_ms + jitter,
                      self.last_arrival_ms)
        self.last_arrival_ms = arrival
        erased = packet.seq in self.config.drop_seqs or (
            self.config.loss_percent > 0
            and self.rng.random() * 100 < self.config.loss_percent
        )
        reason = ("erased" if erased else
                  "stale" if self.config.max_age_ms
                  and arrival - source_ms > self.config.max_age_ms else None)
        item = ScheduledUnit(packet, source_ms, arrival, wire_size)
        self.inflight.append((item, reason))
        self.peak = max(self.peak, len(self.inflight))
        self.wire_sent_bytes += wire_size
        if reason is not None:
            self.dropped.append((packet.seq, reason))

    def snapshot(self, frames: int) -> LinkTrace:
        if self.inflight:
            raise RuntimeError("cannot finalize an undrained virtual link")
        if len(self.delivered) + len(self.dropped) != frames:
            raise RuntimeError("virtual link frame accounting mismatch")
        return LinkTrace(frames, tuple(self.delivered), tuple(self.dropped),
                         self.wire_sent_bytes, self.peak)


class MultiGapFeedbackController:
    """Receiver-side request logic and causal control-message delivery."""

    def __init__(self, config: ControllerConfig):
        if not isinstance(config, ControllerConfig):
            raise ValueError("invalid feedback controller configuration")
        self.config = config
        self.expected_seq = 0
        self.synced = False
        self.last_request_ms: float | None = None
        self.pending: deque[dict] = deque()
        self.requests: list[dict] = []
        self.applied: list[dict] = []
        self.gaps: list[dict] = []
        self.last_observed_ms = -1.0

    def receive(self, packet: VideoAccessUnit, at_ms: float) -> None:
        if (not isinstance(packet, VideoAccessUnit)
                or type(at_ms) not in (int, float)
                or not math.isfinite(at_ms)
                or at_ms < self.last_observed_ms
                or packet.seq < self.expected_seq):
            raise ValueError("noncausal receiver frame observation")
        previous = self.expected_seq
        gap = packet.seq > previous
        if gap:
            self.gaps.append({
                "first_missing_seq": previous,
                "next_arrived_seq": packet.seq,
                "detected_at_ms": round(at_ms, 3),
            })
            self.synced = False
        self.expected_seq = packet.seq + 1
        self.last_observed_ms = at_ms

        if packet.keyframe and is_keyframe(packet.data):
            self.synced = True
            return
        if self.synced or len(self.requests) >= self.config.max_requests:
            return
        if (self.last_request_ms is not None
                and at_ms - self.last_request_ms < self.config.retry_ms):
            return
        request = {
            "request_id": len(self.requests),
            "receiver_observed_seq": packet.seq,
            "detected_at_ms": at_ms,
            "reaches_source_ms": at_ms + self.config.feedback_ms,
        }
        self.requests.append(request)
        self.pending.append(request)
        self.last_request_ms = at_ms

    def requests_due_before(self, source_time_ms: float) -> list[dict]:
        """Only messages delivered STRICTLY before source-frame creation."""
        due = []
        while self.pending and self.pending[0]["reaches_source_ms"] < source_time_ms - 1e-9:
            due.append(self.pending.popleft())
        return due

    def record_applied(self, due: list[dict], source_seq: int) -> None:
        if not due:
            return
        if type(source_seq) is not int or not 0 <= source_seq < 120:
            raise ValueError("invalid source IDR index")
        self.applied.append({
            "source_idr_seq": source_seq,
            "request_ids": [q["request_id"] for q in due],
            "received_control_at_ms": round(
                max(q["reaches_source_ms"] for q in due), 3),
        })

    def report(self):
        return {
            "feedback_ms": self.config.feedback_ms,
            "retry_ms": self.config.retry_ms,
            "max_requests": self.config.max_requests,
            "gaps_detected": self.gaps,
            "requests_sent": len(self.requests),
            "requests": [
                {**r, "detected_at_ms": round(r["detected_at_ms"], 3),
                 "reaches_source_ms": round(r["reaches_source_ms"], 3)}
                for r in self.requests
            ],
            "idr_restarts_applied": len(self.applied),
            "applied": self.applied,
            "requests_not_applied_before_source_end":
                len(self.requests) - sum(len(x["request_ids"]) for x in self.applied),
        }


def simulate_multi_feedback(*, width: int, height: int, fps: int,
                            frames: int, gop: int, link: LinkConfig,
                            controller: ControllerConfig) -> tuple[list[VideoAccessUnit], LinkTrace, dict]:
    """Encode and schedule frames in increasing SOURCE time; receive only past.

    In particular, a request cannot affect packets already encoded
    or packets already on the forward link. A missed final frame may
    never be detected if no later frame arrives.
    """
    _validate_video(width, height, fps, frames, gop)
    if not isinstance(link, LinkConfig):
        raise ValueError("invalid virtual media link")
    if not isinstance(controller, ControllerConfig):
        raise ValueError("invalid controller")

    try:
        import av
        import numpy as np
    except ImportError:
        raise RuntimeError("optional PyAV and NumPy required for multi-feedback test") from None
    from .bridge_video_live import LiveScene, live_background, render_live_rgb

    base = live_background(width, height, np)
    media = VirtualAccessUnitLink(link)
    receiver = MultiGapFeedbackController(controller)
    encoder = _new_encoder(av, width, height, fps, gop)
    segment_start = 0
    packets: list[VideoAccessUnit] = []
    for seq in range(frames):
        source_ms = seq * 1000.0 / fps
        for arrived in media.arrive_by(source_ms):
            receiver.receive(arrived.packet, arrived.arrival_ms)
        due = receiver.requests_due_before(source_ms)
        if due:
            if encoder.encode(None):
                raise RuntimeError("unexpected buffered H.264 frames on IDR restart")
            encoder = _new_encoder(av, width, height, fps, gop)
            segment_start = seq
            receiver.record_applied(due, seq)

        phase = seq / fps
        scene = LiveScene(
            x=.5 + .27 * math.sin(phase * 2),
            y=.5 + .22 * math.cos(phase * 3),
        )
        frame = av.VideoFrame.from_ndarray(render_live_rgb(scene, base, seq),
                                           format="rgb24")
        frame.pts = seq - segment_start
        encoded = encoder.encode(frame)
        if len(encoded) != 1:
            raise RuntimeError("expected one H.264 packet per video frame")
        compressed = bytes(encoded[0])
        packet = VideoAccessUnit(
            seq=seq, pts_ns=round(seq * 1_000_000_000 / fps),
            capture_ns=round(seq * 1_000_000_000 / fps),
            data=compressed,
            keyframe=bool(encoded[0].is_keyframe and is_keyframe(compressed)),
        )
        packets.append(packet)
        media.send(packet, fps=fps)

    if encoder.encode(None):
        raise RuntimeError("unexpected buffered H.264 frames at stream end")
    for arrived in media.arrive_by(math.inf):
        receiver.receive(arrived.packet, arrived.arrival_ms)
    trace = media.snapshot(frames)
    if not packets[0].keyframe:
        raise RuntimeError("H.264 source failed to begin at IDR")
    for applied in receiver.applied:
        if not packets[applied["source_idr_seq"]].keyframe:
            raise RuntimeError("feedback-triggered H.264 IDR not generated")
    return packets, trace, receiver.report()


def compare_multi_feedback(*, width: int = 128, height: int = 72,
                           fps: int = 30, frames: int = 60, gop: int = 24,
                           link: LinkConfig | None = None,
                           controller: ControllerConfig | None = None,
                           buffers_ms: tuple[int, ...] = (40, 80, 120)) -> dict:
    """Compare baseline periodic H.264 to a causal multi-IDR control loop."""
    _validate_video(width, height, fps, frames, gop)
    link = LinkConfig() if link is None else link
    controller = ControllerConfig() if controller is None else controller
    if not isinstance(link, LinkConfig) or not isinstance(controller, ControllerConfig):
        raise ValueError("invalid experiment configuration")
    if (type(buffers_ms) is not tuple or not 1 <= len(buffers_ms) <= 8
            or any(type(x) is not int or not 0 <= x <= 1000 for x in buffers_ms)
            or tuple(sorted(set(buffers_ms))) != buffers_ms):
        raise ValueError("buffer depths must be sorted unique integers in 0..1000")

    baseline_packets = make_h264_packets(width=width, height=height,
                                         fps=fps, frames=frames, gop=gop)
    baseline_trace = simulate_link(baseline_packets, fps, link)
    baseline_decoded = decode_delivered_h264(baseline_trace, width, height)
    adaptive_packets, adaptive_trace, controller_log = simulate_multi_feedback(
        width=width, height=height, fps=fps, frames=frames, gop=gop,
        link=link, controller=controller,
    )
    adaptive_decoded = decode_delivered_h264(adaptive_trace, width, height)
    applied = controller_log["applied"]
    boundary = applied[0]["source_idr_seq"] if applied else frames
    if adaptive_packets[:boundary] != baseline_packets[:boundary]:
        raise RuntimeError("closed-loop control altered frames before first applied feedback")
    if not applied and adaptive_trace != baseline_trace:
        raise RuntimeError("uncontrolled link trace differs from periodic baseline")

    return {
        "schema": 1,
        "mode": "causal-multi-feedback-idr",
        "source": {
            "width": width, "height": height, "fps": fps,
            "frames": frames, "gop": gop,
        },
        "link": {
            "delay_ms": link.delay_ms, "jitter_ms": link.jitter_ms,
            "bandwidth_kbps": link.bandwidth_kbps,
            "bandwidth_steps": [list(s) for s in link.bandwidth_steps],
            "loss_percent": link.loss_percent,
            "drop_seqs": list(link.drop_seqs),
            "max_inflight": link.max_inflight,
            "max_age_ms": link.max_age_ms,
            "seed": link.seed,
        },
        "controller": controller_log,
        "periodic": _summarize_trace(
            baseline_trace, baseline_decoded, baseline_packets, fps, buffers_ms),
        "multi_feedback": _summarize_trace(
            adaptive_trace, adaptive_decoded, adaptive_packets, fps, buffers_ms),
        "scope": (
            "locally encoded/decoded H.264; deterministic virtual-time media "
            "and feedback links; multiple IDR restarts only, no bitrate or "
            "adaptive playout control, real network, Colab, GPU or GUI"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--gop", type=int, default=24)
    parser.add_argument("--delay-ms", type=float, default=40)
    parser.add_argument("--jitter-ms", type=float, default=0)
    parser.add_argument("--bandwidth-kbps", type=int, default=0)
    parser.add_argument("--bandwidth-steps", default="")
    parser.add_argument("--loss-percent", type=float, default=0)
    parser.add_argument("--drop-seqs", default="5,27")
    parser.add_argument("--max-inflight", type=int, default=16)
    parser.add_argument("--max-age-ms", type=float, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--feedback-ms", type=float, default=20)
    parser.add_argument("--retry-ms", type=float, default=100)
    parser.add_argument("--max-requests", type=int, default=8)
    parser.add_argument("--buffers-ms", default="40,80,120")
    args = parser.parse_args(argv)
    try:
        from .bridge_video_feedback import _parse_buffer_list
        config = LinkConfig(
            delay_ms=args.delay_ms, jitter_ms=args.jitter_ms,
            bandwidth_kbps=args.bandwidth_kbps,
            bandwidth_steps=parse_bandwidth_steps(args.bandwidth_steps),
            loss_percent=args.loss_percent, drop_seqs=parse_drop_seqs(args.drop_seqs),
            max_inflight=args.max_inflight, max_age_ms=args.max_age_ms,
            seed=args.seed,
        )
        control = ControllerConfig(
            feedback_ms=args.feedback_ms, retry_ms=args.retry_ms,
            max_requests=args.max_requests,
        )
        result = compare_multi_feedback(
            width=args.width, height=args.height,
            fps=args.fps, frames=args.frames, gop=args.gop,
            link=config, controller=control,
            buffers_ms=_parse_buffer_list(args.buffers_ms),
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
