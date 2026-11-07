"""v26/v27: causal adaptive H.264 encoding-rate experiments on a virtual link.

Each frame is encoded by real libx264. The sender changes the encoder's
target bitrate only AFTER receiving a delayed report derived from media
packets actually observed by the receiver. Any rate switch starts a new
closed-GOP IDR segment (intentionally modeled as an encoder restart).
An opt-in receiver silence timer can also request the low rate without a
new arrival. It knows only elapsed receiver time, never future link state.

The link clock, feedback path, queue limits, jitter and losses are VIRTUAL.
There is no real socket, GPU, Colab runtime, production media protocol,
adaptive receiver clock, or input-to-photon measurement.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import json
import math

from .bridge_video_controller import VirtualAccessUnitLink
from .bridge_video_feedback import _parse_buffer_list, _summarize_trace, _validate_video
from .bridge_video_impairment import (
    LinkConfig, LinkTrace, ScheduledUnit, _make_decoder, decode_delivered_h264,
    parse_bandwidth_steps, parse_drop_seqs,
)
from .bridge_video_stream import VideoAccessUnit, is_keyframe, MAX_PACKET


@dataclass(frozen=True)
class RateConfig:
    high_kbps: int = 650
    low_kbps: int = 95
    congested_age_ms: float = 110.0
    recovered_age_ms: float = 65.0
    recovery_samples: int = 8
    congestion_samples: int = 3
    control_delay_ms: float = 35.0
    min_switch_ms: float = 350.0
    max_switches: int = 8
    silence_timeout_ms: float = 0.0  # disabled preserves the v26 policy

    def __post_init__(self) -> None:
        for name, value, lo, hi in (
            ("high_kbps", self.high_kbps, 50, 20_000),
            ("low_kbps", self.low_kbps, 20, 20_000),
            ("recovery_samples", self.recovery_samples, 1, 30),
            ("congestion_samples", self.congestion_samples, 1, 30),
            ("max_switches", self.max_switches, 0, 20),
        ):
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"invalid {name}")
        if self.low_kbps >= self.high_kbps:
            raise ValueError("low_kbps must be smaller than high_kbps")
        for name, value, lo, hi in (
            ("congested_age_ms", self.congested_age_ms, 1, 2000),
            ("recovered_age_ms", self.recovered_age_ms, 0, 2000),
            ("control_delay_ms", self.control_delay_ms, 0, 1000),
            ("min_switch_ms", self.min_switch_ms, 0, 5000),
            ("silence_timeout_ms", self.silence_timeout_ms, 0, 5000),
        ):
            if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"invalid {name}")
        if self.recovered_age_ms >= self.congested_age_ms:
            raise ValueError("recovery age must be below congestion age")


class RateReceiver:
    """Receiver-side hysteretic high/low rate requests from observed arrivals.

    Observations see only delivered units. The optional silence timer is
    armed by the first arrival, and advances independently of later media
    arrivals. Requests use an ideal fixed-delay reverse path; forward
    media silence is not evidence of reverse-path congestion.
    """

    def __init__(self, settings: RateConfig):
        if not isinstance(settings, RateConfig):
            raise ValueError("invalid rate settings")
        self.settings = settings
        self.expected_seq = 0
        self.last_arrival_ms = -1.0
        self.desired_kbps = settings.high_kbps
        self.last_request_ms = -math.inf
        self.low_age_streak = 0
        self.high_age_streak = 0
        self.pending: deque[dict] = deque()
        self.requests: list[dict] = []
        self.applied: list[dict] = []
        self.observations = 0
        self.gaps = 0
        self.clock_ms = 0.0
        self.silence_expired = False
        self.silence_events: list[dict] = []
        self.watchdog_stopped = False

    def _request(self, target: int, at_ms: float, *, trigger: str,
                 seq: int, age_ms: float | None = None,
                 gap: bool = False, silence_ms: float | None = None) -> None:
        if (len(self.requests) >= self.settings.max_switches
                or at_ms < self.last_request_ms + self.settings.min_switch_ms):
            return
        request = {
            "request_id": len(self.requests),
            "trigger": trigger,
            "observed_seq": seq,
            "observed_age_ms": round(age_ms, 3) if age_ms is not None else None,
            "gap_detected": gap,
            "generated_at_ms": round(at_ms, 3),
            "reaches_source_ms": at_ms + self.settings.control_delay_ms,
            "target_kbps": target,
        }
        if silence_ms is not None:
            request["silence_ms"] = round(silence_ms, 3)
            request["last_arrival_ms"] = round(self.last_arrival_ms, 3)
        self.requests.append(request)
        self.pending.append(request)
        self.desired_kbps = target
        self.low_age_streak = 0
        self.high_age_streak = 0
        self.last_request_ms = at_ms

    def _advance(self, now_ms: float, *, include_deadline: bool) -> None:
        if (type(now_ms) not in (int, float) or not math.isfinite(now_ms)
                or now_ms < self.clock_ms):
            raise ValueError("receiver time must be finite and monotonic")

        def due(deadline: float) -> bool:
            return deadline <= now_ms if include_deadline else deadline < now_ms

        timeout = self.settings.silence_timeout_ms
        if timeout and self.last_arrival_ms >= 0 and not self.watchdog_stopped:
            deadline = self.last_arrival_ms + timeout
            if not self.silence_expired and due(deadline):
                self.silence_expired = True
                # Samples separated by a media blackout do not demonstrate
                # continuous recovery, even if the source is already LOW.
                self.low_age_streak = 0
                self.high_age_streak = 0
                self.silence_events.append({
                    "observed_seq": self.expected_seq - 1,
                    "last_arrival_ms": round(self.last_arrival_ms, 3),
                    "timeout_at_ms": round(deadline, 3),
                })
            eligible = max(deadline, self.last_request_ms + self.settings.min_switch_ms)
            if (self.silence_expired and due(eligible)
                    and self.desired_kbps == self.settings.high_kbps):
                self._request(
                    self.settings.low_kbps, eligible, trigger="receiver_silence",
                    seq=self.expected_seq - 1,
                    silence_ms=eligible - self.last_arrival_ms,
                )
        self.clock_ms = now_ms

    def advance(self, now_ms: float) -> None:
        """Process timers through now, including a deadline exactly at now.

        Call observe for all arrivals through now first. observe itself
        processes timers strictly before its arrival, so a packet arriving
        exactly at the timeout wins that tie and resets the watchdog.
        """
        self._advance(now_ms, include_deadline=True)

    def stop_watchdog(self) -> None:
        """End timer observation with the source; tail arrivals still drain."""
        self.watchdog_stopped = True

    def observe(self, item: ScheduledUnit) -> None:
        if (not isinstance(item, ScheduledUnit)
                or not isinstance(item.packet, VideoAccessUnit)
                or type(item.arrival_ms) not in (int, float)
                or type(item.source_ms) not in (int, float)
                or not math.isfinite(item.arrival_ms)
                or not math.isfinite(item.source_ms)
                or item.source_ms < 0
                or item.arrival_ms < item.source_ms
                or item.arrival_ms < self.last_arrival_ms
                or item.arrival_ms < self.clock_ms
                or item.packet.seq < self.expected_seq):
            raise ValueError("noncausal encoded-frame observation")
        self._advance(item.arrival_ms, include_deadline=False)
        seq = item.packet.seq
        gap = seq > self.expected_seq
        if gap:
            self.gaps += 1
        self.expected_seq = seq + 1
        self.last_arrival_ms = item.arrival_ms
        self.silence_expired = False
        self.observations += 1
        age = item.arrival_ms - item.source_ms
        congested = gap or age >= self.settings.congested_age_ms
        self.high_age_streak = self.high_age_streak + 1 if congested else 0
        if age <= self.settings.recovered_age_ms and not gap:
            self.low_age_streak += 1
        else:
            self.low_age_streak = 0

        target = None
        if (self.desired_kbps == self.settings.high_kbps
                and (gap or self.high_age_streak >= self.settings.congestion_samples)):
            target = self.settings.low_kbps
        elif (self.desired_kbps == self.settings.low_kbps
              and self.low_age_streak >= self.settings.recovery_samples):
            target = self.settings.high_kbps
        if target is not None:
            self._request(
                target, item.arrival_ms,
                trigger=("sequence_gap" if gap else "packet_age"
                         if target == self.settings.low_kbps else "recovery"),
                seq=seq, age_ms=age, gap=gap,
            )

    def apply_due(self, source_seq: int, source_ms: float,
                  current_kbps: int) -> int:
        """Only apply feedback strictly delivered before the source frame."""
        due = []
        while self.pending and self.pending[0]["reaches_source_ms"] < source_ms - 1e-9:
            due.append(self.pending.popleft())
        if not due:
            return current_kbps
        # Multiple queued commands are compressed into the newest one;
        # no historical rate switch is applied retroactively.
        latest = due[-1]
        target = latest["target_kbps"]
        if target != current_kbps:
            self.applied.append({
                "source_seq": source_seq,
                "source_ms": round(source_ms, 3),
                "previous_kbps": current_kbps,
                "target_kbps": target,
                "request_ids_coalesced": [r["request_id"] for r in due],
            })
        return target

    def report(self) -> dict:
        return {
            "packets_observed": self.observations,
            "observed_sequence_gaps": self.gaps,
            "requests": self.requests,
            "requests_sent": len(self.requests),
            "switches_applied": len(self.applied),
            "applied": self.applied,
            "unapplied_commands": len(self.pending),
            "silence_timeout_ms": self.settings.silence_timeout_ms,
            "silence_timeouts": self.silence_events,
        }


def _encoder(av, width: int, height: int, fps: int, gop: int, kbps: int):
    e = av.CodecContext.create("libx264", "w")
    e.width = width
    e.height = height
    e.pix_fmt = "yuv420p"
    e.time_base = Fraction(1, fps)
    e.framerate = Fraction(fps, 1)
    e.thread_count = 1
    e.bit_rate = kbps * 1000
    e.options = {
        "preset": "ultrafast",
        "tune": "zerolatency",
        "x264-params": (
            f"bframes=0:rc-lookahead=0:sync-lookahead=0:"
            f"keyint={gop}:min-keyint={gop}:scenecut=0:open-gop=0:"
            f"vbv-maxrate={kbps}:vbv-bufsize={kbps}:repeat-headers=1:aud=1"
        ),
    }
    e.open()
    return e


def _texture(np, width: int, height: int):
    """Fixed noise texture, translated each frame; reproducible entropy."""
    return np.random.default_rng(2026).integers(
        0, 256, size=(height, width, 3), dtype=np.uint8)


def _raw_frame(np, texture, seq: int):
    rgb = np.roll(texture, shift=(seq * 2, seq * 3), axis=(0, 1)).copy()
    height, width, _ = rgb.shape
    side = min(height // 4, width // 6, 22)
    y = (seq * 3) % (height - side + 1)
    x = (seq * 5) % (width - side + 1)
    rgb[y:y + side, x:x + side] = (255, 80, 30)
    return rgb


def stream_with_rate_control(*, width: int, height: int, fps: int,
                             frames: int, gop: int, link: LinkConfig,
                             rates: RateConfig, adapt: bool,
                             restart_seqs: tuple[int, ...] = ()) -> tuple[
                                 list[VideoAccessUnit], LinkTrace, dict]:
    """Encode in source order, drain only already arrived packets, adapt rate.

    The baseline and adaptive streams start with EXACTLY the same encoder
    configuration. Rate switches require delayed receiver feedback.
    restart_seqs is a fixed-high, offline ablation schedule only; it cannot
    be combined with live feedback and is not a deployable controller.
    """
    _validate_video(width, height, fps, frames, gop)
    if not isinstance(link, LinkConfig) or not isinstance(rates, RateConfig):
        raise ValueError("invalid simulator configuration")
    if type(adapt) is not bool:
        raise ValueError("adapt must be a boolean")
    if (type(restart_seqs) is not tuple or len(restart_seqs) > 20
            or any(type(seq) is not int or not 1 <= seq < frames for seq in restart_seqs)
            or tuple(sorted(set(restart_seqs))) != restart_seqs
            or (adapt and restart_seqs)):
        raise ValueError("restart_seqs must be sorted unique fixed-high source indices")
    try:
        import av
        import numpy as np
    except ImportError:
        raise RuntimeError("v26 requires optional PyAV and NumPy") from None

    source = _texture(np, width, height)
    media = VirtualAccessUnitLink(link)
    receiver = RateReceiver(rates)
    kbps = rates.high_kbps
    e = _encoder(av, width, height, fps, gop, kbps)
    origin_seq = 0
    packets: list[VideoAccessUnit] = []
    for seq in range(frames):
        now = seq * 1000.0 / fps
        arrived = media.arrive_by(now)
        if adapt:
            for item in arrived:
                receiver.observe(item)
            receiver.advance(now)
            target = receiver.apply_due(seq, now, kbps)
        else:
            target = kbps
        if target != kbps or seq in restart_seqs:
            if e.encode(None):
                raise RuntimeError("unexpected H.264 buffer during rate switch")
            e = _encoder(av, width, height, fps, gop, target)
            origin_seq = seq
            kbps = target

        raw = _raw_frame(np, source, seq)
        video = av.VideoFrame.from_ndarray(raw, format="rgb24")
        video.pts = seq - origin_seq
        encoded = e.encode(video)
        if len(encoded) != 1:
            raise RuntimeError("low-delay H.264 encoder did not emit one access unit")
        unit = encoded[0]
        data = bytes(unit)
        if not 0 < len(data) <= MAX_PACKET:
            raise RuntimeError("encoded H.264 access unit exceeds maximum")
        packet = VideoAccessUnit(
            seq=seq, pts_ns=round(seq * 1_000_000_000 / fps),
            capture_ns=round(seq * 1_000_000_000 / fps), data=data,
            keyframe=bool(unit.is_keyframe and is_keyframe(data)),
        )
        if seq == origin_seq and not packet.keyframe:
            raise RuntimeError("encoder segment failed to begin with an IDR")
        packets.append(packet)
        media.send(packet, fps=fps)
    if e.encode(None):
        raise RuntimeError("unexpected H.264 packets at source end")
    receiver.stop_watchdog()
    for item in media.arrive_by(math.inf):
        if adapt:
            receiver.observe(item)
    trace = media.snapshot(frames)
    return packets, trace, receiver.report()


def reconstruction_quality(trace: LinkTrace, *, width: int,
                           height: int, fps: int,
                           include_internal: bool = False) -> dict:
    """Decode delivered H.264 frames with real PyAV; compute RGB distortion.

    On gaps, discard dependent frames until an IDR. PSNR is pooled over
    actually decoded RGB samples and does not hide missing-frame stutter.
    """
    try:
        import av
        import numpy as np
    except ImportError:
        raise RuntimeError("optional PyAV/NumPy required for quality metric") from None
    texture = _texture(np, width, height)
    decoder = None
    expected = 0
    total_squared = 0
    total_samples = 0
    valid_frames = 0
    per_frame_sse: dict[int, int] = {}
    for item in trace.delivered:
        p = item.packet
        if p.seq != expected:
            decoder = None
        expected = p.seq + 1
        if decoder is None:
            if not p.keyframe or not is_keyframe(p.data):
                continue
            decoder = _make_decoder(av)
        output = decoder.decode(av.Packet(p.data))
        if len(output) != 1:
            raise RuntimeError("v26 H.264 decoder did not output one RGB frame")
        actual = output[0].to_ndarray(format="rgb24").astype(np.int16)
        original = _raw_frame(np, texture, p.seq).astype(np.int16)
        error = actual.astype(np.int32) - original.astype(np.int32)
        sse = int(np.sum(error * error, dtype=np.int64))
        total_squared += sse
        per_frame_sse[p.seq] = sse
        total_samples += actual.size
        valid_frames += 1
    mse = total_squared / total_samples if total_samples else None
    psnr = (10 * math.log10(255 * 255 / mse)
            if mse is not None and mse > 0 else
            (float("inf") if mse == 0 else None))
    result = {
        "quality_decoded_frames": valid_frames,
        "rgb_mse_on_decoded_frames": round(mse, 3) if mse is not None else None,
        "rgb_psnr_on_decoded_frames_db": round(psnr, 3)
            if psnr is not None and math.isfinite(psnr) else None,
        "quality_sample_note": "RGB error on correctly decoded frames only; lost frames excluded",
    }
    if include_internal:
        result["_sse_by_seq"] = per_frame_sse
    return result


def compare_rate_control(*, width: int = 128, height: int = 72,
                         fps: int = 30, frames: int = 90, gop: int = 24,
                         link: LinkConfig | None = None,
                         rates: RateConfig | None = None,
                         buffers_ms: tuple[int, ...] = (80, 160, 240)) -> dict:
    """Compare fixed high-bitrate encoding with causal high/low adaptation."""
    _validate_video(width, height, fps, frames, gop)
    link = (LinkConfig(delay_ms=40, jitter_ms=8, bandwidth_steps=(
                (0, 1200), (600, 90), (1750, 1200)),
                max_inflight=10, max_age_ms=350, seed=7)
            if link is None else link)
    rates = RateConfig() if rates is None else rates
    if not isinstance(link, LinkConfig) or not isinstance(rates, RateConfig):
        raise ValueError("invalid configuration")
    if (type(buffers_ms) is not tuple or not 1 <= len(buffers_ms) <= 8
            or any(type(x) is not int or not 0 <= x <= 1000 for x in buffers_ms)
            or tuple(sorted(set(buffers_ms))) != buffers_ms):
        raise ValueError("buffer depths must be sorted unique milliseconds")

    base_packets, base_trace, base_info = stream_with_rate_control(
        width=width, height=height, fps=fps, frames=frames, gop=gop,
        link=link, rates=rates, adapt=False,
    )
    adaptive_packets, adaptive_trace, controller = stream_with_rate_control(
        width=width, height=height, fps=fps, frames=frames, gop=gop,
        link=link, rates=rates, adapt=True,
    )
    first_change = controller["applied"][0]["source_seq"] if controller["applied"] else frames
    if adaptive_packets[:first_change] != base_packets[:first_change]:
        raise RuntimeError("adaptive encoder changed frames before causal feedback")
    if not controller["applied"] and adaptive_trace != base_trace:
        raise RuntimeError("no rate adaptation but link trace differs")

    baseline_decoder = decode_delivered_h264(base_trace, width, height)
    adaptive_decoder = decode_delivered_h264(adaptive_trace, width, height)
    baseline = _summarize_trace(base_trace, baseline_decoder, base_packets, fps, buffers_ms)
    adaptive = _summarize_trace(adaptive_trace, adaptive_decoder, adaptive_packets, fps, buffers_ms)
    base_quality = reconstruction_quality(
        base_trace, width=width, height=height, fps=fps, include_internal=True)
    adaptive_quality = reconstruction_quality(
        adaptive_trace, width=width, height=height, fps=fps, include_internal=True)
    base_sse = base_quality.pop("_sse_by_seq")
    adaptive_sse = adaptive_quality.pop("_sse_by_seq")
    common = sorted(set(base_sse) & set(adaptive_sse))
    sample_count = len(common) * width * height * 3

    def matched_psnr(squared):
        if not sample_count:
            return None
        mse = sum(squared[seq] for seq in common) / sample_count
        return round(10 * math.log10(255 * 255 / mse), 3) if mse > 0 else None

    base_quality["matched_frame_rgb_psnr_db"] = matched_psnr(base_sse)
    adaptive_quality["matched_frame_rgb_psnr_db"] = matched_psnr(adaptive_sse)
    base_quality["matched_frame_count"] = len(common)
    adaptive_quality["matched_frame_count"] = len(common)
    baseline.update(base_quality)
    adaptive.update(adaptive_quality)
    return {
        "schema": 1,
        "mode": "causal-h264-rate-adaptation",
        "source": {"width": width, "height": height, "fps": fps,
                   "frames": frames, "gop": gop, "pattern": "moving-deterministic-texture"},
        "link": {
            "delay_ms": link.delay_ms, "jitter_ms": link.jitter_ms,
            "bandwidth_steps": [list(x) for x in link.bandwidth_steps],
            "bandwidth_kbps": link.bandwidth_kbps,
            "loss_percent": link.loss_percent,
            "drop_seqs": list(link.drop_seqs),
            "max_age_ms": link.max_age_ms, "max_inflight": link.max_inflight,
            "seed": link.seed,
        },
        "rates": {
            "high_kbps": rates.high_kbps, "low_kbps": rates.low_kbps,
            "congested_age_ms": rates.congested_age_ms,
            "recovered_age_ms": rates.recovered_age_ms,
            "recovery_samples": rates.recovery_samples,
            "congestion_samples": rates.congestion_samples,
            "control_delay_ms": rates.control_delay_ms,
            "min_switch_ms": rates.min_switch_ms,
            "max_switches": rates.max_switches,
            "silence_timeout_ms": rates.silence_timeout_ms,
        },
        "controller": controller,
        "fixed_high_rate": baseline,
        "adaptive_rate": adaptive,
        "scope": (
            "real H264 bitrate-constrained encoding and decoding on moving texture; "
            "causal feedback from received packet age/gaps and optional receiver silence; "
            "no real network, Colab/GPU, GUI, physical presentation or bitrate guarantee"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--gop", type=int, default=24)
    parser.add_argument("--delay-ms", type=float, default=40)
    parser.add_argument("--jitter-ms", type=float, default=8)
    parser.add_argument("--bandwidth-steps", default="0:1200,600:90,1750:1200")
    parser.add_argument("--bandwidth-kbps", type=int, default=0)
    parser.add_argument("--loss-percent", type=float, default=0)
    parser.add_argument("--drop-seqs", default="")
    parser.add_argument("--max-inflight", type=int, default=10)
    parser.add_argument("--max-age-ms", type=float, default=350)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--high-kbps", type=int, default=650)
    parser.add_argument("--low-kbps", type=int, default=95)
    parser.add_argument("--congested-age-ms", type=float, default=110)
    parser.add_argument("--recovered-age-ms", type=float, default=65)
    parser.add_argument("--recovery-samples", type=int, default=8)
    parser.add_argument("--congestion-samples", type=int, default=3)
    parser.add_argument("--control-delay-ms", type=float, default=35)
    parser.add_argument("--min-switch-ms", type=float, default=350)
    parser.add_argument("--max-switches", type=int, default=8)
    parser.add_argument("--silence-timeout-ms", type=float, default=0,
                        help="receiver silence timeout; 0 preserves the v26 policy")
    parser.add_argument("--buffers-ms", default="80,160,240,350")
    args = parser.parse_args(argv)
    try:
        from .bridge_video_feedback import _parse_buffer_list
        link = LinkConfig(
            delay_ms=args.delay_ms, jitter_ms=args.jitter_ms,
            bandwidth_kbps=args.bandwidth_kbps,
            bandwidth_steps=parse_bandwidth_steps(args.bandwidth_steps),
            loss_percent=args.loss_percent,
            drop_seqs=parse_drop_seqs(args.drop_seqs),
            max_inflight=args.max_inflight, max_age_ms=args.max_age_ms,
            seed=args.seed,
        )
        rate = RateConfig(
            high_kbps=args.high_kbps, low_kbps=args.low_kbps,
            congested_age_ms=args.congested_age_ms,
            recovered_age_ms=args.recovered_age_ms,
            recovery_samples=args.recovery_samples,
            congestion_samples=args.congestion_samples,
            control_delay_ms=args.control_delay_ms,
            min_switch_ms=args.min_switch_ms,
            max_switches=args.max_switches,
            silence_timeout_ms=args.silence_timeout_ms,
        )
        report = compare_rate_control(
            width=args.width, height=args.height, fps=args.fps,
            frames=args.frames, gop=args.gop, link=link,
            rates=rate, buffers_ms=_parse_buffer_list(args.buffers_ms),
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
