"""v23: deterministic local impaired-link H.264 access-unit experiment.

Real PyAV/libx264 encoding and decoding, but VIRTUAL link time: no sleeps,
external sockets, managed Colab calls, GPU allocations, or remote service.

The modeled link is reliable and FIFO except for explicitly simulated loss
of WHOLE H.264 access units. It approximates finite bandwidth, delay, jitter,
a bounded in-flight queue and age-based dropping. This is NOT a realistic
UDP packet/fragments/FEC/WebRTC emulator, nor an input-to-photon benchmark.

An H.264 decoder must wait for a fresh IDR after any missing predicted frame.
We explicitly suppress dependent frames until a closed-GOP IDR arrives;
we do not pretend a stale P-frame can be dropped without consequences.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import json
import math
import random
from statistics import median

from .bridge_video_stream import VideoAccessUnit, packet_bytes, receive_packet, is_keyframe

MAX_FRAMES = 120
MAX_WIRE_BYTES = 24 * 1024 * 1024


@dataclass(frozen=True)
class LinkConfig:
    delay_ms: float = 40.0
    jitter_ms: float = 12.0
    bandwidth_kbps: int = 2000  # 0 = unlimited when no step profile exists
    bandwidth_steps: tuple[tuple[int, int], ...] = ()
    loss_percent: float = 0.0
    max_inflight: int = 12
    max_age_ms: float = 300.0  # 0 disables age-based dropping
    seed: int = 7
    drop_seqs: tuple[int, ...] = ()

    def __post_init__(self):
        for field, value, lo, hi in (
            ("delay_ms", self.delay_ms, 0, 1000),
            ("jitter_ms", self.jitter_ms, 0, 1000),
            ("loss_percent", self.loss_percent, 0, 100),
            ("max_age_ms", self.max_age_ms, 0, 10000),
        ):
            if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"invalid {field}")
        for field, value, lo, hi in (
            ("bandwidth_kbps", self.bandwidth_kbps, 0, 10_000_000),
            ("max_inflight", self.max_inflight, 1, 128),
            ("seed", self.seed, 0, 2**32 - 1),
        ):
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"invalid {field}")
        if type(self.bandwidth_steps) is not tuple or len(self.bandwidth_steps) > 12:
            raise ValueError("invalid bandwidth step schedule")
        last = -1
        for step in self.bandwidth_steps:
            if (type(step) is not tuple or len(step) != 2
                    or type(step[0]) is not int or type(step[1]) is not int
                    or not 0 <= step[0] <= 10000 or not 1 <= step[1] <= 10_000_000
                    or step[0] <= last):
                raise ValueError("invalid bandwidth step")
            last = step[0]
        if self.bandwidth_steps and self.bandwidth_steps[0][0] != 0:
            raise ValueError("bandwidth step profile must start at t=0 ms")
        if (type(self.drop_seqs) is not tuple
                or len(self.drop_seqs) > MAX_FRAMES
                or any(type(seq) is not int or not 0 <= seq < MAX_FRAMES
                       for seq in self.drop_seqs)
                or tuple(sorted(set(self.drop_seqs))) != self.drop_seqs):
            raise ValueError("drop_seqs must be sorted unique bounded integers")


@dataclass(frozen=True)
class ScheduledUnit:
    packet: VideoAccessUnit
    source_ms: float
    arrival_ms: float
    wire_size: int


@dataclass(frozen=True)
class LinkTrace:
    source_count: int
    delivered: tuple[ScheduledUnit, ...]
    dropped: tuple[tuple[int, str], ...]
    wire_sent_bytes: int
    max_queue_occupancy: int

    @property
    def loss_reasons(self) -> dict[str, int]:
        reasons = {"erased": 0, "queue_full": 0, "stale": 0}
        for _, reason in self.dropped:
            reasons[reason] += 1
        return reasons


def _serialize_ms(start_ms: float, bits: int, config: LinkConfig) -> float:
    """Integrate bits over a piecewise-constant link rate (kbps = bits/ms)."""
    if not config.bandwidth_steps and config.bandwidth_kbps == 0:
        return start_ms
    if config.bandwidth_steps:
        steps = config.bandwidth_steps
        index = max(i for i, (at, _) in enumerate(steps) if at <= start_ms)
        current = start_ms
        remaining = float(bits)
        while True:
            rate = steps[index][1]  # 1 kilobit/s = 1 bit/millisecond
            next_t = steps[index + 1][0] if index + 1 < len(steps) else math.inf
            transferable = rate * (next_t - current)
            if remaining <= transferable:
                return current + remaining / rate
            remaining -= transferable
            current = next_t
            index += 1
    return start_ms + bits / config.bandwidth_kbps


def simulate_link(packets: list[VideoAccessUnit], fps: int,
                  config: LinkConfig) -> LinkTrace:
    """Schedule complete H.264 access units through a bounded FIFO link.

    Drops from erasure/staleness happen AFTER bandwidth consumption; a
    queue-full drop is at admission and does not use bandwidth. The FIFO
    link preserves arrival order, with jitter affecting spacing and
    possible head-of-line blocking. Simulated timestamps are not wall time.
    """
    if type(fps) is not int or not 1 <= fps <= 60:
        raise ValueError("invalid source fps")
    if not isinstance(config, LinkConfig):
        raise ValueError("invalid link configuration")
    if not 1 <= len(packets) <= MAX_FRAMES:
        raise ValueError("source frame count outside limits")
    rng = random.Random(config.seed)
    pending_arrivals: deque[float] = deque()
    delivered: list[ScheduledUnit] = []
    dropped: list[tuple[int, str]] = []
    link_free_at = 0.0
    previous_arrival = 0.0
    wire_sent = 0
    peak = 0

    for seq, packet in enumerate(packets):
        if not isinstance(packet, VideoAccessUnit) or packet.seq != seq:
            raise ValueError("frames must have consecutive source IDs")
        source_time = seq * 1000.0 / fps
        while pending_arrivals and pending_arrivals[0] <= source_time:
            pending_arrivals.popleft()
        # Validate packet bounds before making any transmission decisions.
        wire_size = len(packet_bytes(packet))
        if len(pending_arrivals) >= config.max_inflight:
            dropped.append((seq, "queue_full"))
            continue
        start = max(source_time, link_free_at)
        finish = _serialize_ms(start, wire_size * 8, config)
        link_free_at = finish
        jitter = rng.uniform(-config.jitter_ms, config.jitter_ms) if config.jitter_ms else 0.0
        arrival = max(finish, finish + config.delay_ms + jitter, previous_arrival)
        previous_arrival = arrival
        pending_arrivals.append(arrival)
        peak = max(peak, len(pending_arrivals))
        wire_sent += wire_size

        # Access-unit erasure is explicitly a whole-frame abstraction.
        erased = seq in config.drop_seqs or (
            config.loss_percent > 0 and rng.random() * 100 < config.loss_percent
        )
        if erased:
            dropped.append((seq, "erased"))
        elif config.max_age_ms and arrival - source_time > config.max_age_ms:
            dropped.append((seq, "stale"))
        else:
            delivered.append(ScheduledUnit(packet, source_time, arrival, wire_size))
    if wire_sent > MAX_WIRE_BYTES:
        raise ValueError("total admitted video wire bytes exceeded offline limit")
    if len(delivered) + len(dropped) != len(packets):
        raise RuntimeError("impairment frame accounting mismatch")
    return LinkTrace(len(packets), tuple(delivered), tuple(dropped),
                     wire_sent, peak)


class _WireReader:
    """Simulate fragmented reads over the versioned local binary protocol."""

    def __init__(self, wire: bytes):
        self._wire = memoryview(wire)
        self._at = 0

    def recv(self, count: int) -> bytes:
        if count <= 0:
            return b""
        step = min(count, 4096, len(self._wire) - self._at)
        result = bytes(self._wire[self._at:self._at + step])
        self._at += step
        return result


def parse_virtual_packet(packet: VideoAccessUnit) -> VideoAccessUnit:
    """Validate real bridge packet framing despite using a virtual-time link."""
    restored = receive_packet(_WireReader(packet_bytes(packet)))
    if restored != packet:
        raise RuntimeError("impaired link changed authenticated frame metadata")
    return restored


def _make_decoder(av):
    decoder = av.CodecContext.create("h264", "r")
    decoder.thread_count = 1
    decoder.open()
    return decoder


def decode_delivered_h264(trace: LinkTrace, width: int, height: int) -> dict:
    """Decode only complete consecutive reference chains, restarting at IDR."""
    try:
        import av
    except ImportError:
        raise RuntimeError("optional PyAV required for H.264 impairment decoding") from None

    decoder = None
    expected_seq = 0
    had_synced = False
    recovered = 0
    dependency_skipped = 0
    decoded_seqs: list[int] = []
    decoded_at_ms: list[float] = []
    ages_ms: list[float] = []
    actual_codec_ms: list[float] = []
    from time import perf_counter_ns

    for item in trace.delivered:
        packet = parse_virtual_packet(item.packet)
        if packet.seq != expected_seq:
            decoder = None  # A missing P-frame invalidates its descendants.
        expected_seq = packet.seq + 1

        if decoder is None:
            if not packet.keyframe or not is_keyframe(packet.data):
                dependency_skipped += 1
                continue
            decoder = _make_decoder(av)
            if had_synced:
                recovered += 1
            had_synced = True
        started = perf_counter_ns()
        decoded = decoder.decode(av.Packet(packet.data))
        actual_codec_ms.append((perf_counter_ns() - started) / 1e6)
        if len(decoded) != 1:
            raise RuntimeError("zero-latency H.264 test expected one decoded frame per access unit")
        image = decoded[0]
        if image.width != width or image.height != height:
            raise RuntimeError("recovered H.264 image geometry mismatch")
        pixels = image.to_ndarray(format="rgb24")
        if pixels.shape != (height, width, 3):
            raise RuntimeError("invalid decoded RGB shape")
        decoded_seqs.append(packet.seq)
        decoded_at_ms.append(item.arrival_ms)
        ages_ms.append(item.arrival_ms - item.source_ms)

    runs: list[list[int]] = []
    for seq in decoded_seqs:
        if not runs or seq > runs[-1][1] + 1:
            runs.append([seq, seq])
        else:
            runs[-1][1] = seq
    gaps = [b - a for a, b in zip(decoded_at_ms, decoded_at_ms[1:])]
    virtual_fps = ((len(decoded_seqs) - 1) * 1000 /
                   (decoded_at_ms[-1] - decoded_at_ms[0])
                   if len(decoded_at_ms) > 1 and decoded_at_ms[-1] > decoded_at_ms[0]
                   else 0.0)
    return {
        "decoded_frames": len(decoded_seqs),
        "decoded_runs": runs,
        "dependency_skipped": dependency_skipped,
        "idr_resynchronizations": recovered,
        "simulated_capture_to_arrival_p50_ms":
            round(median(ages_ms), 2) if ages_ms else None,
        "simulated_capture_to_arrival_max_ms":
            round(max(ages_ms), 2) if ages_ms else None,
        "simulated_arrival_gap_max_ms":
            round(max(gaps), 2) if gaps else None,
        "simulated_received_cadence_fps": round(virtual_fps, 2),
        "actual_decoder_call_p50_ms":
            round(median(actual_codec_ms), 3) if actual_codec_ms else None,
    }


def make_h264_packets(*, width: int, height: int,
                      fps: int, frames: int, gop: int) -> list[VideoAccessUnit]:
    """Encode changing local synthetic RGB frames as a closed-GOP H.264 trace."""
    if (type(width) is not int or type(height) is not int
            or width % 2 or height % 2
            or not 64 <= width <= 640 or not 48 <= height <= 360
            or type(fps) is not int or not 1 <= fps <= 60
            or type(frames) is not int or not 4 <= frames <= MAX_FRAMES
            or type(gop) is not int or not 2 <= gop <= 60):
        raise ValueError("invalid bounded H.264 source configuration")
    try:
        import av
        import numpy as np
    except ImportError:
        raise RuntimeError("optional PyAV and NumPy required for H.264 impairment test") from None
    from .bridge_video_live import LiveScene, live_background, render_live_rgb

    encoder = av.CodecContext.create("libx264", "w")
    encoder.width = width
    encoder.height = height
    encoder.pix_fmt = "yuv420p"
    encoder.time_base = Fraction(1, fps)
    encoder.framerate = Fraction(fps, 1)
    encoder.thread_count = 1
    encoder.options = {
        "preset": "ultrafast", "tune": "zerolatency", "crf": "23",
        "x264-params": (
            f"bframes=0:rc-lookahead=0:sync-lookahead=0:keyint={gop}:"
            f"min-keyint={gop}:scenecut=0:open-gop=0:repeat-headers=1:aud=1"
        ),
    }
    encoder.open()
    base = live_background(width, height, np)
    packets: list[VideoAccessUnit] = []
    for seq in range(frames):
        phase = seq / fps
        scene = LiveScene(
            x=.5 + .27 * math.sin(phase * 2),
            y=.5 + .22 * math.cos(phase * 3),
        )
        rgb = render_live_rgb(scene, base, seq)
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = seq
        encoded = encoder.encode(frame)
        if len(encoded) != 1:
            raise RuntimeError("low-delay H.264 encoder did not emit one frame")
        data = bytes(encoded[0])
        if not 0 < len(data) <= 2 * 1024 * 1024:
            raise RuntimeError("generated access unit exceeds local wire cap")
        packet = VideoAccessUnit(
            seq=seq, pts_ns=round(seq * 1_000_000_000 / fps),
            capture_ns=round(seq * 1_000_000_000 / fps),
            data=data,
            keyframe=bool(encoded[0].is_keyframe and is_keyframe(data)),
        )
        packets.append(packet)
    if encoder.encode(None):
        raise RuntimeError("unexpected delayed H.264 frames from zero-latency encoder")
    if not packets[0].keyframe:
        raise RuntimeError("H.264 source did not start with an IDR")
    return packets


def parse_bandwidth_steps(raw: str) -> tuple[tuple[int, int], ...]:
    if not raw.strip():
        return ()
    try:
        entries = tuple(tuple(int(v) for v in chunk.split(":"))
                        for chunk in raw.split(","))
        LinkConfig(bandwidth_steps=entries)
        return entries
    except (ValueError, TypeError):
        raise ValueError("bandwidth steps must be '0:2500,800:250,1600:2500'") from None


def parse_drop_seqs(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    try:
        result = tuple(int(x) for x in raw.split(","))
        LinkConfig(drop_seqs=result)
        return result
    except (ValueError, TypeError):
        raise ValueError("drop sequence list must be sorted unique IDs, e.g. 5,21") from None


def compare_impairments(*, width: int = 320, height: int = 180,
                        fps: int = 30, frames: int = 48, gop: int = 12,
                        config: LinkConfig | None = None) -> dict:
    """Offline scheduled link + actual H.264 decode with conservative IDR recovery."""
    config = config or LinkConfig()
    packets = make_h264_packets(width=width, height=height, fps=fps,
                                frames=frames, gop=gop)
    trace = simulate_link(packets, fps, config)
    observed = decode_delivered_h264(trace, width, height)
    if (observed["decoded_frames"] + observed["dependency_skipped"]
            + len(trace.dropped) != frames):
        raise RuntimeError("source, link and decoder frame accounting mismatch")
    return {
        "schema": 1,
        "mode": "virtual-time-h264-impaired-link",
        "source": {"width": width, "height": height, "fps": fps,
                   "frames": frames, "gop": gop,
                   "encoded_keyframes": sum(p.keyframe for p in packets)},
        "link": {
            "delay_ms": config.delay_ms,
            "jitter_ms": config.jitter_ms,
            "bandwidth_kbps": config.bandwidth_kbps,
            "bandwidth_steps": [list(step) for step in config.bandwidth_steps],
            "loss_percent": config.loss_percent,
            "drop_seqs": list(config.drop_seqs),
            "max_inflight": config.max_inflight,
            "max_age_ms": config.max_age_ms,
            "seed": config.seed,
        },
        "outcome": {
            "admitted_wire_bytes": trace.wire_sent_bytes,
            "link_delivered_units": len(trace.delivered),
            "link_dropped_units": len(trace.dropped),
            "link_drop_reasons": trace.loss_reasons,
            "link_drop_seqs": [seq for seq, _ in trace.dropped],
            "peak_inflight_units": trace.max_queue_occupancy,
            **observed,
        },
        "scope": (
            "deterministic local virtual link over complete H264 access units; "
            "actual codec decode; no real network, GPU, Colab or GUI"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--gop", type=int, default=12)
    parser.add_argument("--delay-ms", type=float, default=40.0)
    parser.add_argument("--jitter-ms", type=float, default=12.0)
    parser.add_argument("--bandwidth-kbps", type=int, default=2000)
    parser.add_argument("--bandwidth-steps", default="",
                        help="piecewise link rate, e.g. 0:2500,800:250,1600:2500")
    parser.add_argument("--loss-percent", type=float, default=0.0)
    parser.add_argument("--drop-seqs", default="",
                        help="erase whole encoded access units by source ID, e.g. 5,17")
    parser.add_argument("--max-inflight", type=int, default=12)
    parser.add_argument("--max-age-ms", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    try:
        config = LinkConfig(
            delay_ms=args.delay_ms, jitter_ms=args.jitter_ms,
            bandwidth_kbps=args.bandwidth_kbps,
            bandwidth_steps=parse_bandwidth_steps(args.bandwidth_steps),
            loss_percent=args.loss_percent, max_inflight=args.max_inflight,
            max_age_ms=args.max_age_ms, seed=args.seed,
            drop_seqs=parse_drop_seqs(args.drop_seqs),
        )
        report = compare_impairments(
            width=args.width, height=args.height,
            fps=args.fps, frames=args.frames, gop=args.gop, config=config,
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
