"""v24: feedback-triggered H.264 IDR and virtual jitter-buffer research.

This is a bounded, LOCAL, offline counterfactual comparison. Real PyAV/libx264
access units are encoded/decoded, but both the media link and feedback message
take place on a deterministic VIRTUAL clock. No Colab, GPU or network access.

The receiver detects its FIRST missing access unit on arrival of the next
later unit. A one-way simulated feedback delay determines the earliest source
frame that can become an IDR. The source is re-encoded with a closed GOP reset
at that frame, then the changed access-unit sizes are re-scheduled over the
same link. The prefix before that decision must be byte-identical. This is
a single-feedback experiment, NOT a closed-loop WebRTC/RTP implementation.

A fixed-deadline playout model separately measures timely decoded frames,
missing frames, late-arriving frames and frozen display slots. It never
claims GUI/compositor presentation or physical input-to-photon latency.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from statistics import median

from .bridge_video_impairment import (
    LinkConfig, LinkTrace, _serialize_ms, decode_delivered_h264,
    make_h264_packets, parse_bandwidth_steps, parse_drop_seqs,
    simulate_link,
)
from .bridge_video_stream import VideoAccessUnit, is_keyframe


def _validate_video(width: int, height: int, fps: int, frames: int, gop: int) -> None:
    if (type(width) is not int or type(height) is not int
            or width % 2 or height % 2
            or not 64 <= width <= 640 or not 48 <= height <= 360
            or type(fps) is not int or not 1 <= fps <= 60
            or type(frames) is not int or not 4 <= frames <= 120
            or type(gop) is not int or not 2 <= gop <= 60):
        raise ValueError("v24 source parameters outside bounded range")


def first_gap_detection(trace: LinkTrace) -> tuple[int, float] | None:
    """First observed gap, which is knowable only when a later packet arrives."""
    expected = 0
    for scheduled in trace.delivered:
        seq = scheduled.packet.seq
        if seq > expected:
            return expected, scheduled.arrival_ms
        expected = seq + 1
    return None


def feedback_idr_plan(trace: LinkTrace, *, fps: int, frames: int,
                      feedback_ms: float) -> dict:
    """Plan earliest causally possible IDR after the first observed gap.

    A feedback request reaches the source at detection_time + feedback_ms.
    Frames must start STRICTLY AFTER receipt to avoid exact-time
    scheduling ambiguities. No oracle access to undetected losses.
    """
    if (type(fps) is not int or not 1 <= fps <= 60
            or type(frames) is not int or not 4 <= frames <= 120
            or type(feedback_ms) not in (int, float)
            or not math.isfinite(feedback_ms) or not 0 <= feedback_ms <= 1000):
        raise ValueError("invalid feedback planning bounds")
    gap = first_gap_detection(trace)
    if gap is None:
        return {"missing_seq_detected": None, "detected_at_ms": None,
                "feedback_reaches_source_ms": None, "forced_idr_seq": None}
    missing, when = gap
    delivered_at = when + feedback_ms
    period = 1000.0 / fps
    # The strictly later frame prevents changing a frame already in flight.
    frame = max(missing + 1, int(math.floor((delivered_at + 1e-8) / period)) + 1)
    return {
        "missing_seq_detected": missing,
        "detected_at_ms": round(when, 4),
        "feedback_reaches_source_ms": round(delivered_at, 4),
        "forced_idr_seq": frame if frame < frames else None,
    }


def _new_encoder(av, width: int, height: int, fps: int, gop: int):
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
    return encoder


def encode_with_forced_idr(*, width: int, height: int, fps: int,
                           frames: int, gop: int, forced_seq: int) -> list[VideoAccessUnit]:
    """Use a new real x264 encoder at forced_seq; its first frame is IDR.

    An encoder reset is one valid way of emitting a self-contained IDR
    with parameter headers. Unlike an in-place keyframe request, its
    CPU initialization cost and resulting rate-control reset are NOT
    included in virtual transport latency. Those are future measurements.
    """
    _validate_video(width, height, fps, frames, gop)
    if type(forced_seq) is not int or not 1 <= forced_seq < frames:
        raise ValueError("forced IDR source index outside bounds")
    try:
        import av
        import numpy as np
    except ImportError:
        raise RuntimeError("optional PyAV and NumPy are required for v24") from None
    from .bridge_video_live import LiveScene, live_background, render_live_rgb

    base = live_background(width, height, np)
    encoder = _new_encoder(av, width, height, fps, gop)
    packets: list[VideoAccessUnit] = []
    for seq in range(frames):
        if seq == forced_seq:
            if encoder.encode(None):
                raise RuntimeError("unexpected delayed H.264 frames before restart")
            encoder = _new_encoder(av, width, height, fps, gop)
        phase = seq / fps
        scene = LiveScene(
            x=.5 + .27 * math.sin(phase * 2),
            y=.5 + .22 * math.cos(phase * 3),
        )
        rgb = render_live_rgb(scene, base, seq)
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        # Reset encoder local PTS with a fresh origin after forcing IDR.
        frame.pts = seq if seq < forced_seq else seq - forced_seq
        result = encoder.encode(frame)
        if len(result) != 1:
            raise RuntimeError("zero-latency encoder did not emit exactly one frame")
        data = bytes(result[0])
        if not 0 < len(data) <= 2 * 1024 * 1024:
            raise RuntimeError("H.264 frame exceeds wire limit")
        packet = VideoAccessUnit(
            seq=seq, pts_ns=round(seq * 1_000_000_000 / fps),
            capture_ns=round(seq * 1_000_000_000 / fps),
            data=data,
            keyframe=bool(result[0].is_keyframe and is_keyframe(data)),
        )
        packets.append(packet)
    if encoder.encode(None):
        raise RuntimeError("unexpected delayed H.264 frames at end")
    if not packets[0].keyframe or not packets[forced_seq].keyframe:
        raise RuntimeError("H.264 encoder restart failed to generate IDR")
    return packets


def _decoded_indices(observed: dict) -> list[int]:
    runs = observed.get("decoded_runs")
    if not isinstance(runs, list):
        raise ValueError("invalid verified decoder runs")
    indices = []
    last = -1
    for run in runs:
        if (not isinstance(run, list) or len(run) != 2
                or any(type(x) is not int for x in run)
                or not 0 <= run[0] <= run[1] <= 120 or run[0] <= last):
            raise ValueError("decoder run sequence invalid")
        indices.extend(range(run[0], run[1] + 1))
        last = run[1]
    if len(indices) != observed.get("decoded_frames"):
        raise ValueError("decoder-run accounting mismatch")
    return indices


def fixed_deadline_playout(trace: LinkTrace, observed: dict,
                           *, fps: int, buffer_ms: float) -> dict:
    """Virtual fixed-rate display slots; only decoded frames can be shown.

    Slot n is at n * frame_period + buffer_ms. Late decoded frames are
    discarded at the DISPLAY layer, without breaking H.264 reference
    state in the decoder. Missed slots repeat the last presented frame,
    or are blank before the first successful frame. This simple model
    has no late-frame repair, display-time rendering, or adaptive PLL.
    """
    if (type(fps) is not int or not 1 <= fps <= 60
            or type(buffer_ms) not in (int, float)
            or not math.isfinite(buffer_ms) or not 0 <= buffer_ms <= 1000):
        raise ValueError("invalid playout buffer")
    valid = set(_decoded_indices(observed))
    arrivals = {s.packet.seq: s.arrival_ms for s in trace.delivered}
    if trace.source_count > 120:
        raise ValueError("frame source count exceeds v24 bounds")
    period = 1000.0 / fps
    on_time: list[int] = []
    late: list[int] = []
    absent: list[int] = []
    held = 0
    blank = 0
    successive_misses = 0
    worst_misses = 0
    seen = False
    for seq in range(trace.source_count):
        deadline = buffer_ms + seq * period
        if seq in valid and seq in arrivals and arrivals[seq] <= deadline + 1e-9:
            on_time.append(seq)
            seen = True
            successive_misses = 0
        else:
            if seq in valid:
                late.append(seq)
            else:
                absent.append(seq)
            successive_misses += 1
            worst_misses = max(worst_misses, successive_misses)
            if seen:
                held += 1
            else:
                blank += 1
    if len(on_time) + len(late) + len(absent) != trace.source_count:
        raise RuntimeError("playout-slot accounting mismatch")
    return {
        "buffer_ms": buffer_ms,
        "on_time_frames": len(on_time),
        "late_decoded_frames": len(late),
        "undecodable_frames": len(absent),
        "held_slots": held,
        "initial_blank_slots": blank,
        "longest_unfresh_run_slots": worst_misses,
        "longest_unfresh_run_ms": round(worst_misses * period, 2),
        "on_time_fraction": round(len(on_time) / trace.source_count, 4),
        "presented_sequence_runs": _runs(on_time),
        "scope": "virtual fixed-rate playout of verified decoded frames, not GUI presentation",
    }


def _runs(seqs: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for seq in seqs:
        if not runs or seq > runs[-1][1] + 1:
            runs.append([seq, seq])
        else:
            runs[-1][1] = seq
    return runs


def _summarize_trace(trace: LinkTrace, observed: dict, packets: list[VideoAccessUnit],
                     fps: int, buffers_ms: tuple[int, ...]) -> dict:
    return {
        "wire_bytes": trace.wire_sent_bytes,
        "encoded_idrs": sum(p.keyframe for p in packets),
        "delivered_frames": len(trace.delivered),
        "link_drops": len(trace.dropped),
        "link_drop_reasons": trace.loss_reasons,
        "actually_decoded_frames": observed["decoded_frames"],
        "decoded_runs": observed["decoded_runs"],
        "dependency_skipped": observed["dependency_skipped"],
        "decoder_idr_resynchronizations": observed["idr_resynchronizations"],
        "simulated_capture_to_arrival_p50_ms":
            observed["simulated_capture_to_arrival_p50_ms"],
        "playout": [
            fixed_deadline_playout(trace, observed, fps=fps, buffer_ms=b)
            for b in buffers_ms
        ],
    }


def compare_recovery_playout(*, width: int = 320, height: int = 180,
                             fps: int = 30, frames: int = 48, gop: int = 12,
                             link: LinkConfig | None = None,
                             feedback_ms: float = 20.0,
                             buffers_ms: tuple[int, ...] = (40, 80, 120, 180)) -> dict:
    """Compare periodic GOP against one feedback-triggered encoder restart.

    The periodic trace is used to detect the earliest gap. Its packet
    prefix is identical in the feedback trace until the forced IDR.
    Afterward, actual changed H.264 packet sizes are re-scheduled.

    The model sends at most ONE feedback request and assumes reverse
    signaling is delivered with the specified virtual one-way delay.
    It is not an adaptive congestion-control or RTP feedback protocol.
    """
    _validate_video(width, height, fps, frames, gop)
    link = LinkConfig() if link is None else link
    if not isinstance(link, LinkConfig):
        raise ValueError("invalid local link")
    if (type(buffers_ms) is not tuple or not 1 <= len(buffers_ms) <= 8
            or any(type(v) is not int or not 0 <= v <= 1000 for v in buffers_ms)
            or tuple(sorted(set(buffers_ms))) != buffers_ms):
        raise ValueError("buffers_ms must be sorted unique integers in 0..1000")
    baseline_packets = make_h264_packets(width=width, height=height,
                                         fps=fps, frames=frames, gop=gop)
    baseline_trace = simulate_link(baseline_packets, fps, link)
    baseline_decode = decode_delivered_h264(baseline_trace, width, height)
    plan = feedback_idr_plan(baseline_trace, fps=fps, frames=frames,
                             feedback_ms=feedback_ms)
    requested = plan["forced_idr_seq"]

    if requested is not None:
        altered_packets = encode_with_forced_idr(
            width=width, height=height, fps=fps, frames=frames,
            gop=gop, forced_seq=requested,
        )
        if altered_packets[:requested] != baseline_packets[:requested]:
            raise RuntimeError("feedback encoder changed source packets before request")
        altered_trace = simulate_link(altered_packets, fps, link)
        # An actual source would receive feedback before emitting its IDR.
        # The prefix invariance verifies causal equivalence for this
        # single-request counterfactual.
        pre = tuple((x.packet.seq, x.arrival_ms) for x in baseline_trace.delivered
                    if x.packet.seq < requested)
        post = tuple((x.packet.seq, x.arrival_ms) for x in altered_trace.delivered
                     if x.packet.seq < requested)
        if pre != post:
            raise RuntimeError("link dynamics changed before feedback IDR")
    else:
        altered_packets, altered_trace = baseline_packets, baseline_trace
    altered_decode = decode_delivered_h264(altered_trace, width, height)
    return {
        "schema": 1,
        "mode": "virtual-h264-feedback-playout",
        "source": {"width": width, "height": height,
                   "fps": fps, "frames": frames, "gop": gop},
        "link": {"delay_ms": link.delay_ms, "jitter_ms": link.jitter_ms,
                 "bandwidth_kbps": link.bandwidth_kbps,
                 "bandwidth_steps": [list(step) for step in link.bandwidth_steps],
                 "loss_percent": link.loss_percent,
                 "drop_seqs": list(link.drop_seqs),
                 "max_inflight": link.max_inflight,
                 "max_age_ms": link.max_age_ms, "seed": link.seed},
        "feedback": {"one_way_control_delay_ms": feedback_ms, **plan},
        "periodic": _summarize_trace(
            baseline_trace, baseline_decode, baseline_packets, fps, buffers_ms),
        "feedback_idr": _summarize_trace(
            altered_trace, altered_decode, altered_packets, fps, buffers_ms),
        "scope": (
            "single causal virtual feedback request, real encoded/decoded H264; "
            "virtual ordered access-unit link and fixed playout deadlines only; "
            "no remote network, Colab, GPU, GUI, loss recovery protocol or physical display"
        ),
    }


def _parse_buffer_list(raw: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(v.strip()) for v in raw.split(","))
        if (not 1 <= len(parsed) <= 8
                or any(not 0 <= v <= 1000 for v in parsed)
                or tuple(sorted(set(parsed))) != parsed):
            raise ValueError()
        return parsed
    except ValueError:
        raise ValueError("playout buffers must be sorted unique milliseconds e.g. 40,80,120,180") from None


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
    parser.add_argument("--bandwidth-steps", default="")
    parser.add_argument("--loss-percent", type=float, default=0.0)
    parser.add_argument("--drop-seqs", default="5")
    parser.add_argument("--max-inflight", type=int, default=12)
    parser.add_argument("--max-age-ms", type=float, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--feedback-ms", type=float, default=20,
                        help="simulated one-way feedback signal latency, 0..1000 ms")
    parser.add_argument("--buffers-ms", default="40,80,120,180",
                        help="virtual playout delays in milliseconds, sorted unique")
    args = parser.parse_args(argv)
    try:
        link = LinkConfig(
            delay_ms=args.delay_ms, jitter_ms=args.jitter_ms,
            bandwidth_kbps=args.bandwidth_kbps,
            bandwidth_steps=parse_bandwidth_steps(args.bandwidth_steps),
            loss_percent=args.loss_percent,
            drop_seqs=parse_drop_seqs(args.drop_seqs),
            max_inflight=args.max_inflight, max_age_ms=args.max_age_ms,
            seed=args.seed,
        )
        report = compare_recovery_playout(
            width=args.width, height=args.height, fps=args.fps,
            frames=args.frames, gop=args.gop, link=link,
            feedback_ms=args.feedback_ms,
            buffers_ms=_parse_buffer_list(args.buffers_ms),
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
