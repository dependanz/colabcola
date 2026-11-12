"""v27: receiver-silence H.264 feedback and a matched-restart ablation.

Four streams share a deterministic RGB source and virtual link settings:
fixed high target, v26 arrival-only feedback, feedback with a receiver
silence timer, and fixed high target with the timer run's restart schedule.
The last is an OFFLINE counterfactual, not an independently causal policy.

Media encoding/decoding is real. Time, bandwidth, playout and reverse
feedback are simulated; there is no network connection or Colab compute.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math

from .bridge_video_feedback import _parse_buffer_list, _summarize_trace, _validate_video
from .bridge_video_impairment import (
    LinkConfig, decode_delivered_h264, parse_bandwidth_steps, parse_drop_seqs,
)
from .bridge_video_rate import (
    RateConfig, reconstruction_quality, stream_with_rate_control,
)


def _summarize_rate_streams(streams: dict, *, width: int, height: int,
                            fps: int, buffers_ms: tuple[int, ...]) -> tuple[dict, list[int]]:
    """Decode each arm and compare RGB error on one common frame set."""
    idrs = lambda packets: [p.seq for p in packets if p.keyframe]
    summaries = {}
    distortions = {}
    for name, (packets, trace, _) in streams.items():
        decoded = decode_delivered_h264(trace, width, height)
        summary = _summarize_trace(trace, decoded, packets, fps, buffers_ms)
        quality = reconstruction_quality(
            trace, width=width, height=height, fps=fps, include_internal=True)
        distortions[name] = quality.pop("_sse_by_seq")
        if quality["quality_decoded_frames"] != summary["actually_decoded_frames"]:
            raise RuntimeError("quality and continuity decoders disagree")
        summary.update(quality)
        payload_bytes = sum(len(p.data) for p in packets)
        arrivals = {item.packet.seq: item.arrival_ms for item in trace.delivered}
        drops = dict(trace.dropped)
        summary.update({
            "encoded_idr_seqs": idrs(packets),
            "idr_delivery": [{
                "source_seq": seq,
                "arrival_ms": round(arrivals[seq], 3) if seq in arrivals else None,
                "drop_reason": drops.get(seq),
            } for seq in idrs(packets)],
            "encoded_payload_bytes": payload_bytes,
            "encoded_source_kbps": round(payload_bytes * 8 * fps / (len(packets) * 1000), 3),
        })
        summaries[name] = summary

    common = sorted(set.intersection(*(set(sse) for sse in distortions.values())))
    for name, summary in summaries.items():
        mse = (sum(distortions[name][seq] for seq in common)
               / (len(common) * width * height * 3)) if common else None
        summary["matched_frame_count"] = len(common)
        summary["matched_frame_rgb_psnr_db"] = (
            round(10 * math.log10(255 * 255 / mse), 3) if mse is not None and mse > 0 else None)

    return summaries, common


def compare_rate_watchdog(*, width: int = 128, height: int = 72,
                          fps: int = 30, frames: int = 90, gop: int = 24,
                          link: LinkConfig | None = None,
                          rates: RateConfig | None = None,
                          buffers_ms: tuple[int, ...] = (80, 160, 240, 350)) -> dict:
    """Compare arrival-only and silence feedback, with a restart control.

    The timer arms only after the first observed packet. No startup loss
    detector is modeled. A silent receiver requests LOW after its timeout
    and request dwell, then waits the fixed reverse feedback delay. The
    source applies it strictly before a future frame, never retroactively.
    """
    _validate_video(width, height, fps, frames, gop)
    link = (LinkConfig(delay_ms=40, jitter_ms=8, bandwidth_steps=(
                (0, 1200), (600, 90), (1750, 1200)),
                max_inflight=10, max_age_ms=350, seed=7)
            if link is None else link)
    rates = RateConfig(silence_timeout_ms=150) if rates is None else rates
    if not isinstance(link, LinkConfig) or not isinstance(rates, RateConfig):
        raise ValueError("invalid watchdog experiment configuration")
    if (type(buffers_ms) is not tuple or not 1 <= len(buffers_ms) <= 8
            or any(type(x) is not int or not 0 <= x <= 1000 for x in buffers_ms)
            or tuple(sorted(set(buffers_ms))) != buffers_ms):
        raise ValueError("buffer depths must be sorted unique milliseconds")

    args = dict(width=width, height=height, fps=fps, frames=frames,
                gop=gop, link=link)
    streams = {
        "fixed_high_rate": stream_with_rate_control(**args, rates=rates, adapt=False),
        "arrival_feedback": stream_with_rate_control(
            **args, rates=replace(rates, silence_timeout_ms=0), adapt=True),
        "silence_feedback": stream_with_rate_control(**args, rates=rates, adapt=True),
    }
    base_packets, base_trace, _ = streams["fixed_high_rate"]
    for name in ("arrival_feedback", "silence_feedback"):
        packets, trace, controller = streams[name]
        boundary = controller["applied"][0]["source_seq"] if controller["applied"] else frames
        if packets[:boundary] != base_packets[:boundary]:
            raise RuntimeError("feedback changed source bytes before its first applied request")
        # Future feedback must not change already scheduled packet arrival times.
        prefix = lambda t: tuple(x for x in t.delivered if x.packet.seq < boundary)
        if prefix(trace) != prefix(base_trace):
            raise RuntimeError("feedback changed a previously scheduled media arrival")
        if not controller["applied"] and trace != base_trace:
            raise RuntimeError("inactive feedback changed the virtual link trace")

    # Both adaptive policies may share early switches. Compare up to the
    # first DIFFERENT action, not just each policy's first action, so the
    # silence timer cannot change source bytes before it changes a decision.
    arrival_packets, arrival_trace, arrival_controller = streams["arrival_feedback"]
    timer_packets, timer_trace, timer_controller = streams["silence_feedback"]
    arrival_actions = {x["source_seq"]: x["target_kbps"] for x in arrival_controller["applied"]}
    timer_actions = {x["source_seq"]: x["target_kbps"] for x in timer_controller["applied"]}
    first_difference = min((seq for seq in arrival_actions.keys() | timer_actions.keys()
                            if arrival_actions.get(seq) != timer_actions.get(seq)), default=frames)
    if arrival_packets[:first_difference] != timer_packets[:first_difference]:
        raise RuntimeError("silence timer changed source bytes before the policies diverged")
    arrival_prefix = lambda t: tuple(x for x in t.delivered if x.packet.seq < first_difference)
    if arrival_prefix(arrival_trace) != arrival_prefix(timer_trace):
        raise RuntimeError("silence timer changed previously scheduled media arrivals")
    if first_difference == frames and arrival_trace != timer_trace:
        raise RuntimeError("identical feedback actions changed the virtual link trace")

    watchdog_packets, _, watchdog_controller = streams["silence_feedback"]
    restarts = tuple(x["source_seq"] for x in watchdog_controller["applied"])
    streams["matched_restart_high_rate"] = stream_with_rate_control(
        **args, rates=rates, adapt=False, restart_seqs=restarts)
    replay_packets = streams["matched_restart_high_rate"][0]
    idrs = lambda packets: [p.seq for p in packets if p.keyframe]
    if idrs(replay_packets) != idrs(watchdog_packets):
        raise RuntimeError("offline restart control did not match all adaptive IDRs")

    summaries, common = _summarize_rate_streams(
        streams, width=width, height=height, fps=fps, buffers_ms=buffers_ms)

    return {
        "schema": 1,
        "mode": "causal-h264-receiver-silence-watchdog",
        "source": {"width": width, "height": height, "fps": fps, "frames": frames,
                   "gop": gop, "pattern": "moving-deterministic-texture"},
        "link": asdict(link),
        "rates": asdict(rates),
        "controllers": {name: streams[name][2]
                        for name in ("arrival_feedback", "silence_feedback")},
        "first_policy_difference_seq": first_difference if first_difference < frames else None,
        "restart_control": {
            "restart_seqs": list(restarts),
            "target_kbps": rates.high_kbps,
            "note": "offline fixed-high ablation conditioned on the silence-feedback restart schedule; "
                    "not an independently causal policy",
        },
        "quality_comparison": {
            "common_frame_seqs": common,
            "note": "matched PSNR pools RGB squared error on the same frame IDs decoded by ALL FOUR arms; "
                    "lost frames excluded, so compare continuity and playout separately",
        },
        **summaries,
        "scope": {
            "time": "all media arrival, timeout, feedback and playout timestamps are virtual milliseconds",
            "feedback": "ideal reliable reverse path with a fixed delay; no reverse bandwidth/loss model "
                        "or control-byte accounting",
            "watchdog": "armed by first usable arrival; no startup silence detection; "
                        "disabled after the final source frame; silence is not proof of congestion",
            "paired_link": "identical parameters and seed, not necessarily identical per-frame jitter/loss "
                           "draws when admission changes; use zero jitter/random loss for a cleaner ablation",
            "limits": "two target rates and encoder restarts; no capacity estimator, actual network, "
                      "Colab/GPU, GUI, physical presentation or bitrate guarantee",
        },
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
    parser.add_argument("--silence-timeout-ms", type=float, default=150)
    parser.add_argument("--buffers-ms", default="80,160,240,350")
    args = parser.parse_args(argv)
    try:
        link = LinkConfig(
            delay_ms=args.delay_ms, jitter_ms=args.jitter_ms,
            bandwidth_kbps=args.bandwidth_kbps,
            bandwidth_steps=parse_bandwidth_steps(args.bandwidth_steps),
            loss_percent=args.loss_percent, drop_seqs=parse_drop_seqs(args.drop_seqs),
            max_inflight=args.max_inflight, max_age_ms=args.max_age_ms, seed=args.seed)
        rate = RateConfig(
            high_kbps=args.high_kbps, low_kbps=args.low_kbps,
            congested_age_ms=args.congested_age_ms, recovered_age_ms=args.recovered_age_ms,
            recovery_samples=args.recovery_samples, congestion_samples=args.congestion_samples,
            control_delay_ms=args.control_delay_ms, min_switch_ms=args.min_switch_ms,
            max_switches=args.max_switches, silence_timeout_ms=args.silence_timeout_ms)
        report = compare_rate_watchdog(
            width=args.width, height=args.height, fps=args.fps, frames=args.frames,
            gop=args.gop, link=link, rates=rate, buffers_ms=_parse_buffer_list(args.buffers_ms))
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
