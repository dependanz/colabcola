"""v28: H.264 decoder recovery independent of encoder rate selection.

A v27 receiver-silence rate controller and the v25 multi-gap IDR controller
observe the same already-arrived access units. Each sends delayed feedback.
The sender combines due requests into one restart, including at unchanged
LOW bitrate. Full media silence does not generate speculative IDR retries.

Four streams compare fixed high rate, watchdog only, watchdog with recovery,
and fixed high rate with the combined run's exact restart schedule. That
last stream is an offline ablation, not an independently causal policy.
Codecs are real; link, feedback and playout times are virtual.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from .bridge_video_controller import ControllerConfig
from .bridge_video_feedback import _parse_buffer_list, _validate_video
from .bridge_video_impairment import LinkConfig, parse_bandwidth_steps, parse_drop_seqs
from .bridge_video_rate import RateConfig, stream_with_rate_control
from .bridge_video_watchdog import _summarize_rate_streams


def compare_rate_recovery(*, width: int = 128, height: int = 72,
                          fps: int = 30, frames: int = 90, gop: int = 24,
                          link: LinkConfig | None = None,
                          rates: RateConfig | None = None,
                          recovery: ControllerConfig | None = None,
                          buffers_ms: tuple[int, ...] = (80, 160, 240, 350)) -> dict:
    """Measure independent recovery with a same-restart fixed-high control."""
    _validate_video(width, height, fps, frames, gop)
    link = (LinkConfig(delay_ms=40, jitter_ms=8, bandwidth_steps=(
                (0, 1200), (600, 90), (1750, 1200)),
                max_inflight=10, max_age_ms=350, seed=7)
            if link is None else link)
    rates = RateConfig(silence_timeout_ms=150) if rates is None else rates
    recovery = (ControllerConfig(feedback_ms=35, retry_ms=150, max_requests=8)
                if recovery is None else recovery)
    if (not isinstance(link, LinkConfig) or not isinstance(rates, RateConfig)
            or not isinstance(recovery, ControllerConfig)):
        raise ValueError("invalid recovery experiment configuration")
    if rates.max_switches + recovery.max_requests > 20:
        raise ValueError("combined rate and recovery request budgets must not exceed 20")
    if (type(buffers_ms) is not tuple or not 1 <= len(buffers_ms) <= 8
            or any(type(x) is not int or not 0 <= x <= 1000 for x in buffers_ms)
            or tuple(sorted(set(buffers_ms))) != buffers_ms):
        raise ValueError("buffer depths must be sorted unique milliseconds")

    args = dict(width=width, height=height, fps=fps, frames=frames,
                gop=gop, link=link, rates=rates)
    streams = {
        "fixed_high_rate": stream_with_rate_control(**args, adapt=False),
        "watchdog_only": stream_with_rate_control(**args, adapt=True),
        "watchdog_with_recovery": stream_with_rate_control(
            **args, adapt=True, recovery=recovery),
    }
    base_packets, base_trace, _ = streams["fixed_high_rate"]
    before_packets, before_trace, before_info = streams["watchdog_only"]
    after_packets, after_trace, after_info = streams["watchdog_with_recovery"]
    for packets, trace, actions in (
        (before_packets, before_trace, before_info["applied"]),
        (after_packets, after_trace, after_info["encoder_restarts"]),
    ):
        boundary = actions[0]["source_seq"] if actions else frames
        if packets[:boundary] != base_packets[:boundary]:
            raise RuntimeError("feedback changed source bytes before its first applied request")
        prefix = lambda t: tuple(x for x in t.delivered if x.packet.seq < boundary)
        if prefix(trace) != prefix(base_trace):
            raise RuntimeError("feedback changed an already scheduled media arrival")
        if not actions and trace != base_trace:
            raise RuntimeError("inactive feedback changed the virtual link trace")

    # Compare effective encoder actions, including same-rate IDR restarts.
    # Feedback may share early rate changes; it cannot alter their prefix.
    before_actions = {x["source_seq"]: x["target_kbps"] for x in before_info["applied"]}
    after_actions = {x["source_seq"]: x["target_kbps"] for x in after_info["encoder_restarts"]}
    first_difference = min((seq for seq in before_actions.keys() | after_actions.keys()
                            if before_actions.get(seq) != after_actions.get(seq)), default=frames)
    if before_packets[:first_difference] != after_packets[:first_difference]:
        raise RuntimeError("recovery changed source bytes before the policies diverged")
    prefix = lambda t: tuple(x for x in t.delivered if x.packet.seq < first_difference)
    if prefix(before_trace) != prefix(after_trace):
        raise RuntimeError("recovery changed previously scheduled media arrivals")
    if first_difference == frames and before_trace != after_trace:
        raise RuntimeError("identical encoder actions changed the virtual link trace")

    restarts = tuple(x["source_seq"] for x in after_info["encoder_restarts"])
    streams["matched_restart_high_rate"] = stream_with_rate_control(
        **args, adapt=False, restart_seqs=restarts)
    idrs = lambda packets: [p.seq for p in packets if p.keyframe]
    if idrs(streams["matched_restart_high_rate"][0]) != idrs(after_packets):
        raise RuntimeError("restart control did not match the combined stream's IDRs")
    summaries, common = _summarize_rate_streams(
        streams, width=width, height=height, fps=fps, buffers_ms=buffers_ms)

    return {
        "schema": 1,
        "mode": "causal-h264-independent-idr-recovery",
        "source": {"width": width, "height": height, "fps": fps, "frames": frames,
                   "gop": gop, "pattern": "moving-deterministic-texture"},
        "link": asdict(link),
        "rates": asdict(rates),
        "recovery": asdict(recovery),
        "controllers": {
            "watchdog_only": before_info,
            "watchdog_with_recovery": after_info,
        },
        "first_policy_difference_seq": first_difference if first_difference < frames else None,
        "restart_control": {
            "restart_seqs": list(restarts),
            "target_kbps": rates.high_kbps,
            "note": "offline fixed-high ablation conditioned on ALL combined-controller restarts; "
                    "not an independently causal policy",
        },
        "quality_comparison": {
            "common_frame_seqs": common,
            "note": "matched PSNR pools RGB squared error on the same frame IDs decoded by ALL FOUR arms; "
                    "lost frames excluded, so compare continuity and playout separately",
        },
        **summaries,
        "scope": {
            "time": "media arrival, feedback and playout timestamps are virtual milliseconds; codecs are real",
            "recovery": "arrival-driven sequence-gap/reference-chain recovery, independent of bitrate; "
                        "bounded retries, no speculative IDRs during full media silence",
            "feedback": "ideal reliable fixed-delay reverse paths; no bandwidth/loss/control-byte model "
                        "or cancellation of in-flight recovery requests",
            "coalescing": "a rate change and any IDR requests due before a source frame share one restart",
            "paired_link": "identical parameters and seed; admission changes may associate random draws "
                           "with different source IDs; zero jitter/random loss is the cleaner ablation",
            "limits": "two target rates, at most 20 configured rate/recovery requests in total; "
                      "no capacity estimator, network, Colab/GPU, GUI or bitrate guarantee",
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
    parser.add_argument("--recovery-feedback-ms", type=float, default=35)
    parser.add_argument("--recovery-retry-ms", type=float, default=150)
    parser.add_argument("--max-recovery-requests", type=int, default=8)
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
        recovery = ControllerConfig(feedback_ms=args.recovery_feedback_ms,
                                    retry_ms=args.recovery_retry_ms,
                                    max_requests=args.max_recovery_requests)
        report = compare_rate_recovery(
            width=args.width, height=args.height, fps=args.fps, frames=args.frames,
            gop=args.gop, link=link, rates=rate, recovery=recovery,
            buffers_ms=_parse_buffer_list(args.buffers_ms))
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
