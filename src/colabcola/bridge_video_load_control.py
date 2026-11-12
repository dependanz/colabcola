"""v29: measured source wire load and delayed receiver feedback for H.264.

Four local streams compare the v28 two-rate policy, a new multiplicative
age/silence policy, that policy with an observed-delivery wire budget, and
fixed high rate replaying the wire policy's encoder restart schedule.
The replay is an offline ablation. Real codecs; virtual media and control.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from .bridge_video_controller import ControllerConfig
from .bridge_video_feedback import _parse_buffer_list, _validate_video
from .bridge_video_impairment import LinkConfig, parse_bandwidth_steps, parse_drop_seqs
from .bridge_video_load import LoadControlConfig, LoadRateReceiver
from .bridge_video_rate import RateConfig, stream_with_rate_control
from .bridge_video_stream import packet_bytes
from .bridge_video_watchdog import _summarize_rate_streams


def _check_action_prefix(before, after, frames: int) -> int | None:
    """Effective restart/target changes must precede any stream difference."""
    old_packets, old_trace, old_report = before
    new_packets, new_trace, new_report = after
    old = {x["source_seq"]: x["target_kbps"] for x in old_report["encoder_restarts"]}
    new = {x["source_seq"]: x["target_kbps"] for x in new_report["encoder_restarts"]}
    boundary = min((seq for seq in old.keys() | new.keys()
                    if old.get(seq) != new.get(seq)), default=frames)
    if old_packets[:boundary] != new_packets[:boundary]:
        raise RuntimeError("source bytes changed before a different encoder action")
    prefix = lambda trace: tuple(x for x in trace.delivered if x.packet.seq < boundary)
    if prefix(old_trace) != prefix(new_trace):
        raise RuntimeError("a policy changed previously scheduled media arrivals")
    if boundary == frames and old_trace != new_trace:
        raise RuntimeError("identical encoder actions produced different link traces")
    return boundary if boundary < frames else None


def _offered_load(packets, trace, info: dict, *, fps: int, high_kbps: int) -> dict:
    """Offline accounting of every produced unit, not just admitted media."""
    wire_sizes = [len(packet_bytes(p)) for p in packets]
    payload_sizes = [len(p.data) for p in packets]
    rejected = {seq for seq, reason in trace.dropped if reason == "queue_full"}
    rejected_bytes = sum(wire_sizes[seq] for seq in rejected)
    offered = sum(wire_sizes)
    if offered != trace.wire_sent_bytes + rejected_bytes:
        raise RuntimeError("offered/admitted/rejected wire accounting mismatch")
    boundaries = [(0, high_kbps)] + [
        (event["source_seq"], event["target_kbps"])
        for event in info.get("encoder_restarts", []) if event["rate_change"]
    ]
    segments = []
    for index, (start, target) in enumerate(boundaries):
        end = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(packets)
        count = end - start
        payload = sum(payload_sizes[start:end])
        wire = sum(wire_sizes[start:end])
        segments.append({
            "start_seq": start, "end_seq_exclusive": end, "target_kbps": target,
            "source_frames": count, "source_duration_ms": round(count * 1000 / fps, 3),
            "payload_bytes": payload, "framing_bytes": wire - payload,
            "offered_wire_bytes": wire,
            "offered_wire_kbps": round(wire * 8 * fps / (count * 1000), 3),
            "queue_rejected_frames": sum(seq in rejected for seq in range(start, end)),
            "encoded_idr_seqs": [p.seq for p in packets[start:end] if p.keyframe],
        })
    return {
        "offered_wire_bytes": offered,
        "offered_source_kbps": round(offered * 8 * fps / (len(packets) * 1000), 3),
        "offered_framing_bytes": offered - sum(payload_sizes),
        "queue_rejected_wire_bytes": rejected_bytes,
        "target_segments": segments,
        "load_note": "all encoded framed units per source time, including queue admission rejects; "
                     "same-target recovery IDRs remain in the same target segment",
    }


def compare_load_control(*, width: int = 128, height: int = 72,
                         fps: int = 30, frames: int = 90, gop: int = 24,
                         link: LinkConfig | None = None,
                         rates: RateConfig | None = None,
                         load: LoadControlConfig | None = None,
                         recovery: ControllerConfig | None = None,
                         buffers_ms: tuple[int, ...] = (80, 160, 240, 350)) -> dict:
    """Compare wire conversion against an otherwise identical new policy."""
    _validate_video(width, height, fps, frames, gop)
    link = (LinkConfig(delay_ms=40, jitter_ms=8, bandwidth_kbps=0,
                       bandwidth_steps=((0, 1200), (600, 90), (1750, 1200)),
                       max_inflight=10, max_age_ms=350, seed=7)
            if link is None else link)
    rates = RateConfig(silence_timeout_ms=150, max_switches=12) if rates is None else rates
    load = LoadControlConfig() if load is None else load
    recovery = (ControllerConfig(feedback_ms=35, retry_ms=150, max_requests=8)
                if recovery is None else recovery)
    if (not isinstance(link, LinkConfig) or not isinstance(rates, RateConfig)
            or not isinstance(load, LoadControlConfig) or not isinstance(recovery, ControllerConfig)):
        raise ValueError("invalid load-control experiment configuration")
    if rates.max_switches + recovery.max_requests > 20:
        raise ValueError("combined rate and recovery budgets must not exceed 20")
    if (type(buffers_ms) is not tuple or not 1 <= len(buffers_ms) <= 8
            or any(type(x) is not int or not 0 <= x <= 1000 for x in buffers_ms)
            or tuple(sorted(set(buffers_ms))) != buffers_ms):
        raise ValueError("buffer depths must be sorted unique milliseconds")
    age_policy = LoadRateReceiver(rates, load, fps=fps, wire_aware=False)
    wire_policy = LoadRateReceiver(rates, load, fps=fps, wire_aware=True)
    args = dict(width=width, height=height, fps=fps, frames=frames,
                gop=gop, link=link, rates=rates)
    streams = {
        "two_rate_recovery": stream_with_rate_control(**args, adapt=True, recovery=recovery),
        "age_only_recovery": stream_with_rate_control(
            **args, adapt=True, recovery=recovery, rate_policy=age_policy),
        "wire_load_recovery": stream_with_rate_control(
            **args, adapt=True, recovery=recovery, rate_policy=wire_policy),
    }
    legacy = streams["two_rate_recovery"]
    age = streams["age_only_recovery"]
    wire = streams["wire_load_recovery"]
    differences = {
        "two_rate_vs_age_only": _check_action_prefix(legacy, age, frames),
        "two_rate_vs_wire_load": _check_action_prefix(legacy, wire, frames),
        "age_only_vs_wire_load": _check_action_prefix(age, wire, frames),
    }
    binding = sum(bool(x["wire_cap_binding"]) for x in wire[2]["decisions"])
    if not binding and (age[0] != wire[0] or age[1] != wire[1]):
        raise RuntimeError("wire conversion changed the stream without a binding decision")
    restarts = tuple(x["source_seq"] for x in wire[2]["encoder_restarts"])
    streams["matched_restart_high_rate"] = stream_with_rate_control(
        **args, adapt=False, restart_seqs=restarts)
    idrs = lambda packets: [p.seq for p in packets if p.keyframe]
    if idrs(streams["matched_restart_high_rate"][0]) != idrs(wire[0]):
        raise RuntimeError("fixed-high ablation did not match all wire-policy IDRs")
    summaries, common = _summarize_rate_streams(
        streams, width=width, height=height, fps=fps, buffers_ms=buffers_ms)
    for name, (packets, trace, info) in streams.items():
        summaries[name].update(_offered_load(
            packets, trace, info, fps=fps, high_kbps=rates.high_kbps))
    return {
        "schema": 1, "mode": "causal-h264-measured-wire-load-control",
        "source": {"width": width, "height": height, "fps": fps,
                   "frames": frames, "gop": gop, "pattern": "moving-deterministic-texture"},
        "link": asdict(link), "rates": asdict(rates), "load_control": asdict(load),
        "recovery": asdict(recovery),
        "controllers": {name: streams[name][2] for name in (
            "two_rate_recovery", "age_only_recovery", "wire_load_recovery")},
        "first_policy_difference_seqs": differences,
        "wire_cap_binding_decisions": binding,
        "restart_control": {
            "restart_seqs": list(restarts), "target_kbps": rates.high_kbps,
            "note": "offline fixed-high ablation conditioned on ALL wire-policy restarts",
        },
        "quality_comparison": {
            "common_frame_seqs": common,
            "note": "RGB PSNR on the same frame IDs decoded by ALL FOUR arms; "
                    "missing frames excluded, so inspect continuity and playout separately",
        },
        **summaries,
        "scope": {
            "time": "real local H.264 codecs; all media, control and playout timestamps are virtual",
            "measurement": "receiver observed delivery is not a guaranteed capacity estimate; "
                           "sender observes actual framed bytes before admission",
            "control": "bounded multiplicative cuts/probes; valid age feedback can impose a measured "
                       "wire budget; silence uses fallback cuts without inventing throughput",
            "ablation": "compare age_only_recovery and wire_load_recovery to isolate wire conversion; "
                        "changes versus two_rate_recovery also include more targets and repeated silence",
            "feedback": "ideal reliable fixed-delay reverse path; no control bandwidth or cancellation",
            "limits": "no sustainable-rate guarantee, packet transport, pacing, Colab/GPU or display; "
                      "minimum target and finite change budget can prevent further reduction",
            "paired_link": "same parameters/seed; admission differences can shift jitter/loss draws; "
                           "zero jitter/random loss is the cleaner comparison",
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
    parser.add_argument("--min-kbps", type=int, default=20)
    parser.add_argument("--congested-age-ms", type=float, default=110)
    parser.add_argument("--recovered-age-ms", type=float, default=65)
    parser.add_argument("--congestion-samples", type=int, default=3)
    parser.add_argument("--recovery-samples", type=int, default=8)
    parser.add_argument("--control-delay-ms", type=float, default=35)
    parser.add_argument("--min-switch-ms", type=float, default=350)
    parser.add_argument("--max-rate-changes", type=int, default=12)
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
        rates = RateConfig(
            high_kbps=args.high_kbps, low_kbps=args.low_kbps,
            congested_age_ms=args.congested_age_ms, recovered_age_ms=args.recovered_age_ms,
            congestion_samples=args.congestion_samples, recovery_samples=args.recovery_samples,
            control_delay_ms=args.control_delay_ms, min_switch_ms=args.min_switch_ms,
            max_switches=args.max_rate_changes, silence_timeout_ms=args.silence_timeout_ms)
        load = LoadControlConfig(min_kbps=args.min_kbps)
        recovery = ControllerConfig(feedback_ms=args.recovery_feedback_ms,
                                    retry_ms=args.recovery_retry_ms,
                                    max_requests=args.max_recovery_requests)
        report = compare_load_control(
            width=args.width, height=args.height, fps=args.fps, frames=args.frames,
            gop=args.gop, link=link, rates=rates, load=load, recovery=recovery,
            buffers_ms=_parse_buffer_list(args.buffers_ms))
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
