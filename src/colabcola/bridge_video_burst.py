"""v30: isolate H.264 encoder burst allowance with the v29 policy held fixed.

Two causal streams use 1000 ms and 100 ms nominal VBV durations. Two offline
crossovers replay each stream's exact target AND restart schedule under the
other encoder setting. Only vbv-bufsize changes; controller, recovery, link
and initial-fill behavior are unchanged. No sender pacer or queue eviction.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from .bridge_video_controller import ControllerConfig
from .bridge_video_feedback import _parse_buffer_list, _validate_video
from .bridge_video_impairment import LinkConfig, parse_bandwidth_steps, parse_drop_seqs
from .bridge_video_load import LoadControlConfig, LoadRateReceiver
from .bridge_video_load_control import _check_action_prefix, _offered_load
from .bridge_video_rate import EncoderBurstConfig, RateConfig, stream_with_rate_control
from .bridge_video_stream import packet_bytes
from .bridge_video_watchdog import _summarize_rate_streams


def _schedule(info: dict) -> tuple[tuple[int, int], ...]:
    return tuple((x["source_seq"], x["target_kbps"]) for x in info["encoder_restarts"])


def _burst_metrics(packets, info: dict, *, fps: int, high_kbps: int,
                   burst: EncoderBurstConfig, windows: tuple[int, ...]) -> dict:
    """Describe source-frame windows; these are not network pacing guarantees."""
    payload = [len(p.data) for p in packets]
    wire = [len(packet_bytes(p)) for p in packets]
    starts = [(0, high_kbps), *_schedule(info)]
    targets = [high_kbps] * len(packets)
    segments = []
    for index, (start, target) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(packets)
        targets[start:end] = [target] * (end - start)
        configured_kbits = burst.buffer_kbits(target, fps)
        initial_windows = []
        for requested in windows:
            count = min(requested, end - start)
            actual_payload = sum(payload[start:start + count])
            actual_wire = sum(wire[start:start + count])
            duration_ms = count * 1000.0 / fps
            initial_windows.append({
                "requested_frames": requested, "measured_frames": count,
                "complete": count == requested, "duration_ms": round(duration_ms, 3),
                "payload_bytes": actual_payload, "wire_bytes": actual_wire,
                "payload_kbps": round(actual_payload * 8 / duration_ms, 3),
                "offered_wire_kbps": round(actual_wire * 8 / duration_ms, 3),
                "payload_to_target_ratio": round(actual_payload * 8 / (duration_ms * target), 5),
            })
        segments.append({
            "start_seq": start, "end_seq_exclusive": end, "target_kbps": target,
            "requested_vbv_ms": burst.vbv_ms,
            "configured_vbv_kbits": configured_kbits,
            "configured_vbv_equivalent_ms": round(configured_kbits * 1000 / target, 3),
            "initial_idr_payload_bytes": payload[start],
            "initial_idr_wire_bytes": wire[start],
            "initial_windows": initial_windows,
        })
    peaks = []
    for count in windows:
        duration_ms = count * 1000.0 / fps
        if count > len(packets):
            peaks.append({"window_frames": count, "duration_ms": round(duration_ms, 3),
                          "full_windows": 0, "peak": None})
            continue
        start = max(range(len(packets) - count + 1), key=lambda seq: sum(wire[seq:seq + count]))
        end = start + count
        actual_wire = sum(wire[start:end])
        peaks.append({
            "window_frames": count, "duration_ms": round(duration_ms, 3),
            "full_windows": len(packets) - count + 1,
            "peak": {
                "start_seq": start, "end_seq_exclusive": end,
                "wire_bytes": actual_wire,
                "offered_wire_kbps": round(actual_wire * 8 / duration_ms, 3),
                "mean_target_kbps": round(sum(targets[start:end]) / count, 3),
                "framing_kbps": round((actual_wire - sum(payload[start:end])) * 8 / duration_ms, 3),
            },
        })
    return {
        "encoder_segments": segments,
        "peak_source_windows": peaks,
        "encoded_idr_payload_bytes": sum(payload[p.seq] for p in packets if p.keyframe),
        "encoded_idr_wire_bytes": sum(wire[p.seq] for p in packets if p.keyframe),
        "burst_note": "windows measure offered framed bytes per source-frame duration, not instantaneous "
                      "network throughput; initial windows report truncation at every encoder restart; "
                      "VBV fields are requested settings after wrapper rounding/floors, not internal occupancy",
    }


def compare_encoder_bursts(*, width: int = 128, height: int = 72,
                           fps: int = 30, frames: int = 90, gop: int = 24,
                           link: LinkConfig | None = None,
                           rates: RateConfig | None = None,
                           load: LoadControlConfig | None = None,
                           recovery: ControllerConfig | None = None,
                           tight: EncoderBurstConfig | None = None,
                           burst_frames: tuple[int, ...] = (1, 3, 9),
                           buffers_ms: tuple[int, ...] = (80, 160, 240, 350)) -> dict:
    """A live comparison plus two explicitly offline same-action crossovers."""
    _validate_video(width, height, fps, frames, gop)
    link = (LinkConfig(delay_ms=40, jitter_ms=8, bandwidth_kbps=0,
                       bandwidth_steps=((0, 1200), (600, 90), (1750, 1200)),
                       max_inflight=10, max_age_ms=350, seed=7)
            if link is None else link)
    rates = RateConfig(silence_timeout_ms=150, max_switches=12) if rates is None else rates
    load = LoadControlConfig() if load is None else load
    recovery = (ControllerConfig(feedback_ms=35, retry_ms=150, max_requests=8)
                if recovery is None else recovery)
    tight = EncoderBurstConfig(vbv_ms=100) if tight is None else tight
    legacy = EncoderBurstConfig(vbv_ms=1000)
    if (not isinstance(link, LinkConfig) or not isinstance(rates, RateConfig)
            or not isinstance(load, LoadControlConfig) or not isinstance(recovery, ControllerConfig)
            or not isinstance(tight, EncoderBurstConfig)):
        raise ValueError("invalid burst experiment configuration")
    if rates.max_switches + recovery.max_requests > 20:
        raise ValueError("combined rate and recovery budgets must not exceed 20")
    for name, values, maximum, size in (
        ("burst windows", burst_frames, 60, 6), ("buffer depths", buffers_ms, 1000, 8),
    ):
        lower = 1 if name == "burst windows" else 0
        if (type(values) is not tuple or not 1 <= len(values) <= size
                or any(type(x) is not int or not lower <= x <= maximum for x in values)
                or tuple(sorted(set(values))) != values):
            raise ValueError(f"{name} must be sorted unique bounded integers")
    args = dict(width=width, height=height, fps=fps, frames=frames,
                gop=gop, link=link, rates=rates)
    streams = {}
    for name, burst in (("legacy_live", legacy), ("tight_live", tight)):
        streams[name] = stream_with_rate_control(
            **args, adapt=True, recovery=recovery, burst=burst,
            rate_policy=LoadRateReceiver(rates, load, fps=fps, wire_aware=True))
    legacy_schedule = _schedule(streams["legacy_live"][2])
    tight_schedule = _schedule(streams["tight_live"][2])
    streams["tight_replay_legacy_schedule"] = stream_with_rate_control(
        **args, adapt=False, burst=tight, replay_schedule=legacy_schedule)
    streams["legacy_replay_tight_schedule"] = stream_with_rate_control(
        **args, adapt=False, burst=legacy, replay_schedule=tight_schedule)
    pairs = (("legacy_live", "tight_replay_legacy_schedule"),
             ("tight_live", "legacy_replay_tight_schedule"))
    for live, replay in pairs:
        if (_schedule(streams[live][2]) != _schedule(streams[replay][2])
                or [p.seq for p in streams[live][0] if p.keyframe]
                != [p.seq for p in streams[replay][0] if p.keyframe]):
            raise RuntimeError("offline crossover changed the target or complete IDR schedule")
    # Prefix equality applies only when encoder settings are the SAME.
    # Different VBV settings may change source bytes from frame zero.
    legacy_boundary = _check_action_prefix(
        streams["legacy_live"], streams["legacy_replay_tight_schedule"], frames)
    tight_boundary = _check_action_prefix(
        streams["tight_live"], streams["tight_replay_legacy_schedule"], frames)
    if legacy_boundary != tight_boundary:
        raise RuntimeError("crossover schedules disagree about first action difference")
    summaries, common = _summarize_rate_streams(
        streams, width=width, height=height, fps=fps, buffers_ms=buffers_ms)
    for name, (packets, trace, info) in streams.items():
        burst = tight if name.startswith("tight") else legacy
        summaries[name].update(_offered_load(packets, trace, info, fps=fps, high_kbps=rates.high_kbps))
        summaries[name].update(_burst_metrics(
            packets, info, fps=fps, high_kbps=rates.high_kbps, burst=burst, windows=burst_frames))
    return {
        "schema": 1, "mode": "causal-h264-encoder-burst-comparison",
        "source": {"width": width, "height": height, "fps": fps, "frames": frames,
                   "gop": gop, "pattern": "moving-deterministic-texture"},
        "link": asdict(link), "rates": asdict(rates), "load_control": asdict(load),
        "recovery": asdict(recovery),
        "encoder_settings": {"legacy": asdict(legacy), "tight": asdict(tight),
                             "varied": "vbv-bufsize only; initial fill and all other options unchanged"},
        "burst_windows_frames": list(burst_frames),
        "controllers": {name: streams[name][2] for name in ("legacy_live", "tight_live")},
        "first_controller_action_difference_seq": legacy_boundary,
        "offline_controls": {
            "legacy_schedule": [list(x) for x in legacy_schedule],
            "tight_schedule": [list(x) for x in tight_schedule],
            "same_schedule_pairs": [list(x) for x in pairs],
            "note": "both replays are offline ablations conditioned on completed live runs; "
                    "every action restarts the encoder, including unchanged-rate IDRs",
        },
        "quality_comparison": {"common_frame_seqs": common,
                               "note": "matched RGB PSNR uses the same frame IDs decoded by ALL FOUR arms; "
                                       "lost frames excluded, so compare continuity and deadlines separately"},
        **summaries,
        "scope": {
            "time": "real local H.264 codecs; link, feedback and playout timestamps are virtual",
            "controller": "same v29 wire-load policy and recovery configuration; "
                          "closed-loop actions may differ because the encoded bytes change",
            "link": "unchanged FIFO link, admission, serialization and age-drop semantics; "
                    "no sender pacer or queue eviction",
            "vbv": "nominal duration rounded up to integer kbits and at least one nominal source frame; "
                   "a codec buffer setting is not a guaranteed per-frame or wire-rate cap",
            "paired_link": "same parameters/seed; different admissions may shift random draws; "
                           "zero jitter/random loss is the cleaner comparison",
            "limits": "bounded synthetic scene, ideal reliable reverse feedback, no physical display, "
                      "network transport, Colab allocation or production congestion-control claim",
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
    parser.add_argument("--vbv-ms", type=int, default=100)
    parser.add_argument("--buffers-ms", default="80,160,240,350")
    args = parser.parse_args(argv)
    try:
        link = LinkConfig(
            delay_ms=args.delay_ms, jitter_ms=args.jitter_ms,
            bandwidth_kbps=args.bandwidth_kbps,
            bandwidth_steps=parse_bandwidth_steps(args.bandwidth_steps),
            loss_percent=args.loss_percent, drop_seqs=parse_drop_seqs(args.drop_seqs),
            max_inflight=args.max_inflight, max_age_ms=args.max_age_ms, seed=args.seed)
        report = compare_encoder_bursts(
            width=args.width, height=args.height, fps=args.fps, frames=args.frames, gop=args.gop,
            link=link, tight=EncoderBurstConfig(vbv_ms=args.vbv_ms),
            buffers_ms=_parse_buffer_list(args.buffers_ms))
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
