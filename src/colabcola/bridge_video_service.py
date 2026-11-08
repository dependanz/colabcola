"""v32: independent sender service allowances and a queue-work drainage objective.

Four live arms retain the same causal encoder/rate/recovery rules: immediate,
v31 target-paced, fixed service and queue-work service. A fifth arm immediately
replays the queue-work source offline. Whole access units and all clocks remain
virtual; service allowances are not estimates of available network capacity.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json

from .bridge_video_controller import ControllerConfig
from .bridge_video_feedback import _parse_buffer_list, _validate_video
from .bridge_video_impairment import LinkConfig, parse_bandwidth_steps, parse_drop_seqs, simulate_link
from .bridge_video_load import LoadControlConfig, LoadRateReceiver
from .bridge_video_pacing import (
    _immediate_sender, _sender_accounting, _source_prefix, stream_with_sender_queue,
)
from .bridge_video_rate import EncoderBurstConfig, RateConfig, stream_with_rate_control
from .bridge_video_sender import SenderQueueConfig
from .bridge_video_sender_service import SenderServiceConfig
from .bridge_video_watchdog import _summarize_rate_streams


LIVE_ARMS = ("immediate_live", "target_paced_live", "fixed_service_live", "queue_service_live")
REPLAY_ARM = "immediate_replay_queue_source"


def compare_sender_service(*, width: int = 128, height: int = 72, fps: int = 30,
                           frames: int = 90, gop: int = 24, link: LinkConfig | None = None,
                           rates: RateConfig | None = None, load: LoadControlConfig | None = None,
                           recovery: ControllerConfig | None = None,
                           burst: EncoderBurstConfig | None = None,
                           sender: SenderQueueConfig | None = None,
                           service: SenderServiceConfig | None = None,
                           buffers_ms: tuple[int, ...] = (80, 160, 240, 350)) -> dict:
    """Keep source-control rules fixed while separating service floor and queue boost.

    The fixed allowance is the initial source ceiling plus nominal framing.
    Queue-work service can exceed that allowance, up to its configured cap,
    using only unpaid debt and locally pending framed bytes. Exact-source
    replay conditions on the queue-work run's realized encoder decisions.
    """
    _validate_video(width, height, fps, frames, gop)
    link = (LinkConfig(delay_ms=40, jitter_ms=8, bandwidth_kbps=0,
                       bandwidth_steps=((0, 1200), (600, 90), (1750, 1200)),
                       max_inflight=10, max_age_ms=350, seed=7) if link is None else link)
    rates = RateConfig(silence_timeout_ms=150, max_switches=12) if rates is None else rates
    load = LoadControlConfig() if load is None else load
    recovery = ControllerConfig(feedback_ms=35, retry_ms=150, max_requests=8) if recovery is None else recovery
    burst = EncoderBurstConfig(vbv_ms=100) if burst is None else burst
    sender = SenderQueueConfig() if sender is None else sender
    service = SenderServiceConfig() if service is None else service
    if (not isinstance(link, LinkConfig) or not isinstance(rates, RateConfig)
            or not isinstance(load, LoadControlConfig) or not isinstance(recovery, ControllerConfig)
            or not isinstance(burst, EncoderBurstConfig) or not isinstance(sender, SenderQueueConfig)
            or not isinstance(service, SenderServiceConfig)):
        raise ValueError("invalid sender service comparison configuration")
    if not sender.pacing:
        raise ValueError("sender service comparison requires pacing")
    if rates.max_switches + recovery.max_requests > 20:
        raise ValueError("existing rate and recovery budgets must not exceed 20")
    if (type(buffers_ms) is not tuple or not 1 <= len(buffers_ms) <= 8
            or any(type(x) is not int or not 0 <= x <= 1000 for x in buffers_ms)
            or tuple(sorted(set(buffers_ms))) != buffers_ms):
        raise ValueError("buffer depths must be sorted unique bounded milliseconds")

    args = dict(width=width, height=height, fps=fps, frames=frames, gop=gop,
                link=link, rates=rates, recovery=recovery, burst=burst)
    immediate = stream_with_rate_control(
        **args, adapt=True, rate_policy=LoadRateReceiver(rates, load, fps=fps, wire_aware=True))
    immediate[2]["sender"] = _immediate_sender(immediate[0], fps=fps)
    fixed = replace(service, queue_boost=False)
    streams = {
        "immediate_live": immediate,
        "target_paced_live": stream_with_sender_queue(**args, load=load, sender=sender),
        "fixed_service_live": stream_with_sender_queue(**args, load=load, sender=sender, service=fixed),
        "queue_service_live": stream_with_sender_queue(**args, load=load, sender=sender, service=service),
    }
    queue_stream = streams["queue_service_live"]
    replay_packets = list(queue_stream[0])
    streams[REPLAY_ARM] = (
        replay_packets, simulate_link(replay_packets, fps, link),
        {"encoder_restarts": queue_stream[2]["encoder_restarts"],
         "sender": _immediate_sender(replay_packets, fps=fps)},
    )
    if replay_packets != queue_stream[0]:
        raise RuntimeError("service replay changed encoded source units")
    service_report = queue_stream[2]["sender"]["service"]
    boost_active = any(row["pacing_kbps"] > service_report["base_kbps"]
                       for row in service_report["decisions"])
    if not boost_active and (queue_stream[0] != streams["fixed_service_live"][0]
                             or queue_stream[1] != streams["fixed_service_live"][1]):
        raise RuntimeError("inactive service boost changed source bytes or the link trace")
    pairs = (("immediate_live", "target_paced_live"),
             ("target_paced_live", "fixed_service_live"),
             ("fixed_service_live", "queue_service_live"),
             ("immediate_live", "queue_service_live"))
    boundaries = {a + "_vs_" + b: _source_prefix(streams[a], streams[b], frames) for a, b in pairs}

    summaries, common = _summarize_rate_streams(
        streams, width=width, height=height, fps=fps, buffers_ms=buffers_ms)
    for name, (packets, trace, info) in streams.items():
        summaries[name].update(_sender_accounting(packets, trace, info["sender"], fps=fps))
        local_drops = {d["source_seq"]: d["reason"] for d in info["sender"]["drops"]}
        for idr in summaries[name]["idr_delivery"]:
            if idr["source_seq"] in local_drops:
                if idr["arrival_ms"] is not None or idr["drop_reason"] is not None:
                    raise RuntimeError("service IDR assigned multiple terminal outcomes")
                idr["drop_reason"] = local_drops[idr["source_seq"]]
            if idr["arrival_ms"] is None and idr["drop_reason"] is None:
                raise RuntimeError("service IDR lacks a terminal outcome")
    original, replay = summaries["queue_service_live"], summaries[REPLAY_ARM]
    for field in ("encoded_source_sha256", "encoded_idr_seqs", "matched_frame_rgb_psnr_db"):
        if original[field] != replay[field]:
            raise RuntimeError("service replay changed source bytes, IDRs or matched decoded quality")
    return {
        "schema": 1, "mode": "causal-h264-sender-service-comparison",
        "source": {"width": width, "height": height, "fps": fps, "frames": frames,
                   "gop": gop, "pattern": "moving-deterministic-texture"},
        "link": asdict(link), "rates": asdict(rates), "load_control": asdict(load),
        "recovery": asdict(recovery), "encoder_burst": asdict(burst),
        "sender_config": asdict(sender), "service_config": asdict(service),
        "fixed_service_config": asdict(fixed),
        "service_budget": {
            "base_kbps": service_report["base_kbps"], "cap_kbps": service_report["cap_kbps"],
            "formula": "min(cap, max(fixed base, (unpaid debt bits + 8 * pending framed bytes) / drain_ms))",
            "fixed_base": "initial encoder ceiling plus nominal source framing rate; unchanged by later targets",
            "interpretation": "configured local service allowance, not a measured capacity or wire-rate guarantee",
            "objective": "rolling work/rate drainage objective including the final unit's debt, assuming the chosen "
                         "rate stayed fixed and no new work arrived; not a completion or per-frame deadline guarantee",
            "updates": "after atomic enqueue, complete expiry chain, or dispatch with full debt charge; held between mutations",
            "queue_boost_ever_active": boost_active,
        },
        "controllers": {name: {k: v for k, v in streams[name][2].items() if k != "sender"}
                        for name in LIVE_ARMS},
        "senders": {name: stream[2]["sender"] for name, stream in streams.items()},
        "first_encoder_action_difference_seqs": boundaries,
        "offline_control": {
            "source_arm": "queue_service_live", "replay_arm": REPLAY_ARM,
            "all_source_bytes_and_idrs_identical": True,
            "note": "offline immediate dispatch of ALL generated queue-service units, including local discards; "
                    "conditions on that realized source and does not isolate queue boost alone",
        },
        "quality_comparison": {
            "common_frame_seqs": common,
            "note": "matched RGB PSNR uses only source IDs decoded by all FIVE arms; "
                    "this cohort can differ from v31; lost/late presentation is measured separately",
        },
        **summaries,
        "scope": {
            "clock": "real local H.264; virtual capture, dispatch, media, feedback and playout clocks",
            "causality": "service sees initial configuration, generated units, unpaid debt, pending bytes and time; "
                         "no future capacity, downstream admission, loss or in-flight occupancy input",
            "pacing": "complete access-unit attempts incur full framed-bit debt even on downstream rejection; "
                      "no idle credit, refunds, packet fragmentation or per-instant rate guarantee",
            "queue": "same unsent count/age/dependency policy and bounded local repairs in all three paced live arms",
            "events": "settle elapsed debt at old service rate; mutate local queue; choose new service; "
                      "ordinary advance/report calls and encoder target changes do not refresh independent service",
            "age": "all queue and presentation ages retain original capture times and full source horizon",
            "tail": "watchdog and source actions stop at final capture; local service may update on actual "
                    "tail dispatch/expiry mutations; no additional encoder actions",
            "attribution": "target-paced versus fixed service changes rate coupling; fixed versus queue service "
                           "measures the complete feedback-coupled boost policy; exact-source replay measures "
                           "combined sender handling conditioned on the queue-service source",
            "paired_link": "same parameters/seed; changed admission can shift jitter/loss draws; "
                           "zero jitter and zero random loss remove this pairing ambiguity",
            "limits": "bounded synthetic source, conservative closed-GOP dependencies, ideal reverse feedback; "
                      "service cap is not known available bandwidth; no physical network/display or Colab allocation",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("width", 128), ("height", 72), ("fps", 30), ("frames", 90), ("gop", 24)):
        parser.add_argument("--" + name, type=int, default=default)
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
    parser.add_argument("--sender-max-pending", type=int, default=3)
    parser.add_argument("--sender-max-age-ms", type=float, default=100)
    parser.add_argument("--sender-max-repair-requests", type=int, default=8)
    parser.add_argument("--drain-ms", type=float, default=50)
    parser.add_argument("--service-cap-factor", type=float, default=2)
    parser.add_argument("--disable-queue-boost", action="store_true")
    parser.add_argument("--buffers-ms", default="80,160,240,350")
    args = parser.parse_args(argv)
    try:
        link = LinkConfig(delay_ms=args.delay_ms, jitter_ms=args.jitter_ms,
                          bandwidth_kbps=args.bandwidth_kbps,
                          bandwidth_steps=parse_bandwidth_steps(args.bandwidth_steps),
                          loss_percent=args.loss_percent, drop_seqs=parse_drop_seqs(args.drop_seqs),
                          max_inflight=args.max_inflight, max_age_ms=args.max_age_ms, seed=args.seed)
        sender = SenderQueueConfig(max_pending=args.sender_max_pending, max_age_ms=args.sender_max_age_ms,
                                   max_repair_requests=args.sender_max_repair_requests)
        service = SenderServiceConfig(queue_boost=not args.disable_queue_boost,
                                      drain_ms=args.drain_ms, max_rate_factor=args.service_cap_factor)
        report = compare_sender_service(
            width=args.width, height=args.height, fps=args.fps, frames=args.frames, gop=args.gop,
            link=link, burst=EncoderBurstConfig(vbv_ms=args.vbv_ms), sender=sender, service=service,
            buffers_ms=_parse_buffer_list(args.buffers_ms))
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
