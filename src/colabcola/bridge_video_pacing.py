"""v31: causal whole-access-unit pacing and a bounded local sender queue.

All arms retain the v29 rate/recovery policy and v30 tight encoder setting.
The sender sees only generated units, applied targets and elapsed time.
Local expiry never removes bytes already dispatched to the virtual link.
An exact-packet offline replay separates source changes from sender handling.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from statistics import median

from .bridge_video_controller import ControllerConfig, MultiGapFeedbackController, VirtualAccessUnitLink
from .bridge_video_feedback import _parse_buffer_list, _validate_video
from .bridge_video_impairment import LinkConfig, LinkTrace, parse_bandwidth_steps, parse_drop_seqs, simulate_link
from .bridge_video_load import LoadControlConfig, LoadRateReceiver
from .bridge_video_rate import EncoderBurstConfig, RateConfig, _encoder, _raw_frame, _texture, stream_with_rate_control
from .bridge_video_sender import LocalSenderQueue, SenderQueueConfig
from .bridge_video_sender_service import SenderServiceConfig, ServiceSenderQueue
from .bridge_video_stream import MAX_PACKET, PACKET_HEADER, VideoAccessUnit, is_keyframe, packet_bytes
from .bridge_video_watchdog import _summarize_rate_streams


def _dispatch_record(packet, at_ms: float, pacing_kbps: float, queue_delay_ms: float) -> dict:
    return {"source_seq": packet.seq, "source_ms": packet.capture_ns / 1e6,
            "dispatch_ms": at_ms, "wire_size": len(packet_bytes(packet)),
            "pacing_kbps": pacing_kbps, "queue_delay_ms": queue_delay_ms}


def _immediate_sender(packets, *, fps: int) -> dict:
    """Uniform accounting for the existing immediate and offline replay arms."""
    return {
        "config": asdict(SenderQueueConfig(pacing=False, max_pending=120,
                                          max_age_ms=0, max_repair_requests=0)),
        "dispatches": [_dispatch_record(p, p.seq * 1000.0 / fps, 0.0, 0.0) for p in packets],
        "drops": [], "requests": [], "applied": [], "cancelled": [],
        "need_idr": False, "note": "immediate dispatch; no local queue or pacing budget",
    }


def stream_with_sender_queue(*, width: int, height: int, fps: int, frames: int,
                             gop: int, link: LinkConfig, rates: RateConfig,
                             load: LoadControlConfig, recovery: ControllerConfig,
                             burst: EncoderBurstConfig, sender: SenderQueueConfig,
                             service: SenderServiceConfig | None = None) -> tuple[list, LinkTrace, dict]:
    """Capture at fixed source times; dispatch queued units on a separate clock.

    Queued expiry/dispatch events through each capture time run first. Arrivals
    through that time then reach the unchanged receiver policy. Encoder changes
    require feedback/local repair strictly before capture. Newly captured units
    can dispatch at that same time if eligible. Tail dispatch/expiry continues
    after the last capture, but no further encoder actions are applied.
    """
    _validate_video(width, height, fps, frames, gop)
    if (not isinstance(link, LinkConfig) or not isinstance(rates, RateConfig)
            or not isinstance(load, LoadControlConfig) or not isinstance(recovery, ControllerConfig)
            or not isinstance(burst, EncoderBurstConfig) or not isinstance(sender, SenderQueueConfig)):
        raise ValueError("invalid sender experiment configuration")
    if rates.max_switches + recovery.max_requests > 20:
        raise ValueError("existing rate and recovery budgets must not exceed 20")
    if service is not None and not isinstance(service, SenderServiceConfig):
        raise ValueError("invalid sender service configuration")
    if service is not None and not sender.pacing:
        raise ValueError("sender service requires pacing")
    try:
        import av
        import numpy as np
    except ImportError:
        raise RuntimeError("sender experiment requires optional PyAV and NumPy") from None

    texture = _texture(np, width, height)
    receiver = LoadRateReceiver(rates, load, fps=fps, wire_aware=True)
    recovery_state = MultiGapFeedbackController(recovery)
    local = (LocalSenderQueue(sender, fps=fps, initial_target_kbps=rates.high_kbps)
             if service is None else ServiceSenderQueue(
                 sender, fps=fps, initial_target_kbps=rates.high_kbps, service=service))
    media = VirtualAccessUnitLink(link)
    waiting_arrivals = []
    dispatches = []
    packets = []
    restarts = []
    kbps = rates.high_kbps
    encoder = _encoder(av, width, height, fps, gop, kbps, burst=burst)
    origin_seq = 0

    def dispatch(events):
        for event in events:
            # The transport drains due arrivals to manage its own slots. These
            # observations do not feed the sender queue or refund pacing debt.
            waiting_arrivals.extend(media.arrive_by(event.dispatch_ms))
            media.send(event.packet, fps=fps, dispatch_ms=event.dispatch_ms)
            dispatches.append(_dispatch_record(
                event.packet, event.dispatch_ms, event.pacing_kbps, event.queue_delay_ms))

    def observe_waiting():
        for item in waiting_arrivals:
            receiver.observe(item)
            recovery_state.receive(item.packet, item.arrival_ms)
        waiting_arrivals.clear()

    for seq in range(frames):
        now = seq * 1000.0 / fps
        dispatch(local.advance(now))
        waiting_arrivals.extend(media.arrive_by(now))
        observe_waiting()
        receiver.advance(now)
        target = receiver.apply_due(seq, now, kbps)
        due_recovery = recovery_state.requests_due_before(now)
        due_local = local.take_repair_before(now, seq)
        if target != kbps or due_recovery or due_local is not None:
            if encoder.encode(None):
                raise RuntimeError("unexpected H.264 buffer during sender restart")
            restarts.append({
                "source_seq": seq, "source_ms": round(now, 3),
                "previous_kbps": kbps, "target_kbps": target,
                "rate_change": target != kbps,
                "recovery_request_ids": [r["request_id"] for r in due_recovery],
                "local_repair_request_ids": [] if due_local is None else [due_local["request_id"]],
            })
            encoder = _encoder(av, width, height, fps, gop, target, burst=burst)
            origin_seq = seq
            kbps = target
            recovery_state.record_applied(due_recovery, seq)
        local.set_target(kbps, now)
        frame = av.VideoFrame.from_ndarray(_raw_frame(np, texture, seq), format="rgb24")
        frame.pts = seq - origin_seq
        encoded = encoder.encode(frame)
        if len(encoded) != 1:
            raise RuntimeError("low-delay sender encoder did not emit one access unit")
        unit = encoded[0]
        data = bytes(unit)
        if not 0 < len(data) <= MAX_PACKET:
            raise RuntimeError("sender access unit exceeds packet limit")
        packet = VideoAccessUnit(
            seq=seq, pts_ns=round(seq * 1_000_000_000 / fps),
            capture_ns=round(seq * 1_000_000_000 / fps), data=data,
            keyframe=bool(unit.is_keyframe and is_keyframe(data)))
        if seq == origin_seq and not packet.keyframe:
            raise RuntimeError("sender encoder segment did not begin with verified IDR")
        packets.append(packet)
        # Preserve v29's source observation: every generated frame, at capture,
        # including units that the local sender subsequently discards.
        receiver.observe_encoded(packet, now, kbps)
        local.enqueue(packet, now)
        dispatch(local.advance(now))

    if encoder.encode(None):
        raise RuntimeError("unexpected H.264 buffer at sender source end")
    receiver.stop_watchdog()
    tail_events = 0
    while local.next_event_ms is not None:
        at_ms = local.next_event_ms
        if not math.isfinite(at_ms):
            raise RuntimeError("sender tail event is not finite")
        dispatch(local.advance(at_ms))
        tail_events += 1
        if tail_events > frames * 3:
            raise RuntimeError("sender tail did not terminate within its frame bound")
    if local.pending_count:
        raise RuntimeError("sender retained units without a terminal event")
    waiting_arrivals.extend(media.arrive_by(math.inf))
    observe_waiting()
    physical = media.snapshot(len(dispatches))
    # The source horizon includes locally discarded units. Link loss reasons
    # remain link-only; sender losses are separate, checked terminal outcomes.
    trace = LinkTrace(frames, physical.delivered, physical.dropped,
                      physical.wire_sent_bytes, physical.max_queue_occupancy)
    sender_report = local.report()
    sender_report["dispatches"] = dispatches
    sender_report["last_capture_ms"] = (frames - 1) * 1000.0 / fps
    sender_report["last_dispatch_ms"] = dispatches[-1]["dispatch_ms"] if dispatches else None
    sender_report["last_admitted_arrival_ms"] = media.last_arrival_ms if media.wire_sent_bytes else None
    sender_report["post_source_dispatches"] = sum(
        d["dispatch_ms"] > sender_report["last_capture_ms"] for d in dispatches)
    report = receiver.report()
    report.update({"recovery": recovery_state.report(), "encoder_restarts": restarts,
                   "encoder_burst": asdict(burst), "sender": sender_report})
    _sender_accounting(packets, trace, sender_report, fps=fps)
    return packets, trace, report


def _sender_accounting(packets, trace: LinkTrace, sender: dict, *, fps: int) -> dict:
    """Validate disjoint terminal outcomes before reporting any improvements."""
    wire = {p.seq: len(packet_bytes(p)) for p in packets}
    dispatches, drops = sender["dispatches"], sender["drops"]
    attempts = [x["source_seq"] for x in dispatches]
    local_ids = [x["source_seq"] for x in drops]
    arrived = [x.packet.seq for x in trace.delivered]
    link_drops = [seq for seq, _ in trace.dropped]
    if (len(set(attempts)) != len(attempts) or len(set(local_ids)) != len(local_ids)
            or set(attempts) & set(local_ids) or set(attempts) | set(local_ids) != set(wire)
            or len(set(arrived + link_drops)) != len(arrived + link_drops)
            or set(arrived + link_drops) != set(attempts)
            or trace.source_count != len(packets)):
        raise RuntimeError("sender/source/link frame partition mismatch")
    if any(x["wire_size"] != wire[x["source_seq"]] for x in [*dispatches, *drops]):
        raise RuntimeError("sender byte record mismatch")
    if attempts != sorted(attempts) or any(
            b["dispatch_ms"] < a["dispatch_ms"] for a, b in zip(dispatches, dispatches[1:])):
        raise RuntimeError("sender changed dispatch ordering")
    for item in trace.delivered:
        if not math.isclose(item.source_ms, item.packet.seq * 1000.0 / fps, abs_tol=1e-7):
            raise RuntimeError("dispatch reset a frame's source timestamp")
    rejected_ids = [seq for seq, reason in trace.dropped if reason == "queue_full"]
    encoded_bytes = sum(wire.values())
    local_bytes = sum(wire[seq] for seq in local_ids)
    attempted_bytes = sum(wire[seq] for seq in attempts)
    rejected_bytes = sum(wire[seq] for seq in rejected_ids)
    if (encoded_bytes != local_bytes + attempted_bytes
            or attempted_bytes != trace.wire_sent_bytes + rejected_bytes):
        raise RuntimeError("sender/source/link byte conservation mismatch")
    reasons = {}
    for item in drops:
        reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
    waits = [x["queue_delay_ms"] for x in dispatches]
    peaks = []
    for duration in (100, 300):
        if dispatches:
            windows = [(d["dispatch_ms"], sum(
                x["wire_size"] for x in dispatches
                if d["dispatch_ms"] <= x["dispatch_ms"]
                and x["dispatch_ms"] < d["dispatch_ms"] + duration - 1e-9))
                for d in dispatches]
            start, count = max(windows, key=lambda x: x[1])
            peak = {"start_ms": round(start, 3), "wire_bytes": count,
                    "attempted_wire_kbps": round(count * 8 / duration, 3)}
        else:
            peak = None
        peaks.append({"window_ms": duration, "peak": peak})
    return {
        "encoded_wire_bytes": encoded_bytes,
        "encoded_idr_wire_bytes": sum(wire[p.seq] for p in packets if p.keyframe),
        "encoded_source_sha256": hashlib.sha256(b"".join(packet_bytes(p) for p in packets)).hexdigest(),
        "sender_discarded_frames": len(local_ids), "sender_discarded_wire_bytes": local_bytes,
        "sender_drop_reasons": reasons,
        "dispatch_attempts": len(attempts), "dispatch_attempted_wire_bytes": attempted_bytes,
        "link_admission_rejected_frames": len(rejected_ids),
        "link_admission_rejected_wire_bytes": rejected_bytes,
        "total_dropped_frames": len(local_ids) + len(link_drops),
        "sender_wait_p50_ms": round(median(waits), 3) if waits else None,
        "sender_wait_max_ms": round(max(waits), 3) if waits else None,
        "last_capture_ms": round((len(packets) - 1) * 1000.0 / fps, 3),
        "last_dispatch_ms": round(dispatches[-1]["dispatch_ms"], 3) if dispatches else None,
        "last_delivered_arrival_ms": round(trace.delivered[-1].arrival_ms, 3) if trace.delivered else None,
        "post_source_dispatches": sum(x["dispatch_ms"] > (len(packets) - 1) * 1000.0 / fps for x in dispatches),
        "peak_dispatch_windows": peaks,
        "accounting_note": "encoded = sender discarded + dispatch attempted; dispatch attempted = "
                           "link admitted + link admission rejected; erased/stale admitted units consume wire bytes",
        "dispatch_window_note": "half-open virtual dispatch-time windows anchored at each complete unit; "
                                "1e-9 ms tolerance excludes exact endpoint ties; "
                                "one atomic access unit may exceed the pacing budget of a short window",
    }


def _source_prefix(a, b, frames: int) -> int | None:
    actions = lambda stream: {x["source_seq"]: x["target_kbps"] for x in stream[2]["encoder_restarts"]}
    aa, bb = actions(a), actions(b)
    different = sorted(seq for seq in aa.keys() | bb.keys() if aa.get(seq) != bb.get(seq))
    boundary = different[0] if different else frames
    if a[0][:boundary] != b[0][:boundary]:
        raise RuntimeError("source bytes changed before the first different encoder action")
    return different[0] if different else None


def compare_sender_queues(*, width: int = 128, height: int = 72, fps: int = 30,
                          frames: int = 90, gop: int = 24, link: LinkConfig | None = None,
                          rates: RateConfig | None = None, load: LoadControlConfig | None = None,
                          recovery: ControllerConfig | None = None,
                          burst: EncoderBurstConfig | None = None,
                          sender: SenderQueueConfig | None = None,
                          buffers_ms: tuple[int, ...] = (80, 160, 240, 350)) -> dict:
    _validate_video(width, height, fps, frames, gop)
    link = (LinkConfig(delay_ms=40, jitter_ms=8, bandwidth_kbps=0,
                       bandwidth_steps=((0, 1200), (600, 90), (1750, 1200)),
                       max_inflight=10, max_age_ms=350, seed=7) if link is None else link)
    rates = RateConfig(silence_timeout_ms=150, max_switches=12) if rates is None else rates
    load = LoadControlConfig() if load is None else load
    recovery = ControllerConfig(feedback_ms=35, retry_ms=150, max_requests=8) if recovery is None else recovery
    burst = EncoderBurstConfig(vbv_ms=100) if burst is None else burst
    sender = SenderQueueConfig() if sender is None else sender
    if (not isinstance(link, LinkConfig) or not isinstance(rates, RateConfig)
            or not isinstance(load, LoadControlConfig) or not isinstance(recovery, ControllerConfig)
            or not isinstance(burst, EncoderBurstConfig) or not isinstance(sender, SenderQueueConfig)):
        raise ValueError("invalid sender comparison configuration")
    if not sender.pacing:
        raise ValueError("bounded comparison arm requires pacing")
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
    fifo = SenderQueueConfig(pacing=True, max_pending=120, max_age_ms=0, max_repair_requests=0)
    streams = {
        "immediate_live": immediate,
        "paced_fifo_live": stream_with_sender_queue(**args, load=load, sender=fifo),
        "bounded_paced_live": stream_with_sender_queue(**args, load=load, sender=sender),
    }
    bounded = streams["bounded_paced_live"]
    if not bounded[2]["sender"]["drops"] and (
            bounded[0] != streams["paced_fifo_live"][0]
            or bounded[1] != streams["paced_fifo_live"][1]):
        raise RuntimeError("nonbinding local queue changed the paced stream")
    replay_packets = list(bounded[0])
    replay_trace = simulate_link(replay_packets, fps, link)
    replay_info = {"encoder_restarts": bounded[2]["encoder_restarts"],
                   "sender": _immediate_sender(replay_packets, fps=fps)}
    streams["immediate_replay_bounded_source"] = (replay_packets, replay_trace, replay_info)
    if replay_packets != bounded[0]:
        raise RuntimeError("offline replay changed encoded source units")
    boundaries = {
        "immediate_vs_fifo": _source_prefix(immediate, streams["paced_fifo_live"], frames),
        "immediate_vs_bounded": _source_prefix(immediate, bounded, frames),
        "fifo_vs_bounded": _source_prefix(streams["paced_fifo_live"], bounded, frames),
    }
    summaries, common = _summarize_rate_streams(
        streams, width=width, height=height, fps=fps, buffers_ms=buffers_ms)
    for name, (packets, trace, info) in streams.items():
        summaries[name].update(_sender_accounting(packets, trace, info["sender"], fps=fps))
        local_drops = {d["source_seq"]: d["reason"] for d in info["sender"]["drops"]}
        for idr in summaries[name]["idr_delivery"]:
            if idr["source_seq"] in local_drops:
                if idr["arrival_ms"] is not None or idr["drop_reason"] is not None:
                    raise RuntimeError("IDR assigned multiple terminal outcomes")
                idr["drop_reason"] = local_drops[idr["source_seq"]]
            if idr["arrival_ms"] is None and idr["drop_reason"] is None:
                raise RuntimeError("IDR lacks a terminal outcome")
    if (summaries["bounded_paced_live"]["encoded_source_sha256"]
            != summaries["immediate_replay_bounded_source"]["encoded_source_sha256"]
            or summaries["bounded_paced_live"]["matched_frame_rgb_psnr_db"]
            != summaries["immediate_replay_bounded_source"]["matched_frame_rgb_psnr_db"]):
        raise RuntimeError("exact-source replay changed source bytes or matched decoded quality")
    return {
        "schema": 1, "mode": "causal-h264-sender-queue-comparison",
        "source": {"width": width, "height": height, "fps": fps, "frames": frames,
                   "gop": gop, "pattern": "moving-deterministic-texture"},
        "link": asdict(link), "rates": asdict(rates), "load_control": asdict(load),
        "recovery": asdict(recovery), "encoder_burst": asdict(burst),
        "sender_config": asdict(sender), "fifo_sender_config": asdict(fifo),
        "bounded_queue_ever_dropped": bool(bounded[2]["sender"]["drops"]),
        "pacing_rate": {"formula": "current encoder target + nominal source framing rate",
                        "framing_kbps": PACKET_HEADER.size * 8 * fps / 1000,
                        "idle_credit_bits": 0, "granularity": "one complete access unit"},
        "controllers": {name: {k: v for k, v in streams[name][2].items() if k != "sender"}
                        for name in ("immediate_live", "paced_fifo_live", "bounded_paced_live")},
        "senders": {name: stream[2]["sender"] for name, stream in streams.items()},
        "first_encoder_action_difference_seqs": boundaries,
        "offline_control": {
            "source_arm": "bounded_paced_live", "replay_arm": "immediate_replay_bounded_source",
            "all_source_bytes_and_idrs_identical": True,
            "note": "offline immediate dispatch of ALL generated bounded-run units, including those "
                    "locally discarded; isolates sender handling conditioned on that realized source",
        },
        "quality_comparison": {"common_frame_seqs": common,
                               "note": "matched RGB PSNR uses source IDs decoded by all four arms; "
                                       "lost/late presentation is measured separately"},
        **summaries,
        "scope": {
            "clock": "real local H.264; virtual capture, dispatch, media, feedback and playout clocks",
            "causality": "sender queue uses generated units, elapsed time and applied target only; "
                         "no future capacity, admission, loss or in-flight occupancy input",
            "pacing": "one immediate whole access unit then framed-bit debt; every attempt is charged "
                      "even on link rejection; no packet fragmentation or per-instant rate guarantee",
            "queue": "only unsent local units can expire; descendants suppressed until a verified IDR; "
                     "at most eight configured local repair requests, coalesced with rate/remote recovery",
            "events": "queued events through capture precede receiver decisions; expiry wins dispatch tie; "
                      "repair/feedback at exact capture cannot alter that frame; new units may dispatch at capture",
            "age": "original capture time survives local waiting; full capture horizon used for all playout slots",
            "tail": "watchdog stops at source end; queued units drain/expire at the frozen target; "
                    "tail arrivals observed but no further encoder action applies",
            "attribution": "FIFO versus bounded includes eviction, dependency suppression and local repair; "
                           "exact-source replay isolates their combined sender handling with pacing",
            "paired_link": "same parameters/seed; changed admission can shift jitter/loss draws; zero jitter "
                           "and zero random loss remove this pairing ambiguity",
            "limits": "bounded synthetic source, conservative closed-GOP dependencies, ideal reliable "
                      "reverse feedback; no physical network/display, Colab allocation or capacity guarantee",
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
        report = compare_sender_queues(
            width=args.width, height=args.height, fps=args.fps, frames=args.frames, gop=args.gop,
            link=link, burst=EncoderBurstConfig(vbv_ms=args.vbv_ms), sender=sender,
            buffers_ms=_parse_buffer_list(args.buffers_ms))
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
