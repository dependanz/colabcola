"""v33: identical-source H.264 packet scheduling and bounded reassembly.

A causal v30 source pilot generates real H.264 once. Six offline replays
freeze all its bytes, timestamps and encoder actions while comparing atomic
frame batches with individual custom datagrams. Neither replay receiver nor
link outcomes can influence the frozen source or its local sender.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math

from .bridge_video_controller import ControllerConfig
from .bridge_video_datagram import (
    DATAGRAM_HEADER, AccessUnitReassembler, DatagramConfig, datagram_bytes,
    fragment_unit, parse_datagram,
)
from .bridge_video_feedback import _parse_buffer_list, _validate_video, fixed_deadline_playout
from .bridge_video_impairment import (
    MAX_FRAMES, MAX_WIRE_BYTES, LinkConfig, LinkTrace, ScheduledUnit,
    decode_delivered_h264, parse_bandwidth_steps, parse_drop_seqs,
)
from .bridge_video_load import LoadControlConfig, LoadRateReceiver
from .bridge_video_packet_transport import BytePacketLink, PacketSenderConfig, PacketSenderQueue
from .bridge_video_rate import (
    EncoderBurstConfig, RateConfig, reconstruction_quality, stream_with_rate_control,
)
from .bridge_video_sender_service import SenderServiceConfig
from .bridge_video_stream import PACKET_HEADER, VideoAccessUnit, packet_bytes


ARMS = {
    "batch_immediate": ("batch", "none"),
    "batch_fixed": ("batch", "fixed"),
    "batch_service": ("batch", "queue"),
    "packet_burst": ("packet", "none"),
    "packet_fixed": ("packet", "fixed"),
    "packet_service": ("packet", "queue"),
}


def _prepare_source(packets, datagram: DatagramConfig, fps: int):
    if (type(packets) not in (list, tuple) or not 1 <= len(packets) <= MAX_FRAMES
            or type(fps) is not int or not 1 <= fps <= 60
            or not isinstance(datagram, DatagramConfig)):
        raise ValueError("invalid bounded packet replay source")
    framed = []
    total = 0
    for seq, packet in enumerate(packets):
        stamp = round(seq * 1_000_000_000 / fps)
        if (not isinstance(packet, VideoAccessUnit) or packet.seq != seq
                or packet.pts_ns != stamp or packet.capture_ns != stamp):
            raise ValueError("packet replay requires contiguous original fixed-FPS source clocks")
        wire = packet_bytes(packet)
        count = (len(wire) + datagram.body_capacity - 1) // datagram.body_capacity
        total += len(wire) + count * DATAGRAM_HEADER.size
        if total > MAX_WIRE_BYTES:
            raise ValueError("packetized source exceeds the aggregate wire budget")
        framed.append(wire)
    fragments = tuple(fragment_unit(p, datagram) for p in packets)
    packet_wire = [datagram_bytes(d, datagram) for group in fragments for d in group]
    return fragments, {
        "encoded_source_sha256": hashlib.sha256(b"".join(framed)).hexdigest(),
        "datagram_source_sha256": hashlib.sha256(b"".join(packet_wire)).hexdigest(),
        "encoded_payload_bytes": sum(len(p.data) for p in packets),
        "original_framed_bytes": sum(map(len, framed)),
        "datagram_header_bytes": len(packet_wire) * DATAGRAM_HEADER.size,
        "datagram_wire_bytes": total,
        "datagram_count": len(packet_wire),
        "encoded_idr_seqs": [p.seq for p in packets if p.keyframe],
        "frame_datagram_counts": [len(group) for group in fragments],
        "largest_datagram_bytes": max(map(len, packet_wire)),
    }


def run_packet_replay(packets: list[VideoAccessUnit], *, fps: int, link: LinkConfig,
                      sender: PacketSenderConfig, initial_target_kbps: int = 650,
                      datagram: DatagramConfig | None = None,
                      service: SenderServiceConfig | None = None,
                      max_inflight_bytes: int = 16_384,
                      reassembly_max_age_ms: float | None = None,
                      reassembly_max_bytes: int = 65_536,
                      erase_first_seqs: tuple[int, ...] = ()) -> tuple[LinkTrace, dict]:
    """Replay current captures causally at the sender, then evaluate the link.

    Precomputing dispatches is valid only because this experiment explicitly
    freezes the source and gives the sender no downstream observations. The
    reassembler receives only wire datagrams and their arrival times; source
    equality is checked separately by the evaluator after reconstruction.
    """
    datagram = DatagramConfig() if datagram is None else datagram
    service = SenderServiceConfig() if service is None else service
    if not isinstance(link, LinkConfig):
        raise ValueError("invalid packet replay link")
    age = link.max_age_ms if reassembly_max_age_ms is None else reassembly_max_age_ms
    receiver = AccessUnitReassembler(datagram, fps=fps, max_age_ms=age,
                                     max_buffer_bytes=reassembly_max_bytes)
    byte_link = BytePacketLink(link, datagram, max_inflight_bytes=max_inflight_bytes,
                              erase_first_seqs=erase_first_seqs)
    queue = PacketSenderQueue(sender, datagram, fps=fps,
                              initial_target_kbps=initial_target_kbps, service=service)
    fragments, source = _prepare_source(packets, datagram, fps)
    dispatches = []
    for packet, group in zip(packets, fragments):
        at_ms = packet.seq * 1000.0 / fps
        dispatches.extend(queue.advance(at_ms))
        queue.enqueue(packet, group, at_ms)
        dispatches.extend(queue.advance(at_ms))
    # Every finite event either dispatches data or discards an expired chain.
    events = 0
    while queue.next_event_ms is not None:
        dispatches.extend(queue.advance(queue.next_event_ms))
        events += 1
        if events > source["datagram_count"] + len(packets):
            raise RuntimeError("packet sender tail failed to terminate")
    sender_report = queue.report()

    arrivals = []
    for attempt in dispatches:
        arrivals.extend(byte_link.arrive_by(attempt.dispatch_ms))
        byte_link.send(attempt.datagrams, dispatch_ms=attempt.dispatch_ms,
                       source_ms=attempt.source_ms)
    arrivals.extend(byte_link.arrive_by(math.inf))
    link_report = byte_link.report()
    complete = []
    for item in arrivals:
        fragment = parse_datagram(datagram_bytes(item.fragment, datagram), datagram)
        unit = receiver.receive(fragment, item.arrival_ms)
        if unit is not None:
            if unit != packets[unit.seq]:
                raise RuntimeError("reassembly changed the frozen encoded source")
            complete.append(ScheduledUnit(unit, unit.seq * 1000.0 / fps,
                                          item.arrival_ms, len(packet_bytes(unit))))
    receiver_report = receiver.finish(len(packets), math.inf)
    if [x.packet.seq for x in complete] != sorted(receiver_report["completed_seqs"]):
        raise RuntimeError("FIFO transport changed reconstructed frame ordering")

    attempted_ids = {tuple(row["fragment_id"]) for row in link_report["packets"]}
    local_ids = {tuple(row["fragment_id"]) for row in sender_report["drops"]}
    all_ids = {(d.frame_seq, d.index) for group in fragments for d in group}
    if (attempted_ids & local_ids or attempted_ids | local_ids != all_ids
            or len(attempted_ids) != sender_report["dispatched_packets"]
            or len(local_ids) != sender_report["dropped_packets"]
            or sender_report["pending_packets"] or receiver_report["pending_frames"]):
        raise RuntimeError("source/sender/link fragment partition mismatch")
    accounting = {
        "generated_wire_bytes": source["datagram_wire_bytes"],
        "local_discarded_wire_bytes": sender_report["dropped_wire_bytes"],
        "attempted_wire_bytes": sender_report["dispatched_wire_bytes"],
        "admission_rejected_wire_bytes": link_report["rejected_wire_bytes"],
        "admitted_wire_bytes": link_report["admitted_wire_bytes"],
        "erased_wire_bytes": link_report["erased_wire_bytes"],
        "received_wire_bytes": receiver_report["received_wire_bytes"],
        "reassembled_wire_bytes": receiver_report["completed_wire_bytes"],
        "unusable_received_wire_bytes": receiver_report["dropped_wire_bytes"],
    }
    a = accounting
    if (a["generated_wire_bytes"] != a["local_discarded_wire_bytes"] + a["attempted_wire_bytes"]
            or a["attempted_wire_bytes"] != a["admission_rejected_wire_bytes"] + a["admitted_wire_bytes"]
            or a["admitted_wire_bytes"] != a["erased_wire_bytes"] + a["received_wire_bytes"]
            or a["received_wire_bytes"] != a["reassembled_wire_bytes"] + a["unusable_received_wire_bytes"]
            or a["received_wire_bytes"] != link_report["delivered_wire_bytes"]):
        raise RuntimeError("source/sender/link/reassembly wire conservation mismatch")

    frame_rows = []
    link_by_frame = {seq: [] for seq in range(len(packets))}
    for row in link_report["packets"]:
        link_by_frame[row["source_seq"]].append(row)
    for packet, local, terminal in zip(packets, sender_report["frames"], receiver_report["terminal_frames"]):
        seq = packet.seq
        if local["source_seq"] != seq or terminal["frame_seq"] != seq:
            raise RuntimeError("terminal frame accounting changed source identity")
        records = link_by_frame[seq]
        source_ms = seq * 1000.0 / fps
        completion_ms = terminal["terminal_ms"] if terminal["reason"] == "complete" else None
        frame_rows.append({
            "source_seq": seq, "source_ms": source_ms, "keyframe": packet.keyframe,
            "generated_packets": len(fragments[seq]),
            "local_discarded_packets": local["dropped_packets"],
            "attempted_packets": len(records),
            "admission_rejected_packets": sum(r["status"] == "queue_full" for r in records),
            "erased_packets": sum(r["status"] == "erased" for r in records),
            "received_packets": terminal["received_datagrams"],
            "received_wire_bytes": terminal["received_wire_bytes"],
            "unusable_received_wire_bytes": terminal["unusable_wire_bytes"],
            "local_state": local["state"], "receiver_state": terminal["reason"],
            "receiver_detail": terminal["detail"],
            "last_fragment_dispatch_ms": local["last_dispatch_ms"],
            "last_fragment_dispatch_age_ms": (local["last_dispatch_ms"] - source_ms
                                              if local["last_dispatch_ms"] is not None else None),
            "complete_dispatch_ms": local["complete_dispatch_ms"],
            "complete_dispatch_age_ms": (local["complete_dispatch_ms"] - source_ms
                                         if local["complete_dispatch_ms"] is not None else None),
            "reassembly_completion_ms": completion_ms,
            "reassembly_age_ms": completion_ms - source_ms if completion_ms is not None else None,
        })
    trace = LinkTrace(len(packets), tuple(complete), (), link_report["admitted_wire_bytes"], 0)
    return trace, {
        **source, "source_frames": len(packets), "reassembled_frames": len(complete),
        "unavailable_frames": len(packets) - len(complete),
        "accounting": accounting, "sender": sender_report, "link": link_report,
        "receiver": receiver_report, "frames": frame_rows,
        "accounting_note": "generated = local discard + attempted; attempted = admission rejected + admitted; "
                           "admitted = erased + received; received = reassembled + unusable received. "
                           "Reassembled frames may still be undecodable or late for presentation.",
    }


def compare_packetization(*, width: int = 128, height: int = 72, fps: int = 30,
                          frames: int = 90, gop: int = 24, link: LinkConfig | None = None,
                          datagram: DatagramConfig | None = None,
                          service: SenderServiceConfig | None = None,
                          max_inflight_bytes: int = 16_384,
                          sender_max_pending: int = 3, sender_max_age_ms: float = 100,
                          reassembly_max_age_ms: float | None = None,
                          reassembly_max_bytes: int = 65_536,
                          erase_first_seqs: tuple[int, ...] = (),
                          buffers_ms: tuple[int, ...] = (80, 160, 240, 350)) -> dict:
    _validate_video(width, height, fps, frames, gop)
    link = (LinkConfig(delay_ms=40, jitter_ms=8, bandwidth_kbps=0,
                       bandwidth_steps=((0, 1200), (600, 90), (1750, 1200)),
                       max_inflight=10, max_age_ms=350, seed=7) if link is None else link)
    datagram = DatagramConfig() if datagram is None else datagram
    service = SenderServiceConfig() if service is None else service
    if (type(buffers_ms) is not tuple or not 1 <= len(buffers_ms) <= 8
            or any(type(x) is not int or not 0 <= x <= 1000 for x in buffers_ms)
            or tuple(sorted(set(buffers_ms))) != buffers_ms):
        raise ValueError("buffer depths must be sorted unique bounded milliseconds")
    # Validate controls before invoking the real codec, including ignored legacy drops.
    BytePacketLink(link, datagram, max_inflight_bytes=max_inflight_bytes,
                   erase_first_seqs=erase_first_seqs)
    AccessUnitReassembler(datagram, fps=fps,
                         max_age_ms=link.max_age_ms if reassembly_max_age_ms is None else reassembly_max_age_ms,
                         max_buffer_bytes=reassembly_max_bytes)
    configs = {name: PacketSenderConfig(grouping, pacing, sender_max_pending, sender_max_age_ms)
               for name, (grouping, pacing) in ARMS.items()}
    rates = RateConfig(silence_timeout_ms=150, max_switches=12)
    load = LoadControlConfig()
    recovery = ControllerConfig(feedback_ms=35, retry_ms=150, max_requests=8)
    burst = EncoderBurstConfig(vbv_ms=100)
    for config in configs.values():
        PacketSenderQueue(config, datagram, fps=fps, initial_target_kbps=rates.high_kbps, service=service)
    packets, pilot_trace, pilot_controller = stream_with_rate_control(
        width=width, height=height, fps=fps, frames=frames, gop=gop, link=link,
        rates=rates, recovery=recovery, burst=burst, adapt=True,
        rate_policy=LoadRateReceiver(rates, load, fps=fps, wire_aware=True))
    summaries = {}
    distortions = {}
    for name, config in configs.items():
        trace, summary = run_packet_replay(
            packets, fps=fps, link=link, sender=config, initial_target_kbps=rates.high_kbps,
            datagram=datagram, service=service, max_inflight_bytes=max_inflight_bytes,
            reassembly_max_age_ms=reassembly_max_age_ms, reassembly_max_bytes=reassembly_max_bytes,
            erase_first_seqs=erase_first_seqs)
        decoded = decode_delivered_h264(trace, width, height)
        quality = reconstruction_quality(trace, width=width, height=height, fps=fps, include_internal=True)
        distortions[name] = quality.pop("_sse_by_seq")
        if quality["quality_decoded_frames"] != decoded["decoded_frames"]:
            raise RuntimeError("continuity and quality decoders disagree")
        summary.update({
            "actually_decoded_frames": decoded["decoded_frames"],
            "decoded_runs": decoded["decoded_runs"], "dependency_skipped": decoded["dependency_skipped"],
            "decoder_idr_resynchronizations": decoded["idr_resynchronizations"],
            "simulated_capture_to_arrival_p50_ms": decoded["simulated_capture_to_arrival_p50_ms"],
            "simulated_capture_to_arrival_max_ms": decoded["simulated_capture_to_arrival_max_ms"],
            "playout": [fixed_deadline_playout(trace, decoded, fps=fps, buffer_ms=b) for b in buffers_ms],
            **quality,
        })
        valid = set(distortions[name])
        for row in summary["frames"]:
            row["actually_decoded"] = row["source_seq"] in valid
        summary["idr_outcomes"] = [dict(row) for row in summary["frames"] if row["keyframe"]]
        summaries[name] = summary
    common = sorted(set.intersection(*(set(sse) for sse in distortions.values())))
    for name, summary in summaries.items():
        if any(distortions[name][seq] != distortions["batch_immediate"][seq] for seq in common):
            raise RuntimeError("identical source bytes changed matched decoded image quality")
        mse = (sum(distortions[name][seq] for seq in common) / (len(common) * width * height * 3)
               if common else None)
        summary["matched_frame_count"] = len(common)
        summary["matched_frame_rgb_psnr_db"] = (
            round(10 * math.log10(255 * 255 / mse), 3) if mse is not None and mse > 0 else None)
    for field in ("encoded_source_sha256", "datagram_source_sha256", "encoded_idr_seqs"):
        if any(row[field] != summaries["batch_immediate"][field] for row in summaries.values()):
            raise RuntimeError("packetization arms changed source bytes, headers or IDRs")
    return {
        "schema": 1, "mode": "frozen-h264-packetization-comparison",
        "source": {"width": width, "height": height, "fps": fps, "frames": frames,
                   "gop": gop, "pattern": "moving-deterministic-texture"},
        "source_pilot": {
            "kind": "v30-tight-immediate-causal-reference", "link": asdict(link),
            "rates": asdict(rates), "load_control": asdict(load), "recovery": asdict(recovery),
            "encoder_burst": asdict(burst), "controller": pilot_controller,
            "delivered_frames": len(pilot_trace.delivered), "wire_sent_bytes": pilot_trace.wire_sent_bytes,
            "note": "source provenance only; its historical frame-slot link differs from every byte-bounded replay",
        },
        "datagram_config": {**asdict(datagram), "header_bytes": DATAGRAM_HEADER.size,
                            "original_access_unit_header_bytes": PACKET_HEADER.size},
        "link_max_inflight_bytes": max_inflight_bytes,
        "arm_configs": {name: asdict(config) for name, config in configs.items()},
        "quality_comparison": {"common_frame_seqs": common,
                               "note": "same encoded source in all SIX arms; matched RGB quality excludes "
                                       "missing frames and uses only source IDs decoded by all six arms"},
        "arms": summaries,
        "scope": {
            "clock": "real local H.264; virtual source, packet dispatch, serialization, arrival and playout",
            "source": "all replays freeze every pilot byte, timestamp and encoder action; no new feedback or repair",
            "headers": "all arms charge identical CCP1 custom datagrams including every 16-byte fragment header",
            "service": "configured independent allowance retains initial target plus nominal 30-byte AU framing; "
                       "fragment headers are fully charged but do not increase that inherited base; not known capacity",
            "rejection": "sender schedule is computed without admission or erasure input; rejected fragments "
                         "do not abort later attempts for the same frame",
            "granularity": "batch arms admit and serialize all frame datagrams atomically; packet arms do so individually",
            "attribution": "batch_fixed versus packet_fixed isolates granularity under fixed service; "
                           "batch_service versus packet_service also changes service refresh cadence",
            "receiver": "only complete validated exact access units enter H.264; no partial decode or concealment; "
                        "never-observed frames are evaluator-known missing, not receiver-detected timeouts",
            "bounds": "local bounds count frames with unsent data; receiver byte bound counts stored fragment bodies; "
                      "source clocks and full presentation horizon are preserved",
            "limits": "bounded synthetic frozen-source experiment; no RTP/FU-A interoperability, UDP/IP accounting, "
                      "retransmission, FEC, physical network/display, Colab allocation or transport feedback controller",
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
    parser.add_argument("--pilot-max-inflight", type=int, default=10)
    parser.add_argument("--max-age-ms", type=float, default=350)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-datagram-bytes", type=int, default=1200)
    parser.add_argument("--link-max-inflight-bytes", type=int, default=16_384)
    parser.add_argument("--sender-max-pending", type=int, default=3)
    parser.add_argument("--sender-max-age-ms", type=float, default=100)
    parser.add_argument("--reassembly-max-age-ms", type=float, default=None)
    parser.add_argument("--reassembly-max-bytes", type=int, default=65_536)
    parser.add_argument("--erase-first-seqs", default="")
    parser.add_argument("--drain-ms", type=float, default=50)
    parser.add_argument("--service-cap-factor", type=float, default=2)
    parser.add_argument("--buffers-ms", default="80,160,240,350")
    args = parser.parse_args(argv)
    try:
        link = LinkConfig(delay_ms=args.delay_ms, jitter_ms=args.jitter_ms,
                          bandwidth_kbps=args.bandwidth_kbps,
                          bandwidth_steps=parse_bandwidth_steps(args.bandwidth_steps),
                          loss_percent=args.loss_percent, max_inflight=args.pilot_max_inflight,
                          max_age_ms=args.max_age_ms, seed=args.seed)
        report = compare_packetization(
            width=args.width, height=args.height, fps=args.fps, frames=args.frames, gop=args.gop,
            link=link, datagram=DatagramConfig(args.max_datagram_bytes),
            service=SenderServiceConfig(drain_ms=args.drain_ms, max_rate_factor=args.service_cap_factor),
            max_inflight_bytes=args.link_max_inflight_bytes,
            sender_max_pending=args.sender_max_pending, sender_max_age_ms=args.sender_max_age_ms,
            reassembly_max_age_ms=args.reassembly_max_age_ms, reassembly_max_bytes=args.reassembly_max_bytes,
            erase_first_seqs=parse_drop_seqs(args.erase_first_seqs),
            buffers_ms=_parse_buffer_list(args.buffers_ms))
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
