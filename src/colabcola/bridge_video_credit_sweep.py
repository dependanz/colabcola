"""v33k: offline-first 1 -> 2 -> 1 credit sweep on ONE owned H.264 Comm.

This worker never allocates compute or creates transport connections. Its caller
owns the authenticated kernel Comm and T4 lifecycle. A single encoder on the
remote side, a single host decoder, and a single ordered receive window persist
across all phases; only the *host request-credit cap* changes.
"""
from __future__ import annotations

import json
import queue
from statistics import median
import time

from .bridge_colab import offer_update
from .bridge_video_comm_pipeline import KernelH264PullWindow
from .bridge_video_comm_session import WIDTH, HEIGHT, MAX_NO_FRESH_SECONDS, make_host_decoder
from .lifecycle_io import LifecycleError

PHASES = (("A", 1), ("B", 2), ("A_repeat", 1))
POLL_WAIT_SECONDS = .03


def _summarize_phase(name: str, cap: int, stats: dict, *, measured_s: float,
                     end_gap_ms: float) -> dict:
    """Never mix phase drain/warmup work into scored samples."""
    assert measured_s > 0
    samples = stats["fresh_gaps_ms"] + [end_gap_ms]
    return {
        "name": name,
        "credits": cap,
        "measurement_seconds": measured_s,
        "decoded_frames": stats["decoded"],
        "decoded_fps": round(stats["decoded"] / measured_s, 3),
        "committed_replies": stats["replies"],
        "suppressed_dependent_units": stats["suppressed"],
        "max_outstanding": stats["max_outstanding"],
        "h264_kib_s": round(stats["h264_bytes"] / measured_s / 1024, 3),
        "ccv1_kib_s": round(stats["ccv1_bytes"] / measured_s / 1024, 3),
        "base64_kib_s": round(stats["base64_chars"] / measured_s / 1024, 3),
        "pull_rtt_p50_ms": (round(median(stats["rtts"]), 2) if stats["rtts"] else None),
        "decode_rgb_p50_us": (round(median(stats["decode_us"]), 2)
                               if stats["decode_us"] else None),
        "input_ack_observed_p50_ms": (round(median(stats["acks_ms"]), 2)
                                      if stats["acks_ms"] else None),
        "longest_no_new_decoded_frame_ms": round(max(samples) if samples else 0, 2),
    }


def _empty_phase() -> dict:
    return {"decoded": 0, "replies": 0, "suppressed": 0,
            "max_outstanding": 0, "h264_bytes": 0, "ccv1_bytes": 0,
            "base64_chars": 0, "rtts": [], "decode_us": [],
            "acks_ms": [], "fresh_gaps_ms": []}


def run_h264_credit_sweep(peer, gpu: str, outgoing: queue.Queue,
                          frames: queue.Queue, stopped, deadline: float,
                          poll_hz: int, *, phase_seconds: int = 16,
                          warmup_seconds: int = 2) -> dict:
    """Controlled 1/2/1 credit A/B/A' experiment without reconnect or replay.

    A phase stops *submitting* requests at its deadline, drains every already
    outstanding reply (including out-of-order buffered replies), then starts
    the next phase with the same decoder and same sequence gate. Warmup and
    draining are excluded from scored phase metrics. No input-to-photon or
    cross-machine capture-to-decode measurement is claimed.
    """
    if (type(poll_hz) is not int or not 1 <= poll_hz <= 12
            or type(phase_seconds) is not int or not 6 <= phase_seconds <= 40
            or type(warmup_seconds) is not int
            or not 1 <= warmup_seconds <= min(5, phase_seconds - 3)):
        raise ValueError("invalid H.264 credit sweep timing")
    if getattr(peer, "video_codec", None) != "h264":
        raise LifecycleError("H.264 credit sweep requires opted-in video Comm")
    if not isinstance(deadline, (float, int)) or isinstance(deadline, bool):
        raise ValueError("invalid sweep deadline")

    av, decoder = make_host_decoder()
    window = KernelH264PullWindow(peer, credits=2)
    header = b"P6\n640 360\n255\n"
    input_seq = 0
    last_ack = 0
    input_sent_at: dict[int, float] = {}
    next_pull = 1
    next_due = time.monotonic()
    last_fresh_global = next_due
    results = []
    print("[colabcola] H.264 single-runtime 1/2/1 credit sweep started", flush=True)

    try:
        for phase_name, credit_cap in PHASES:
            phase_start = time.monotonic()
            phase_end = phase_start + phase_seconds
            scored_start = phase_start + warmup_seconds
            stats = _empty_phase()
            last_scored_frame = scored_start
            drain = False
            offer_update(frames, ("status", f"H.264 sweep {phase_name}: {credit_cap} credit(s)"))

            while not stopped.is_set():
                now = time.monotonic()
                if now >= deadline:
                    raise LifecycleError("H.264 credit sweep time budget reached")

                # The viewer's input sequence never resets at phase boundaries.
                latest = None
                while True:
                    try:
                        latest = outgoing.get_nowait()
                    except queue.Empty:
                        break
                if latest is not None:
                    input_seq += 1
                    input_sent_at[input_seq] = time.monotonic()
                    peer.send_input(input_seq, latest)
                    # Bound retained input metadata even if ACKs stall.
                    if len(input_sent_at) > 128:
                        for seq in sorted(input_sent_at)[:-128]:
                            del input_sent_at[seq]

                now = time.monotonic()
                if now >= phase_end:
                    drain = True
                if not drain and now >= next_due and window.outstanding < credit_cap:
                    window.submit(next_pull)
                    next_pull += 1
                    next_due = time.monotonic() + 1 / poll_hz  # never catch up

                if window.outstanding:
                    committed = window.poll(POLL_WAIT_SECONDS)
                else:
                    committed = []
                    if drain:
                        break  # all pending and out-of-order replies settled
                    time.sleep(min(.02, max(.001, min(next_due, phase_end) - time.monotonic())))

                for unit, ack, sizes, rtt in committed:
                    observed = time.monotonic()
                    scored = scored_start <= observed < phase_end
                    if scored:
                        stats["replies"] += 1
                        raw, framed, base64_count = sizes
                        stats["h264_bytes"] += raw
                        stats["ccv1_bytes"] += framed
                        stats["base64_chars"] += base64_count
                        stats["rtts"].append(rtt)
                        if unit is None:
                            stats["suppressed"] += 1

                    if ack > last_ack:
                        if scored and ack in input_sent_at:
                            stats["acks_ms"].append((observed - input_sent_at[ack]) * 1000)
                        last_ack = ack
                        for seq in list(input_sent_at):
                            if seq <= ack:
                                del input_sent_at[seq]

                    if unit is None:
                        continue
                    started_ns = time.perf_counter_ns()
                    try:
                        packet = av.Packet(unit.data)
                        packet.pts = unit.seq
                        images = decoder.decode(packet)
                        if len(images) > 1:
                            raise LifecycleError("credit sweep H.264 decoder emitted extra frames")
                        for image in images:
                            if image.width != WIDTH or image.height != HEIGHT:
                                raise LifecycleError("credit sweep decoded dimensions invalid")
                            pixels = image.to_ndarray(format="rgb24").tobytes()
                            if len(pixels) != WIDTH * HEIGHT * 3:
                                raise LifecycleError("credit sweep decoded RGB size invalid")
                            offer_update(frames, ("frame", header + pixels))
                            last_fresh_global = time.monotonic()
                            if scored and scored_start <= last_fresh_global < phase_end:
                                stats["decoded"] += 1
                                stats["fresh_gaps_ms"].append(
                                    (last_fresh_global - last_scored_frame) * 1000)
                                last_scored_frame = last_fresh_global
                        if scored:
                            stats["decode_us"].append(
                                (time.perf_counter_ns() - started_ns) / 1000)
                    except LifecycleError:
                        raise
                    except Exception:
                        raise LifecycleError("credit sweep host H.264 decode failed") from None

                now = time.monotonic()
                window.check_timeouts(now=now)
                if now - last_fresh_global > MAX_NO_FRESH_SECONDS:
                    raise LifecycleError("credit sweep H.264 frame freshness watchdog fired")
                if scored_start <= now < phase_end:
                    stats["max_outstanding"] = max(
                        stats["max_outstanding"], window.outstanding)

            if stopped.is_set():
                raise LifecycleError("H.264 credit sweep interrupted by viewer shutdown")
            # The phase cutoff, not drain completion, ends measurement.
            end_gap = max(0, (phase_end - last_scored_frame) * 1000)
            result = _summarize_phase(
                phase_name, credit_cap, stats,
                measured_s=phase_seconds - warmup_seconds,
                end_gap_ms=end_gap)
            results.append(result)
            print("[colabcola] H.264 sweep phase " + json.dumps(result, sort_keys=True), flush=True)
            next_due = time.monotonic()  # new phase; no catch-up burst

        report = {"schema": 1, "mode": "single-owned-comm-h264-credit-aba",
                  "phases": results,
                  "request_count": next_pull - 1,
                  "maximum_window_outstanding": window.maximum_outstanding,
                  "continuous_decoder_and_sequence_gate": True,
                  "all_phase_boundaries_drained": True,
                  "scope": "one owned Jupyter Comm; host request/receive/decode and Tk configuration, not network bytes or input-to-photon latency"}
        print("[colabcola] H.264 sweep result " + json.dumps(report, sort_keys=True), flush=True)
        return report
    except Exception as exc:
        print("[colabcola] H.264 credit sweep stopped: " + type(exc).__name__, flush=True)
        raise LifecycleError("H.264 credit sweep interrupted") from None
    finally:
        stopped.set()
