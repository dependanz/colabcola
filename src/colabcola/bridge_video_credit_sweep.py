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
from .bridge_video_comm_pipeline import (
    KernelH264PullWindow, MAX_EXPERIMENTAL_VIDEO_CREDITS,
)
from .bridge_video_comm_session import WIDTH, HEIGHT, MAX_NO_FRESH_SECONDS, make_host_decoder
from .lifecycle_io import LifecycleError

PHASES = (("A", 1), ("B", 2), ("A_repeat", 1))
HIGH_RATE_PHASES = tuple((f"C{credit}", credit) for credit in (1, 2, 3, 4, 8, 16, 32))
POLL_WAIT_SECONDS = .03
# High-rate capacity mode only: keep the receive-side wait shorter than a
# 30/60 Hz frame period. The 12 Hz A/B/A worker retains its established wait.
EXPERIMENTAL_POLL_WAIT_SECONDS = .004


def receive_poll_timeout(*, experimental: bool, draining: bool,
                         now: float, next_due: float) -> float:
    """Bound a blocking IOPub receive so it cannot monopolize send pacing.

    One worker owns the Comm; it never issues concurrent or repeated requests.
    A short timeout does not declare a reply missing: the receive window owns
    the existing per-request, eight-second fail-closed deadline.
    """
    if not experimental or draining:
        return POLL_WAIT_SECONDS
    return min(EXPERIMENTAL_POLL_WAIT_SECONDS,
               max(.001, next_due - now))


def _record_producer_window(stats: dict, diagnostics: dict | None) -> None:
    """Count each independently generated producer interval once per phase.

    A repeated H.264 pull often sees the same one-second sidecar. Conflicting
    readings for the same interval are excluded without dropping video replies.
    """
    window = diagnostics.get("producer_window") if diagnostics else None
    if window is None:
        stats["producer_missing_replies"] += 1
        return
    wid = window["window_seq"]
    if wid in stats["producer_conflicted_ids"]:
        return
    previous = stats["producer_windows"].get(wid)
    if previous is None:
        stats["producer_windows"][wid] = window
    elif previous != window:
        stats["producer_window_conflicts"] += 1
        del stats["producer_windows"][wid]
        stats["producer_conflicted_ids"].add(wid)


def _summarize_phase(name: str, cap: int, stats: dict, *, measured_s: float,
                     end_gap_ms: float) -> dict:
    """Never mix phase drain/warmup work into scored samples."""
    assert measured_s > 0
    samples = stats["fresh_gaps_ms"] + [end_gap_ms]
    # One sample per independent producer window: NOT one per H.264 reply.
    windows = list(stats["producer_windows"].values())
    def window_median(field):
        return round(median(w[field] for w in windows), 3) if windows else None
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
        # Distinct CUDA source IDs; never conflate H.264 units with fresh renders.
        "distinct_renderer_sequences_in_replies": (
            len(stats["renderer_seqs"]) if stats["source_read_us"] else None),
        "distinct_renderer_reply_fps": (
            round(len(stats["renderer_seqs"]) / measured_s, 3)
            if stats["source_read_us"] else None),
        "duplicate_renderer_replies": stats["duplicate_renderer_replies"],
        "remote_diagnostics_missing": stats["remote_diagnostics_missing"],
        "producer_windows_observed": len(windows),
        "producer_window_metrics_missing_replies": stats["producer_missing_replies"],
        "producer_window_conflicts": stats["producer_window_conflicts"],
        "producer_renderer_fps_p50": window_median("renderer_fps"),
        "producer_renderer_fps_min": (min(w["renderer_fps"] for w in windows)
                                      if windows else None),
        "producer_renderer_fps_max": (max(w["renderer_fps"] for w in windows)
                                      if windows else None),
        "producer_render_copy_p50_ms": window_median("render_copy_ms"),
        "producer_compress_p50_ms": window_median("compress_ms"),
        "producer_publish_p50_ms": window_median("publish_ms"),
        "remote_snapshot_read_p50_us": (round(median(stats["source_read_us"]), 2)
                                        if stats["source_read_us"] else None),
        "remote_encode_p50_us": (round(median(stats["encode_us"]), 2)
                                 if stats["encode_us"] else None),
        "host_pull_submit_interval_p50_ms": (
            round(median(stats["submit_intervals_ms"]), 2)
            if stats["submit_intervals_ms"] else None),
        "host_receive_poll_elapsed_p50_ms": (
            round(median(stats["receive_poll_elapsed_ms"]), 2)
            if stats["receive_poll_elapsed_ms"] else None),
        "host_receive_poll_count": stats["receive_poll_count"],
        "host_receive_polls_without_commit": stats["receive_polls_without_commit"],
    }


def _empty_phase() -> dict:
    return {"decoded": 0, "replies": 0, "suppressed": 0,
            "max_outstanding": 0, "h264_bytes": 0, "ccv1_bytes": 0,
            "base64_chars": 0, "rtts": [], "decode_us": [],
            "acks_ms": [], "fresh_gaps_ms": [],
            "renderer_seqs": set(), "duplicate_renderer_replies": 0,
            "remote_diagnostics_missing": 0, "source_read_us": [],
            "encode_us": [], "submit_intervals_ms": [],
            "last_submit_at": None, "receive_poll_elapsed_ms": [],
            "receive_poll_count": 0, "receive_polls_without_commit": 0,
            "producer_windows": {}, "producer_missing_replies": 0,
            "producer_window_conflicts": 0,
            "producer_conflicted_ids": set()}


def run_h264_credit_sweep(peer, gpu: str, outgoing: queue.Queue,
                          frames: queue.Queue, stopped, deadline: float,
                          poll_hz: int, *, phase_seconds: int = 16,
                          warmup_seconds: int = 2,
                          experimental: bool = False,
                          phases: tuple[tuple[str, int], ...] | None = None) -> dict:
    """Controlled 1/2/1 credit A/B/A' experiment without reconnect or replay.

    A phase stops *submitting* requests at its deadline, drains every already
    outstanding reply (including out-of-order buffered replies), then starts
    the next phase with the same decoder and same sequence gate. Warmup and
    draining are excluded from scored phase metrics. No input-to-photon or
    cross-machine capture-to-decode measurement is claimed.
    """
    if type(experimental) is not bool:
        raise ValueError("invalid experimental sweep selector")
    if phases is None:
        phases = PHASES
    if experimental:
        if (phases != HIGH_RATE_PHASES or type(poll_hz) is not int
                or poll_hz not in (30, 60)):
            raise ValueError("invalid experimental H.264 capacity sweep")
    elif phases != PHASES:
        raise ValueError("custom H.264 phases require experimental opt-in")
    if (type(poll_hz) is not int or not 1 <= poll_hz <= (60 if experimental else 12)
            or type(phase_seconds) is not int or not 6 <= phase_seconds <= 40
            or type(warmup_seconds) is not int
            or not 1 <= warmup_seconds <= min(5, phase_seconds - 3)):
        raise ValueError("invalid H.264 credit sweep timing")
    if getattr(peer, "video_codec", None) != "h264":
        raise LifecycleError("H.264 credit sweep requires opted-in video Comm")
    if not isinstance(deadline, (float, int)) or isinstance(deadline, bool):
        raise ValueError("invalid sweep deadline")

    av, decoder = make_host_decoder()
    window = (KernelH264PullWindow(
        peer, credits=MAX_EXPERIMENTAL_VIDEO_CREDITS, experimental=True)
        if experimental else KernelH264PullWindow(peer, credits=2))
    header = b"P6\n640 360\n255\n"
    input_seq = 0
    last_ack = 0
    input_sent_at: dict[int, float] = {}
    next_pull = 1
    # Windows time.monotonic() can have ~15.625 ms resolution on Python 3.12.
    # Use a single HOST-ONLY high-resolution timebase for experimental pacing
    # and phase statistics. The externally supplied owner deadline and the
    # receiver's eight-second per-request timers remain on time.monotonic().
    phase_clock = ((lambda: time.perf_counter_ns() / 1_000_000_000)
                   if experimental else time.monotonic)
    next_due = phase_clock()
    last_fresh_global = next_due
    results = []
    print("[colabcola] H.264 single-runtime " +
          ("1/2/3/4/8/16/32 capacity" if experimental else "1/2/1 credit") +
          " sweep started", flush=True)

    try:
        for phase_name, credit_cap in phases:
            phase_start = phase_clock()
            phase_end = phase_start + phase_seconds
            scored_start = phase_start + warmup_seconds
            stats = _empty_phase()
            last_scored_frame = scored_start
            drain = False
            offer_update(frames, ("status", f"H.264 sweep {phase_name}: {credit_cap} credit(s)"))

            while not stopped.is_set():
                now = phase_clock()
                if time.monotonic() >= deadline:
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
                    input_sent_at[input_seq] = phase_clock()
                    peer.send_input(input_seq, latest)
                    # Bound retained input metadata even if ACKs stall.
                    if len(input_sent_at) > 128:
                        for seq in sorted(input_sent_at)[:-128]:
                            del input_sent_at[seq]

                now = phase_clock()
                if now >= phase_end:
                    drain = True
                if not drain and now >= next_due and window.outstanding < credit_cap:
                    window.submit(next_pull)
                    submitted_at = phase_clock()
                    if scored_start <= submitted_at < phase_end:
                        previous_submit = stats["last_submit_at"]
                        if previous_submit is not None:
                            stats["submit_intervals_ms"].append(
                                (submitted_at - previous_submit) * 1000)
                        stats["last_submit_at"] = submitted_at
                    next_pull += 1
                    next_due = submitted_at + 1 / poll_hz  # never catch up

                if window.outstanding:
                    poll_started = phase_clock()
                    if experimental and not drain:
                        # Never enter Windows' coarse timed Queue.get wait:
                        # timeout=0 returns an available IOPub message now.
                        # Sleep below, using the finer host clock, to yield CPU.
                        committed = window.poll(0.0)
                    else:
                        wait = receive_poll_timeout(
                            experimental=experimental, draining=drain,
                            now=poll_started, next_due=next_due)
                        committed = window.poll(wait)
                    poll_finished = phase_clock()
                    if scored_start <= poll_finished < phase_end:
                        stats["receive_poll_count"] += 1
                        if not committed:
                            # May include an out-of-order buffered reply.
                            stats["receive_polls_without_commit"] += 1
                        stats["receive_poll_elapsed_ms"].append(
                            (poll_finished - poll_started) * 1000)
                    if experimental and not drain and not committed:
                        # A tiny time.sleep is observed to yield far more
                        # precisely than Queue.get(timeout=small) on Windows.
                        # Always advance even when the credit window is full.
                        time.sleep(.001)
                else:
                    committed = []
                    if drain:
                        break  # all pending and out-of-order replies settled
                    if experimental:
                        time.sleep(min(.002, max(.001, min(next_due, phase_end) - phase_clock())))
                    else:
                        time.sleep(min(.02, max(.001, min(next_due, phase_end) - phase_clock())))

                # Optional, ordered side channel: a legacy/fake window has no
                # diagnostics, while the production window aligns 1:1 to commits.
                diagnostic_batch = getattr(window, "last_committed_diagnostics", None)
                if diagnostic_batch is None:
                    diagnostic_batch = [None] * len(committed)
                if len(diagnostic_batch) != len(committed):
                    raise LifecycleError("H.264 diagnostic/commit alignment invalid")
                for (unit, ack, sizes, rtt), diagnostics in zip(committed, diagnostic_batch):
                    observed = phase_clock()
                    scored = scored_start <= observed < phase_end
                    if scored:
                        stats["replies"] += 1
                        raw, framed, base64_count = sizes
                        stats["h264_bytes"] += raw
                        stats["ccv1_bytes"] += framed
                        stats["base64_chars"] += base64_count
                        stats["rtts"].append(rtt)
                        if diagnostics is None:
                            stats["remote_diagnostics_missing"] += 1
                        else:
                            renderer_seq = diagnostics["renderer_seq"]
                            if renderer_seq in stats["renderer_seqs"]:
                                stats["duplicate_renderer_replies"] += 1
                            else:
                                stats["renderer_seqs"].add(renderer_seq)
                            stats["source_read_us"].append(diagnostics["source_read_us"])
                            stats["encode_us"].append(diagnostics["encode_us"])
                        # Do not count a repeated producer window once per
                        # H.264 pull; this telemetry is independent of pull FPS.
                        _record_producer_window(stats, diagnostics)
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
                            last_fresh_global = phase_clock()
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

                now = phase_clock()
                window.check_timeouts(now=time.monotonic())
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
            next_due = phase_clock()  # new phase; no catch-up burst

        report = {"schema": 1,
                  "mode": ("single-owned-comm-h264-capacity" if experimental
                           else "single-owned-comm-h264-credit-aba"),
                  "requested_poll_hz": poll_hz,
                  "host_phase_clock": ("perf_counter_ns" if experimental
                                       else "monotonic"),
                  "nonblocking_iopub_poll": bool(experimental),
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


def run_h264_capacity_sweep(peer, gpu: str, outgoing: queue.Queue,
                            frames: queue.Queue, stopped, deadline: float,
                            poll_hz: int, *, phase_seconds: int = 10,
                            warmup_seconds: int = 2) -> dict:
    """Offline-first experimental 30/60 Hz sweep: credits 1,2,3,4,8,16,32.

    The authenticated Comm and GPU lifecycle remain owned by the caller.
    This function never allocates GPU resources or creates a socket.
    """
    return run_h264_credit_sweep(
        peer, gpu, outgoing, frames, stopped, deadline, poll_hz,
        phase_seconds=phase_seconds, warmup_seconds=warmup_seconds,
        experimental=True, phases=HIGH_RATE_PHASES,
    )
