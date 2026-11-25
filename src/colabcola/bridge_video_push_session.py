"""v34 opt-in sender-driven H.264 transport over one authenticated Jupyter Comm.

No host video_pull messages. Four bounded credits are granted initially, then
one returned only when a correlated video response is fully processed.
All latency timestamps use the same Windows-host perf_counter_ns clock.
"""
from __future__ import annotations

import json
import queue
from statistics import median
import time

from .bridge_pipeline_trace import PipelineTrace, validated_credit_snapshot

from .bridge_colab import offer_update
from .bridge_viewer_stats import LocalFrameTiming, pooled_percentiles
from .bridge_video_comm_pipeline import validated_remote_video_diagnostics
from .bridge_video_comm_protocol import (
    MAX_PULL_SEQ, VideoCommProtocolError, VideoSequenceGate,
    decode_video_reply,
)
from .bridge_video_comm_session import WIDTH, HEIGHT, make_host_decoder
from .lifecycle_io import LifecycleError

PUSH_CREDITS = 4
PUSH_CREDIT_CHOICES = (4, 6, 8, 10, 12)
PUSH_RECEIVE_WATCHDOG_S = 12.0
MAX_INPUT_PENDING = 128


def validate_push_message(message, *, comm_id, run_id):
    """Filter unrelated IOPub traffic, but fail closed on bad matched video."""
    if not isinstance(message, dict):
        return None
    header = message.get("header")
    content = message.get("content")
    if (not isinstance(header, dict) or header.get("msg_type") != "comm_msg"
            or not isinstance(content, dict)
            or content.get("comm_id") != comm_id):
        return None
    payload = content.get("data")
    if not isinstance(payload, dict) or payload.get("run_id") != run_id:
        return None
    if type(payload.get("v")) is not int or payload["v"] != 1:
        raise LifecycleError("push Comm version invalid")
    if payload.get("kind") == "error":
        raise LifecycleError("remote push video reported error")
    if payload.get("kind") not in ("video_access_unit", "video_no_new"):
        return None
    return payload


def decode_push_message(payload, *, expected_seq, run_id, last_ack,
                        last_renderer_seq, gate):
    """One monotone coded access unit or a strictly correlated empty response."""
    if (type(expected_seq) is not int or not 1 <= expected_seq <= MAX_PULL_SEQ
            or type(payload.get("pull_seq")) is not int
            or payload["pull_seq"] != expected_seq):
        raise LifecycleError("push video sequence invalid")
    try:
        if payload["kind"] == "video_no_new":
            ack = payload.get("input_ack")
            if (type(ack) is not int or not last_ack <= ack < 10_000_000
                    or set(payload) != {"v", "kind", "run_id", "pull_seq",
                                        "input_ack"}):
                raise VideoCommProtocolError("invalid push empty frame")
            return None, ack, last_renderer_seq
        unit, ack = decode_video_reply(
            payload, run_id=run_id, expected_pull_seq=expected_seq)
        if ack < last_ack:
            raise VideoCommProtocolError("push input ACK regressed")
        source = payload.get("renderer_seq")
        if (type(source) is not int or not 0 <= source < 100_000_000
                or source <= last_renderer_seq):
            raise VideoCommProtocolError("push source sequence not fresh")
        if not gate.accept(unit):
            raise VideoCommProtocolError("push H.264 lost decoder dependency")
        return unit, ack, source
    except VideoCommProtocolError:
        raise LifecycleError("push H.264 response failed validation") from None


def validated_sender_diagnostics(payload: object) -> dict | None:
    """Optional v35B remote-only durations; no remote/host clock subtraction."""
    if not isinstance(payload, dict):
        return None
    keys = ("sender_credit_wait_us_total", "sender_credit_wait_iterations")
    if not any(k in payload for k in (*keys, "sender_comm_previous_us")):
        return None
    wait = payload.get(keys[0])
    iterations = payload.get(keys[1])
    previous = payload.get("sender_comm_previous_us")
    if (type(wait) not in (int, float) or not 0 <= wait <= 1_800_000_000
            or type(iterations) is not int or not 0 <= iterations <= 10_000_000
            or (previous is not None and (
                type(previous) not in (int, float) or not 0 <= previous <= 10_000_000))):
        return None
    import math
    if not math.isfinite(wait) or (previous is not None and not math.isfinite(previous)):
        return None
    return {"credit_wait_us_total": float(wait), "credit_wait_iterations": iterations,
            "comm_previous_us": float(previous) if previous is not None else None}


def producer_stage_summary(windows: dict) -> dict:
    """Report *window statistics*, not fictitious pooled producer samples."""
    result = {}
    for prefix, field in (
            ("render_copy", "render_copy_ms"),
            ("compress", "compress_ms"),
            ("publish", "publish_ms")):
        medians = [w[field] for w in windows.values()]
        tails = [w[field.replace("_ms", "_p95_ms")] for w in windows.values()
                 if field.replace("_ms", "_p95_ms") in w]
        result[f"producer_{prefix}_window_median_p50_ms"] = (
            round(median(medians), 2) if medians else None)
        result[f"producer_{prefix}_window_median_p95_ms"] = (
            pooled_percentiles(medians)["p95"])
        result[f"producer_{prefix}_within_window_p95_median_ms"] = (
            round(median(tails), 2) if tails else None)
    return result


def run_push_comm_h264(peer, gpu, outgoing, frames, stopped, deadline,
                       *, credits: int = PUSH_CREDITS, trace_path=None):
    """One host event loop, bounded credits, no per-frame pull or replay."""
    if type(credits) is not int or credits not in PUSH_CREDIT_CHOICES:
        raise ValueError("invalid sender-driven credit configuration")
    if getattr(peer, "push_credits", credits) != credits:
        raise LifecycleError("push credit configuration mismatch")
    target_fps = getattr(peer, "fps", 30)
    if (type(target_fps) is not int or target_fps not in (30, 60)
            or (target_fps == 30 and credits not in (4, 6, 8))
            or (target_fps == 60 and credits not in (8, 10, 12))):
        raise LifecycleError("invalid push FPS/credit pairing")
    if (getattr(peer, "video_push", False) is not True
            or getattr(peer, "video_codec", None) != "h264"
            or peer.wire is None or not peer.wire.connection_ready.is_set()):
        raise LifecycleError("push video requires an attached opted-in Comm")
    if trace_path is not None and target_fps != 30:
        raise ValueError("pipeline trace requires 30Hz push")
    trace = (PipelineTrace(trace_path, run_id=peer.run_id)
             if trace_path is not None else None)
    av, decoder = make_host_decoder()
    gate = VideoSequenceGate()
    input_seq = last_ack = 0
    last_renderer_seq = -1
    expected_seq = 1
    received_frames = empty_replies = input_observations = 0
    input_callback_ns = {}
    latest_acked_callback_ns = None
    first_receive_input_ms = []
    first_decode_input_ms = []
    renderer_lags = []
    renderer_skipped = 0
    remote_encode_us = []
    remote_source_read_us = []
    remote_comm_previous_us = []
    sender_credit_wait_us_total = None
    sender_credit_wait_iterations = None
    sender_diag_observations = 0
    producer_windows = {}
    receive_interval_ms = []
    previous_received_ns = None
    started = last_frame_at = time.monotonic()
    header = b"P6\n640 360\n255\n"
    print(f"[colabcola] v34 sender-driven H.264 Comm; {credits}-credit receive window",
          flush=True)
    offer_update(frames, ("status", f"Push H.264 Comm · {gpu}"))
    outcome = "failed"
    phase = "initial_credit"
    try:
        credit_start = time.perf_counter_ns() if trace else None
        peer._send("video_credit", count=credits)
        if trace:
            trace.add("initial_credit", count=credits, start_ns=credit_start,
                      end_ns=time.perf_counter_ns())
        while not stopped.is_set() and time.monotonic() < deadline:
            # Synthetic input coalesces; gameplay preserves all key transitions.
            phase = "host_input"
            pending = []
            if getattr(peer, "application", "synthetic") in ("freedoom", "inputlab"):
                for _ in range(16):
                    try:
                        pending.append(outgoing.get_nowait())
                    except queue.Empty:
                        break
            else:
                newest = None
                while True:
                    try:
                        newest = outgoing.get_nowait()
                    except queue.Empty:
                        break
                if newest is not None:
                    pending.append(newest)
            for fresh in pending:
                event = dict(fresh)
                callback_ns = event.pop("_ui_callback_ns", None)
                if (callback_ns is not None and
                        (type(callback_ns) is not int or callback_ns <= 0)):
                    raise LifecycleError("invalid local UI event clock")
                input_seq += 1
                send_start = time.perf_counter_ns() if trace else None
                peer.send_input(input_seq, event)
                if trace:
                    trace.add("input_sent", input_seq=input_seq, callback_ns=callback_ns,
                              send_start_ns=send_start, send_end_ns=time.perf_counter_ns(),
                              outgoing_queue_depth=outgoing.qsize())
                if callback_ns is not None:
                    input_callback_ns[input_seq] = callback_ns
                if len(input_callback_ns) > MAX_INPUT_PENDING:
                    for old in sorted(input_callback_ns)[:-MAX_INPUT_PENDING]:
                        del input_callback_ns[old]

            phase = "host_receive"
            try:
                # Avoid coarse Windows Queue.get(timeout) scheduling (often
                # approximately 15.6 ms) in the latency-sensitive mode.
                message = peer.wire.iopub_channel.get_msg(timeout=0)
                dequeue_ns = time.perf_counter_ns() if trace else None
            except queue.Empty:
                message = None
                time.sleep(.001)
            except Exception:
                raise LifecycleError("push IOPub receive failed") from None
            if not peer.wire.connection_ready.is_set():
                raise LifecycleError("push kernel websocket disconnected")
            payload = validate_push_message(
                message, comm_id=peer.comm_id, run_id=peer.run_id)
            if payload is None:
                if time.monotonic() - last_frame_at > PUSH_RECEIVE_WATCHDOG_S:
                    phase = "freshness_watchdog"
                    raise LifecycleError("push video freshness watchdog fired")
                continue

            received_ns = time.perf_counter_ns()
            phase = "video_validation"
            unit, ack, renderer_seq = decode_push_message(
                payload, expected_seq=expected_seq, run_id=peer.run_id,
                last_ack=last_ack, last_renderer_seq=last_renderer_seq,
                gate=gate)
            expected_seq += 1
            if trace:
                trace.add("video_received", video_seq=expected_seq - 1,
                          renderer_seq=renderer_seq if unit is not None else None,
                          input_ack=ack, host_dequeue_ns=dequeue_ns,
                          host_matched_ns=received_ns, host_validated_ns=time.perf_counter_ns(),
                          access_unit_bytes=len(unit.data) if unit is not None else 0)
            decoded_ns = None
            diag = sender_diag = None
            newly_acked_callback_ns = None
            if ack > last_ack:
                matched = [seq for seq in input_callback_ns if last_ack < seq <= ack]
                if matched:
                    newest_ack = max(matched)
                    latest_acked_callback_ns = input_callback_ns[newest_ack]
                    newly_acked_callback_ns = latest_acked_callback_ns
                    first_receive_input_ms.append(
                        (received_ns - latest_acked_callback_ns) / 1_000_000)
                    input_observations += 1
                for seq in list(input_callback_ns):
                    if seq <= ack:
                        del input_callback_ns[seq]
            last_ack = ack
            if unit is not None:
                if last_renderer_seq >= 0:
                    renderer_skipped += max(0, renderer_seq - last_renderer_seq - 1)
                last_renderer_seq = renderer_seq
                if previous_received_ns is not None:
                    receive_interval_ms.append(
                        (received_ns - previous_received_ns) / 1_000_000)
                previous_received_ns = received_ns
                phase = "host_decode"
                decode_start_ns = time.perf_counter_ns() if trace else None
                try:
                    packet = av.Packet(unit.data)
                    packet.pts = unit.seq
                    images = decoder.decode(packet)
                    if len(images) != 1:
                        raise ValueError("unexpected push decoder output count")
                    image = images[0]
                    if (image.width, image.height) != (WIDTH, HEIGHT):
                        raise ValueError("push frame dimensions invalid")
                    pixels = image.to_ndarray(format="rgb24").tobytes()
                    if len(pixels) != WIDTH * HEIGHT * 3:
                        raise ValueError("push RGB image size invalid")
                except Exception:
                    raise LifecycleError("push H.264 decoder failed") from None
                decoded_ns = time.perf_counter_ns()
                if newly_acked_callback_ns is not None:
                    first_decode_input_ms.append(
                        (decoded_ns - newly_acked_callback_ns) / 1_000_000)
                if trace:
                    trace.add("frame_offered", video_seq=expected_seq - 1,
                              renderer_seq=renderer_seq, input_ack=ack,
                              decode_start_ns=decode_start_ns, decoded_ns=decoded_ns,
                              offer_start_ns=time.perf_counter_ns())
                offer_update(frames, ("frame-timed", (
                    header + pixels,
                    LocalFrameTiming(
                        captured_ns=received_ns, decoded_ns=decoded_ns,
                        input_callback_ns=latest_acked_callback_ns,
                        **({"video_seq": expected_seq - 1, "renderer_seq": renderer_seq,
                            "input_ack": ack} if trace else {})))))
                received_frames += 1
                last_frame_at = time.monotonic()
                diag = validated_remote_video_diagnostics(payload)
                if diag:
                    if "renderer_buffer_lag" in diag:
                        renderer_lags.append(diag["renderer_buffer_lag"])
                    remote_encode_us.append(diag["encode_us"])
                    remote_source_read_us.append(diag["source_read_us"])
                    window = diag.get("producer_window")
                    if window is not None:
                        wid = window["window_seq"]
                        previous = producer_windows.get(wid)
                        if previous is None:
                            producer_windows[wid] = window
                            if trace:
                                trace.add("producer_window", video_seq=expected_seq - 1,
                                          host_dequeue_ns=dequeue_ns, window=window)
                        elif previous != window:
                            raise LifecycleError("conflicting push producer windows")
                sender_diag = validated_sender_diagnostics(payload)
                if sender_diag is not None:
                    wait = sender_diag["credit_wait_us_total"]
                    iterations = sender_diag["credit_wait_iterations"]
                    if (sender_credit_wait_us_total is None
                            or (wait >= sender_credit_wait_us_total and
                                iterations >= sender_credit_wait_iterations)):
                        sender_credit_wait_us_total = wait
                        sender_credit_wait_iterations = iterations
                        sender_diag_observations += 1
                    previous_us = sender_diag["comm_previous_us"]
                    if previous_us is not None:
                        remote_comm_previous_us.append(previous_us)
            else:
                empty_replies += 1
            # Credit is returned only AFTER complete validated processing.
            phase = "credit_return"
            credit_start = time.perf_counter_ns() if trace else None
            peer._send("video_credit", count=1)
            if trace:
                trace.add("video_processed", video_seq=expected_seq - 1,
                          renderer_seq=renderer_seq if unit is not None else None,
                          input_ack=ack, host_decoded_ns=decoded_ns,
                          credit_return_start_ns=credit_start,
                          credit_return_end_ns=time.perf_counter_ns(),
                          remote_video=({k: v for k, v in diag.items()
                                         if k != "producer_window"} if diag else None),
                          remote_sender=sender_diag,
                          remote_credit=validated_credit_snapshot(payload, credits, expected_seq - 1))

        phase = "session_summary"
        elapsed = max(.001, time.monotonic() - started)
        covered_ms = sum(w["window_elapsed_ms"] for w in producer_windows.values())
        covered_frames = sum(w["published_count"] for w in producer_windows.values())
        report = {
            "schema": 1, "mode": "external-credit-push-h264",
            "decoded_frames": received_frames, "decoded_distinct_fps": round(
                received_frames / elapsed, 3),
            "elapsed_seconds": round(elapsed, 3),
            "video_no_new": empty_replies,
            "last_renderer_seq": last_renderer_seq,
            "renderer_frames_skipped_total": renderer_skipped,
            "configured_credits": credits,
            "target_fps": target_fps,
            "application": getattr(peer, "application", "synthetic"),
            "producer_zlib_level": (
                0 if getattr(peer, "producer_zlib_level0", False) else 1),
            "producer_windows_observed": len(producer_windows),
            "producer_frames_covered": covered_frames,
            "producer_elapsed_ms_covered": round(covered_ms, 3),
            "producer_renderer_fps_weighted_covered": (
                round(1000 * covered_frames / covered_ms, 3)
                if covered_ms else None),
            **producer_stage_summary(producer_windows),
            "producer_renderer_fps_p50": (
                round(median(w["renderer_fps"] for w in producer_windows.values()), 3)
                if producer_windows else None),
            "remote_encode_p50_ms": (
                round(median(remote_encode_us) / 1000, 3)
                if remote_encode_us else None),
            **{f"remote_encode_{key}_ms": value for key, value in
               pooled_percentiles([v / 1000 for v in remote_encode_us]).items()
               if key != "p50"},
            **{f"remote_source_read_{key}_ms": value for key, value in
               pooled_percentiles([v / 1000 for v in remote_source_read_us]).items()},
            **{f"remote_comm_previous_send_{key}_ms": value for key, value in
               pooled_percentiles([v / 1000 for v in remote_comm_previous_us]).items()},
            "remote_sender_diagnostic_observations": sender_diag_observations,
            "remote_sender_credit_wait_ms_total_observed": (
                round(sender_credit_wait_us_total / 1000, 2)
                if sender_credit_wait_us_total is not None else None),
            "remote_sender_credit_wait_iterations_observed": sender_credit_wait_iterations,
            "host_receive_interval_p50_ms": (
                round(median(receive_interval_ms), 2)
                if receive_interval_ms else None),
            "host_receive_interval_max_ms": (
                round(max(receive_interval_ms), 2)
                if receive_interval_ms else None),
            **{f"host_receive_interval_{key}_ms": value
               for key, value in pooled_percentiles(receive_interval_ms).items()},
            "input_to_decoded_observations": input_observations,
            **{f"input_callback_to_receive_{key}_ms": value
               for key, value in pooled_percentiles(first_receive_input_ms).items()},
            **{f"input_callback_to_decode_{key}_ms": value
               for key, value in pooled_percentiles(first_decode_input_ms).items()},
            "renderer_buffer_lag_p50_frames": (
                median(renderer_lags) if renderer_lags else None),
            "timing_scope": "host callback to host frame receive; Tk configuration reported separately; no physical scanout claim",
        }
        print("[colabcola] v34 push result "
              + json.dumps(report, sort_keys=True), flush=True)
        outcome = "completed"
        return report
    finally:
        stopped.set()
        if trace:
            trace.finish(outcome=outcome, phase=phase, decoded_frames=received_frames,
                         last_video_seq=expected_seq - 1, last_input_ack=last_ack)
