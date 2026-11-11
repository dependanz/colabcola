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

from .bridge_colab import offer_update
from .bridge_viewer_stats import LocalFrameTiming
from .bridge_video_comm_pipeline import validated_remote_video_diagnostics
from .bridge_video_comm_protocol import (
    MAX_PULL_SEQ, VideoCommProtocolError, VideoSequenceGate,
    decode_video_reply,
)
from .bridge_video_comm_session import WIDTH, HEIGHT, make_host_decoder
from .lifecycle_io import LifecycleError

PUSH_CREDITS = 4
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


def run_push_comm_h264(peer, gpu, outgoing, frames, stopped, deadline):
    """One host event loop, 4 credits, no per-frame request or automatic replay."""
    if (getattr(peer, "video_push", False) is not True
            or getattr(peer, "video_codec", None) != "h264"
            or peer.wire is None or not peer.wire.connection_ready.is_set()):
        raise LifecycleError("push video requires an attached opted-in Comm")
    av, decoder = make_host_decoder()
    gate = VideoSequenceGate()
    input_seq = last_ack = 0
    last_renderer_seq = -1
    expected_seq = 1
    received_frames = empty_replies = input_observations = 0
    input_callback_ns = {}
    latest_acked_callback_ns = None
    first_decode_input_ms = []
    renderer_lags = []
    started = last_frame_at = time.monotonic()
    header = b"P6\n640 360\n255\n"
    print("[colabcola] v34 sender-driven H.264 Comm; 4-credit receive window",
          flush=True)
    offer_update(frames, ("status", f"Push H.264 Comm · {gpu}"))
    peer._send("video_credit", count=PUSH_CREDITS)
    try:
        while not stopped.is_set() and time.monotonic() < deadline:
            # Coalesce user motion; input submission is never blocked on video.
            newest = None
            while True:
                try:
                    newest = outgoing.get_nowait()
                except queue.Empty:
                    break
            if newest is not None:
                event = dict(newest)
                callback_ns = event.pop("_ui_callback_ns", None)
                if (callback_ns is not None and
                        (type(callback_ns) is not int or callback_ns <= 0)):
                    raise LifecycleError("invalid local UI event clock")
                input_seq += 1
                peer.send_input(input_seq, event)
                if callback_ns is not None:
                    input_callback_ns[input_seq] = callback_ns
                if len(input_callback_ns) > MAX_INPUT_PENDING:
                    for old in sorted(input_callback_ns)[:-MAX_INPUT_PENDING]:
                        del input_callback_ns[old]

            try:
                # Avoid coarse Windows Queue.get(timeout) scheduling (often
                # approximately 15.6 ms) in the latency-sensitive mode.
                message = peer.wire.iopub_channel.get_msg(timeout=0)
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
                    raise LifecycleError("push video freshness watchdog fired")
                continue

            received_ns = time.perf_counter_ns()
            unit, ack, renderer_seq = decode_push_message(
                payload, expected_seq=expected_seq, run_id=peer.run_id,
                last_ack=last_ack, last_renderer_seq=last_renderer_seq,
                gate=gate)
            expected_seq += 1
            if ack > last_ack:
                matched = [seq for seq in input_callback_ns if last_ack < seq <= ack]
                if matched:
                    newest_ack = max(matched)
                    latest_acked_callback_ns = input_callback_ns[newest_ack]
                    first_decode_input_ms.append(
                        (received_ns - latest_acked_callback_ns) / 1_000_000)
                    input_observations += 1
                for seq in list(input_callback_ns):
                    if seq <= ack:
                        del input_callback_ns[seq]
            last_ack = ack
            if unit is not None:
                last_renderer_seq = renderer_seq
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
                offer_update(frames, ("frame-timed", (
                    header + pixels,
                    LocalFrameTiming(
                        captured_ns=received_ns, decoded_ns=decoded_ns,
                        input_callback_ns=latest_acked_callback_ns))))
                received_frames += 1
                last_frame_at = time.monotonic()
                diag = validated_remote_video_diagnostics(payload)
                if diag and "renderer_buffer_lag" in diag:
                    renderer_lags.append(diag["renderer_buffer_lag"])
            else:
                empty_replies += 1
            # Credit is returned only AFTER complete validated processing.
            peer._send("video_credit", count=1)

        elapsed = max(.001, time.monotonic() - started)
        report = {
            "schema": 1, "mode": "external-credit-push-h264",
            "decoded_frames": received_frames, "decoded_distinct_fps": round(
                received_frames / elapsed, 3),
            "elapsed_seconds": round(elapsed, 3),
            "video_no_new": empty_replies,
            "last_renderer_seq": last_renderer_seq,
            "configured_credits": PUSH_CREDITS,
            "input_to_decoded_observations": input_observations,
            "input_callback_to_receive_p50_ms": (
                round(median(first_decode_input_ms), 2)
                if first_decode_input_ms else None),
            "renderer_buffer_lag_p50_frames": (
                median(renderer_lags) if renderer_lags else None),
            "timing_scope": "host callback to host frame receive; Tk configuration reported separately; no physical scanout claim",
        }
        print("[colabcola] v34 push result "
              + json.dumps(report, sort_keys=True), flush=True)
        return report
    finally:
        stopped.set()
