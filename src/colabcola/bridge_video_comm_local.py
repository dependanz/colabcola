"""v33b: local, allocation-free H.264 -> Comm envelope -> H.264 decoder proof.

This is an in-memory protocol harness, NOT a Jupyter Comm, network transport,
Colab GPU allocation, or a remotely synchronized clock. It uses real PyAV
encoding/decoding when run. All frame messages pass through the v33a CCV1
Base64 contract. Loss injection omits whole replies before the receiver sees
them; the receiver must wait for an IDR after a source sequence gap.

This module is deliberately separate from the existing PPM Comm implementation.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import queue
from statistics import median
import threading
import time
import uuid

from .bridge_colab import offer_update
from .bridge_video_comm_protocol import (
    VideoCommProtocolError, VideoSequenceGate,
    decode_video_reply, encode_video_reply,
)
from .bridge_video_live import (
    LiveScene, live_background, make_live_codecs, render_live_rgb, validate_live,
)
from .bridge_video_stream import VideoAccessUnit
from .bridge_viewer_stats import LocalFrameTiming


class LocalVideoCommInbox:
    """Sequence- and pull-aware inbox for a SINGLE in-flight video reply.

    Missing replies may create gaps in pull IDs. Every received reply is still
    correlated to the caller's most recently issued outstanding pull. The
    actual Jupyter transport must enforce outstanding-pull ownership itself.
    """

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.last_received_pull = 0
        self.gate = VideoSequenceGate()
        self.suppressed_units = 0

    def receive(self, message: object, *, expected_pull_seq: int):
        if (type(expected_pull_seq) is not int
                or expected_pull_seq <= self.last_received_pull):
            raise VideoCommProtocolError("stale or repeated local video pull")
        unit, ack = decode_video_reply(
            message, run_id=self.run_id, expected_pull_seq=expected_pull_seq)
        self.last_received_pull = expected_pull_seq
        if not self.gate.accept(unit):
            self.suppressed_units += 1
            return None, ack
        return unit, ack


class LocalVideoCommProducer:
    """Real live PyAV source, exposed through one synthetic Comm-style pull.

    Rendering happens ON PULL in this harness; this is not a continuously
    publishing remote renderer. `capture_ns` and callback timing are local only.
    """

    def __init__(self, *, run_id: str, width: int, height: int, fps: int):
        self.run_id = run_id
        self.width, self.height, self.fps = width, height, fps
        self.av, self.np, self.encoder, self.decoder = make_live_codecs(
            width, height, fps)
        self.base = live_background(width, height, self.np)
        self.scene = LiveScene()
        self.source_seq = 0
        self.last_pull = 0
        self.input_ack = 0
        self.latest_callback_ns = None

    def send_input(self, seq: int, event: dict) -> bool:
        if (type(seq) is not int or not self.input_ack < seq <= 9_999_999
                or not isinstance(event, dict)):
            return False
        # Keep internal viewer timestamps out of the putative provider payload.
        allowed = {k: event[k] for k in ("event", "x", "y", "key") if k in event}
        if not self.scene.apply(allowed):
            return False
        self.input_ack = seq
        callback = event.get("_ui_callback_ns")
        now = time.perf_counter_ns()
        self.latest_callback_ns = (
            callback if type(callback) is int
            and 0 < callback <= now
            and now - callback <= 180 * 1_000_000_000 else None)
        return True

    def pull(self, pull_seq: int) -> dict:
        if (type(pull_seq) is not int
                or not self.last_pull < pull_seq <= 100_000_000):
            raise VideoCommProtocolError("repeated or invalid local video pull")
        seq = self.source_seq
        captured_ns = time.perf_counter_ns()
        rgb = render_live_rgb(self.scene, self.base, seq)
        frame = self.av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = seq
        encoded = self.encoder.encode(frame)
        if len(encoded) != 1:
            raise RuntimeError("local video Comm expected one encoded access unit per frame")
        packet = encoded[0]
        unit = VideoAccessUnit(
            seq=seq, pts_ns=round(seq * 1_000_000_000 / self.fps),
            capture_ns=captured_ns, data=bytes(packet),
            keyframe=bool(packet.is_keyframe),
        )
        reply = encode_video_reply(
            run_id=self.run_id, pull_seq=pull_seq,
            input_ack=self.input_ack, unit=unit)
        self.source_seq += 1
        self.last_pull = pull_seq
        return reply


def _validated_drops(drop_source_seqs: tuple[int, ...], frames: int) -> frozenset[int]:
    if (type(drop_source_seqs) is not tuple
            or tuple(sorted(set(drop_source_seqs))) != drop_source_seqs
            or any(type(x) is not int or x < 0 or x >= frames
                   for x in drop_source_seqs)):
        raise ValueError("source reply omissions must be unique sorted in-range integers")
    return frozenset(drop_source_seqs)


def run_local_video_comm(outgoing: queue.Queue, frames: queue.Queue, stopped,
                         *, width: int = 128, height: int = 72, fps: int = 12,
                         max_frames: int = 24,
                         drop_source_seqs: tuple[int, ...] = (),
                         timed_frames: bool = False) -> dict:
    """Run the real local codec with synthetic Comm messages and optional loss.

    With no missing replies, all frames should decode. After an injected loss,
    P-frames are suppressed until an IDR, so decoded count can be lower. The
    only reported capture-to-decode durations use the SAME local process clock.
    """
    validate_live(width, height, fps, max_frames)
    if type(timed_frames) is not bool:
        raise ValueError("timed_frames must be boolean")
    missing = _validated_drops(drop_source_seqs, max_frames)
    run_id = uuid.uuid4().hex
    source = LocalVideoCommProducer(
        run_id=run_id, width=width, height=height, fps=fps)
    receiver = LocalVideoCommInbox(run_id)
    pending = deque()
    decoded_seqs = []
    decoded_ages = []
    input_decoded_ms = []
    accepted_inputs = 0
    observed_input_ack = 0
    reply_count = 0
    last_callback_ns = None
    ppm_header = f"P6\n{width} {height}\n255\n".encode("ascii")
    start = time.perf_counter()
    offer_update(frames, ("status", "Local H.264 Comm-envelope proof (no Colab)"))

    def process_decoded(images):
        nonlocal observed_input_ack
        for image in images:
            if not pending:
                raise RuntimeError("local video Comm decoder produced unsolicited frame")
            seq, capture_ns, ack, callback_ns = pending.popleft()
            pixels = image.to_ndarray(format="rgb24").tobytes()
            if len(pixels) != width * height * 3:
                raise RuntimeError("local video Comm decoded unexpected RGB size")
            decoded_ns = time.perf_counter_ns()
            decoded_seqs.append(seq)
            decoded_ages.append((decoded_ns - capture_ns) / 1e6)
            if ack > observed_input_ack:
                observed_input_ack = ack
                if callback_ns is not None:
                    input_decoded_ms.append((decoded_ns - callback_ns) / 1e6)
            ppm = ppm_header + pixels
            if timed_frames:
                offer_update(frames, ("frame-timed", (ppm, LocalFrameTiming(
                    captured_ns=capture_ns, decoded_ns=decoded_ns,
                    input_callback_ns=callback_ns))))
            else:
                offer_update(frames, ("frame", ppm))

    try:
        for source_seq in range(max_frames):
            while not stopped.is_set():
                delay = start + source_seq / fps - time.perf_counter()
                if delay <= 0:
                    break
                time.sleep(min(.05, delay))
            if stopped.is_set():
                break
            event = None
            while True:
                try:
                    event = outgoing.get_nowait()
                except queue.Empty:
                    break
            if event is not None and source.send_input(accepted_inputs + 1, event):
                accepted_inputs += 1
                last_callback_ns = source.latest_callback_ns
            pull_seq = source_seq + 1
            message = source.pull(pull_seq)
            if source_seq in missing:
                # No reply delivered to receiver: this simulates an absent
                # whole Comm reply, NOT a packet-level network implementation.
                continue
            reply_count += 1
            unit, ack = receiver.receive(message, expected_pull_seq=pull_seq)
            if unit is None:
                continue
            pending.append((unit.seq, unit.capture_ns, ack, last_callback_ns))
            packet = source.av.Packet(unit.data)
            packet.pts = unit.seq
            process_decoded(source.decoder.decode(packet))
        if not stopped.is_set():
            process_decoded(source.decoder.decode(None))
            if pending:
                raise RuntimeError("local video Comm decoded-frame metadata remained pending")
            if not missing and len(decoded_seqs) != max_frames:
                raise RuntimeError("lossless local video Comm did not decode every source frame")
        return {
            "schema": 1,
            "mode": "local-h264-comm-envelope-live",
            "produced_frames": source.source_seq,
            "received_envelopes": reply_count,
            "synthetic_missing_replies": len(set(range(source.source_seq)) & missing),
            "suppressed_dependent_units": receiver.suppressed_units,
            "decoded_frames": len(decoded_seqs),
            "decoded_source_seqs": decoded_seqs,
            "accepted_input_events": accepted_inputs,
            "last_input_ack_observed_in_decoded_frame": observed_input_ack,
            "capture_to_decoded_p50_ms": round(median(decoded_ages), 3)
            if decoded_ages else None,
            "input_callback_to_decoded_p50_ms": round(median(input_decoded_ms), 3)
            if input_decoded_ms else None,
            "scope": "local PyAV and in-memory Comm reply envelopes, not Jupyter Comm, Colab, remote networking, or input-to-photon timing",
        }
    finally:
        stopped.set()


def _parse_drops(raw: str) -> tuple[int, ...]:
    if not raw:
        return ()
    try:
        values = tuple(int(x) for x in raw.split(","))
    except ValueError:
        raise ValueError("--drop-source-seqs must be comma-separated integers") from None
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--max-seconds", type=int, default=10)
    parser.add_argument("--drop-source-seqs", default="")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--viewer-poll-ms", type=int, default=30)
    args = parser.parse_args(argv)
    if not 1 <= args.max_seconds <= 180:
        parser.error("--max-seconds must be 1..180")
    if args.frames > args.fps * args.max_seconds:
        parser.error("--frames exceeds explicit time budget")
    if not 5 <= args.viewer_poll_ms <= 100:
        parser.error("--viewer-poll-ms must be 5..100")
    try:
        validate_live(args.width, args.height, args.fps, args.frames)
        drops = _parse_drops(args.drop_source_seqs)
        _validated_drops(drops, args.frames)
    except ValueError as exc:
        parser.error(str(exc))
    worker = lambda outgoing, frames, stopped: run_local_video_comm(
        outgoing, frames, stopped,
        width=args.width, height=args.height, fps=args.fps,
        max_frames=args.frames, drop_source_seqs=drops,
        timed_frames=args.show)
    if args.show:
        from .bridge_demo import run_viewer
        run_viewer(
            "127.0.0.1", 0, "", width=args.width, height=args.height,
            title="Colab Cola · local H.264 Comm-envelope proof",
            worker=worker, viewer_stats=True, timestamp_local_input=True,
            poll_interval_ms=args.viewer_poll_ms)
    else:
        result = worker(queue.Queue(maxsize=64), queue.Queue(maxsize=2),
                        threading.Event())
        print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
