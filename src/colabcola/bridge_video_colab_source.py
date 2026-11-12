"""v33d: optional kernel-side H.264 encoding of owned CUDA PPM snapshots.

This standalone source is uploaded to a managed kernel only by a later
explicit video-mode bootstrap. It does NOT open a socket or allocate GPUs.
A pull samples the existing CCB1 latest-frame file, then feeds that RGB
image into one persistent PyAV x264 encoder. Video seq counts ENCODED pulls,
not GPU renderer frames; renderer_seq is reported separately. All clock
values in CCV1 are REMOTE and are never host capture-to-decode timestamps.
"""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path
import struct
import time
import zlib

WIDTH, HEIGHT = 640, 360
MAX_PPM_COMPRESSED = 2 * 1024 * 1024
MAX_UNIT_BYTES = 2 * 1024 * 1024
MAX_PULL_SEQ = 100_000_000
CCB1 = struct.Struct("!4sIIII")
CCV1 = struct.Struct("!4sBBIQQI")


class RemoteVideoSourceError(RuntimeError):
    """Failure safe to report only by type, never by content or credentials."""


def provision_remote_pyav() -> None:
    """Opt-in, one-shot PyPI binary wheel provisioning in the OWNED kernel.

    The host must explicitly authorize this through --install-remote-pyav.
    Do not use shell invocation, source builds, retries, user indexes, or
    unbounded pip output. Never install anything in the host environment.
    """
    import importlib
    import subprocess
    import sys

    try:
        import av  # noqa: F401
        return  # Already importable: do not modify the runtime environment.
    except ImportError:
        pass

    command = [
        sys.executable, "-m", "pip", "--isolated", "install",
        "--disable-pip-version-check", "--no-input", "--no-cache-dir",
        "--only-binary=:all:", "--no-deps", "--retries", "0",
        "--timeout", "8", "--index-url", "https://pypi.org/simple",
        "av==17.0.0",
    ]
    try:
        completed = subprocess.run(
            command, check=False, timeout=40,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        raise RemoteVideoSourceError("remote PyAV installation timed out") from None
    except OSError:
        raise RemoteVideoSourceError("remote PyAV installation failed") from None
    if completed.returncode != 0:
        raise RemoteVideoSourceError("remote PyAV installation failed")
    importlib.invalidate_caches()
    try:
        importlib.import_module("av")
    except Exception:
        raise RemoteVideoSourceError("remote PyAV installation failed") from None


def _make_encoder(width: int, height: int, fps: int):
    """Create a single low-delay encoder (optional PyAV is not installed here)."""
    try:
        import av
    except ImportError:
        raise RemoteVideoSourceError("remote PyAV unavailable") from None
    try:
        import numpy as np
    except ImportError:
        raise RemoteVideoSourceError("remote NumPy unavailable") from None
    from fractions import Fraction

    try:
        codec = av.CodecContext.create("libx264", "w")
    except Exception:
        raise RemoteVideoSourceError("remote libx264 encoder unavailable") from None
    codec.width, codec.height = width, height
    codec.pix_fmt = "yuv420p"
    codec.time_base = Fraction(1, fps)
    codec.framerate = Fraction(fps, 1)
    codec.thread_count = 1
    codec.options = {
        "preset": "ultrafast", "tune": "zerolatency", "crf": "23",
        "x264-params": (
            f"bframes=0:rc-lookahead=0:sync-lookahead=0:keyint={fps}:"
            "min-keyint=1:repeat-headers=1:aud=1"
        ),
    }
    try:
        codec.open()
    except Exception:
        raise RemoteVideoSourceError("remote libx264 encoder unavailable") from None

    def encode(pixels: bytes, coded_seq: int) -> tuple[bytes, bool]:
        rgb = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 3))
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = coded_seq
        packets = codec.encode(frame)
        if len(packets) != 1:
            raise RemoteVideoSourceError("encoder output was not exactly one access unit")
        packet = packets[0]
        return bytes(packet), bool(packet.is_keyframe)

    return encode


def _contains_idr(data: bytes) -> bool:
    """Conservative Annex B IDR NAL detection; no byte-pattern substring guess."""
    pos = 0
    while pos < len(data):
        start3 = data.find(b"\x00\x00\x01", pos)
        if start3 < 0:
            return False
        header = start3 + 3
        if header < len(data) and (data[header] & 0x1f) == 5:
            return True
        pos = header + 1
    return False


def read_latest_rgb(path: Path, *, width: int = WIDTH, height: int = HEIGHT):
    """Validate the existing atomic CCB1 snapshot before using its RGB bytes."""
    try:
        if not path.is_file() or path.stat().st_size > CCB1.size + MAX_PPM_COMPRESSED:
            raise RemoteVideoSourceError("remote snapshot missing or too large")
        body = path.read_bytes()
    except (OSError, ValueError):
        raise RemoteVideoSourceError("remote snapshot unavailable") from None
    if not CCB1.size < len(body) <= CCB1.size + MAX_PPM_COMPRESSED:
        raise RemoteVideoSourceError("remote snapshot size invalid")
    magic, renderer_seq, input_ack, w, h = CCB1.unpack_from(body)
    if magic != b"CCB1" or (w, h) != (width, height):
        raise RemoteVideoSourceError("remote snapshot header invalid")
    prefix = f"P6\n{w} {h}\n255\n".encode("ascii")
    expected = len(prefix) + 3 * w * h
    decoder = zlib.decompressobj()
    try:
        ppm = decoder.decompress(body[CCB1.size:], expected + 1)
    except zlib.error:
        raise RemoteVideoSourceError("remote snapshot compressed data invalid") from None
    if (len(ppm) != expected or ppm[:len(prefix)] != prefix or not decoder.eof
            or decoder.unconsumed_tail or decoder.unused_data):
        raise RemoteVideoSourceError("remote snapshot integrity check failed")
    return renderer_seq, input_ack, ppm[len(prefix):]


RING_SLOTS = 8  # bounded publication history; never scans arbitrary files
MAX_BUFFER_LAG_FRAMES = 2  # preserve near-real-time interaction freshness


def read_next_buffered_rgb(directory: Path, *, after_seq: int | None,
                           width: int = WIDTH, height: int = HEIGHT,
                           max_lag_frames: int = MAX_BUFFER_LAG_FRAMES):
    """Read one distinct, sequence-increasing snapshot from an eight-slot ring.

    Scan only fixed slot headers; validate the selected payload with the same
    bounded CCB1/zlib decoder as legacy. A concurrent overwrite may invalidate
    a candidate, in which case it is skipped rather than replayed. No file
    polling waits or unbounded accumulation is introduced.
    """
    if type(max_lag_frames) is not int or not 0 <= max_lag_frames <= 2:
        raise RemoteVideoSourceError("invalid renderer lag bound")
    candidates = []
    for slot in range(RING_SLOTS):
        path = directory / f"frame-ring-{slot:02d}.bin"
        try:
            with path.open("rb") as source:
                head = source.read(CCB1.size)
        except (OSError, ValueError):
            continue
        if len(head) != CCB1.size:
            continue
        magic, seq, ack, w, h = CCB1.unpack(head)
        if (magic == b"CCB1" and (w, h) == (width, height)
                and seq % RING_SLOTS == slot and
                (after_seq is None or seq > after_seq)):
            candidates.append((seq, path))
    if not candidates:
        return None
    newest = max(seq for seq, _ in candidates)
    # A queue can improve delivered FPS while silently adding latency.
    # A controlled freshness experiment chooses 0, 1, or 2 older frames.
    # Stale frames outside that cap are skipped before encoding.
    candidates = [(seq, path) for seq, path in candidates
                  if seq >= newest - max_lag_frames]
    candidates.sort(key=lambda item: item[0], reverse=after_seq is None)
    for seq, path in candidates:
        try:
            result = read_latest_rgb(path, width=width, height=height)
        except RemoteVideoSourceError:
            continue
        if result[0] != seq:
            continue
        skipped = (max(0, seq - after_seq - 1)
                   if after_seq is not None else 0)
        return (*result, skipped, newest - seq)
    return None


def read_producer_window(path: Path, run_id: str) -> dict | None:
    """Best-effort snapshot of INDEPENDENT CUDA producer measurements.

    Failures are optional telemetry failures, never codec failures. The
    sidecar is atomically replaced by the producer. No remote timestamps are
    compared with host clocks; no new socket or request is introduced.
    """
    try:
        if not path.is_file() or path.stat().st_size > 4096:
            return None
        raw = path.read_bytes()
        if len(raw) > 4096:
            return None
        obj = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    if (not isinstance(obj, dict) or type(obj.get("schema")) is not int
            or obj["schema"] != 1 or obj.get("run_id") != run_id):
        return None
    ints = {"window_seq": (1, 10_000_000),
            "seq": (0, 100_000_000), "published_count": (1, 10_000),
            "target_fps": (1, 60)}
    values = {"window_elapsed_ms": (1, 10_000),
              "renderer_fps": (0, 120),
              "render_copy_ms": (0, 10_000),
              "compress_ms": (0, 10_000),
              "publish_ms": (0, 10_000)}
    if any(type(obj.get(k)) is not int or not lo <= obj[k] <= hi
           for k, (lo, hi) in ints.items()):
        return None
    if any(type(obj.get(k)) not in (int, float)
           or not math.isfinite(obj[k]) or not lo <= obj[k] <= hi
           for k, (lo, hi) in values.items()):
        return None
    # A remote-only consistency check: the producer's own counted windows
    # must agree with its own published rate, allowing round-off to .01 FPS.
    measured_rate = obj["published_count"] * 1000 / obj["window_elapsed_ms"]
    if abs(measured_rate - obj["renderer_fps"]) > 0.02:
        return None
    # v35B fields are optional for compatibility with earlier sidecars.
    tails = {}
    for field, median_field in (
            ("render_copy_p95_ms", "render_copy_ms"),
            ("compress_p95_ms", "compress_ms"),
            ("publish_p95_ms", "publish_ms")):
        if field in obj:
            value = obj[field]
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not obj[median_field] <= value <= 10_000):
                return None
            tails[field] = float(value)
    return {**{k: obj[k] for k in ints},
            **{k: float(obj[k]) for k in values}, **tails}


class RemoteH264PullSource:
    """Single ordered source per owned Comm; request IDs must increase.

    Legacy mode encodes the latest RGB snapshot once per pull. Opt-in
    sequence-buffered mode consumes distinct renderer sequence numbers from
    a fixed eight-slot publication ring. It returns correlated video_no_new
    replies without encoding when no fresh snapshot exists, and bounds lag
    to two producer frames so improved throughput does not mask stale input.
    """

    def __init__(self, directory: str | Path, run_id: str, *, fps: int,
                 width: int = WIDTH, height: int = HEIGHT, encoder_factory=None,
                 sequence_buffered: bool = False,
                 max_buffer_lag_frames: int = MAX_BUFFER_LAG_FRAMES):
        if (not isinstance(run_id, str) or len(run_id) != 32
                or type(fps) is not int or not 1 <= fps <= 60
                or (width, height) != (WIDTH, HEIGHT)):
            raise RemoteVideoSourceError("invalid remote video source configuration")
        self.directory = Path(directory)
        self.run_id = run_id
        self.fps = fps
        self.width, self.height = width, height
        if type(sequence_buffered) is not bool:
            raise RemoteVideoSourceError("invalid sequence-buffered selector")
        if (type(max_buffer_lag_frames) is not int
                or not 0 <= max_buffer_lag_frames <= 2
                or (not sequence_buffered and
                    max_buffer_lag_frames != MAX_BUFFER_LAG_FRAMES)):
            raise RemoteVideoSourceError("invalid renderer lag mode")
        self.max_buffer_lag_frames = max_buffer_lag_frames
        self.sequence_buffered = sequence_buffered
        self.last_renderer_seq = None
        self.encoder = (encoder_factory or _make_encoder)(width, height, fps)
        self.coded_seq = 0
        self.last_pull = 0
        self.last_ack = 0
        self.failed = False

    def pull(self, pull_seq: int) -> dict:
        if self.failed:
            raise RemoteVideoSourceError("remote H.264 source previously failed")
        if (type(pull_seq) is not int or not self.last_pull < pull_seq <= MAX_PULL_SEQ):
            raise RemoteVideoSourceError("invalid or repeated remote video pull")
        if self.coded_seq >= 1_000_000:
            self.failed = True
            raise RemoteVideoSourceError("remote video sequence budget exceeded")
        # Never repeat a possibly executed request after a failure.
        self.last_pull = pull_seq
        try:
            read_started_ns = time.perf_counter_ns()
            if self.sequence_buffered:
                candidate = read_next_buffered_rgb(
                    self.directory, after_seq=self.last_renderer_seq,
                    width=self.width, height=self.height,
                    max_lag_frames=self.max_buffer_lag_frames)
                if candidate is None:
                    # A positive, correlated response releases the host credit.
                    # No encoder call or video sequence increment occurs; the
                    # stateful H.264 stream remains contiguous across empties.
                    return {"v": 1, "kind": "video_no_new",
                            "run_id": self.run_id, "pull_seq": pull_seq,
                            "input_ack": self.last_ack}
                renderer_seq, ack, pixels, skipped_frames, buffer_lag = candidate
            else:
                renderer_seq, ack, pixels = read_latest_rgb(
                    self.directory / "frame.bin", width=self.width, height=self.height)
                skipped_frames = 0
                buffer_lag = 0
            source_read_us = (time.perf_counter_ns() - read_started_ns) / 1000
            # The one-second producer sidecar may be missing at startup.
            # Its status is independent of the snapshot used by this pull.
            producer_window = read_producer_window(
                self.directory / "render-metrics.json", self.run_id)
            if ack < self.last_ack or ack >= 10_000_000:
                raise RemoteVideoSourceError("remote applied-input ACK invalid")
            recorded_ns = time.perf_counter_ns()  # remote encoding-input timestamp
            payload, keyframe = self.encoder(pixels, self.coded_seq)
            encode_us = (time.perf_counter_ns() - recorded_ns) / 1000
            if (not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_UNIT_BYTES
                    or type(keyframe) is not bool or not payload.startswith(
                        (b"\x00\x00\x01", b"\x00\x00\x00\x01"))
                    or keyframe != _contains_idr(payload)):
                raise RemoteVideoSourceError("remote H.264 access unit invalid")
            coded = self.coded_seq
            wire = CCV1.pack(b"CCV1", 1, int(keyframe), coded,
                             round(coded * 1_000_000_000 / self.fps),
                             recorded_ns, len(payload)) + payload
            reply = {
                "v": 1, "kind": "video_access_unit", "codec": "h264-annexb",
                "run_id": self.run_id, "pull_seq": pull_seq, "input_ack": ack,
                "renderer_seq": renderer_seq,
                **({"renderer_frames_skipped": skipped_frames,
                    "renderer_buffer_lag": buffer_lag}
                   if self.sequence_buffered else {}),
                # Independent remote CLOCK DURATIONS; never subtract these
                # timestamps from host monotonic/perf-counter readings.
                "source_read_us": round(source_read_us, 2),
                "encode_us": round(encode_us, 2),
                "data": base64.b64encode(wire).decode("ascii"),
            }
            if producer_window is not None:
                reply["producer_window"] = producer_window
            self.coded_seq += 1
            self.last_ack = ack
            if self.sequence_buffered:
                self.last_renderer_seq = renderer_seq
            return reply
        except Exception:
            self.failed = True
            raise RemoteVideoSourceError("remote H.264 pull failed") from None
