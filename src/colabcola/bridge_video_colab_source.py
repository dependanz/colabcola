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


class RemoteH264PullSource:
    """Single ordered source per owned Comm; request IDs must increase.

    Encoded sequence is independent of source renderer_seq. Each successful
    video_pull encodes the latest RGB snapshot even if the same renderer
    snapshot was previously read. This keeps one access unit per pull and
    prevents gaps caused solely by renderer/pull cadence mismatch.
    """

    def __init__(self, directory: str | Path, run_id: str, *, fps: int,
                 width: int = WIDTH, height: int = HEIGHT, encoder_factory=None):
        if (not isinstance(run_id, str) or len(run_id) != 32
                or type(fps) is not int or not 1 <= fps <= 60
                or (width, height) != (WIDTH, HEIGHT)):
            raise RemoteVideoSourceError("invalid remote video source configuration")
        self.directory = Path(directory)
        self.run_id = run_id
        self.fps = fps
        self.width, self.height = width, height
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
            renderer_seq, ack, pixels = read_latest_rgb(
                self.directory / "frame.bin", width=self.width, height=self.height)
            if ack < self.last_ack or ack >= 10_000_000:
                raise RemoteVideoSourceError("remote applied-input ACK invalid")
            recorded_ns = time.perf_counter_ns()  # remote encoding-input timestamp
            payload, keyframe = self.encoder(pixels, self.coded_seq)
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
                "data": base64.b64encode(wire).decode("ascii"),
            }
            self.coded_seq += 1
            self.last_ack = ack
            return reply
        except Exception:
            self.failed = True
            raise RemoteVideoSourceError("remote H.264 pull failed") from None
