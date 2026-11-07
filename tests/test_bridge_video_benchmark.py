"""CPU-only tests for the optional H.264 codec experiment."""
from __future__ import annotations

import shutil

import pytest

from colabcola.bridge_video_benchmark import (
    _validate_dimensions, compare_codecs, decode_h264,
    encode_h264, scene_like_rgb_frames, synthetic_rgb_frames,
)


def test_codec_frames_are_deterministic_and_bounded():
    a = synthetic_rgb_frames(64, 48, 8)
    b = synthetic_rgb_frames(64, 48, 8)
    assert a == b
    assert all(len(f) == 64 * 48 * 3 for f in a)
    assert a[0] != a[1]
    scene = scene_like_rgb_frames(64, 48, 8, 12)
    assert all(len(f) == 64 * 48 * 3 for f in scene)
    assert scene[0] != scene[1]
    for width, height, count in ((31, 48, 8), (63, 48, 8), (64, 47, 8),
                                 (64, 48, 0), (640, 360, 49), (True, 48, 8)):
        with pytest.raises(ValueError):
            _validate_dimensions(width, height, count)


def test_codec_rejects_malformed_inputs_before_invoking_ffmpeg():
    with pytest.raises(ValueError):
        encode_h264([b"bad"] * 8, 64, 48, 12)
    with pytest.raises(ValueError):
        encode_h264(synthetic_rgb_frames(64, 48, 8), 64, 48, 0)
    with pytest.raises(ValueError):
        compare_codecs(width=64, height=48, frames=8, pattern="bad")
    with pytest.raises(ValueError):
        compare_codecs(width=64, height=48, frames=8, fps=0)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="optional FFmpeg unavailable")
@pytest.mark.parametrize("pattern", ["scene", "square"])
def test_h264_actual_encode_decode_from_synthetic_frames(pattern):
    report = compare_codecs(width=96, height=64, frames=8, fps=12, pattern=pattern)
    assert report["frames"] == 8
    assert report["h264_bytes"] > 0
    assert report["h264_bytes"] < report["zlib_bytes"]
    assert report["psnr_db"] > 25
    assert report["scope"].startswith("offline synthetic")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="optional FFmpeg unavailable")
def test_h264_roundtrip_frame_count_no_decoder_duplicates():
    original = scene_like_rgb_frames(64, 48, 8, 12)
    encoded = encode_h264(original, 64, 48, 12)
    decoded = decode_h264(encoded, 64, 48)
    assert len(decoded) == 8
    assert all(len(v) == 64 * 48 * 3 for v in decoded)
