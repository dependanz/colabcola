"""Optional, local-only in-process H.264 codec comparison."""
from __future__ import annotations

import pytest

from colabcola.bridge_video_inprocess import compare_inprocess


def test_inprocess_requires_valid_configuration_without_importing_pyav():
    for kwargs in ({"fps": 0}, {"fps": 61}, {"pattern": "invalid"},
                   {"width": 63}, {"height": 47}, {"frames": 49}):
        arguments = {"width": 64, "height": 48, "frames": 8,
                     "fps": 12, "pattern": "square"}
        arguments.update(kwargs)
        with pytest.raises(ValueError):
            compare_inprocess(**arguments)


def _pyav_available():
    av = pytest.importorskip("av", reason="optional PyAV not installed")
    pytest.importorskip("numpy", reason="NumPy not installed")
    try:
        av.CodecContext.create("libx264", "w")
    except Exception:
        pytest.skip("PyAV build does not include libx264 encoder")


@pytest.mark.parametrize("fps,frames", [(12, 24), (60, 48)])
def test_inprocess_streams_before_end_with_direct_codec_packets(fps, frames):
    _pyav_available()
    report = compare_inprocess(width=96, height=64, frames=frames,
                               fps=fps, pattern="square")
    assert report["decoded_frames"] == frames
    assert report["decoded_before_source_finished"] >= frames // 2
    assert report["h264_bytes"] > 0
    assert report["h264_bytes"] < report["zlib_bytes"]
    assert report["capture_to_decoded_p50_ms"] > 0
    for key in ("capture_to_packet_p50_ms", "packet_to_socket_p50_ms",
                "socket_to_decoded_p50_ms"):
        assert report[key] >= 0
    assert report["scope"].startswith("in-process optional PyAV")


def test_inprocess_scene_decodes_all_frames():
    _pyav_available()
    report = compare_inprocess(width=64, height=48, frames=8,
                               fps=12, pattern="scene")
    assert report["decoded_frames"] == 8
    assert report["keyframes"] >= 1
