"""No-Colab tests for the live in-process H.264 source and local viewer path."""
from __future__ import annotations

import queue
import threading

import pytest

from colabcola.bridge_video_live import (
    LiveScene, validate_live, live_background, render_live_rgb,
    run_live_stream,
)


def test_live_input_validation_and_motion():
    scene = LiveScene()
    assert not scene.apply(None)
    assert not scene.apply({"event": "mouse", "x": float("nan"), "y": .2})
    assert not scene.apply({"event": "mouse", "x": 1.3, "y": .2})
    assert not scene.apply({"event": "key", "key": "Escape"})
    assert scene.apply({"event": "mouse", "x": .75, "y": .25})
    assert (scene.x, scene.y) == (.75, .25)
    assert scene.apply({"event": "key", "key": "Left"})
    assert scene.x == pytest.approx(.67)
    assert scene.apply({"event": "key", "key": "Down"})
    assert scene.y == pytest.approx(.33)


def test_live_limits_reject_invalid_inputs():
    for width, height, fps, frames in (
        (63, 64, 12, 12), (96, 47, 12, 12), (96, 64, 0, 12),
        (96, 64, 61, 12), (96, 64, 12, 3),
        (96, 64, 12, 241), (True, 64, 12, 12),
    ):
        with pytest.raises(ValueError):
            validate_live(width, height, fps, frames)


def test_live_source_renders_different_images_for_input_and_time():
    np = pytest.importorskip("numpy")
    base = live_background(96, 64, np)
    a = render_live_rgb(LiveScene(), base, 0)
    b = render_live_rgb(LiveScene(x=.85, y=.2), base, 0)
    c = render_live_rgb(LiveScene(), base, 4)
    assert a.shape == (64, 96, 3)
    assert a.tobytes() != b.tobytes()
    assert a.tobytes() != c.tobytes()


def _codec_available():
    av = pytest.importorskip("av", reason="optional PyAV unavailable")
    pytest.importorskip("numpy")
    try:
        av.CodecContext.create("libx264", "w")
    except Exception:
        pytest.skip("PyAV build lacks libx264 encoder")


def test_live_interactive_h264_roundtrips_input_while_source_runs():
    _codec_available()
    input_queue = queue.Queue(maxsize=16)
    input_queue.put_nowait({"event": "mouse", "x": .8, "y": .2})
    images = queue.Queue(maxsize=32)
    stopped = threading.Event()
    result = run_live_stream(
        input_queue, images, stopped,
        width=96, height=64, fps=12, max_frames=16,
    )
    assert stopped.is_set()
    assert result["produced_frames"] == result["decoded_frames"] == 16
    assert 5 <= result["live_decoded_fps"] <= 25
    assert result["capture_to_decoded_p50_ms"] is not None
    assert result["input_events_observed_in_decoded_frames"] >= 1
    assert result["input_applied_to_decoded_p50_ms"] is not None
    assert "no Colab" in result["scope"]

    frames = []
    while not images.empty():
        kind, data = images.get_nowait()
        if kind == "frame":
            frames.append(data)
    assert len(frames) >= 8
    assert all(frame.startswith(b"P6\n96 64\n255\n") for frame in frames)
    assert frames[0] != frames[-1]


def test_live_scene_headless_60_fps_decodes_without_gpu():
    _codec_available()
    result = run_live_stream(
        queue.Queue(), queue.Queue(maxsize=2), threading.Event(),
        width=96, height=64, fps=60, max_frames=24,
    )
    assert result["decoded_frames"] == 24
    assert 35 <= result["live_decoded_fps"] <= 100
