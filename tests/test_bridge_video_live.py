"""No-Colab tests for the live in-process H.264 source and local viewer path."""
from __future__ import annotations

import queue
import threading
import time

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
        (96, 64, 12, 10801), (96, 64, 1, 181),
        (True, 64, 12, 12),
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



def test_local_callback_timestamp_travels_through_encoded_packet_to_gui_queue():
    _codec_available()
    from colabcola.bridge_viewer_stats import LocalFrameTiming, ViewerPresentationMeter

    events = queue.Queue(maxsize=16)
    callback_ns = time.perf_counter_ns()
    events.put_nowait({
        "event": "mouse", "x": .75, "y": .25,
        "_ui_callback_ns": callback_ns,
    })
    images = queue.Queue(maxsize=64)
    result = run_live_stream(
        events, images, threading.Event(), width=96, height=64,
        fps=12, max_frames=12, timed_frames=True,
    )
    assert result["decoded_frames"] == 12
    assert result["input_callback_to_decoded_p50_ms"] is not None
    assert result["input_callback_to_decoded_p50_ms"] >= 0

    timed = []
    while not images.empty():
        kind, data = images.get_nowait()
        if kind == "frame-timed":
            timed.append(data)
    assert len(timed) == 12
    assert all(isinstance(sample[1], LocalFrameTiming) for sample in timed)
    assert all(sample[0].startswith(b"P6\n96 64\n255\n") for sample in timed)
    assert timed[0][1].input_callback_ns == callback_ns

    meter = ViewerPresentationMeter(started_at=10.0)
    for pixels, timing in timed[:2]:
        meter.presented(1.0)
        meter.configured_timed_frame(timing, time.perf_counter_ns())
    summary = meter.report(12.0)
    assert "input callback→Tk p50" in summary
    assert "(1 changed frames)" in summary


def test_live_h264_tk_viewer_opens_and_exits_cleanly():
    """Exercise the actual GUI consumer under Xvfb or a Windows desktop."""
    _codec_available()
    import tkinter as tk
    try:
        probe = tk.Tk()
        probe.withdraw()
        probe.destroy()
    except tk.TclError as exc:
        pytest.skip(f"no Tk graphical display available: {exc}")
    from colabcola.bridge_demo import run_viewer
    run_viewer(
        "127.0.0.1", 0, "", width=96, height=64,
        title="Colab Cola · Live H.264 Tk smoke",
        worker=lambda outgoing, frames, stopped: run_live_stream(
            outgoing, frames, stopped, width=96, height=64,
            fps=12, max_frames=8,
        ),
        viewer_stats=True,
    )
