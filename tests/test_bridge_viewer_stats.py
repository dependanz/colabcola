"""Viewer-side telemetry tests; no T4 allocation or remote API calls."""
from __future__ import annotations

import pytest

from colabcola.bridge_viewer_stats import LocalFrameTiming, ViewerPresentationMeter


def test_gui_meter_counts_configured_images_independently_of_network_fps():
    meter = ViewerPresentationMeter(started_at=10.0)
    meter.presented(4.0)
    meter.presented(8.0)
    meter.presented(12.0)
    assert meter.report(11.9) is None
    message = meter.report(12.0)
    assert message == "GUI configured 1.5 FPS · Tk image+label p50 8.0 ms"
    # A reset after reporting is crucial: do not publish stale frame stats.
    assert meter.report(14.1) == "GUI configured 0.0 FPS · Tk image+label p50 n/a ms"
    meter.presented(5.0)
    assert "Tk image+label p50 5.0 ms" in meter.report(16.2)


def test_gui_meter_rejects_invalid_timings():
    meter = ViewerPresentationMeter(started_at=0)
    for bad in (-1.0, float("inf"), float("nan"), 60001.0):
        with pytest.raises(ValueError):
            meter.presented(bad)
    for bad in (0, -1, 35):
        with pytest.raises(ValueError):
            meter.report(2.5, every_seconds=bad)


def test_viewer_meter_is_opt_in_and_does_not_allocate_during_preflight(monkeypatch):
    from colabcola.cli import main
    from colabcola import bridge_colab
    selected = []

    monkeypatch.setattr(bridge_colab, "run_colab",
                        lambda args: selected.append(args.viewer_stats) or 0)
    base = ["--profile", "example", "bridge", "colab", "--preflight-only"]
    assert main(base + ["--transport", "comm", "--viewer-stats"]) == 0
    assert main(base) == 0
    assert selected == [True, False]


def test_local_input_callback_to_tk_configuration_is_measured_once():
    meter = ViewerPresentationMeter(started_at=10.0)
    sample = LocalFrameTiming(
        captured_ns=1_040_000_000,
        decoded_ns=1_043_000_000,
        input_callback_ns=1_020_000_000,
    )
    meter.presented(6.0)
    meter.configured_timed_frame(sample, 1_050_000_000)
    # The same applied input persists in later video frames, but must
    # not be double-counted as another input acknowledgment.
    meter.presented(5.0)
    meter.configured_timed_frame(
        LocalFrameTiming(1_070_000_000, 1_073_000_000, 1_020_000_000),
        1_080_000_000,
    )
    report = meter.report(12.0)
    assert "GUI configured 1.0 FPS" in report
    assert "capture→Tk p50 10.0 ms" in report
    assert "decode→Tk p50 7.0 ms" in report
    assert "input callback→Tk p50 30.0 ms (1 changed frames)" in report

    # Per-window frame timings reset, but the last acknowledged input
    # identity persists across reporting windows.
    meter.presented(5.0)
    meter.configured_timed_frame(
        LocalFrameTiming(1_090_000_000, 1_094_000_000, 1_020_000_000),
        1_100_000_000,
    )
    report = meter.report(14.1)
    assert "input callback→Tk" not in report


def test_viewer_timing_rejects_invalid_or_future_input_before_counting():
    meter = ViewerPresentationMeter(started_at=0)
    with pytest.raises(ValueError):
        meter.configured_timed_frame(
            LocalFrameTiming(100, 150, 110), 200,
        )  # Input cannot be applied after this frame's capture.
    with pytest.raises(ValueError):
        meter.configured_timed_frame(
            LocalFrameTiming(150, 100, None), 200,
        )
    with pytest.raises(ValueError):
        meter.configured_timed_frame(
            LocalFrameTiming(100, 150, True), 200,
        )
    assert meter.report(2.0) == (
        "GUI configured 0.0 FPS · Tk image+label p50 n/a ms"
    )


def test_console_viewer_summary_is_cp1252_safe_without_losing_metrics():
    import io
    from colabcola.bridge_viewer_stats import console_safe_viewer_summary

    meter = ViewerPresentationMeter(started_at=10.0)
    meter.presented(1.25)
    meter.configured_timed_frame(
        LocalFrameTiming(captured_ns=1_040_000_000,
                         decoded_ns=1_043_000_000,
                         input_callback_ns=1_020_000_000),
        now_ns=1_050_000_000,
    )
    summary = meter.report(12.0)
    assert "capture→Tk" in summary
    safe = console_safe_viewer_summary(summary)
    assert "capture->Tk" in safe
    assert "decode->Tk" in safe
    assert "input callback->Tk p50 30.0 ms" in safe
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    print("[colabcola] " + safe, file=stream, flush=True)
    assert "input callback->Tk" in stream.buffer.getvalue().decode("cp1252")
