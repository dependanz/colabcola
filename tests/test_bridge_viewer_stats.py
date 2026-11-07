"""Viewer-side telemetry tests; no T4 allocation or remote API calls."""
from __future__ import annotations

import pytest

from colabcola.bridge_viewer_stats import ViewerPresentationMeter


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
