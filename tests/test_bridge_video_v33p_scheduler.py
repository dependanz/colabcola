"""v33p: allocation-free timing contract for experimental receive pacing."""
from __future__ import annotations

import pytest

from colabcola.bridge_video_credit_sweep import (
    EXPERIMENTAL_POLL_WAIT_SECONDS,
    POLL_WAIT_SECONDS,
    _empty_phase,
    _summarize_phase,
    receive_poll_timeout,
)


@pytest.mark.parametrize("rate", [30, 60])
def test_high_rate_poll_budget_fits_below_frame_period(rate):
    period = 1 / rate
    wait = receive_poll_timeout(
        experimental=True, draining=False, now=10.0, next_due=10.0 + period)
    assert wait == EXPERIMENTAL_POLL_WAIT_SECONDS
    assert 0.001 <= wait < period


def test_high_rate_receive_time_respects_nearby_send_deadline():
    assert receive_poll_timeout(
        experimental=True, draining=False,
        now=10.0, next_due=10.002) == pytest.approx(.002)
    assert receive_poll_timeout(
        experimental=True, draining=False,
        now=10.0, next_due=9.9) == .001


def test_legacy_and_phase_drain_preserve_original_poll_wait():
    for experimental, draining in ((False, False), (False, True), (True, True)):
        assert receive_poll_timeout(
            experimental=experimental, draining=draining,
            now=10.0, next_due=10.003) == POLL_WAIT_SECONDS


def test_phase_report_separates_poll_timing_from_pull_rtt():
    stats = _empty_phase()
    stats.update(decoded=4, replies=4, rtts=[140, 141, 141, 160],
                 receive_poll_elapsed_ms=[2, 4, 5, 8],
                 receive_poll_count=4, receive_polls_without_commit=2)
    report = _summarize_phase("C4", 4, stats, measured_s=2, end_gap_ms=0)
    assert report["decoded_fps"] == 2
    assert report["pull_rtt_p50_ms"] == 141
    assert report["host_receive_poll_elapsed_p50_ms"] == 4.5
    assert report["host_receive_poll_count"] == 4
    assert report["host_receive_polls_without_commit"] == 2
