"""v35 offline acceptance: opt-in 60Hz push + pooled GUI latency/pacing."""
from __future__ import annotations

import argparse
import math

import pytest

from colabcola.bridge_colab_attach import validate_attach_options
from colabcola.bridge_comm import KernelCommBridge
from colabcola.bridge_video_push_runtime import CreditPushSender
from colabcola.bridge_video_push_session import run_push_comm_h264
from colabcola.bridge_viewer_stats import (
    LocalFrameTiming, ViewerPresentationMeter, pooled_percentiles,
)
from colabcola.lifecycle_io import LifecycleError

RUN_ID = "f" * 32


def args_for(*, fps=60, credits=10, push=True, benchmark=False):
    return argparse.Namespace(
        transport="comm", video_codec="h264", benchmark_h264_capacity=benchmark,
        fps=fps, poll_hz=fps, inflight=1, confirm_allocation=False,
        preflight_only=True, confirm_attach=False, push_video=push,
        sequence_buffered=push, renderer_lag_frames=1 if push else 2,
        push_credits=credits, benchmark_h264_credits=False,
    )


@pytest.mark.parametrize("credits", [8, 10, 12])
def test_external_opt_in_60hz_push_is_validated_at_attach(credits):
    validate_attach_options(args_for(credits=credits))


@pytest.mark.parametrize("fps,credits", [
    (60, 4), (60, 6), (60, 9), (60, 13), (30, 10), (30, 12),
])
def test_external_attachment_rejects_mismatched_rate_and_credits(fps, credits):
    with pytest.raises(LifecycleError):
        validate_attach_options(args_for(fps=fps, credits=credits))


def test_legacy_30hz_pull_sweep_is_still_accepted():
    validate_attach_options(args_for(
        fps=30, credits=4, push=False, benchmark=True))
    with pytest.raises(LifecycleError):
        validate_attach_options(args_for(
            fps=60, credits=4, push=False, benchmark=True))


@pytest.mark.parametrize("fps,credits", [(30, 4), (30, 6), (30, 8),
                                        (60, 8), (60, 10), (60, 12)])
def test_host_bridge_accepts_only_intended_pairs(fps, credits):
    bridge = KernelCommBridge(
        None, None, RUN_ID, fps=fps, video_codec="h264",
        sequence_buffered=True, video_push=True, push_credits=credits)
    assert (bridge.fps, bridge.push_credits) == (fps, credits)


@pytest.mark.parametrize("fps,credits", [(30, 10), (30, 12), (60, 4),
                                        (60, 6), (60, 9)])
def test_host_bridge_rejects_invalid_pairs_before_any_connection(fps, credits):
    with pytest.raises(ValueError):
        KernelCommBridge(
            None, None, RUN_ID, fps=fps, video_codec="h264",
            sequence_buffered=True, video_push=True, push_credits=credits)


@pytest.mark.parametrize("fps,credits", [(60, 8), (60, 10), (60, 12)])
def test_60hz_cli_preflight_makes_no_provider_calls(monkeypatch, fps, credits):
    from colabcola.cli import main
    from colabcola import bridge_colab
    observed = []
    monkeypatch.setattr(bridge_colab, "run_colab",
                        lambda args: observed.append(args) or 0)
    argv = ["--profile", "high-credit-dev", "bridge", "colab",
            "--attach-existing", "--sequence-buffered", "--push-video",
            "--push-credits", str(credits), "--renderer-lag-frames", "1",
            "--transport", "comm", "--video-codec", "h264",
            "--fps", str(fps), "--poll-hz", str(fps), "--inflight", "1",
            "--preflight-only", "--max-seconds", "120"]
    assert main(argv) == 0
    assert len(observed) == 1
    assert observed[0].fps == fps
    assert observed[0].push_credits == credits
    assert observed[0].confirm_attach is False


@pytest.mark.parametrize("mutate", [
    lambda a: [x for x in a if x != "--push-video"],
    lambda a: [x for x in a if x != "--attach-existing"],
    lambda a: a + ["--benchmark-h264-capacity"],
    lambda a: a + ["--confirm-allocation"],
    lambda a: a[:a.index("--push-credits")+1] + ["6"] +
              a[a.index("--push-credits")+2:],
])
def test_cli_rejects_unsafe_60hz_modes(monkeypatch, mutate):
    from colabcola.cli import main
    from colabcola import bridge_colab
    monkeypatch.setattr(bridge_colab, "run_colab",
                        lambda args: pytest.fail("must not reach provider"))
    argv = ["--profile", "high-credit-dev", "bridge", "colab",
            "--attach-existing", "--sequence-buffered", "--push-video",
            "--push-credits", "10", "--renderer-lag-frames", "1",
            "--transport", "comm", "--video-codec", "h264",
            "--fps", "60", "--poll-hz", "60", "--inflight", "1",
            "--preflight-only", "--max-seconds", "120"]
    with pytest.raises(SystemExit):
        main(mutate(argv))


def test_pooled_quantiles_are_interpolated_individual_observations():
    assert pooled_percentiles([]) == {
        "p50": None, "p95": None, "p99": None}
    assert pooled_percentiles([12]) == {
        "p50": 12, "p95": 12, "p99": 12}
    assert pooled_percentiles([0, 10, 20, 30, 40]) == {
        "p50": 20, "p95": 38, "p99": 39.6}
    for value in (True, -1, math.inf, math.nan, "1"):
        with pytest.raises(ValueError):
            pooled_percentiles([value])


def test_session_gui_statistics_survive_window_reset_and_deduplicate_inputs():
    meter = ViewerPresentationMeter(started_at=10.0)
    samples = [
        (1_050_000_000, 1_045_000_000, 1_047_000_000, 1_020_000_000),
        (1_070_000_000, 1_065_000_000, 1_067_000_000, 1_020_000_000),
        (1_100_000_000, 1_095_000_000, 1_097_000_000, 1_080_000_000),
        (1_200_000_000, 1_195_000_000, 1_197_000_000, None),
    ]
    for configured, received, decoded, input_ns in samples[:2]:
        meter.presented(2.0, configured_ns=configured)
        meter.configured_timed_frame(
            LocalFrameTiming(received, decoded, input_ns), configured)
    assert meter.report(12.0) is not None
    for configured, received, decoded, input_ns in samples[2:]:
        meter.presented(2.0, configured_ns=configured)
        meter.configured_timed_frame(
            LocalFrameTiming(received, decoded, input_ns), configured)

    result = meter.session_summary(12.5)
    assert result["gui_configured_frames"] == 4
    assert result["gui_frame_spacing_observations"] == 3
    assert result["gui_frame_spacing_p50_ms"] == 30.0
    assert result["gui_frame_spacing_p95_ms"] == 93.0
    assert result["host_receipt_to_tk_observations"] == 4
    assert result["host_receipt_to_tk_p50_ms"] == 5.0
    assert result["input_callback_to_tk_observations"] == 2
    assert result["input_callback_to_tk_p50_ms"] == 25.0
    assert result["input_callback_to_tk_p95_ms"] == 29.5
    assert result["input_callback_to_tk_p99_ms"] == 29.9
    assert "not physical display scanout" in result["timing_scope"]


def test_session_metrics_reject_nonmonotone_gui_configurations():
    meter = ViewerPresentationMeter(started_at=1.0)
    meter.presented(2.0, configured_ns=1_000_000)
    with pytest.raises(ValueError):
        meter.presented(2.0, configured_ns=1_000_000)
    assert meter.session_count == 1
    assert meter.session_summary(2.0)["gui_frame_spacing_observations"] == 0



@pytest.mark.parametrize("capacity", [10, 12])
def test_60hz_credit_sender_cannot_exceed_configured_window(tmp_path, capacity):
    import struct
    from types import SimpleNamespace

    class Source:
        sequence_buffered = True
        last_renderer_seq = None
        def pull(self, send_seq):
            self.last_renderer_seq = send_seq - 1
            return {"kind": "video_access_unit", "pull_seq": send_seq}

    class Comm:
        def __init__(self):
            self.sent = []
        def send(self, packet):
            self.sent.append(packet)

    from colabcola.bridge_video_push_runtime import CreditPushSender
    source, comm = Source(), Comm()
    sender = CreditPushSender(comm=comm, source=source,
                              directory=tmp_path, run_id=RUN_ID,
                              credits=capacity)
    sender.grant(capacity)
    with pytest.raises(ValueError):
        sender.grant(1)  # No frame sent: never issue speculative credit.
    for seq in range(capacity):
        (tmp_path / "frame.bin").write_bytes(
            struct.pack("!4sIIII", b"CCB1", seq, 0, 640, 360))
        assert sender.step()
    assert sender.available == 0
    with pytest.raises(ValueError):
        sender.grant(2)
    sender.grant(1)
    assert sender.available == 1
    with pytest.raises(ValueError):
        sender.grant(capacity)  # Initial grant cannot be replayed.
    (tmp_path / "frame.bin").write_bytes(
        struct.pack("!4sIIII", b"CCB1", capacity, 0, 640, 360))
    assert sender.step()
    assert len(comm.sent) == capacity + 1
    assert sender.available == 0
    assert sender.available + sender.sent_count - sender.replenished == capacity
    sender.close()


@pytest.mark.parametrize("credits", [8, 10, 12])
def test_remote_kernel_60hz_configuration_is_valid_without_gpu(credits):
    # Validate the remote bootstrap's pre-render arguments using a
    # fake runpy loader. A deliberately invalid renderer path must not run.
    from colabcola import bridge_colab_comm_runtime as remote
    from pathlib import Path
    from unittest.mock import patch

    with patch.object(remote.runpy, "run_path", side_effect=RuntimeError(
            "fake renderer loaded; no kernel/GPU code executed")):
        with pytest.raises(RuntimeError, match="fake renderer loaded"):
            remote.start(
                directory="/nonexistent/v35",
                run_id=RUN_ID, fps=60, renderer_path="/nonexistent/renderer.py",
                sequence_buffered=True, preflight_video=True,
                video_source_path="/nonexistent/h264.py",
                video_push=True, video_push_source_path="/nonexistent/push.py",
                push_credits=credits)


@pytest.mark.parametrize("fps,credits", [(60, 6), (60, 4), (30, 10)])
def test_remote_kernel_rejects_bad_rate_credit_before_renderer(fps, credits):
    from colabcola import bridge_colab_comm_runtime as remote
    from unittest.mock import patch

    with patch.object(remote.runpy, "run_path", side_effect=AssertionError(
            "remote renderer must not be imported")):
        with pytest.raises(ValueError, match="sender FPS/credit"):
            remote.start(
                directory="/nonexistent/v35",
                run_id=RUN_ID, fps=fps, renderer_path="/nonexistent/renderer.py",
                sequence_buffered=True, preflight_video=True,
                video_source_path="/nonexistent/h264.py",
                video_push=True, video_push_source_path="/nonexistent/push.py",
                push_credits=credits)
