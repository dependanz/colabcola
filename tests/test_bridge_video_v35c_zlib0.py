"""v35C opt-in zlib level 0 intermediate frames: offline safety and parity.

No Colab assignment, CUDA allocation, network connection, or apt install.
"""
from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path
from unittest.mock import patch

import pytest

from colabcola.bridge_colab_attach import validate_attach_options
from colabcola.bridge_colab_runtime import (
    HEADER, HEIGHT, MAGIC, WIDTH, encode_intermediate_ppm,
)
from colabcola.bridge_comm import KernelCommBridge
from colabcola.bridge_video_colab_source import read_latest_rgb
from colabcola.lifecycle_io import LifecycleError


RUN = "c" * 32


def cli_args(*, fps=60, enabled=True, push=True, buffered=True,
             credits=10, benchmark=False):
    return argparse.Namespace(
        transport="comm", video_codec="h264", benchmark_h264_capacity=benchmark,
        benchmark_h264_credits=False, sequence_buffered=buffered,
        push_video=push, producer_zlib_level0=enabled,
        fps=fps, poll_hz=fps, inflight=1, push_credits=credits,
        confirm_allocation=False, preflight_only=True, confirm_attach=False,
        renderer_lag_frames=1 if buffered else 2,
    )


def test_level0_opt_in_only_on_60hz_external_push():
    validate_attach_options(cli_args())
    validate_attach_options(cli_args(enabled=False))
    validate_attach_options(cli_args(fps=30, enabled=False, credits=6))
    for args in (
        cli_args(fps=30, credits=6),
        cli_args(push=False, benchmark=True, credits=4),
        cli_args(buffered=False),
    ):
        with pytest.raises(LifecycleError):
            validate_attach_options(args)


def test_host_bridge_rejects_unsupported_compression_modes():
    valid = KernelCommBridge(
        None, None, RUN, fps=60, video_codec="h264",
        sequence_buffered=True, video_push=True,
        push_credits=10, producer_zlib_level0=True)
    assert valid.producer_zlib_level0 is True

    for kwargs in (
        dict(fps=30, video_codec="h264", sequence_buffered=True,
             video_push=True, push_credits=6, producer_zlib_level0=True),
        dict(fps=60, video_codec="h264", sequence_buffered=True,
             video_push=False, push_credits=4, producer_zlib_level0=True),
        dict(fps=60, video_codec="h264", sequence_buffered=True,
             video_push=True, push_credits=10, producer_zlib_level0=1),
    ):
        with pytest.raises(ValueError):
            KernelCommBridge(None, None, RUN, **kwargs)


def test_remote_kernel_rejects_invalid_level0_before_loading_renderer():
    from colabcola import bridge_colab_comm_runtime as remote
    with patch.object(remote.runpy, "run_path", side_effect=AssertionError(
            "renderer must not run")) as loader:
        for fps, video_push in ((30, True), (60, False)):
            with pytest.raises(ValueError, match="level0 compression"):
                remote.start(
                    directory="/nonexistent", run_id=RUN, fps=fps,
                    renderer_path="/nonexistent/renderer.py",
                    sequence_buffered=True, preflight_video=True,
                    video_push=video_push,
                    video_push_source_path="/nonexistent/push.py" if video_push else None,
                    video_source_path="/nonexistent/encoder.py",
                    push_credits=6 if fps == 30 else 4,
                    producer_zlib_level0=True,
                )
        loader.assert_not_called()


def test_valid_level0_reaches_renderer_loader_but_not_gpu():
    from colabcola import bridge_colab_comm_runtime as remote
    with patch.object(remote.runpy, "run_path", side_effect=RuntimeError(
            "fake renderer selected")) as loader:
        with pytest.raises(RuntimeError, match="fake renderer selected"):
            remote.start(
                directory="/nonexistent", run_id=RUN, fps=60,
                renderer_path="/nonexistent/renderer.py",
                sequence_buffered=True, preflight_video=True,
                video_push=True,
                video_push_source_path="/nonexistent/push.py",
                video_source_path="/nonexistent/encoder.py",
                push_credits=10, producer_zlib_level0=True,
            )
        loader.assert_called_once()


@pytest.mark.parametrize("level", [0, 1])
def test_zlib_intermediate_roundtrip_in_existing_ccb1_decoder(tmp_path, level):
    pixels = bytes(range(256)) * (WIDTH * HEIGHT * 3 // 256)
    assert len(pixels) == WIDTH * HEIGHT * 3
    prefix = f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii")
    ppm = prefix + pixels
    compressed = encode_intermediate_ppm(ppm, zlib_level=level)
    assert 0 < len(compressed) <= 2 * 1024 * 1024
    assert zlib.decompress(compressed) == ppm
    (tmp_path / "frame.bin").write_bytes(
        HEADER.pack(MAGIC, 17, 4, WIDTH, HEIGHT) + compressed)
    assert read_latest_rgb(tmp_path / "frame.bin") == (17, 4, pixels)


def test_level0_and_level1_restore_byte_identical_rgb_and_keep_bounds(tmp_path):
    pixels = bytes([15, 200, 50]) * (WIDTH * HEIGHT)
    ppm = f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii") + pixels
    raw = encode_intermediate_ppm(ppm, zlib_level=0)
    compressed = encode_intermediate_ppm(ppm, zlib_level=1)
    assert len(raw) > len(compressed)
    assert len(raw) <= 2 * 1024 * 1024
    assert zlib.decompress(raw) == zlib.decompress(compressed) == ppm
    for level, blob in ((0, raw), (1, compressed)):
        path = tmp_path / f"frame-{level}.bin"
        path.write_bytes(HEADER.pack(MAGIC, 22, 9, WIDTH, HEIGHT) + blob)
        assert read_latest_rgb(path) == (22, 9, pixels)


@pytest.mark.parametrize("invalid", [-1, 2, True, 1.0, "0"])
def test_invalid_compression_levels_fail_closed(invalid):
    ppm = f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii") + b"\0" * (WIDTH * HEIGHT * 3)
    with pytest.raises(ValueError):
        encode_intermediate_ppm(ppm, zlib_level=invalid)


def test_malformed_intermediate_frame_fails_closed():
    good = f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii") + b"\0" * (WIDTH * HEIGHT * 3)
    for malformed in (good[:-1], b"x" * len(good), good + b"\0"):
        with pytest.raises(ValueError):
            encode_intermediate_ppm(malformed, zlib_level=0)


def test_cli_producer_zlib_level0_preflight_is_gated_without_provider(monkeypatch):
    from colabcola.cli import main
    from colabcola import bridge_colab
    observed = []
    monkeypatch.setattr(bridge_colab, "run_colab",
                        lambda args: observed.append(args) or 0)
    base = [
        "--profile", "high-credit-dev", "bridge", "colab",
        "--attach-existing", "--sequence-buffered", "--push-video",
        "--producer-zlib-level0", "--push-credits", "10",
        "--renderer-lag-frames", "1", "--transport", "comm",
        "--video-codec", "h264", "--fps", "60", "--poll-hz", "60",
        "--inflight", "1", "--max-seconds", "120", "--preflight-only",
    ]
    assert main(base) == 0
    assert len(observed) == 1 and observed[0].producer_zlib_level0
    for invalid in (
        [x for x in base if x != "--push-video"],
        [x for x in base if x != "--attach-existing"],
        [x for x in base if x != "--sequence-buffered"],
        base + ["--confirm-allocation"],
        base + ["--benchmark-h264-capacity"],
    ):
        with pytest.raises(SystemExit):
            main(invalid)
    for idx in (base.index("--fps") + 1, base.index("--poll-hz") + 1):
        modified = list(base)
        modified[idx] = "30"
        with pytest.raises(SystemExit):
            main(modified)
    assert len(observed) == 1


def test_default_cli_preserves_zlib_level_one(monkeypatch):
    from colabcola.cli import main
    from colabcola import bridge_colab
    choices = []
    monkeypatch.setattr(bridge_colab, "run_colab",
                        lambda args: choices.append(args) or 0)
    argv = [
        "--profile", "high-credit-dev", "bridge", "colab",
        "--attach-existing", "--sequence-buffered", "--push-video",
        "--push-credits", "10", "--renderer-lag-frames", "1",
        "--transport", "comm", "--video-codec", "h264",
        "--fps", "60", "--poll-hz", "60", "--inflight", "1",
        "--max-seconds", "120", "--preflight-only",
    ]
    assert main(argv) == 0
    assert choices[0].producer_zlib_level0 is False


def test_confirmed_attach_preserves_external_ownership_and_level0_journal(
        tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from colabcola import bridge_colab_attach, bridge_comm

    class Provider:
        def assignments(self):
            return [SimpleNamespace(endpoint="external-t4", accelerator="T4")]
        def _session(self, endpoint):
            assert endpoint == "external-t4"
            return object()
    monkeypatch.setattr(bridge_colab_attach, "single_kernel_id",
                        lambda *_: "single-kernel")
    observed = []
    class Peer:
        def __init__(self, *args, **kwargs):
            observed.append(kwargs)
        def open(self, *, expected_kernel_id):
            assert expected_kernel_id == "single-kernel"
            return {"gpu": "Tesla T4"}
        def close(self):
            pass
    monkeypatch.setattr(bridge_comm, "KernelCommBridge", Peer)
    monkeypatch.setattr(bridge_colab_attach, "run_viewer",
                        lambda *a, **k: None)

    args = cli_args()
    args.run_dir = tmp_path / "unique-v35c-run"
    args.profile = "high-credit-dev"
    args.confirm_attach = True
    args.preflight_only = False
    args.max_seconds = 120
    args.install_remote_pyav = False
    args.viewer_stats = True
    assert bridge_colab_attach.run_attached_colab(args, Provider()) == 0
    assert observed[0]["producer_zlib_level0"] is True
    journal = json.loads((args.run_dir / "state.json").read_text())
    assert journal["producer_zlib_level"] == 0
    assert journal["ownership"] == "external"
    assert journal["phase"] == "detached"
    assert journal["close_status"] == "host_disconnected"
    assert "allocation_id" not in journal
