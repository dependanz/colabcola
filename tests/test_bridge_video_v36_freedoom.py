"""v36A Freedoom adapter offline contract tests. No Colab/Xvfb/game launched."""
from __future__ import annotations

import argparse
import queue
import struct
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from colabcola.bridge_colab_attach import validate_attach_options
from colabcola.bridge_comm import KernelCommBridge
from colabcola.bridge_game_freedoom_runtime import (
    DISPLAY, FRAME_BYTES, FRAME_HEADER, FRAME_PREFIX, FreedoomSource,
    KEY_MAP, MAX_EVENTS, validate_event, _read_exact,
)
from colabcola.bridge_video_colab_source import read_latest_rgb
from colabcola.lifecycle_io import LifecycleError

RUN = "a" * 32


def options(*, app="freedoom", fps=30, credits=6, push=True, buffered=True,
            zlib0=False):
    return argparse.Namespace(
        application=app, transport="comm", video_codec="h264", fps=fps,
        poll_hz=fps, inflight=1, push_video=push,
        sequence_buffered=buffered, producer_zlib_level0=zlib0,
        push_credits=credits, benchmark_h264_capacity=False,
        benchmark_h264_credits=False, confirm_allocation=False,
        preflight_only=True, confirm_attach=False,
        renderer_lag_frames=1 if buffered else 2,
    )


def test_explicit_game_attachment_guard_preserves_synthetic():
    validate_attach_options(options())
    validate_attach_options(options(app="synthetic"))
    with pytest.raises(LifecycleError):
        validate_attach_options(options(fps=60, credits=10))
    with pytest.raises(LifecycleError):
        validate_attach_options(options(push=False))
    with pytest.raises(LifecycleError):
        validate_attach_options(options(buffered=False))
    with pytest.raises(LifecycleError):
        validate_attach_options(options(zlib0=True))


@pytest.mark.parametrize("credits", [4, 6, 8])
def test_explicit_game_bridge_configuration(credits):
    peer = KernelCommBridge(
        None, None, RUN, fps=30, video_codec="h264",
        sequence_buffered=True, video_push=True,
        push_credits=credits, application="freedoom")
    assert peer.application == "freedoom"
    assert peer.push_credits == credits


@pytest.mark.parametrize("kwargs", [
    {"fps": 60, "push_credits": 10},
    {"fps": 30, "push_credits": 10},
    {"fps": 30, "video_push": False},
    {"fps": 30, "sequence_buffered": False},
    {"fps": 30, "producer_zlib_level0": True},
    {"fps": 30, "application": "untrusted"},
])
def test_game_bridge_rejects_unsupported_modes(kwargs):
    base = dict(fps=30, video_codec="h264", sequence_buffered=True,
                video_push=True, push_credits=6, application="freedoom")
    base.update(kwargs)
    with pytest.raises(ValueError):
        KernelCommBridge(None, None, RUN, **base)


@pytest.mark.parametrize("credits", [4, 6, 8])
def test_game_cli_is_preflight_only_until_explicit_confirm(monkeypatch, credits):
    from colabcola.cli import main
    from colabcola import bridge_colab
    seen = []
    monkeypatch.setattr(bridge_colab, "run_colab",
                        lambda args: seen.append(args) or 0)
    args = ["--profile", "high-credit-dev", "bridge", "colab",
            "--application", "freedoom", "--attach-existing",
            "--sequence-buffered", "--push-video", "--push-credits",
            str(credits), "--renderer-lag-frames", "1", "--transport", "comm",
            "--video-codec", "h264", "--fps", "30", "--poll-hz", "30",
            "--inflight", "1", "--max-seconds", "120", "--preflight-only"]
    assert main(args) == 0
    assert seen[-1].application == "freedoom"
    assert seen[-1].confirm_attach is False
    assert seen[-1].preflight_only is True


def test_game_cli_rejects_benchmarks_managed_mode_and_level0(monkeypatch):
    from colabcola.cli import main
    from colabcola import bridge_colab
    monkeypatch.setattr(bridge_colab, "run_colab",
                        lambda args: pytest.fail("must fail before provider"))
    base = ["--profile", "example", "bridge", "colab",
            "--application", "freedoom", "--attach-existing",
            "--sequence-buffered", "--push-video", "--push-credits", "6",
            "--transport", "comm", "--video-codec", "h264", "--fps", "30",
            "--poll-hz", "30", "--inflight", "1", "--max-seconds", "120",
            "--preflight-only"]
    invalid = (
        [x for x in base if x != "--attach-existing"],
        base + ["--benchmark-h264-capacity"],
        base + ["--producer-zlib-level0"],
        base + ["--confirm-allocation"],
        base[:base.index("--fps") + 1] + ["60"] +
        base[base.index("--fps") + 2:],
    )
    for args in invalid:
        with pytest.raises(SystemExit):
            main(args)


@pytest.mark.parametrize("kind,key", [
    ("key_down", "Up"), ("key_up", "Up"),
    ("key_down", "Control_L"), ("key_up", "space"),
    ("key_down", "Escape"), ("key_up", "Tab"),
])
def test_game_key_contract_is_explicit(kind, key):
    event = {"seq": 17, "event": kind, "key": key}
    assert validate_event(event) == event


@pytest.mark.parametrize("event", [
    {"seq": True, "event": "key_down", "key": "Up"},
    {"seq": 1, "event": "key_down", "key": "F12"},
    {"seq": 1, "event": "key_down", "key": "Up", "cmd": "rm"},
    {"seq": 1, "event": "key", "key": "Up"},
    {"seq": 1, "event": "mouse_delta", "dx": 100, "dy": 0},
    {"seq": 1, "event": "mouse_delta", "dx": 1.0, "dy": 0},
    {"seq": 1, "event": "mouse_button_down", "button": True},
    {"seq": 1, "event": "mouse_button_up", "button": 2},
    {"seq": 1, "event": "mouse_delta", "dx": 1, "dy": 0, "extra": 1},
])
def test_game_input_rejects_malformed_events(event):
    with pytest.raises(ValueError):
        validate_event(event)


def test_game_input_queued_in_order_without_coalescing(tmp_path):
    source = FreedoomSource(tmp_path, RUN)
    assert source.submit_input({"seq": 1, "event": "key_down", "key": "Up"})
    assert source.submit_input({"seq": 2, "event": "key_up", "key": "Up"})
    assert not source.submit_input({"seq": 2, "event": "key_up", "key": "Up"})
    assert source.events.get_nowait()["event"] == "key_down"
    assert source.events.get_nowait()["event"] == "key_up"
    source.close()


def test_game_input_rejects_full_queue_before_changing_queued_seq(tmp_path):
    source = FreedoomSource(tmp_path, RUN)
    for seq in range(1, MAX_EVENTS + 1):
        assert source.submit_input({"seq": seq, "event": "key_down", "key": "Up"})
    assert not source.submit_input({
        "seq": MAX_EVENTS + 1, "event": "key_up", "key": "Up"})
    assert source._latest_queued == MAX_EVENTS
    source.close()


def test_game_input_executor_only_uses_allowlisted_commands(tmp_path):
    source = FreedoomSource(tmp_path, RUN)
    commands = []
    source._xdotool_command = lambda *args: commands.append(args)
    source._apply({"seq": 1, "event": "key_down", "key": "Up"})
    source._apply({"seq": 2, "event": "key_down", "key": "Up"})
    source._apply({"seq": 3, "event": "mouse_delta", "dx": 6, "dy": -3})
    source._apply({"seq": 4, "event": "mouse_button_down", "button": 1})
    source._release_all()
    assert commands == [
        ("keydown", "Up"),
        ("mousemove_relative", "--", "6", "-3"),
        ("mousedown", "1"),
        ("keyup", "Up"),
        ("mouseup", "1"),
    ]
    assert source._applied_seq == 4
    source.close()


def test_existing_h264_source_accepts_game_ccb1_frame(tmp_path):
    pixels = bytes((17, 42, 201)) * (FRAME_BYTES // 3)
    packed = FRAME_HEADER.pack(b"CCB1", 9, 4, 640, 360)
    (tmp_path / "frame.bin").write_bytes(
        packed + zlib.compress(FRAME_PREFIX + pixels, level=1))
    assert read_latest_rgb(tmp_path / "frame.bin") == (9, 4, pixels)


def test_game_source_rejects_wrong_geometry_fps_or_identity(tmp_path):
    for kwargs in (
        {"fps": 60}, {"sequence_buffered": False},
        {"run_id": "invalid"}, {"fps": True},
    ):
        values = dict(directory=tmp_path, run_id=RUN,
                      fps=30, sequence_buffered=True)
        values.update(kwargs)
        with pytest.raises(ValueError):
            FreedoomSource(**values)


def test_remote_bootstrap_rejects_unselected_game_without_loading_renderer():
    from colabcola import bridge_colab_comm_runtime as remote
    with patch.object(remote.runpy, "run_path", side_effect=AssertionError(
            "must not load game or renderer")) as loader:
        with pytest.raises(ValueError):
            remote.start(
                directory="/not-real", run_id=RUN, fps=60,
                renderer_path="/not-real/renderer.py",
                video_source_path="/not-real/source.py",
                preflight_video=True, sequence_buffered=True,
                video_push=True,
                video_push_source_path="/not-real/push.py",
                push_credits=10, application="freedoom",
                game_renderer_path="/not-real/game.py")
        loader.assert_not_called()


def test_remote_bootstrap_selects_game_module_before_cuda_work():
    from colabcola import bridge_colab_comm_runtime as remote
    chosen = []
    def loader(path):
        chosen.append(path)
        raise RuntimeError("game module selected")
    with patch.object(remote.runpy, "run_path", side_effect=loader):
        with pytest.raises(RuntimeError, match="game module selected"):
            remote.start(
                directory="/not-real", run_id=RUN, fps=30,
                renderer_path="/not-real/renderer.py",
                video_source_path="/not-real/source.py",
                preflight_video=True, sequence_buffered=True,
                video_push=True, push_credits=6,
                video_push_source_path="/not-real/push.py",
                application="freedoom", game_renderer_path="/not-real/game.py")
    assert chosen == ["/not-real/game.py"]


def test_game_no_automatic_package_install_or_network_open():
    import inspect
    import colabcola.bridge_game_freedoom_runtime as game
    text = inspect.getsource(game)
    assert "pip install" not in text
    assert "apt-get" not in text
    assert "shell=True" not in text
    assert "requests." not in text


def test_missing_game_dependency_fails_without_child_processes(tmp_path, monkeypatch):
    from colabcola import bridge_game_freedoom_runtime as mod
    def unavailable():
        raise RuntimeError("freedoom dependency missing: Xvfb")
    monkeypatch.setattr(mod, "binary_inventory", unavailable)
    source = FreedoomSource(tmp_path / "isolated", RUN)
    worker = source.start()
    worker.join(timeout=2)
    assert not worker.is_alive()
    status = __import__("json").loads(
        (source.root / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["error_type"] == "RuntimeError"
    assert status["application"] == "freedoom"
    assert not source._processes
    source.close()


def test_game_capture_pipe_read_is_exact_and_fails_on_truncation():
    import io
    assert _read_exact(io.BytesIO(b"xyz123"), 6) == b"xyz123"
    with pytest.raises(RuntimeError, match="pipe ended"):
        _read_exact(io.BytesIO(b"xyz"), 6)


def test_game_source_rejects_arbitrary_key_types():
    with pytest.raises(ValueError):
        validate_event({"seq": 1, "event": "key_down", "key": ["Up"]})


def test_doom_warp_arguments_match_freedoom_episode_or_map():
    from colabcola.bridge_game_freedoom_runtime import (
        IWAD_CANDIDATES, doom_warp_arguments)
    assert doom_warp_arguments(IWAD_CANDIDATES[0]) == ("-warp", "1", "1")
    assert doom_warp_arguments(IWAD_CANDIDATES[1]) == ("-warp", "1")
    with pytest.raises(ValueError):
        doom_warp_arguments("/home/user/custom.wad")


def test_host_game_input_does_not_coalesce_key_releases(monkeypatch):
    import threading
    import time
    from colabcola import bridge_video_push_session as mod
    done = threading.Event()
    connected = threading.Event()
    connected.set()
    sent = []

    class EmptyChannel:
        def get_msg(self, timeout=0):
            raise queue.Empty

    class Peer:
        fps = 30
        application = "freedoom"
        video_push = True
        video_codec = "h264"
        push_credits = 6
        run_id = RUN
        comm_id = "owned"
        wire = SimpleNamespace(connection_ready=connected,
                               iopub_channel=EmptyChannel())
        def _send(self, kind, **data):
            sent.append((kind, data))
        def send_input(self, seq, event):
            sent.append(("input", seq, event))
            if seq == 2:
                done.set()

    class FakeDecoder:
        pass

    monkeypatch.setattr(mod, "make_host_decoder",
                        lambda: (None, FakeDecoder()))
    events = queue.Queue()
    events.put({"event": "key_down", "key": "Up"})
    events.put({"event": "key_up", "key": "Up"})
    result = mod.run_push_comm_h264(
        Peer(), "Tesla T4", events, queue.Queue(),
        done, time.monotonic() + 0.5, credits=6)
    assert result["application"] == "freedoom"
    assert sent[:3] == [
        ("video_credit", {"count": 6}),
        ("input", 1, {"event": "key_down", "key": "Up"}),
        ("input", 2, {"event": "key_up", "key": "Up"}),
    ]
    assert result["decoded_frames"] == 0


def test_chocolate_doom_packaged_path_is_used_when_path_omits_games(
        monkeypatch, tmp_path):
    from colabcola import bridge_game_freedoom_runtime as game
    binaries = {}
    for name in game.BINARY_NAMES:
        fake = tmp_path / name
        fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake.chmod(0o755)
        binaries[name] = str(fake)

    monkeypatch.setattr(game, "PACKAGED_CHOCOLATE_DOOM",
                        binaries["chocolate-doom"])

    def which(name):
        # Mimic Colab PATH without /usr/games.
        return None if name == "chocolate-doom" else binaries[name]

    discovered = game.binary_inventory(which=which)
    assert discovered == binaries


def test_packaged_chocolate_doom_must_be_executable_and_regular(
        monkeypatch, tmp_path):
    from colabcola import bridge_game_freedoom_runtime as game
    missing = tmp_path / "missing-chocolate-doom"
    monkeypatch.setattr(game, "PACKAGED_CHOCOLATE_DOOM", str(missing))
    with pytest.raises(RuntimeError, match="dependency missing: chocolate-doom"):
        game.binary_inventory(which=lambda name: (
            None if name == "chocolate-doom" else __import__("sys").executable))

    # A directory cannot be executed as the game binary on any platform.
    invalid = tmp_path / "chocolate-doom"
    invalid.mkdir()
    monkeypatch.setattr(game, "PACKAGED_CHOCOLATE_DOOM", str(invalid))
    with pytest.raises(RuntimeError, match="dependency missing: chocolate-doom"):
        game.binary_inventory(which=lambda name: (
            None if name == "chocolate-doom" else __import__("sys").executable))
