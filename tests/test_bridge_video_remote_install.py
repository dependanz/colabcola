"""v33g: bounded, explicit remote PyAV binary wheel provisioning; offline."""
from __future__ import annotations

import importlib
from pathlib import Path
import runpy
import subprocess
import sys
import types

import pytest

from colabcola import bridge_video_colab_source as source


def test_already_importable_pyav_does_not_install(monkeypatch):
    fake_av = types.ModuleType("av")
    monkeypatch.setitem(sys.modules, "av", fake_av)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("pip must not run"))
    source.provision_remote_pyav()


def test_missing_pyav_installs_fixed_wheel_once_silently(monkeypatch):
    calls = []
    real_import = __import__
    def missing_av(name, *a, **kw):
        if name == "av":
            raise ImportError("av not present")
        return real_import(name, *a, **kw)
    monkeypatch.setattr("builtins.__import__", missing_av)
    monkeypatch.setattr(importlib, "import_module", lambda name: object())
    monkeypatch.setattr(importlib, "invalidate_caches", lambda: None)
    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, "run", fake_run)
    source.provision_remote_pyav()
    assert len(calls) == 1
    argv, options = calls[0]
    assert argv[:4] == [sys.executable, "-m", "pip", "--isolated"]
    assert "av==17.0.0" in argv
    assert "--only-binary=:all:" in argv and "--no-deps" in argv
    assert "https://pypi.org/simple" in argv
    assert options["timeout"] == 40
    assert options["stdout"] == subprocess.DEVNULL
    assert options["stderr"] == subprocess.DEVNULL
    assert "shell" not in options


@pytest.mark.parametrize("mode,expected", [
    ("error", "remote PyAV installation failed"),
    ("timeout", "remote PyAV installation timed out"),
    ("oserror", "remote PyAV installation failed"),
])
def test_remote_install_failure_is_redacted_and_bounded(monkeypatch, mode, expected):
    real_import = __import__
    def missing_av(name, *a, **kw):
        if name == "av":
            raise ImportError("secret")
        return real_import(name, *a, **kw)
    monkeypatch.setattr("builtins.__import__", missing_av)
    def bad_run(*args, **kwargs):
        if mode == "timeout":
            raise subprocess.TimeoutExpired(cmd="secret", timeout=40)
        if mode == "oserror":
            raise OSError("private token")
        return types.SimpleNamespace(returncode=1)
    monkeypatch.setattr(subprocess, "run", bad_run)
    with pytest.raises(source.RemoteVideoSourceError) as caught:
        source.provision_remote_pyav()
    assert str(caught.value) == expected
    assert "secret" not in str(caught.value)


def test_cli_remote_install_requires_h264(monkeypatch):
    from colabcola.cli import main
    from colabcola import bridge_colab
    selected = []
    monkeypatch.setattr(bridge_colab, "run_colab", lambda args: selected.append(args) or 0)
    base = ["--profile", "example", "bridge", "colab", "--preflight-only"]
    assert main(base + ["--transport", "comm", "--video-codec", "h264",
                        "--install-remote-pyav"]) == 0
    assert selected[-1].install_remote_pyav
    assert selected[-1].inflight == 1
    with pytest.raises(SystemExit) as exc:
        main(base + ["--transport", "comm", "--install-remote-pyav"])
    assert exc.value.code == 2


def test_peer_rejects_pip_permission_in_ppm_mode():
    from colabcola.bridge_comm import KernelCommBridge
    with pytest.raises(ValueError, match="requires opt-in H.264"):
        KernelCommBridge(None, None, "a" * 32, 12, install_remote_pyav=True)
    assert not KernelCommBridge(None, None, "a" * 32, 12, video_codec="h264").install_remote_pyav
    assert KernelCommBridge(None, None, "a" * 32, 12, video_codec="h264",
                            install_remote_pyav=True).install_remote_pyav


def test_bootstrap_diagnostic_maps_remote_install_failures(tmp_path, monkeypatch):
    from colabcola import bridge_colab_comm_runtime as remote
    # Reuse the established fake renderer and fake kernel from v33f.
    from test_bridge_video_preflight_diagnostics import _fake_renderer, _install_fake_kernel, RUN
    comms = _install_fake_kernel(monkeypatch)
    renderer = _fake_renderer(tmp_path / "renderer.py")
    fake_source = tmp_path / "source.py"
    flag = tmp_path / "installed"
    fake_source.write_text(
        "from pathlib import Path\n"
        "def provision_remote_pyav():\n"
        f" Path({str(flag)!r}).write_text('installed')\n"
        "class RemoteH264PullSource:\n"
        " def __init__(self, *args, **kwargs): pass\n",
        encoding="utf-8")
    channel = remote.start(str(tmp_path), RUN, 12, renderer,
                           video_source_path=str(fake_source), preflight_video=True,
                           install_remote_pyav=True)
    assert channel is comms[0]
    assert flag.read_text() == "installed"
    assert channel.ready["kind"] == "ready"
    assert channel.ready["video_codec"] == "h264-annexb"


@pytest.mark.parametrize("message,code", [
    ("remote PyAV installation failed", "pyav_install_failed"),
    ("remote PyAV installation timed out", "pyav_install_timeout"),
])
def test_remote_preflight_reports_sanitized_install_failure(tmp_path, monkeypatch, message, code):
    from colabcola import bridge_colab_comm_runtime as remote
    from test_bridge_video_preflight_diagnostics import _fake_renderer, _install_fake_kernel, RUN
    comms = _install_fake_kernel(monkeypatch)
    renderer = _fake_renderer(tmp_path / "renderer.py")
    fake_source = tmp_path / "failing_source.py"
    fake_source.write_text(
        "class RemoteVideoSourceError(RuntimeError): pass\n"
        "def provision_remote_pyav():\n"
        f" raise RemoteVideoSourceError({message!r})\n"
        "class RemoteH264PullSource:\n"
        " def __init__(self, *args, **kwargs): pass\n",
        encoding="utf-8")
    channel = remote.start(str(tmp_path), RUN, 12, renderer,
                           video_source_path=str(fake_source), preflight_video=True,
                           install_remote_pyav=True)
    assert channel is comms[0]
    assert channel.ready["kind"] == "video_preflight_failed"
    assert channel.ready["failure_code"] == code
    assert "details" not in str(channel.ready)
