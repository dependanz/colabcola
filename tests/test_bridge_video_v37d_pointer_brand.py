"""v37D: official website branding and fail-safe Windows cursor clipping.

These tests do not manipulate the desktop cursor; all native APIs are fakes.
"""
from __future__ import annotations

import hashlib
from importlib import resources
from pathlib import Path
from types import SimpleNamespace

import pytest

from colabcola.bridge_input_lab_pointer import Win32CenterPointerCapture
from colabcola.bridge_input_lab_host import InputLabHostControls
from colabcola.bridge_input_lab_runtime import LabPainter, LabState, WIDTH, HEIGHT


class FakeWindowsPointerApi:
    def __init__(self, bounds=(100, 200, 740, 560),
                 old=(0, 0, 2560, 1600)):
        self.bounds = bounds
        self.old = old
        self.current_clip = old
        self.position = (0, 0)
        self.actions = []
        self.fail_recenter = False

    def client_bounds(self, label):
        self.actions.append(("bounds",))
        return self.bounds

    def previous_clip(self):
        self.actions.append(("get_clip",))
        return self.current_clip

    def set_clip(self, rect):
        self.actions.append(("set_clip", rect))
        self.current_clip = rect

    def set_position(self, x, y):
        if self.fail_recenter:
            raise OSError("cannot recenter")
        self.position = (x, y)
        self.actions.append(("set_position", x, y))

    def attempt_fast_motion(self, x, y):
        left, top, right, bottom = self.current_clip
        self.position = (
            max(left, min(right - 1, x)),
            max(top, min(bottom - 1, y)),
        )


def test_center_capture_prevents_extreme_cursor_escape_and_restores_exact_clip():
    api = FakeWindowsPointerApi()
    lock = Win32CenterPointerCapture(api)
    assert not lock.active
    lock.acquire(object())
    assert lock.active
    assert api.current_clip == (330, 290, 511, 471)
    assert api.position == (420, 380)
    api.attempt_fast_motion(100_000, -100_000)
    assert api.position == (510, 290)
    lock.recenter()
    assert api.position == (420, 380)
    lock.release()
    assert not lock.active
    assert api.current_clip == api.old
    lock.release()
    assert api.current_clip == api.old


def test_center_capture_failure_restores_prior_clip():
    api = FakeWindowsPointerApi()
    api.fail_recenter = True
    clip = Win32CenterPointerCapture(api)
    with pytest.raises(OSError, match="recenter"):
        clip.acquire(object())
    assert not clip.active
    assert api.current_clip == api.old


def test_center_capture_rejects_tiny_viewport_without_clipping():
    api = FakeWindowsPointerApi(bounds=(1, 2, 20, 30))
    clip = Win32CenterPointerCapture(api)
    with pytest.raises(OSError, match="too small"):
        clip.acquire(object())
    assert api.current_clip == api.old


class FakeLabel:
    def __init__(self):
        self.cursors = []
    def configure(self, **kw):
        self.cursors.append(kw)
    def focus_set(self):
        pass


def _host():
    events = []
    api = FakeWindowsPointerApi()
    capture = Win32CenterPointerCapture(api)
    label = FakeLabel()
    root = SimpleNamespace()
    host = InputLabHostControls(
        root, label, events.append,
        pointer_warp=capture.recenter, pointer_capture=capture)
    host.stage = 3
    return host, capture, api, label, events


def test_pointer_lock_escape_unlock_and_focus_loss_restore_even_after_fast_motion():
    host, capture, api, label, events = _host()
    assert host.lock_pointer() is True
    assert capture.active
    api.attempt_fast_motion(10_000, 10_000)
    assert api.position == (510, 470)
    host.motion(SimpleNamespace(x=390, y=200))
    assert any(e.get("event") == "mouse_delta" for e in events)
    assert api.position == (420, 380)
    host.key_down(SimpleNamespace(keysym="Escape"))
    assert not capture.active and not host.locked
    assert api.current_clip == api.old
    assert label.cursors[-1] == {"cursor": ""}
    assert events[-1] == {"event": "pointer_lock", "enabled": False}

    # Focus loss is an independent restoration path.
    assert host.lock_pointer()
    assert capture.active
    host.release_all()
    assert not capture.active and not host.locked
    assert api.current_clip == api.old


def test_window_configure_unlock_only_on_toplevel_move():
    host, clip, api, _, events = _host()
    assert host.lock_pointer()
    child = object()
    host._on_window_configure(SimpleNamespace(widget=child))
    assert host.locked and clip.active
    host._on_window_configure(SimpleNamespace(widget=host.root))
    assert not host.locked and not clip.active
    assert api.current_clip == api.old


def test_pointer_capture_restores_on_cursor_hide_failure():
    host, capture, api, label, events = _host()
    def fail_hide(**kwargs):
        if kwargs.get("cursor") == "none":
            raise RuntimeError("cursor cannot be hidden")
    label.configure = fail_hide
    # On Windows, Tk's missing cursor="none" is expected to fall back to
    # Win32 ShowCursor. Simulate failure of BOTH hide mechanisms.
    host._windows_hide_cursor = lambda: (_ for _ in ()).throw(
        OSError("native pointer hide failed"))
    assert host.lock_pointer() is False
    assert not host.locked and not capture.active
    assert api.current_clip == api.old
    assert not any(e["event"] == "pointer_lock" for e in events)


def test_website_official_banner_asset_is_unmodified_and_packaged():
    image = resources.files("colabcola").joinpath(
        "assets", "colabcola-banner.png").read_bytes()
    assert len(image) == 600210
    data = b"blob " + str(len(image)).encode("ascii") + b"\x00" + image
    assert hashlib.sha1(data).hexdigest() == (
        "f15d2f9b68f0678df7f772d399bb11b45f23ebbe"
    )


def test_finished_image_uses_actual_website_banner_pixel_for_pixel():
    pytest.importorskip("PIL")
    painter = LabPainter()
    state = LabState()
    state.stage = 4
    state.finished_at = 10.0
    body = painter.frame(state, 11.0)
    prefix = b"P6\n640 360\n255\n"
    assert body.startswith(prefix)
    from PIL import Image
    full = Image.frombytes("RGB", (WIDTH, HEIGHT), body[len(prefix):])
    # The original approved banner was downscaled for the 640x360 viewer,
    # not replaced by an invented circular logo.
    box = full.crop((35, 92, 605, 235))
    assert box.tobytes() == painter.brand_banner.tobytes()
    assert box.size == (570, 143)
