"""Windows game pointer clipping with exact restoration of prior desktop bounds.

Unlike repeated SetCursorPos calls, ClipCursor prevents fast pointer
movement from escaping the viewport. Nothing changes unless the user
explicitly opts into the Input Lab's stage-four LOCK CURSOR action.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes


class _Win32CursorApi:
    def __init__(self):
        self.user32 = ctypes.windll.user32

    def client_bounds(self, label) -> tuple[int, int, int, int]:
        hwnd = label.winfo_id()
        if not isinstance(hwnd, int) or hwnd <= 0:
            raise OSError("game window handle unavailable")
        rect = wintypes.RECT()
        origin = wintypes.POINT(0, 0)
        if not self.user32.GetClientRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
            raise OSError("cannot read game viewport")
        if not self.user32.ClientToScreen(wintypes.HWND(hwnd), ctypes.byref(origin)):
            raise OSError("cannot locate game viewport")
        left, top = origin.x, origin.y
        right = left + rect.right - rect.left
        bottom = top + rect.bottom - rect.top
        if right - left < 64 or bottom - top < 64:
            raise OSError("game viewport is too small for pointer lock")
        return left, top, right, bottom

    def previous_clip(self) -> tuple[int, int, int, int]:
        bounds = wintypes.RECT()
        if not self.user32.GetClipCursor(ctypes.byref(bounds)):
            raise OSError("cannot read existing pointer clipping")
        return bounds.left, bounds.top, bounds.right, bounds.bottom

    def set_clip(self, bounds: tuple[int, int, int, int]) -> None:
        rect = wintypes.RECT(*bounds)
        if not self.user32.ClipCursor(ctypes.byref(rect)):
            raise OSError("cannot set Windows pointer clipping")

    def set_position(self, x: int, y: int) -> None:
        if not self.user32.SetCursorPos(x, y):
            raise OSError("cannot center Windows pointer")


class Win32CenterPointerCapture:
    """Small bounded center region; paired restoration, no global hooks."""

    def __init__(self, api=None):
        self._api = api if api is not None else _Win32CursorApi()
        self._previous = None
        self._center = None

    @property
    def active(self) -> bool:
        return self._previous is not None

    def acquire(self, label) -> None:
        if self.active:
            raise RuntimeError("pointer already captured")
        left, top, right, bottom = self._api.client_bounds(label)
        cx = (left + right) // 2
        cy = (top + bottom) // 2
        # Permit enough displacement to measure relative movement, but
        # clamp rapid OS motion to a small region INSIDE the game viewport.
        radius = min(96, (right - left) // 4, (bottom - top) // 4)
        if radius < 16:
            raise OSError("game viewport is too small for clipping")
        bounded = (cx - radius, cy - radius, cx + radius + 1, cy + radius + 1)
        previous = self._api.previous_clip()
        self._api.set_clip(bounded)
        self._previous = previous
        self._center = (cx, cy)
        try:
            self.recenter()
        except Exception:
            self.release()
            raise

    def recenter(self) -> None:
        if not self.active or self._center is None:
            raise RuntimeError("pointer capture not active")
        self._api.set_position(*self._center)

    def release(self) -> None:
        if not self.active:
            return
        # Preserve the restoration target if Win32 release fails, allowing
        # another release attempt on the next focus or shutdown callback.
        self._api.set_clip(self._previous)
        self._previous = None
        self._center = None
