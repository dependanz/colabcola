"""Opt-in Windows/Tk Input Lab capture controls with explicit safe pointer lock.

All input remains on the established authenticated bridge. Locking is armed
ONLY by clicking the dedicated final-stage button; Escape, focus loss and
window closure unlock and restore the pointer. No global keyboard hooks.
"""
from __future__ import annotations

import os
import time

from .bridge_input_lab_runtime import KEYS, NEXT_RECT, LOCK_RECT, WIDTH, HEIGHT


def _inside(x, y, rect):
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


class InputLabHostControls:
    def __init__(self, root, label, enqueue, *,
                 width=WIDTH, height=HEIGHT, pointer_warp=None):
        if width != WIDTH or height != HEIGHT:
            raise ValueError("Input Lab requires 640x360 viewer")
        self.root = root
        self.label = label
        self.enqueue = enqueue
        self.width = width
        self.height = height
        self.stage = 0
        self.locked = False
        self.keys = set()
        self.buttons = set()
        self._next_armed = False
        self._previous_motion_time = 0.0
        self._warp = pointer_warp or self._warp_pointer
        self._native_hide_count = 0

    def _coords(self, event):
        x = max(0, min(self.width, int(event.x)))
        y = max(0, min(self.height, int(event.y)))
        return x, y

    def _normalized(self, event):
        x, y = self._coords(event)
        return {"x": x / self.width, "y": y / self.height}

    def _warp_pointer(self):
        x = self.label.winfo_rootx() + self.width // 2
        y = self.label.winfo_rooty() + self.height // 2
        if os.name == "nt":
            import ctypes
            if ctypes.windll.user32.SetCursorPos(x, y) == 0:
                raise OSError("Windows pointer recenter failed")
        else:
            self.label.event_generate(
                "<Motion>", warp=True, x=self.width // 2, y=self.height // 2)

    def _windows_hide_cursor(self):
        # Tk 8.6 does not implement cursor="none" on Windows.
        # ShowCursor has a counter; restore exactly the calls we make.
        import ctypes
        user32 = ctypes.windll.user32
        count = 0
        for _ in range(32):
            value = user32.ShowCursor(False)
            count += 1
            if value < 0:
                self._native_hide_count = count
                return
        for _ in range(count):
            user32.ShowCursor(True)
        raise OSError("Windows cursor hide limit exceeded")

    def _restore_native_cursor(self):
        count = self._native_hide_count
        self._native_hide_count = 0
        if not count:
            return
        import ctypes
        for _ in range(count):
            ctypes.windll.user32.ShowCursor(True)

    def lock_pointer(self):
        if self.locked or self.stage != 3:
            return False
        try:
            try:
                self.label.configure(cursor="none")
            except Exception:
                if os.name != "nt":
                    raise
                self._windows_hide_cursor()
            self._warp()
        except Exception:
            try:
                self.label.configure(cursor="")
            except Exception:
                pass
            self._restore_native_cursor()
            return False
        self.locked = True
        self.enqueue({"event": "pointer_lock", "enabled": True})
        return True

    def unlock_pointer(self):
        if not self.locked:
            return
        self.locked = False
        try:
            self.label.configure(cursor="")
        except Exception:
            pass
        finally:
            self._restore_native_cursor()
        self.enqueue({"event": "pointer_lock", "enabled": False})

    def motion(self, event):
        x, y = self._coords(event)
        now = time.monotonic()
        if self.locked:
            dx = x - self.width // 2
            dy = y - self.height // 2
            # Ignore the synthetic Windows mouse event from recentring.
            if abs(dx) <= 1 and abs(dy) <= 1:
                return
            self.enqueue({
                "event": "mouse_delta",
                "dx": max(-64, min(64, dx)),
                "dy": max(-64, min(64, dy)),
            })
            try:
                self._warp()
            except Exception:
                # Never leave a hidden pointer with broken recentering.
                self.unlock_pointer()
            return
        if now - self._previous_motion_time < .02:
            return
        self._previous_motion_time = now
        self.enqueue({"event": "mouse_move", **self._normalized(event)})

    def key_down(self, event):
        key = event.keysym
        if key not in KEYS or key in self.keys:
            return
        self.keys.add(key)
        self.enqueue({"event": "key_down", "key": key})
        if key == "Escape":
            self.unlock_pointer()

    def key_up(self, event):
        key = event.keysym
        if key not in self.keys:
            return
        self.keys.remove(key)
        self.enqueue({"event": "key_up", "key": key})

    def button_down(self, event):
        button = event.num
        if button not in (1, 2, 3):
            return
        if button == 1:
            try:
                self.label.focus_set()
            except Exception:
                pass
        if button not in self.buttons:
            self.buttons.add(button)
            self.enqueue({
                "event": "mouse_button_down", "button": button,
                **self._normalized(event),
            })
        x, y = self._coords(event)
        self._next_armed = (button == 1 and not self.locked
                            and _inside(x, y, NEXT_RECT))

    def button_up(self, event):
        button = event.num
        if button not in (1, 2, 3):
            return
        if button in self.buttons:
            self.buttons.remove(button)
            self.enqueue({
                "event": "mouse_button_up", "button": button,
                **self._normalized(event),
            })
        x, y = self._coords(event)
        if self._next_armed and button == 1 and _inside(x, y, NEXT_RECT):
            if self.stage < 3:
                self.stage += 1
            else:
                # The remote game verifies look distance and completion.
                self.unlock_pointer()
        elif (button == 1 and self.stage == 3 and not self.locked
              and _inside(x, y, LOCK_RECT)):
            self.lock_pointer()
        self._next_armed = False

    def wheel(self, event):
        if self.locked:
            return
        delta = getattr(event, "delta", 0)
        if not isinstance(delta, int) or delta == 0:
            return
        ticks = min(8, max(1, int(round(abs(delta) / 120))))
        self.enqueue({"event": "mouse_wheel", "ticks": ticks if delta > 0 else -ticks})

    def wheel_linux_up(self, event):
        if not self.locked:
            self.enqueue({"event": "mouse_wheel", "ticks": 1})

    def wheel_linux_down(self, event):
        if not self.locked:
            self.enqueue({"event": "mouse_wheel", "ticks": -1})

    def release_all(self, _event=None):
        self.unlock_pointer()
        for key in sorted(self.keys):
            self.enqueue({"event": "key_up", "key": key})
        self.keys.clear()
        for button in sorted(self.buttons):
            self.enqueue({"event": "mouse_button_up", "button": button,
                          "x": .5, "y": .5})
        self.buttons.clear()
        self._next_armed = False

    def bind(self):
        self.label.bind("<Motion>", self.motion)
        self.label.bind("<ButtonPress-1>", self.button_down)
        self.label.bind("<ButtonRelease-1>", self.button_up)
        self.label.bind("<ButtonPress-2>", self.button_down)
        self.label.bind("<ButtonRelease-2>", self.button_up)
        self.label.bind("<ButtonPress-3>", self.button_down)
        self.label.bind("<ButtonRelease-3>", self.button_up)
        self.label.bind("<MouseWheel>", self.wheel)
        self.label.bind("<Button-4>", self.wheel_linux_up)
        self.label.bind("<Button-5>", self.wheel_linux_down)
        self.root.bind("<KeyPress>", self.key_down)
        self.root.bind("<KeyRelease>", self.key_up)
        self.root.bind("<FocusOut>", self.release_all)
        self.root.focus_set()
        return self.release_all
