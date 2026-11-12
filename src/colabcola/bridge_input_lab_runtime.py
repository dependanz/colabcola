"""Colab Cola Input Lab: remote four-stage interactive input diagnostics.

Runs only after explicit external T4 attachment. No shell, arbitrary commands,
X11 process, download, or new listening socket. Output is the existing bounded
CCB1 frame ring consumed by the authenticated H.264 Comm push sender.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import queue
import re
import struct
import threading
import time
import zlib
from statistics import median

WIDTH, HEIGHT, FPS = 640, 360, 30
RUN_RE = re.compile(r"[0-9a-f]{32}\Z")
HEADER = struct.Struct("!4sIIII")
MAGIC = b"CCB1"
FRAME_LIMIT = 2 * 1024 * 1024
SLOTS = 8
QUEUE_LIMIT = 128
# Fail closed if the host never grants its initial video credits, or if its
# authenticated credit heartbeat disappears after first grant. These clocks
# are remote-only: no comparison against Windows-host timestamps.
INITIAL_HOST_CREDIT_TIMEOUT_S = 120.0
ACTIVE_HOST_CREDIT_TIMEOUT_S = 20.0
P6 = b"P6\n640 360\n255\n"
NEXT_RECT = (492, 310, 626, 350)
LOCK_RECT = (167, 268, 472, 315)
STAGE_NAMES = ("KEYBOARD", "MOUSE BUTTONS", "SCROLL WHEEL", "POINTER LOCK")
LETTERS = tuple("abcdefghijklmnopqrstuvwxyz")
DIGITS = tuple("0123456789")
FUNCTIONS = tuple(f"F{i}" for i in range(1, 13))
PUNCT = (
    "minus", "equal", "bracketleft", "bracketright", "backslash",
    "semicolon", "apostrophe", "grave", "comma", "period", "slash",
)
SYSTEM = (
    "Escape", "Tab", "Caps_Lock", "Shift_L", "Shift_R",
    "Control_L", "Control_R", "Alt_L", "Alt_R", "Super_L", "Super_R",
    "BackSpace", "Return", "space",
)
NAVIGATION = (
    "Left", "Right", "Up", "Down", "Insert", "Delete",
    "Home", "End", "Prior", "Next", "Pause", "Print",
)
NUMPAD = tuple(f"KP_{i}" for i in range(10)) + (
    "KP_Add", "KP_Subtract", "KP_Multiply", "KP_Divide",
    "KP_Decimal", "KP_Enter",
)
KEYS = frozenset(LETTERS + DIGITS + FUNCTIONS + PUNCT + SYSTEM + NAVIGATION + NUMPAD)
BUTTONS = (1, 2, 3)
_CONTROLLER = None


def validate_event(event):
    """Whitelisted, bounded, version-independent remote input payload."""
    if (not isinstance(event, dict)
            or type(event.get("seq")) is not int
            or not 1 <= event["seq"] < 10_000_000):
        raise ValueError("invalid lab input sequence")
    kind = event.get("event")
    if kind in ("key_down", "key_up"):
        if (set(event) != {"seq", "event", "key"}
                or type(event["key"]) is not str or event["key"] not in KEYS):
            raise ValueError("invalid lab key")
    elif kind in ("mouse_button_down", "mouse_button_up"):
        if (set(event) != {"seq", "event", "button", "x", "y"}
                or type(event["button"]) is not int
                or event["button"] not in BUTTONS):
            raise ValueError("invalid lab button")
        _coords(event)
    elif kind == "mouse_move":
        if set(event) != {"seq", "event", "x", "y"}:
            raise ValueError("invalid lab pointer location")
        _coords(event)
    elif kind == "mouse_wheel":
        if (set(event) != {"seq", "event", "ticks"}
                or type(event["ticks"]) is not int
                or not -8 <= event["ticks"] <= 8 or event["ticks"] == 0):
            raise ValueError("invalid lab wheel")
    elif kind == "mouse_delta":
        if (set(event) != {"seq", "event", "dx", "dy"}
                or type(event["dx"]) is not int or type(event["dy"]) is not int
                or not -64 <= event["dx"] <= 64
                or not -64 <= event["dy"] <= 64):
            raise ValueError("invalid lab mouse delta")
    elif kind == "pointer_lock":
        if (set(event) != {"seq", "event", "enabled"}
                or type(event["enabled"]) is not bool):
            raise ValueError("invalid lab pointer-lock state")
    else:
        raise ValueError("unknown lab input")
    return dict(event)


def _coords(event):
    for key in ("x", "y"):
        val = event.get(key)
        if (type(val) not in (float, int) or not math.isfinite(val)
                or not 0.0 <= val <= 1.0):
            raise ValueError("invalid normalized pointer coordinates")


def _inside(x, y, rect):
    px, py = x * WIDTH, y * HEIGHT
    return rect[0] <= px <= rect[2] and rect[1] <= py <= rect[3]


def _atomic_bytes(path, body):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(body)
    os.replace(temporary, path)


def _p95(items):
    values = sorted(items)
    pos = (len(values) - 1) * .95
    index = int(pos)
    return round(values[index] + (pos - index) *
                 (values[min(index + 1, len(values) - 1)] - values[index]), 2)


class LabState:
    """Pure logical state machine, suitable for offline deterministic testing."""

    def __init__(self):
        self.stage = 0
        self.seen_keys = set()
        self.held_keys = set()
        self.seen_buttons = set()
        self.clicks = []
        self.mouse_x = .5
        self.mouse_y = .5
        self.wheel_up = 0
        self.wheel_down = 0
        self.locked = False
        self.yaw = 0.0
        self.pitch = 0.0
        self.look_distance = 0
        self.finished_at = None
        self.last_input_seq = 0
        self._next_armed = False

    def apply(self, raw, *, now=None):
        event = validate_event(raw)
        seq = event["seq"]
        if seq <= self.last_input_seq:
            return False
        self.last_input_seq = seq
        kind = event["event"]
        now = time.monotonic() if now is None else now
        if kind == "key_down":
            self.held_keys.add(event["key"])
            if self.stage == 0:
                self.seen_keys.add(event["key"])
            if self.stage == 3 and event["key"] == "Escape":
                self.locked = False
        elif kind == "key_up":
            self.held_keys.discard(event["key"])
        elif kind == "mouse_move":
            self.mouse_x, self.mouse_y = event["x"], event["y"]
        elif kind == "mouse_button_down":
            self.mouse_x, self.mouse_y = event["x"], event["y"]
            if self.stage == 1:
                self.seen_buttons.add(event["button"])
                self.clicks.append((event["x"], event["y"], event["button"]))
                self.clicks = self.clicks[-16:]
            self._next_armed = (
                event["button"] == 1 and not self.locked
                and _inside(event["x"], event["y"], NEXT_RECT))
        elif kind == "mouse_button_up":
            self.mouse_x, self.mouse_y = event["x"], event["y"]
            if (event["button"] == 1 and self._next_armed
                    and not self.locked
                    and _inside(event["x"], event["y"], NEXT_RECT)):
                if self.stage < 3:
                    self.stage += 1
                    self._next_armed = False
                elif self.stage == 3 and self.look_distance >= 10:
                    self.stage = 4
                    self.finished_at = now
            elif (event["button"] == 1 and self.stage == 3
                  and _inside(event["x"], event["y"], LOCK_RECT)):
                # Host must independently activate actual cursor hiding/centering.
                # Remote click alone does not claim pointer lock succeeded.
                pass
            self._next_armed = False
        elif kind == "mouse_wheel" and self.stage == 2:
            ticks = event["ticks"]
            if ticks > 0:
                self.wheel_up = min(10000, self.wheel_up + ticks)
            else:
                self.wheel_down = min(10000, self.wheel_down - ticks)
        elif kind == "pointer_lock" and self.stage == 3:
            self.locked = event["enabled"]
        elif kind == "mouse_delta" and self.stage == 3 and self.locked:
            dx, dy = event["dx"], event["dy"]
            self.yaw = (self.yaw + dx * .005) % (math.pi * 2)
            self.pitch = max(-.82, min(.82, self.pitch - dy * .005))
            self.look_distance = min(1_000_000, self.look_distance + abs(dx) + abs(dy))
        return True


class LabPainter:
    def __init__(self):
        from PIL import Image, ImageDraw, ImageFont
        self.Image = Image
        self.Draw = ImageDraw
        self.fonts = {}
        # Website-approved Colab Cola hero banner (exact packaged PNG).
        # Remote launch uploads it beside this script; local offline tests
        # read its identical package resource from the source checkout.
        asset = Path(__file__).with_name("colabcola-banner.png")
        if not asset.is_file():
            asset = Path(__file__).parent / "assets" / "colabcola-banner.png"
        if not asset.is_file():
            raise RuntimeError("Input Lab official branding asset unavailable")
        with Image.open(asset) as banner:
            if banner.format != "PNG" or banner.size != (1584, 396):
                raise RuntimeError("Input Lab official branding asset invalid")
            self.brand_banner = banner.convert("RGB").resize(
                (570, 143), Image.Resampling.LANCZOS)
        for size in (10, 12, 14, 16, 18, 23, 27, 42):
            font = None
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
            except (OSError, ValueError):
                font = ImageFont.load_default()
            self.fonts[size] = font

    def text(self, draw, xy, value, size=14, color="#edf5ff", anchor=None):
        draw.text(xy, str(value), font=self.fonts[size], fill=color, anchor=anchor)

    def button(self, draw, rect, label, active=True):
        fill = "#2fdbb2" if active else "#273448"
        fg = "#071a22" if active else "#99a6b8"
        draw.rounded_rectangle(rect, radius=12, fill=fill)
        self.text(draw, ((rect[0] + rect[2]) // 2,
                         (rect[1] + rect[3]) // 2),
                  label, 14, fg, "mm")

    def base(self, state):
        image = self.Image.new("RGB", (WIDTH, HEIGHT), "#0b1423")
        draw = self.Draw.Draw(image)
        draw.rounded_rectangle((13, 9, 628, 49), radius=13, fill="#17253b")
        self.text(draw, (26, 28), "COLAB COLA", 18, "#f7fafc", "lm")
        self.text(draw, (608, 28), "INPUT LAB", 12, "#83e4e0", "rm")
        if state.stage < 4:
            for i, name in enumerate(STAGE_NAMES):
                left = 16 + 152 * i
                color = "#32ddb3" if state.stage == i else (
                    "#667e98" if i < state.stage else "#253954")
                draw.rounded_rectangle((left, 57, left + 142, 61),
                                       radius=2, fill=color)
            self.text(draw, (22, 74), STAGE_NAMES[state.stage], 16,
                      "#78e1da")
        return image, draw

    def render_keyboard(self, draw, state):
        self.text(draw, (22, 98),
                  f"Press keys to light them up  -  {len(state.seen_keys)} registered",
                  12, "#adbdcd")
        rows = [
            ("Escape",) + FUNCTIONS + ("Print", "Pause"),
            ("grave",) + tuple("1234567890") + ("minus", "equal", "BackSpace"),
            ("Tab",) + tuple("qwertyuiop") + (
                "bracketleft", "bracketright", "backslash"),
            ("Caps_Lock",) + tuple("asdfghjkl") + (
                "semicolon", "apostrophe", "Return"),
            ("Shift_L",) + tuple("zxcvbnm") + (
                "comma", "period", "slash", "Shift_R"),
            ("Control_L", "Super_L", "Alt_L", "space", "Alt_R",
             "Super_R", "Control_R", "Left", "Down", "Up", "Right"),
            ("Insert", "Delete", "Home", "End", "Prior", "Next",
             "KP_7", "KP_8", "KP_9", "KP_Add"),
            ("KP_4", "KP_5", "KP_6", "KP_Subtract", "KP_1",
             "KP_2", "KP_3", "KP_Multiply", "KP_0", "KP_Decimal",
             "KP_Divide", "KP_Enter"),
        ]
        labels = {
            "BackSpace": "Bksp", "Caps_Lock": "Caps", "Shift_L": "Shift",
            "Shift_R": "Shift", "Control_L": "Ctrl", "Alt_L": "Alt",
            "Control_R": "Ctrl", "Alt_R": "Alt", "Return": "Enter",
            "space": "SPACE", "bracketleft": "[", "bracketright": "]",
            "minus": "-", "equal": "=", "semicolon": ";",
            "apostrophe": "'", "comma": ",", "period": ".", "slash": "/",
            "grave": "`", "backslash": "\\", "Escape": "Esc",
            "Super_L": "Win", "Super_R": "Win", "Pause": "Pause",
            "Print": "PrtSc", "Prior": "PgUp", "Next": "PgDn",
            "Insert": "Ins", "Delete": "Del", "Home": "Home", "End": "End",
            "KP_Enter": "N-Ent", "KP_Decimal": "N-.",
        }
        for row_index, row in enumerate(rows):
            y = 111 + row_index * 23
            slot = 596 / len(row)
            for col, key in enumerate(row):
                x1 = int(22 + col * slot)
                x2 = int(22 + (col + 1) * slot - 3)
                active = key in state.seen_keys
                draw.rounded_rectangle(
                    (x1, y, x2, y + 20), radius=5,
                    fill=("#31d6a7" if active else "#293b52"))
                val = labels.get(key, key.upper() if len(key) == 1 else key)
                self.text(draw, ((x1+x2)//2, y+10), val, 10,
                          "#052223" if active else "#d8e3f1", "mm")
        extra = len(state.seen_keys - set(k for row in rows for k in row))
        self.text(draw, (24, 306),
                  "Win keys/shortcuts are OS-reserved and may not register.",
                  10, "#93aac1")
        self.text(draw, (24, 321),
                  "No need to press every key. Click NEXT TEST when ready.",
                  10, "#a9c5d8")
        self.button(draw, NEXT_RECT, "NEXT TEST")

    def render_buttons(self, draw, state):
        self.text(draw, (22, 101), "Click with LEFT, MIDDLE, and RIGHT anywhere in the window.",
                  12, "#b3c4d8")
        for i, (button, label, x) in enumerate((
                (1, "LEFT", 127), (2, "MIDDLE", 320), (3, "RIGHT", 513))):
            active = button in state.seen_buttons
            draw.ellipse((x-51, 134, x+51, 236),
                         fill=("#2fdbb2" if active else "#253850"),
                         outline="#5d7896", width=2)
            self.text(draw, (x, 185), label, 14,
                      "#08252a" if active else "#dbeafa", "mm")
            self.text(draw, (x, 252), "DETECTED" if active else "TRY IT", 12,
                      "#36e1b2" if active else "#9bb1ca", "mm")
        for x, y, button in state.clicks:
            if y < .84:
                px, py = int(x*WIDTH), int(y*HEIGHT)
                draw.ellipse((px-4, py-4, px+4, py+4),
                             fill={1:"#38edae", 2:"#f5c76a",3:"#a48df5"}[button])
        self.button(draw, NEXT_RECT, "NEXT TEST")

    def render_wheel(self, draw, state):
        self.text(draw, (22, 99), "Scroll UP and DOWN over the viewer.",
                  14, "#b3c4d8")
        for i,(label,count) in enumerate((
                ("SCROLL UP",state.wheel_up),("SCROLL DOWN",state.wheel_down))):
            y=145+i*75
            draw.rounded_rectangle((65,y,560,y+46),radius=13,fill="#22354c")
            draw.rounded_rectangle((66,y+1,66+int(min(1,count/12)*491),y+45),
                                   radius=12,fill="#35d6af" if i==0 else "#6fbbff")
            self.text(draw,(87,y+23),f"{label}    {count} ticks",16,
                      "#eaf7fd","lm")
        self.text(draw,(22,285),"Wheel movement is counted remotely, not inferred locally.",
                  12,"#8caac1")
        self.button(draw,NEXT_RECT,"NEXT TEST")

    def project(self, x, y, z, yaw, pitch):
        # Camera at (0, 2.8, -7), facing the positive world Z axis.
        x0,y0,z0=x,y-2.8,z+7
        cy,sy=math.cos(yaw),math.sin(yaw)
        cp,sp=math.cos(pitch),math.sin(pitch)
        vx=cy*x0-sy*z0
        vz=sy*x0+cy*z0
        vy=cp*y0-sp*vz
        vz2=sp*y0+cp*vz
        if vz2<=.3:
            return None
        f=265.0/vz2
        return (WIDTH/2+vx*f, HEIGHT/2-vy*f)

    def voxel_world(self, image, state):
        draw=self.Draw.Draw(image)
        for y in range(84,HEIGHT):
            t=(y-84)/max(1,HEIGHT-84)
            draw.line((0,y,WIDTH,y),fill=(
                int(40+64*t),int(97+68*t),int(152+45*t)))
        # Floating toy-brick world with flat green tops, painted back-to-front.
        cells=[]
        for z in range(3,19):
            for x in range(-9,10):
                elevation=max(0, int(1+1.3*math.sin(x*.71)*math.cos(z*.47)))
                cells.append((x,z,elevation))
        cells.sort(key=lambda c:(c[1]+7)**2+c[0]**2,reverse=True)
        for x,z,elevation in cells:
            y=float(elevation)
            corners=(
                (x,y,z),(x+1,y,z),(x+1,y,z+1),(x,y,z+1),
                (x,y-1,z),(x+1,y-1,z),(x+1,y-1,z+1),(x,y-1,z+1))
            points=[self.project(*v,state.yaw,state.pitch) for v in corners]
            if any(p is None for p in points):
                continue
            top=[points[i] for i in (0,1,2,3)]
            side_a=[points[i] for i in (0,1,5,4)]
            side_b=[points[i] for i in (1,2,6,5)]
            side_c=[points[i] for i in (2,3,7,6)]
            shade=(x*37+z*13)%3
            draw.polygon(side_a,fill=("#346d59","#3a795d","#487d55")[shade])
            draw.polygon(side_b,fill=("#2d5f52","#3c6156","#456a51")[shade])
            draw.polygon(side_c,fill=("#254c45","#306353","#3e6652")[shade])
            draw.polygon(top,fill=("#77c879","#89d37c","#70bb83")[shade])
            draw.line(top+[top[0]],fill="#264b4e",width=1)
        draw.line((WIDTH//2-9,HEIGHT//2,WIDTH//2+9,HEIGHT//2),
                  fill="#f9f9ed",width=2)
        draw.line((WIDTH//2,HEIGHT//2-9,WIDTH//2,HEIGHT//2+9),
                  fill="#f9f9ed",width=2)
        draw.rounded_rectangle((12,65,628,106),radius=10,fill="#162b39")
        self.text(draw,(25,84),"VOXEL LOOK TEST",16,"#dafff6","lm")
        self.text(draw,(615,84),
                  f"YAW {math.degrees(state.yaw):05.1f}   PITCH {math.degrees(state.pitch):+05.1f}",
                  12,"#c8e0e5","rm")
        draw.rounded_rectangle((14,319,475,352),radius=10,fill="#172b3a")
        self.text(draw,(28,336),
                  "Cursor centered and hidden" if state.locked else
                  "Click LOCK CURSOR, move mouse, then Escape to release",
                  12,"#def9f4","lm")
        if not state.locked:
            self.button(draw,LOCK_RECT,"LOCK CURSOR / LOOK AROUND")
        self.button(draw,NEXT_RECT,
                    "FINISH TEST" if state.look_distance>=10 else "LOOK AROUND FIRST",
                    active=state.look_distance>=10 and not state.locked)

    def complete(self, image, state, now):
        from PIL import ImageDraw
        elapsed=max(0.0,now-state.finished_at)
        draw=ImageDraw.Draw(image)
        image.paste("#0a1a29",(0,0,WIDTH,HEIGHT))
        draw=ImageDraw.Draw(image)
        radius=int(min(800,elapsed*780))
        cx,cy=WIDTH//2,HEIGHT//2
        draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),
                     fill="#ffffff")
        if elapsed >= .52:
            # The PNG is the exact approved banner displayed at colabcola.com,
            # not a guess at the visual identity or a hand-drawn linked mark.
            image.paste(self.brand_banner, (35, 92))
            draw = ImageDraw.Draw(image)
            self.text(draw,(320,272),"interactive bridge test finished",
                      16,"#243647","mm")
            self.text(draw,(320,307),"Close the viewer when ready",
                      12,"#63758a","mm")
        return image

    def frame(self, state, now):
        image,draw=self.base(state)
        if state.stage==0:
            self.render_keyboard(draw,state)
        elif state.stage==1:
            self.render_buttons(draw,state)
        elif state.stage==2:
            self.render_wheel(draw,state)
        elif state.stage==3:
            self.voxel_world(image,state)
        else:
            image=self.complete(image,state,now)
        return P6+image.tobytes()


def visual_frame_key(state, now):
    """Return only state visible in the remote image.

    Input ACK is NOT part of the key: every new output frame gets a fresh CCB1
    header/ACK even if the compressed RGB image can safely be reused.
    """
    if state.stage == 0:
        return (0, tuple(sorted(state.seen_keys)))
    if state.stage == 1:
        return (1, tuple(sorted(state.seen_buttons)), tuple(state.clicks))
    if state.stage == 2:
        return (2, state.wheel_up, state.wheel_down)
    if state.stage == 3:
        return (3, state.locked, state.yaw, state.pitch,
                state.look_distance >= 10)
    if state.stage == 4:
        # Refresh at 30Hz while the white circle expands, then retain the
        # completed splash image without re-drawing it on every video tick.
        elapsed = max(0.0, now - state.finished_at)
        return (4, min(35, int(elapsed * FPS)))
    raise ValueError("invalid Input Lab stage")


class InputLabFrameCache:
    """Reuse exact zlib-encoded image bytes between visual state changes."""

    def __init__(self):
        self._key = None
        self._compressed = None

    def frame(self, painter, state, now):
        signature = visual_frame_key(state, now)
        if signature == self._key and self._compressed is not None:
            return self._compressed, 0.0, 0.0, True
        started = time.monotonic()
        ppm = painter.frame(state, now)
        render_ms = (time.monotonic() - started) * 1000
        if len(ppm) != len(P6) + WIDTH * HEIGHT * 3 or not ppm.startswith(P6):
            raise RuntimeError("invalid lab image dimensions")
        compress_start = time.monotonic()
        compressed = zlib.compress(ppm, level=1)
        compress_ms = (time.monotonic() - compress_start) * 1000
        if not 0 < len(compressed) <= FRAME_LIMIT:
            raise RuntimeError("lab intermediate image exceeds bound")
        self._key = signature
        self._compressed = compressed
        return compressed, render_ms, compress_ms, False


class InputLabSource:
    def __init__(self,directory,run_id,fps=30,sequence_buffered=True):
        if (not isinstance(run_id,str) or RUN_RE.fullmatch(run_id) is None
                or type(fps) is not int or fps!=FPS
                or sequence_buffered is not True):
            raise ValueError("Input Lab requires 30Hz buffered source")
        self.root=Path(directory)
        self.run_id=run_id
        self.events=queue.Queue(maxsize=QUEUE_LIMIT)
        self._seq=0
        self.state=LabState()
        self._lock=threading.Lock()
        self._lease_lock=threading.Lock()
        self._started_at=time.monotonic()
        self._last_host_credit_at=None
        self._host_credit_count=0
        self._stop=threading.Event()
        self._thread=None

    def submit_input(self,event):
        record=validate_event(event)
        with self._lock:
            if self._stop.is_set() or record["seq"]<=self._seq:
                return False
            try:
                self.events.put_nowait(record)
            except queue.Full:
                return False
            self._seq=record["seq"]
            return True

    def note_host_credit(self):
        """Called only after a validated, authenticated video-credit grant."""
        with self._lease_lock:
            self._last_host_credit_at=time.monotonic()
            self._host_credit_count+=1

    def host_credit_lease_expired(self,now=None):
        now=time.monotonic() if now is None else now
        with self._lease_lock:
            latest=self._last_host_credit_at
            started=self._started_at
        if latest is None:
            return now-started>=INITIAL_HOST_CREDIT_TIMEOUT_S
        return now-latest>=ACTIVE_HOST_CREDIT_TIMEOUT_S

    def _status(self,state,gpu=None,error_type=None,reason=None):
        payload={"state":state,"run_id":self.run_id,"application":"inputlab"}
        if gpu is not None:
            payload.update(device="cuda",gpu=gpu,width=WIDTH,height=HEIGHT)
        if error_type is not None:
            payload["error_type"]=error_type
        if reason is not None:
            payload["reason"]=reason
        _atomic_bytes(self.root/"status.json",
                      (json.dumps(payload,sort_keys=True)+"\n").encode())

    def _worker(self):
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("external T4 GPU unavailable")
            gpu=torch.cuda.get_device_name(0)
            if "T4" not in gpu:
                raise RuntimeError("external GPU is not T4")
            painter=LabPainter()
            cache=InputLabFrameCache()
            frame_seq=0
            window_seq=0
            window_since=time.monotonic()
            render_times=[]
            compress_times=[]
            publish_times=[]
            count=0
            stop_reason="normal"
            while not self._stop.is_set() and not (self.root/"stop").exists():
                tick=time.monotonic()
                if self.host_credit_lease_expired(tick):
                    stop_reason="host_credit_lease_expired"
                    break
                for _ in range(32):
                    try:
                        e=self.events.get_nowait()
                    except queue.Empty:
                        break
                    with self._lock:
                        self.state.apply(e,now=tick)
                # Snapshot small logical state under lock, then render
                # outside it; expensive Pillow drawing must never block the
                # notebook's authenticated Comm input or credit callbacks.
                with self._lock:
                    state_snapshot=copy.deepcopy(self.state)
                    ack=state_snapshot.last_input_seq
                compressed,render_ms,compress_ms,_=cache.frame(
                    painter,state_snapshot,time.monotonic())
                packed=HEADER.pack(MAGIC,frame_seq,ack,WIDTH,HEIGHT)+compressed
                publish_start=time.monotonic()
                _atomic_bytes(self.root/f"frame-ring-{frame_seq % SLOTS:02d}.bin",packed)
                _atomic_bytes(self.root/"frame.bin",packed)
                publish_ms=(time.monotonic()-publish_start)*1000
                if frame_seq==0:
                    self._status("ready",gpu=gpu)
                render_times.append(render_ms)
                compress_times.append(compress_ms)
                publish_times.append(publish_ms)
                count+=1
                now=time.monotonic()
                if now-window_since>=1:
                    elapsed=now-window_since
                    window_seq+=1
                    payload={
                        "schema":1,"run_id":self.run_id,
                        "seq":frame_seq,"window_seq":window_seq,
                        "published_count":count,
                        "window_elapsed_ms":round(elapsed*1000,3),
                        "target_fps":FPS,"rate_seq":0,
                        "application":"inputlab",
                        "input_lab_stage":state_snapshot.stage,
                        "input_lab_credit_count":self._host_credit_count,
                        "renderer_fps":round(count/elapsed,2),
                        "render_copy_ms":round(median(render_times),2),
                        "render_copy_p95_ms":_p95(render_times),
                        "compress_ms":round(median(compress_times),2),
                        "compress_p95_ms":_p95(compress_times),
                        "publish_ms":round(median(publish_times),2),
                        "publish_p95_ms":_p95(publish_times),
                    }
                    _atomic_bytes(self.root/"render-metrics.json",
                                  (json.dumps(payload,separators=(",",":"))+"\n").encode())
                    window_since=now
                    render_times.clear()
                    compress_times.clear()
                    publish_times.clear()
                    count=0
                frame_seq+=1
                self._stop.wait(max(0,1/FPS-(time.monotonic()-tick)))
            self._status("stopped",gpu=gpu,reason=stop_reason)
        except Exception as exc:
            if not self._stop.is_set():
                self._status("failed",error_type=type(exc).__name__)
        finally:
            self._stop.set()

    def start(self):
        if self._thread is not None:
            raise ValueError("Input Lab already started")
        self.root.mkdir(parents=True,exist_ok=True)
        self._thread=threading.Thread(
            target=self._worker,daemon=True,name="colabcola-input-lab")
        self._thread.start()
        return self._thread

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


def start(directory,fps,run_id,*,sequence_buffered=False):
    global _CONTROLLER
    if _CONTROLLER is not None:
        raise ValueError("Input Lab controller already active")
    _CONTROLLER=InputLabSource(directory,run_id,fps,sequence_buffered)
    return _CONTROLLER.start()


def submit_input(event):
    if _CONTROLLER is None:
        raise ValueError("Input Lab not running")
    return _CONTROLLER.submit_input(event)


def note_host_credit():
    if _CONTROLLER is None:
        raise ValueError("Input Lab source not running")
    _CONTROLLER.note_host_credit()


def stop():
    global _CONTROLLER
    if _CONTROLLER is not None:
        _CONTROLLER.close()
        _CONTROLLER=None
