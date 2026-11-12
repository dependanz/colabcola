"""Input Lab: offline UI state, app protocol, pointer safety and frame parity."""
from __future__ import annotations

import argparse
import json
import math
import queue
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from colabcola.bridge_input_lab_runtime import (
    WIDTH, HEIGHT, KEYS, P6, NEXT_RECT, LOCK_RECT,
    LabState, InputLabSource, validate_event, HEADER, MAGIC,
)
from colabcola.bridge_input_lab_host import InputLabHostControls
from colabcola.bridge_comm import KernelCommBridge
from colabcola.bridge_colab_attach import validate_attach_options
from colabcola.lifecycle_io import LifecycleError

RUN = "d" * 32


def evt(seq, event, **extra):
    return {"seq": seq, "event": event, **extra}


def press_next(state, seq):
    x = (NEXT_RECT[0]+NEXT_RECT[2])/(2*WIDTH)
    y = (NEXT_RECT[1]+NEXT_RECT[3])/(2*HEIGHT)
    assert state.apply(evt(seq, "mouse_button_down", button=1, x=x, y=y))
    assert state.apply(evt(seq+1, "mouse_button_up", button=1, x=x, y=y))


def test_keyboard_grid_records_many_keys_and_releases():
    state=LabState()
    many=("a","z","1","0","F1","F12","Shift_L",
          "space","Escape","Up","KP_5","Return")
    for idx,key in enumerate(many, 1):
        assert key in KEYS
        assert state.apply(evt(2*idx-1,"key_down",key=key))
        assert state.apply(evt(2*idx,"key_up",key=key))
    assert state.seen_keys==set(many)
    assert not state.held_keys
    assert state.stage==0
    assert not state.apply(evt(2*len(many),"key_down",key="b"))


def test_mouse_clicks_and_wheel_progress_through_all_four_stages():
    state=LabState()
    press_next(state, 1)
    assert state.stage==1
    for i,button in enumerate((1,2,3)):
        assert state.apply(evt(3+2*i,"mouse_button_down",button=button,x=.3,y=.5))
        assert state.apply(evt(4+2*i,"mouse_button_up",button=button,x=.3,y=.5))
    assert state.seen_buttons=={1,2,3}
    press_next(state, 10)
    assert state.stage==2
    assert state.apply(evt(12,"mouse_wheel",ticks=4))
    assert state.apply(evt(13,"mouse_wheel",ticks=-5))
    assert state.wheel_up==4 and state.wheel_down==5
    press_next(state,14)
    assert state.stage==3
    assert state.apply(evt(16,"mouse_delta",dx=50,dy=20))
    assert state.look_distance==0
    assert state.apply(evt(17,"pointer_lock",enabled=True))
    assert state.locked
    assert state.apply(evt(18,"mouse_delta",dx=50,dy=-20))
    assert state.look_distance==70
    assert state.yaw>0 and state.pitch>0
    # Can't click Finish while locked.
    press_next(state,19)
    assert state.stage==3
    assert state.apply(evt(21,"key_down",key="Escape"))
    assert state.locked is False
    press_next(state,22)
    assert state.stage==4 and state.finished_at is not None


def test_no_accidental_navigation_or_premature_completion():
    state=LabState()
    x=(NEXT_RECT[0]+NEXT_RECT[2])/(2*WIDTH)
    y=(NEXT_RECT[1]+NEXT_RECT[3])/(2*HEIGHT)
    assert state.apply(evt(1,"mouse_button_up",button=1,x=x,y=y))
    assert state.stage==0
    # Down outside button / up inside does not advance.
    assert state.apply(evt(2,"mouse_button_down",button=1,x=.1,y=.1))
    assert state.apply(evt(3,"mouse_button_up",button=1,x=x,y=y))
    assert state.stage==0
    press_next(state,4)
    press_next(state,6)
    press_next(state,8)
    assert state.stage==3
    press_next(state,10)
    assert state.stage==3
    assert state.finished_at is None


@pytest.mark.parametrize("bad",[
    evt(0,"key_down",key="a"),
    evt(1,"key_down",key="unknown"),
    evt(1,"key_down",key=[]),
    evt(1,"key_down",key="a",cmd="sudo"),
    evt(1,"mouse_move",x=float("nan"),y=.5),
    evt(1,"mouse_move",x=2,y=.5),
    evt(1,"mouse_button_up",button=True,x=.5,y=.5),
    evt(1,"mouse_button_down",button=4,x=.5,y=.5),
    evt(1,"mouse_delta",dx=65,dy=0),
    evt(1,"mouse_delta",dx=.5,dy=0),
    evt(1,"mouse_wheel",ticks=9),
    evt(1,"mouse_wheel",ticks=0),
    evt(1,"pointer_lock",enabled=1),
    evt(1,"pointer_lock",enabled=True,extra="bad"),
])
def test_input_validation_strict(bad):
    with pytest.raises(ValueError):
        validate_event(bad)


def test_remote_bounded_ordered_fifo(tmp_path):
    src=InputLabSource(tmp_path,RUN)
    assert src.submit_input(evt(1,"key_down",key="a"))
    assert src.submit_input(evt(2,"key_up",key="a"))
    assert not src.submit_input(evt(2,"key_down",key="z"))
    assert src.events.get_nowait()["event"]=="key_down"
    assert src.events.get_nowait()["event"]=="key_up"
    src.close()
    assert not src.submit_input(evt(3,"key_down",key="b"))


def opts(app="inputlab",fps=30,credits=6,**extra):
    values=dict(
        application=app,transport="comm",video_codec="h264",
        push_video=True,sequence_buffered=True,producer_zlib_level0=False,
        fps=fps,poll_hz=fps,inflight=1,push_credits=credits,
        benchmark_h264_capacity=False,benchmark_h264_credits=False,
        confirm_allocation=False,preflight_only=True,confirm_attach=False,
        renderer_lag_frames=1,
    )
    values.update(extra)
    return argparse.Namespace(**values)


def test_opt_in_inputlab_bridge_and_attach_guard():
    validate_attach_options(opts())
    valid=KernelCommBridge(None,None,RUN,30,video_codec="h264",
                           video_push=True,sequence_buffered=True,
                           push_credits=6,application="inputlab")
    assert valid.application=="inputlab"
    for x in (opts(fps=60,credits=10),
              opts(push_video=False),
              opts(sequence_buffered=False),
              opts(producer_zlib_level0=True),
              opts(application="unknown")):
        with pytest.raises(LifecycleError):
            validate_attach_options(x)
    with pytest.raises(ValueError):
        KernelCommBridge(None,None,RUN,60,video_codec="h264",
                         video_push=True,sequence_buffered=True,
                         push_credits=10,application="inputlab")


def test_remote_bootstrap_selects_inputlab_before_any_cuda():
    from colabcola import bridge_colab_comm_runtime as remote
    with patch.object(remote.runpy,"run_path",
                      side_effect=RuntimeError("input lab source selected")) as fn:
        with pytest.raises(RuntimeError,match="input lab source selected"):
            remote.start("/tmp/not-an-actual-run",RUN,30,
                         "/synthetic.py",preflight_video=True,
                         sequence_buffered=True,video_push=True,
                         video_source_path="/source.py",
                         video_push_source_path="/push.py",
                         push_credits=6,application="inputlab",
                         game_renderer_path="/inputlab.py")
        fn.assert_called_once_with("/inputlab.py")


def test_host_pointer_lock_is_explicit_and_restores_on_esc_and_focus():
    events=[]
    class Label:
        def __init__(self): self.cursors=[]
        def configure(self,**kw): self.cursors.append(kw)
        def focus_set(self): pass
    root=SimpleNamespace()
    label=Label()
    warps=[]
    host=InputLabHostControls(root,label,events.append,
                               pointer_warp=lambda:warps.append(1))
    class Mouse:
        def __init__(self,x=0,y=0,num=1,delta=0,keysym=""):
            self.x,self.y,self.num,self.delta,self.keysym=x,y,num,delta,keysym
    x=(NEXT_RECT[0]+NEXT_RECT[2])//2
    y=(NEXT_RECT[1]+NEXT_RECT[3])//2
    for _ in range(3):
        host.button_down(Mouse(x,y))
        host.button_up(Mouse(x,y))
    assert host.stage==3
    lx=(LOCK_RECT[0]+LOCK_RECT[2])//2
    ly=(LOCK_RECT[1]+LOCK_RECT[3])//2
    host.button_down(Mouse(lx,ly))
    host.button_up(Mouse(lx,ly))
    assert host.locked and len(warps)==1
    assert any(e=={"event":"pointer_lock","enabled":True} for e in events)
    host.motion(Mouse(350,180))
    assert any(e["event"]=="mouse_delta" for e in events)
    host.key_down(Mouse(keysym="Escape"))
    assert not host.locked
    assert label.cursors[-1]=={"cursor":""}
    assert events[-1]=={"event":"pointer_lock","enabled":False}
    # Lock again and lose focus: no hidden cursor survives.
    host.button_down(Mouse(lx,ly))
    host.button_up(Mouse(lx,ly))
    host.release_all()
    assert not host.locked
    assert label.cursors[-1]=={"cursor":""}


def test_host_wheel_and_middle_right_clicks_are_forwarded():
    events=[]
    host=InputLabHostControls(object(),SimpleNamespace(
        focus_set=lambda:None),events.append,pointer_warp=lambda:None)
    mouse=SimpleNamespace(x=240,y=150,num=2,delta=0)
    host.button_down(mouse);host.button_up(mouse)
    mouse.num=3
    host.button_down(mouse);host.button_up(mouse)
    host.wheel(SimpleNamespace(delta=240))
    host.wheel(SimpleNamespace(delta=-120))
    assert [e["event"] for e in events]==[
        "mouse_button_down","mouse_button_up",
        "mouse_button_down","mouse_button_up",
        "mouse_wheel","mouse_wheel",
    ]
    assert events[-2]["ticks"]==2 and events[-1]["ticks"]==-1


def test_host_cursor_lock_failure_does_not_claim_lock():
    class Label:
        def configure(self,**kw):pass
    events=[]
    host=InputLabHostControls(object(),Label(),events.append,
                              pointer_warp=lambda: (_ for _ in ()).throw(
                                  RuntimeError("no pointer warp")))
    host.stage=3
    assert host.lock_pointer() is False
    assert host.locked is False
    assert not events


def test_remote_pillow_render_and_ccb1_roundtrip():
    pytest.importorskip("PIL")
    from colabcola.bridge_input_lab_runtime import LabPainter
    import zlib
    painter=LabPainter()
    state=LabState()
    image=painter.frame(state,1.0)
    assert image.startswith(P6)
    assert len(image)==len(P6)+WIDTH*HEIGHT*3
    data=HEADER.pack(MAGIC,17,5,WIDTH,HEIGHT)+zlib.compress(image,1)
    # Existing CCB1 reader must accept Input Lab image without custom decoder.
    from colabcola.bridge_video_colab_source import read_latest_rgb
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        path=Path(root)/"frame.bin"
        path.write_bytes(data)
        sequence,ack,pixels=read_latest_rgb(path)
    assert (sequence,ack)==(17,5)
    assert len(pixels)==WIDTH*HEIGHT*3


def test_voxel_pointer_lock_changes_image_and_finish_wipes_white():
    pytest.importorskip("PIL")
    from colabcola.bridge_input_lab_runtime import LabPainter
    painter=LabPainter()
    state=LabState()
    state.stage=3
    before=painter.frame(state,10.0)
    state.apply(evt(1,"pointer_lock",enabled=True))
    state.apply(evt(2,"mouse_delta",dx=60,dy=25))
    after=painter.frame(state,10.1)
    assert before!=after
    state.apply(evt(3,"pointer_lock",enabled=False))
    press_next(state,4)
    assert state.stage==4
    start=painter.frame(state,state.finished_at)
    end=painter.frame(state,state.finished_at+2)
    assert start!=end
    center=len(P6)+(HEIGHT//2*WIDTH+WIDTH//2)*3
    assert end[center:center+3]!=start[center:center+3]


def test_lab_cli_is_explicit_external_30hz(monkeypatch):
    from colabcola.cli import main
    from colabcola import bridge_colab
    seen=[]
    monkeypatch.setattr(bridge_colab,"run_colab",
                        lambda args:seen.append(args) or 0)
    args=["--profile","high-credit-dev","bridge","colab",
          "--application","inputlab","--attach-existing","--sequence-buffered",
          "--push-video","--push-credits","6","--renderer-lag-frames","1",
          "--transport","comm","--video-codec","h264","--fps","30",
          "--poll-hz","30","--inflight","1","--max-seconds","120",
          "--preflight-only"]
    assert main(args)==0 and seen[-1].application=="inputlab"
    with pytest.raises(SystemExit):
        main([x for x in args if x!="--attach-existing"])


def test_input_lab_remote_worker_produces_acknowledged_ccb1_frames(
        monkeypatch, tmp_path):
    pytest.importorskip("PIL")
    import sys
    import time
    from types import SimpleNamespace
    from colabcola.bridge_video_colab_source import read_latest_rgb

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda index: "Tesla T4"))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    src = InputLabSource(tmp_path / "lab-run", RUN)
    worker = src.start()
    deadline = time.monotonic() + 3
    status_path = src.root / "status.json"
    try:
        while time.monotonic() < deadline and not status_path.exists():
            time.sleep(.01)
        assert status_path.exists(), "remote lab never reported ready"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        assert status["state"] == "ready", status
        assert status["application"] == "inputlab"
        assert src.submit_input(evt(1, "key_down", key="space"))

        ack = 0
        while time.monotonic() < deadline:
            seq, ack, pixels = read_latest_rgb(src.root / "frame.bin")
            if ack == 1:
                assert len(pixels) == WIDTH * HEIGHT * 3
                break
            time.sleep(.02)
        assert ack == 1, "keyboard event was not acknowledged in video"
    finally:
        src.close()
    worker.join(timeout=2)
    assert not worker.is_alive()
    final = json.loads(status_path.read_text(encoding="utf-8"))
    assert final["state"] == "stopped"


def test_windows_cursor_hide_fallback_is_paired(monkeypatch):
    from colabcola import bridge_input_lab_host as host_mod
    events = []
    order = []

    class Label:
        def configure(self, *, cursor):
            order.append(("cursor", cursor))
            if cursor == "none":
                raise RuntimeError("Tk Windows does not support none cursor")

    monkeypatch.setattr(host_mod, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        host_mod.InputLabHostControls, "_windows_hide_cursor",
        lambda self: order.append(("hide",)))
    monkeypatch.setattr(
        host_mod.InputLabHostControls, "_restore_native_cursor",
        lambda self: order.append(("restore",)))
    host = host_mod.InputLabHostControls(
        object(), Label(), events.append, pointer_warp=lambda: order.append(("warp",)))
    host.stage = 3
    assert host.lock_pointer() is True
    host.unlock_pointer()
    assert order == [
        ("cursor", "none"), ("hide",), ("warp",),
        ("cursor", ""), ("restore",),
    ]
    assert events == [
        {"event": "pointer_lock", "enabled": True},
        {"event": "pointer_lock", "enabled": False},
    ]
