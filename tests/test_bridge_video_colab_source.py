"""v33d: remote H.264 source/opt-in handler, entirely offline."""
from __future__ import annotations

import importlib.resources
from pathlib import Path
import runpy
import struct
import sys
import types
import zlib

import pytest

from colabcola.bridge_video_colab_source import (
    RemoteH264PullSource, RemoteVideoSourceError, read_latest_rgb,
)
from colabcola.bridge_video_comm_protocol import decode_video_reply, VideoSequenceGate

RUN = "7" * 32
W, H = 640, 360
IDR = b"\x00\x00\x00\x01\x65\x88\x84"
P = b"\x00\x00\x00\x01\x41\x88\x84"


def snapshot(path: Path, *, renderer_seq=0, ack=0, rgb=None):
    if rgb is None:
        rgb = bytes((51, 62, 74)) * (W * H)
    ppm = b"P6\n640 360\n255\n" + rgb
    path.write_bytes(struct.pack("!4sIIII", b"CCB1", renderer_seq, ack, W, H)
                     + zlib.compress(ppm, 1))


def fake_encoder_factory(width, height, fps):
    assert (width, height, fps) == (W, H, 12)
    def encode(pixels, seq):
        assert len(pixels) == W * H * 3
        return (IDR if seq % 12 == 0 else P), seq % 12 == 0
    return encode


def test_snapshot_validation_rejects_corruption_and_unbounded_data(tmp_path):
    path = tmp_path / "frame.bin"
    snapshot(path, renderer_seq=9, ack=4)
    seq, ack, rgb = read_latest_rgb(path)
    assert (seq, ack, rgb[:3], len(rgb)) == (9, 4, bytes((51, 62, 74)), W * H * 3)
    path.write_bytes(path.read_bytes() + b"junk")
    with pytest.raises(RemoteVideoSourceError):
        read_latest_rgb(path)
    snapshot(path)
    path.write_bytes(b"BAD!" + path.read_bytes()[4:])
    with pytest.raises(RemoteVideoSourceError):
        read_latest_rgb(path)
    path.write_bytes(b"x" * (2 * 1024 * 1024 + 21))
    with pytest.raises(RemoteVideoSourceError):
        read_latest_rgb(path)


def test_pulls_create_contiguous_encoded_seqs_from_noncontiguous_render_frames(tmp_path):
    file = tmp_path / "frame.bin"
    source = RemoteH264PullSource(
        tmp_path, RUN, fps=12, encoder_factory=fake_encoder_factory)
    gate = VideoSequenceGate()
    for pull, renderer_seq in ((1, 2), (2, 19), (3, 19), (4, 60)):
        snapshot(file, renderer_seq=renderer_seq, ack=pull)
        result = source.pull(pull)
        unit, ack = decode_video_reply(result, run_id=RUN, expected_pull_seq=pull)
        assert unit.seq == pull - 1
        assert result["renderer_seq"] == renderer_seq
        assert ack == pull
        assert gate.accept(unit)
    assert source.coded_seq == 4


def test_source_rejects_duplicate_pull_without_duplicating_access_unit(tmp_path):
    snapshot(tmp_path / "frame.bin")
    source = RemoteH264PullSource(
        tmp_path, RUN, fps=12, encoder_factory=fake_encoder_factory)
    assert source.pull(1)["pull_seq"] == 1
    with pytest.raises(RemoteVideoSourceError):
        source.pull(1)
    assert source.coded_seq == 1
    assert source.pull(2)["pull_seq"] == 2


def test_source_poisoned_when_applied_ack_regresses(tmp_path):
    file = tmp_path / "frame.bin"
    source = RemoteH264PullSource(
        tmp_path, RUN, fps=12, encoder_factory=fake_encoder_factory)
    snapshot(file, ack=3)
    assert source.pull(1)["input_ack"] == 3
    snapshot(file, ack=2)
    with pytest.raises(RemoteVideoSourceError):
        source.pull(2)
    assert source.failed
    snapshot(file, ack=4)
    with pytest.raises(RemoteVideoSourceError):
        source.pull(3)


def test_mismatched_fake_idr_claim_fails_without_payload_leak(tmp_path):
    snapshot(tmp_path / "frame.bin")
    source = RemoteH264PullSource(
        tmp_path, RUN, fps=12,
        encoder_factory=lambda *_: (lambda pixels, seq: (P, True)))
    with pytest.raises(RemoteVideoSourceError, match="remote H.264 pull failed"):
        source.pull(1)
    assert source.failed


def test_optional_real_pyav_produces_idr_and_non_idr(tmp_path):
    av = pytest.importorskip("av", reason="optional PyAV not installed")
    pytest.importorskip("numpy")
    try:
        av.CodecContext.create("libx264", "w")
    except Exception:
        pytest.skip("libx264 unavailable")
    file = tmp_path / "frame.bin"
    source = RemoteH264PullSource(tmp_path, RUN, fps=12)
    got = []
    for i in range(14):
        snapshot(file, renderer_seq=i, ack=0)
        got.append(decode_video_reply(source.pull(i + 1), run_id=RUN,
                                      expected_pull_seq=i + 1)[0])
    assert [u.seq for u in got] == list(range(14))
    assert got[0].keyframe
    assert any(not u.keyframe for u in got[1:12])
    assert got[12].keyframe


def _fake_kernel(monkeypatch):
    comms = []
    class FakeComm:
        def __init__(self, **kwargs):
            self.open = kwargs
            self.replies = []
            self.closed = False
            comms.append(self)
        def on_msg(self, callback):
            self.callback = callback
        def send(self, reply):
            self.replies.append(reply)
        def close(self):
            self.closed = True
        def call(self, kind, **fields):
            self.callback({"content": {"data": {
                "v": 1, "run_id": RUN, "kind": kind, **fields,
            }}})
    kernel = types.ModuleType("ipykernel")
    kernel_comm = types.ModuleType("ipykernel.comm")
    kernel_comm.Comm = lambda **kwargs: FakeComm(**kwargs)
    kernel.comm = kernel_comm
    monkeypatch.setitem(sys.modules, "ipykernel", kernel)
    monkeypatch.setitem(sys.modules, "ipykernel.comm", kernel_comm)
    return comms


def _fake_renderer(tmp_path):
    path = tmp_path / "renderer.py"
    path.write_text(
        "from pathlib import Path\nimport json\n"
        "def start(directory, fps, run_id):\n"
        " p=Path(directory)\n"
        " (p/'status.json').write_text(json.dumps(dict(run_id=run_id,state='ready',"
        "device='cuda',gpu='Tesla T4',width=640,height=360)))\n"
        "def atomic_bytes(path, body):\n Path(path).write_bytes(body)\n"
    )
    return str(path)


def test_kernel_handler_is_off_by_default_and_legacy_pull_works(tmp_path, monkeypatch):
    comms = _fake_kernel(monkeypatch)
    snapshot(tmp_path / "frame.bin", renderer_seq=3, ack=2)
    runtime = runpy.run_path(str(importlib.resources.files("colabcola") /
                                 "bridge_colab_comm_runtime.py"), run_name="test_module")
    runtime["start"](str(tmp_path), RUN, 12, _fake_renderer(tmp_path))
    comm = comms[0]
    comm.call("video_pull", pull_seq=1)
    assert comm.replies == []  # No legacy behavior changed, video not enabled.
    comm.call("pull", pull_seq=2)
    assert len(comm.replies) == 1
    assert comm.replies[0]["kind"] == "frame"
    comm.call("close")
    assert comm.closed


def test_opt_in_kernel_handler_validates_identity_order_and_reports_safe_error(tmp_path, monkeypatch):
    comms = _fake_kernel(monkeypatch)
    snapshot(tmp_path / "frame.bin", renderer_seq=18, ack=7)
    fake_source_file = tmp_path / "fake_video_source.py"
    fake_source_file.write_text(
        "class RemoteH264PullSource:\n"
        " def __init__(self, directory, run_id, **kwargs):\n"
        "  self.run_id=run_id\n"
        " def pull(self, pull_seq):\n"
        "  return dict(v=1, kind='video_access_unit', run_id=self.run_id, "
        "pull_seq=pull_seq, input_ack=7, codec='h264-annexb', data='AA==')\n"
    )
    runtime = runpy.run_path(str(importlib.resources.files("colabcola") /
                                 "bridge_colab_comm_runtime.py"), run_name="test_module")
    runtime["start"](str(tmp_path), RUN, 12, _fake_renderer(tmp_path),
                     video_source_path=str(fake_source_file))
    comm = comms[0]
    comm.call("video_pull", pull_seq=1)
    assert len(comm.replies) == 1
    assert comm.replies[0]["kind"] == "video_access_unit"
    assert comm.replies[0]["pull_seq"] == 1
    comm.call("video_pull", pull_seq=1)  # No duplicate answer.
    comm.call("video_pull", pull_seq=True)
    assert len(comm.replies) == 1
    comm.call("video_pull", pull_seq=2)
    assert len(comm.replies) == 2
    assert comm.replies[-1]["pull_seq"] == 2
    # Malformed source path is fail-closed; never return exception content.
    runtime["start"](str(tmp_path), RUN, 12, _fake_renderer(tmp_path),
                     video_source_path="/missing/private/path.py")
    second = comms[1]
    second.call("video_pull", pull_seq=1)
    assert second.replies[-1]["kind"] == "error"
    assert second.replies[-1]["run_id"] == RUN
    assert "private" not in str(second.replies[-1])
