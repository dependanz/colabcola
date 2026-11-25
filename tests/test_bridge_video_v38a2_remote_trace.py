"""Remote trace and Input Lab/sender lifecycle tests, entirely offline."""
import json
import struct
import sys
import time
from types import SimpleNamespace

import pytest

from colabcola import bridge_input_lab_runtime as labmod
from colabcola.bridge_remote_trace import RemotePipelineTrace
from colabcola.bridge_video_push_runtime import CreditPushSender

RUN = "a" * 32


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_remote_trace_bounded_idempotent_and_run_scoped(tmp_path):
    path = tmp_path / "trace.jsonl"
    trace = RemotePipelineTrace(path, RUN, "credit-sender", limit=2)
    for i in range(4):
        trace.add("credit_granted", count=i)
    assert not path.exists()
    assert trace.finish(outcome="stopped")
    rows = read_rows(path)
    assert rows[0]["run_id"] == RUN
    assert rows[0]["clock"] == "remote_monotonic_ns"
    assert [r["record_seq"] for r in rows[1:3]] == [3,4]
    assert rows[-1]["dropped_oldest_records"] == 2
    assert rows[-1]["total_records"] == 4
    assert rows[1]["remote_ns"] <= rows[2]["remote_ns"] <= rows[-1]["remote_ns"]
    original = path.read_bytes()
    trace.add("late")
    assert not trace.finish(outcome="failed")
    assert path.read_bytes() == original
    other = RemotePipelineTrace(path, RUN, "credit-sender")
    assert not other.finish(outcome="stopped")
    assert path.read_bytes() == original


def fake_gpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True,
                             get_device_name=lambda _i: "Tesla T4")))
    monkeypatch.setattr(labmod, "LabPainter", lambda: SimpleNamespace(
        frame=lambda state, now: labmod.P6+b"\0"*(640*360*3)))


@pytest.mark.parametrize("enabled", [False, True])
def test_lab_input_frame_correlation_and_cache_parity(tmp_path, monkeypatch, enabled):
    fake_gpu(monkeypatch)
    lab = labmod.InputLabSource(tmp_path, RUN, trace_pipeline=enabled)
    assert lab.submit_input({"seq":1, "event":"key_down", "key":"space"})
    original = labmod._atomic_bytes
    snapshots = []
    def publish(path, body):
        original(path, body)
        if path.name == "frame.bin":
            snapshots.append(body)
            if len(snapshots) == 2:
                lab._stop.set()
    monkeypatch.setattr(labmod, "_atomic_bytes", publish)
    lab._worker()
    assert len(snapshots) == 2
    assert [labmod.HEADER.unpack(s[:labmod.HEADER.size])[1:3]
            for s in snapshots] == [(0,1),(1,1)]
    assert snapshots[0][labmod.HEADER.size:] == snapshots[1][labmod.HEADER.size:]
    path = tmp_path/"remote-producer-trace.jsonl"
    if not enabled:
        assert not path.exists()
        return
    rows = read_rows(path)
    enqueued = next(r for r in rows if r["event"]=="input_enqueued")
    applied = next(r for r in rows if r["event"]=="input_applied")
    frames = [r for r in rows if r["event"]=="frame_published"]
    assert enqueued["input_seq"] == applied["input_seq"] == frames[0]["input_ack"] == 1
    assert enqueued["submit_start_ns"] <= enqueued["remote_ns"] <= applied["apply_start_ns"]
    assert applied["remote_ns"] <= frames[0]["snapshot_start_ns"]
    assert [f["renderer_seq"] for f in frames] == [0,1]
    assert [f["cache_hit"] for f in frames] == [False,True]
    for f in frames:
        assert f["snapshot_start_ns"] <= f["render_start_ns"] <= f["publish_start_ns"] <= f["remote_ns"]
    assert rows[-1]["outcome"] == "stopped"
    assert rows[-1]["last_published_seq"] == 1
    assert "space" not in path.read_text()
    assert '"key_down"' not in path.read_text()


def test_remote_trace_survives_lease_expiry_without_media(tmp_path, monkeypatch):
    fake_gpu(monkeypatch)
    lab = labmod.InputLabSource(tmp_path, RUN, trace_pipeline=True)
    lab._started_at = time.monotonic()-labmod.INITIAL_HOST_CREDIT_TIMEOUT_S-1
    lab._worker()
    rows = read_rows(tmp_path/"remote-producer-trace.jsonl")
    assert rows[-1]["outcome"] == "stopped"
    assert rows[-1]["reason"] == "host_credit_lease_expired"
    assert rows[-1]["last_published_seq"] == -1
    assert not (tmp_path/"frame.bin").exists()


def test_remote_trace_retains_renderer_failure_without_exception_text(tmp_path, monkeypatch):
    fake_gpu(monkeypatch)
    def broken():
        raise RuntimeError("private detail")
    monkeypatch.setattr(labmod, "LabPainter", broken)
    lab = labmod.InputLabSource(tmp_path, RUN, trace_pipeline=True)
    lab._worker()
    path = tmp_path/"remote-producer-trace.jsonl"
    rows = read_rows(path)
    assert rows[-1]["outcome"] == "failed"
    assert rows[-1]["phase"] == "initialization"
    assert "private detail" not in path.read_text()


@pytest.mark.parametrize("fail_send", [False,True])
def test_sender_independent_timeline_and_credit_accounting(tmp_path, fail_send):
    class Source:
        fps=30
        sequence_buffered=True
        last_renderer_seq=None
        def pull(self, seq):
            self.last_renderer_seq=seq-1
            return {"kind":"video_access_unit","pull_seq":seq,"input_ack":7}
    def send(reply):
        if fail_send:
            raise OSError("private detail")
    sender=CreditPushSender(comm=SimpleNamespace(send=send), source=Source(),
                            directory=tmp_path,run_id=RUN,credits=6,trace_pipeline=True)
    sender.grant(6)
    assert sender.step() is False
    (tmp_path/"frame.bin").write_bytes(struct.pack("!4sIIII",b"CCB1",0,7,640,360))
    assert sender.step() is (not fail_send)
    if not fail_send:
        sender.grant(1)
    sender.close()
    path=tmp_path/"remote-sender-trace.jsonl"
    rows=read_rows(path)
    events=[r["event"] for r in rows]
    assert events[:6]==["header","credit_granted","credit_reserved","source_pull_begin",
                        "source_pull_end","comm_send_begin"]
    assert events[6]==("send_failed" if fail_send else "comm_send_end")
    assert rows[2]["cumulative_source_misses"]==1
    assert rows[4]["video_seq"]==1 and rows[4]["renderer_seq"]==0
    assert rows[4]["input_ack"]==7
    for r in rows:
        if r["event"] in ("credit_granted","credit_reserved","terminal"):
            assert r["available"]+r["sent"]-r["replenished"]==6
    assert rows[-1]["outcome"]==("failed" if fail_send else "stopped")
    assert "private detail" not in path.read_text()
