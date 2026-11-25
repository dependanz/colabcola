"""v38A.1: opt-in diagnostics, credit parity, bounded failure evidence."""
import json
import queue
import struct
import threading
import time
from types import SimpleNamespace

import pytest
from colabcola.bridge_pipeline_trace import PipelineTrace, validated_credit_snapshot
from colabcola.bridge_video_push_runtime import CreditPushSender
from colabcola.bridge_video_push_session import validated_sender_diagnostics
from colabcola.bridge_comm import KernelCommBridge
from colabcola.lifecycle_io import LifecycleError

RUN = "a" * 32


@pytest.mark.parametrize("opt_in", [False, True])
def test_30hz_sender_trace_is_opt_in(tmp_path, opt_in):
    class Source:
        fps = 30
        sequence_buffered = True
        last_renderer_seq = None
        def pull(self, seq):
            self.last_renderer_seq = seq - 1
            return {"kind": "video_access_unit", "pull_seq": seq}

    packets = []
    sender = CreditPushSender(comm=SimpleNamespace(send=packets.append), source=Source(),
                              directory=tmp_path, run_id=RUN, credits=6,
                              trace_pipeline=opt_in)
    sender.grant(6)
    for seq in range(2):
        (tmp_path / "frame.bin").write_bytes(
            struct.pack("!4sIIII", b"CCB1", seq, 0, 640, 360))
        assert sender.step()
        sender.grant(1)
    assert sender.available == 6
    assert sender.sent_count == sender.replenished == 2
    if opt_in:
        assert validated_sender_diagnostics(packets[0]) is not None
        assert "sender_comm_previous_us" not in packets[0]
        assert packets[1]["sender_comm_previous_us"] >= 0
        assert validated_credit_snapshot(packets[1], 6, 2) == {
            "available": 5, "sent": 2, "replenished": 1}
    else:
        assert all(not any(k.startswith("sender_") for k in p) for p in packets)
    sender.close()


@pytest.mark.parametrize("bad", [
    {}, {"sender_available_credits": True, "sender_sent_count": 2, "sender_replenished_count": 1},
    {"sender_available_credits": 6, "sender_sent_count": 2, "sender_replenished_count": 1},
    {"sender_available_credits": 5, "sender_sent_count": 3, "sender_replenished_count": 2},
])
def test_credit_snapshot_invalid_is_ignored(bad):
    assert validated_credit_snapshot(bad, 6, 2) is None


def test_trace_bounded_and_never_overwrites(tmp_path):
    path = tmp_path / "trace.jsonl"
    trace = PipelineTrace(path, limit=2)
    for i in range(4):
        trace.add("input_sent", input_seq=i)
    assert not path.exists()
    trace.finish(outcome="failed", phase="freshness_watchdog",
                 decoded_frames=0, last_video_seq=0, last_input_ack=0)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["header", "input_sent", "input_sent", "terminal"]
    assert [r["record_seq"] for r in rows[1:3]] == [3, 4]
    assert rows[-1]["dropped_oldest_records"] == 2
    original = path.read_bytes()
    trace.finish(outcome="completed", phase="session_summary",
                 decoded_frames=0, last_video_seq=0, last_input_ack=0)
    assert path.read_bytes() == original


def test_trace_requires_buffered_30hz_at_host():
    with pytest.raises(ValueError, match="pipeline trace"):
        KernelCommBridge(None, None, RUN, fps=60, video_codec="h264",
                         sequence_buffered=True, video_push=True, push_credits=10,
                         trace_pipeline=True)


def test_watchdog_retains_trace_and_original_failure(tmp_path, monkeypatch):
    from colabcola import bridge_video_push_session as mod
    class Channel:
        def get_msg(self, timeout=0):
            raise queue.Empty
    ready = threading.Event()
    ready.set()
    stopped = threading.Event()
    peer = SimpleNamespace(fps=30, video_push=True, video_codec="h264",
                           push_credits=6, run_id=RUN, comm_id="owned",
                           wire=SimpleNamespace(connection_ready=ready, iopub_channel=Channel()),
                           _send=lambda *args, **kw: None)
    monkeypatch.setattr(mod, "make_host_decoder", lambda: (None, None))
    monkeypatch.setattr(mod, "PUSH_RECEIVE_WATCHDOG_S", -1.)
    path = tmp_path / "pipeline-trace.jsonl"
    with pytest.raises(LifecycleError, match="freshness watchdog fired"):
        mod.run_push_comm_h264(peer, "Tesla T4", queue.Queue(), queue.Queue(), stopped,
                               time.monotonic() + 1., credits=6, trace_path=path)
    assert stopped.is_set()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["header", "initial_credit", "terminal"]
    assert rows[-1]["outcome"] == "failed"
    assert rows[-1]["phase"] == "freshness_watchdog"
