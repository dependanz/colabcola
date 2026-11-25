"""v38A.3 correlation at the pinned WebSocket callback and host viewer."""
import json
import queue
import threading
import time
from types import SimpleNamespace, MethodType

import pytest

from colabcola.bridge_pipeline_trace import HostEventTrace
from colabcola.bridge_ws_probe import install_video_ingress, matched_video_sequence
from colabcola.bridge_viewer_stats import LocalFrameTiming, frame_trace_fields
from colabcola.bridge_demo import MeteredFrameQueue
from colabcola.bridge_colab import offer_update
from colabcola._vendor.jupyter_kernel_client.wsclient import (
    KernelWebSocketClient, JupyterSubprotocol,
)

RUN = "a" * 32


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def message(seq, *, run=RUN, comm="owned", kind="video_access_unit"):
    return {"channel":"iopub", "msg_id":"not-retained", "msg_type":"comm_msg",
            "header":{"msg_type":"comm_msg"},
            "content":{"comm_id":comm, "data":{"v":1, "kind":kind,
                        "run_id":run, "pull_seq":seq, "data":"private media"}}}


def test_ingress_wraps_real_pinned_vendor_callback_without_mutation(tmp_path):
    path = tmp_path / "wire.jsonl"
    trace = HostEventTrace(path, RUN, "websocket-ingress")
    wire = SimpleNamespace(_subprotocol=JupyterSubprotocol.DEFAULT,
                           _iopub_msg_queue=queue.Queue(),
                           _message_received=threading.Event(),
                           log=SimpleNamespace(debug=lambda *_a, **_k: None))
    wire._on_message = MethodType(KernelWebSocketClient._on_message, wire)
    current_comm = [None]
    install_video_ingress(wire, trace=trace, run_id=RUN,
                          comm_id_getter=lambda: current_comm[0])
    before_ready = message(1)
    wire._on_message(None, json.dumps(before_ready))
    assert wire._iopub_msg_queue.get_nowait() == before_ready
    current_comm[0] = "owned"
    inputs = [message(2, run="b"*32), message(3, comm="other"),
              message(4, kind="not-video"), message(5), message(6, kind="video_no_new")]
    for msg in inputs:
        wire._on_message(None, json.dumps(msg))
    assert wire._message_received.is_set()
    assert [wire._iopub_msg_queue.get_nowait() for _ in inputs] == inputs
    # Direct queue puts are not WebSocket callbacks.
    wire._iopub_msg_queue.put_nowait(message(7))
    assert wire._iopub_msg_queue.get_nowait() == message(7)
    assert trace.finish(outcome="completed")
    rows = read_rows(path)
    assert [r["video_seq"] for r in rows[1:-1]] == [5,6]
    assert [r["iopub_depth_before"] for r in rows[1:-1]] == [3,4]
    for row in rows[1:-1]:
        assert row["callback_ns"] <= row["enqueue_start_ns"] <= row["enqueue_return_ns"]
    assert "private media" not in path.read_text()
    assert "not-retained" not in path.read_text()


@pytest.mark.parametrize("bad", [True,0,-1,100_000_001,"1",None])
def test_ingress_rejects_bad_candidate_sequence(bad):
    assert matched_video_sequence(message(bad), RUN, "owned") is None


def test_ingress_failed_original_put_never_fabricates_enqueue(tmp_path):
    trace = HostEventTrace(tmp_path/"wire.jsonl", RUN, "websocket-ingress")
    q = queue.Queue(maxsize=1)
    q.put_nowait("already full")
    wire = SimpleNamespace(_iopub_msg_queue=q)
    wire._on_message = lambda _socket, msg: q.put_nowait(msg)
    install_video_ingress(wire, trace=trace, run_id=RUN,
                          comm_id_getter=lambda:"owned")
    with pytest.raises(queue.Full):
        wire._on_message(None, message(1))
    q.get_nowait()
    q.put_nowait(message(2))
    trace.finish(outcome="failed")
    assert [r["event"] for r in read_rows(trace.path)] == ["header","terminal"]


def test_viewer_queue_correlates_discard_and_remaining_frame(tmp_path):
    trace = HostEventTrace(tmp_path/"gui.jsonl", RUN, "tk-viewer")
    q = MeteredFrameQueue(maxsize=2, presentation_trace=trace)
    def item(seq):
        return ("frame-timed",(b"not-recorded",LocalFrameTiming(
            100,110,None,video_seq=seq,renderer_seq=2*seq,input_ack=seq)))
    offer_update(q, item(1))
    offer_update(q, ("status","ready"))
    offer_update(q, item(2))
    assert q.get_nowait()[0] == "status"
    q.trace_pending()
    assert trace.finish(outcome="completed")
    rows = read_rows(trace.path)
    assert [r["event"] for r in rows] == [
        "header","gui_discarded","gui_pending_at_close","gui_queue_terminal","terminal"]
    assert rows[1]["video_seq"]==1 and rows[2]["video_seq"]==2
    assert rows[3]["coalesced_video_frames"]==1
    assert rows[3]["pending_items"]==1
    assert q.qsize()==1
    assert "not-recorded" not in trace.path.read_text()
    assert frame_trace_fields(LocalFrameTiming(100,110)) is None


def test_host_trace_bound_and_concurrent_record_order(tmp_path):
    trace = HostEventTrace(tmp_path/"trace.jsonl",RUN,"tk-viewer",limit=5)
    threads = [threading.Thread(target=lambda: [trace.add("event", value=i)
                                                  for i in range(10)])
               for _ in range(2)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not trace.path.exists()
    assert trace.finish(outcome="completed")
    rows=read_rows(trace.path)
    assert [r["record_seq"] for r in rows[1:-1]] == list(range(16,21))
    assert rows[-1]["dropped_oldest_records"]==15
    assert rows[-1]["total_records"]==20
    original=trace.path.read_bytes()
    trace.add("late")
    assert not trace.finish(outcome="failed")
    assert trace.path.read_bytes()==original


def test_host_video_ids_follow_validated_frame_into_viewer(tmp_path,monkeypatch):
    from colabcola import bridge_video_push_session as mod
    from colabcola.bridge_video_comm_protocol import encode_video_reply
    from colabcola.bridge_video_stream import VideoAccessUnit
    from test_bridge_video_v34_push import IDR
    ready=threading.Event()
    ready.set()
    stopped=threading.Event()
    incoming=queue.Queue()
    outgoing=queue.Queue()
    frames=MeteredFrameQueue(maxsize=2)
    sent=[]
    class Channel:
        def get_msg(self,timeout=0):
            return incoming.get_nowait()
    class Peer:
        fps=30
        video_push=True
        video_codec="h264"
        push_credits=6
        run_id=RUN
        comm_id="owned"
        wire=SimpleNamespace(connection_ready=ready,iopub_channel=Channel())
        def _send(self,kind,**kw):
            sent.append((kind,kw))
            if kw.get("count")==6:
                unit=VideoAccessUnit(seq=0,pts_ns=0,capture_ns=1,
                                     data=IDR,keyframe=True)
                payload=encode_video_reply(run_id=RUN,pull_seq=1,input_ack=0,unit=unit)
                payload["renderer_seq"]=42
                incoming.put({"header":{"msg_type":"comm_msg"},
                              "content":{"comm_id":"owned","data":payload}})
            else:
                stopped.set()
    frame=SimpleNamespace(width=640,height=360,
                          to_ndarray=lambda **_k: SimpleNamespace(tobytes=lambda:b"x"*(640*360*3)))
    monkeypatch.setattr(mod,"make_host_decoder",lambda:(
        SimpleNamespace(Packet=lambda _data:SimpleNamespace(pts=None)),
        SimpleNamespace(decode=lambda _packet:[frame])))
    path=tmp_path/"pipeline-trace.jsonl"
    report=mod.run_push_comm_h264(Peer(),"Tesla T4",outgoing,frames,stopped,
                                  time.monotonic()+1,credits=6,trace_path=path)
    assert report["decoded_frames"]==1
    assert sent==[("video_credit",{"count":6}),("video_credit",{"count":1})]
    item=frames.get_nowait()
    if item[0]=="status":
        item=frames.get_nowait()
    timing=item[1][1]
    assert frame_trace_fields(timing)=={"video_seq":1,"renderer_seq":42,"input_ack":0}
    rows=read_rows(path)
    assert rows[0]["run_id"]==RUN
    received=next(r for r in rows if r["event"]=="video_received")
    offered=next(r for r in rows if r["event"]=="frame_offered")
    assert received["video_seq"]==offered["video_seq"]==timing.video_seq
    assert received["host_dequeue_ns"]<=received["host_matched_ns"]<=received["host_validated_ns"]
    assert received["host_validated_ns"]<=offered["decode_start_ns"]<=offered["decoded_ns"]
    assert offered["decoded_ns"]==timing.decoded_ns
