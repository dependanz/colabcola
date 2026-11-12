"""v37C: Input Lab remote lease, sender observability and exact-run stop marker.

All tests are offline: no network, GPU allocation, Colab runtime or OS changes.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from colabcola.bridge_comm import KernelCommBridge
from colabcola.bridge_input_lab_runtime import (
    ACTIVE_HOST_CREDIT_TIMEOUT_S, INITIAL_HOST_CREDIT_TIMEOUT_S,
    InputLabSource,
)
from colabcola.bridge_video_push_runtime import CreditPushSender
from colabcola.lifecycle_io import LifecycleError

RUN = "a" * 32


def test_inputlab_lease_grace_before_first_credit_then_active_timeout(tmp_path):
    lab = InputLabSource(tmp_path / "run", RUN)
    before = lab._started_at
    assert not lab.host_credit_lease_expired(
        before + INITIAL_HOST_CREDIT_TIMEOUT_S - 1)
    assert lab.host_credit_lease_expired(
        before + INITIAL_HOST_CREDIT_TIMEOUT_S + 1)
    lab.note_host_credit()
    after = lab._last_host_credit_at
    assert lab._host_credit_count == 1
    assert not lab.host_credit_lease_expired(
        after + ACTIVE_HOST_CREDIT_TIMEOUT_S - 1)
    assert lab.host_credit_lease_expired(
        after + ACTIVE_HOST_CREDIT_TIMEOUT_S + 1)
    lab.note_host_credit()
    assert lab._host_credit_count == 2
    lab.close()


def test_inputlab_orphan_worker_self_stops_without_host_credits(
        tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    import sys
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _i: "Tesla T4")))
    lab = InputLabSource(tmp_path / "isolated", RUN)
    # Force expiration before running the worker. We don't wait 120 seconds.
    lab._started_at = time.monotonic() - INITIAL_HOST_CREDIT_TIMEOUT_S - 1
    worker = lab.start()
    worker.join(timeout=3)
    assert not worker.is_alive()
    status = json.loads((lab.root / "status.json").read_text())
    assert status["state"] == "stopped"
    assert status["reason"] == "host_credit_lease_expired"
    assert not (lab.root / "frame.bin").exists()
    lab.close()


def test_game_sender_sidecar_exposes_bounded_credit_balance(tmp_path):
    from colabcola.bridge_video_push_runtime import newest_frame_seq
    class Source:
        sequence_buffered = True
        last_renderer_seq = None
        def pull(self, seq):
            self.last_renderer_seq = seq - 1
            return {"kind": "video_access_unit", "pull_seq": seq}

    class Comm:
        def send(self, data):
            pass

    sender = CreditPushSender(
        comm=Comm(), source=Source(), directory=tmp_path, run_id=RUN,
        credits=6, enable_status_sidecar=True)
    sender._write_sidecar("active", force=True)
    path = tmp_path / "push-sender-status.json"
    first = json.loads(path.read_text())
    assert first["state"] == "active"
    assert first["initialized_credits"] is False
    assert first["capacity"] == 6

    (tmp_path / "frame.bin").write_bytes(
        __import__("struct").pack("!4sIIII", b"CCB1", 0, 0, 640, 360))
    sender.grant(6)
    assert newest_frame_seq(tmp_path) == 0
    assert sender.step() is True
    sender._write_sidecar("active", force=True)
    second = json.loads(path.read_text())
    assert second["sent_count"] == 1
    assert second["replenished_count"] == 0
    assert second["available_credits"] == 5
    assert second["latest_renderer_seq"] == 0
    sender.close()


def test_sender_failure_durably_records_phase_without_exception_text(tmp_path):
    class Source:
        sequence_buffered = True
        last_renderer_seq = None
        def pull(self, seq):
            self.last_renderer_seq = seq - 1
            return {"kind": "video_access_unit", "pull_seq": seq}
    class BadComm:
        def send(self, data):
            raise RuntimeError("must not be written to sidecar: private secret")
    sender = CreditPushSender(
        comm=BadComm(), source=Source(), directory=tmp_path, run_id=RUN,
        credits=6, enable_status_sidecar=True)
    (tmp_path / "frame.bin").write_bytes(
        __import__("struct").pack("!4sIIII", b"CCB1", 0, 0, 640, 360))
    sender.grant(6)
    assert sender.step() is False
    data = json.loads((tmp_path / "push-sender-status.json").read_text())
    assert data["state"] == "failed"
    assert data["failed"] is True
    assert data["error_phase"] == "comm_send"
    assert data["sent_count"] == 1
    assert "private secret" not in json.dumps(data)
    sender.close()


def test_sender_sidecar_is_opt_in_and_never_changes_legacy(tmp_path):
    sender = CreditPushSender(
        comm=SimpleNamespace(send=lambda _data: None),
        source=SimpleNamespace(sequence_buffered=True),
        directory=tmp_path, run_id=RUN, credits=4)
    sender._write_sidecar("active", force=True)
    assert not (tmp_path / "push-sender-status.json").exists()
    sender.close()


def _host_peer(provider):
    peer = KernelCommBridge(
        provider, SimpleNamespace(), RUN, fps=30,
        video_codec="h264", sequence_buffered=True,
        video_push=True, push_credits=6, application="inputlab")
    peer.external_attachment = True
    return peer


def test_close_uploads_only_exact_run_stop_marker_via_authenticated_provider():
    uploads = []
    provider = SimpleNamespace(
        _upload=lambda session, path, data: uploads.append((path, data)))
    peer = _host_peer(provider)
    peer.close()
    assert peer.remote_stop_marker_uploaded is True
    assert uploads == [(f"content/colabcola-bridge/{RUN}/stop", b"1")]
    peer.close()
    assert len(uploads) == 1


def test_close_does_not_upload_for_other_applications():
    uploads = []
    provider = SimpleNamespace(
        _upload=lambda session, path, data: uploads.append(path))
    peer = KernelCommBridge(
        provider, SimpleNamespace(), RUN, fps=30,
        video_codec="h264", sequence_buffered=True, video_push=True,
        push_credits=6, application="freedoom")
    peer.external_attachment = True
    peer.close()
    assert uploads == []


def test_unconfirmed_stop_marker_is_failure_requiring_manual_review():
    def fail_upload(session, path, data):
        raise OSError("private runtime token must not leak")
    peer = _host_peer(SimpleNamespace(_upload=fail_upload))
    with pytest.raises(LifecycleError, match="stop marker upload unconfirmed"):
        peer.close()
    assert peer._closed is True
    assert peer.remote_stop_marker_uploaded is False


def test_non_external_inputlab_peer_never_uploads_stop_marker():
    peer = KernelCommBridge(
        SimpleNamespace(), SimpleNamespace(), RUN, fps=30,
        video_codec="h264", sequence_buffered=True,
        video_push=True, push_credits=6, application="inputlab")
    peer.close()
    assert peer.remote_stop_marker_uploaded is False
