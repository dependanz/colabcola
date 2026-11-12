"""v33c: opt-in host-side H.264 adapter for an OWNED Jupyter kernel Comm.

This is a wire-level adapter, not an enabled production video mode. It uses
an already authenticated KernelCommBridge and its single established Comm ID;
it neither opens a socket nor starts a kernel. The remote runtime does NOT
respond to ``video_pull`` yet. Only offline/fake-kernel tests use this class.

Only one video request may be in flight. Timeouts poison this adapter rather
than replaying an uncertain request. The owner must always close the parent
KernelCommBridge and release its allocation according to the normal journal.
"""
from __future__ import annotations

import queue
import time

from .bridge_video_comm_protocol import (
    MAX_PULL_SEQ, VideoCommProtocolError, VideoSequenceGate,
    decode_video_reply,
)
from .lifecycle_io import LifecycleError
from .bridge_video_stream import PACKET_HEADER


class KernelVideoCommReceiver:
    """Strictly correlate CCV1 replies on the existing Jupyter Comm IOPub.

    ``next_video`` returns (VideoAccessUnit | None, input_ack) where None means
    that the frame is dependent on a missing earlier unit and cannot be sent
    to a stateful H.264 decoder. The producer's capture_ns is an independent
    clock and must not be subtracted from host timestamps.
    """

    def __init__(self, peer):
        if (not isinstance(getattr(peer, "run_id", None), str)
                or len(peer.run_id) != 32
                or not isinstance(getattr(peer, "comm_id", None), str)
                or not peer.comm_id
                or getattr(peer, "wire", None) is None):
            raise LifecycleError("kernel video receiver needs attached owned Comm")
        self.peer = peer
        self.pending_pull: int | None = None
        self.last_pull = 0
        self.last_input_ack = 0
        self.gate = VideoSequenceGate()
        self.suppressed_dependent_units = 0
        # (H.264 bytes, complete CCV1 bytes, Base64 characters), latest valid reply.
        # This is application payload accounting, NOT WebSocket/network bytes.
        self.last_reply_sizes: tuple[int, int, int] | None = None
        self.failed = False

    def _healthy(self) -> None:
        wire = self.peer.wire
        if self.failed or wire is None or not wire.connection_ready.is_set():
            self.failed = True
            raise LifecycleError("kernel video Comm unavailable or failed")

    def request_video(self, pull_seq: int) -> None:
        """Submit exactly one numbered pull. Never replay an ambiguous send."""
        self._healthy()
        if (self.pending_pull is not None
                or type(pull_seq) is not int
                or not self.last_pull < pull_seq <= MAX_PULL_SEQ):
            raise LifecycleError("kernel video pull not monotonic or already outstanding")
        # Mark it outstanding BEFORE send; an exception leaves an uncertain
        # submission outcome and this adapter must not retry.
        self.pending_pull = pull_seq
        self.last_reply_sizes = None
        self.last_pull = pull_seq
        try:
            self.peer._send("video_pull", pull_seq=pull_seq)
        except Exception:
            self.failed = True
            raise LifecycleError("kernel video pull submission failed") from None

    def next_video(self, seconds: float):
        """Wait for a single matched response; fail closed on timeout/ambiguity."""
        self._healthy()
        if self.pending_pull is None:
            raise LifecycleError("kernel video response requested without pull")
        if (type(seconds) not in (int, float)
                or isinstance(seconds, bool) or not .01 <= seconds <= 8):
            raise ValueError("kernel video receive timeout outside bounds")
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._healthy()
            try:
                msg = self.peer.wire.iopub_channel.get_msg(
                    timeout=min(.25, max(.01, deadline - time.monotonic())))
            except queue.Empty:
                continue
            except Exception:
                self.failed = True
                raise LifecycleError("kernel video Comm receive failed") from None
            if not isinstance(msg, dict):
                continue
            header = msg.get("header")
            if not isinstance(header, dict) or header.get("msg_type") != "comm_msg":
                continue
            content = msg.get("content")
            if not isinstance(content, dict) or content.get("comm_id") != self.peer.comm_id:
                continue
            payload = content.get("data")
            if not isinstance(payload, dict):
                continue
            if payload.get("v") != 1 or payload.get("run_id") != self.peer.run_id:
                continue
            if payload.get("kind") == "error":
                self.failed = True
                raise LifecycleError("remote video Comm reported an error")
            if payload.get("kind") != "video_access_unit":
                continue  # Other control messages belong to the legacy path.
            try:
                unit, ack = decode_video_reply(
                    payload, run_id=self.peer.run_id,
                    expected_pull_seq=self.pending_pull)
                if ack < self.last_input_ack:
                    raise VideoCommProtocolError("video input acknowledgment regressed")
                keep = self.gate.accept(unit)
            except VideoCommProtocolError:
                self.failed = True
                raise LifecycleError("kernel video Comm response invalid") from None
            self.pending_pull = None
            self.last_input_ack = ack
            self.last_reply_sizes = (
                len(unit.data), PACKET_HEADER.size + len(unit.data), len(payload["data"])
            )
            if not keep:
                self.suppressed_dependent_units += 1
                return None, ack
            return unit, ack
        self.failed = True
        raise LifecycleError("kernel video Comm response timed out")
