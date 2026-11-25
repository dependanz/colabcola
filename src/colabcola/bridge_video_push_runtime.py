"""v34 experimental sender-driven video flow control inside a user-owned kernel.

Standalone uploaded source, no package dependencies. Background Comm emission
is experimental and requires a live Jupyter kernel trial after offline tests.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import struct
import threading
import time

DEFAULT_PUSH_CREDITS = 4
# Experimental bounds; larger windows may increase queued-video age.
PUSH_CREDIT_CHOICES = (4, 6, 8, 10, 12)
MAX_PUSH_CREDITS = DEFAULT_PUSH_CREDITS  # backward-compatible default
_FRAME_HEADER = struct.Struct("!4sIIII")


def newest_frame_seq(directory):
    """Inspect ONLY the latest atomic CCB1 header; no frame decompression."""
    try:
        with (Path(directory) / "frame.bin").open("rb") as stream:
            header = stream.read(_FRAME_HEADER.size)
        if len(header) != _FRAME_HEADER.size:
            return None
        magic, seq, _ack, width, height = _FRAME_HEADER.unpack(header)
        if magic != b"CCB1" or (width, height) != (640, 360):
            return None
        return seq
    except OSError:
        return None


class CreditPushSender:
    """Strict bounded grants; no replay or per-frame host pull RPC."""

    def __init__(self, *, comm, source, directory, run_id,
                 credits=MAX_PUSH_CREDITS, enable_status_sidecar=False,
                 trace_pipeline=False):
        if (type(credits) is not int or credits not in PUSH_CREDIT_CHOICES
                or not isinstance(run_id, str) or len(run_id) != 32
                or not getattr(source, "sequence_buffered", False)):
            raise ValueError("invalid experimental push sender configuration")
        if type(enable_status_sidecar) is not bool:
            raise ValueError("sender telemetry selector must be boolean")
        self.enable_status_sidecar = enable_status_sidecar
        self.comm = comm
        self.source = source
        self.directory = Path(directory)
        self.run_id = run_id
        self.capacity = credits
        self.available = 0
        self.sent_count = 0
        self.replenished = 0
        self.initialized_credits = False
        self.failed = False
        self.stopped = threading.Event()
        self._condition = threading.Condition()
        self._thread = None
        if (type(trace_pipeline) is not bool or (trace_pipeline and
                getattr(source, "fps", 30) != 30)):
            raise ValueError("pipeline trace requires 30Hz source")
        self.trace_pipeline = trace_pipeline
        self._trace = None
        self._source_misses = 0
        if trace_pipeline:
            trace_class = runpy.run_path(str(Path(__file__).with_name(
                "bridge_remote_trace.py")))["RemotePipelineTrace"]
            self._trace = trace_class(self.directory / "remote-sender-trace.jsonl",
                                      run_id, "credit-sender")
        # Preserve the original default behavior; explicitly opt in at 30Hz.
        self.diagnostics = getattr(source, "fps", 30) == 60 or trace_pipeline
        self._credit_wait_ns = 0
        self._credit_wait_iterations = 0
        self._previous_comm_send_us = None
        self._last_sidecar_write = 0.0
        self._send_error_phase = None

    def _write_sidecar(self, state, *, force=False):
        """Best-effort remote diagnostic only; cannot block video on errors.

        Written at most once per second while active, plus terminal states.
        An unexpectedly stale active sidecar indicates blocked sender work.
        """
        if not self.enable_status_sidecar:
            return
        now = time.monotonic()
        if not force and now - self._last_sidecar_write < 1.0:
            return
        self._last_sidecar_write = now
        with self._condition:
            data = {
                "schema": 1, "run_id": self.run_id, "state": state,
                "sent_count": self.sent_count,
                "replenished_count": self.replenished,
                "available_credits": self.available,
                "capacity": self.capacity,
                "initialized_credits": self.initialized_credits,
                "failed": self.failed,
                "latest_renderer_seq": getattr(self.source,"last_renderer_seq",None),
                "error_phase": self._send_error_phase,
                "monotonic_ns": time.monotonic_ns(),
            }
        target = self.directory / "push-sender-status.json"
        temp = self.directory / "push-sender-status.tmp"
        try:
            temp.write_bytes((json.dumps(data, separators=(",",":"),
                                         allow_nan=False)+"\n").encode())
            os.replace(temp, target)
        except (OSError, TypeError, ValueError):
            # Telemetry is observational. It must never break the sender.
            pass

    def grant(self, count):
        """Grant from authenticated Comm only; never accept unlimited credit."""
        if type(count) is not int or count < 1 or count > self.capacity:
            raise ValueError("invalid video credit grant")
        with self._condition:
            if self.failed or self.stopped.is_set():
                raise ValueError("video sender already closed")
            if not self.initialized_credits:
                if count != self.capacity or self.sent_count:
                    raise ValueError("push initial grant must equal configured window")
                self.initialized_credits = True
                self.available = count
            else:
                # A return credit must acknowledge one previously sent frame.
                # available + (sent_count - replenished) == capacity.
                if count != 1 or self.replenished >= self.sent_count:
                    raise ValueError("unsolicited or repeated push credit")
                self.replenished += 1
                self.available += 1
            if self.available > self.capacity:
                raise ValueError("push credit window overflow")
            if self._trace:
                self._trace.add("credit_granted", count=count, available=self.available,
                                sent=self.sent_count, replenished=self.replenished)
            self._condition.notify_all()

    def step(self):
        """Attempt at most one frame, returning True only on a real send.

        This is deterministic and can be driven entirely by offline fakes.
        """
        with self._condition:
            if self.available <= 0 or self.failed or self.stopped.is_set():
                return False
        latest = newest_frame_seq(self.directory)
        last = getattr(self.source, "last_renderer_seq", None)
        if latest is None or (last is not None and latest <= last):
            if self._trace:
                self._source_misses += 1
            return False
        with self._condition:
            if self.available <= 0 or self.failed or self.stopped.is_set():
                return False
            self.available -= 1
            self.sent_count += 1
            send_seq = self.sent_count
            if self._trace:
                self._trace.add("credit_reserved", video_seq=send_seq,
                                latest_header_seq=latest, available=self.available,
                                sent=self.sent_count, replenished=self.replenished,
                                cumulative_source_misses=self._source_misses)
        phase = "source_pull"
        try:
            if self._trace:
                self._trace.add("source_pull_begin", video_seq=send_seq)
            reply = self.source.pull(send_seq)
            if reply.get("kind") not in ("video_access_unit", "video_no_new"):
                raise ValueError("invalid push video response")
            if self._trace:
                self._trace.add("source_pull_end", video_seq=send_seq,
                                renderer_seq=(getattr(self.source, "last_renderer_seq", None)
                                              if reply["kind"] == "video_access_unit" else None),
                                input_ack=reply.get("input_ack"),
                                has_video=reply["kind"] == "video_access_unit")
            # An encoded frame carries bounded, previous-send diagnostics.
            # We cannot time the current send until after it has completed.
            if self.diagnostics and reply["kind"] == "video_access_unit":
                with self._condition:
                    extras = {
                        "sender_credit_wait_us_total": round(
                            self._credit_wait_ns / 1000, 2),
                        "sender_credit_wait_iterations": self._credit_wait_iterations,
                    }
                    if self._previous_comm_send_us is not None:
                        extras["sender_comm_previous_us"] = self._previous_comm_send_us
                    if self.trace_pipeline:
                        extras.update(sender_available_credits=self.available,
                                      sender_sent_count=self.sent_count,
                                      sender_replenished_count=self.replenished)
                reply = {**reply, **extras}
            phase = "comm_send"
            if self._trace:
                self._trace.add("comm_send_begin", video_seq=send_seq)
            started_ns = time.perf_counter_ns()
            self.comm.send(reply)
            if self.diagnostics:
                elapsed_us = (time.perf_counter_ns() - started_ns) / 1000
                with self._condition:
                    self._previous_comm_send_us = round(elapsed_us, 2)
            if self._trace:
                self._trace.add("comm_send_end", video_seq=send_seq)
            return True
        except Exception:
            with self._condition:
                self._send_error_phase = phase
                self.failed = True
                self.stopped.set()
                self._condition.notify_all()
            if self._trace:
                self._trace.add("send_failed", video_seq=send_seq, phase=phase)
            self._write_sidecar("failed", force=True)
            try:
                self.comm.send({"v": 1, "kind": "error", "run_id": self.run_id,
                                "code": "push_video_unavailable"})
            except Exception:
                pass
            return False

    def _finish_trace(self, outcome):
        if self._trace:
            with self._condition:
                terminal=dict(sent=self.sent_count, replenished=self.replenished,
                              available=self.available, initialized=self.initialized_credits,
                              error_phase=self._send_error_phase,
                              cumulative_source_misses=self._source_misses)
            self._trace.finish(outcome=outcome, **terminal)

    def _worker(self):
        outcome = "interrupted"
        try:
            while not self.stopped.is_set():
                self._write_sidecar("active")
                with self._condition:
                    if self.available == 0 and not self.stopped.is_set():
                        since = (time.perf_counter_ns()
                                 if self.diagnostics and self.initialized_credits else None)
                        wait_start_ns = (time.monotonic_ns() if self._trace and
                                         self.initialized_credits else None)
                        self._condition.wait(timeout=0.05)
                        if since is not None:
                            self._credit_wait_ns += max(
                                0, time.perf_counter_ns() - since)
                            self._credit_wait_iterations += 1
                        if wait_start_ns is not None:
                            self._trace.add("credit_wait", wait_start_ns=wait_start_ns,
                                            available_after=self.available,
                                            sent=self.sent_count, replenished=self.replenished)
                if self.stopped.is_set():
                    break
                if not self.step():
                    self.stopped.wait(0.004)
            outcome = "failed" if self.failed else "stopped"
        finally:
            try:
                self._write_sidecar(
                    "failed" if self.failed else "stopped", force=True)
            finally:
                self._finish_trace(outcome)

    def start(self):
        if self._thread is not None:
            raise ValueError("push sender already started")
        self._thread = threading.Thread(
            target=self._worker, name="colabcola-video-push", daemon=True)
        self._thread.start()

    def close(self):
        with self._condition:
            self.stopped.set()
            self._condition.notify_all()
        # Never block the notebook's Comm callback indefinitely.
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        else:
            self._finish_trace("failed" if self.failed else "stopped")
