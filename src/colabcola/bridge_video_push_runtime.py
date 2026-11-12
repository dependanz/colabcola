"""v34 experimental sender-driven video flow control inside a user-owned kernel.

Standalone uploaded source, no package dependencies. Background Comm emission
is experimental and requires a live Jupyter kernel trial after offline tests.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
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
                 credits=MAX_PUSH_CREDITS, enable_status_sidecar=False):
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
        # v35B optional diagnostics: 30Hz sender payloads remain untouched.
        self.diagnostics = getattr(source, "fps", 30) == 60
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
            return False
        with self._condition:
            if self.available <= 0 or self.failed or self.stopped.is_set():
                return False
            self.available -= 1
            self.sent_count += 1
            send_seq = self.sent_count
        phase = "source_pull"
        try:
            reply = self.source.pull(send_seq)
            if reply.get("kind") not in ("video_access_unit", "video_no_new"):
                raise ValueError("invalid push video response")
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
                reply = {**reply, **extras}
            started_ns = time.perf_counter_ns()
            phase = "comm_send"
            self.comm.send(reply)
            if self.diagnostics:
                elapsed_us = (time.perf_counter_ns() - started_ns) / 1000
                with self._condition:
                    self._previous_comm_send_us = round(elapsed_us, 2)
            return True
        except Exception:
            with self._condition:
                self._send_error_phase = phase
                self.failed = True
                self.stopped.set()
                self._condition.notify_all()
            self._write_sidecar("failed", force=True)
            try:
                self.comm.send({"v": 1, "kind": "error", "run_id": self.run_id,
                                "code": "push_video_unavailable"})
            except Exception:
                pass
            return False

    def _worker(self):
        try:
            while not self.stopped.is_set():
                self._write_sidecar("active")
                with self._condition:
                    if self.available == 0 and not self.stopped.is_set():
                        since = (time.perf_counter_ns()
                                 if self.diagnostics and self.initialized_credits else None)
                        self._condition.wait(timeout=0.05)
                        if since is not None:
                            self._credit_wait_ns += max(
                                0, time.perf_counter_ns() - since)
                            self._credit_wait_iterations += 1
                if self.stopped.is_set():
                    break
                if not self.step():
                    self.stopped.wait(0.004)
        finally:
            self._write_sidecar(
                "failed" if self.failed else "stopped", force=True)

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
