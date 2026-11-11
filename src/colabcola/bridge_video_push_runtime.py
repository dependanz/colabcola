"""v34 experimental sender-driven video flow control inside a user-owned kernel.

Standalone uploaded source, no package dependencies. Background Comm emission
is experimental and requires a live Jupyter kernel trial after offline tests.
"""
from __future__ import annotations

from pathlib import Path
import struct
import threading
import time

MAX_PUSH_CREDITS = 4
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
    """At most four granted unacknowledged frame sends; no replay or polling RPC."""

    def __init__(self, *, comm, source, directory, run_id, credits=MAX_PUSH_CREDITS):
        if (type(credits) is not int or credits != MAX_PUSH_CREDITS
                or not isinstance(run_id, str) or len(run_id) != 32
                or not getattr(source, "sequence_buffered", False)):
            raise ValueError("invalid experimental push sender configuration")
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

    def grant(self, count):
        """Grant from authenticated Comm only; never accept unlimited credit."""
        if type(count) is not int or count < 1 or count > self.capacity:
            raise ValueError("invalid video credit grant")
        with self._condition:
            if self.failed or self.stopped.is_set():
                raise ValueError("video sender already closed")
            if not self.initialized_credits:
                if count != self.capacity or self.sent_count:
                    raise ValueError("push initial grant must equal four credits")
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
        try:
            reply = self.source.pull(send_seq)
            if reply.get("kind") not in ("video_access_unit", "video_no_new"):
                raise ValueError("invalid push video response")
            # A no-new race remains correlated; the host returns one credit.
            self.comm.send(reply)
            return True
        except Exception:
            with self._condition:
                self.failed = True
                self.stopped.set()
                self._condition.notify_all()
            try:
                self.comm.send({"v": 1, "kind": "error", "run_id": self.run_id,
                                "code": "push_video_unavailable"})
            except Exception:
                pass
            return False

    def _worker(self):
        while not self.stopped.is_set():
            with self._condition:
                if self.available == 0 and not self.stopped.is_set():
                    self._condition.wait(timeout=0.05)
            if self.stopped.is_set():
                break
            if not self.step():
                self.stopped.wait(0.004)

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
