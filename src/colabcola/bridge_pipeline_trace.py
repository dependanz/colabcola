"""v38A.1 bounded, opt-in host diagnostic trace; no hot-path file writes."""
from __future__ import annotations
from collections import deque
from itertools import chain
import json
from pathlib import Path
import re
import threading
import time

TRACE_MAX_RECORDS = 65_536


def validated_credit_snapshot(payload, capacity, video_seq):
    """Validate optional remote credit counters without affecting media acceptance."""
    if not isinstance(payload, dict):
        return None
    available = payload.get("sender_available_credits")
    sent = payload.get("sender_sent_count")
    replenished = payload.get("sender_replenished_count")
    if (any(type(v) is not int for v in (available, sent, replenished))
            or not 0 <= available <= capacity
            or not 0 <= replenished <= sent == video_seq
            or available + sent - replenished != capacity):
        return None
    return {"available": available, "sent": sent, "replenished": replenished}


class PipelineTrace:
    """Single host worker owns the bounded tail; flush on loop exit."""

    def __init__(self, path, *, limit=TRACE_MAX_RECORDS, run_id=None):
        if type(limit) is not int or not 1 <= limit <= TRACE_MAX_RECORDS:
            raise ValueError("invalid pipeline trace limit")
        if run_id is not None and (not isinstance(run_id, str)
                                   or re.fullmatch(r"[0-9a-f]{32}", run_id) is None):
            raise ValueError("invalid pipeline trace run identity")
        self.run_id = run_id
        self.path = Path(path)
        self.records = deque(maxlen=limit)
        self.total = 0
        self.started_ns = time.perf_counter_ns()

    def add(self, event, **fields):
        self.total += 1
        self.records.append({"event": event, "record_seq": self.total, **fields})

    def finish(self, *, outcome, phase, decoded_frames, last_video_seq, last_input_ack):
        # The caller supplies only allowlisted numeric/structured telemetry.
        # No raw errors, credentials, endpoints, event contents or media bytes.
        header = {"schema": 1, "event": "header", "host_clock": "perf_counter_ns",
                  "remote_clock": "remote durations only", "started_ns": self.started_ns,
                  "limit": self.records.maxlen}
        if self.run_id is not None:
            header["run_id"] = self.run_id
        terminal = {"event": "terminal", "outcome": outcome, "phase": phase,
                    "host_finished_ns": time.perf_counter_ns(), "decoded_frames": decoded_frames,
                    "last_video_seq": last_video_seq, "last_input_ack": last_input_ack,
                    "total_records": self.total, "retained_records": len(self.records),
                    "dropped_oldest_records": self.total - len(self.records)}
        try:
            with self.path.open("x", encoding="utf-8", newline="\n") as stream:
                for record in chain((header,), self.records, (terminal,)):
                    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        except (OSError, ValueError, TypeError):
            print("[colabcola] pipeline trace could not be fully written", flush=True)


class HostEventTrace:
    """Independent bounded host timeline, safe for GUI and worker callbacks.

    Internal call sites supply fixed event names and numeric metadata only.
    Closing detaches the bounded tail before doing any file I/O.
    """

    def __init__(self, path, run_id, component, *, limit=TRACE_MAX_RECORDS):
        if (not isinstance(run_id, str)
                or re.fullmatch(r"[0-9a-f]{32}", run_id) is None
                or component not in ("websocket-ingress", "tk-viewer")
                or type(limit) is not int or not 1 <= limit <= TRACE_MAX_RECORDS):
            raise ValueError("invalid host event trace configuration")
        self.path = Path(path)
        self.run_id = run_id
        self.component = component
        self.limit = limit
        self._records = deque(maxlen=limit)
        self._total = 0
        self._lock = threading.Lock()
        self.started_ns = time.perf_counter_ns()

    def add(self, event, **fields):
        with self._lock:
            if self._records is None:
                return
            self._total += 1
            self._records.append({"event": event, "record_seq": self._total,
                                  "host_ns": time.perf_counter_ns(), **fields})

    def finish(self, *, outcome, **fields):
        with self._lock:
            if self._records is None:
                return False
            records, self._records = self._records, None
            total = self._total
            finished_ns = time.perf_counter_ns()
        header = {"schema": 1, "event": "header", "run_id": self.run_id,
                  "component": self.component, "clock": "host_perf_counter_ns",
                  "started_ns": self.started_ns, "limit": self.limit}
        terminal = {"event": "terminal", "outcome": outcome,
                    "host_ns": finished_ns, "total_records": total,
                    "retained_records": len(records),
                    "dropped_oldest_records": total-len(records), **fields}
        try:
            with self.path.open("x", encoding="utf-8", newline="\n") as stream:
                for record in chain((header,), records, (terminal,)):
                    stream.write(json.dumps(record, sort_keys=True, allow_nan=False)+"\n")
        except (OSError, TypeError, ValueError):
            return False
        return True
