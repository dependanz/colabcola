"""Standalone v38A.2 remote trace, uploaded only for opted-in sessions.

Each worker owns an independent bounded tail; callbacks can append under a
small lock. No file I/O on the active media path, no raw input or media data.
"""
from __future__ import annotations

from collections import deque
from itertools import chain
import json
from pathlib import Path
import re
import threading
import time

REMOTE_TRACE_LIMIT = 65_536


class RemotePipelineTrace:
    def __init__(self, path, run_id, component, *, limit=REMOTE_TRACE_LIMIT):
        if (not isinstance(run_id, str) or re.fullmatch(r"[0-9a-f]{32}", run_id) is None
                or component not in ("inputlab-producer", "credit-sender")
                or type(limit) is not int or not 1 <= limit <= REMOTE_TRACE_LIMIT):
            raise ValueError("invalid remote trace configuration")
        self.path = Path(path)
        self.run_id = run_id
        self.component = component
        self.limit = limit
        self._records = deque(maxlen=limit)
        self._total = 0
        self._lock = threading.Lock()
        self.started_ns = time.monotonic_ns()

    def add(self, event, **fields):
        # Only internal call sites supply fixed names and numeric/boolean
        # metadata. Never pass an entire input message, frame or exception.
        with self._lock:
            if self._records is None:
                return
            self._total += 1
            self._records.append({"event": event, "record_seq": self._total,
                                  "remote_ns": time.monotonic_ns(), **fields})

    def finish(self, *, outcome, **fields):
        with self._lock:
            if self._records is None:
                return False
            records, self._records = self._records, None
            total = self._total
            finished_ns = time.monotonic_ns()
        header = {"schema": 1, "event": "header", "run_id": self.run_id,
                  "component": self.component, "clock": "remote_monotonic_ns",
                  "started_ns": self.started_ns, "limit": self.limit}
        terminal = {"event": "terminal", "outcome": outcome,
                    "remote_ns": finished_ns, "total_records": total,
                    "retained_records": len(records),
                    "dropped_oldest_records": total-len(records), **fields}
        try:
            # A fresh, exact-run path; never overwrite a previous trace.
            with self.path.open("x", encoding="utf-8", newline="\n") as stream:
                for record in chain((header,), records, (terminal,)):
                    stream.write(json.dumps(record, sort_keys=True, allow_nan=False)+"\n")
        except (OSError, TypeError, ValueError):
            # Absence of a complete terminal record means incomplete evidence.
            # Diagnostic I/O must not change worker cleanup or expose errors.
            return False
        return True
