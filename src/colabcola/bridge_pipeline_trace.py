"""v38A.1 bounded, opt-in host diagnostic trace; no hot-path file writes."""
from __future__ import annotations
from collections import deque
from itertools import chain
import json
from pathlib import Path
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

    def __init__(self, path, *, limit=TRACE_MAX_RECORDS):
        if type(limit) is not int or not 1 <= limit <= TRACE_MAX_RECORDS:
            raise ValueError("invalid pipeline trace limit")
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
