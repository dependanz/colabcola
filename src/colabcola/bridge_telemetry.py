"""Request-level observability for interactive compute connections.

Transport timings are measured on the local machine and deliberately omit
provider tokens, endpoints, file paths, user inputs and pixel data.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import median


class BridgeTransferMeter:
    """Keep only window-local samples; correct skipped/repeated ACK accounting."""

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._sent: dict[int, float] = {}
        self._acked = 0

    def record(self, stage: str, seconds: float) -> None:
        if stage not in ("get", "put", "decode"):
            raise ValueError("unknown bridge timing stage")
        if not 0 <= seconds < 3600:
            raise ValueError("invalid bridge transfer duration")
        self._samples[stage].append(seconds * 1000)

    def sent(self, sequence: int, started_at: float) -> None:
        """Record input timestamp BEFORE the remote upload, not afterward."""
        if sequence <= self._acked or sequence in self._sent:
            raise ValueError("repeated or already acknowledged input sequence")
        self._sent[sequence] = started_at

    def received_ack(self, sequence: int, observed_at: float) -> float | None:
        """Return ack-observation time only when a NEW, known input is acknowledged.

        If a remote worker skips intermediate input events and applies the
        latest, their timestamps must not accumulate indefinitely.
        """
        if sequence <= self._acked:
            return None
        relevant = [n for n in self._sent if n <= sequence]
        if not relevant:
            return None
        newest = max(relevant)
        elapsed = max(0.0, observed_at - self._sent[newest]) * 1000
        self._acked = sequence
        for n in relevant:
            del self._sent[n]
        self._samples["ack"].append(elapsed)
        return elapsed

    def flush(self) -> dict[str, float | None]:
        """Median request timings and ACK latency since the previous flush.

        Return None for metrics with no new samples; never carry a stale input
        latency into the next reporting window.
        """
        result = {
            key: (round(median(self._samples[key])) if self._samples[key] else None)
            for key in ("get", "put", "decode", "ack")
        }
        self._samples.clear()
        return result
