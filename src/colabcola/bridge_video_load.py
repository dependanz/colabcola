"""v29: delayed receiver reports and measured sender load, on virtual time.

Receiver delivery rate is an observation, not an estimate of unused capacity.
Congested age permits a conservative load-based cut; healthy arrivals permit
a bounded multiplicative probe. Silence cuts carry no invented byte sample.
Sender accounting includes every encoded and framed unit before admission.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math

from .bridge_video_impairment import ScheduledUnit
from .bridge_video_rate import RateConfig, RateReceiver
from .bridge_video_stream import MAX_PACKET, PACKET_HEADER, VideoAccessUnit, packet_bytes


MAX_LOAD_REPORTS = 128


@dataclass(frozen=True)
class LoadControlConfig:
    min_kbps: int = 20
    source_window_ms: float = 300.0
    receiver_window_ms: float = 300.0
    min_delivery_span_ms: float = 200.0
    feedback_freshness_ms: float = 400.0
    safety_fraction: float = 0.8
    decrease_factor: float = 0.5
    increase_factor: float = 1.25

    def __post_init__(self) -> None:
        if type(self.min_kbps) is not int or not 20 <= self.min_kbps <= 20_000:
            raise ValueError("invalid minimum load-control target")
        for name in ("source_window_ms", "receiver_window_ms", "min_delivery_span_ms"):
            value = getattr(self, name)
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not 1 <= value <= 5000):
                raise ValueError(f"invalid {name}")
        if self.min_delivery_span_ms > self.receiver_window_ms:
            raise ValueError("delivery span must fit the receiver window")
        for name, lower, upper, upper_inclusive in (
            ("feedback_freshness_ms", -1, 5000, True),
            ("safety_fraction", 0, 1, True),
            ("decrease_factor", 0, 1, False),
            ("increase_factor", 1, 4, True),
        ):
            value = getattr(self, name)
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or value <= lower
                    or (value > upper if upper_inclusive else value >= upper)):
                raise ValueError(f"invalid {name}")
        if self.feedback_freshness_ms < 0:
            raise ValueError("invalid feedback_freshness_ms")


class LoadRateReceiver(RateReceiver):
    """Separate receiver observations from delayed sender-side rate decisions.

    ``rates.max_switches`` bounds actual target changes here. Receiver reports
    have an independent cap and dwell interval; ignored reports cannot spend
    the source's change budget. No method receives queue or link-profile state.
    """

    def __init__(self, rates: RateConfig, load: LoadControlConfig, *,
                 fps: int, wire_aware: bool = True):
        super().__init__(rates)
        if not isinstance(load, LoadControlConfig):
            raise ValueError("invalid load-control configuration")
        if load.min_kbps >= rates.high_kbps:
            raise ValueError("minimum target must be below the high target")
        if type(fps) is not int or not 1 <= fps <= 60 or type(wire_aware) is not bool:
            raise ValueError("invalid load-control source configuration")
        self.load = load
        self.fps = fps
        self.wire_aware = wire_aware
        self.decisions: list[dict] = []
        self._arrivals: deque[tuple[int, float, int]] = deque(maxlen=120)
        self._source: deque[tuple[int, float, int, int, int]] = deque(maxlen=120)
        self._high_first_seq: int | None = None
        self._low_first_seq: int | None = None
        self._next_silence_report_ms: float | None = None
        self._target_epoch_seq = 0
        self._last_encoded_ms = -1.0
        self._last_encoded_target = rates.high_kbps
        self._sender_clock_ms = 0.0
        self._offered_payload_bytes = 0
        self._offered_wire_bytes = 0

    def observe_encoded(self, packet: VideoAccessUnit, source_ms: float,
                        target_kbps: int) -> None:
        """Observe a prior sender output, including units later rejected by the link."""
        if (not isinstance(packet, VideoAccessUnit)
                or type(source_ms) not in (int, float) or not math.isfinite(source_ms)
                or source_ms < 0 or source_ms <= self._last_encoded_ms
                or packet.seq != self.source_frames_encoded or packet.seq >= 120
                or not math.isclose(source_ms, packet.seq * 1000.0 / self.fps,
                                    rel_tol=0, abs_tol=1e-6)
                or type(target_kbps) is not int
                or not self.load.min_kbps <= target_kbps <= self.settings.high_kbps):
            raise ValueError("noncausal sender load observation")
        wire_bytes = len(packet_bytes(packet))
        payload_bytes = len(packet.data)
        if target_kbps != self._last_encoded_target:
            self._target_epoch_seq = packet.seq
        self._last_encoded_target = target_kbps
        self._last_encoded_ms = source_ms
        self._source.append((packet.seq, source_ms, target_kbps, payload_bytes, wire_bytes))
        self._offered_payload_bytes += payload_bytes
        self._offered_wire_bytes += wire_bytes
        super().observe_encoded(packet, source_ms, target_kbps)

    def _delivery_sample(self) -> dict:
        values = list(self._arrivals)
        first = values[0] if values else None
        last = values[-1] if values else None
        span = last[1] - first[1] if values else 0.0
        wire_bytes = sum(value[2] for value in values[1:])
        reason = ("insufficient_arrivals" if len(values) < 3 else
                  "short_delivery_span" if span < self.load.min_delivery_span_ms else
                  "sequence_gap" if any(b[0] != a[0] + 1
                                        for a, b in zip(values, values[1:])) else None)
        return {
            "delivery_first_seq": first[0] if first else None,
            "delivery_last_seq": last[0] if last else None,
            "delivery_packet_count": len(values),
            "delivery_span_ms": span,
            "delivery_wire_bytes_after_anchor": wire_bytes,
            "observed_delivery_kbps": wire_bytes * 8 / span if reason is None else None,
            "delivery_unavailable_reason": reason,
        }

    def _emit(self, trigger: str, at_ms: float, first_seq: int,
              age_ms: float | None = None) -> bool:
        if (len(self.requests) >= MAX_LOAD_REPORTS
                or at_ms < self.last_request_ms + self.settings.min_switch_ms):
            return False
        sample = self._delivery_sample()
        if trigger == "receiver_silence":
            sample = {**sample, "observed_delivery_kbps": None,
                      "delivery_unavailable_reason": "receiver_silence"}
        request = {
            "request_id": len(self.requests),
            "trigger": trigger,
            "observed_first_seq": first_seq,
            "observed_seq": self.expected_seq - 1,
            "observed_age_ms": age_ms,
            "generated_at_ms": at_ms,
            "reaches_source_ms": at_ms + self.settings.control_delay_ms,
            **sample,
        }
        if trigger == "receiver_silence":
            request["last_arrival_ms"] = self.last_arrival_ms
            request["silence_ms"] = at_ms - self.last_arrival_ms
        self.requests.append(request)
        self.pending.append(request)
        self.last_request_ms = at_ms
        self.low_age_streak = self.high_age_streak = 0
        self._low_first_seq = self._high_first_seq = None
        return True

    def _advance(self, now_ms: float, *, include_deadline: bool) -> None:
        if (type(now_ms) not in (int, float) or not math.isfinite(now_ms)
                or now_ms < self.clock_ms):
            raise ValueError("receiver time must be finite and monotonic")

        def due(deadline: float) -> bool:
            return deadline <= now_ms if include_deadline else deadline < now_ms

        timeout = self.settings.silence_timeout_ms
        if timeout and self.last_arrival_ms >= 0 and not self.watchdog_stopped:
            deadline = self.last_arrival_ms + timeout
            if not self.silence_expired and due(deadline):
                self.silence_expired = True
                self.low_age_streak = self.high_age_streak = 0
                self._low_first_seq = self._high_first_seq = None
                self.silence_events.append({
                    "observed_seq": self.expected_seq - 1,
                    "last_arrival_ms": self.last_arrival_ms,
                    "timeout_at_ms": deadline,
                })
                self._next_silence_report_ms = max(
                    deadline, self.last_request_ms + self.settings.min_switch_ms)
            while (self.silence_expired and self._next_silence_report_ms is not None
                   and due(self._next_silence_report_ms)
                   and len(self.requests) < MAX_LOAD_REPORTS):
                at_ms = self._next_silence_report_ms
                if not self._emit("receiver_silence", at_ms, self.expected_seq - 1):
                    break
                self._next_silence_report_ms = at_ms + max(
                    timeout, self.settings.min_switch_ms)
        self.clock_ms = now_ms

    def observe(self, item: ScheduledUnit) -> None:
        if (not isinstance(item, ScheduledUnit)
                or not isinstance(item.packet, VideoAccessUnit)
                or type(item.arrival_ms) not in (int, float)
                or type(item.source_ms) not in (int, float)
                or not math.isfinite(item.arrival_ms) or not math.isfinite(item.source_ms)
                or item.source_ms < 0 or item.arrival_ms < item.source_ms
                or item.arrival_ms < self.last_arrival_ms or item.arrival_ms < self.clock_ms
                or item.packet.seq < self.expected_seq
                or type(item.wire_size) is not int
                or not PACKET_HEADER.size < item.wire_size <= MAX_PACKET + PACKET_HEADER.size):
            raise ValueError("noncausal encoded-frame observation")
        self._advance(item.arrival_ms, include_deadline=False)
        seq = item.packet.seq
        gap = seq > self.expected_seq
        self.gaps += int(gap)
        self.expected_seq = seq + 1
        self.last_arrival_ms = item.arrival_ms
        self.silence_expired = False
        self._next_silence_report_ms = None
        self.observations += 1
        self._arrivals.append((seq, item.arrival_ms, item.wire_size))
        while self._arrivals and self._arrivals[0][1] < item.arrival_ms - self.load.receiver_window_ms:
            self._arrivals.popleft()
        age = item.arrival_ms - item.source_ms
        if age >= self.settings.congested_age_ms:
            if not self.high_age_streak:
                self._high_first_seq = seq
            self.high_age_streak += 1
        else:
            self.high_age_streak = 0
            self._high_first_seq = None
        if age <= self.settings.recovered_age_ms and not gap:
            if not self.low_age_streak:
                self._low_first_seq = seq
            self.low_age_streak += 1
        else:
            self.low_age_streak = 0
            self._low_first_seq = None
        # Erasure alone is handled by independent IDR recovery, not by a cut.
        if self.high_age_streak >= self.settings.congestion_samples:
            self._emit("packet_age", item.arrival_ms, self._high_first_seq, age)
        elif self.low_age_streak >= self.settings.recovery_samples:
            self._emit("healthy_probe", item.arrival_ms, self._low_first_seq, age)

    def _source_sample(self, now_ms: float, current_kbps: int) -> dict:
        values = [x for x in self._source
                  if x[0] >= self._target_epoch_seq and x[2] == current_kbps
                  and now_ms - self.load.source_window_ms <= x[1] < now_ms]
        duration = len(values) * 1000.0 / self.fps
        payload_bytes = sum(x[3] for x in values)
        framing_bytes = sum(x[4] - x[3] for x in values)
        return {
            "source_sample_first_seq": values[0][0] if values else None,
            "source_sample_last_seq": values[-1][0] if values else None,
            "source_sample_frames": len(values),
            "source_sample_duration_ms": duration,
            "measured_payload_kbps": payload_bytes * 8 / duration if values else None,
            "measured_framing_kbps": framing_bytes * 8 / duration if values else None,
        }

    def apply_due(self, source_seq: int, source_ms: float, current_kbps: int) -> int:
        if (type(source_seq) is not int or not 0 <= source_seq < 120
                or source_seq < self.source_frames_encoded
                or type(source_ms) not in (int, float) or not math.isfinite(source_ms)
                or source_ms < self._sender_clock_ms or source_ms < self.clock_ms
                or source_ms < self._last_encoded_ms
                or type(current_kbps) is not int
                or not self.load.min_kbps <= current_kbps <= self.settings.high_kbps):
            raise ValueError("noncausal sender rate decision")
        self._sender_clock_ms = source_ms
        due = []
        while self.pending and self.pending[0]["reaches_source_ms"] < source_ms - 1e-9:
            due.append(self.pending.popleft())
        if not due:
            return current_kbps
        latest = due[-1]
        sample = self._source_sample(source_ms, current_kbps)
        decision = {
            "source_seq": source_seq,
            "source_ms": source_ms,
            "previous_kbps": current_kbps,
            "target_epoch_seq": self._target_epoch_seq,
            "selected_request_id": latest["request_id"],
            "request_ids_coalesced": [r["request_id"] for r in due],
            "trigger": latest["trigger"],
            "cut_base_kbps": None,
            **sample,
            "observed_delivery_kbps": latest["observed_delivery_kbps"],
            "wire_budget_kbps": None,
            "wire_candidate_kbps": None,
            "wire_cap_used": False,
            "wire_cap_binding": False,
            "fallback_reason": None,
            "min_target_saturated": False,
            "overhead_budget_failure": False,
        }
        candidate = float(current_kbps)
        ignored = False
        if source_ms - latest["generated_at_ms"] > self.load.feedback_freshness_ms:
            decision["fallback_reason"] = "stale_feedback"
            ignored = True
        elif (latest["trigger"] != "receiver_silence"
              and latest["observed_first_seq"] < self._target_epoch_seq):
            decision["fallback_reason"] = "previous_target_epoch"
            ignored = True
        if not ignored and latest["trigger"] == "healthy_probe":
            candidate = math.ceil(current_kbps * self.load.increase_factor)
        elif not ignored:
            base = math.floor(current_kbps * self.load.decrease_factor)
            decision["cut_base_kbps"] = base
            candidate = float(base)
            reason = ("silence_has_no_delivery_sample" if latest["trigger"] == "receiver_silence" else
                      "wire_feedback_disabled" if not self.wire_aware else
                      latest["delivery_unavailable_reason"] if latest["observed_delivery_kbps"] is None else
                      "delivery_window_previous_epoch" if latest["delivery_first_seq"] < self._target_epoch_seq else
                      "insufficient_source_samples" if sample["source_sample_frames"] < 3 else None)
            decision["fallback_reason"] = reason
            if reason is None:
                budget = self.load.safety_fraction * latest["observed_delivery_kbps"]
                framing = sample["measured_framing_kbps"]
                wire_candidate = current_kbps * max(0.0, budget - framing) / sample["measured_payload_kbps"]
                decision.update({
                    "wire_budget_kbps": budget,
                    "wire_candidate_kbps": wire_candidate,
                    "wire_cap_used": True,
                    "overhead_budget_failure": budget <= framing,
                })
                candidate = min(candidate, wire_candidate)

        def clipped(value: float) -> int:
            return max(self.load.min_kbps, min(self.settings.high_kbps, math.floor(value)))

        decision["min_target_saturated"] = candidate < self.load.min_kbps
        target = clipped(candidate)
        exhausted = target != current_kbps and len(self.applied) >= self.settings.max_switches
        if exhausted:
            target = current_kbps
            decision["fallback_reason"] = "rate_change_budget_exhausted"
        decision["chosen_target_kbps"] = target
        decision["wire_cap_binding"] = bool(
            decision["wire_cap_used"] and target < clipped(decision["cut_base_kbps"]))
        decision["action_status"] = ("ignored_report" if ignored else
                                     "budget_exhausted" if exhausted else
                                     "applied" if target != current_kbps else "unchanged_at_bound")
        self.decisions.append(decision)
        if target != current_kbps:
            self.applied.append({
                "source_seq": source_seq,
                "source_ms": source_ms,
                "previous_kbps": current_kbps,
                "target_kbps": target,
                "request_ids_coalesced": decision["request_ids_coalesced"],
                "trigger": latest["trigger"],
            })
            self._target_epoch_seq = source_seq
            self.desired_kbps = target
        return target

    def report(self) -> dict:
        return {
            **super().report(),
            "policy": "wire_load_feedback" if self.wire_aware else "age_only_multirate",
            "load_control": asdict(self.load),
            "rate_change_budget": self.settings.max_switches,
            "feedback_report_budget": MAX_LOAD_REPORTS,
            "source_frames_encoded": self.source_frames_encoded,
            "source_offered_payload_bytes": self._offered_payload_bytes,
            "source_offered_wire_bytes": self._offered_wire_bytes,
            "wire_caps_used": sum(x["wire_cap_used"] for x in self.decisions),
            "wire_caps_binding": sum(x["wire_cap_binding"] for x in self.decisions),
            "decisions": self.decisions,
            "scope": "receiver delivery rate is not unused link capacity; healthy reports permit probes; "
                     "source load includes framing, IDRs and admission-rejected units; "
                     "all clocks and fixed-delay feedback are virtual",
        }
