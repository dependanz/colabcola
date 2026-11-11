# v33p — experimental high-rate receive-wait pacing

**Status:** allocation-free candidate. Requires full Windows bridge regression and
managed preflight before considering another paid T4 experiment.

## Motivation

The v33o 30 Hz sweep yielded ~18–19 decoded units/s at 3–32 credits, with
host pull-submit p50 **47 ms** despite a nominal **33.3 ms** period. Pull RTT
p50 was ~125–141 ms and remote snapshot-read plus H.264 encode durations
had medians around 7–9 ms combined. This is *consistent with*, not proof of,
a host-side scheduling limitation. Distinct renderer sequence IDs were seen
in only ~17–18.4 received replies/s; the renderer's independent production
rate remains unknown.

The existing worker uses a **30 ms blocking Comm receive** regardless of
its pending next-send deadline. This patch changes **only the experimental
30/60 Hz capacity sweep** to poll the existing authenticated receiver for at
most **4 ms**, further bounded by the next-send deadline (minimum 1 ms).
The established 12 Hz A/B/A sweep, ordinary 1/2-credit H.264, PPM streaming,
remote renderer/encoder, and allocation/cleanup code are not changed.
Outstanding request IDs, in-order H.264 access-unit commits, per-request
8-second fail-closed timeout, and no ambiguous replay remain unchanged.
During a phase transition/drain, use the old 30 ms receive timeout.

The phase telemetry now includes actual host receive-poll elapsed p50, number
of receive polls, and receive polls without a newly ordered commit. An empty
poll can reflect an out-of-order buffered reply; it is not necessarily a
transport timeout. Shorter polls may increase CPU usage; no benefit is claimed
until measured on Windows with a real owned T4.

## Validation

Run `pytest tests/test_bridge_video_v33p_scheduler.py` and the complete
`test_bridge*.py` Windows test suite, then `--preflight-only` at 30 and 60 Hz.
The first live trial, if approved after validation, should be **30 Hz** and
compare the 3/4/8-credit phases to v33o. Do not infer screen scanout or
input-to-photon latency from this telemetry.
