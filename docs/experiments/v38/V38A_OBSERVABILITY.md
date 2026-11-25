# v38A.1 — bounded, opt-in 30 Hz pipeline trace

Base: accepted v37D `a04c8c6a8348a0e6b041c3e275ea701e0bd4aa3c`. Development branch only. No Colab run or release approval is implied.

## Historical evidence

v37D: 23.927 producer FPS (covered windows), 20.947 distinct decoded FPS, 17.460 Tk-configured FPS; 428 input-to-Tk observations, p50/p95/p99 262.43/1441.38/2944.38 ms. During severe stalls, median decode-to-Tk remains near 10–16 ms even when input-to-Tk exceeds one second. Sender timing fields were deliberately disabled at 30 Hz. v37B ended with a freshness watchdog and no normal sender report. Different historical sessions are observational comparisons, not controlled A/B experiments.

## First increment

`--trace-pipeline` explicitly opts in to diagnostics for external 30 Hz buffered H.264 push. The host bootstrap passes and validates the flag. At 30 Hz this adds the previously 60-Hz-only cumulative credit wait and previous Comm send duration, plus a validated sender credit snapshot. These counters describe sender state, not transport RTT. Existing 30 Hz default media payloads and 60 Hz diagnostics remain unchanged.

The host media worker retains a bounded tail of at most 65,536 JSONL records in memory, then writes `pipeline-trace.jsonl` under the fresh run directory on normal or exceptional exit from the loop, including the freshness watchdog. No per-record file I/O occurs in the hot path. The file is created exclusively and never overwrites evidence. Header records clock domains and cap; terminal records outcome, phase, and exact retained/dropped counts. Forced process termination, bootstrap/decoder initialization failure before the loop, and storage failure may leave no complete trace. Failure text is not recorded.

Events: initial credit; local input sequence/callback/send timing (no event contents); validated video sequence, renderer sequence, ACK, host dequeue and validation timestamps and access-unit byte count; distinct producer reporting windows; processing/credit-return timestamps with validated remote encode/source-read/sender durations and credit counters. All host timestamps are same-host perf_counter_ns; no cross-machine subtraction or inferred one-way network delay. Comm send duration is previous send, not delivery time.

## Deliberate limitations and next increment

This is the first slice of v38A, not full pipeline attribution. Host dequeue is not actual WebSocket arrival; no wire-arrival queue timing, remote input receipt/application timing, per-producer-frame cache and stage traces, or frame-correlated Tk presentation traces yet. Remote sender metadata rides media, so a silent remote interval may have no observations until a frame arrives. The bounded tail explicitly reports discarded records. Trace overhead must be quantified with matched workloads. No intentional change to credit policy, frame rate, lag, codecs, input ordering, pointer containment, ownership or runtime lease.

## Acceptance gates

1. Inspect diff and run focused v38A, v35B, v34 and v37 tests; full bridge regressions on Windows and Ubuntu. Confirm default behavior unchanged and opt-in only.
2. Next increment: remote producer/input timeline and host wire/Tk correlation, including bounded failure retention. Predeclare static/dynamic input workloads before paid experiments.
3. Only after offline gates and explicit live authorization: disposable user-owned T4, read-only single-kernel preflight, 640x360 / 30 Hz / C6 / lag 1; manually disconnect after verifying stop state. No bridge cleanup for external sessions. Keep this branch and PR draft; do not merge into dev/main based only on offline tests.
