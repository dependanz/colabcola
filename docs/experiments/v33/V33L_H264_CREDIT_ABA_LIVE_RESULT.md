# v33l — controlled single-runtime H.264 credit A/B/A′: observed T4 result

**Evidence:** operator-supplied Windows PowerShell output, November 10, 2026.
**Repository checkpoint:** `6aa8a8f` (`feat(bridge): integrate managed H264 credit ABA sweep`).
**Run ID in local folder name:** `v33l-h264-aba-20261110-123447`.
**Console log:** `v33l-h264-aba-20261110-123447.log` (operator machine; not bundled in this patch).
**Status:** completed real Colab T4 run, exit code `0`, owned-runtime journal `phase: released`, `transport: comm`, `video_codec: h264`.

## Configuration and methodology

- One managed Tesla T4, one authenticated owned Jupyter Comm connection, persistent remote CPU H.264/PyAV encoder and one host decoder; remote renderer uses CUDA.
- `--video-codec h264 --transport comm --install-remote-pyav --inflight 1 --gpu T4` plus `--benchmark-h264-credits --h264-credit-phase-seconds 16 --fps 12 --poll-hz 12 --max-seconds 180 --viewer-stats --confirm-allocation`.
- The three consecutive phases were A (one credit), B (two credits), A′ (one credit). Each phase lasted 16 seconds, with its first two seconds excluded from scoring; measured denominator 14 seconds per phase.
- All pending and out-of-order buffered replies drained at phase boundaries. Decoder and sequence gate persisted, and request numbering did not restart; 208 requests across entire experiment. The 180 scored replies below exclude warmup and transition/drain portions of the run.
- The same runtime reduces allocation-level confounding, but phases were not randomized or simultaneous; scene evolution, temporal drift and input intensity could still differ.

## Observed per-phase results

| Observed metric | A: 1 credit | B: 2 credits | A′: 1 credit |
|:--|--:|--:|--:|
| Measurement duration (s) | 14 | 14 | 14 |
| Decoded frames | 45 | 90 | 45 |
| Decoded FPS | **3.214** | **6.429** | **3.214** |
| Committed replies | 45 | 90 | 45 |
| H.264 application bytes (KiB/s) | 14.860 | 26.239 | 15.071 |
| CCV1 application bytes (KiB/s) | 14.955 | 26.427 | 15.166 |
| Base64 application payload (KiB/s) | 19.943 | 35.244 | 20.225 |
| Video-pull RTT p50 (ms) | 312 | 312 | 297 |
| Input ACK observed p50 (ms) | 391 | 375 | 390 |
| Host decode + RGB p50 (µs) | 794.10 | 775.15 | 783.70 |
| Longest gap with no new decoded frame (ms) | 375 | 235 | 422 |
| Maximum outstanding requests | 1 | 2 | 1 |
| Dependent units suppressed | 0 | 0 | 0 |

The GUI reported about 3.0–3.5 FPS in A, 6.0–7.0 FPS in B, and 3.0–4.0 FPS in A′. These are separate interval-level GUI reports, not pooled display-timing statistics.

## Interpretation

1. **Window-size throughput effect:** B delivered exactly twice the scored frame count of A and A′ (`90 / 45 = 2.0`). Its decoded FPS was approximately **+100%** relative to either single-credit phase (subject to rounding). The repeated A′ returned to the original A throughput, supporting an effect from request-credit availability *under this session's conditions*.
2. **Per-request latency did not halve:** Pull RTT p50 was 312 ms in both A and B. Concurrent credits allow requests to overlap and increase delivery frequency without requiring an individual request to complete faster. The rough relation `credits / RTT` yields 1 / 0.312 ≈ 3.21 and 2 / 0.312 ≈ 6.41 FPS, close to the observed frame rates; this is a heuristic based on median RTT, not a general throughput identity.
3. **Input ACK did not substantially change:** Observed input ACK p50 was 391 → 375 → 390 ms. The B improvement over A is only ~4%. This is *observation of input acknowledgment in received frames*, not input-to-photon latency.
4. **Frame gap improvement:** B's longest reported no-new-decoded-frame period was 235 ms, versus 375 and 422 ms in the one-credit phases. Zero dependency skips were reported throughout.
5. **Between-allocation variation remains material:** The separate, earlier v33j two-credit T4 trial achieved mean ~10.60 decoded FPS and ~78 ms median-of-interval pull-RTT medians; this v33l B phase achieved 6.429 FPS and 312 ms phase-level pull-RTT p50. Different Colab allocations and measurement formats preclude treating the cross-run difference as a controlled codec or pipeline regression.

## Evidence boundaries and next experiment

- The reported phase p50 values are summaries of individual phase samples; they are not raw RTT distributions or guarantees. Rate figures use phase-level scored totals divided by 14 s; no confidence intervals or repetitions across independent sessions have been established.
- Application H.264, CCV1 and Base64 byte counts **exclude** full Jupyter JSON, WebSocket, proxy/network overhead and retransmissions. Host decode timing does not measure remote encode time or physical display latency. No synchronized remote-to-host one-way frame age was measured.
- The run printed `all_phase_boundaries_drained: true`, `continuous_decoder_and_sequence_gate: true`, and `maximum_window_outstanding: 2`. It exited `0`, then explicitly confirmed release of the owned T4 runtime and journal `phase: released`.
- **Recommended next:** Keep two credits as the current verified live cap. Add stage-specific host-side timing/queue metrics and, where feasible, safe remote timing (relative durations measured on the *same* remote clock) to explain why some allocations exhibit ~78-ms RTT and others ~312 ms. Do not increase credits just because this sweep showed a two-fold throughput improvement; first establish what is saturated and whether increased concurrency worsens freshness or input latency. Preserve the recorded console log separately if archival provenance is required.
