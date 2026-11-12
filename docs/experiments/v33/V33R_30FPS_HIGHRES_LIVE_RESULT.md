# v33r — live 30 Hz high-resolution H.264 capacity sweep

**Evidence status:** Operator-provided Windows console transcript and captured managed T4 console log; November 11, 2026. The independently produced renderer metrics file and source machine were not accessed. These are application reports, not a controlled cross-allocation network benchmark.

**Source code checkpoint:** `f3f7e38` (`perf(bridge): use high-resolution nonblocking H264 pacing`) on `feature/interactive-bridge-prototype`, as reported by operator.

**Owned run directory:** `C:\Users\Danze\colabcola-bridge-runs\v33r-30fps-highres-20261111-095112`.

## Configuration and outcome

- Managed Tesla T4, authenticated kernel Comm, experimental H.264, explicit remote pinned PyAV installation permission.
- `--fps 30 --poll-hz 30 --benchmark-h264-capacity --h264-capacity-phase-seconds 10 --max-seconds 300`.
- Seven sequential phases: 1, 2, 3, 4, 8, 16, 32 requested outstanding credits. Two seconds of warmup excluded from each ten-second phase; scored duration eight seconds per phase.
- One owned GPU, one connected Comm, continuous renderer/encoder/decoder and ordered video pull numbering; phase boundaries were drained.
- **Trial exit code 0; journal phase `released`; owned Tesla T4 release explicitly confirmed.**
- **1,822 total requests**, including unscored phase warmup and transition requests; zero reported H.264 dependent-frame suppressions; every scored reply had usable remote diagnostics.

## Scored phase telemetry

| Credit cap | Decoded FPS | Distinct renderer IDs / s in replies | Duplicate renderer replies | Pull-submit p50 (ms) | Pull RTT p50 (ms) | Input ACK p50 (ms) | Max outstanding |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13.125 | 13.125 | 0 | 75.36 | 78.00 | 117.32 | 1 |
| 2 | 24.250 | 22.750 | 12 | 38.01 | 78.00 | 119.55 | 2 |
| 3 | 28.000 | 25.500 | 20 | 34.41 | 78.00 | 122.11 | 3 |
| 4 | 29.000 | 25.500 | 28 | 34.35 | 78.00 | 118.65 | 4 |
| 8 | 29.250 | 26.625 | 21 | 34.30 | 78.00 | 112.62 | 4 |
| 16 | 29.000 | 25.375 | 29 | 34.38 | 78.00 | 121.62 | 7 |
| 32 | 29.000 | 25.500 | 28 | 34.41 | 78.00 | 120.13 | 7 |

| Credit cap | Remote snapshot read p50 (ms) | Remote H.264 encode p50 (ms) | Host decode+RGB p50 (ms) | Longest decoded-frame gap (ms) | Base64 payload KiB/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.051 | 3.469 | 0.772 | 114.26 | 47.846 |
| 2 | 4.280 | 3.688 | 0.782 | 97.36 | 75.337 |
| 3 | 4.864 | 3.817 | 0.774 | 130.61 | 82.479 |
| 4 | 4.125 | 3.343 | 0.683 | 75.57 | 83.055 |
| 8 | 3.772 | 2.927 | 0.586 | 77.13 | 83.637 |
| 16 | 4.324 | 3.528 | 0.573 | 130.71 | 83.659 |
| 32 | 4.278 | 3.519 | 0.629 | 133.83 | 84.776 |

## Interpretation and comparisons

- **Host scheduling correction supported.** At three or more credits, observed median submission intervals are **34.30–34.41 ms**, much closer to the nominal **33.33 ms** 30-Hz period than the **47 ms** observed in the preceding v33p run. The new report confirms `host_phase_clock: perf_counter_ns` and `nonblocking_iopub_poll: true`.
- The highest delivered rate is **29.25 decoded access units/s** at eight credits (234 in eight scored seconds), approximately 97.5% of the configured 30-FPS target. The previous v33p peak was **21.25 decoded units/s** in a different allocation; peak-to-peak improvement is approximately **37.6%**. This comparison is observational, not a randomized causal estimate of the scheduler change alone.
- At eight credits, only **213 of 234 replies** carried distinct `renderer_seq` values: **26.625 distinct renderer IDs observed/s**, with **21/234 = 8.97%** duplicate-snapshot replies. A decoded H.264 unit is not necessarily a newly produced CUDA frame.
- The four-credit phase delivered 29.00 decoded FPS and 25.50 distinct renderer IDs/s. Credit caps above four did not materially increase decoded FPS. At eight credits the measured outstanding maximum was four; for 16/32 it was seven. Credit caps should not be confused with actual concurrency.
- Median pull RTT was **78 ms** across every phase. Median input ACK observation was **112.62–122.11 ms** in phases with three or more credits. This ACK is *not* input-to-photon latency.
- Remote file-read-plus-zlib decode and CPU x264 encoding were measured separately on the remote clock, typically on the order of **3–5 ms** and **3–4 ms** respectively. Their medians do not identify a full causal service-time distribution, nor exclude other remote work, kernel dispatch, or scheduling overhead.
- Host nonblocking receive polls were approximately **5,000 per eight-second scored phase**, with most polls producing no committed access unit. While effective for scheduling, this could have CPU utilization/power implications; no CPU percentage was measured.
- **Distinct snapshot IDs among received replies do not measure total independently produced renderer frames.** A renderer could publish snapshots that the host never sampled, or the host could request more frames than the renderer published. Actual onscreen scanout and absolute frame age were not measured.

## Recommended next investigation

Keep the verified one-/two-credit and 12-FPS paths unchanged. Before claiming 30 genuinely fresh displayed frames/s or moving to a 60-Hz T4 test, measure independent renderer production using the already-generated `render-metrics.json` sidecar (run-ID validated), record renderer `seq` progress and producer FPS by scored phase, and compare with unique renderer IDs in replies. Investigate file publication/CPU compression and frame sampling only if producer-versus-consumer evidence supports them. Validate changes offline and preserve owned-runtime cleanup checks before further allocation.

**Scope:** One 30-Hz seven-phase experiment in one T4 allocation. No network-byte, input-to-photon, refresh scanout, or cross-machine one-way age measurement is claimed. The Windows console printout and log are the underlying evidence.
