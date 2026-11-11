# v33n: managed T4 30 FPS H.264 credit-capacity sweep (live result)

**Evidence:** Operator-provided Windows PowerShell transcript and console log, November 10, 2026. This report records observed application telemetry, not a provider trace or independent bandwidth measurement.

**Run:** `C:\Users\Danze\colabcola-bridge-runs\v33n-30fps-capacity-20261110-193905`

## Configuration

- One owned managed Tesla T4, authenticated kernel Jupyter Comm, 640x360 CUDA latest-frame renderer, remote CPU PyAV/libx264 and local CPU decoder/Tk viewer.
- `--fps 30 --poll-hz 30 --benchmark-h264-capacity --h264-capacity-phase-seconds 10 --max-seconds 300`.
- Explicit `--video-codec h264 --install-remote-pyav --inflight 1 --gpu T4 --confirm-allocation`.
- One kernel connection, persistent encoder and decoder, seven phases with credit caps 1, 2, 3, 4, 8, 16, 32. Each phase measured 8 seconds after a 2-second warmup; phase transitions drained outstanding replies.

## Observed phase results

| Configured credits | Decoded frames | Decoded FPS | Pull RTT p50 (ms) | Input ACK p50 (ms) | H.264 KiB/s | Max observed outstanding | Longest output gap (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 57 | 7.125 | 140 | 187 | 23.137 | 1 | 172 |
| 2 | 112 | 14.000 | 141 | 188 | 37.754 | 2 | 125 |
| 3 | 151 | 18.875 | 140 | 187 | 44.511 | 3 | 140 |
| 4 | 154 | 19.250 | 140 | 188 | 45.126 | 4 | 109 |
| 8 | 156 | 19.500 | 140 | 187 | 46.756 | 4 | 109 |
| 16 | 157 | 19.625 | 141 | 188 | 44.618 | 4 | 125 |
| 32 | 159 | 19.875 | 141 | 188 | 46.094 | 6 | 218 |

Additional application throughput (CCV1 / Base64, KiB/s):

| Credits | CCV1 KiB/s | Base64 KiB/s | Local decode+RGB p50 (µs) |
|---:|---:|---:|---:|
| 1 | 23.345 | 31.137 | 739.70 |
| 2 | 38.164 | 50.906 | 767.40 |
| 3 | 45.064 | 60.107 | 811.70 |
| 4 | 45.690 | 60.945 | 702.65 |
| 8 | 47.327 | 63.130 | 645.80 |
| 16 | 45.193 | 60.278 | 634.00 |
| 32 | 46.677 | 62.259 | 572.20 |

## Interpretation

- Throughput rose from **7.125 FPS (1 credit)** to **14.000 (2)** and **18.875 (3)**; this supports a strong overlapping-request benefit in the low-credit regime.
- It then flattened: **19.250 FPS at 4**, **19.500 at 8**, **19.625 at 16**, and **19.875 at 32**, only about **3.25% improvement from 4 to 32**. Best observed decoded rate was **66.25% of the 30 FPS target**. The first four credits carried most of the benefit.
- Pull RTT p50 was **140-141 ms throughout**, so faster individual replies did not explain throughput scaling. Request-window size alone did not remove the apparent ~20 FPS throughput ceiling; the responsible bottleneck was **not isolated**.
- The actual outstanding count reached at most **4** in the 8- and 16-credit phases and **6** in the 32-credit phase; large configured windows were not fully utilized. This is consistent with request pacing, service capacity and scheduling interactions, but does not identify a particular implementation defect.
- Input ACK p50 was **187-188 ms** across phases. This is observation-of-ACK delay, **not input-to-photon latency**. The median host decode+RGB cost was **0.57-0.81 ms**, and Tk configuration commonly ran around 19-20 FPS in the high-credit phases. The log does not separately isolate server encoding time, host scheduler pacing or GUI presentation timing.
- Longest no-new-decoded-frame gap at 32 credits was **218 ms**, versus **109 ms** at 4/8 credits. These are per-phase maxima from short runs, not robust jitter distributions. Higher window caps may have freshness costs worth instrumenting.
- The H.264 source may encode **the same latest CUDA-rendered snapshot more than once**. This experiment establishes **decoded access-unit FPS**, not **unique renderer FPS**, scene freshness, or actual display refresh FPS.

## Completion and safety

- `request_count`: **1186** over the whole sweep (including warmup and drain), `maximum_window_outstanding`: **7** overall.
- All seven measured phases reported **zero suppressed dependent H.264 units**.
- The final report states `all_phase_boundaries_drained: true` and `continuous_decoder_and_sequence_gate: true`.
- The CLI exited with code **0**; operator's journal check reported `phase: released`, and the runtime release was confirmed.
- These statements are limited to this one owned allocation; later runs require independent cleanup checks.

## Next experimental step

Do **not** infer that a 60 FPS request will produce 60 unique GPU frames from this trial. First instrument: (i) distinct renderer sequence IDs delivered to host, (ii) remote per-pull decompression/encoding time and serialized kernel handler service, (iii) host send/poll/commit scheduling and queue occupancy, and (iv) host frame presentation/freshness using same-clock intervals only. Compare single-credit versus four-credit behavior before widening the concurrency window further. Preserve the verified two-credit ordinary path as baseline. Avoid subtracting remote monotonic timestamps from host monotonic timestamps.

Data source: user-provided `v33n-30fps-capacity-20261110-193905.log`; values above are transcribed from the seven printed phase JSON objects.
