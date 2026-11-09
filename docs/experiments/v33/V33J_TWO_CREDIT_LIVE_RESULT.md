# v33j: first managed two-credit H.264 Comm result

**Evidence status:** user-observed Windows PowerShell output, November 9, 2026.
**Run directory:** `C:\Users\Danze\colabcola-bridge-runs\v33j-h264-2credit-20261109-071040`
**Source:** terminal transcript supplied by the operator; no provider-side telemetry file has been separately inspected.

## Configuration

- Owned managed Google Colab Tesla T4; authenticated persistent Jupyter Comm transport
- `--video-codec h264 --install-remote-pyav --inflight 2 --gpu T4`
- `--fps 12 --poll-hz 12 --max-seconds 240 --viewer-stats`
- 640×360 CUDA-rendered latest-frame snapshots encoded through the experimental CPU PyAV/libx264 path
- Explicit `--confirm-allocation`; no public listener or tunnel

## Observed outcome

- Remote GPU verified as Tesla T4; opt-in H.264 Comm pipeline connected with two credits
- 34 interval-level video reports, totaling 723 received replies
- Decoded rate across interval reports: **mean 10.60 FPS**, range **10.0–11.0 FPS**
- GUI configuration: **mean 10.59 FPS** across 33 GUI interval reports
- H.264 payload throughput: **mean 37.59 KiB/s** across interval reports
- CCV1-framed throughput: **mean 37.90 KiB/s**; Base64 application payload: **mean 50.54 KiB/s**
- Video-pull RTT: **78 ms**, median of reported interval p50 values
- Input ACK: **152.5 ms**, median of reported interval p50 values
- Local H.264 decode plus RGB conversion: **697.1 µs**, median of reported interval p50 values
- All displayed video intervals reported **0 dependent-frame skips**
- Exit code **0**, journal `phase: released`, `transport: comm`, `video_codec: h264`

The arithmetic averages above treat reporting intervals equally; they are **not** per-frame weighted distributions. The median-of-medians values are **not** pooled request-level p50 values. App-level Base64 bytes exclude Jupyter/WebSocket and other transport overhead.

## Comparison to earlier v33g single-credit trial

| Indicator | v33g: one credit | v33j: two credits | Approximate observed change |
| --- | ---: | ---: | ---: |
| Decoded video | ~5.8 FPS | 10.60 FPS | +82.8% |
| Pull RTT p50 | ~172 ms | 78 ms | −54.7% |
| Input ACK p50 | ~344 ms | 152.5 ms | −55.7% |
| H.264 payload | ~24 KiB/s | 37.59 KiB/s | +56.6% |
| Dependent-unit skips | 0 reported | 0 reported | No loss seen |

The one-credit and two-credit observations came from **different T4 allocations**. These results are strong operational evidence for the two-credit configuration **in the observed run**, but the cross-run comparison does not identify a causal effect of credits alone. Environment, provider proxy behavior, load and input activity were not rigorously matched.

## Interpretation and next experiment

The two-credit receiver can overlap outstanding Jupyter Comm requests, which plausibly reduces the single-request round-trip throughput constraint. The configured ceiling remains 12 FPS; the measured mean is 10.60 FPS. The synchronous kernel-side encoder, proxy scheduling, message pacing, client delivery and GUI processing have not been individually isolated as residual bottlenecks.

Before expanding to three or four credits, perform an **in-session A/B/A** experiment (1 credit → 2 credits → 1 credit) on the same owned T4, with stabilized phases and comparable interaction patterns. Preserve an explicit cap of two credits in the current live implementation until that experiment is validated. Measure decoded FPS, byte accounting, request RTT, ACK observation, actual Tk configuration, decoder resynchronizations and cleanup. An A/B/A sweep requires new, allocation-free offline implementation and lifecycle tests before authorizing compute.

## Scope and caution

- Rendering is remote CUDA; H.264 encoding is remote CPU and decoding is local CPU.
- Input ACK means the input was observed in a received frame, not monitor scanout or input-to-photon latency.
- Remote and host monotonic clocks are not synchronized; do not compute cross-machine one-way capture age from them.
- A successful exit and released journal demonstrate this allocation's cleanup. Do not infer future sessions are safe without verifying each run's journal.
- The report is a transcription/aggregation of observed console telemetry, not a formal network benchmark or a statistical comparison of independent samples.
