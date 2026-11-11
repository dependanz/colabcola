# v33p — 30 Hz H.264 receive-pacing T4 result

**Evidence:** supplied Windows terminal transcript and `v33p-30fps-pacing-20261111-081451.log`, November 11, 2026. This report transcribes operator observations; it is not independently collected Colab telemetry.
**Source checkpoint:** `030458a` (local user-reported commit, `perf(bridge): improve high-rate H264 receive pacing`).
**Run:** `C:\Users\Danze\colabcola-bridge-runs\v33p-30fps-pacing-20261111-081451`.

Owned managed Tesla T4, authenticated Jupyter Comm, remote CPU H.264 via PyAV/libx264 with explicit remote provisioning, `--fps 30 --poll-hz 30`, seven credit caps 1,2,3,4,8,16,32. Each phase lasted 10 s, with 2 s unscored warmup. One continuous encoder, decoder and request stream; phase boundary drains. The terminal output confirms **exit code 0 and owned T4 release**, no dependent-frame skips and no missing remote timing fields.

## Observations and comparison

| Credits | v33o decoded FPS | v33p decoded FPS | Cross-run delta | v33p distinct renderer IDs/s | v33p submit p50 ms | v33p RTT p50 ms | v33p receive poll p50 ms | v33p max outstanding | v33p duplicate renderer IDs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7.125 | 8.500 | +19.3% | 8.500 | 125 | 125 | 15.0 | 1 | 0 |
| 2 | 14.125 | 16.750 | +18.6% | 16.500 | 62 | 110 | 0.0 | 2 | 2 |
| 3 | 18.250 | 21.250 | +16.4% | 20.750 | 47 | 110 | 0.0 | 3 | 4 |
| 4 | 18.500 | 21.125 | +14.2% | 21.125 | 47 | 110 | 0.0 | 4 | 0 |
| 8 | 18.625 | 21.000 | +12.8% | 19.500 | 47 | 125 | 0.0 | 5 | 12 |
| 16 | 18.250 | 21.250 | +16.4% | 21.125 | 47 | 110 | 0.0 | 4 | 1 |
| 32 | 18.500 | 21.250 | +14.9% | 20.875 | 47 | 110 | 0.0 | 3 | 3 |

The v33p run reported **1,316 requested video pulls in total**, including unscored portions, **no suppressed dependent frames**, `maximum_window_outstanding=5`, and all phase boundaries drained. At credits 3, 4, 16 and 32, decoded rates were roughly 21.1–21.25 FPS; this is below 30 FPS. Distinct renderer IDs appearing in replies reached 21.125/s, but this is **not** independent CUDA renderer production or actual monitor scanout rate.

From v33o to v33p, decoded frame rates improved by about 12.8–19.3% across all seven credit limits. **Because the trials used separate T4 allocations, this is not a controlled estimate of the receive-wait patch's causal effect.**

## Most diagnostic facts

- For 3–32 credits, host pull-submit interval median remained **47 ms**, versus a nominal 33.33 ms at 30 Hz. This is very close to the reciprocal of the observed ~21 FPS plateau.
- v33p's shorter receive-poll timeout produced a reported **0.0 ms median receive-poll elapsed** for windows of 2 or more credits (and 15.0 ms at one credit). This median alone does not show the tails or CPU overhead: the capacity run counted roughly 1,100–1,320 receive polls per 8 s phase, many without newly committed frames.
- For 3–32 credits, video pull RTT medians were about 110–125 ms. Input ACK observation medians were ~156–172 ms, not input-to-photon delays.
- Remote source read/zlib decode was roughly 3.5–4.3 ms p50; remote CPU H.264 encoding was roughly 2.8–3.7 ms p50, and host decoding plus RGB conversion generally <1 ms. These separate duration medians do not determine the end-to-end critical path.
- The 4-credit phase had 169 decoded frames and 169 distinct renderer IDs in the scored 8 s window. At 8 credits, there were 12 duplicate renderer IDs (168 decoded frames, 156 distinct IDs). Higher window sizes did not remove the saturation.

## Interpretation and next experiment

The reduced receive timeout is **not sufficient** to reach the nominal 30 FPS request rate: the median submission interval remains 47 ms. The new polling median is often rounded to 0.0 ms, but this does not mean the receive path consumes no time. Plausible remaining causes include Windows scheduling granularity, kernel/Jupyter queue behavior, host loop interactions, or renderer/kernel service cadence. The log alone does not distinguish these.

Before another owned GPU trial, run a **local standard-library timing probe** measuring Windows monotonic timing, sub-5 ms sleeps, queue receive waits, and synthetic 30/60 Hz loops; use that result to choose an instrumentation or scheduler correction. The probe does not emulate Colab proxy latency, and a successful 30 Hz synthetic loop cannot prove the full H.264 path can reach 30 FPS.

Preserve the validated 12 FPS production options, strict request IDs/credit bounds, decoder ordering, and owned allocation cleanup. Do not infer end-to-end frame age or input-to-photon latency from cross-machine clocks.
