# v35A — opt-in 60 Hz sender-driven H.264 and session-pooled latency instrumentation

**Status:** Offline-first implementation on `feature/interactive-bridge-prototype`. NOT yet validated on a live 60 Hz Colab T4. Keep the proven 30 Hz `v34C` six-credit configuration available unchanged.

## Established baseline (v34C)

On one browser-owned T4 in profile `high-credit-dev`, v34C delivered **3,088 distinct H.264-decoded renderer frames in 104.234 s = 29.626 FPS**; independent CUDA producer p50 29.845 FPS (104 sidecar windows), only 3 source-frame sequence skips, median selection lag 0 frames, no `video_no_new` replies. Input callback→host receipt p50 148.65 ms; callback→host decode completion p50 149.64 ms (406 observations); GUI period medians generally 29–30 FPS and input callback→Tk about 150–175 ms. These were not pooled per-event GUI statistics and not hardware input-to-photon latency. Run journal ended `detached`, `ownership=external`, `host_disconnected`. No allocation/release by bridge.

## New experimental configuration

Sender-driven H.264 can now explicitly request `--fps 60 --poll-hz 60` on an existing, **single** browser-provisioned Tesla T4 and single owned kernel only. The host CLI, external attachment validator, host bridge, and remote Jupyter bootstrap independently enforce:
- Push mode requires `--attach-existing --sequence-buffered --push-video --transport comm --video-codec h264 --inflight 1`.
- Push FPS and poll-hz must match and be **30/30 or 60/60**.
- **30 Hz** push credits are 4/6/8 (the 4-credit default is unchanged).
- **60 Hz** push credits are bounded at 8/10/12. Recommended first experiment: **10 credits**, *not* an automatic increase. Capacity 10 covers approximately 167 ms of 60 FPS production in a fully utilized window; the real push-credit RTT and frame queueing remain unknown. More credits are not automatically better for interactivity.
- Legacy external 30 Hz pull capacity benchmark remains unchanged; managed allocation, silent confirmation, and other transports cannot activate 60 Hz push.

The T4 CUDA source and PyAV encoder already accept 60 FPS. H.264 frame pacing uses 60 Hz codec metadata; the eight-slot renderer ring remains fixed and the source-selection lag stays explicitly bounded (recommend one frame). The Windows Tk periodic poll is 5 ms for opt-in 60 Hz push and remains 10 ms for 30 Hz push; neither setting guarantees hardware display scanout every 16.67 ms.

## New trustworthy measurement output

**Host receiver summary:** The experimental push result includes the target FPS, selected credit capacity, *pooled individual frame-arrival intervals* p50/p95/p99 and max, and *pooled individual* callback→receive and callback→decode p50/p95/p99 along with observation counts. Individual input callbacks are counted only on the first renderer frame acknowledging an input. `input_callback_ns` is a host `perf_counter_ns` value, not a remote clock.

**Tk GUI session summary:** On viewer shutdown, `[colabcola] v35 GUI session {JSON}` reports full-session Tk-configured FPS and frames, pooled per-frame GUI spacing p50/p95/p99, host receipt→Tk, host decoding→Tk, and input callback→Tk p50/p95/p99 and sample counts. The existing two-second console telemetry remains unchanged. Only actual configured Tk labels count; dropped or coalesced frames are not falsely reported as presented. The GUI session stats remain available even when reporting-window medians reset.

The percentile estimator sorts the raw event durations and uses linear interpolation between adjacent order statistics at fractional index `(n-1)p`; p50/p95/p99 are finite, non-negative milliseconds rounded to 2 decimals; empty lists yield JSON `null`. This is not a confidence interval and does not prove display scanout. Both renderer and decoder are in different clock domains from the Windows GUI; never subtract their raw monotonic timestamps.

## First live acceptance (only after offline and CI)

0. Verify pinned upstream provider and full `tests/test_bridge*.py` on Windows and Ubuntu, plus local Windows checkout. Confirm branch commit and preserve untracked v32 evidence.
1. Connect exactly one disposable Tesla T4 notebook under the higher-credit development profile, and perform a **read-only** `--preflight-only` with `--fps 60 --poll-hz 60 --push-credits 10 --renderer-lag-frames 1`. No GPU allocation, notebook execution, or code upload is allowed in preflight.
2. Only after preflight, explicitly `--confirm-attach` with a unique run-dir and `--max-seconds 120`. This executes experimental code and may install PyAV in the user-owned notebook.
3. Exercise controlled mouse/arrow input; inspect JSON summaries, frame gaps, upstream producer FPS, H.264 sequence skip count, Tk-configured FPS, and tail latency. Never claim input-to-photon; report comparison to v34C as observational unless matched identical hardware, input schedule, and network path.
4. The existing external lifecycle must not allocate/release or retry. The user must manually disconnect/delete the disposable T4 in the browser. Never call managed `bridge cleanup` against an external journal.

**Do not claim 60 FPS achievement from offline tests.** If the sender cannot match CUDA production, instrument remote credit starvation, `comm.send` blocking, and kernel IOPub queue depth in the next iteration before increasing credits beyond 12.
