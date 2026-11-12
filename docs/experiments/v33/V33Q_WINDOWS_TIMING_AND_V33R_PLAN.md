# v33q Windows timer probe and v33r experimental scheduler fix

Evidence: user-provided Windows host probe (`colabcola-v33q-windows-timing.json`), November 11, 2026, Python 3.12.1, `win32`. Probe involved no Colab, GPU or network.

| Target rate | `time.sleep` pacing | `Queue.get(timeout=...)` pacing | `Event.wait(timeout=...)` pacing |
|---:|---:|---:|---:|
| 30 Hz | 29.681 Hz; interval p50 33.661 ms | 21.456 Hz; p50 46.658 ms | 21.482 Hz; p50 46.606 ms |
| 60 Hz | 58.699 Hz; p50 17.048 ms | 32.244 Hz; p50 31.049 ms | 32.151 Hz; p50 31.067 ms |

The probe reports `time.monotonic()` resolution **15.625 ms**, versus `perf_counter()` resolution **0.0001 ms**. Its short `Queue.get`/`Event.wait` calls typically rounded to approximately 15.5–16 ms even for requested 1–4 ms timeouts. This directly reproduces the observed ~47 ms host submission interval at 30 Hz; it does not prove the sole remaining bottleneck or guarantee higher GPU-rendered FPS in an owned Colab runtime.

## v33r instrumentation-only experimental change

For the **explicit 30/60 Hz capacity sweep only**, use `time.perf_counter_ns()` converted to seconds for *host-local phase scheduling*, scored timings, ACK observation, and freshness accounting. The externally supplied owned-runtime `deadline` and the existing `KernelH264PullWindow` per-request 8 s timeout remain on `time.monotonic()`; no clock domains are subtracted. `poll(0)` performs a truly nonblocking `iopub_channel.get_msg(timeout=0)` read. A short `time.sleep(.001)` yields CPU after empty polls rather than using `Queue.get(timeout=.004)`, which the Windows probe showed may actually block ~15.6 ms. Legacy one-/two-credit video workers and the established 12 Hz A/B/A sweep retain the prior clock and wait behavior; ordering, run/comm correlation, fail-closed errors, no ambiguous retry, and T4 ownership are unchanged.

The phase report adds `host_phase_clock` and `nonblocking_iopub_poll` identifiers. Existing `host_receive_poll_elapsed_p50_ms` measures the nonblocking receive operation, **not** the separate 1 ms host sleep. It must not be interpreted as total loop overhead.

This is an allocation-free candidate, validated in an isolated virtual-clock reconstruction. Require complete Windows `test_bridge*.py` regression and `--preflight-only` at 30 Hz before any further owned T4 allocation. For a live comparison, compare the host pull-submit p50 against the previous ~47 ms and count distinct renderer sequence IDs, request RTT, input ACK, GUI FPS and runtime cleanup. **Do not infer 30 distinct CUDA-rendered FPS or input-to-photon latency from decoded access-unit FPS.**
