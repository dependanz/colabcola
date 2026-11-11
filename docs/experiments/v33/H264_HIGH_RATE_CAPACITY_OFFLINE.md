# v33m — offline 30/60 FPS H.264 capacity sweep

**Status:** experimental host/remote building blocks only; no new managed CLI selection and no new T4 allocation. The previous `--benchmark-h264-credits` 1→2→1 / 12 FPS experiment and ordinary 1/2-credit streaming are unchanged.

## Intent

Test whether increasing the in-flight video-pull window helps approach 30 and 60 decoded frames/s. The new opt-in `run_h264_capacity_sweep` exercises credits **1, 2, 3, 4, 8, 16, 32** at a *single* configured poll frequency (30 Hz or 60 Hz), on one authenticated Comm with a continuous decoder and strictly increasing pull IDs. Each phase excludes warmup and draining from scored metrics. The hard cap is 32 outstanding requests, counting both pending and out-of-order replies. Uncertain submissions are never retried; protocol errors and 8-second unanswered requests fail closed.

The existing `KernelH264PullWindow(peer)` remains exactly **two-credit only**. Higher limits require constructing the window with `experimental=True`. No ordinary viewer/PPM mode can silently use a higher window.

`RemoteH264PullSource` and the owned CUDA renderer's startup FPS validation expand from 12 to 60 so a later opt-in CLI can configure higher rates. The legacy CLI itself still limits all managed configurations to 12 FPS. The remote rate-control protocol retains its existing 12 FPS cap in this stage; high-rate experiments will configure the renderer at startup rather than dynamically changing its rate.

## Interpretation requirements

A decoded H.264 access unit is produced **for each pull**, even when the underlying CCB1 latest-frame renderer snapshot has not changed. Therefore decoded FPS can exceed distinct CUDA-rendered frames per second. **Do not claim 30/60 unique GPU frames** based on host decoded FPS alone. Existing remote replies include `renderer_seq`; a later instrumentation stage should count distinct renderer sequences, remote encoding time, and host frame age before claiming production-rate feasibility.

The renderer currently does GPU render → CPU readback → CPU zlib compression → atomic file publication, and the kernel H.264 encoder decompresses the file and CPU-encodes it in the synchronous Comm handler. More credits may increase the request pipeline while failing to improve either of these compute stages. Very high credit counts may increase queueing and input/frame staleness. A capacity sweep should be stopped on timeout or high latency; do not automatically run 32 credits against a paid runtime without a separate validation/approval step.

## Validation

This patch is tested offline with a fake owned Comm and synthetic H.264 units, including reverse-order replies at window=32, bound failures, non-opt-in rejection, and virtual-clock 30/60 Hz sweeps. The user's full Windows `tests/test_bridge*.py` regression and a separate managed CLI integration/ownership test are required before any real GPU experiment.
