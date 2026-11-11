# v33n — experimental managed 30/60 Hz H.264 capacity sweep

**Status:** managed integration candidate; allocation-free tests must pass in the user's complete Windows checkout before any live T4 experiment. This mode is opt-in. It does not change the normal 12 FPS PPM or H.264 sessions.

`--benchmark-h264-capacity` requires authenticated `--transport comm --video-codec h264 --inflight 1`, `--fps` and `--poll-hz` equal to **30** or **60**, and cannot combine with any other benchmark or link probe. Phase duration is 6–30 seconds, default 10. The run budget must include seven phases and at least 120 seconds of bootstrap/cleanup margin. The selected worker exercises credit caps **1, 2, 3, 4, 8, 16, 32**, with one remote encoder, one kernel Comm, and one local decoder. Requests are never replayed; all outstanding replies drain between phases. Each phase excludes 2 seconds of warmup and boundary drains from its scored interval.

The kernel source and CUDA renderer startup validation were expanded to 60 FPS by v33m, but **no performance claim** has been established at 30 or 60 FPS. The actual CUDA renderer and CPU x264 encoder may run below the configured rate. The H.264 source re-encodes the latest published RGB frame for each video pull, even when that frame has not changed: decoded access-unit FPS **is not** unique GPU frame FPS. We must measure distinct `renderer_seq`, remote CPU encode time, frame freshness, and real input-to-display latency before claiming smooth interactive 30/60 FPS.

High credit windows can increase pending work and frame age. Any transport timeout, protocol mismatch, dependency failure or viewer error must terminate the experiment and release only the owned T4 allocation. Never launch a second allocation until the first run journal is `phase: released`. Keep a locally recorded log and review results after a first 30 Hz run **before** attempting 60 Hz.

Preflight (no allocation):

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m colabcola --profile pypi-new-account-test bridge colab `
    --transport comm --video-codec h264 --install-remote-pyav `
    --inflight 1 --fps 30 --poll-hz 30 `
    --benchmark-h264-capacity --h264-capacity-phase-seconds 10 `
    --max-seconds 300 --preflight-only
```

Live T4 allocation, when explicitly approved separately, needs a *fresh* `--run-dir`, `--confirm-allocation`, and owned release verification; the `--install-remote-pyav` flag remains a distinct opt-in. Never use this experiment mode as a replacement for the previously verified two-credit viewer until measured and reviewed.
