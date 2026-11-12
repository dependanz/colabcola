# v35C — opt-in zlib level-0 intermediate snapshots at 60 Hz

**Status:** Offline-first controlled optimization. v35B live C10 decoded 5,392 distinct frames / 105.297s = 51.208 FPS and Tk configured 4,794 frames = 45.515 FPS. Independent weighted observed producer rate 51.416 FPS. The producer one-second-window median cost was: CUDA render+CPU readback 1.22ms, zlib compression **11.07ms**, atomic publication 0.76ms. The median across within-window compression p95 values was 18.25ms. Remote credit wait was only 856.37ms observed cumulative time, not an end-to-end credit RTT.

## Exact experimental change

Opt-in flag: **--producer-zlib-level0**. Only accepted for one externally owned, sequence-buffered, H.264 Comm, sender-driven **60 FPS / 60 Hz** session. It replaces the internal operation zlib.compress(ppm, level=1) with zlib.compress(ppm, level=0). It does **not** change Comm H.264 video coding, the 640x360 P6 RGB data, the CCB1 framing, producer ring size, decoder, source freshness, input format, or 60Hz/C8-C10-C12 credit bounds. It adds no network listener and has no impact on 30Hz push or the normal mode, both of which keep zlib level 1.

Independent checks are enforced by the CLI, attach validator, host bridge, remote kernel bootstrap, and renderer. The remote handshake confirms level 0 before the host accepts an opted-in session; journal and host push report identify the compression mode. The bridge never allocates/releases browser-owned Colab runtimes.

Level 0 remains a fully valid zlib stream with uncompressed DEFLATE blocks; the existing bounded source decoder restores identical bytes. At 640x360 RGB, pixel data alone requires 691,200 bytes. The renderer writes the latest and one fixed-ring snapshot for every produced frame, approximately 83 MB/s for 60 FPS before filesystem overhead. That extra I/O can offset the CPU savings. The existing 2MiB compressed-frame bound remains active.

## Validation gates

1. Both Windows/Ubuntu GitHub CI and local Windows focused/full bridge regressions must pass. Verify pinned upstream Colab and kernel provider sources and preserve untracked v32 evidence.
2. Use a fresh disposable, user-owned T4 with the high-credit-dev profile. First run one read-only external preflight; then explicitly confirm a single 120-second level0 run with a fresh run directory. Use the same 60Hz/C10/one-frame lag/640x360 input as v35B. Do not use bridge cleanup for externally owned kernels. Disconnect the notebook manually after the run.
3. Compare covered-window weighted producer FPS, decoded distinct FPS, GUI-configured FPS, stage timing medians and within-window p95s (render/copy, zlib, publish), source-read and H.264 encode p50/p95/p99, credit-wait, previous Comm send duration, GUI frame coalescing and pacing p95/p99, and input callback-to-Tk p50/p95/p99.
4. PowerShell Start-Transcript alone does not capture all native Python subprocess output in the previous sessions. Use an explicit stdout/stderr capture, for example native stderr/stdout redirected through Tee-Object.
5. Comparisons to a different runtime/day/input schedule are observational, not a randomized same-environment A/B.

## Decision

- If producer publication reaches at least 57 FPS and Tk improves, continue testing the level0 path, including I/O and tail-latency regressions.
- If compression improves but file publication and source reads grow enough to offset it, evaluate bounded shared-memory or direct-to-encoder transport rather than increasing video credits.
- If the host decodes ~60 but GUI coalescing remains high, profile Tk consumption and scheduling without allowing unbounded stale frames.
- Do not change C10 to C12 without evidence of credit starvation.

## Future game integration (separate milestone)

Freedoom game assets with a Linux DOOM source port under a virtual display would be a suitable first real application. That integration is **not** present in this commit: current remote CUDA renderer is hard-coded, input has only mouse coordinates and arrow keys, and there is no app capture or general key-down/up injection yet. A future isolated source adapter will need explicit environment/bootstrap, process supervision, bounded RGB capture, keyboard/mouse translation, focus and cleanup, then reuse existing H.264/Comm. Vanilla DOOM is CPU-rendered; treat it as an interactivity test, not a T4 acceleration benchmark.
