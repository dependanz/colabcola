# v33g live T4 H.264 result and v33h host telemetry

## Observed live run (November 8, 2026)

Source: operator-provided PowerShell transcript, not an automatically archived
raw performance trace. Project branch `feature/interactive-bridge-prototype`.
Run directory: `C:\Users\Danze\colabcola-bridge-runs\v33g-h264-install-20261108-193915`.

Invocation: Colab Cola managed T4, authenticated kernel Comm, opt-in H.264
and opt-in installation of pinned remote PyAV, `--inflight 1 --fps 12
--poll-hz 12 --max-seconds 240 --viewer-stats --confirm-allocation`.

Observed after remote PyAV setup:

- Remote GPU was verified as `Tesla T4`; an H.264 single-credit Comm stream started.
- Approximately 5.6–6.4 decoded FPS per two-second reporting interval;
  approximately 5.8–6.0 typically. Local Tk configuration was about 5.5–6.5 FPS.
- H.264 encoded access-unit data reported approximately 22–25 KiB/s; these
  readings **excluded** CCV1 headers, Base64 expansion, Jupyter JSON and
  WebSocket framing.
- Video pull round-trip median generally 171–172 ms (some 156 ms intervals).
- Input ACK observation medians were about 328–344 ms. This is not
  physical input-to-photon latency.
- No dependent units were suppressed in the reported intervals.
- Previous host decode telemetry printed `0.0 ms` (low display precision and
  timer choice); this is not evidence of zero decoding cost.
- Exit code `0`, journal `phase: released`, `transport: comm`.

A subsequent full Windows test discovery found 33 bridge test modules:
**355 passed in 24.85 seconds**. `git diff --check` succeeded.

The remote scene is a CUDA-rendered demo; CPU-side libx264 encodes snapshots.
This does not yet show arbitrary GPU application streaming or general remote
application virtualization. Provider/network conditions can vary per session.

## v33h allocation-free host instrumentation

The v33h patch preserves single-credit H.264 behavior and the default PPM
transport. It only changes host reporting: local `perf_counter_ns` measures
H.264 decode **plus RGB conversion and local bounded-frame-queue submission**,
reported in microseconds rather than one-decimal milliseconds. Reports distinguish:

- `H.264 KiB/s`: encoded access-unit bytes in all validated replies;
- `CCV1 KiB/s`: the complete CCV1 record (header + access unit);
- `Base64 KiB/s`: ASCII payload length inside the Jupyter Comm reply.

All exclude outer Comm JSON, WebSocket frames, HTTP/proxy transport overhead,
network retransmissions and actual screen presentation. Rate observations are
per reporting interval and cannot be compared as a controlled network A/B
against a past run without matching runtime conditions. Remote capture timestamps
must never be subtracted from host timestamps.

No extra request credits, renderer rate changes, codec or sender policy changes
are introduced. Proceed to H.264 multi-credit pipelining separately only after
local protocol tests establish strictly ordered decoder input and bounded queues.
