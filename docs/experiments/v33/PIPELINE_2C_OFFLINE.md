# v33i: two-credit H.264 Comm receive window (offline checkpoint)

**Status: offline host-side experiment only; not wired into `bridge colab` CLI.**
The v33g remote H.264 T4 experiment achieved ~5.8 decoded FPS, 156–172 ms
single-credit pull RTT, ~24 KiB/s compressed H.264 throughput and confirmed
owned-runtime release. The v33h patch recorded that result and added fine-grain
payload/timing instrumentation. v33i prepares a safe two-credit host receiver
without altering either established single-credit H.264 or PPM streaming.

`bridge_video_comm_pipeline.py` introduces a `KernelH264PullWindow` that sends
strictly consecutive pull IDs through an existing authenticated owned Jupyter
Comm. The maximum of **two** includes both pending and received-but-blocked
replies. Sends are paced without catch-up bursts. The remote kernel currently
handles each `video_pull` synchronously; two requests may overlap *round trips*
but do not guarantee 12 FPS or imply parallel remote encoding.

IOPub replies may arrive out of order. The window correlates each with an
outstanding pull ID, buffers it until all earlier requests are complete, then
advances the H.264 dependency gate and input ACK state in ordered-pull order.
A duplicate, unsolicited, malformed, stale, disconnected or timed-out request
fails closed. It never replays an ambiguous request. A missing sequence causes
non-IDR frames to be suppressed until the next IDR. The existing Colab owner
remains responsible for closing the Comm and releasing precisely its owned T4.

`stream_comm_h264_pipeline` is a separate optional host worker, not selected by
the CLI yet. Its metrics distinguish compressed H.264 bytes, CCV1-framed bytes,
and Base64 character count (none are full WebSocket bytes); local decode uses
`perf_counter_ns`, and no remote-to-host monotonic clock subtraction is done.

## Tests (no T4 allocation)

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m pytest -q -rs tests/test_bridge_video_comm_pipeline.py `
    tests/test_bridge_video_comm_session.py `
    tests/test_bridge_video_wire_accounting.py `
    tests/test_bridge_video_telemetry.py
```

Follow with the full discovered `tests/test_bridge*.py` regression. Only after
both pass should a later, separately reviewed patch expose a `--inflight 2`
H.264 option, with explicit T4 authorization and a new run directory.

## Not established

No managed T4 has been run with the two-credit H.264 window. No Colab FPS,
network throughput, input-to-photon delay, per-frame deadline guarantee,
video corruption recovery, or performance improvement claim is made here.
