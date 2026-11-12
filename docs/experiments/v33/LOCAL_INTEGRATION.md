# v33b: local H.264 Comm-envelope integration

**Status: local integration proof; no actual Jupyter Comm session, Colab allocation,
network transfer, remote renderer or physical input-to-photon measurement.**

`bridge_video_comm_local.py` integrates the real local PyAV H.264 encoder and
decoder with the v33a Comm envelope. Each generated access unit is wrapped as
`video_access_unit` carrying a bounded `CCV1` record, decoded with strict owned
run-ID and pull-ID correlation, and passed through `VideoSequenceGate` before
reaching the decoder. An input event updates the local scene; subsequent
frames carry source-observed input acknowledgments. The existing Tk viewer
can display the resulting PPM images without modifying the original remote
`CCB1` or default batch transport.

The producer **renders on request** in the same local Python process. It does
not model a continuously publishing Colab worker or Jupyter proxy behavior.
It processes **one request at a time**. No new sockets or background network
listeners exist; `--show` uses the normal local Tk GUI.

## Test and run

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m pytest -q tests/test_bridge_video_comm_local.py tests/test_bridge_video_comm_protocol.py
& $python -m colabcola.bridge_video_comm_local --width 128 --height 72 --fps 12 --frames 24
& $python -m colabcola.bridge_video_comm_local --width 128 --height 72 --fps 12 --frames 24 --drop-source-seqs 1
& $python -m colabcola.bridge_video_comm_local --width 640 --height 360 --fps 12 --frames 120 --show
```

The injected missing reply is a deterministic, local **whole-reply omission**;
it does not simulate transport backpressure, partial H.264 access units,
provider rate limiting or a bandwidth-adaptive sender. After a gap the
receiver rejects dependent non-IDR frames until an IDR arrives.

`capture_to_decoded_p50_ms` is valid **only for this single-machine harness**
whose producer and receiver share the same `perf_counter_ns` clock; do not
apply that subtraction to remote frames. Input callback-to-decode timing is
not input-to-display timing. The Tk `configured` metric is not proof of
monitor scanout.

The patch is additive. Existing PPM Comm runtime, authentication, owned T4
allocation/cleanup, batch runner and sender-control experiments are unchanged.

## Next

v33c needs a separately opt-in remote H.264 producer and real Jupyter Comm
adapter with identity checks, bounded input/reply traffic, accurate IDR-safe
loss handling, decode on the host, and tests for release on every outcome.
Do not allocate a T4 until those offline integration tests pass.
