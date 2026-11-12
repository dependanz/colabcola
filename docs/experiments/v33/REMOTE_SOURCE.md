# v33d: remote H.264 source and gated kernel-side video_pull

Status: **offline validation only**, not yet wired into the host's managed
Colab video bootstrap/CLI. No GPU has been allocated for v33d.

The existing CUDA scene worker writes atomic, independent CCB1 PPM snapshots.
The optional `RemoteH264PullSource` reads and bounds each CCB1 snapshot,
extracts RGB, and encodes one **new** whole H.264 access unit in a persistent
low-delay PyAV/libx264 encoder **for each permitted `video_pull`**. The
independent `coded_seq` increments on every successful pull, *not* on each
CUDA renderer frame. A `renderer_seq` field documents which CUDA snapshot was
encoded. Because the codec is persistent, the receiver must process units
in order and conservatively wait for IDR after a missing sequence.

A source `capture_ns` timestamp in CCV1 is **when the remote snapshot was
sampled for encoding**, not original CUDA capture time. It is on the **kernel**
clock and cannot be subtracted from the host's clock. The actual applied input
ACK comes from the CCB1 snapshot, not from the kernel Comm handler's input
acceptance counter. The remote encoder is CPU-side PyAV; CUDA still renders
its original scene, but no claim of GPU H.264 encoding is made.

The existing `bridge_colab_comm_runtime.start` gains an optional, trusted
`video_source_path` argument. It is **None by default**. Only when it is
present does the kernel handler accept strictly numbered `video_pull`
requests. Those requests use the preexisting authenticated kernel Comm ID and
run ID, and return the v33a `video_access_unit` envelope. Any error returns
an identity-scoped, content-free `kind=error` response and fails closed.
The legacy `pull` (PPM), input/rate/ping, close, ownership, and cleanup paths
are unchanged. No new socket, tunnel, or remote listener is opened.

`bridge_comm.KernelCommBridge.open` does NOT currently upload or enable the
new source, nor does the CLI expose video mode. Do not call `video_pull` against
an ordinary managed T4 bridge; the server will not answer it. The next step
is a separately opt-in, tested host bootstrap (including PyAV availability)
and decoded-viewer adapter with owned-resource cleanup tests.

### Limits and necessary tests

* Each pull re-encodes a *latest snapshot*: repeated snapshots can generate
  new video access units. This is not direct capture from arbitrary apps.
* The encoder runs inside the kernel Comm callback. Its CPU work and any
  kernel scheduling contention require a real, **explicitly authorized**
  benchmark before performance claims.
* PyAV/libx264 availability is not assumed on managed Colab. No automatic
  installation and no added batch-runner dependency.
* Bounded at one request at a time, ordered whole access units only. There
  is no negotiated queue-aware service policy, congestion sensing, or media
  playout/transport guarantee in this milestone.
* Tests use a fake kernel Comm and fake encoded access units. An optional
  PyAV smoke test checks real codec output where that dependency exists.

Run locally:

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m pytest -q -rs tests/test_bridge_video_colab_source.py tests/test_bridge_video_comm_protocol.py tests/test_bridge_video_comm_wire.py tests/test_bridge_comm.py
```
