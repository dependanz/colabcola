# v33c: optional host-side H.264 Jupyter Comm wire adapter

**Status: offline fake-kernel wiring only; NOT a working remote-video mode.**

`KernelVideoCommReceiver` attaches to the **existing** authenticated, run-ID-scoped
`KernelCommBridge` instance and expects the normal Jupyter `comm_msg` IOPub
wrapper. It issues an opt-in `video_pull` request using the existing shell
channel, validates the bounded `video_access_unit` CCV1 reply, correlates the
one outstanding pull ID, enforces non-regressing input acknowledgment, and
runs the dependency-aware sequence gate before any caller decodes H.264.

The existing Colab worker and kernel-side Comm handler do **not** implement
`video_pull` yet; invoking this against a live T4 would time out, waste compute
and require owned-allocation cleanup. Do not add a CLI selector or allocate a
runtime for v33c. This adapter is used only with offline fake peers.

Retries after a timed-out, disconnected, or uncertain send are intentionally
not supported. The owning managed bridge remains responsible for closing the
Comm and reconciling/releasing the exact owned allocation. Foreign comm IDs
and run IDs are ignored; malformed correlated replies fail closed without
logging content. This does not create a new socket or public endpoint.

Real Colab integration will require a remote H.264 source, bounded access-unit
buffering and a new opt-in kernel handler; the host will then decode CCV1 and
use the existing Tk viewer. Source `capture_ns` and host `perf_counter_ns`
remain **different clock domains**, so subtracting them is invalid.

## Offline validation

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m pytest -q -rs tests/test_bridge_video_comm_wire.py tests/test_bridge_video_comm_protocol.py tests/test_bridge_video_comm_local.py tests/test_bridge_comm.py
```
