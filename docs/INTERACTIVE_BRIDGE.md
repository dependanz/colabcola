# Interactive compute bridge (experimental)

Colab Cola's long-term goal is to make remote compute usable from a local
machine through a continuous connection. **The Python job runner remains a
primary product**, alongside future interactive and general-program execution.
Google Colab is the first compute provider, not a permanent architectural
restriction.

## Status

This branch contains a **loopback transport prototype**, not a managed Colab
graphics session. It never allocates Colab compute, modifies a current job,
opens a public listening port, or changes checkpoint-recovery behavior. It
demonstrates the independent input/output loop that a future provider adapter
will transport.

## Try it on Windows (two PowerShell terminals)

From an editable checkout of the feature branch:

```powershell
python -m pip install -e .
colabcola bridge serve --fps 8
```

In a second terminal:

```powershell
colabcola bridge view
```

The viewer opens a Tk window showing a changing image. Move the mouse over
the image to move the glow; press Left or Right to alter the pattern, and Up
or Down to move its vertical center. Close the window to disconnect.

To have the **server** render its scene using a local CUDA GPU, install an
appropriate CUDA-enabled PyTorch in that Python environment and run:

```powershell
colabcola bridge serve --device cuda --fps 8
```

This uses the GPU on the machine running the server. It is **not** evidence
of streaming from a Google Colab T4.

For a shared local secret, set the same `COLABCOLA_BRIDGE_TOKEN` environment
variable in both terminals before connecting. An absent variable means an
empty secret, appropriate only for this loopback-only demo.

## Managed Colab T4 proof (new, not yet live-validated)

This proof uses the same authorized Colab provider as the Python job runner.
A standalone PyTorch scene computes **on the remote Tesla T4**; the viewer
displays 640×360 frames on the host and sends mouse/arrow-key events back.

The transport is deliberately conservative: an authenticated Jupyter Contents
API transfers zlib-compressed frame snapshots and the latest input event.
This is a *polling bridge*, **not** TCP tunneling, remote desktop, or general
application virtualization. We expect lower FPS and higher latency than the
localhost demo. A faster Jupyter websocket or permitted provider-native
streaming channel can replace it without changing the session interface.

**Before allocating compute**, close other assigned Colab runtimes and run:

```powershell
colabcola --profile pypi-new-account-test bridge colab --preflight-only
```

For an actual allocation, choose a **fresh**, empty run directory and confirm:

```powershell
colabcola --profile pypi-new-account-test bridge colab `
  --run-dir I:\colabcola-bridge-t4-proof-v1 `
  --gpu T4 --fps 6 --poll-hz 3 --max-seconds 600 `
  --confirm-allocation
```

This allocates a real GPU and may consume compute units. The viewer should
open automatically after the remote renderer reports a Tesla T4. Close it to
release the runtime. The maximum session budget is measured from allocation
start, not from the first rendered frame.

If cleanup cannot be confirmed, retry only the owned allocation using the
durable local journal:

```powershell
colabcola --profile pypi-new-account-test bridge cleanup `
  --run-dir I:\colabcola-bridge-t4-proof-v1
```

Do **not** delete a run directory reporting `needs_cleanup` until the
owned allocation has been confirmed absent. Allocation identity is recorded
before the allocation request. Other people's or independently allocated
runtimes must never be released automatically.

The first managed T4 integration is experimental, and still needs the live
Windows/Colab proof. There is no claim of measured remote frame rate,
interactive latency, or confirmed cross-provider behavior yet.

### Verification

```powershell
python -m pytest -q tests/test_bridge.py tests/test_bridge_colab.py
```

The offline suite uses a fake provider for allocation, frame decoding, and
cleanup cases; it does not allocate a GPU.

### Usage boundaries

Google's managed Colab service restricts remote-control activity on the free
tier with no positive compute-unit balance, and separately restricts abusive
proxy, tunneling, and media-serving workloads. This experiment uses the normal
authenticated Colab kernel and file APIs for a research visualization. Users
remain responsible for the current Google Colab terms and their plan's
compute-unit balance. It does not bypass usage limits.

## Architecture boundary

The initial protocol is a bounded, versioned length-prefixed JSON stream.

- Local to remote: handshake and normalized mouse/keyboard input events.
- Remote to local: handshake result and sequential frames, currently base64
  encoded binary PPM images. The viewer drops stale queued frames.
- Transport: asyncio streams over **loopback TCP only**. Arbitrary binds are
  rejected deliberately; authentication alone does not encrypt traffic.
- Renderer: tiny Python scene with CPU execution or optional torch/CUDA.
- Presentation: optional Tk local viewer, loaded only for the view command.

The protocol lives in `src/colabcola/bridge_protocol.py`, distinct from the
Colab allocator, credential handling, and frozen batch execution plans.

## Remaining integration work (not implemented here)

1. **Colab transport adapter:** use an authorized, owned Colab kernel/session
   and an explicitly designed, outbound-compatible channel. Do not assume
   that arbitrary incoming TCP ports can be opened on a Colab runtime.
2. **Session management:** attach, reconnect, cancel, release, and distinguish
   network disconnection from runtime loss. Preserve existing run journals.
3. **Frame pipeline:** use a suitable image/video codec and frame pacing,
   with throughput, latency, and input-to-photon measurements.
4. **Unmodified applications:** isolate a supported capture/input backend
   from the runtime supervisor; general GUI applications need a graphical
   environment, rendering capture, and device event injection.
5. **Other providers:** implement the same session/transport contracts over
   private GPUs and cloud backends, subject to their policy constraints.

Interactive session continuity is different from durable checkpoint recovery.
A lost interactive application may restart from an application-specific
checkpoint rather than preserving its exact on-screen state.

## Verification

```bash
python -m pytest -q tests/test_bridge.py
```

The tests cover protocol bounds, version checks, malformed inputs,
authenticated handshakes, real loopback frame transmission and input acceptance,
and CLI routing. Live Colab, Windows Tk rendering, network delays, and
GPU input-to-photon latency remain unverified by this suite.

Colab Cola and its use of Google services remain subject to provider terms
and applicable workload restrictions.
