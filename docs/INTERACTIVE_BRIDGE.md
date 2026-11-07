# Interactive compute bridge (experimental)

Colab Cola's long-term goal is to make remote compute usable from a local
machine through a continuous connection. **The Python job runner remains a
primary product**, alongside future interactive and general-program execution.
Google Colab is the first compute provider, not a permanent architectural
restriction.

## Status

This branch contains two experimental paths: a **loopback-only** Python
graphics/input demonstration and an opt-in **managed Colab T4** renderer over
the authenticated Jupyter Contents API. The T4 path can allocate compute
only with explicit confirmation. Neither path changes the existing batch
runner, checkpoint-recovery behavior, nor opens a public listening port.
The Jupyter Contents polling bridge was verified against a real T4. The
persistent Jupyter Comm path remains experimental and has not yet established
a successful live Colab connection.

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

## Managed Colab T4 proof (file-polling path verified)

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

The managed T4 connection completed its first real Windows/Colab test on
November 6, 2026. At 640×360, the initial successful live run observed about
0.3–0.7 displayed FPS and input-acknowledgement estimates around 1–2 seconds.
A subsequent instrumented proof (v3) observed intervals from 0.1–1.7 FPS,
GET p50 mostly 516–656 ms, PUT p50 mostly 375–625 ms, and negligible local
decompression. The 0.1 FPS interval coincided with a 7,562 ms GET outlier.
The T4 was released successfully in both runs, with an independent cleanup
check after the earlier run. This validates the authenticated remote
interaction path but **not** a low-latency user experience or other providers.

Subsequent instrumentation reports local GET/PUT/decompression request costs,
each as a median over the latest reporting interval. The ACK statistic is
measured from the beginning of a local input upload until a fetched frame
acknowledges that input; it is **not** an end-to-end input-to-photon latency
measurement. Intervals with no new acknowledgements deliberately omit the
ACK statistic instead of recycling stale values.

Do not infer CUDA render throughput from the current displayed FPS: each
frame is separately retrieved using an authenticated HTTP Contents API call
and input writes are also performed through that API. The next architectural
experiment is a lower-latency persistent transport with explicit ownership
and cleanup boundaries.

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

## Persistent Jupyter Comm transport (development preview)

The first T4 proof confirmed the Jupyter Contents polling bottleneck: typical
frame GET requests took roughly 500–650 ms and input PUT requests roughly
380–625 ms; local decompression was generally under 10 ms. A single GET also
stalled for 7.56 seconds. Request-by-request file transfer is fundamentally
unsuited to low-latency graphics.

The experimental `--transport comm` option connects to the **existing owned
Colab kernel** using its authenticated Jupyter WebSocket rather than reading
a file over HTTP for every frame and writing a file over HTTP for every input.

The same remote CUDA renderer still computes and publishes frames inside the
runtime. A small kernel Comm handler sends the latest compressed frame through
IOPub when the local viewer grants one frame credit, and receives normalized
keyboard/mouse messages on the shell channel. No unauthenticated TCP listener,
remote desktop service, or tunnel is created. A single frame credit bounds
in-flight video data, and the host discards redundant frames.

To prepare locally without using a T4:

```powershell
python -m pytest -q tests/test_bridge.py tests/test_bridge_colab.py tests/test_bridge_comm.py tests/test_bridge_comm_local_kernel.py
colabcola --profile pypi-new-account-test bridge colab --transport comm --preflight-only
```

For an explicitly authorized trial on a paid/eligible Colab account, use a
**new run directory**:

```powershell
colabcola --profile pypi-new-account-test bridge colab `
  --run-dir I:\colabcola-bridge-t4-proof-v5 `
  --transport comm --gpu T4 --fps 6 --poll-hz 6 `
  --max-seconds 300 --confirm-allocation
```

The first real Comm-mode trial (v4) reached a T4 allocation but failed its
Python bootstrap before displaying frames. The original output did not
identify the exception; it also did not display an explicit release
confirmation, so the v4 journal should be checked. Colab's documented
`ipykernel.comm.Comm` interface is now preferred, with a `comm.create_comm`
fallback. The remote worker writes a small failure report containing only
the bootstrap stage and Python exception **type**, and the host includes
that sanitized information when the bootstrap fails. The new Comm path
still requires live verification; no throughput or latency improvements are
claimed yet. Local real-kernel integration tests verify Jupyter Comm
messaging, but Google's proxy behavior may differ. The runtime is still journal-owned and must be
confirmed released after a trial; if uncertain, run `bridge cleanup` using
the matching profile and run directory. The existing `files` option remains
the proven fallback and is the default.

The expected advantage is fewer per-frame HTTP round trips, not guaranteed
60 FPS, network quality, or support for unmodified graphical applications.

## Architecture boundary

The initial protocol is a bounded, versioned length-prefixed JSON stream.

- Local to remote: handshake and normalized mouse/keyboard input events.
- Remote to local: handshake result and sequential frames, currently base64
  encoded binary PPM images. The viewer drops stale queued frames.
- Loopback transport: asyncio TCP streams on localhost only; arbitrary public
  binds are rejected.
- Managed Colab transport: authenticated Jupyter file snapshots, zlib-compressed
  PPM frames, and bounded latest-input writes to the **owned** T4 session.
- Renderer: tiny Python scene on CPU, local torch/CUDA, or remote Tesla T4.
- Presentation: Tk graphical viewer on the user's local machine.

The protocol lives in `src/colabcola/bridge_protocol.py`, distinct from the
Colab allocator, credential handling, and frozen batch execution plans.

## Remaining integration work (not implemented here)

1. **Colab acceptance:** validate the real managed T4 launch, authenticated
   frame retrieval, input acknowledgements, and guaranteed owned cleanup.
2. **Session management:** reattach, recover after local supervisor restart,
   and distinguish network disconnection from runtime loss.
3. **Frame pipeline:** move beyond REST polling to lower-latency transport,
   use a suitable codec, and verify frame pacing and input-to-photon latency.
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
python -m pytest -q tests/test_bridge.py tests/test_bridge_colab.py tests/test_bridge_comm.py tests/test_bridge_comm_local_kernel.py
```

The offline tests cover bounded protocol framing, actual loopback
input/frame exchange, Tk frame decoding, fake-provider allocation ownership,
frame integrity, cleanup, and CLI routing. Live Colab, network delays, and
GPU input-to-photon latency remain unverified by this suite.

Colab Cola and its use of Google services remain subject to provider terms
and applicable workload restrictions.
