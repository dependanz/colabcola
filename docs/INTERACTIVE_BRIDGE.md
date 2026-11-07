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
  --run-dir I:\colabcola-bridge-t4-proof-v7 `
  --transport comm --gpu T4 --fps 12 --poll-hz 12 `
  --max-seconds 180 --confirm-allocation
```

Earlier Comm-mode trials (v4 and v5) failed before streaming.
The v4 experiment reached the remote Python bootstrap but reported an
unspecified execution error; its journal later confirmed `released`.
The v5 experiment allocated its owned T4 but the kernel WebSocket readiness
flag was unset after attachment; **the CLI confirmed release**. Its output
did not prove whether the WebSocket was rejected, closed, or timed out.
Colab's documented `ipykernel.comm.Comm` is preferred for the remote
Comm handler, with `comm.create_comm` as a fallback.

The v6 real T4 trial **successfully established persistent Jupyter Comm**
with streamed CUDA-rendered frames and bidirectional input. At a configured
6 FPS render/pull target, observed delivered frame rates were about **5.2–5.9
FPS**, with **552–832 KiB/s** of compressed-frame throughput and fresh
input-to-ACK observation medians around **321–383 ms**. The runtime was
explicitly confirmed released. Compared with the v3 polling experiment, this
was a substantial improvement, but it is **not** a measurement of peak
streaming capacity or input-to-photon latency.

The WebSocket attachment probe reports only whether the connection opened,
is currently connected, subsequently closed, whether its thread remains
alive, an optional numeric close code, and an allowlisted exception type.
It never exposes endpoints, credentials, exception strings, or content.
The client may retry one *connection* before submitting any code, but never
replays an uncertain kernel execution. Both attempts are bounded and release
the connection they opened if unsuccessful.

### Optional v7 throughput characterization

The Comm transport can now run with up to **12 FPS rendering** and **12 Hz frame
pulls**, while the files transport retains its 6 Hz polling cap. Defaults are
unchanged. The client additionally reports median frame-pull request/response
time (`pull RTT p50`) and the fraction of received frames that are fresh
(`fresh A/B`); it fails if the worker sends no new frame for 30 seconds.

Use a new run directory for each explicit allocation:

```powershell
colabcola --profile pypi-new-account-test bridge colab `
  --run-dir I:\colabcola-bridge-t4-proof-v7 `
  --transport comm --gpu T4 --fps 12 --poll-hz 12 `
  --max-seconds 180 --confirm-allocation
```

### v7 results and v8 diagnostic

The live v7 trial at `--fps 12 --poll-hz 12` regressed sharply from v6:
only **1.9–2.4 unique FPS**, with **406–492 ms** median frame-pull
round trips and **828–1070 ms** fresh input acknowledgments.
Every reply was fresh (`fresh 4/4` or `5/5`), so duplicated/stale frames
were not responsible for the low delivered rate. The request/response RTT
alone imposes an approximate upper bound near 2.3 FPS at 440 ms.
The owned Tesla T4 was confirmed **released**.

The cause is *not yet established*. Relative to v6, both the remote
renderer target and the host pull cap doubled, and resource contention,
kernel scheduling, transfer overhead, or GPU-to-CPU synchronization could
be responsible. The new profiling sidecar measures actual **remote renderer
FPS**, median **CUDA-render-and-copy** time (GPU kernels + synchronization +
device-to-host transfer), **zlib compression** time, and **frame-file publish**
time. The renderer and Comm handler pass those values through the same
authenticated frame stream, without an additional remote HTTP request.
The client prints them alongside the already measured pull RTT, fresh
responses, and input ACK observation.

The next optional diagnostic trial changes **only one variable from v6**:
leave rendering at 6 FPS but increase host pull limit to 12 Hz. Use a
fresh directory and verify release before proceeding:

```powershell
colabcola --profile pypi-new-account-test bridge colab `
  --run-dir I:\colabcola-bridge-t4-proof-v8 `
  --transport comm --gpu T4 --fps 6 --poll-hz 12 `
  --max-seconds 180 --confirm-allocation
```

### v8 observed results

The real v8 trial (`--fps 6 --poll-hz 12`) recovered to roughly **5.0–6.2
unique FPS** after startup. The remote renderer itself reported **5.9–6.0
FPS**; frame-pull RTT medians were mostly **156–203 ms**, input ACK
observation medians **313–429 ms**, and the T4 released successfully.
Remote **CUDA render-and-copy** median was around **1 ms**, **zlib compression**
around **5–10 ms**, and atomic publication around **0–1 ms**.

This result isolates the changed variable more cleanly: raising the host
pull cap from 6 to 12 Hz, while retaining 6 FPS rendering, did **not**
reproduce v7's ~2.3 FPS slowdown. The extra pull capacity is generally
unused because only one request is in flight and its RTT is longer than
the ideal 83 ms 12 Hz period. The remaining candidate cause is linked to
the higher 12 FPS remote-rendering setting (or between-run differences),
not proven GPU throughput saturation.

The local Tk viewer is now fixed at **frame width × (frame height + 28 px)**.
The status footer uses concise telemetry, independent of the long, varying
performance strings in the terminal, so its text should not repeatedly
resize the graphical window.

### v9 observed results

The real v9 trial (`--fps 12 --poll-hz 6`) **did not reproduce v7's
slowdown**. The remote Tesla T4 consistently rendered around **11.9–12.0 FPS**;
the local client delivered **5.3–5.9 FPS** at a 6 Hz pull limit, with
**171–187 ms** typical pull RTT and **328–359 ms** input acknowledgments.
CUDA render plus device-to-host copy was around **1 ms**, compression around
**6–9 ms**, and frame publication around **0–1 ms**. The owned runtime was
confirmed released.

Across separate sessions, 6/6, 6/12, and 12/6 settings all performed near
5–6 received FPS, while 12/12 delivered only ~2.3 FPS. This suggests a
possible **interaction** between those settings, but per-session Colab proxy
variation remains an alternative explanation. We should not claim a
specific GPU, GIL, codec, or network root cause yet.

### v10 in-session controlled sweep (new; not live-verified)

A single optional `--benchmark-sweep` run now compares **five phases within
the same owned Colab runtime and Jupyter kernel connection**:

| Phase | Render rate | Host pull cap |
| --- | --- | --- |
| 1 | 6 FPS | 6 Hz |
| 2 | 12 FPS | 6 Hz |
| 3 | 6 FPS | 12 Hz |
| 4 | 12 FPS | 12 Hz |
| 5 | 6 FPS | 6 Hz (baseline repeat) |

Rate changes use an authenticated and run-ID-scoped Jupyter Comm control
message. The remote worker validates and applies a new rate in its own loop;
its timing sidecar includes a control sequence and active target so the host
can distinguish requested from actually applied changes. Each phase lasts
12 seconds by default; its first 2 seconds are excluded from measurements.
A per-phase line reports received FPS, median frame-pull RTT, fresh replies,
rate confirmation, and measured remote renderer throughput. The final
baseline checks for drift across the one session.

Only one frame request remains in flight. The original streaming default
and file-polling fallback are unchanged. A requested control change is
never allowed to allocate additional GPUs or start another kernel.

For another **real, optional T4 allocation**, with a new run directory:

```powershell
colabcola --profile pypi-new-account-test bridge colab `
  --run-dir I:\colabcola-bridge-t4-proof-v10 `
  --transport comm --gpu T4 `
  --fps 6 --poll-hz 6 `
  --benchmark-sweep --benchmark-phase-seconds 12 `
  --max-seconds 240 --confirm-allocation
```

The viewport opens automatically, and the benchmark closes it after all
phases complete. Closing it early terminates the sweep and releases the
owned runtime. A new allocation may consume Colab compute units; no live
v10 performance figures are claimed. Afterward confirm the journal has
`phase: released`, or retry `bridge cleanup` using the same profile and
run directory. The existing Python job runner and checkpoint recovery are
unaffected.

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
