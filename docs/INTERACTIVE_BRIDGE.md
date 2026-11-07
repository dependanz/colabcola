# Interactive compute bridge (experimental)

Colab Cola's long-term goal is to make remote compute usable from a local
machine through a continuous connection. **The Python job runner remains a
primary product**, alongside future interactive and general-program execution.
Google Colab is the first compute provider, not a permanent architectural
restriction.

## Status

The branch contains a **loopback-only** Python graphics/input demo
and a **managed Colab Tesla T4** bridge. The original Jupyter Contents
polling transport and the persistent authenticated Jupyter Comm transport
have both been verified on real managed T4 allocations. The latter now
defaults to two bounded in-flight frame credits (only when explicitly
selected); its real v14 run delivered approximately 9.42 unique frames
per second and matched the local Tk image-configuration rate. Live
performance varies between allocations and is not guaranteed.
The default transport remains file polling. All T4 allocations require
explicit confirmation, and none of this modifies the released Python
batch runner, checkpoint recovery, or public network bindings.

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

### v10 in-session controlled sweep (live verified)

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

The real v10 sweep ran successfully in
`I:\colabcola-bridge-t4-proof-v10-2ndtry` after its original run directory
was already occupied. It allocated one owned Tesla T4, ran all five phases
and confirmed `released` in the journal. Phase results were:

| Render FPS / pull Hz | Received unique FPS | Pull RTT p50 | Fresh replies |
| --- | ---: | ---: | ---: |
| 6 / 6 | 5.56 | 172 ms | 56/56 |
| 12 / 6 | 5.66 | 172 ms | 57/57 |
| 6 / 12 | 5.59 | 172 ms | 56/56 |
| 12 / 12 | 5.66 | 172 ms | 57/57 |
| 6 / 6 (repeat) | 5.47 | 172 ms | 55/56 |

Across the sweep, 281 of 282 frame responses were fresh. Each phase
confirmed its active renderer target (including 11.9 FPS at 12 FPS).
The 12/12 phase **did not reproduce** v7's ~2.3 FPS regression.
The repeated baseline was stable over this run. A transient provider or
network condition in v7 is a stronger hypothesis than a deterministic
12/12 workload interaction, but the cause of v7 remains unknown.

The current host loop permits only one outstanding frame request. With an
observed 172 ms RTT, its approximate throughput ceiling is
`1 / 0.172 ≈ 5.81 FPS`, consistent with the observed 5.5–5.7 FPS.
This is a ceiling from the protocol's sequential request/response design,
not proof of an immutable Colab bandwidth ceiling or GPU limitation.

### Optional tiny Comm RTT probe (new; not live verified)

The experimental `--probe-link` option sends eight tiny identity-scoped
Jupyter Comm ping/pong exchanges **before ordinary frame streaming starts**.
It logs the median small-message round trip without transferring frame
bytes or making additional HTTP requests. Comparing that value to the
normal `pull RTT p50` distinguishes a fixed kernel/proxy round-trip
overhead from the added cost of transferring a compressed frame.

It uses the same owned kernel and does not alter the default transport.
The option requires `--transport comm` and cannot be combined with the
in-session rate sweep. It can consume compute units only if the separately
explicitly confirmed Colab allocation is requested:

```powershell
colabcola --profile pypi-new-account-test bridge colab `
  --run-dir I:\colabcola-bridge-t4-proof-v11 `
  --transport comm --gpu T4 --fps 12 --poll-hz 12 `
  --probe-link --max-seconds 180 --confirm-allocation
```

Choose a fresh run directory, and confirm `phase: released` afterward.
The Python job runner and checkpoint recovery are unchanged.

### v11 observed results and opt-in bounded frame pipelining

The real v11 trial (`--fps 12 --poll-hz 12 --probe-link`) reported a
**281 ms median tiny-message Comm RTT** across eight requests without frame
payloads. The subsequent compressed-frame pull RTT was usually **421–469
ms**, with **2.0–2.4 unique FPS**, fresh frames on each reply, and
**~813–1093 ms** input acknowledgments. Remote CUDA rendering continued
at **11.9–12.0 FPS**, with render+device-to-host copy ~1 ms and
compression ~6–10 ms. The T4 was explicitly confirmed released.

This reproduces v7's low-throughput behavior, whereas the single-runtime
v10 sweep achieved ~5.6 FPS in the same 12/12 configuration. The
inter-session performance difference appears associated with Jupyter
proxy/network/kernel messaging conditions; we cannot yet identify
whether it is a transient provider limitation, bandwidth, runtime
scheduling, or another factor. The tiny-message and full-frame probes
were sequential rather than simultaneous, so subtracting the median
values is **not** a reliable estimate of payload serialization cost.

The existing one-request-at-a-time implementation is constrained by
`throughput <= 1 / RTT`. To investigate whether overlapping round trips
could improve throughput, `--inflight N` (N=2..4) is now available as
an **opt-in development experiment only** on `--transport comm`.
It keeps a maximum of N outstanding frame requests, sends them at
the configured `--poll-hz` without catch-up bursts, tags each request
and reply with an ID, and displays only strictly increasing renderer
frame sequence numbers. It fails on unsolicited/duplicate request
IDs, late responses, or stale frames and then performs ordinary owned
runtime release. Standard `--inflight 1` behavior is unchanged.
This experiment cannot be combined with `--probe-link` or
`--benchmark-sweep`; it does not allocate additional kernels or T4s.

Optional example using a **new** run directory:

```powershell
colabcola --profile pypi-new-account-test bridge colab `
  --run-dir I:\colabcola-bridge-t4-proof-v12 `
  --transport comm --gpu T4 --fps 12 --poll-hz 12 `
  --inflight 3 --max-seconds 180 --confirm-allocation
```

This allocation can consume Colab compute units. Output includes
received FPS, frame-pull RTT, fresh replies, and outstanding credit count.
**No v12 performance result has been measured**. More outstanding
requests may produce duplicate/stale frames or worsen input latency;
speedup is not guaranteed. Validate offline before running, and always
confirm the journal ends in `released` or retry owned cleanup.

### v12 observed bounded pipeline result

A real managed Tesla T4 completed the `--inflight 3 --fps 12 --poll-hz 12`
trial successfully. All 68 CI checks passed on Linux, and the local Windows
checkout passed **67 with one GUI-related skip**. During streaming,
the local client received **8.4–10.4 unique FPS**, with an approximate
mean across twelve reporting intervals of **9.38 FPS**. Frame request
RTT medians were mostly **156–172 ms**, fresh replies totaled **227/231
(98.3%)**, and input ACK observation was **187–281 ms**. Remote rendering
continued at **11.8–12.0 FPS**, with ~1 ms render+copy and ~5–12 ms
compression. The owned T4 explicitly confirmed release.

This measured improvement over the previous ~5.6 FPS single-in-flight
example is encouraging but **not a controlled one-session comparison**:
network and Colab conditions varied substantially between earlier trials.
The pipeline uses the same authenticated kernel WebSocket, never more than
three frame requests in flight in this test, and displays only new frames.
The delivered FPS measures unique frames queued for the Tk viewer,
not input-to-photon latency or guaranteed application display FPS.

### v13 credit-window sweep (live verified)

The successful v12 pipeline delivered roughly **9.38 unique FPS** with
`--inflight 3` at a 12 FPS remote renderer and 12 Hz pull rate. RTT
medians were typically **156–172 ms**, input ACK observation **187–281 ms**,
and 227/231 replies were fresh. The owned T4 was confirmed released.

The new opt-in `--benchmark-credits` mode holds remote rendering at
12 FPS and request pacing at 12 Hz, varying **only the maximum number of
outstanding frame requests** within one owned T4 and one kernel WebSocket:

| Phase | Maximum outstanding frame requests |
| --- | ---: |
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |
| 4 | 4 |
| 5 | 1 (repeated baseline) |

Each phase has a 12-second duration by default, with the first two
seconds excluded from performance measurements. At each boundary the host
stops issuing new requests and **drains all pending responses** before
changing its credit limit. Monotonic request IDs and frame sequences
remain intact throughout the run. Per-phase reports show unique received
FPS, response RTT median, fresh/total response count, observed input
acknowledgment median, and most recent remote renderer rate.

The comparison was completed in v13; the results are documented below. The single-credit default and existing rate sweep
remain unchanged. The benchmark automatically ends after all five
phases and invokes the same owned T4 cleanup. This path is experimental
and should not be interpreted as production-ready streaming.

Example of an **optional** real T4 allocation (may consume compute units):

```powershell
colabcola --profile pypi-new-account-test bridge colab `
  --run-dir I:\colabcola-bridge-t4-proof-v13 `
  --transport comm --gpu T4 --fps 12 --poll-hz 12 `
  --benchmark-credits --credit-phase-seconds 12 `
  --max-seconds 240 --confirm-allocation
```

Use a new run directory. Verify `phase: released` afterward, and
retry `bridge cleanup` only if release is unconfirmed. The benchmark
cannot be combined with the link probe, old renderer-rate sweep,
or a manually selected `--inflight > 1`.

### v13 observed results: two frame credits are sufficient for this scene

The real v13 credit sweep used one managed Colab T4 and one kernel
connection, with 12 FPS remote rendering and 12 Hz request pacing
throughout. It ran all five phases and confirmed runtime release.

| Credits | New frames received / second | Frame RTT p50 | Input ACK p50 | Fresh replies |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 5.75 FPS | 172 ms | 250 ms | 58/58 |
| **2** | **9.53 FPS** | **157 ms** | **250 ms** | **96/97** |
| 3 | 9.19 FPS | 156 ms | 235 ms | 93/93 |
| 4 | 9.58 FPS | 156 ms | 242 ms | 97/97 |
| 1 (baseline repeat) | 6.30 FPS | 156 ms | 234 ms | 64/64 |

Increasing the credit limit from one to two produced roughly **66% more
received unique FPS** compared with the first baseline. Four credits
provided only **0.05 FPS** beyond two in this controlled run.
The remote renderer remained near 12 FPS; only 1/409 replies
repeated an older renderer sequence. The repeated single-credit
baseline was somewhat faster, consistent with mild within-run variation.

For this specific scene, **two credits are the preferred experimental
configuration**, now used as the ordinary Comm-streaming default after
the additional v14 verification. This is not a general performance
guarantee across workloads, resolutions, or Colab sessions.

### Local Tk viewer measurement (live verified in v14)

`--viewer-stats` independently measures frames that the local Tk
event loop actually decodes and configures on the image label.
Its two-second terminal reports are separate from the network-side
received FPS and include median local Tk image+label processing time.
This is **not** an OS compositor/presentation or input-to-photon
measurement. It can reveal whether the GUI itself falls behind when
the bridge receives 9–10 FPS. The window remains fixed-size, and
the meter is disabled by default.

For an optional live comparison with **two credits** (a new T4
allocation that may consume compute units):

```powershell
colabcola --profile pypi-new-account-test bridge colab `
  --run-dir I:\colabcola-bridge-t4-proof-v14 `
  --transport comm --gpu T4 --fps 12 --poll-hz 12 `
  --inflight 2 --viewer-stats `
  --max-seconds 180 --confirm-allocation
```

Look for `GUI configured` and `kernel Comm` lines in the same
terminal. Always verify the owned journal ends with `phase: released`.

### v14 observed results and Comm credit default

The real v14 T4 run used **12 FPS remote rendering, 12 Hz pull pacing,
two outstanding request credits, and the Tk viewer meter**. Across
32 network reporting windows and 32 GUI measurement windows, the mean
of the logged frame rates was **9.42 FPS on both sides**. Network
delivery ranged **8.0–10.5 FPS**; Tk image/label configuration was
**7.9–10.3 FPS**. The mean of the per-window Tk processing medians was
**6.2 ms** (range **2.6–7.6 ms**).

The network returned **606 fresh replies out of 612 (99.0%)**.
Reported pull RTT medians averaged **168.8 ms** and the available
input-ACK medians averaged approximately **263 ms**. The T4 was confirmed
released. These data indicate Tk image decoding and label updates are
not substantially limiting throughput at 640×360 and ~9–10 FPS.

Because v12, the controlled v13 credit sweep, and v14 all supported
the two-credit configuration, the **ordinary experimental Comm
streamer now defaults to two outstanding frame requests**. This does
not change the globally default `files` transport, which retains
single-request polling. Existing users can explicitly select
`--inflight 1` to retain sequential Comm behavior. The controlled
benchmark modes and tiny-message probe also retain one-credit
operation. The remote renderer's existing default 6 FPS and pull
limit 3 Hz are not raised automatically.

No screen compositor confirmation, GPU-to-display frame timing,
or input-to-photon latency is claimed. This feature remains
experimental and does not imply general application virtualization.

## Feasibility of low-latency encoded video (codec proof only)

The next research question is whether a **persistent temporal video codec**
can replace independent zlib-compressed RGB frames, while using only the
existing authenticated kernel communication path for Colab. This is
**technically feasible**, but a suitable *continuous packet sender* and
receiver have **not yet been implemented or Colab-live-validated**.

### Offline reproducible codec comparison

Install FFmpeg locally if available (with the optional `libx264` encoder).
This script uses only local CPU-generated frames and has **no Colab API
calls, runtime allocations, sockets, streaming server, or package-level
FFmpeg dependency**:

```powershell
python -m colabcola.bridge_video_benchmark --width 640 --height 360 --fps 12 --frames 24 --pattern scene
```

A separate offline Linux run of that experiment generated 24 synthetic
frames approximating the remote glow-and-wave scene:

| Codec/metric | Observed local result |
| --- | ---: |
| zlib(level=1), independent RGB frames | 3,777,090 bytes total |
| H.264/AVC libx264, ultrafast/zerolatency, CRF 23 | 81,267 bytes total |
| Size ratio H.264/zlib | 0.0215 (about 46× smaller) |
| H.264 PSNR vs original 8-bit RGB frames | 43.74 dB |
| Decoded frames | 24 of 24 |

The H.264 output is a **single encoded sequence with inter-frame
prediction**, whereas the zlib baseline compresses each RGB frame
independently. H.264 is lossy and the comparison is strongly
**scene-dependent**; moving games, camera changes and complex geometry
may compress very differently. FFmpeg batch encode/decode process times
do not measure live, per-frame latency. The official real T4 bridge
render+compression measurements were produced by a different pipeline.

The current hosted Jupyter frame protocol remains request-tagged
Base64-encoded zlib frames. The offline proof is an existence test for
a video codec, **not** a claim of real-time H.264 streaming through
Colab, 60 FPS, or support for unmodified game applications.

### Next experimental design (not yet implemented)

1. Introduce bounded, versioned **encoded-video packet metadata**
   (session ID, sequence, presentation timestamp, codec, keyframe marker,
   packet size), independently of the provider.
2. Add an authenticated, paced **push** sender instead of requiring one
   shell-channel frame request per encoded frame; cap the receiver buffer
   and define keyframe recovery on loss or resynchronization.
3. Decode the packets in a local video decoder. Compare end-to-end
   delivered FPS, frame age, input ACK, and bandwidth against the proven
   two-credit zlib baseline **on the same synthetic application**.
4. Only after local tests, evaluate whether a bounded managed-Colab
   research visualization is consistent with the latest Google Colab
   usage rules. **Do not deploy a general media-serving platform,
   remote desktop, public streaming endpoint, reverse tunnel, or
   provider-limit bypass.**

Google's current restrictions distinguish media serving unrelated
to interactive compute (restricted for all managed runtimes) from
additional free-tier restrictions on remote control without a positive
compute-unit balance. The service also requires authorized access.
Do not assume a custom Jupyter-based transport is automatically
approved. Official references:

- https://research.google.com/colaboratory/faq.html
- https://research.google.com/colaboratory/tos_v6.html

For providers supporting dedicated streaming, a separately implemented
WebRTC/RTP or similar media transport could be substituted, without
changing the application-facing frame/input interface.

## Persistent H.264 streaming on local loopback (offline development proof)

A second independent experiment now tests **actual progressive streaming**,
not just whole-sequence codec size. The new module
`src/colabcola/bridge_video_stream.py` uses:

1. Precomputed deterministic RGB frames paced at the target frame rate;
2. One **persistent FFmpeg/libx264** encoder, with B-frames disabled,
   ultrafast/zerolatency settings and Annex B access-unit delimiters;
3. A bounded versioned packet header containing codec identity, frame
   sequence, source presentation timestamp, monotonic capture timestamp,
   keyframe flag and encoded byte length, transmitted over a **local
   socket pair** (no remotely accessible listener);
4. One persistent FFmpeg H.264 decoder operating concurrently with
   the producer, including decoder `-threads 1` to reduce buffering.

The parser needs the **next access-unit delimiter** to identify the end
of the previous access unit; this imposes a one-frame lookahead. The
decoder may buffer additional frames. The implementation measures how
many frames decode *before the source finishes* and the median local
capture-to-RGB-decode age. It also validates packet lengths, source
frame ordering and decoded-frame count. This proof does **not** implement
cross-network error recovery, lost-reference recovery or a GUI viewer.

Optional local command, with FFmpeg executable available:

```powershell
python -m pytest -q tests/test_bridge_video_benchmark.py tests/test_bridge_video_stream.py

python -m colabcola.bridge_video_stream `
  --width 640 --height 360 `
  --frames 24 --fps 12 --pattern scene
```

A developer-side Linux dry run using the matching synthetic scene
produced **24/24 decoded frames**, of which **20 appeared before the
source finished**, at approximately **12 FPS active decoding**.
The median local source-to-decoded-frame age was about **256 ms**.
Encoded size was about **81 KB**, versus **3.78 MB** for independent
zlib frames. These performance observations depend on FFmpeg, CPU,
scene and OS, and are not guaranteed on Windows or Colab. CI tests
also verify progressive decoding and full frame recovery.

A faster-rate local synthetic experiment (still precomputed frames):

```powershell
python -m colabcola.bridge_video_stream `
  --width 640 --height 360 `
  --frames 48 --fps 60 --pattern square
```

This is a **local media-codec demonstration**, not general application
or game streaming. No CUDA, Colab allocation, owned runtime, Jupyter
Comm, remote traffic, public server or provider-policy exception is
involved. Its timestamps share the same local clock and must **not**
be interpreted as synchronized timestamps across machines. The
reported decoded-frame cadence is not compositor-presented FPS.

**Important result:** even a functioning persistent video path can
still buffer multiple frames. The local 12 FPS test's ~256 ms median
source-to-decode age is far from the low latency expected of responsive
game streaming. Before substituting video into the managed T4 bridge,
improve the packetization/decoder delay, maintain bounded recovery
semantics and verify the design against Colab's authorized usage rules.

### v17 local latency decomposition (offline, no provider allocation)

The Windows v16 tests confirm that the persistent H.264 loopback path
decoded **24/24** glow-scene frames at **12.03 active FPS** with a median
capture-to-decode age of **250.0 ms**. The 640×360, 60 FPS moving-square
test decoded **48/48** frames at **61.17 active FPS**, median age **47.0 ms**
and maximum age **63.0 ms**. The respective medians correspond to
approximately **3.0 and 2.82 frame periods**. Both tests decoded all
frames, with 20/24 and 44/48 produced before the sources finished.

This frame-period similarity suggests a small **frame-count pipeline
lag**, rather than a fixed 250 ms network latency. It does not prove
which stage is responsible: the existing x264 Annex B access-unit parser
waits for the *next* AUD, and the FFmpeg decoder may buffer frames.

New optional instrumentation breaks the local age of each frame into:

- `capture_to_packet_p50_ms`: original RGB capture through encoder
  output and access-unit parser readiness (including lookahead);
- `packet_to_socket_p50_ms`: packet enqueue through complete local
  socket receipt (including scheduling);
- `socket_to_decoded_p50_ms`: packet receipt through reconstructed RGB
  output (including decoder buffering);
- `capture_to_decoded_frame_periods`: full median age expressed in
  target-frame periods.

These values use a single process-local monotonic clock. Individual
per-frame stages sum exactly to total frame age, but **medians of
separate distributions need not sum to the total median**.
They do not measure a real network, remote CUDA, or screen presentation.

To compare the two conditions without purchasing compute:

```powershell
python -m colabcola.bridge_video_stream --width 640 --height 360 --frames 24 --fps 12 --pattern scene
python -m colabcola.bridge_video_stream --width 640 --height 360 --frames 48 --fps 60 --pattern square
```

The next change should target whichever stage dominates, while keeping
the current bounded packet format and testability. No change to the
Colab allocator, installed FFmpeg dependencies, production batch runner,
or live authenticated Comm path has been made here.

### v17 measured stage contributions and decoder flag A/B

Linux CI confirms that the apparent three-frame H.264 latency originates
primarily in **access-unit packetization (~1 frame interval)** and the
**FFmpeg CLI decoder input/output pipeline (~2 frame intervals)**.
For the same local 640×360 synthetic scenes:

| Stage (median, Linux) | 12 FPS glow scene | 60 FPS square |
| --- | ---: | ---: |
| Capture → complete encoded packet | ~85.6 ms | ~18.7 ms |
| Complete packet → local socket receipt | ~0.1 ms | ~0.1 ms |
| Socket receipt → decoded RGB output | ~168.0 ms | ~34.5 ms |
| Total | ~253.7 ms | ~53.2 ms |

An opt-in `--decoder-low-delay` switch applies FFmpeg's
`-flags +low_delay` before opening the H.264 decoder. The Linux A/B
at 12 FPS returned **253.7 ms baseline vs 253.5 ms low-delay**;
the 60 FPS square returned **53.2 ms baseline vs 52.7 ms low-delay**.
It decoded all frames, but **did not materially reduce** latency.
The default remains unchanged; the flag is purely diagnostic.

### v18 in-process H.264 codec comparison (Linux CI verified, local only)

Using PyAV's direct FFmpeg CodecContext API, the v18 experiment emits
**complete encoded access units directly from libx264**, wraps them in
the same bounded local frame packet format as the v16 socketpair test,
and feeds them directly into an in-process H.264 decoder. The original
v16 experiment instead uses two persistent FFmpeg subprocesses and
needs the *next Annex B access-unit delimiter* to complete each encoded
frame's packet.

A matched Linux CI A/B, using the same 640×360 scenes, showed:

| Median stage latency | 12 FPS: subprocess | 12 FPS: PyAV | 60 FPS: subprocess | 60 FPS: PyAV |
| --- | ---: | ---: | ---: | ---: |
| Capture to encoded packet | 85.0 ms | **2.1 ms** | 17.9 ms | **1.4 ms** |
| Packet to local socket receipt | 0.1 ms | 0.1 ms | ~0.0 ms | ~0.0 ms |
| Local socket receipt to decoded RGB | 167.7 ms | **0.4 ms** | 33.9 ms | **0.4 ms** |
| **Total capture to decoded RGB** | **252.8 ms** | **2.6 ms** | **51.8 ms** | **1.8 ms** |
| Frames decoded | 24/24 | 24/24 | 48/48 | 48/48 |
| Frames decoded before input end | 20/24 | 23/24 | 44/48 | 47/48 |

The in-process trials maintained approximately **12.01 FPS** and
**60.18 FPS** active decode cadence, respectively. Encoded sizes
differ somewhat between the two implementations because their
encoder configurations are not identical, so this is a controlled
**architectural feasibility comparison**, not a microbenchmark
attributing every millisecond exclusively to one specific FFmpeg
component. It demonstrates that H.264 itself need not introduce
a three-frame pipeline delay on these test cases.

These are **CPU-only, precomputed RGB scenes with a same-process
local socketpair**. No render/capture workload, real network
conditions, Jupyter proxy, remote CUDA, external listeners, GUI
compositor, or game input are included. In-process stream
performance for the full 640×360 benchmark on Windows
**has now been measured**, while real managed Colab video streaming
remains untested.

A Windows Python 3.12 / PyAV 19.0.2 test exposed a timing-clock
precision issue for very small (96×64) frames: the previously
reported median latency rounded to `0.0 ms` even though encoding
and decoding completed successfully. In-process stage timestamps
now use `time.perf_counter_ns()` for high-resolution elapsed time;
the report includes `capture_to_decoded_p50_us` and three-decimal
millisecond timings. The regression still requires a positive
measurement. Windows CI now installs PyAV and NumPy explicitly:
**94 tests passed, 6 optional FFmpeg-dependent tests skipped**.
Linux CI passed **100 tests** on the same updated code. These
are test-suite results, not Windows 640×360 performance claims.

PyAV and NumPy remain **optional local-only development
dependencies**, not dependencies of the released `colabcola`
installation:

```powershell
python -m pip install av numpy
python -m pytest -q tests/test_bridge_video_inprocess.py

python -m colabcola.bridge_video_inprocess `
  --width 640 --height 360 --frames 24 --fps 12 --pattern scene

python -m colabcola.bridge_video_inprocess `
  --width 640 --height 360 --frames 48 --fps 60 --pattern square
```

**Distribution licensing:** PyAV binary wheels may embed
FFmpeg/x264 components subject to GPL licensing obligations.
A production codec integration would need a licensing and binary
distribution review. The demonstrated low-latency in-process
packet interface could alternatively be implemented using another
appropriately licensed FFmpeg integration or a platform-native
codec API. Do not assume the existing Colab Jupyter kernel can or
should be used as a general video-serving platform.

### v18 measured on Windows (PyAV 19.0.2, Python 3.12)

After the high-resolution timing fix, the user reran the optional in-process
H.264 test at 640×360 on Windows. **All four PyAV-specific regression tests
passed**; both live-paced synthetic scenes decoded every frame:

| Metric | 12 FPS glow scene | 60 FPS moving square |
| --- | ---: | ---: |
| Decoded frames | 24/24 | 48/48 |
| Decoded before source finished | 23/24 | 47/48 |
| Active decoded cadence | 12.04 FPS | 60.58 FPS |
| Median capture-to-decoded RGB | **2.285 ms** | **1.865 ms** |
| Median capture-to-packet | 1.710 ms | 1.403 ms |
| Median packet-to-local-socket | 0.121 ms | 0.098 ms |
| Median socket-to-decoded RGB | 0.456 ms | 0.354 ms |
| H.264 bytes | 76,329 | 26,377 |

These Windows results corroborate Linux's approximately 2–3 ms
in-process benchmark. They are not measurements of GPU frame capture,
network transfer, local display configuration, or input-to-photon latency.

### v19 live, input-responsive H.264 local demo (headless CI verified)

`src/colabcola/bridge_video_live.py` introduces a separate,
bounded **live frame source** rather than replaying precomputed images.
It renders an animated moving stripe and an input-controlled marker.
Mouse movement and arrow keys alter the marker in the producer before
the next H.264 frame is encoded. A persistent in-process PyAV encoder
sends complete access units over the existing bounded binary socketpair
format to an independent decoder, which forwards RGB images through the
existing Tk viewer's bounded frame queue.

The new local pipeline measures capture-to-decoded RGB latency, encoding
and decoding cost, and **input-application-to-decoded-frame** latency.
This last metric starts when the producer processes an input event,
**not** when the user physically presses a key or when the desktop
compositor presents the resulting frame. The GUI additionally reports
local Tk image-configuration FPS when enabled.

Linux headless CI has now exercised a continuously generated
**640×360 scene at 12 FPS**, producing and decoding **24/24 frames**
at **12.04 decoded FPS** and a **2.735 ms median local
capture-to-decoded-RGB age**. This includes fresh CPU scene generation,
H.264 encoding, bounded local packet transfer and in-process decoding,
not just playback of precomputed frames. It does not include network or
display time. The separate tests also inject a mouse event, validate
that a subsequent decoded frame acknowledges its application,
and verify RGB output changes with input and animation.

The GUI route is opt-in: it uses the same fixed-size Tk viewer as the
bridge, measures local widget frame configuration, and receives the
input-controlled marker and animated stripe. A graphical Tk smoke
test exercises clean automatic viewer closure when a short session
finishes. Continuous sessions are intentionally bounded by
`--max-seconds` (1–180) and can also be interrupted by closing
the window.

For a no-GPU, no-Colab *headless* smoke check (PyAV/NumPy must be installed):

```powershell
python -m pytest -q tests/test_bridge_video_live.py
python -m colabcola.bridge_video_live --width 640 --height 360 --fps 12 --frames 24 --max-seconds 5
```

To show the interactive local viewer:

```powershell
python -m colabcola.bridge_video_live --show --width 640 --height 360 --fps 30 --max-seconds 30
```

Move the mouse over the image or use the arrow keys. The viewer
closes automatically after its bounded frame budget (or can be
closed manually). This test **does not allocate a Colab GPU, access the
network, open a public listener, or modify the existing batch runner**.
It does not prove remote encoded-video streaming or lost-reference
recovery. The new live module is a separate optional experiment; no
provider-side media integration is implied.

### v19 Windows live interactive H.264 verification

The user ran `bridge_video_live --show --width 640 --height 360
--fps 30 --max-seconds 30` on Windows with PyAV installed.
All **6 local live-source regression tests passed**. The run was
closed after **508 frames** (approximately 16.9 seconds of decoded
frames at 30 FPS), without a crash:

| Local measurement | Result |
| --- | ---: |
| Produced and decoded frames | 508 / 508 |
| Active decoded cadence | **30.01 FPS** |
| Median source capture to decoded RGB | **2.689 ms** |
| Median H.264 encoding cost | 2.000 ms |
| Median H.264 decoding call | 0.191 ms |
| Input changes reflected in decoded frames | 345 |
| Median input *application* to decoded RGB | 2.677 ms |

Tk image-label configuration generally tracked 30 FPS.
The source-input-application metric specifically excludes GUI callback
queueing and waiting for the next render tick. It is not the
perceived response time to a physical mouse movement or keystroke.

### v20 local Tk input-callback-to-image-configuration timing

The next opt-in **local-only** instrumentation timestamps an accepted
mouse or keyboard callback on the Tk main thread, carries that timestamp
in private local frame metadata alongside the in-process encoded video
and reconstructed pixels, and records the time at which the matching
frame is configured onto the Tk image label. The remote input protocol,
Colab transport, packaged job runner, and viewer's existing PPM path
are not changed.

The console GUI meter now also reports:

- `capture→Tk p50`: local scene capture through image-label update;
- `decode→Tk p50`: local decoded RGB readiness through label update,
  including queueing and Tk processing;
- `input callback→Tk p50`: accepted Tk mouse/key callback through
  label update **for the first displayed frame reflecting that input**.
  Consecutive frames carrying the same acknowledged input are counted
  only once. Dropped or superseded input changes are not claimed as
  individually displayed.

A finite-duration Tk demo remains available:

```powershell
python -m pytest -q tests/test_bridge_video_live.py tests/test_bridge_viewer_stats.py
python -m colabcola.bridge_video_live --show --width 640 --height 360 --fps 30 --max-seconds 30
```

This timestamp interval includes event callback-to-producer scheduling,
wait for the next frame, CPU RGB generation, H.264 encoding/decoding,
bounded frame queueing and Tk label configuration. It **does not**
include device-to-Tk OS event dispatch or actual physical display scanout,
and it is not an input-to-photon measurement. No Colab compute is
allocated; no public listener or network transport is involved.

### v20 verified on Windows: callback-to-Tk latency

On the user’s Windows machine, a 640×360, 30 FPS, 30-second local
H.264 session completed **900 produced / 900 decoded frames** at
**30.00 decoded FPS**, with **12/12 targeted regression tests passing**.
The GUI's per-window frame configuration rate was generally near
30 FPS. Input changes appeared in **309 decoded frames**.

| Window-level latency statistic | Observed v20 |
| --- | ---: |
| Capture → decoded RGB, whole-session p50 | **2.141 ms** |
| Input callback → decoded RGB, whole-session p50 | **17.662 ms** |
| Decode → Tk configuration, reported-window p50 | ~18–26 ms |
| Capture → Tk configuration, reported-window p50 | ~21–28 ms |
| **Input callback → Tk configuration**, reported-window p50 | **35.1–42.1 ms** |

The reported-window input-to-GUI medians were around 40 ms in
steady state. They measure the delay from an **accepted Tk event
callback** to configuring the **first locally decoded frame that
includes that event's applied state**. They do not include physical
input device dispatch or display-compositor scanout. Per-stage
medians over different windows/samples are not exactly additive.

The PyAV encoder and decoder are not the dominant source of
local interaction latency: the event can wait for the next
30 FPS render tick (33.3 ms spacing) and the GUI currently
processes its bounded frame queue using a 30 ms Tk timer.

### v21 Tk polling interval A/B (Windows live verified)

The same local 640×360 scene ran on Windows at **30 FPS** with
`--viewer-poll-ms 8`. The user’s **14 targeted local tests passed**
and the **full 30-second session produced and decoded 900/900 frames**
at **30.01 FPS**. The same test previously ran at the unchanged
30 ms viewer default (v20) with 900/900 decoded at 30.00 FPS.
The 8 ms mode remains **opt-in**, scoped to the local H.264 demo.
The existing managed-Colab viewer stays at 30 ms polling.

| Local metric | v20, 30 ms poll | v21, 8 ms poll | Change |
| --- | ---: | ---: | ---: |
| Decoded FPS | 30.00 | 30.01 | Stable |
| Callback → configured Tk image | **40.5 ms** | **29.0 ms** | **−11.5 ms (−28.4%)** |
| Decoded RGB → configured Tk image | 22.4 ms | 13.8 ms | −8.6 ms (−38.4%) |
| Scene capture → configured Tk image | 24.4 ms | 16.2 ms | −8.2 ms (−33.6%) |
| Capture → decoded RGB, whole-session p50 | 2.141 ms | 2.080 ms | Similar |
| Inputs reflected in decoded frames | 309 | 433 | Different interaction pattern |

The GUI timing figures above are **medians of per-window medians** from
the terminal logs (not pooled frame-level p50s). There were 13
baseline and 12 faster-poll windows with usable callback-to-Tk
readings; some windows contained no eligible newly acknowledged input.
Two separate runs with different amounts of mouse movement do not
establish a definitive causal latency effect, but they support that
faster Tk polling reduces local queue wait without sacrificing the
30 FPS throughput of the test scene.

This is **not** physical device-input-to-display latency. The timing
starts when Tk accepts a mouse/keyboard callback and ends when the
label is configured, before OS compositor/scanout.

#### v22: 60 FPS local H.264 application path verified on Windows

Using the same 8 ms Tk queue polling as v21, the user ran the live
640×360 H.264 scene for 30 seconds with a **60 FPS source**.
The experiment produced and decoded **1,800/1,800 frames** at
**60.01 decoded FPS** and ended normally.

| Metric | v21: 30 FPS, 8 ms poll | v22: 60 FPS, 8 ms poll |
| --- | ---: | ---: |
| Input callback → Tk configured image | **29.0 ms** | **22.05 ms** |
| Decoded RGB → Tk configured image | 13.8 ms | 11.8 ms |
| Tk frame configuration (typical) | ~30 FPS | ~60 FPS |
| Input callback → decoded RGB, whole-session p50 | 16.11 ms | **9.13 ms** |
| Capture → decoded RGB, whole-session p50 | 2.080 ms | 2.184 ms |
| Produced / decoded frames | 900/900 | 1800/1800 |
| Inputs reflected in decoded frames | 433 | 623 |

The callback→Tk figures are **medians of per-window medians**:
12 qualifying windows for v21 and 14 for v22. Raising the source
rate from 30 to 60 FPS corresponded to a **6.95 ms (~24%)**
lower callback-to-Tk delay while local H.264 processing stayed
near 2 ms. Both experiments were independent live sessions with
different input activity, and the window medians do not constitute
a paired event-by-event statistical test.

Tk configuration ran close to 60 FPS for **10 of 14** reporting
windows. Four windows reported 48.9–58.0 FPS, even though decoded
frames remained at approximately 60 FPS. This shows a working
local 60 FPS H.264 application demonstration but **not perfectly
stable or compositor-verified presentation**.

The next substantive bottleneck to study is not further small
Tk timer tuning: it is how encoded video behaves under
**variable network latency, jitter, bandwidth limits and packet
loss**, while keeping freshness and input responsiveness. A
safe next experiment would be a **local-only impaired-link
harness** between the persistent H.264 packet producer and decoder,
with a bounded queue and controlled delay/jitter. Predictive
H.264 frames cannot be arbitrarily dropped without considering
reference dependencies and IDR/keyframe recovery. This is a
design direction, **not an implemented or measured remote
transport**.

No managed Colab allocation, public socket, GPU, network
proxy, or changes to the released batch runner are involved.
The latency measurements end at Tk label configuration, not
physical display scanout or input-to-photon.

## v23 local-only H.264 network-impairment experiment

This new **deterministic virtual-time test harness** addresses what
makes continuous encoded video fragile when the connection fluctuates.
It is deliberately separate from every Colab/provider transport.
It creates real synthetic-scene H.264 access units using an optional
in-process PyAV/libx264 encoder, then schedules those units through
a simulated FIFO access-unit link. At the receiver it uses an
**actual in-process H.264 decoder** and validates the decoded RGB.

The virtual link models **fixed delay, seeded jitter, a finite
bandwidth rate, piecewise bandwidth steps, whole-access-unit erasure,
a bounded number of admitted in-flight frames, and a maximum
frame-age cutoff**. Only queued/admitted frames consume link bandwidth;
an erased frame does consume bandwidth, whereas a queue-full frame
never enters the link. Every unit is counted once as delivered or
dropped. Actual wire framing is validated through the versioned
bounded `VideoAccessUnit` format used by prior local experiments.

**Important predictive-video safety property:** when an H.264
predicted frame is erased or dropped, dependent frames must not be
blindly fed to the decoder. This experiment marks the reference
chain broken, suppresses dependent frames, and starts a fresh
H.264 decoder only at a **verified IDR** carrying its own SPS/PPS.
A fixed closed-GOP encoding interval controls the next possible
recovery point. This is conservative and intentionally sacrifices
continuity to avoid presenting corrupted reconstructed frames.
No dynamic keyframe-request feedback or retransmission is implemented.

### Verified local results (initial v23 CI)

The first branch CI proved **real H.264 reference-chain recovery**
using PyAV on synthetic 128×72, 24 FPS frames with an IDR every
12 frames. The virtual FIFO link tests produced:

| Model condition | Link delivered | Actually decoded | Decoded contiguous runs | IDR recoveries |
| --- | ---: | ---: | --- | ---: |
| 30 ms delay, ±10 ms jitter, 2,000 kbps, no loss | 36 / 36 | **36 / 36** | 0–35 | 0 |
| 30 ms delay, explicit access-unit erasure at index 5 | 35 / 36 | **29 / 36** | 0–4, 12–35 | 1 |

The erased predicted frame cost **7 decodable frames**: the
erased frame itself plus six reference-dependent frames 6–11.

A third Linux CI example held an encoded **320×180, 30 FPS,
48-frame** stream at a piecewise bandwidth of **2,500 kbps
(0–500 ms), 120 kbps (500–1,000 ms), then 2,500 kbps**
with 50 ms base delay, ±25 ms jitter, a six-unit in-flight
queue and a 250 ms age cutoff. **All 48/48 frames decoded**
and peak in-flight occupancy was five, but the *virtual*
source-to-arrival latency increased to **166.27 ms maximum**
(median **54.84 ms**) and the largest gap between actually
decoded frame arrivals was **94.69 ms**. The received-cadence
estimate over the entire simulated span was **29.54 FPS**.

This is instructive: **near-30 FPS average delivery can coexist
with a ~95 ms interruption in a continuous frame stream**.
The model has no playout jitter buffer and no real network
capture, so it is evidence for the behavior of this specified
virtual link, not evidence about a particular cloud service.

The modeled receiver did **not** falsely count dependent frames as
decoded; it opened a new H.264 decoder at IDR index 12.
Virtual capture-to-arrival medians were about **28.6 ms** and
**30.0 ms**, respectively. These are controlled simulator outcomes,
not measured network or GUI timing.

Branch CI passed **121 tests on Ubuntu**, exercising PyAV and real
H.264 decoder recovery, and **115 tests on Windows**, with six
unrelated tests skipped because that runner lacks the external
FFmpeg executable. Neither OS test suite allocated Colab compute.

### Three useful offline runs

From the experimental branch, with PyAV and NumPy already installed:

```powershell
# 1. Delay and jitter alone: intact encoded stream
python -m colabcola.bridge_video_impairment `
  --width 320 --height 180 --fps 30 --frames 48 --gop 12 `
  --delay-ms 50 --jitter-ms 20 --bandwidth-kbps 2000 `
  --max-inflight 12 --max-age-ms 300 --seed 7

# 2. Lose a predicted frame (source index 5): wait for IDR at index 12
python -m colabcola.bridge_video_impairment `
  --width 320 --height 180 --fps 30 --frames 48 --gop 12 `
  --delay-ms 40 --jitter-ms 0 --bandwidth-kbps 0 `
  --max-inflight 12 --max-age-ms 0 --drop-seqs 5

# 3. Throughput drops sharply and then recovers (virtual millisecond clock)
python -m colabcola.bridge_video_impairment `
  --width 320 --height 180 --fps 30 --frames 48 --gop 12 `
  --delay-ms 50 --jitter-ms 25 --seed 7 `
  --bandwidth-steps "0:2500,500:120,1000:2500" `
  --max-inflight 6 --max-age-ms 250
```

The JSON report separates **link delivered frames**, **erased,
queue-full and stale drops**, **frames skipped due to missing H.264
dependencies**, **IDR resynchronizations**, and **contiguous runs
of actually decoded frames**. It also reports virtual source-to-
arrival age, arrival cadence/gaps, queued bytes, and separately
the actual codec decoder call time. If a keyframe itself is lost,
the decoder must wait for the *next* valid IDR to resume.

Use `--seed` to reproduce jitter and random loss, and
`--loss-percent` to add repeatable whole-frame erasures. All
simulated times are **virtual**, not measured network latency;
real H.264 encoding and decoding occur locally but do not delay
the simulated clock. A bandwidth step is a piecewise-constant
bit rate, integrated over packets that cross a change boundary.

**Limitations:** this is a bounded access-unit-level **ordered
virtual link**, not an actual UDP/IP packet network, WebRTC
implementation, FEC/retransmission layer, network jitter buffer,
congestion-control algorithm, or live GUI/input test.
It does not fragment frames into MTU packets or simulate reordering.
The model uses fixed periodic IDRs and does not request a keyframe
in response to loss. Thus it provides a transparent baseline for
correctness under impairment, **not a demonstrated solution for
cloud game streaming** or a promise of smooth playback.

Its frame count is capped at 120, queue at 128, link rate and
timings bounded, and it has no Colab allocation, real network
interface, GUI listener or GPU use. It leaves the existing remote
Jupyter Comm and released batch runner unchanged. These experiments
need only the optional local PyAV/NumPy research dependencies.

## v24: feedback-triggered IDR and virtual playout-buffer comparison

This next **local-only research experiment** adds two distinct
questions to v23: (1) can a receiver's **loss-triggered keyframe
request** shorten the period of undecodable H.264 frames compared
with waiting for a periodic IDR, and (2) how much *added playback
latency* is necessary for a receiver to present decoded frames at
a regular cadence under modeled jitter/bandwidth fluctuations?

Implementation: `src/colabcola/bridge_video_feedback.py`. It
reuses the v23 encoded access-unit link and actual PyAV decoder,
but does **not** alter the working managed-Colab bridge or existing
real Tk viewer.

### Verified v24 CI comparison (local virtual link only)

Actual H.264 encoding/decoding and the scheduling model passed
**132 tests on Ubuntu** and **126 on Windows, with six unrelated
external-FFmpeg-dependent tests skipped**. The optional PyAV-based
feedback and playout tests exercised genuine encoded packets on
both operating systems. All runs are offline and require no
managed Colab allocation.

#### Lost predicted frame and single feedback IDR

For the **128×72**, **30 FPS**, **36-frame** stream, periodic
GOP=12 and explicit erasure at source index 5, the Linux CI
reports:

| Virtual result | Periodic IDR | Single feedback IDR |
| --- | ---: | ---: |
| Missing frame detected | frame 5 at 240 ms | same |
| Feedback arrives at sender | n/a | 260 ms |
| Next usable IDR | frame 12 | **frame 8** |
| Actually decoded frames | **29/36** | **33/36** |
| Valid runs | 0–4, 12–35 | 0–4, 8–35 |
| Delivered media wire bytes | 12,852 | 13,165 |
| Actual encoded IDRs | 3 | 4 |
| 40 ms virtual playout buffer: timely frames | 29 | **33** |

The receiver did not learn about missing frame 5 until
a later access unit arrived at 240 ms. One modeled
20 ms feedback trip reaches the sender at virtual 260 ms,
and the **first source frame strictly after that** is 8
(266.7 ms). A real encoder reset at frame 8 produces
a verified, self-contained IDR; subsequent frames decode
normally. This recovers **four extra frames** with
approximately **2.4% more admitted media bytes** in
this small, fixed source test. There is no feedback packet
bandwidth accounting, CPU encoder-reset timing or repeated
loss handling. The improvement is neither guaranteed under
other network conditions nor proof of a live feedback protocol.

#### Fluctuating link and playout latency

For the no-loss **128×72**, **30 FPS**, **48-frame** stream
under 50 ms base delay, ±20 ms jitter and piecewise
2,500 → 130 → 2,000 kbps, actual H.264 decoding
succeeded for 48/48 frames. The simplified virtual
fixed-deadline presentation model produced:

| Intentionally added playout delay | Timely frames | Late decoded frames |
| --- | ---: | ---: |
| 40 ms | **8/48** | 40 |
| 80 ms | **42/48** | 6 |
| 120 ms | **48/48** | 0 |
| 180 ms | 48/48 | 0 |
| 250 ms | 48/48 | 0 |

A fixed 120 ms delay hides the modeled jitter for these
frames, at the expense of adding 120 ms to the source-to-
scheduled-display age. A shorter 40 ms buffer keeps a
fresher target deadline but drops most frames from
their intended presentation slots. Neither approach
can reconstruct a missing H.264 reference frame.
This simplified model does not simulate a true OS
display pipeline, receiver clock recovery or
interactive input response.

### Two H.264 keyframe policies

- **Periodic:** encode real synthetic H.264 with fixed closed GOP and
  verify valid decoded runs using v23's IDR re-synchronization behavior.
- **Single feedback IDR:** the first missing frame is detected by the
  receiver only **when a later access unit arrives** (no oracle
  knowledge of a missing packet). A modeled one-way control delay
  (`--feedback-ms`) determines when the encoder learns of the gap.
  The next frame **strictly after** feedback arrival starts a
  new real libx264 encoder instance, producing a self-contained
  IDR with parameter sets. After the restart the GOP counter
  resets. Actual altered H.264 packet sizes are sent through the
  simulated link again; the bytes and delivery times of all
  frames before the decision are required to match the
  periodic counterpart.

This **single-request counterfactual** is causal for the first
detected gap because the pre-request source and link prefix
must match exactly. It is **not** a production keyframe-request
control loop, RTCP PLI/FIR implementation, or robust recovery
strategy for repeated erasures. Resetting an encoder is also
not equivalent to requesting an IDR from an already-running
encoder: CPU initialization time, rate-control resets and
encoder control behavior must be measured separately.

### Fixed-deadline virtual playout

Given *actually decoded* frames and virtual link arrivals, each
source frame `n` has a scheduled presentation slot at

```text
source_time(n) = n * (1000 / source_fps) milliseconds
deadline(n) = source_time(n) + playout_buffer_ms
```

A correctly decoded frame arriving by its deadline is eligible
to be shown once in the model. An undecodable, missing, or
late frame cannot appear in its intended slot; the player
holds its last good frame (or shows an initial blank until the
first valid frame). Delayed decoded frames are **dropped from
display**, but *not* from the H.264 decoder's reference chain.
The virtual output records on-time unique frames, late decoded
frames, undecodable frames, held/blank slots and longest
unfresh-frame run for each requested buffer depth. The
buffer-depth setting is **intentional latency**, not a codec
cost estimate or a measured real GUI frame time.

Compare the same impairment conditions using increasing
playout buffers. A larger buffer can improve regularity but
unavoidably adds to source-to-intended-display latency. It
cannot reconstruct missing references or repair a damaged
GOP. This simplified model uses fixed source-time deadlines,
not an actual playout-clock PLL, decoder scheduling pipeline,
RTP/UDP stack, FEC or adaptive bitrate algorithm.

### Reproduce locally on Windows (PyAV + NumPy already installed)

```powershell
git pull --ff-only origin feature/interactive-bridge-prototype
python -m pytest -q tests/test_bridge_video_feedback.py

# Compare periodic GOP with a simulated receiver request after frame 5 is erased
python -m colabcola.bridge_video_feedback `
  --width 320 --height 180 --fps 30 --frames 48 --gop 12 `
  --delay-ms 40 --jitter-ms 0 --bandwidth-kbps 0 `
  --max-inflight 12 --max-age-ms 0 --drop-seqs 5 `
  --feedback-ms 20 --buffers-ms "40,80,120,180"

# Compare fixed playout delays under fluctuating bandwidth, no erasures
python -m colabcola.bridge_video_feedback `
  --width 320 --height 180 --fps 30 --frames 48 --gop 12 `
  --delay-ms 50 --jitter-ms 20 `
  --bandwidth-steps "0:2500,300:130,850:2000" `
  --max-inflight 8 --max-age-ms 250 --drop-seqs "" `
  --buffers-ms "40,80,120,180,250" --seed 7
```

Both scenarios perform genuine local H.264 encoding/decoding but
simulate media and feedback delivery with virtual time. **No Colab
allocation, GPU rendering, real network traffic, public socket or
external streaming service is involved.** Only two temporary
encoder/decoder test paths run locally. Model outputs include
actual reference-chain validity, changed keyframe count and
admitted wire bytes, so benefits and overhead can be compared.
Provider policy and licensing review remain prerequisites for
any future production/cloud streaming implementation.

## v24 Windows verification and v25 multi-gap feedback research

### Windows v24 result (local, deterministic virtual link)

On Windows, the user ran **11/11 v24 feedback tests** successfully
and independently reproduced both modeled outcomes:

- For the 36-frame / 30 FPS / GOP-12 stream with access unit 5
  erased, periodic IDRs decoded **29/36** frames, while a single
  feedback-triggered IDR at frame 8 decoded **33/36** frames.
  The longest virtual interval with no fresh frame fell from
  **233.33 ms to 100.0 ms**. Encoded media bytes increased from
  12,852 to 13,165. A single missing P-frame caused dependency
  invalidation until the next verified IDR.
- With no dropped access units but fluctuating virtual bandwidth,
  48/48 frames decoded successfully. Virtual fixed-deadline playout
  presented **8/48** frames on time with a 40 ms buffer,
  **42/48** with 80 ms, and **48/48** with 120 ms or more.
  The virtual arrival-time median was 53.28 ms.

These are exact outputs of the local virtual-time model, not measured
physical network/GUI properties.

### v25: causal multi-request IDR controller (experimental)

The opt-in `src/colabcola/bridge_video_controller.py` advances the
same bounded link in *source-frame time* and allows the receiver to
observe **only access units whose virtual arrival has occurred**.
The producer encodes **real H.264 one frame at a time**. A receiver
notices a sequence gap when a later packet arrives and sends a virtual
feedback request; only after its configured one-way delay may the
sender begin a fresh closed-GOP libx264 encoder at the next frame.
This can happen **multiple times** when independent losses occur.

The state machine includes a minimum retry interval to avoid
request storms, a strict upper bound on issued requests, an
explicit limit on in-flight encoded packets, and the original
link's bandwidth/jitter/loss/age admission rules. A lost
feedback-generated IDR can cause the receiver to request another,
after its retry interval. The first segment *before any applied
feedback* is required to be byte-for-byte identical to the periodic
baseline. If there are no controller actions, the entire source
bitstream and virtual-link trace must match the baseline exactly.

Report fields distinguish **detected gaps**, requests made at the
receiver, virtual feedback delivery time, source IDR restarts
actually applied, requests that arrived too late for the bounded
source, decoder-verified valid frame runs, admitted bytes, and
fixed-depth virtual playout outcomes. A request can only be made
after a received frame exposes a loss; a final lost frame with
no later packet may remain undetected.

The first scenario has now been verified by cross-platform CI:
**140 passing tests on Ubuntu** and **134 passed / 6 skipped
on Windows** (the skips require an external FFmpeg executable).
With a real synthetic **60-frame H.264 stream, 30 FPS, GOP 24**
and explicitly erased media frames **5 and 27**, the virtual
receiver detected both gaps only on arrival of a subsequent
access unit. Feedback requests were generated at virtual times
**240.0 ms** and **973.333 ms**, reached the source at
**260.0 ms** and **993.333 ms**, and caused IDRs at frame
indices **8 and 30**, respectively. Both new keyframes
were verified through real H.264 decoding.

| v25 modeled outcome | Periodic GOP 24 | Two receiver-feedback IDRs |
| --- | ---: | ---: |
| Frames actually decoded | **20/60** | **54/60** |
| On-time unique frames at 40 ms fixed playout | 20/60 | **54/60** |
| Media wire bytes | 20,465 | 20,527 |
| Encoded IDRs | 3 | 4 |
| Lost access units | 2 | 2 |
| Applied feedback requests | 0 | 2 |

There are **34 more valid decoded frames** than with periodic
keyframes alone in this deliberately challenging small example.
The media-byte difference (62 bytes) is content-dependent,
and it excludes feedback-message bytes and encoder-reset CPU
cost. Real link timings, congestion feedback, and physical
presentation remain unmeasured.

The second verified Linux CI scenario had a piecewise bandwidth
profile of 2,500 → 130 → 2,200 kbps, ±15 ms seeded jitter,
bounded queue 6 and max modeled frame age 200 ms. It delivered
and decoded **60/60 frames without any requested IDRs**;
a static 40 ms playout buffer met 8 deadlines, 80 ms met 57,
and 160 ms or 240 ms met all 60. This reinforces the
separation between reference-chain recovery and jitter-buffer
tradeoffs.

Streaming adaptation remains limited to
**IDR recovery**: the H.264 CRF/bitrate is fixed and the playback
buffers remain a set of fixed comparisons. A rate controller and
adaptive buffer clock are separate future experiments; neither
should be inferred from this implementation.

```powershell
git pull --ff-only origin feature/interactive-bridge-prototype
python -m pytest -q tests/test_bridge_video_controller.py

# Two independent losses; the receiver can issue two causal IDR requests
python -m colabcola.bridge_video_controller `
  --width 128 --height 72 --fps 30 --frames 60 --gop 24 `
  --delay-ms 40 --jitter-ms 0 --bandwidth-kbps 0 `
  --max-age-ms 0 --max-inflight 16 --drop-seqs "5,27" `
  --feedback-ms 20 --retry-ms 100 --buffers-ms "40,80,120"

# No explicit erasure, but a bandwidth outage and finite queue
python -m colabcola.bridge_video_controller `
  --width 128 --height 72 --fps 30 --frames 60 --gop 24 `
  --delay-ms 50 --jitter-ms 15 `
  --bandwidth-steps "0:2500,400:130,1100:2200" `
  --max-inflight 6 --max-age-ms 200 --drop-seqs "" `
  --retry-ms 150 --buffers-ms "40,80,160,240"
```

**Scope and provider constraints:** the v25 controller runs entirely
inside a local test process on virtual time. It uses real PyAV/libx264
encoding and decoding but no real socket, GPU, Colab API, cloud
allocation or Tk viewer, and it makes no claims about round-trip
network performance, encoder-reset wall time, retransmission/FEC,
loss of IP packet fragments, or gameplay input-to-photon response.
PyAV/NumPy remain optional development dependencies. The
existing authenticated Colab bridge and batch runner are unchanged.

## v25 Windows confirmation and v26 causal adaptive H.264 rate experiment

### Windows v25 reproduction (virtual media/feedback)

The user ran `tests/test_bridge_video_controller.py` on Windows:
**8/8 tests passed**. A locally encoded 60-frame H.264 stream at
30 FPS, GOP 24, with explicit access-unit erasures at frames 5
and 27 reproduced two feedback requests, applied at IDRs **8 and
30**. **54/60 frames** reconstructed under multi-feedback,
compared with **20/60** for periodic GOPs alone; longest
stale-frame run fell from **700 ms to 100 ms** in the virtual
fixed-deadline playout model.

The no-erasure bandwidth-outage experiment decoded **60/60**
frames under the fixed-rate source. A 40 ms virtual buffer
was on time for **8/60**, an 80 ms buffer **57/60**, and
160/240 ms buffers **60/60**. No actual network latency or
screen-presentation timing was measured.

### v26 verified results and limitations (cross-platform CI)

The opt-in adaptive encoder, causal feedback state machine and
actual RGB quality tests pass on both platforms: **147 tests on
Ubuntu**, and **141 passed / 6 external-FFmpeg-related skips on
Windows**. The integrated 90-frame / 30 FPS / 128×72
textured-scene experiment was run through a deterministic link
whose virtual bandwidth changed from 1,200 to 90 and back to
1,200 kbps. All frame sizes came from actual libx264 encoding.

| v26 virtual-link outcome | Fixed target 650 kbps | Adaptive target 650/95 kbps |
| --- | ---: | ---: |
| Admitted media wire bytes | 211,788 | **193,760** |
| Link-delivered units | 54 / 90 | 54 / 90 |
| Actual decoded frames | **36 / 90** | **46 / 90** |
| Link-dropped units | 36 | 36 |
| 40 ms fixed playout: timely frames | 0 | **6** |
| 80 ms fixed playout: timely frames | 29 | **32** |
| 160 ms fixed playout: timely frames | 36 | **46** |
| Pooled RGB PSNR over each policy's decoded frames | 12.969 dB | 12.477 dB |
| RGB PSNR on exactly the same **36** decoded frame IDs | 12.969 dB | **12.585 dB** |
| Actual IDRs in encoded stream | 4 | 6 |
| Source rate switches | 0 | 4 |

The adaptive policy sent about **8.5% fewer admitted media
bytes**, while preserving **10 more decodable frames**
than the fixed high-rate stream. Its quality on the
**same decoded frames** was approximately **0.384 dB lower**.
This is a real quality/freshness tradeoff, not a claim that
the adaptive version is always better.

A first, more reactive two-level controller switched
**eight** times on this trace, exhausting its switch budget
and recovering 48/90 frames; it also met only 22 deadlines
at the 80 ms buffer. Adding three consecutive high-delay
observations before downshift, eight low-delay observations
before upshift, and a 350 ms minimum between requests
produced the four-switch result above. The more stable
policy traded two total reconstructed frames for better
80 ms on-time delivery (32 versus 22 in these independent
controller runs).

**Important causal limitation:** the adapted sender switched
at source frames **7, 18, 62 and 75**. The modeled bandwidth
outage ran between virtual 600 and 1,750 ms, but the
critical second downshift did not occur until frame 62
(about 2,067 ms), *after* the outage ended. Under
congestion, the forward media packets supplying the receiver's
evidence can be delayed or filtered out as stale. No new arrival
means no new rate decision. The reverse control path itself is
ideal and has a fixed delay; this experiment does not model
reverse-feedback congestion. This controller makes real causal decisions,
but its observed improvements should not be confused
with prompt adaptation to the outage. Its early
low-rate segment and IDR resets also affect the outcome.

This experiment deliberately uses a deterministic moving
high-entropy texture, not representative gameplay video.
PSNR computed on different decoded-frame subsets is
not directly comparable; the matched-frame PSNR is the
appropriate narrower quality comparison. The simulator's
media byte counts are transmitted/admitted access units,
not only delivered payloads. All bandwidth, latency,
timeliness and feedback times are **virtual**. The actual
encoder/decoder run locally; no GPU/Colab/real network or
physical display was involved.

### v26: receiver-feedback high/low H.264 bitrate adaptation

The separate research module `src/colabcola/bridge_video_rate.py`
extends the incremental v25 link with a **causal two-rate
encoder controller**:

- The source creates a deterministic **textured, moving RGB
  scene**, rather than the earlier highly compressible glow
  pattern. It encodes each frame with an actual in-process
  libx264 **target bitrate** and a matching VBV limit.
- A receiver observes **only delivered access units** and
  their simulated source-to-arrival ages and sequence gaps.
  If a packet is old enough or a sequence gap is observed,
  it requests a reduced bitrate. Several low-delay
  observations can trigger a recovery to the higher rate.
- The control message is delivered only after a bounded
  simulated feedback delay. A **future** source frame
  switches encoder rate by restarting a closed-GOP encoder,
  which also creates a new verified IDR. The first bytes
  before the applied rate change must exactly match the
  no-adaptation baseline.
- The policy uses hysteresis thresholds, a required number
  of good samples, minimum switch interval, a finite maximum
  number of switches, and bounded virtual queue admission
  and frame-age constraints. It does **not** peek at future
  bandwidth steps.
- In addition to reconstructed frame counts, dropped
  units, byte totals and fixed playout deadlines, actual
  decoded RGB frames are compared with the known original
  textured images, producing pooled **MSE and PSNR on valid
  decoded frames**. A good PSNR cannot compensate for
  missing frames, so both quality and continuity must
  be evaluated.

This is deliberately simpler than WebRTC's real congestion
control: just two configured rates, not a production bitrate
estimator, frame-rate selector, keyframe request signaling,
retransmission/FEC, or adaptive playout clock. A bitrate
target is **not a guaranteed exact output bit rate**;
measure actual media bytes and per-frame sizes. Encoder
restart time and packet-delivery timings are not measured
in real time.

The first comparison is a deterministic local media link at
30 FPS with a capacity drop from **1,200 kbps** to **90 kbps**
at virtual 600 ms and a recovery to 1,200 kbps at 1,750 ms.
The nominal H.264 encoder targets are **650 kbps**
(high) and **95 kbps** (low). The receiver has no access
to this schedule; it can act only on packet arrivals.

```powershell
git pull --ff-only origin feature/interactive-bridge-prototype
python -m pytest -q tests/test_bridge_video_rate.py

python -m colabcola.bridge_video_rate `
  --width 128 --height 72 --fps 30 --frames 90 --gop 24 `
  --delay-ms 40 --jitter-ms 8 `
  --bandwidth-steps "0:1200,600:90,1750:1200" `
  --max-inflight 10 --max-age-ms 350 `
  --high-kbps 650 --low-kbps 95 `
  --congested-age-ms 110 --recovered-age-ms 65 `
  --control-delay-ms 35 --min-switch-ms 350 `
  --buffers-ms "80,160,240,350" --seed 7
```

Inspect `controller.requests` and `controller.applied`
separately: a delayed request may never take effect before
the end of the source trace. Compare `fixed_high_rate`
against `adaptive_rate` for actual wire bytes,
`decoded_runs`, playout deadline compliance,
and PSNR on successfully reconstructed frames.
Do **not** assume adaptive rate is better under every
network scenario or that a higher bitrate means higher
quality after packet loss.

**Scope:** this is a seeded virtual-time complete-access-unit
link and two in-process PyAV codecs; no Colab resource,
GPU, real transport, GUI, physical display measurement,
or released batch-runner change. PyAV/NumPy remain optional
development-only dependencies and production use requires
separate provider-policy and binary licensing review.

## v27: receiver-silence feedback and restart attribution

The optional `bridge_video_watchdog.py` experiment addresses the
v26 observation gap: a receiver can now request the low encoder
target when no usable media packet has arrived for a configured
interval. Its timer does not inspect the link's bandwidth profile,
drop decisions, queue contents, or future arrivals. A request still
waits for the fixed reverse-path delay and can only change a future
source frame. The original `bridge_video_rate` command leaves this
timer **disabled by default** (`--silence-timeout-ms 0`).

The four-arm comparison measures:

1. Fixed high target, with the normal periodic GOP.
2. v26 feedback driven only by arriving packets.
3. The same feedback with a receiver silence timeout.
4. Fixed high target, with exactly the timeout run's encoder restart
   indices and verified IDR schedule. This is an **offline ablation**
   conditioned on another run, not an independently causal controller.

The source pixels, link parameters and seed are shared. The simulator
draws random jitter/loss per admitted unit, so different admissions
can assign draws to different source frames. A second zero-jitter,
zero-random-loss run removes that source of ambiguity. All reported
matched PSNR values use the same frame IDs decoded by **all four**
arms; the JSON records that intersection explicitly.

### Result: earlier action does not yet improve video continuity

These retained local measurements use PyAV 19.0.2, real libx264
encoding/decoding, 128x72 source images, 90 frames at 30 FPS, GOP 24,
650/95 kbps targets, 350 ms request dwell, and a 150 ms receiver
silence timeout. The virtual capacity changes from 1,200 to 90 kbps
at 600 ms and returns to 1,200 kbps at 1,750 ms.

With the historical 40 ms base delay and +/-8 ms jitter, the
outage-related rate reduction moves from source frame **62
(2,066.667 ms)** to frame **28 (933.333 ms)**. The receiver's last
usable pre-outage arrival is at 616.490 ms; the silence timer
expires at 766.490 ms, but the request dwell holds it until
889.425 ms. Its 35 ms reverse delay makes the request available at
924.425 ms, strictly before source frame 28.

| Historical virtual-link result | Fixed high | Arrival feedback | Silence feedback | Fixed high, matched restarts |
| --- | ---: | ---: | ---: | ---: |
| Admitted media wire bytes | 211,788 | 193,760 | 181,834 | 257,541 |
| Actually decoded frames | 36/90 | 46/90 | 39/90 | 39/90 |
| On time with 40 ms buffer | 0 | 6 | 4 | 0 |
| On time with 80 ms buffer | 29 | 32 | 26 | 20 |
| On time with 160 ms buffer | 36 | 46 | 39 | 39 |
| Encoded IDRs | 4 | 6 | 6 | 6 |
| RGB PSNR on the same 36 frames | 12.969 dB | 12.585 dB | 12.698 dB | 13.025 dB |

The watchdog sends fewer bytes but reconstructs **seven fewer frames
than arrival-only feedback**. The timer run reconstructs frames
0-17 and 69-89; arrival-only feedback reconstructs 0-17 and 62-89.
Earlier downshifting resets the GOP while the link is still impaired;
it does not guarantee that a usable IDR reaches the decoder. The
`idr_delivery` evidence shows the watchdog's IDRs at frames 28 and 52
both rejected at queue admission. Its next usable IDR is frame 69;
arrival-only feedback instead generates a usable IDR at frame 62.
The current rate policy has no independent IDR retry while it remains LOW.

The zero-jitter comparison makes the attribution limitation clearer:

| Zero-jitter result | Fixed high | Arrival feedback | Silence feedback | Fixed high, matched restarts |
| --- | ---: | ---: | ---: | ---: |
| Admitted media wire bytes | 211,788 | 221,881 | 208,025 | 258,052 |
| Actually decoded frames | 36/90 | 46/90 | 39/90 | 39/90 |
| On time with 80 ms buffer | 29 | 34 | 27 | 27 |
| RGB PSNR on the same 36 frames | 12.969 dB | 12.983 dB | 13.021 dB | 13.021 dB |

Here the watchdog lowers the rate at frame **25 (833.333 ms)**,
while arrival-only feedback still waits until frame 62. Yet the
watchdog and matched-restart high-rate stream have identical decoded
runs, 80 ms deadline counts and matched PSNR. The rate change saves
50,027 admitted bytes relative to that restart control, but provides
**no observed continuity improvement in this scenario**. The 95 kbps
low target also exceeds the 90 kbps bottleneck before framing overhead;
it is not a sustainable-rate estimate. These results do not establish
that a silence timeout improves perceived quality.

**Decision:** keep the timer opt-in and retain v26 as the comparison.
The next distinct improvement should separate decoder recovery from
rate selection: request/retry an IDR after lost references even if
the source is already LOW, then compare against the same restart
control. A future capacity estimator must also account for actual
encoded bytes and framing overhead.

### Reproduce and inspect the evidence

```powershell
Set-Location C:\dev\colabcola-bridge-prototype
git pull --ff-only origin feature/interactive-bridge-prototype

python -m pytest -q tests/test_bridge_video_rate.py tests/test_bridge_video_watchdog.py

# Same impairment profile as v26, with all four comparison arms.
python -m colabcola.bridge_video_watchdog `
  --silence-timeout-ms 150 --buffers-ms "40,80,160,240,350"

# Remove random jitter to clarify the restart-versus-rate comparison.
python -m colabcola.bridge_video_watchdog `
  --jitter-ms 0 --silence-timeout-ms 150 `
  --buffers-ms "40,80,160,240,350"
```

Inspect `controllers.silence_feedback.requests` for
`trigger: "receiver_silence"`, `controllers.*.applied` for actual
source changes, and `restart_control` for the ablation schedule.
The source bytes and already scheduled arrivals must remain identical
until the two feedback policies first take different actions.

Full measured outputs and codec versions are retained in
[`experiments/v27/historical.json`](experiments/v27/historical.json),
[`experiments/v27/zero-jitter.json`](experiments/v27/zero-jitter.json), and
[`experiments/v27/environment.json`](experiments/v27/environment.json).
All initial and restarted encoder segments now verify their first
packet is an IDR, including low-to-high rate changes.

The timer arms only after the first usable packet; total startup
loss remains undetected. Silence clears the good-sample recovery
streak, including when already LOW. Packet arrivals win exact ties
with timer deadlines. Timers stop after the final source frame;
tail arrivals still drain, and late commands cannot change past frames.
Reverse feedback remains ideal, reliable and fixed-delay, with no
reverse bandwidth or feedback-byte model. These are local deterministic
experiments, not real-network or physical input-to-photon measurements.

**Correction to the earlier v26 write-up:** the 29/32 on-time frame
counts belong to an **80 ms** buffer, not 40 ms. Re-running the original
v26 code confirms 0/6 at 40 ms. The table above and v26 table have been
corrected; the full v26 reproduction command now uses the documented
350 ms dwell rather than the earlier 150 ms example.

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
