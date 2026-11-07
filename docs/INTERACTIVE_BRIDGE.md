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

### v21 optional Tk polling interval A/B (not yet benchmarked)

The local live-H.264 demo now supports
`--viewer-poll-ms N` with **N between 5 and 100 milliseconds**.
The existing Tk default stays **30 ms** for all other viewers,
including the managed Colab viewer. A smaller interval polls the
bounded decoded-frame queue more frequently; it might lower
`decode→Tk` and `input callback→Tk` latency at the cost of
more frequent GUI wakeups. No improvement is claimed until measured.

Compare **two separate local runs** with the same scene size and
30 FPS target. Move the mouse comparably throughout both, and
compare the reported GUI frame rate, decode-to-Tk median and
input-callback-to-Tk median:

```powershell
python -m pytest -q tests/test_bridge_video_live.py tests/test_bridge_viewer_stats.py

# Existing viewer schedule (30 ms)
python -m colabcola.bridge_video_live --show --width 640 --height 360 --fps 30 --max-seconds 30 --viewer-poll-ms 30

# Experimental faster polling (8 ms)
python -m colabcola.bridge_video_live --show --width 640 --height 360 --fps 30 --max-seconds 30 --viewer-poll-ms 8
```

This experiment is **local only**, and no GPU, Colab runtime,
network listener, provider API, or product dependency is added.
A reduction in widget-configuration latency is *not* an
input-to-photon improvement claim.

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
