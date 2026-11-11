# v34C — sender-driven H.264 credit-window experiment (offline first)

## What the first healthy v34B run actually showed

The user-owned browser T4 ran a complete 108.50-second sender-driven H.264 session under feature commit `485100c`. The Unicode Tk telemetry crash was fixed. At credit capacity 4 the host decoded **2,679 distinct renderer frames**, or **24.691 FPS**, while independently reported producer windows had median **29.69 FPS**. The host saw **511 skipped renderer sequence IDs** (4.71/s), and its frame-arrival interval p50 was 37.66 ms (maximum 209.73 ms). Median H.264 encoding took 3.519 ms. Tk configured images at approximately 24–25 FPS over the reporting windows, typically taking 2.2 ms for image+label; local host-receipt to Tk-configure was around 11–12 ms. The median input callback→host receipt was 221.03 ms; callback→decode complete was 221.72 ms, and the median of the periodic input callback→Tk p50 windows was approximately 233 ms, **not an aggregate per-event percentile**.

The journal reported external ownership and `host_disconnected`, and the process exited zero. Browser runtime disposal remains the user's responsibility.

At 30 FPS with ~125 ms median request RTT (from earlier **pull-mode** experiments), four concurrent credits are only just enough by `FPS × RTT ≈ 3.75`. The push path's **credit return round trip has not been measured**. The observed 24.691 FPS would correspond to approximately 162 ms per four-credit round trip **if** the window were continuously full; that is a diagnostic hypothesis rather than measured network latency. Raising credits might remove a credit bottleneck, but could increase end-to-end latency if in-flight video queues accumulate.

## Controlled v34C modification

Add `--push-credits {4,6,8}` to **external-only sender-driven video**; default **4** preserves v34B. One configured immutable credit capacity is forwarded from host CLI through authenticated kernel bootstrap to `CreditPushSender` and back into the host `run_push_comm_h264` controller. A push session must explicitly opt in to `--attach-existing --sequence-buffered --push-video`; other modes reject nondefault capacity. Per-frame host `video_pull` remains forbidden.

The server enforces:
- An initial grant **equal to configured capacity**, once.
- Every subsequent grant **exactly one** per previously sent frame; repeated/unsolicited returns rejected.
- Available plus unacknowledged sent-frame credits **never exceed configured capacity**.
- No provider allocation, release, kernel restart, network listener, or retry of ambiguous video messages.

The existing eight-slot source ring stays fixed. `--renderer-lag-frames 1` stays the target for a future controlled live comparison; do not change both lag and capacity in the same trial. New acceptance tests cover 4/6/8 on both sides, invalid/bool/out-of-range capacities, and CLI gating, plus prior H.264 sequence/ACK invariants.

## Decision test and limitations

The first *optional supervised* live run after full GitHub Windows+Ubuntu CI, local regression, pinned provider integrity, and read-only external T4 preflight should use six credits. Compare it with **v34B four credits** under the same 30 FPS target and one-frame selection lag. Capture:
- Independent CUDA producer FPS (expected around 29.7 if unchanged)
- Host decoded distinct FPS and Tk configured FPS
- Input callback→host receipt/decode and input callback→Tk configuration (report observation counts, use same measurement point)
- Total skipped source IDs, median/max host inter-frame gaps, no-new counts, provider GPU ownership

Only test eight credits if six demonstrates credible throughput improvement and no unacceptable latency regression. If 6/8 does not improve, stop increasing credits and instead instrument remote credit starvation, per-frame `comm.send` duration, and Jupyter IOPub backlog.

The current `input callback→Tk` measurement is a *host-clock response time* to an applied input reported in a frame; it is not synchronized remote capture latency and not physical monitor scanout. v33w input ACK measurements are not directly comparable to this end-to-end-ish metric.

No live Colab session was used to author this commit; do not use `bridge cleanup` with external journals. User must manually disconnect the disposable T4 when finished.
