# v35B — attribution of producer, sender credit flow, and Tk coalescing (offline first)

**Status:** v35A achieved 47.216 delivered FPS (target 60) on one user-owned T4. v35B is instrumentation only; no live run is claimed by this commit. Keep v34C 30Hz/C6 as the live-known-good fallback, and preserve the external-only 60Hz/C8-C10-C12 safety envelope. The 60Hz/C10 benchmark remains the first choice; don't increase to C12 without evidence of credit starvation.

## Source of the observed deficit

- Live v35A 60Hz/C10: 5,421 decoded distinct frames / 114.813 s = **47.216 FPS**; remote producer p50 of 114 one-second FPS windows = **50.84 FPS** (not a session mean); 46 skipped renderer sequence IDs; 4,958 Tk-configured images / 114.828 s = **43.178 FPS**. Host arrival gaps p50/p95/p99 = 18.27/55.83/74.04 ms; callback-to-Tk p50/p95/p99 = 168.87/224.24/259.39 ms, n=1,882. Journal detached cleanly; external runtime was not released.
- A median producer-window FPS and a session-average decoded FPS are **not directly subtractable** to infer transport loss. Likewise, decoded minus configured frames do **not** by themselves establish how many frames were dropped due to GUI queue coalescing.

## Added measurements and precise meanings

1. **Weighted observed producer FPS:** sum of `published_count` over the distinct one-second sidecars actually seen by the host, divided by the sum of their `window_elapsed_ms`. Also report `producer_frames_covered` and `producer_elapsed_ms_covered`. This is a weighted rate over **observed windows only**, not necessarily all wall-clock time in the run.
2. **Producer pipeline phases:** every one-second producer window includes `render_copy_ms`, `compress_ms`, `publish_ms` (each is the *within-window median*) and new `*_p95_ms` fields (the *within-window interpolated 95th percentile*). The host reports `producer_{phase}_window_median_p50_ms` and `..._p95_ms` (p50/p95 **across the series of within-window medians**), plus `producer_{phase}_within_window_p95_median_ms` (median **across the observed one-second p95 values**). **None of these is a pooled per-frame producer p95**. Sidecars without new fields remain valid.
3. **Remote read and encode:** `remote_source_read_p50/p95/p99_ms` and `remote_encode_p95/p99_ms` are pooled from the received per-access-unit remote durations. Existing `remote_encode_p50_ms` is retained. These are remote duration measurements and must never be formed by subtracting a host timestamp from a remote timestamp.
4. **Credit wait:** with the 60Hz push sender only, the remote sender accumulates time spent waiting on its condition variable **after the initial credit grant**, when no credit was available at entry. Each encoded reply carries a cumulative `sender_credit_wait_us_total` and wait-iteration count. Host reports the latest observed cumulative totals (not necessarily final lifetime totals). This tracks credit-wait blocking time, not a direct end-to-end credit RTT; repeated 50ms condition timeouts can create multiple wait iterations within one starvation episode. A sender that is still encoding/sending does not count as waiting.
5. **Comm send:** each 60Hz encoded reply carries the duration of the **previous completed** `comm.send` (if any). The host pools these durations into p50/p95/p99; the first and last sends can be underrepresented. This is time inside the sender's Comm API call, **not wire delivery or Jupyter IOPub queueing time**.
6. **GUI queue loss:** `gui_queue_coalesced_video_frames` counts actual video frame items discarded by the bounded two-slot host-to-Tk queue in `offer_update` (including a rejected incoming video item when the queue remains full). It excludes evicted status messages and does not conflate counted coalescing with other decoded-to-Tk differences.

The sender sends no extra Comm message; diagnostics attach only to its existing 60Hz **video_access_unit** payload. `video_no_new` contract and 30Hz sender payloads remain unmodified. Optional invalid diagnostics are ignored by the host and must not affect decoding, credit grants, or identity/sequence validation.

## Offline acceptance before a new paid T4 experiment

- CI must pass full `tests/test_bridge*.py` on Windows and Ubuntu.
- On local Windows feature checkout, verify commit, pinned provider integrity, focused `tests/test_bridge_video_v35b_attribution.py` with v34/v35 regression, full `tests/test_bridge*.py`, and untracked v32 evidence intact.
- Use the `high-credit-dev` account and **one new disposable user-owned T4**, read-only preflight, then one explicitly confirmed 60Hz/C10/one-frame-lag run for at most 120 seconds. The receiver should report the weighted observed producer rate, the source phase profiles, sender wait/previous-send durations, and explicit Tk coalescing.
- Exercise a controlled input pattern, preserve JSON summaries and sanitized logs, compare like-for-like. Do not assert physical input-to-photon or infer true IOPub queue depth from `comm.send` blocking.
- After a run, only the user disconnects the browser-owned T4. The external bridge never allocates/releases it; do not use managed `bridge cleanup`.

## Decision gate

- If weighted producer FPS remains below 60 and phase timings are high, profile the remote CUDA-to-CPU/zlib/file publication path before changing credits.
- If producer approaches 60 but sender credit wait is substantial, benchmark a carefully controlled 60Hz/C10 vs 60Hz/C12 A/B without increasing unbounded queueing.
- If sender and receiver approach 60 but Tk coalescing is significant, optimize only host decode/queue scheduling or GUI display, preserving freshness.
- If video gaps or p95/p99 spacing remain bad with healthy average FPS, investigate Jupyter IOPub queueing with additional **new** instrumentation rather than treating `comm.send` duration as its direct measurement.
