# v37C — sender-side evidence and fail-safe Input Lab producer shutdown

**Status: offline-first repair; no new Colab paid run until Windows/Ubuntu CI and local Windows suite pass.**

## Evidence from v37B run

Input Lab run ID `759f2811ab0c464d89b027d973a21e99`, remote root `/content/colabcola-bridge/<run_id>`, was supervised 30 Hz / C6, with a ten-minute maximum. Windows reported a video freshness watchdog failure after ~190 s. Host Tk displayed 3004 frames, logged 420 queue-coalesced frames, and its terminal journal marked host detached. This does **not** confirm remote producer shutdown.

A subsequent read-only remote notebook inspection showed:
- `status.json`: state **ready**, application inputlab, T4
- `render-metrics.json`: window 567, seq 15418, produced 28 frames over 1031.544 ms = **27.14 FPS**
- `frame.bin`: last seq **15443**, input ACK **3904**
- all eight ring slots present and recently updated; **no `stop` marker**

The remote producer was still making frames long after host video delivery stopped, so the failure cannot be attributed to a crashed Pillow worker. The particular sender/transport fault remains unknown: possible credit exhaustion, blocked `comm.send`, sender failure, or Jupyter IOPub disruption. Do not assert a cause without sender evidence. The resulting worker leak is a real lifecycle bug, because an apparently successful host disconnect does not imply a remote worker stop.

## v37C changes (Input Lab only)

1. **Authenticated host credit lease.** The Input Lab producer records only validated video-credit grants handled by the existing kernel Comm. It allows at most 120s without any initial credit (permitting slow legitimate bootstrap), then stops after **20s without a subsequent credit**. Neither arbitrary user input nor frame creation refreshes the lease. `status.json` reports `stopped` with `reason=host_credit_lease_expired`. Normal stop-file shutdown and thread cleanup remain supported.
2. **Independent shutdown marker.** After a host Input Lab session, the host sends the existing authenticated Comm close and also issues a one-shot, run-ID-specific `PUT /api/contents/content/colabcola-bridge/<run_id>/stop` through the already authenticated provider contents API. The marker is the same `b"1"` polled by the renderer, so it works even when Jupyter shell→Comm delivery has failed. No new remote listener, no arbitrary paths, no GPU allocation/release and **no automatic replay of an uncertain PUT**. A failed PUT leaves the external journal at `needs_manual_review`, never falsely claiming a verified shutdown. On success, the journal records `remote_stop_marker_uploaded=true`, explicitly a stop **request**, not proof of worker exit.
3. **Sender evidence sidecar.** Only Input Lab's credit sender receives `enable_status_sidecar=True`. It writes `push-sender-status.json` at most once per second, plus terminal updates: state active/stopped/failed, counts sent/replenished, bounded available credits, initialized grant status, last renderer sequence, sanitized failure phase (`source_pull` or `comm_send`), and a **remote-only** monotonic timestamp. It never includes secrets or exception strings. If `comm.send` blocks indefinitely, a stale `active` status timestamp is evidence of the stall.
4. **Reduce input lock contention.** Snapshot the small Input Lab state under its lock before drawing. Pillow drawing and zlib compression run outside the lock, allowing the kernel input and credit callbacks to proceed while an expensive frame renders. The snapshot still determines both frame contents and input ACK, preserving the established CCB1 semantics.
5. **Remote-only metrics.** Existing `render-metrics.json` gains optional `input_lab_stage` and `input_lab_credit_count` to correlate late-stage rendering with sender stalls. These are diagnostic extras, not protocol authorization. The source's existing 30Hz frame ring, H.264 encoder, credit bounds and no-socket rules are unchanged. Other applications and the default synthetic mode preserve their current behavior.

## Acceptance gates

- Cross-platform offline tests: initial and active lease expiry, sender sidecar counts and sanitized send errors, remote self-stop, independent exact-run stop marker (positive/negative), preserved legacy sender behavior, preserved frame ACK semantics, full bridge regression.
- On the user's Windows machine: fast-forward only the Input Lab feature branch with clean tracked worktree, run focused and complete `tests/test_bridge*.py` regression and pinned provider integrity; do not modify or remove untracked v32 receipts.
- Only after green CI and local checks: explicitly connect a disposable browser-owned T4, confirm one assignment/kernel read-only, verify Pillow and use a **fresh** bounded Input Lab session. Keep 30 Hz / C6. **Do not** increase credit window or extend the host freshness watchdog as a workaround.
- Observe pointer hiding/recentering and Escape release, stage progression, voxel rendering, final wipe. On success/host failure, inspect `status.json`, `push-sender-status.json`, `render-metrics.json` and `stop` marker from **that exact run ID**. The GPU remains externally owned: disconnect manually in the Colab browser.
- If the sender status remains active but stops advancing while producer metrics still update, investigate `comm.send`/IOPub. If sender available credits=0 while producer continues, investigate host credit return/shell delivery. If sender state failed, inspect only its sanitized error phase; do not surface raw kernel tracebacks or tokens.

## Existing orphan recovery

If the original v37B notebook kernel is still connected, the user can explicitly create only its run-owned stop marker after validating `status.json` run ID and application, then confirm `status.json` becomes stopped. This does not release the T4 or terminate unrelated processes. The v37C fix applies only to **future** sessions; updating the Windows Git branch does not retroactively alter already executing remote code.
