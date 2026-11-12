# v33w — bounded freshness-versus-input-latency experiment

## Motivation: v33v succeeded, with a latency trade-off

The opt-in eight-slot ring and `video_no_new` protocol eliminated all duplicate renderer snapshots in a complete browser-owned T4 experiment. v33v C4/C8/C16/C32 distinct delivered rates were 28.75/29.0/29.125/29.125 FPS compared with v33u 27.0/26.375/26.25/27.125. Independent production remained approximately 29.6–29.8 FPS. Every measured v33v phase reported median **2 producer frames of selected-buffer lag**, approximately **67 ms at 30 FPS**.

Observed input ACK p50, however, worsened: C4 163.44 → 229.75 ms, C8 167.91 → 218.53 ms, C16 164.88 → 222.81 ms, and C32 158.55 → 223.90 ms. This is consistent with consuming older renderer snapshots, but is **not a controlled causal input-to-visible-pixel study**: both test runs used live, unscripted viewer interactions, and observed ACK excludes physical display scanout.

## Small offline-only change

This feature adds `--renderer-lag-frames {0,1,2}` to the opt-in, user-owned `--attach-existing --sequence-buffered` 30 Hz H.264 capacity mode. **Default remains 2**, preserving the previously verified v33v behavior. No managed allocation, vendor integrity, Jupyter kernel ownership or H.264 codec settings change.

The producer still maintains eight atomic snapshot slots. The source selects the oldest available distinct frame not more than the configured `N` sequence positions behind the newest (where `N = 0,1,2`). A smaller bound may reduce snapshot age but also reduce fresh delivered FPS or cause correlated `video_no_new` replies. H.264 coded sequence remains contiguous across source-sequence skips. The selected lag and skipped frame count remain in per-phase metrics; the run report additionally records `configured_renderer_lag_frames`. Buffer slots are not changed or dynamically resized.

Safety: only external, explicitly confirmed attach may enable this switch. Out-of-range values and nonbuffered use with non-default lag are rejected before remote action. No module is imported from a new untrusted source, no network listener is added, and external Colab runtime termination is still a user action.

## Offline acceptance before any live trial

Run the new `tests/test_bridge_video_v33w_freshness_policy.py`, the preceding v33v tests, the entire `tests/test_bridge*.py` regression, the pinned provider integrity check, and both GitHub Actions platforms. Do not run a GPU experiment if any gate fails.

## Supervised lag=1 A/B

After validation, attach to a **fresh disposable browser-provisioned T4** with exactly one existing kernel and read-only preflight. Execute at most one normal seven-phase capacity run using `--sequence-buffered --renderer-lag-frames 1 --confirm-attach` and a new run directory; skip `--install-remote-pyav` if already installed in the disposable kernel.

Compare v33w lag=1 against v33v lag=2 **at C4/C8/C16/C32**. Report independently produced FPS, decoded/distinct FPS, input ACK p50, selected producer-frame lag, skipped-frame count, and no-new responses. A desirable result improves input responsiveness without forfeiting the 29+ distinct FPS achieved by v33v. If performance degrades, retain lag=2 as an explicit throughput-oriented option and design a **controlled input marker-to-visible-frame** measurement before making stronger latency claims. User must manually terminate the browser-owned runtime.

Tracking: https://github.com/dependanz/colabcola/issues/21 .
