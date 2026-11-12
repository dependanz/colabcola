# v33s — independent CUDA renderer producer windows

**Base:** GitHub feature branch after the verified synchronization merge `a918620a0b54bc86d9a5c7e2718c32a6947a648d`.
**Scope:** measurement-only candidate. No Colab allocation in development or acceptance tests. No additional listener, socket, tunnel, virtual desktop, encoder, decoder, or change to host input / video-pull request authorization.

## Research question

v33r reached **29.25 decoded H.264 FPS** at eight credits, but only **26.625 unique renderer sequences in scored replies per second** (213 distinct snapshot IDs across 234 decoded replies, an 8-second scored interval). The difference could reflect an independently slower CUDA producer, timing alignment of snapshot sampling, or another pipeline interaction. Reply counts alone cannot decide.

## Instrumentation contract

The existing CUDA background producer already writes an atomic `render-metrics.json` sidecar about once per second. v33s adds `window_seq`, `published_count`, and `window_elapsed_ms`; retained fields include `seq` (last published frame ID), `renderer_fps`, `target_fps`, and the median render+copy, zlib compression, and atomic publication durations. This production rate is **measured by the producer thread**, independently of host `video_pull` cadence. Existing `status.json` and `frame.bin` semantics remain unchanged.

During an explicitly bootstrapped H.264 pull, the remote source best-effort reads the existing atomic sidecar (maximum 4 KiB), verifies run identity, field types, finite bounded values, and agreement between window count and reported FPS, then adds an **optional** `producer_window` field to its existing authenticated Comm H.264 reply. Missing or malformed metadata is ignored without failing video. The host validates the field independently of H.264 codec/sequence checks and keeps the metadata aligned with the already-ordered reply commit side channel. No remote timestamps are compared with host clock values.

For every scored phase, the host reports:

- `producer_windows_observed`: number of distinct valid producer windows seen among that phase's replies (not number of video pulls).
- `producer_renderer_fps_p50`, `producer_renderer_fps_min`, `producer_renderer_fps_max`: the reported producer FPS across **distinct window IDs**, without overweighting windows that appeared in repeated pulls.
- `producer_render_copy_p50_ms`, `producer_compress_p50_ms`, `producer_publish_p50_ms`: median producer stage timing per unique window.
- `producer_window_metrics_missing_replies`: scored H.264 replies without validated producer-window metadata (early startup, sidecar absent, invalid, or older remote worker).
- `producer_window_conflicts`: count of inconsistencies where the same producer window ID is reported with differing contents; such windows are excluded from aggregations.

Each producer window covers approximately one second on the remote clock, and **can straddle host phase boundaries**. Therefore producer FPS aggregates are diagnostic context, **not exact per-phase CUDA production rates**. The host's `distinct_renderer_reply_fps` still separately measures unique frame IDs sampled by H.264 replies. An optional sidecar may start reporting after the first second; an absent value is represented as `null`, never zero renderer FPS.

## Acceptance and next action

1. Commit v33s directly to `feature/interactive-bridge-prototype` on GitHub, retaining `dev` and `main` unchanged.
2. Run `tests/test_bridge_video_v33s_renderer_production.py` and the full discovered Windows `test_bridge*.py` regression; check both 30/60 FPS managed **preflight-only** modes; run `git diff --check`.
3. Commit the feature files only, excluding unrelated v32 JSON evidence. Merge into `dev` only after review and a green regression in the `dev` worktree. `main` remains stable.
4. Only then consider a supervised managed 30 FPS T4 capacity sweep (same seven phases, same owned-runtime cleanup), and compare unique producer windows with distinct IDs received. Keep 60 FPS experiments on hold until the 30-fresh-FPS discrepancy is understood.

Note: the earlier feature-branch divergence was resolved by merge commit `a918620`. Do not force-push or merge this feature into `dev` until the merged-tree Windows regression and live experiment have been reviewed.
