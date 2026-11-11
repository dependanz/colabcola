# v33o — offline H.264 capacity diagnostics

**Scope:** Adds nonblocking, bounded diagnostic fields and per-phase aggregates to the existing H.264 capacity sweep. It does not increase concurrency, alter protocol authorization, change GPU allocation, or expand the normal streaming mode. **No paid Colab trial has been run with this instrumentation.**

The remote `RemoteH264PullSource` already included `renderer_seq` in every H.264 reply, but the host receiver did not retain it. The source now also reports `source_read_us` (remote same-clock wall duration of reading and zlib-decoding the CCB1 file) and `encode_us` (remote same-clock wall duration of NumPy/PyAV RGB conversion and x264 encoding). Timing includes scheduling interruptions and is **not** CPU process time.

The bounded host H.264 window validates optional metadata and preserves it in **ordered commit** order, even when IOPub replies arrive out of order. `poll()` still returns the original four-field tuples, so the standard two-credit streaming API is unchanged. Older test/fake peers may omit diagnostics; reports indicate missing samples rather than claiming zero remote processing cost. Values outside safe bounds are ignored as optional diagnostics; invalid H.264 access units still fail closed as before.

For each scored phase, reports now include:

- `distinct_renderer_sequences_in_replies` and `distinct_renderer_reply_fps`: distinct CUDA snapshot sequence IDs **among scored H.264 replies**, not screen presentation FPS or total unique CUDA production rate. These values may be lower than decoded FPS if snapshots were re-encoded.
- `duplicate_renderer_replies`: number of scored H.264 replies whose renderer sequence was already seen in that same phase.
- `remote_snapshot_read_p50_us` and `remote_encode_p50_us`: remote-only duration medians; not transport RTT and not subtractable from host timestamps.
- `host_pull_submit_interval_p50_ms`: host-only elapsed interval between consecutive scored pull submissions. A rate below the configured poll frequency can reveal scheduler/service restrictions.
- `remote_diagnostics_missing`: scored replies without usable diagnostic metadata.

These values do **not** identify actual Tk scanout, input-to-photon time, total network bytes or absolute source-image age. No unsynchronized remote/host clock subtraction is introduced. Remote renderer `render-metrics.json` can later help distinguish GPU rendering from H.264 source work.

## Acceptance

First run `tests/test_bridge_video_v33o_diagnostics.py`, then discover all `tests/test_bridge*.py` and run the full Windows regression. This patch must be checked for application and reversal against the v33n checkpoint. Only after those pass should a new T4 experiment be considered.
