# v34B — first live push observation, Windows Tk safety fix, diagnostics

## First live experiment

On 2026-11-11 the first `--attach-existing --push-video --sequence-buffered --renderer-lag-frames 1` experiment passed read-only single-T4/single-kernel preflight and established the background-thread remote `Comm.send` path can deliver decoded H.264 frames to Windows.

Run `v34a-push-20261111-162938` reported:
- 666 decoded, distinct frames in 28.86 s, or **23.077 decoded frames/s** over the run
- input callback to **host receive**, p50 242.21 ms, 66 observed input events
- one-frame producer selection lag p50
- `video_no_new=0`, `configured_credits=4`
- exit 0; journal `phase=detached`, `ownership=external`, `backend=google-colab-external`, `video_push=true`, `close_status=host_disconnected`.

However, the Tk main-thread `poll()` callback crashed on `UnicodeEncodeError` because the user's Windows CP1252 console could not encode the arrows in `ViewerPresentationMeter.report()` (e.g. `input callback→Tk`). The exception happened before the next `root.after`, halting viewer polling and preventing a meaningful input-to-Tk measurement. The successful Python exit/journal only establishes host detachment, not a healthy viewer or full experiment. **Do not treat 23.077 FPS as a clean benchmark or equate the 242 ms receive observation with v33w input ACK p50.**

## Isolated fixes

1. `console_safe_viewer_summary` emits ASCII-only telemetry text (`->`, `|`) while leaving the original Unicode reporting API intact. Tk logging catches stdout `UnicodeError`/`OSError` and keeps the `root.after` poll loop active. A strict `cp1252` stream regression confirms the complete input-callback-to-Tk message prints.
2. The v34 push session reports independently sampled remote producer FPS from validated one-second producer windows; renderer frame IDs skipped across received frames; host frame arrival interval p50/max; remote encoder p50; and host callback-to-decode timing as separate diagnostics. The existing callback-to-receive p50 remains so the two clocks/measurement points are not confused. All duration comparisons use one host `perf_counter_ns` clock.
3. Offline tests verify producer-window deduplication, source sequence gaps and callback-to-decode versus callback-to-receipt ordering.

## Next gate

Only after Windows and Ubuntu full CI and local `tests/test_bridge*.py` pass should another disposable browser-owned T4 be used. Never retry the failed session run directory; create a fresh one. First read-only `--preflight-only`, then one supervised `--confirm-attach` invocation with a 120-second upper bound and user-owned lifecycle.

Interpret results carefully:
- If remote independent production stays ~29.7 FPS but received FPS ~23, compare remote encode time and host inter-arrival jitter and source gaps; avoid assuming a codec bottleneck.
- A viewer can fall below receiver FPS due to Tk queue coalescing or UI processing; report decoded FPS and configured FPS separately.
- Input callback-to-host receive and input callback-to-Tk config do not equal input-to-photon latency. At least one controlled marker experiment remains necessary before claims about CRD-like responsiveness.

No GPU/provider code, allocation lifecycle, vendor sources, or default pull video behavior are modified by this correction.
