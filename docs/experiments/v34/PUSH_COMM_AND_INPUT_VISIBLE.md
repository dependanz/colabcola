# v34A — Sender-driven H.264 over authenticated Colab Comm (experimental)

## Evidence

- v33v with a two-frame selection lag eliminated duplicate H.264 snapshots and reached 29.125 distinct frames/s at C16/C32, but observed input ACK p50 was roughly 220–230 ms.
- v33w reduced the allowed lag from 2 to 1, preserving ~28.9–29.1 distinct FPS at high credit counts and reducing input ACK p50 at C4/C8/C16/C32 to 194.20/195.34/184.42/181.71 ms (from 229.75/218.53/222.81/223.90 ms). This is **observational**, not a controlled end-to-end latency estimate.
- The pull route still requires a host `video_pull` for each encoded frame, despite continuous CUDA publication. The underlying authenticated Jupyter Comm may support unsolicited messages from a remote sender thread; this remains **unverified in a real Colab kernel**.

## First testable design

Use a new opt-in `--push-video` flag **only** with `--attach-existing --sequence-buffered --confirm-attach --transport comm --video-codec h264 --fps 30 --poll-hz 30 --inflight 1`. The attached browser-provisioned T4 must be disposable. The normal pull H.264 route and managed allocation path remain unchanged.

The attached kernel loads an isolated `bridge_video_push_runtime.py` module. A background thread checks the latest atomically published CCB1 header, then encodes a new renderer frame with the existing monotone H.264 source only when a bounded credit is available. The host initially grants **four credits** via the existing authenticated Comm and returns exactly one credit **after** validating and decoding each frame. The producer never opens a new port or issues a client pull request; the sender cannot send more than four frames outstanding. Initial grant is precisely four, later grants exactly one per previously sent frame, and duplicate/unsolicited grants fail closed. Source gaps are permitted while H.264 access unit sequences must remain contiguous.

The receiver validates the same run ID, Comm ID, monotonically increasing `pull_seq` and `renderer_seq`, video framing, and keyframe/dependency sequence gate. Any malformed message or lost H.264 dependency ends the run instead of replaying unknown frames. `video_no_new` is a correlated, bounded zero-frame acknowledgement and never enters the decoder.

Existing `run_viewer` has a local `LocalFrameTiming` structure. The v34 push mode uses `timestamp_local_input=True`, a 10 ms Tk poll instead of the legacy 30 ms, and `frame-timed` messages with `captured_ns` = **host receipt**, `decoded_ns` = **host decoding completed**, and `input_callback_ns` = local GUI callback time for the most recently acknowledged input. The viewport reports callback→Tk label-configuration p50. This DOES NOT establish physical screen presentation time, and sender-side/host-side timebases are never subtracted.

A short bounded local-only study is required before interpreting the results: compare v34 push 30 FPS and fixed four credits to v33w C4 across independent producer rate, distinct decoded FPS, buffer age, input callback→Tk p50 and renderer-sequence gaps. User inputs must be intentionally exercised; in their absence, input latency stays `null`, not zero.

## Operational boundaries

- No provider allocation POST, runtime release, kernel restart, new socket, or external TURN/STUN/WebRTC transport.
- Only the user's explicitly selected single browser-owned T4 is supported; `bridge cleanup` must never release it.
- Remote H.264 `comm.send` from a background thread is intentionally **experimental**; offline fakes cannot prove the notebook's Jupyter implementation permits it.
- The browser runtime continues consuming resources until the user disconnects/deletes it. No automatic retry after ambiguous send, bootstrap or websocket failure. On any failure, preserve sanitized logs and manually terminate the disposable browser session.
- Remote sender poll interval is capped at 4 ms; host nonblocking IOPub read yields with a 1 ms sleep. This is a producer-driven *application-layer* prototype and does not claim equivalence to Chrome Remote Desktop/WebRTC.
- `--max-seconds 120` bounds a supervised test. This includes bootstrap and video time; the GUI closes after its deadline.

## Validation

Run `tests/test_bridge_video_v34_push.py`, full `tests/test_bridge*.py`, pinned provider integrity checks, and Windows/Ubuntu GitHub Actions. Do not run live before these pass. A live test requires a read-only `--preflight-only` against a newly connected disposable T4 and explicit `--confirm-attach`; `--push-video` must not be combined with the legacy pull capacity benchmark.

Working feature branch: `feature/interactive-bridge-prototype`.
Tracking: https://github.com/dependanz/colabcola/issues/21.
