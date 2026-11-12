# v33v — sequence-aware bounded producer-to-H.264 handoff

## Evidence and hypothesis

The first successful user-browser-provisioned T4 attachment (v33u, `20261111-135420`) completed its seven-phase 30 Hz H.264 sweep. At 32 credits: independent renderer production p50 **29.76 FPS**, host decoded **29.25 FPS**, distinct renderer source IDs **27.125 FPS** (217 distinct of 234 decoded; 17 duplicates); median pull RTT ~125 ms. Four credits nearly saturated decoded throughput. These figures support a producer/consumer sampling mismatch: legacy `RemoteH264PullSource.pull` encodes the **latest** atomic `frame.bin` again for every request, even if `renderer_seq` is unchanged.

The next question is whether a small bounded producer history and explicit empty-pull responses improve genuinely distinct delivered frames **without adding unacceptable latency**.

## Opt-in implementation

Only `bridge colab --attach-existing --sequence-buffered` uses this experimental contract; ordinary managed allocation, external attachment without the flag, file polling and existing H.264 legacy modes are unchanged.

- A CUDA worker writes its normal latest frame and, only when opted in, an atomically replaced `frame-ring-{slot:02d}.bin` for `slot = renderer_seq % 8`. Eight fixed slots cap disk usage. Each contains a complete CCB1 frame with monotonically increasing renderer sequence and validated dimensions.
- The remote source scans only eight **headers**, then decompresses and validates one chosen frame. It never emits a previously encoded renderer sequence. The first selected frame is the newest available; later reads choose the oldest frame newer than the last encoded one, **no more than two frames behind the newest available**. This caps intentional ring lag and counts skipped/stale producer sequence IDs.
- When no newer usable snapshot exists, `video_pull` receives a correlated `video_no_new` response with `run_id`, `pull_seq` and last applied input ACK. This consumes the request's credit without encoding, repeating a frame, advancing `coded_seq`, or advancing the host H.264 dependency gate.
- The host accepts `video_no_new` **only** for opt-in sequence-buffered peers, after the same Comm/run/pull correlation checks. Corrupt/mismatched/duplicate replies fail closed. Empty responses maintain input-ACK monotonicity at ordered commit but never enter PyAV. No video pulls are replayed after uncertain transport outcomes.
- Telemetry includes empty replies, total skipped renderer sequences, and p50/max buffered lag in producer frames. H.264 access-unit sequence remains contiguous even when renderer sequences legitimately skip; input-to-photon latency is still **not** directly measurable.
- External ownership is preserved: no `assign`/`unassign`, no kernel restart, remote Comm and worker close best-effort, browser notebook remains user-owned. The benchmark report identifies an external runtime rather than misleadingly calling it owned.

## Safety and limits

The ring absorbs short producer/consumer timing jitter; it is **not** an unlimited throughput queue. Producer frames older than two frame periods relative to the newest are deliberately skipped. Better fresh FPS must not be claimed at the expense of large queue lag. On ring reader races, return `video_no_new` rather than re-encode an old frame.

The producer may slightly slow because it writes one extra atomic frame per iteration; measure that overhead. Empty responses lower decoded FPS but do not count as H.264 suppression. Measurements remain host-configured images, not physical display scanout. No provisioned GPU or notebook was contacted to create this commit.

## Offline acceptance and next test

Run `tests/test_bridge_video_v33v_sequence_buffer.py` plus all `tests/test_bridge*.py`; validate GitHub CI on Windows and Ubuntu and private vendor checks. Confirm legacy source tests unchanged. Any failed test blocks live use. For a supervised comparison later, provision a disposable browser T4, run read-only attach preflight, then invoke the same 30 FPS seven-phase command with `--sequence-buffered` and `--confirm-attach`. If PyAV is absent, only explicitly install it in the disposable notebook. Compare at C4/C8/C32 against v33u: independent producer FPS, **distinct renderer FPS**, decoded FPS, empty responses, skipped frames and buffer lag. Stop if hardware watchdog or ownership uncertainty occurs. Disconnect the browser notebook manually when done.

Related issue: https://github.com/dependanz/colabcola/issues/21 .
