# v38A.2 — independent remote producer and sender timelines

Base v38A.1 `0eec1c1ca236a6321512888467a60847af7b049b`. Local DANZIEPC: 72 passed/2 skipped focused, 898 passed/5 skipped full. CI workflow 45709666382 passed Windows and Ubuntu including v33 report comparisons. A.2 is an offline-first implementation, not live acceptance.

## Trace contract

The existing `--trace-pipeline` flag uploads a standalone standard-library trace helper. Only opted-in sessions create traces. The remote Input Lab records accepted input submission, application and producer frames; the sender records its independent credit, source-pull and Comm-send timeline for any opted-in 30 Hz app. No new socket, package install, automatic download or changed media/control policy.

- `remote-producer-trace.jsonl` (Input Lab only): authenticated input submission after Comm validation, accepted sequence, approximate queue depth, application start/end and stage, frame start, snapshot/render/publish boundaries, per-frame render/compress/publication costs, cache hit, bytes, renderer sequence and input ACK. Input ACK means logical application, not a guarantee of a distinct visual change.
- `remote-sender-trace.jsonl`: accepted credit grants/reservations with balance, cumulative observations of absent/not-fresh source headers (not duration), source pull begin/end, renderer/video sequence and ACK, Comm send begin/end, failures and initialized zero-credit condition waits. Comm-send completion does not prove network delivery.

Both use `time.monotonic_ns()` within the remote runtime; correlate via run ID, input sequence, renderer sequence and video sequence. The host trace uses host perf_counter_ns; never subtract clocks across machines. Remote input submission is not actual packet arrival.

Each independent trace retains a tail of at most 65,536 records. No active-path file writes; terminal flush is exclusive-create and records exact lost/retained counts. Trace writes happen after worker exit, outside state locks. They do not change runtime ownership or extend the credit lease. A stalled sender or a killed runtime can prevent its trace from being finalized; the producer trace is independent and can still be written on producer lease expiry or stop marker. Confirm complete terminal records, not merely file existence. No credentials, raw input values, images, error text, or media bytes are retained. Timing metadata introduces measurement overhead; quantify before comparing performance.

## Retrieval and gates

After a future explicitly authorized diagnostic run, inspect the exact remote run directory `/content/colabcola-bridge/<run_id>/` while the browser-owned runtime still exists. Verify status.json, stop marker, push-sender-status.json and both trace terminal records; download the two exact-run JSONL files manually using the Colab file browser before manually disconnecting the runtime. Retain the host pipeline-trace.jsonl from the corresponding fresh local run directory. A missing trace must be reported as incomplete evidence, not interpreted as absence of stalls.

Offline CI and local full bridge tests required. A.3 will add host wire ingress and Tk frame correlation; no paid test in A.2. Keep all v37D safety/ownership constraints and no release merge.
