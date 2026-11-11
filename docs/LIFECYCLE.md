# Unattended lifecycle (design contract 0.18)

The opt-in lifecycle CLI packages a small trusted job, stages verified inputs, allocates
Colab compute, runs a detached worker, publishes checkpoints/logs/results to Drive,
restores after confirmed runtime loss, verifies local artifacts and releases owned compute.
Live allocation, GPU execution, deliberate runtime replacement, checkpoint continuation,
Drive publication, local verification and compute release passed the bounded WSL/T4 proof.

## Current setup (v0.1.0)

~~~bash
python -m pip install colabcola
colabcola login --profile research
colabcola login --profile research --status
~~~

New profiles use the bundled production Desktop OAuth client. The
execution project is separate from the OAuth publisher's project.
Colab and Drive grants must belong to the same account; tokens remain
in private host configuration and never enter job archives.
Login checks account and workspace permissions without allocating.
`--status` is offline; actual launch preflight rechecks cloud access.

Native Windows CPU runtime management/recovery and Ubuntu WSL have
passed bounded live tests. Other host/GPU configurations require their
own validation. Keep the supervisor connected for monitoring/cleanup.
See [user setup](USER_SETUP.md).

The dated experiment reports below preserve historical test scope.

## Job and CLI

A job opts in with:

    "managed_recovery": {
      "checkpoint_pattern": "outputs/fit/state-*.pt.json",
      "log_patterns": ["outputs/fit/events.ndjson", "outputs/fit/metrics.json"]
    }

The policy changes identity only when present and cannot mix with durable_checkpoints.
Each checkpoint JSON declares schema=1, integer step, sibling state file, bytes and
sha256. Write immutable state first, then atomically publish its JSON. Restored paths
reach the workload through reserved COLABCOLA_RESUME_MANIFEST.

The default pinned CLI backend supports runtime.version=latest. The opt-in REST
backend requests dated images through Google's documented Runtime.version field; use
--provider-backend rest for a job declaring YYYY.MM. It never substitutes Latest.
This new backend is offline-tested, not live-verified on this account. Exact
Python/package/GPU preflight remains mandatory. --high-mem requests host RAM, not
a particular GPU memory capacity. See archived LongCat readiness notes for the migration gates.

    colabcola lifecycle prepare JOB.colajob --run-dir NEW_RUN_DIR --drive-parent EXACT_FOLDER_ID --token-command HELPER_ARGV.json --gpu T4 --max-attempts 2 --max-seconds 900 --fault-after-checkpoint

Prepare is local. Review the job, destination, GPU, limits and file hashes, then use
its printed plan SHA-256:

    colabcola lifecycle start RUN_DIR --approve EXACT_PLAN_SHA256
    colabcola lifecycle status RUN_DIR
    colabcola lifecycle logs RUN_DIR
    colabcola lifecycle cancel RUN_DIR

Start detaches the supervisor. Status is JSON; logs emits NDJSON lifecycle events.
Workload logs appear under RUN_DIR/logs; verified results under RUN_DIR/artifacts.
Repeating start resumes persisted state; a lock prevents concurrent owners.
Cancellation is durable but requires a running supervisor to release compute.

## Recovery

Allocation UUIDs are persisted before calls. Uncertain responses reuse the UUID;
cleanup reconciles it even without a recorded endpoint. Connection failures mean
unknown. Two successful observations of absence precede replacement.

The worker publishes immutable members with readback, then their commit manifest.
Partial uploads cannot become resumable checkpoints. Restoration verifies all bytes
before exposing its directory. Final verification checks identity, ZIP paths, exact
declared members and hashes.

With --fault-after-checkpoint, success additionally requires destruction of the first
owned runtime after a verified remote checkpoint, a distinct replacement, matching
resume commit, positive restored step and later progress. Finishing before the fault
is recovery_proof_not_demonstrated, even when training succeeds. The checkpoint selected
before deliberate destruction is persisted with the stop intent; later publications cannot
replace it. Only the current attempt can supply a terminal result, and first-attempt
finals are ignored after deliberate-stop intent.

## Bounds

Prepared inputs: 512 MiB total, including the compiled notebook and worker files.
Inputs above 32 MiB use verified Drive chunks; only the small worker/bootstrap files
travel through the runtime API. Notebook compilation still materializes JSON in memory.
Schema1 snapshots: 64 MiB. Schema2 snapshots: 6 GiB including ZIP/metadata/log
overhead, transferred in 8 MiB chunks. Final workload outputs: 5 GiB aggregate
uncompressed, plus up to 64 MiB result.json metadata. Attempts: 1-3.
Shared wall budget: 60-86400 seconds, explicitly selected; default remains 900.
It covers staging, allocation, setup, work and recovery across all attempts. The
service may stop a runtime earlier. Drive copies are retained. The existing
large-payload controller transport remains separate.

The laptop and WSL must remain awake and online for replacement and compute release.
The worker stops training at the shared deadline even when the host disconnects;
new plans use renewable short-lived Drive access. A host thread renews the token
independently of artifact transfers, using the existing host credential helper. It
checks every 60 seconds and refreshes access with at most 600 seconds remaining.
Only access_token/expires_at reach the runtime, atomically outside job artifacts;
refresh credentials remain on the host. A separate runtime watchdog stops the
execution process group 120 seconds before access expires, or at the shared deadline.
Final artifact publication can still fail; only committed snapshots are recoverable.
The watchdog stops work, not the VM: the host supervisor owns runtime deletion.
In-flight calls, provider outages, sleep or supervisor termination can delay release. The deadline is not an exact
billing ceiling. Resume supervision to reconcile and release uncertain allocations.

Local evidence covers Windows regressions, transport faults and actual Linux worker
destruction/restoration with a filesystem store. The bounded live recovery proof also
passes as recorded below. Hyperplus reconstruction metrics are measured; no convergence or quality threshold is claimed. Public-product OAuth onboarding was validated separately for v0.1.0.

Live update 2026-10-15: T4 allocation, exact current-runtime preflight, embedded-wheel
installation, 2000-step Hyperplus GPU training, Drive checkpoint/log/final publication,
local verification and compute release passed. Final snapshots include terminal logs
even for fast failures. The corrected recovery proof then verified checkpoint 25 before
destroying the first T4, restored it on a distinct T4, and completed steps 26 through 2000.
All 1975 resumed image-sampling batches and the final sampler state match checkpoint
continuation exactly; the first 25 losses are preserved. Seven ZIP outputs, eight final
Drive members and both restored checkpoint members passed independent byte verification.
Both runtimes were released after 261.246272 seconds of the 900-second allowance; an
independent provider query confirmed zero active assignments. This proves the bounded
WSL/T4 lifecycle after initial sign-in, not native Windows or general production readiness.


## Single-command managed workflow (2026-10-15)

A job declaring managed_recovery now routes bare submission into the managed supervisor:

    colabcola --prepare-only JOB.colajob
    colabcola JOB.colajob

An adjacent JOB.run.json supplies drive_parent, token_command (path to the helper argv
JSON), gpu, max_attempts and max_seconds; optional fault_after_checkpoint requests the
bounded recovery proof. Optional provider_backend (legacy or rest) and high_mem
(boolean) are bound into new plans. --execution-config overrides that path. --run-dir overrides the
default JOB.run journal. All CLI flags precede JOB; trailing tokens remain workload args.

Preparation is offline. Actual invocation is the execution request; --approve SHA256 can
bind it to a previously reviewed plan. Progress is human-readable, final status is JSON,
and failures return nonzero. Repeating the command reconnects to the same journal, without
fresh allocations or a reset budget. Terminal completed runs are verified locally.

--resume-from TERMINAL_RUN with a new --run-dir explicitly starts a new bounded segment
from that same job's verified checkpoint. The initial checkpoint is bound in the new plan.
It does not permit changed data, arguments or source to inherit a checkpoint.

Schema1 checkpoint manifests remain supported. Schema2 adds an artifacts object mapping
safe relative paths to bytes/sha256, beside the immutable state file and JSON marker.
The worker verifies and publishes these completed artifacts before committing the
checkpoint. A restored workload receives their directory through the existing
COLABCOLA_RESUME_MANIFEST interface. The Kodak example uses this to skip completed fits.

Schema2 transfer snapshots support 8 MiB chunk objects, at most 4096 members and 6 GiB
aggregate. Files and the final ZIP are hashed as streams. Content-addressed remote and
local reuse avoids retransferring completed fit artifacts for every checkpoint.
Prepared inputs above 32 MiB now use schema2 Drive chunks, capped at 512 MiB total.
The worker verifies snapshot identity, member hashes and sizes before executing the notebook.
Existing small-input plans keep their original transport; attempt/time bounds are unchanged.
The Kodak example and its pilot are described in examples/kodak_grid/README.md.


## Terminal recovery and log ordering

A worker-reported final commit is fetched by its exact ID and verified against the
run, job and attempt even when Drive listing has not indexed it yet. Reopening a
terminal journal now searches the run-scoped catalogue for later logs,completed-output
commits and final results after runtime release. This path creates no provider and
preserves the terminal state,attempts and deadline;late results never turn a failed
run into success. The same recovery pass runs automatically after cleanup when
there is time remaining in the shared budget. Reopen the same command to retry
read-only recovery if Drive indexing or access was delayed.
Log snapshots at the same step/attempt are ordered by Drive creation time, not by
arbitrary file ID. Historical event replay in the terminal UI remains a separate UX issue.


## Long-run compatibility and ownership (contract 0.17)

New plans bind authorization_policy=renewable-v1, provider_backend, runtime_version
and high_mem. Existing prepared plans retain their original hash, worker bytes,
bounds and authorization policy; prepare a new journal to adopt these features.
Portable JobSpec schema 1 and compiler contract 0.10 are unchanged.

REST creation persists a UUID4 request ID and stable owned runtime name before POST,
then records the asynchronous operation. Retries reuse those identities. A dated
version or runtime specification mismatch fails and cleans up that owned runtime.
Cleanup never creates another runtime. An unknown create outcome remains pending
until reconciled; a missing GET alone cannot prove an outstanding creation failed.
If the create response and operation name were lost and the resource never becomes
visible, cleanup can remain unresolved and requires operator investigation.
Deletion is confirmed with GET before recording release. No quota evasion, automatic
GPU upgrades, or unbounded retries to obtain more VRAM are implemented.

The beta REST service is allowlist-only, then requires enablement on the consumer
project. Its first live probe failed with SERVICE_DISABLED before creation. REST
adoption is deferred; it is not required by the primary official CLI session path.
The CLI supports Latest only in the inspected interface. LongCat dated-image/Python
compatibility is therefore still unresolved; constraints are never silently weakened.


## Five-GiB results and renewal evidence (contract 0.18)

Output validation and ZIP creation stream files; ZIP64 is selected when needed.
The 5 GiB workload-output allowance excludes result.json. The 6 GiB transfer
envelope leaves room for ZIP overhead, terminal logs and metadata without reducing
the workload allowance. Prepared inputs remain capped at 512 MiB. The packager's
default per-glob output allowance stays 1 GiB; explicitly request 5368709120 bytes
for a 5 GiB glob. Multiple globs must fit the aggregate recovered-output limit.

Recovery caches only verified chunks and reuses them after an interrupted download.
A partial snapshot is never exposed as a verified result. Whole members are still
stream-hashed before acceptance, and corrupt cached bytes fail integrity checks.
Reserve disk for the archive, recovered bytes and caches: the initial check requires
twice the snapshot size plus 64 MiB free; filesystems without hard links may need
another full copy. Caches are retained; no automatic deletion of user artifacts is
introduced. The remote job also needs space for original outputs plus the ZIP.

New renewable workers include public authorization.json evidence in the final
snapshot: initial/latest expiry, actual token changes used by Drive requests, and
verified publication timestamps. No token values, token hashes or refresh credentials
are recorded. The ordinary examples/runtime_renewal probe requires verified
publication with a changed token at least 120 seconds after initial expiry.
Existing prepared workers are frozen: prepare a new journal for these features.


## Primary non-REST path (contract 0.19)

The stable configuration value `legacy` denotes the official Colab CLI session
adapter, not the beta REST service. It remains the default, including for historical
plans that omit provider_backend. Explicit unknown backends and legacy dated plans
fail; no backend is silently substituted. The CLI adapter now confirms the exact
owned endpoint is absent after unassign before the supervisor records release.
A delayed response or observation keeps cleanup pending and consumes no new runtime.

Existing bounded GPU proofs used this adapter. Revalidation is tracked in
archived non-REST revalidation notes. The fresh CPU renewal probe explicitly requests Latest
and records the actual Python version without claiming a dated environment.


## Live progress from the single command

`colabcola JOB.colajob` prints verified stdout/stderr and declared `.log` files,
with attempt number and file name. Module sequences also print command start/exit
markers. Workloads do not need a special progress format. Existing Kodak structured
progress remains supported. Python runs unbuffered by default; an explicit job
environment override is respected. Other programs must flush their own output.

Runtime capture forwards available bytes immediately instead of waiting for a
64 KiB buffer. The worker checks for changed logs every five seconds, plus transfer
and supervisor polling time. This is a polling interval, not a delivery guarantee. Completed newline/carriage-return updates are displayed;
a final snapshot also flushes unterminated text. This is periodic log delivery, not
a websocket stream. No new snapshot means no repeated output. A reconnect shows a
bounded recent tail; in-process log cursors distinguish attempts and file resets.
Each snapshot displays up to 20 recent updates plus up to eight earlier error lines
per log (up to 32 logs), with long lines shortened. This retains a root CUDA OOM
when a launcher appends a long wrapper traceback. Full logs stay in the verified snapshot directory printed by the
command. Terminal control characters and conventional credential fields are removed
from the display; workloads must still avoid writing secrets into their raw logs.

Kernel control connections now initialize cleanup state before constructing their
manager, close channels/socket idempotently and never shut down compute during
connection teardown. Connection timeouts retain bounded retries; executing a command
is not retried after an uncertain reply. Real execution/cleanup errors remain visible
with credential-safe exception types. Runtime release still uses the supervisor's
separate unassign-and-confirm-absence procedure.

The 2026-10-15 fixes passed live GaussianAvatar acceptance through the public command:
one T4, 963.540 seconds within 3600, all 5000 training iterations completed. The command
showed 298 training updates across 15 verified snapshots, beginning at step330 more
than five minutes before the completion marker. The first training stderr snapshot
was 13,333 bytes, below the former 64 KiB buffering threshold. No client cleanup
warning occurred. All 63 outputs (including 41 PNGs) passed recovery and fresh Drive
readback, and the owned runtime was independently confirmed absent. Evidence is under
nonrest-revalidation-20261015/gaussian-live-progress/acceptance.json in the mapped
output resource. This verifies the bounded WSL/Latest/T4 case, not native Windows
provider execution. Future runs require a fresh journal and their own compute budget.

Remote input downloads now write `runtime/transfer.log` through the same verified
snapshot path: received/declared bytes, verified-file counts, current file bytes,
MiB/s, retry state and final size/SHA-256 verification. Updates are throttled to
two seconds in the runtime, with immediate file/retry/completion markers. The host
still sees periodic snapshots. In-flight bytes are distinct from verified bytes;
URLs and authorization headers are never printed by transfer reporting.

The supervisor caches verified immutable commit bytes for its current process.
Every cache key includes the complete run/job/kind/reference identity; changed
references are fetched again and caller mutations cannot alter cached commits.
Output chunks and recovered files retain their original hash checks. Polling no
longer downloads every old log manifest on every pass. Reconnection re-verifies
the catalog; no disk cache is trusted implicitly. Only the current attempt's logs
are selected. The subsequent dev7 LongCat run measured the remaining delay described below.
The dev9 repair has local coverage; its cloud latency remains unmeasured.


### Local latency and renewal-health repair (dev9)

Unchanged log members reuse the previous publication's remotely verified references.
Recovery rehashes matching local log bytes and fetches only changed/missing members;
a cold readback still fetches every member. Reuse is confined to the same run, job
and attempt. Worker-reported exact log commits bypass Drive search-index delay,
with listing fallback for older workers. The CLI receives verified logs before
subsequent provider checks can block it.

The host checks renewal every minute but delivers only a changed token, or retries
an unconfirmed delivery. Public health separates valid runtime access from a
credential-refresh or delivery retry. A worker's successful publication confirms a
lost acknowledgement only for an expiry already known to the host. Successful
recovery removes the active error; the last delivery error can remain as history.
The 600-second refresh margin and 120-second expiry stop are unchanged. No access
or refresh token is written to public health. CLI health messages show transitions
without repeating unchanged states.

The prior dev7 LongCat run completed all five clips and released its sole A100 in
83m38s, but source-stamped logs had a 52-second median and 88-second maximum delay,
and host delivery health retained a timeout. Dev9 addresses the local causes;
cloud latency after this repair has not been measured. No new runtime is authorized
by these tests or documentation.

## Completed outputs before job completion (development 0.1.0.dev10)

Declare JSON completion markers in `.colaml`;the workload can use its existing
manifest format. A matching marker is a promise that the listed files are complete,
closed and immutable. Write files first,then atomically rename the JSON marker last.
A marker path is immutable;use a new directory for a new version.

```toml
[[outputs.completed]]
marker = "outputs/smoke/*/manifest.json"
files = ["video.mp4"]

[outputs.completed.ready_when]
status = "completed"
```

The worker checks these groups during execution. Paths in `files` are relative to
the marker's directory. The marker itself is always included. Every member must
match the job's normal output declarations. Incomplete,failed and missing markers
do not publish a group. A ready marker with missing files is a persistence error.
The worker freezes completed files,checks output/file-count/5GiB aggregate bounds,
uploads immutable chunks,and writes the hash-bound `artifacts` commit last. It
reports the exact commit reference only after authenticated readback. Unchanged
members reuse their verified chunk references;earlier groups remain in each
cumulative snapshot. No periodic archive of still-changing outputs is accepted.

The host automatically verifies/downloads these commits to `<run>/artifacts/<sha>`
and prints the protected file count and local directory. The final archive remains
separate. If the runtime disappears before final packaging,the completed-output
snapshot remains recoverable without allocating another runtime. This is output
durability;it does not teach arbitrary programs to skip completed work or replace
their checkpoint/resume logic. A loss during a group's first upload can still lose
that group;only a verified commit is protected.

The JobSpec representation is optional `managed_recovery.artifact_groups`,an array
of `marker`, `files`, `ready_when` objects. Existing jobs without this field retain
their portable identities. New policies require a fresh job/prepared journal and
dev10 or later;frozen prior consumers are never rewritten in place.

## Runtime-loss evidence and cleanup (development 0.1.0.dev10)

Missing assignment listings are observations,not immediate proof of termination.
The supervisor checks the exact allocation UUID through a read-only GET and records
a90-second quiet grace. A present direct assignment prevents loss confirmation;
newly recovered Drive progress restarts the quiet interval. Timeouts remain unknown
connectivity and never authorize a new allocation. The shared time/attempt bounds,
expiry protection and deliberate restart proof remain enforced.

After grace,the controller sends an unassign request for the exact durable-owned
endpoint even when listing is empty. Only an accepted request or authenticated404,
followed by absence confirmation,allows it to mark that attempt released and
consider another already-authorized attempt. A404 is reported as already absent,
not an acknowledged deletion. The journal retains last worker heartbeat,missing
observations and loss timestamps;service-side termination cause can remain unknown.
No user-facing claim infers CUDA OOM from loss alone.

## Genuine tracked execution (development 0.1.0.dev11)

Fresh non-REST preparations freeze `execution_mode = "kernel-v1"`. The worker
executes as the actual Jupyter request for its entire lifetime;it is not launched
as a detached subprocess behind an idle notebook kernel. The controller keeps the
owned channel open,records kernel identity,busy/idle state and execution receipts,
and reattaches to that same kernel after connection loss without replaying work.
A durable submission intent with no execution receipt is uncertain and cannot
silently start another copy. Host channel cleanup never interrupts or shuts down
the kernel;owned-runtime release remains the supervisor's explicit operation.

Drive renewal uses the contents API to upload a pending token. The worker validates
and atomically adopts newer,unexpired access in its ordinary publication/watchdog
paths,so renewal does not queue another execute request behind the busy workload.
Partial,invalid,expired or older uploads cannot replace the current lease. Existing
shared deadlines,expiry protection,artifact commits and attempt limits remain.

Historical prepared plans without this execution mode retain their original
semantics. Prepare a fresh journal with the new controller/worker to adopt the
repair;never rewrite an old approval or consumed journal. Frozen development
controllers are not upgraded in place.

The dev9/dev10 losses were observed about21 minutes after the last confirmed short
kernel operation,well before credential expiry;the successful dev7 controller
continued short token-delivery executions. A local real Jupyter kernel reproduces
the detached-launch idle defect and verifies the tracked repair. This supports an
idle-execution hypothesis;it does not reveal Google's service-side termination
reason or establish live reliability. Colab documents variable idle timeouts and
continuous code execution in its [official FAQ](https://research.google.com/colaboratory/faq.html).
No dummy activity or timer-based idle-timeout bypass is implemented.


### Bounded cloud follow-up â€” 2026-10-16

The subsequent private eight-case LongCat proof on installed `0.1.0.dev11` passed
on one A100 80 GB with a fresh `kernel-v1` journal. The same kernel/request was
tracked across 363 busy observations. All 8 videos, 8 pre-final Drive commits,
three natural token rotations, post-expiry publication, cold recovery and
independent runtime absence passed within 10800 seconds. See
`archived non-REST revalidation notes` and the mapped `danzel_longcat_example/acceptance.json`.
This observes continuity for that bounded run. The older Google termination
causes remain unconfirmed and old journals are not migrated. CLI logs still
show measured delivery delay; timely progress remains a separate follow-up.


## Built-in preflight (development controller dev12)

Run a managed job with one command; admission checks run automatically before staging:

```powershell
colabcola .\experiment.colajob
```

An existing prepared plan can be selected explicitly:

```powershell
colabcola --approve PLAN_SHA256 .\experiment.colajob
```

`--prepare-only` stays offline. `--preflight-only` performs the same admission checks
without staging or allocating. Place controller options before the job filename.

The gate checks installed wheel RECORD integrity (damage detection, not a publisher
signature), pinned provider dependencies, approval/path/job identity, frozen hashes,
consumer bindings, local recovery space, Colab sign-in/listing, existing assignments,
Drive token lifetime (>600 seconds), and the exact owned untrashed writable parent.
Editable installs must be replaced with a built development wheel before launch.
Normal OAuth refresh may update the local credential store; no interactive login is
started by preflight. Assignment listing is a point-in-time check, not a reservation.
Existing assignments block a fresh run unless EACH observed endpoint is explicitly
listed using repeatable `--allow-active-assignment ENDPOINT`. This never cancels them.

Sanitized reports (including failures and interruption) live in RUN.run/preflight/;
state.json points to the latest. Failed admission preserves prepared/zero attempts
and starts no work deadline. A passing standalone check is rerun on actual launch.
Reconnection and cleanup do not apply fresh-journal admission to an owned runtime.
No arbitrary consumer hooks run inside this gate. Metadata checks do not prove Drive
quota, future renewal, future upload success, or guaranteed runtime lifetime.

New preparations declare gpu-model-v1. Runtime preflight records model names, memory
bytes, device count and requested/observed comparison before extraction, installs or
remote downloads; a mismatch prevents workload execution and triggers owned-runtime
cleanup. A100 variants are accepted only with the job's memory floor; H100 is not an
A100 substitute. Unknown models fail closed. G4 maps to RTX PRO 6000 Blackwell Server
Edition (https://docs.cloud.google.com/compute/docs/gpus); other workstation variants
are not inferred equivalent. Reports appear in live progress, state.json and recovered
hardware.json/result.json. CPU jobs record their requested CPU policy.

Existing dev11 prepared plans are never rewritten. Their frozen worker/notebook code
cannot acquire new model enforcement through a host upgrade; preflight reports this
as legacy_model_check_unavailable. Reprepare in a NEW directory to adopt runtime
checks, then use the newly printed approval. Keep the frozen dev11 controller for
already reviewed dev11 runs. A package version alone is not a portable job identity,
but new preparation changes run identity and can change frozen notebook hashes.

Local admission reserves four maximum 6-GiB snapshots plus 64 MiB on the journal
filesystem. Each chunked recovery rechecks three times its actual size plus 64 MiB
for chunks, reconstruction and copy fallback. Retained snapshots/logs can exceed that
initial reserve. Existing bounds remain: 5-GiB outputs, 6-GiB snapshots, 4096 members,
8-MiB individual collected logs and 10000 Drive objects per listed kind. Five-hour
renewal is covered by virtual-clock tests, not a new live five-hour acceptance run.


## Run a recipe in one command

```sh
colabcola run --colaml experiment.colaml --run-dir "$HOME/colabcola-runs/experiment-1"
```

This uses the general packager, prepares the managed journal, runs built-in preflight,
stages inputs, executes, recovers artifacts and releases owned compute. The recipe is
read from its original location, so relative data, lockfile and helper paths keep their
meaning. Its output setting is redirected into the NEW workflow folder. Existing
unrelated folders, including older prepared jobs, are refused without modification.
A new folder cannot be nested inside an existing managed job folder.

No consumer-written launcher or copying script is needed. The CLI writes timestamped
`launch.log`, `cli-exit.json`, `workflow.json`, packaging evidence, the generated
`.colajob`/`.files`/`.run.json`, and the standard `.run` journal beneath that folder.
Repeat the SAME command and run directory to reconnect/recover; it reuses frozen inputs,
plan and budget and does not repackage or create another runtime after a terminal run.
Changing or removing the original recipe does not replace an existing workspace's job.

Add `--prepare-only` for packaging/preparation without network, or `--preflight-only`
for admission checks without staging/allocation. `--non-interactive` requires complete
recipe answers. `--approve SHA256` and `--expect-job-id SHA256` are optional exact
identity guards; an expected job guard is retained on reconnect. Existing assignment
overrides use the same repeatable `--allow-active-assignment ENDPOINT` as managed jobs.
Recipe execution uses the account profile created by `colabcola login`. Explicit
`execution.drive_parent` and `execution.token_command` remain available for legacy
configurations. Packaging failure retains its evidence; do not reset or overwrite
the incomplete workspace.

For an existing portraits recipe, select a NEW SIBLING folder; never the dev11 folder:

```sh
colabcola run --colaml portraits-2026-10/longcat.colaml --run-dir portraits-2026-10-dev13
```

The run-dir is the workflow root; its generated journal is longcat.run inside it.
These commands run in the installed controller environment (Ubuntu WSL for the current
live-verified provider path). They do not migrate or upgrade an older controller.

## Saved results and Drive status (dev17)

Before runtime bootstrap confirms a lease, progress says `Drive access pending runtime
bootstrap`. Actual expired leases and renewal failures still report their real health;
the running-job expiry guard remains fail-closed.

Repeating a terminal job opens its saved result: `Viewing existing completed run; no
new runtime will be started`. Historical allocation events and workload logs are not
replayed as current activity. If final artifacts were missing, the command can still
recover late uploads from Drive without allocating compute, and identifies these as
newly recovered snapshots from the existing run.

The result retains `result_directory` and `completed_artifact_directory` and adds:

- `final_artifacts_status`: `verified` or `missing`.
- `completed_outputs_status`: `verified`, `not_configured`, or `not_available` for
  optional intermediate completed-output commits.
- `viewing_existing_run`: whether this command displayed a terminal journal.
- `final_terminal_state` and `run_success`: a successful command requires a completed
  journal and verified final artifacts whose workload terminal state is `success`.

A null intermediate directory is normal when intermediate commits were not configured;
it does not indicate missing final results. Corrupt saved artifacts fail verification.


## Worker heartbeat and host telemetry (issue #18, development)

A tracked `kernel-v1` worker reports `kernel_execution` failure evidence with
sanitized exception class and receipt completion time. Confirmed failed
execution transitions through a durable owned-runtime release barrier before
a bounded retry can allocate anything. A failed final commit is not assumed
from a stale heartbeat.

For every current-attempt `running` worker with a valid heartbeat, the supervisor
checks its age against `max(120 seconds, 10 * poll_seconds)`, where the default
poll interval is 5 seconds. A stale heartbeat records a single `worker_stalled`
event with `heartbeat_age`, `last_heartbeat`, `observed_at`, and the threshold.
The warning persists across supervisor restart. Once a fresh heartbeat is
observed, a `worker_stall_cleared` event closes that warning episode.
**A stalled heartbeat by itself never releases compute or starts a replacement.**
Unknown, invalid, and future-dated heartbeat values are not interpreted as
evidence of worker failure.

Worker log commits include `runtime/host-telemetry.json` with bounded readings
of Linux `/proc/meminfo` (MemTotal, MemAvailable, Cached, Dirty, Writeback);
cgroup v1/v2 memory limits, usage and available event counters; host and
available cgroup memory/I/O pressure; and free bytes on job/control filesystems.
`runtime/execution-tail.log` is capped at 64 KiB. The initial
`hardware.json` includes `host_memory.mem_total_bytes` and
`host_memory.cgroup_memory_limit_bytes`. Unavailable metrics are null or
empty, not invented zeros. Telemetry carries no token values or remote URLs;
the execution log tail is ordinary workload log material and may contain
content emitted by the workload, so protect recovered logs accordingly.

These changes have offline Windows/Linux CI coverage. The specific real-Jupyter
interrupt/restart acceptance tests, large-input page-cache control, and long
real-Colab workload are still separate release gates for issue #18.


## Large remote-input cache control (issue #18)

For remote inputs, the notebook retains the exact target paths, byte-count and
SHA-256 verification, and HTTP Range resumption semantics. Before each file
and every retry it checks free disk against the bytes still outstanding plus
the existing 1 GiB margin. After each 64 MiB of streamed file output, it
flushes Python buffers and calls `fdatasync` (or `fsync` when unavailable),
then **attempts** `posix_fadvise(POSIX_FADV_DONTNEED)` on the synchronized
range. Remaining bytes are synchronized on exit, including resumable partial
downloads. The disk check also repeats after each window, so an unexpected
concurrent decrease in free space is detected before the next window.
Cache eviction is an advisory kernel operation and is not a guaranteed hard
RAM limit; unsupported or failed advice is represented by counters, not
reported as successful eviction.

Verified run metadata in `transport.remote_inputs.cache_control` reports
`window_bytes`, `sync_calls`, `advice_supported`, `advice_calls` and
`advice_errors`. Jobs declaring no remote inputs omit the entire downloader
from their compiled notebook, preserving the compact controller size.
No local GPU or Colab allocation is needed for the synthetic regression tests.

A two-GiB Linux proof on GitHub Actions (November 11, 2026) streamed exactly
2,147,483,648 deterministic bytes, successfully performed 32 synchronization
and 32 cache-advice operations with no advice errors, and observed 0 resident
bytes using `mincore` immediately after download. A subsequent independent
read verified the final file's SHA-256. This is one filesystem/kernel result,
not a promise of 0 resident pages on a Colab runtime. To repeat the opt-in
local proof on Linux with at least 5 GiB free:

    COLABCOLA_LARGE_INPUT_PROOF=1 python -m pytest -q -s tests/test_remote_input_cache_large.py

The dedicated GitHub workflow for the issue executes this proof without
Colab credentials, network inputs, or GPU allocation. Real long-duration
Colab execution and kernel-interruption/restart acceptance remain outstanding.


## Scratch filesystem assessment (issue #18)

When remote inputs are declared, the downloader performs a **read-only**
capacity comparison before its existing disk preflight. The evaluation
records the bytes required including the 1 GiB safety margin, free bytes on
the current job filesystem, and available fixed candidate directories
(`/mnt/local-scratch`, `/mnt/data`, `/scratch`). For each existing
non-symlink directory, it records free bytes, whether `st_dev` differs
from the current job filesystem, and whether the capacity is sufficient.
Missing candidates and symlinks are omitted. Candidate contents are never
read, created, deleted, or modified.

The evaluation is stored in
`result.json -> transport.remote_input_storage_evaluation`, as
`storage-assessment.json` in the runtime results, and under
`runtime/storage-assessment.json` in worker log snapshots. This makes the
assessment visible even while a large input is still downloading.

**This is diagnostic only:** it does not choose a different disk, relocate
files, use symlinks or bind mounts, or modify the working directory or
remote-input target paths. A larger filesystem with enough reported free
space is only a possible candidate for a future separately tested placement
strategy. Distinct device IDs and available bytes do not prove that a path
can safely be mounted or that its data will survive provider interruption.
Actual placement remains out of scope until evaluated on the intended
Colab runtime.


## Missing tracked-execution receipts (issue #18, closure review)

A `kernel-v1` runtime whose `status.json` still reports `running` but
whose owned `kernel-execution.json` **and** `kernel-request.json` are both
missing is *not* treated as a dead worker: missing files alone cannot
establish a kernel failure, a process-group exit, or permission to
release/replace an owned runtime. The supervisor retains a sanitized
connectivity diagnostic and respects the existing wall-time limit.
It never submits the workload again merely because the receipt is missing.

If both execution artifacts are missing **but** `status.json` explicitly
reports `failed` or `persistence_failed` for the exact current attempt,
that terminal status remains actionable: the supervisor journals
`request_state=receipt_missing`, follows the existing failed-status
cleanup path, and never fabricates `KeyboardInterrupt` or other kernel
exception evidence. A mismatched attempt cannot use this fallback.

Legacy plans without `kernel-v1` remain status-driven and do not depend
on execution receipts. This rule is covered by offline provider and
supervisor tests; it does not claim that the original provider's unknown
termination cause has been established.
