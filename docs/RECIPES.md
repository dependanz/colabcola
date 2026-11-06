# Colab Cola — Managed workflows and recipes

[← Back to the main README](../README.md)

## Unattended lifecycle

The opt-in lifecycle CLI prepares, starts, inspects, cancels and resumes bounded managed
jobs. See [LIFECYCLE.md](LIFECYCLE.md) for setup and evidence boundaries.
The provider has passed native Windows CPU execution and a live
checkpoint/fresh-runtime recovery proof. WSL has also been tested; native Linux
and macOS still require separate validation.

Drive token renewal and publication after initial-token expiry passed the bounded
CPU probe and later multi-hour LongCat runs on the primary adapter. See
non-REST revalidation (historical evidence archived) for the exact tested workloads and
limits. Beta REST integration remains deferred.

Managed results now support 5 GiB of workload artifacts with streamed ZIP64 and
verified chunk recovery. See [the bounded renewal probe](../examples/runtime_renewal/README.md)
for the CPU/Latest/token workflow; its bounded non-REST renewal probe passed live.


The single managed command now displays ordinary stdout/stderr and declared log
files, including module-sequence progress. Runtime capture forwards small updates
promptly; the worker checks for changed logs every five seconds, plus transfer
and polling time. Unchanged verified members are reused, and the CLI displays a
recovered snapshot before slower provider checks. This is periodic delivery,
not a five-second latency guarantee. See [live progress](LIFECYCLE.md#live-progress-from-the-single-command)
for display bounds, full log locations and local/live verification status.


Managed jobs include built-in preflight before uploading or allocating:
`colabcola JOB.colajob`. For checks only use
`colabcola --preflight-only JOB.colajob`; `--prepare-only` remains offline.
See [LIFECYCLE.md](LIFECYCLE.md) for existing-assignment overrides, integrity,
GPU model/memory reporting and development-controller support boundaries.


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
Recipe execution uses the selected account profile; pass the same `--profile NAME`
used for login, or keep the same environment selection. Explicit
`execution.drive_parent` and `execution.token_command` remain available for legacy
configurations. Packaging failure retains its evidence; do not reset or overwrite
the incomplete workspace.

For another recipe, choose a separate run directory rather than reusing the
workspace of an unrelated job. The run directory is the workflow root;
its generated journal and artifacts live beneath it.
These commands run in the installed controller environment. Native Windows
PowerShell and Ubuntu WSL have both been exercised; they do not migrate or
upgrade an older controller.
