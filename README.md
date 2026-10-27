# colabcola

Development wheel installation: [DEVELOPMENT.md](DEVELOPMENT.md). Core packaging
works on native Windows; unattended execution currently uses Linux/WSL.

Package the command you already run; answer questions about its data, dependencies,
GPU, outputs and time limit:

```powershell
colabcola package -- python experiment.py --epochs 100
colabcola package -- python -m myproject.train
```

Optionally save TOML answers as a `.colaml` file, then reuse them:

```powershell
colabcola package --colaml training.colaml
```

Packaging is local. See [PACKAGING.md](PACKAGING.md) and the ordinary
[Kodak run.py](examples/kodak_grid/run.py).

A batch-job compiler for running trusted Python scripts and their declared data on Google Colab.

The primary development path is unattended `package -> run -> verified download`
through Google's official Colab CLI session adapter (configuration value `legacy`).
It uses Google sign-in and the existing Drive transport, without the allowlisted
beta `colaboratory.googleapis.com` API. Package a job with its execution settings,
then invoke `colabcola JOB.colajob`; `--prepare-only` freezes it without cloud actions.
See [the non-REST acceptance plan](NONREST_REVALIDATION.md) and [LIFECYCLE.md](LIFECYCLE.md).
WSL is the current provider host; native Windows support remains planned.

The manual controller flow below remains available for existing jobs.

The primary transport keeps large bytes out of notebook JSON:

```text
JOB.colajob -> deterministic .colabcola bundle -> temporary Google Drive staging
            -> tiny controller.ipynb -> Colab runtime -> verified artifacts
```

One-time Google setup uses two OAuth clients from the same Cloud project: a Desktop client for
local staging and a **TVs and Limited Input devices** client for Colab. Enable the Drive API, then
download both JSON files outside the repository and authorize colabcola locally:

```console
colabcola auth google-drive \
  --desktop-client-secrets desktop-client.json \
  --colab-client-secrets colab-client.json
```

The desktop user token is stored in the operating-system credential manager, not in a job or
notebook. Add the complete `colab-client.json` content to Colab Secrets under
`COLABCOLA_GOOGLE_DRIVE_OAUTH` and grant generated controllers access to that secret.

Submit a job while forwarding every argument after the job path unchanged:

```console
colabcola --output dist/job.colabcola job_config.colajob --steps 100
```

The bare command compiles the deterministic bundle, stages or reuses it through the Google Drive
API, reads back its exact identity, and writes a tiny bound controller plus a credential-free
receipt. Open the controller in Colab, select the runtime version named in its summary, and use
**Run all**. The controller requests only `drive.file` access, downloads the exact file ID, and
verifies its parent, name, byte count, MIME type, app properties, and SHA-256 before extraction.

For a compile-only operation with no network access:

```console
colabcola compile --output dist/job.colabcola job_config.colajob --steps 100
```

For small jobs that genuinely need a single hard-coded notebook, the warned secondary mode remains:

```console
colabcola --transport embedded --output dist/job.ipynb job_config.colajob --steps 100
```

Embedded payload cells stay hidden by default. Do not reveal them: large base64 source can make the
Colab frontend slow or unresponsive.

Drive staging uses Google's supported Python client, resumable uploads, `drive.file`, and the
operating-system keyring. In the manual controller flow, launch and Colab runtime selection remain explicit user
actions. See `DESIGN.md` for the full integrity and cleanup state machine.

Long-running jobs may declare `durable_checkpoints`. Their controller automatically uploads each
complete checkpoint pair, commits it by uploading a hash-bound manifest last, and restores the
newest unique committed pair when that same controller is rerun. Remote retention keeps at least
two committed pairs and moves older commit markers and members to Drive trash only after a newer
pair is committed. This behavior uses the controller's `drive.file` grant and is absent unless the
JobSpec opts in.

Large inputs that already have a stable, hash-verifiable home, such as model weights on Hugging
Face, can be declared as `remote_inputs` instead of being bundled: each names a target path, an
HTTPS URL, the byte count, and the SHA-256. The compiler stays offline and the declaration becomes
part of the job identity; the runtime downloads each file into the job root after extraction,
resumes interrupted transfers, and accepts a file only when its size and hash match. Nothing
undeclared is fetched.

## Unattended lifecycle (experimental)

The opt-in lifecycle CLI prepares, starts, inspects, cancels and resumes small managed
jobs. See [LIFECYCLE.md](LIFECYCLE.md) for setup and evidence boundaries. The current
provider proof uses WSL; native Windows support remains a product goal.

Drive token renewal and publication after initial-token expiry passed a 67-minute CPU probe. Multi-hour workloads beyond that duration remain unverified
on the primary adapter. [LongCat readiness](LONGCAT_READINESS.md) records the
remaining Python/date and resource gates. Beta REST integration is deferred.

Managed results now support 5 GiB of workload artifacts with streamed ZIP64 and
verified chunk recovery. See [the bounded renewal probe](examples/runtime_renewal/README.md)
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


## Run a recipe in one command (dev13)

```sh
colabcola run --colaml experiment.colaml --run-dir runs/experiment-1
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
Recipe execution requires execution.drive_parent and execution.token_command. Packaging
failure retains its evidence; do not reset or overwrite the incomplete workspace.

For an existing portraits recipe, select a NEW SIBLING folder; never the dev11 folder:

```sh
colabcola run --colaml portraits-2026-10/longcat.colaml --run-dir portraits-2026-10-dev13
```

The run-dir is the workflow root; its generated journal is longcat.run inside it.
These commands run in the installed controller environment (Ubuntu WSL for the current
live-verified provider path). They do not migrate or upgrade an older controller.
