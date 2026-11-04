# Package an ordinary Python workload

`colabcola package` creates a job from code you already run. It does not execute
or import your program, install packages locally, access credentials, upload files
or allocate a runtime. No workload-specific builder or directory layout is required.

## From installation to a first run

Use Python 3.12+ on Linux or Ubuntu WSL for managed execution. Native Windows can
package jobs; runtime control currently uses Linux/WSL. This preview is not on PyPI.
Install the publisher-provided development wheel in a virtual environment:

```bash
python3.12 -m venv "$HOME/.venvs/colabcola-dev24"
source "$HOME/.venvs/colabcola-dev24/bin/activate"
python -m pip install /path/to/colabcola-0.1.0.dev24-py3-none-any.whl
colabcola login --profile danzel
```

Follow [README: user-owned project setup](README.md#user-owned-project-setup)
for the complete illustrated first login. The example profile is stored at
`~/.colabcola/configs/danzel`; choose an unused name for a fresh setup. Check
`colabcola login --profile danzel --status` first: an unused name reports
`not_configured` without network access. Existing named logins are reused, and
`--force` renews consent rather than resetting them. Automatic Cloud project setup
needs `gcloud` in the same Linux/WSL environment; the manual route records its project/API/billing checks as
self-reported. Login prompts are in the terminal, with Google Console and consent
steps in the browser. After you create the Desktop client, keep its downloaded
`client_secret_*.json` in Downloads; login can detect the stable matching file and
continue without a path prompt. Distinct matching clients require your selection;
`--drive-client PATH` or `--no-auto-client` keeps explicit control. Client creation
and Google consent remain manual. Login does not allocate compute.

From your existing project, package its normal command:

```bash
cd /path/to/your/project
colabcola package --profile danzel --save-colaml training.colaml -- python train.py --data data
```

The questions cover data mappings, dependencies, GPU, outputs, checkpoints and
workload time. For a bounded first GPU run, choose T4, declare the program's real
outputs (for example `outputs/**`), choose no checkpoint if it cannot resume, and
choose 10 minutes for workload time. With no explicit limit overrides this gives
one runtime maximum and 900 seconds for the entire managed run. Inputs already
inside the project need no extra mapping. Review dependencies; `none` means require
the detected packages in Colab, not that those imports are ignored. Packaging is
offline and never imports or executes the workload locally.

```bash
colabcola --profile danzel train.colajob
```

That command performs preflight, uploads the frozen inputs, requests compute,
shows progress, recovers artifacts and releases the owned runtime. Keep the
supervising computer awake and connected. Repeating the same job/journal opens
its existing run; after completion it does not launch another runtime.

A new run from the saved recipe needs a new workspace outside any existing
managed-job folder:

```bash
colabcola run --profile danzel --colaml training.colaml \
  --run-dir "$HOME/colabcola-runs/training-002" --non-interactive
```

This last command starts a new bounded run. Reusing its exact run directory
reconnects to that frozen run and does not reset its time or runtime allowance.
A future published release replaces only the wheel-install line with
`python -m pip install colabcola`; the personal-project setup remains part of login.

## Select the login used for execution

`--profile NAME` is accepted after `login`, `package` and `run`. For direct job
execution, use `colabcola --profile NAME JOB.colajob`. Packaging stays offline and
does not bind a portable job to an account: use the intended profile again when
preparing or executing, and keep it unchanged when resuming a frozen journal.
Existing explicit helpers and prepared journals retain their absolute paths;
they are not migrated when a profile is selected.

To keep examples without flags on one profile for a shell session:

```bash
# Clear a custom root only if that is the selection you intend to replace.
unset COLABCOLA_CONFIG_HOME
export COLABCOLA_PROFILE=danzel
```

An explicit profile overrides environment selections. Otherwise
`COLABCOLA_CONFIG_HOME` takes precedence over `COLABCOLA_PROFILE`, followed by an
existing `~/.config/colabcola`, or `~/.colabcola/configs/default` on a new install.
Legacy files are left in place. Profiles do not create a persistent active-account
switch. See [profile selection and existing logins](README.md#profile-selection-and-existing-logins)
for names, custom paths and offline status.

## Command-first guided packaging

From the directory where the command normally runs:

```powershell
colabcola package -- python experiment.py --epochs 100
colabcola package -- python -m myproject.train --config configs/train.yaml
colabcola package -- accelerate launch train.py --mixed_precision bf16
```

This inspects the current project without importing or running it. It includes the
project tree with the exclusions below, follows ordinary local imports by retaining
that tree, reads requirements.lock/requirements.txt or PEP 621 pyproject dependencies,
and proposes dependency names from static imports when no declaration exists.
Import-to-distribution guesses need your review; dynamic imports and file accesses
cannot be discovered reliably. It never freezes the host environment or runs --help.
A src/ module layout adds PYTHONPATH=src. Editable project installation and native
system-package provisioning are not automatic.

Questions cover extra input mappings, unresolved dependencies, accelerator, output
globs, checkpoint manifests and workload time. Existing external --data/--input/
--config path arguments suggest mappings. Mapped paths are rewritten in argv to
runtime-relative paths. Other arguments are preserved literally; no shell expands
them. Review the proposed package, then optionally save answers. No runtime starts.
An absolute script outside the current directory uses its parent as project context;
--project explicitly selects a different root. --include/--exclude select files.

```powershell
colabcola package --colaml training.colaml -- python -m myproject.train
colabcola package --colaml training.colaml
colabcola package --colaml training.colaml --non-interactive
colabcola package --save-colaml training.colaml -- python experiment.py
colabcola package --command "python -m myproject.train" --command "python -m myproject.evaluate"
```

`--colaml` loads the configuration; `--save-colaml` writes a new one.
The earlier `--answers` and `--save-answers` spellings remain compatibility aliases.

A .colaml file is TOML. Fields answer questions; omitted unresolved fields are still
asked. --non-interactive reports the first missing answer without creating a job.
Explicit CLI options override saved answers (lists replace, rather than append).
A trailing argv or --command sequence overrides saved commands. Paths inside the
file resolve relative to that file; the command executes relative to project.

```toml
commands = [["python", "-m", "myproject.train", "--data", "data"]]
project = "."
inputs = [] # Explicitly no extra mappings; files under project are already included.

[environment]
python = ">=3.11"
requirements = "requirements.txt"

[compute]
gpu = "T4" # false/NONE for CPU; true selects T4
minimum_gpu_memory_gib = 16

[outputs]
include = ["outputs/**"]
checkpoints = [] # Or one outputs/... JSON checkpoint-manifest glob.
logs = ["outputs/progress.json"]
maximum_bytes = 1073741824

[limits]
timeout_minutes = 60
```

Replace inputs=[] with [[inputs]] tables containing source and target for external
files/directories. Optional environment fields: dependencies (ordinary PyPI strings),
runtime_requirements (assert installed), wheels (local .whl paths), isolated_python
(currently "3.12"), env (string-valued table). Top-level remote_input_files names
JSON manifests described below. outputs.resume_arg supplies a single command's
checkpoint option. No checkpoint state exists unless the program writes it.

limits also accepts timeout_seconds, max_seconds (total unattended lifecycle bound),
max_runtimes and seed. Default total bound leaves at least 60 seconds beyond the
workload timeout; execution uses the account profile selected when preparing/running.
Optional [execution] holds drive_parent, token_command, runtime_version,
provider_backend and [compute] high_mem / minimum_gpu_memory_bytes. The default
backend is legacy/non-REST with runtime latest. A100 does not guarantee 80 GB;
minimum memory is checked when executing. These settings never authorize a launch.

An omitted output filename defaults to <script/module/command>.colajob in the current
directory. --output changes it. The packager also creates a sibling .files immutable
snapshot, and a .run.json containing the requested GPU and bounds. Its default
account selector resolves the selected local login only at preparation/execution;
it does not pin a named profile during packaging. Generated
answers default to no save and never overwrite an existing .colaml. Answers and
host-only credential helper configuration are excluded from the portable payload.

The existing explicit --output CLI remains available without questions; pass
--interactive to enable inspection/questions for such commands. Console commands
must be installed by the declared workload dependencies. Commands run without a
shell, so use repeated --command for sequential tasks instead of && or pipelines.

## Explicit script packaging


```bash
colabcola package -o jobs/experiment.colajob --gpu T4 --artifact 'results/**' train.py --epochs 10
```

Only `train.py` is included by default. If it imports sibling files or uses project
configuration, explicitly include the project tree:

```bash
colabcola package -o jobs/experiment.colajob --project . --requirements requirements.txt --data /datasets/images=data --artifact 'results/**' -- python train.py --data data
```

CPU jobs are the default (`--gpu NONE`). GPU jobs require CUDA. Script paths are
relative to `--project` when given; otherwise they are relative to the current host
working directory. Files keep their relative paths and commands run at the packaged
project root. Workload arguments are copied literally: there is no shell expansion.
Host absolute data paths inside arguments are not rewritten; map data explicitly
and pass the corresponding relative runtime path.

## Explicit module packaging

```bash
colabcola package -o jobs/train.colajob --project . --gpu T4 --artifact 'outputs/**' -- python -m myproject.train --epochs 10
```

`--project` must contain `myproject/train.py` (or a package with `__main__.py`).
Relative imports work normally. For an uninstalled `src/` layout, use `--project src`
when that is the root from which the original Python command works. Automatic editable
installation of projects and native build-system provisioning are not implemented.
The [GaussianAvatars example](examples/gaussian_avatar/README.md) prepares an ordinary
module-based native-build pilot and documents FLAME/data acquisition. Its bounded T4 CUDA build, 200-iteration training and saved-model rendering
pilots passed; those results do not establish convergence or held-out quality.

## Train, then evaluate

Write `commands.json` using argument arrays, not shell command strings:

```json
[
  ["python", "-m", "myproject.train", "--output", "outputs/model.pt"],
  ["python", "-m", "myproject.evaluate", "--model", "outputs/model.pt"]
]
```

```bash
colabcola package -o jobs/pipeline.colajob --project . --commands commands.json --artifact 'outputs/**'
```

Commands run in fresh Python processes, in order, sharing the job filesystem and a
single workload timeout. The first failed command stops the sequence and its exit
code is recorded. Extra arguments at job submission are supported for one command;
sequences reject ambiguous extra arguments. An interrupted sequence restarts from
its first command; it does not silently skip a command whose outputs happen to exist.

## Files, dependencies and outputs

- `--project DIR` snapshots that tree; `--data SOURCE=TARGET` adds files or directories.
  Targets are relative to the reconstructed root. Windows sources such as
  `C:/datasets/images=data` work. Collisions, path traversal, symlinks and junctions
  are rejected. The generated `_colabcola/` directory is reserved.
- Common caches, environments, VCS folders, outputs/build/dist directories, `.env`
  files, known credential filenames and key files are excluded from tree capture.
  This is not a secret scanner: inspect the generated job and source snapshot before
  upload. Explicit single-file inputs are deliberate inclusions.
- `--exclude GLOB` and `.colabcolaignore` provide additional simple project-relative
  or basename globs. These are not full Gitignore semantics; `.gitignore` is not
  interpreted. Included files are enumerated explicitly in the generated job.
- `--requirements FILE` accepts ordinary PyPI requirement lines, extras, version
  constraints and markers. Pip installs them in Colab; installed versions are
  recorded in result evidence. Unlocked dependencies can resolve differently later.
  Existing fully hash-pinned requirement lines retain the strict hash-checked path.
  URL/VCS/local requirements, pip option lines and recursive requirement files are
  rejected with an actionable error. They are not silently ignored or host-resolved.
- `--runtime-requirement 'torch==VERSION'` checks an existing runtime package without
  installing it. `--python '>=3.11'` controls the Python compatibility constraint.
  Packaging does not infer imports or assume the local environment matches Colab.
- Repeat `--artifact GLOB` for required outputs. Without it, `outputs/**` is optional,
  so a print-only script works. `--artifact-max-bytes` bounds each glob's aggregate.
  Stdout, stderr and terminal result metadata are captured by the runtime.

The `.colajob` references a sibling `.files` snapshot. Keep both together; unlike the
original working files, modifying that snapshot changes job identity. Packaging refuses
to overwrite an existing destination. A generated launcher and argv JSON are product
implementation details; the workload remains ordinary Python. Existing handwritten
JobSpecs continue to work.

## Execute with bounded managed Colab supervision

After `colabcola login --profile danzel`, a new package can run through
`colabcola --profile danzel JOB.colajob`. The generated `.run.json` resolves the
selected local account at preparation/execution. For an explicit legacy
Drive configuration, these options remain available:

```text
--drive-parent DRIVE_FOLDER_ID --token-command /path/to/drive-helper.json
--gpu T4 --max-runtimes 2 --max-seconds 900 --timeout 840
```

The helper file contains a credential-free command argv array. Credentials remain in
host stores; the helper is never included in the remote payload. The generated config
retains the requested GPU model. Without these explicit host settings, execution
resolves the selected local login. Packaging itself needs no login and never reads
credentials. See LIFECYCLE.md.
The current managed provider supports explicit 60ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“86400-second segments, 1ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“3 attempts,
and the configured WSL host. Native Windows packaging works; native Windows provider
execution remains separate work.

```bash
colabcola --prepare-only jobs/experiment.colajob
colabcola jobs/experiment.colajob
```

Preparation verifies the frozen plan without allocation. Execution stages the job,
allocates compatible compute, reports progress, saves logs/results to Drive, verifies
local recovery and releases owned compute. Allocation remains service-controlled.
A laptop supervising the run must stay awake and connected.

Prepared notebook/worker inputs are capped at512MiB total. Above32MiB, the managed
workflow stages verified8MiB Drive chunks and sends only the small bootstrap through
the runtime API. The worker verifies the exact input snapshot before execution.
Compilation still materializes the embedded notebook in memory; this is a bounded
input path, not an unlimited dataset streamer.

Checkpoint continuation is optional. Use `--checkpoint 'outputs/state-*.json'` for the
existing integrity-bound manifest protocol and `--resume-arg=--resume` to pass its
restored path to a single command. The program decides how to restore its model,
optimizer and random state. `--log 'outputs/progress.json'` adds a live log pattern.
No arbitrary Python in-memory checkpointing is claimed.


## Dated images and multi-hour execution

Package options now include --runtime-version YYYY.MM, --provider-backend rest,
--high-mem and --minimum-gpu-memory-bytes N. The default backend remains legacy
with Latest. Dated selection requires rest; unsupported dates are never changed
to Latest. Backend and host-memory shape live in the host execution configuration;
Python constraints, runtime version and minimum GPU memory remain portable job
requirements. When no .run.json is generated, configure its backend explicitly
before preparing a dated job.

Set --max-seconds explicitly for longer execution, and leave setup/transfer margin
around --timeout. New plans renew short-lived Drive access while the supervisor
remains awake and connected. Existing plans keep their original bounds and policy.
The 24-hour ceiling is a colabcola limit, not a promised Colab runtime lifetime.
See [LongCat preparation](LONGCAT_READINESS.md) for the supplied workload's proposed
three-hour envelope and remaining live-validation and output-size gates.


## Large output artifacts

For a 5 GiB allowance use `--artifact-max-bytes 5368709120`. Managed recovery
accepts 5 GiB of aggregate expanded workload outputs, with separate result-metadata
space and a 6 GiB snapshot envelope for ZIP/log overhead. Output hashing, ZIP64
creation and recovery stream files in bounded memory. Interrupted downloads reuse
verified chunks. This does not change the 512 MiB prepared-input limit.

The [ordinary runtime renewal probe](examples/runtime_renewal/README.md) demonstrates
a prepared CPU-only dated-image and sustained Drive test through the same packager.
Its timed live execution remains a separate authorization.

## Explicit Python3.12 workload isolation

Add `--isolated-python 3.12 --python '>=3.12,<3.13'` to the same script,
module or sequence packaging command. Colab's control interpreter stays unchanged.
The job declares a pinned Linux x86_64 CPython3.12.13 archive with its URL, size and
SHA-256 in `python.isolation`. The runtime verifies it before extracting, creates
an independent venv, and routes pip, package probes, entrypoint and Python children
through it. No host/system/user-site Python packages are inherited. Host Python/pip
configuration is stripped; conflicting job overrides are rejected. GPU drivers and
OS libraries remain shared with Colab; this is not a dated Colab image or container.

An isolated job installs dependencies and runs pip check before downloading its
remote dataset/weights. Hash-pinned dependencies remain hash checked; missing
previously preinstalled dependencies must be declared explicitly. Setup logs appear
as `runtime/environment.log` in managed snapshots and final recovery. Result evidence
records control Python separately from workload Python, executable/prefix and the
verified archive identity. Invalid hashes, unsupported hosts, failed installs,
wrong Python or failed package checks stop the workload without a host fallback.

The first profile supports Linux x86_64 only; native Windows can package it but
cannot execute its Linux interpreter. Recovery starts with a new isolated environment
on each new runtime. Workload checkpoint/continuation remains the script's responsibility.

Isolated installs select a complete host GCC/G++ toolchain, or Clang/Clang++ when
GCC is unavailable, and explicitly set CC, CXX and LDSHARED. Compiler paths are
recorded in isolation evidence and streamed to environment.log. Colab must supply
OS build tools; colabcola does not install system packages. Binary-only jobs may
run without a compiler. Isolated pip runs disable the wheel cache so a cached local
extension cannot hide a missing source-build prerequisite during rehearsal.

## Local wheels and remote data

Repeat `--wheel /path/to/package.whl` to snapshot wheels under `wheels/` and install
them in the runtime. Repeat `--remote-inputs /path/to/inputs.json` for JSON arrays:

```json
[{"target":"weights/model.bin","url":"https://example.com/model.bin","bytes":1234,"sha256":"<64 hexadecimal characters>"}]
```

These are declarations, not builder scripts. Remote targets, HTTPS URLs, byte counts,
hashes and collisions use the normal JobSpec validation. Packaging never downloads
remote inputs or installs wheels on the host. Colab downloads and verifies the declared
bytes/SHA-256 before running the command. The input file limit includes both bundled
and remote members. The portable job holds the declarations directly; its original
requirements/remote manifest files are not needed after packaging.

## Protect completed outputs during execution

Optional `.colaml` completion groups let ordinary programs retain their own output
layout and JSON manifests:

```toml
[[outputs.completed]]
marker = "outputs/smoke/*/manifest.json"
files = ["video.mp4"]
[outputs.completed.ready_when]
status = "completed"
```

`files` uses paths/globs relative to each marker directory;the marker is included
automatically. All files must be within declared outputs. Write/close files before
atomically creating the ready JSON marker,then keep that group immutable. The
managed worker protects each group on Drive and the host verifies/downloads it
before final job packaging. Existing log/checkpoint declarations remain independent.
The equivalent advanced flag is `--artifact-group` with one JSON object. Groups
are optional and saved/replayed by `--save-colaml`;no custom job builder is needed.
See LIFECYCLE.md for commit,interruption,recovery and size-limit boundaries.


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
Recipe execution uses the selected local account profile. Keep the same explicit
`--profile NAME` or environment selection used for login and preparation. Explicit
`execution.drive_parent` and `execution.token_command` remain supported for legacy
configurations. Packaging failure retains its evidence; do not reset or overwrite
the incomplete workspace.

For an existing portraits recipe, select a NEW SIBLING folder; never the dev11 folder:

```sh
colabcola run --colaml portraits-2026-10/longcat.colaml --run-dir portraits-2026-10-dev13
```

The run-dir is the workflow root; its generated journal is longcat.run inside it.
These commands run in the installed controller environment (Ubuntu WSL for the current
live-verified provider path). They do not migrate or upgrade an older controller.
