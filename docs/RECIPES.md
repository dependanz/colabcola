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

## Named environments and ordered steps

A recipe can declare several workload environments and select one for each step.
Use `environments` and `steps` together. The existing `commands` format continues
to use one workload environment. Named steps run sequentially in the shared job
directory; a failed step stops the sequence.

```toml
project = "."
inputs = []

[environments.prepare]
python = "3.10"

[environments.evaluate]
python = "3.12"

[[steps]]
name = "prepare"
environment = "prepare"
command = ["python", "prepare.py"]

[[steps]]
name = "evaluate"
environment = "evaluate"
command = ["python", "evaluate.py", "--input", "../outputs/data.json"]
cwd = "analysis"
```

Step names and environment names are unique lowercase hyphen-separated slugs.
Each step names a declared environment. `command` is a literal argument array;
Python scripts, Python modules and commands installed in that environment are
supported. Scripts must be inside the packaged project. Arguments added to the
generated job at launch time are refused: put each step's arguments in the recipe.
A fixed Bash interpreter is also supported for packaged script files:

```toml
[[steps]]
name = "prepare-shell"
environment = "prepare"
command = ["bash", "scripts/prepare.sh", "literal argument"]
```

The script path is relative to the step's `cwd`, must end in `.sh`, and must be a
regular included project source file. Use normalized POSIX paths (no leading `./`,
`..`, absolute paths, backslashes, or symlinks). Colabcola invokes `/bin/bash`
without startup files, after activating the selected environment, and passes each
argument separately. The script's bytes are frozen into the job and checked again before dispatch. Arbitrary shell
strings, `bash -c`, interpreter flags and other shell programs remain unsupported.
Write Bash files with LF line endings. As with Python workloads, script code is
trusted workload code; this is not a shell sandbox. Bash does not need to be a
conda dependency, but commands and libraries used by the script must be available.

`cwd` defaults to `.` and is relative to the project root. Script paths and ordinary
file arguments are relative to that step's working directory. The working directory
must exist in the packaged project. Requirement file paths below are always relative
to the project root, independently of a step's `cwd`.

The controller still uses its own Python installation. The environments' `python`
values choose workload interpreter versions, such as `3.10` or an exact patch
version. Package resolution happens on the allocated Linux x86_64 runtime using a
checksum-pinned micromamba executable; conda need not be installed on the packaging
machine or in the runtime image. All declared environments are built before the
first step starts. A build or dependency check failure prevents workload execution.

### Declare dependencies per environment

```toml
[environments.evaluate]
python = "3.12"

[environments.evaluate.conda]
channels = ["conda-forge"]
packages = ["ffmpeg"]

[environments.evaluate.pip]
packages = ["numpy==2.2.6"]
requirements = ["requirements/base.txt", "requirements/extensions.txt"]
no_build_isolation = true

[environments.evaluate.env]
PYTHONUNBUFFERED = "1"
MAX_JOBS = "2"
```

Conda channels are explicit and default to `conda-forge`. Pip installs the declared
package list first, then each requirement file in order. This lets an extension
requirement use dependencies installed by an earlier file. Use
`no_build_isolation = true` only when the build needs that environment's already
installed dependencies. A bounded `pip check` runs before the environment is ready.

Use `pip.wheels` to install local, already-built wheels into one named environment:

```toml
[environments.train]
python = "3.12"

[environments.train.pip]
packages = ["numpy==2.2.6"]
wheels = ["wheels/my_extension-1.0-cp312-cp312-manylinux_2_28_x86_64.whl"]
```

Wheel paths are normalized POSIX paths relative to the packaged project. Include
these files in `include` if you use an allowlist. They are snapshotted with role
`wheel` and exact SHA-256/size, just like other payload files. Wheel filenames
must be valid wheel filenames; absolute paths, traversal, excluded sources, links
and duplicate distributions in one environment are rejected. Multiple environments
may reference the same wheel, provided it is compatible with each interpreter.
After the declared package and requirements installs, the selected environment
installs the wheel set with `--no-index --no-deps --force-reinstall`. Dependencies
must already be present or `pip check` fails; a matching preinstalled index version
cannot silently replace the declared wheel. The runtime rechecks each wheel against
the frozen manifest and records its path, byte count and SHA-256 in the environment
record (`embedded_wheels`), along with the resolved package inventory. The controller
Python is unaffected. This is distinct from the legacy top-level `--wheel` option,
which remains incompatible with named environments.

Pip supports ordinary package requirements, project-relative local packages beginning
with `./`, and named `git+https` direct references pinned to a full 40-character
commit. A Git reference can include a project-relative `#subdirectory=...` fragment.
Local package sources and requirement files must be included in the project
selection; excluded or missing files fail packaging.
Requirement files are bounded UTF-8 files and cannot include another requirements
or constraints file, editable installs, or arbitrary pip options. Index requirements
can include SHA-256 hashes. Use `pip.index_url` and `pip.extra_index_urls` for HTTPS
package indexes; credential-bearing URLs and query strings are refused.

Declared environment variables can set application options and build limits. Relative
`PYTHONPATH` entries are interpreted from the job root. Interpreter, activation,
package-manager and library-search overrides are reserved for colabcola; credentials
do not belong in a recipe. An ordinary console executable missing from the selected environment fails
instead of falling through to a command from the runtime image. The fixed runtime
`/bin/bash` is the explicit exception for packaged shell scripts.

For named jobs, dependency and isolation settings belong inside each environment.
Global `environment.dependencies`, requirements, wheels, runtime requirements and
`isolated_python` cannot be combined with named environments. The optional global
`environment.python` constraint describes the controller interpreter in Colab, and
ordinary global `environment.env` values still apply to every step.

### Opt-in environment caching

```toml
[environments.train]
python = "3.12"
cache = true
```

Managed runs can save a fully provisioned, validated environment to the selected
Drive parent before the first workload step. A fresh runtime restores it at the
same deterministic installation prefix only when the full cache identity agrees:
normalized environment declaration, wheel and requirements identities, manager
and implementation versions, actual runtime release and host ABI, and observed
GPU compatibility. Local source builds conservatively bind the entire frozen job.
`runtime_version = "latest"` alone is not a runtime image identity. When the
runtime does not expose reliable compatibility facts, or storage is unavailable,
colabcola bypasses caching. Standalone notebooks do not mount or discover Drive.

CPU feature matching is exact by default (`cache_cpu = "exact"`). Intel and AMD
hosts can therefore have different keys even for identical recipes. An explicit
`cache_cpu = "x86-64-v2"` (requires `cache = true`) is a portability assertion:
all installed code, including native wheels and package/source builds, must run
on any Linux x86-64-v2 host with the remaining recorded compatibility constraints.
The runtime checks the required CPU flags on every visible CPU, binds the target
rather than the full flag set, and constrains micromamba's solver archspec for
portable cache builds. Missing required flags bypass caching. The runtime image,
ABI, GPU, dependencies and verification constraints are unchanged. The solver
constraint does **not** constrain arbitrary package build scripts or prove binary
portability. Do not opt in for `-march=native` builds or unknown ISA requirements;
wheel platform tags alone do not prove portability. In that case keep exact
matching. Actual host fingerprint and selected policy are retained in cache
result evidence. The cache implementation identity changes with this release, so
older cache entries are not reused by the new implementation.

Cache uploads use immutable, verified chunks and a commit published last. A hit
must pass compressed archive integrity, safe extraction, interpreter activation,
package inventory, `pip check`, and applicable CUDA checks. Failed restoration
falls back to a fresh build within the remaining lifecycle budget. A failed cache
publication does not by itself fail the workload. Cache transfers are optional
and bounded (4 GiB compressed, 12 GiB expanded, 200,000 entries, and up to five
minutes per operation subject to remaining time and workload reserve). Cache
objects include installed code, including any proprietary dependencies; enable
this only with an appropriate private Drive parent. Invalid or old entries are
not automatically deleted, and the cache is not a dependency lockfile: unpinned
packages may retain older resolved versions. Change the declaration or disable
caching to request a fresh resolution.

The named environment record includes `cache.state` (`miss`, `hit`, `rebuild`,
`bypass`), the key when available, and timing/publication evidence. The reported
`estimated_seconds_saved` compares a prior cold build's duration with the current
warm environment setup; actual end-to-end savings depend on the runtime and
Drive throughput. Local integration has been tested using simulated provisioning
and Drive HTTP transport. Real Colab cold/warm acceptance is still pending.
See [the cache design](ENVIRONMENT_CACHE_DESIGN.md).

### CUDA toolkits and native builds

Declare toolkit packages in the environment that builds or runs the extension.
For example, a CUDA 12.8 environment can select a matching toolkit explicitly:

```toml
[environments.gpu]
python = "3.10"
cuda = "12.8"

[environments.gpu.conda]
channels = ["nvidia", "conda-forge"]
packages = ["cuda-toolkit=12.8"]
```

`cuda` is a validation expectation; it does not install a toolkit or choose a
PyTorch build. Declare compatible framework packages in that environment's pip or
conda specification. Colabcola locates `nvcc` inside the environment, checks its
major/minor version against `cuda`, and checks a CUDA-enabled PyTorch installation
against that compiler. A missing or mismatched compiler fails during environment
setup, before a step can accidentally use the host toolkit.

The runtime adds the toolkit's target include directory and installed NVIDIA wheel
include/library directories to the build environment. It excludes stub libraries
from runtime library search. This supports extensions whose framework headers also
include CUDA library headers. GPU driver support and the extension's own source
compatibility remain separate requirements; verify those with a small compilation
test on the intended GPU.

### Evidence, deadlines and recovery

The frozen manifest and prepared plan contain all environment declarations and the
ordered steps. Preflight verifies that those declarations match the packaged job.
Environment setup is recorded as its own stage. Resolved interpreter paths, versions,
installed packages, toolkit facts and setup timing are retained in
`environments.json` and `result.json`; per-step timing and exit evidence are retained
in `steps.json` and `result.json`. These metadata files accompany live log snapshots,
including partial setup or execution failures.

`limits.timeout_seconds` bounds the workload sequence. `limits.max_seconds` is the
shared managed budget, including allocation, transfer, environment setup and work.
Named jobs reserve 30 seconds of that budget for finalization. Choose a total budget
that allows package downloads and native builds; exhausting setup time prevents the
steps from starting. The existing supervisor and recovery limits still apply.

Each run currently builds fresh environments. Persistent environment and native-wheel
caching is a separate feature. Completed output groups work across the ordered steps:
write each marker only after its declared files are complete. A later failed step
does not turn unfinished outputs into completed artifacts.

The [two-environment worked example](../examples/multi_environment/README.md) has a
complete CPU recipe and expected recovered results. The separate
[CUDA environment example](../examples/cuda_environment/README.md) builds and runs
a small PyTorch CUDA extension with an environment-local toolkit, records compiler
and header selection, and verifies the recovered results. Its acceptance check
also confirms that the runtime's host toolkit is newer than the selected toolkit.
