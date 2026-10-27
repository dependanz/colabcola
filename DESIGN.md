# colabcola design contract

**Contract version:** 0.20<br>
**Status:** The primary Drive-staged transport and both Phase 3 pilot segments are live-verified and receipt-cleaned. Contract `0.14` adds declared remote inputs: HTTPS files bound by target, URL, byte count, and SHA-256 that the runtime downloads and verifies into the job root after extraction, so large weights need not travel through Drive; the `remote_inputs` manifest key appears only when declared, and compiler contract `0.10` is unchanged. Contract `0.13` adds hash-bound, manifest-committed Drive checkpoint publication and strict same-job restoration for interruption-safe sustained runs; compiler contract `0.10` binds that optional policy into job identity.<br>
**Last updated:** 2026-10-15

This document defines the first implementable contract for compiling a trusted Python batch job
and its declared input files into a deterministic staged bundle plus a small Google Colab
controller. A warned self-contained notebook remains available as a secondary transport. The
deterministic bundle, both notebook forms, runtime failure evidence, atomic compiler output, and
generated-audio CPU round trip are implemented locally. The HiFi-GAN source verifier, provenance
payload planner, modern-runtime lock, versioned exact-continuation overlay, and count-parameterized
deterministic LJ Speech subset builder/verifier are also implemented. One exact 100-step workload
and its 10-step checkpoint recovery completed on Colab T4s. The 512+32 pilot's 0-to-2,500 segment
completed and was independently verified; its first A100 resume attempt failed before step 2,501
while restoring a CPU RNG tensor that checkpoint loading had relocated to CUDA. The corrected A100
resume then completed step 5,000, its result independently verified, and its staged payload cleaned.

## 1. Product boundary

colabcola is a compiler for batch-job notebooks. Its stable core is:

```text
local JobSpec + trusted files -> deterministic .colabcola -> temporary Drive staging
                              -> tiny .ipynb controller -> Colab runtime -> result bundle
```

Version 1 targets one Python process on one Colab CUDA GPU. It does not provide a queue, cron service,
distributed worker pool, remote shell, guaranteed accelerator allocation, or model-quality
evaluation. A later launcher may open or submit a controller, but the controller and its staging
receipt must remain independently inspectable and runnable with Colab's normal UI.

After the Colab proof, the intended product may generalize into a provider-selectable library for
whole-script/data transport, GPU VM execution, and pre-run cost estimation. That direction must
preserve a provider-neutral compiler/job/result core with replaceable execution and pricing
adapters. It is recorded now to avoid accidental Colab-only coupling, but it does not broaden the
version 1 implementation or defer the Colab and HiFi-GAN milestones.

"Complete serialization" means that the staged `.colabcola` object contains every declared source,
configuration, and data byte. It does not mean that it contains a Python interpreter, CUDA runtime,
PyTorch wheel, or every system library. Those belong to an explicit runtime contract and must be
checked before the job runs. Secondary embedded mode additionally hard-codes the staged job bytes
as base64 inside the notebook itself.

The deterministic ZIP is framed with its canonical manifest into the primary binary deliverable.
That object is temporarily staged in the user's Drive and streamed into `/content`; the controller
verifies the exact Drive identity, byte count, staged SHA-256, canonical manifest, archive SHA-256,
portable job ID, and every extracted file. A declared remote input (section 3.1, contract `0.14`) is
downloaded only from the HTTPS URL the manifest names and is accepted only when its byte count and
SHA-256 match the declaration; nothing else is downloaded from a third-party URL at runtime.

## 2. Trust and serialization boundary

- Inputs are trusted local files selected explicitly by the caller.
- Version 1 packages files and JSON-compatible metadata. It does not use `pickle`, `cloudpickle`,
  live interpreter state, captured closures, or arbitrary object deserialization.
- Every bundled target is a normalized relative POSIX path. Absolute paths, `..`, drive prefixes,
  empty components, NUL bytes, and paths escaping the job root are rejected.
- Symlinks, junctions, devices, sockets, and other special files are rejected in version 1.
- Secret values are forbidden in the `JobSpec`, bundle, generated notebook, receipt, logs, and
  result manifest. A controller may name a Colab Secret containing the limited-input OAuth client
  JSON, but neither that JSON nor a user access/refresh token may be embedded in portable output.
- Generated notebooks execute arbitrary trusted code. Opening or running a notebook from another
  party is equivalent to running that party's program and must be treated accordingly.

## 3. Public interface

The source job document uses the `.colajob` extension and contains strict JSON conforming to the
authoritative portable schema. A standard-library Python `JobSpec` API mirrors it. Version 1 does
not expose a decorator or callable serializer.

The primary command-line grammar is:

```text
colabcola [COLABCOLA_OPTIONS] JOB.colajob [JOB_ARGS...]
```

The `.colajob` path is the invocation boundary. Colabcola consumes its own options only before
that path; every token after it is forwarded verbatim to the packaged program. The effective
entrypoint argument array is the configured `entrypoint.argv` followed by the invocation's
`JOB_ARGS`. Colabcola never shell-parses, interpolates, or semantically rewrites those strings.
The normalized effective arguments are stored in the portable manifest and therefore affect the
job identity: identical inputs and arguments reproduce one identity, while changing a forwarded
argument produces a distinct job.

```python
from pathlib import Path

from colabcola import JobSpec, prepare_drive_bundle

spec = JobSpec.from_json(Path("hifigan.colajob"))
result = prepare_drive_bundle(
    spec,
    Path("dist/hifigan.colabcola"),
    job_args=("--epochs", "20", "--batch-size", "16"),
)
print(result.job_id, result.staged_sha256, result.warnings)
```

Equivalent primary CLI:

```text
colabcola --output dist/hifigan.colabcola hifigan.colajob --epochs 20 --batch-size 16
```

The bare `colabcola JOB.colajob ...` form is the stable high-level submission interface. It
validates and compiles the job, assigns its content-derived job ID, writes the exact binary object,
authenticates through the configured desktop client, stages or reuses the object, reads back the
exact Drive identity, and writes a bound controller plus credential-free receipt. Success reports
`awaiting_manual_launch`: staging is verified, but Colab has not necessarily accepted, started, or
completed the job. Compilation receipts also contain a `warnings` array.

Google Drive setup is a separate, explicit one-time operation:

```text
colabcola auth google-drive \
  --desktop-client-secrets DESKTOP_CLIENT.json \
  --colab-client-secrets LIMITED_INPUT_CLIENT.json
```

Both clients must be distinct and belong to the same Google Cloud project. The command requests
only `https://www.googleapis.com/auth/drive.file`, stores the user's desktop refresh token in the
operating-system keyring, and persists only public client identity plus the path to the user's
desktop client file. The user separately places the complete limited-input client JSON in the
named Colab Secret (default `COLABCOLA_GOOGLE_DRIVE_OAUTH`). No user credential or client secret is
copied into the `.colajob`, bundle, controller, receipt, log, or devbrain.

Explicit lifecycle commands remain available for narrower operations:

```text
colabcola validate JOB.colajob
colabcola compile [COLABCOLA_OPTIONS] JOB.colajob [JOB_ARGS...]
colabcola auth google-drive --desktop-client-secrets FILE --colab-client-secrets FILE
colabcola status JOB_ID
colabcola results JOB_ID
```

`compile` is local and performs no network access. It never starts Colab, uploads a bundle, or runs
the job, and Drive mode reports `awaiting_drive_staging`. The lower-level `controller` operation is
retained for diagnostics and binds a small notebook to independently observed Drive metadata;
normal users do not need it. `status` and `results` arrive with the launch/result adapter.
`--transport embedded` explicitly selects the warned
self-contained base64 notebook and reports `awaiting_manual_launch` in bare mode. Local CPU
fixtures have reconstructed and executed both transports; remote Drive/Colab work is claimed only
from separately recovered evidence.

### 3.1 Source `JobSpec`

The initial schema has these required top-level fields:

```json
{
  "schema_version": 1,
  "name": "hifigan-ljspeech-smoke",
  "payload": [],
  "entrypoint": {},
  "python": {},
  "runtime": {"provider": "google_colab", "version": "2026.07"},
  "resources": {},
  "run": {},
  "outputs": [],
  "limits": {}
}
```

#### `name`

- Human-readable lowercase slug matching `^[a-z0-9]+(?:-[a-z0-9]+)*$`.
- Used in display text only; it is not the content identity.

#### `payload[]`

Each entry declares a compile-time source and portable target:

```json
{
  "role": "source",
  "source": "C:/path/on/compiler/train.py",
  "target": "workload/train.py"
}
```

- `role` is `source`, `data`, `config`, `license`, or `wheel`.
- `source` is a local file or directory used only during compilation. Its absolute spelling is not
  copied into the portable manifest or content identity.
- `target` is a unique relative POSIX path beneath the reconstructed job root.
- Directories expand recursively into individual sorted file records after exclusions are applied.
- Version 1 has no implicit `.gitignore` behavior. Every exclusion must be declared by the caller.
- Exclusions are normalized POSIX globs relative to the declared source directory. `*`, `?`, and
  bracket expressions match within one path component; a complete `**` component matches zero or
  more components. Exclusions on single-file sources, absolute patterns, `..`, empty components,
  backslashes, drive prefixes, and embedded `**` fragments are rejected.

#### `remote_inputs` (optional, contract `0.14`)

```json
{
  "target": "weights/model.safetensors",
  "url": "https://huggingface.co/<owner>/<repo>/resolve/<commit>/model.safetensors",
  "bytes": 5669789744,
  "sha256": "a50ac8f3..."
}
```

- A remote input is a file the runtime downloads into the job root instead of carrying it in the
  bundle: weights or datasets that already have a stable, hash-verifiable home. The bundle stays
  small and the Drive staging unchanged.
- `target` is a unique portable path. It may not equal, contain, or lie beneath a payload target or
  another remote input target, and remote inputs count toward `limits.maximum_file_count`.
- `url` is HTTPS only, with no user information, whitespace, or fragment. Pin it to an immutable
  revision where the host offers one (a Hugging Face `resolve/<commit>` URL, a release asset); the
  identity that matters is the declared `bytes` and `sha256`, which the runtime verifies before the
  file is accepted. The URL may not carry a credential; a private source needs a transport of its
  own and is outside this contract.
- The compiler never downloads a remote input; `compile` stays offline. The declaration enters the
  portable manifest and therefore the job identity (section 4).
- At runtime the acquisition happens at the end of the extraction cell group, after the bundle is
  verified and before environment preparation (section 6); its facts are recorded in `result.json`
  (section 7).

#### `entrypoint`

```json
{
  "kind": "python_script",
  "path": "workload/train_wrapper.py",
  "argv": ["--config", "workload/config_v1.json"],
  "cwd": "."
}
```

- `kind` is `python_script` in version 1.
- `path` and `cwd` resolve beneath the reconstructed job root.
- `argv` contains reproducible defaults from the `.colajob` document. Invocation-time `JOB_ARGS`
  are appended to it, and the effective array is passed directly to `subprocess`; it is never
  interpolated into a shell string.

#### `python`

```json
{
  "requires_python": ">=3.11,<3.13",
  "runtime_requirements": ["torch>=2.6"],
  "install_requirements": [],
  "embedded_wheels": []
}
```

- `runtime_requirements` must already be satisfied by the selected Colab runtime. The preflight
  cell reports installed versions and fails before payload execution if a constraint is false.
- `install_requirements` contains exact version pins plus hashes. The environment cell may install
  them from the default package index; this is declared network access and is not offline.
- `embedded_wheels` references `payload` targets with role `wheel`; these install without network.
- Unpinned package names, VCS branches, mutable URLs, and `pip install -U` are rejected.
- System-package installation is outside version 1. A workload requiring it fails compilation.

#### `runtime`

```json
{
  "provider": "google_colab",
  "version": "2026.07"
}
```

- Version 1 requires provider `google_colab` and a runtime version of `latest` or `YYYY.MM`.
- The version is carried in portable identity, controller metadata, summary, and result evidence.
- Because runtime selection remains an explicit Colab UI action, exact Python/package/resource
  preflight is the execution authority and fails safely if the selected environment disagrees.

#### `resources`

```json
{
  "accelerator": "gpu",
  "gpu_count": 1,
  "cuda_required": true,
  "minimum_gpu_memory_bytes": null
}
```

- `accelerator` is `cpu` or `gpu`; the HiFi-GAN profile requires `gpu`.
- Resource fields are requirements to check, not promises about Colab allocation.
- `minimum_gpu_memory_bytes` is optional until a workload has measured evidence for a threshold.
- A GPU JobSpec requests a compatible capability, not a portable model guarantee. The user may
  select an offered T4, A100, or another compatible CUDA GPU in Colab; runtime preflight and
  `result.json` record the actual model, count, CUDA state, and memory, and throughput/budget
  estimates must be recalculated from that allocation.
- TPU is not an alias for this CUDA profile. Colab accelerator types and availability vary, and a
  PyTorch TPU path requires a separately implemented and verified XLA device/optimizer/data-loader
  backend. JobSpec v1 therefore rejects `tpu` rather than silently running CUDA-only code on it.
  See the official [Colab FAQ](https://research.google.com/colaboratory/faq.html) and
  [PyTorch/XLA device guidance](https://docs.pytorch.org/xla/master/learn/pytorch-on-xla-devices.html).

#### `run`

```json
{
  "timeout_seconds": 3600,
  "seed": 1234,
  "environment": {},
  "capture_stdout": true,
  "capture_stderr": true
}
```

- `timeout_seconds` is enforced by the notebook runner where the runtime permits it.
- Environment keys and literal values are included in the content identity and may not be secrets.
- The seed is passed as the reserved `COLABCOLA_SEED` process-environment variable; callers may not
  override that key in `run.environment`. Reproducibility still depends on the framework and
  hardware and is not inferred from the configured value.

#### `outputs[]`

```json
{
  "pattern": "outputs/cp_hifigan/**",
  "required": true,
  "maximum_bytes": 2147483648
}
```

- Patterns are relative to the job root and may not escape it.
- Every matched regular file is hashed into the result manifest.
- A missing required output or exceeded limit changes the terminal state to
  `output_validation_failed`.

#### `limits`

```json
{
  "maximum_file_count": 10000
}
```

- JobSpec v1 has no configured archive-size ceiling. The former `maximum_archive_bytes` field was
  removed before release because no measured provider limit justified it.
- At or above 1,024 MiB (`1,073,741,824` bytes) of deterministic ZIP data, compilation continues but
  emits a `RuntimeWarning` and records a structured warning in the bundle, Python compilation
  result, CLI JSON receipt, and notebook metadata/summary. The warning includes the exact primary
  Drive transfer floor and secondary standard-base64 byte count before notebook JSON overhead.
- `maximum_file_count` remains a hard job-level expansion limit.
- The compiler reports raw input bytes, archive bytes, base64 bytes, notebook bytes, and file count.
- Drive submission JSON reports separate compilation, Drive staging/reuse, and controller-generation
  durations. These are local operational measurements and do not imply remote execution.

## 4. Portable bundle and content identity

Compilation expands the source `JobSpec` into a portable bundle manifest. Host source paths are
removed. Each file record contains:

```json
{
  "role": "data",
  "path": "data/LJSpeech-1.1/wavs/LJ001-0001.wav",
  "size": 123456,
  "sha256": "...",
  "mode": "0644"
}
```

The manifest also contains the normalized effective entrypoint, Python environment, resource, run,
output, limit, archive-format, source-schema, and compiler-contract fields. A job that declares
remote inputs also carries `remote_inputs`, sorted by target with each entry's `bytes`, `sha256`,
`target`, and `url`; the key is absent when none are declared, so jobs without remote inputs keep
the identities they had under earlier contracts and `compiler_contract_version` remains `0.10`.
The display-only job name is carried separately and does not affect content identity. The manifest contains no
compilation timestamp, hostname, absolute source path, username, temporary directory, Git
credential, or secret.

Canonical JSON is UTF-8 encoded with sorted keys, no insignificant whitespace, `ensure_ascii=false`,
and one terminal newline. The compiler computes:

1. every file SHA-256;
2. the canonical manifest SHA-256;
3. the deterministic archive SHA-256; and
4. `job_id = sha256(canonical_manifest_bytes || archive_sha256_bytes)`.

The implementation frames each component with an unsigned eight-byte big-endian length. The
archive component is the raw 32-byte SHA-256 digest, not its hexadecimal display spelling. The
final notebook SHA-256 is reported separately and is not part of its own content identity.

## 5. Deterministic archive and transport encoding

- Files are ordered by Unicode code-point order of normalized target paths.
- The archive is ZIP with DEFLATE compression and standard-library-compatible extraction.
- Every ZIP entry uses a fixed `1980-01-01T00:00:00` timestamp, normalized creator/platform fields,
  UTF-8 names, no comments or arbitrary extras, and mode `0644` for data or `0755` only for an
  explicitly executable source file. Entries whose size, compressed size, or header offset exceeds
  Python's `ZIP64_LIMIT` carry exactly one standard `0x0001` ZIP64 extra containing only the required
  64-bit fields in specification order and use ZIP version 4.5; all other entries retain empty extras
  and ZIP version 2.0.
- Compression level is fixed at DEFLATE level 9 and recorded in the manifest.
- Primary Drive mode emits one binary `.colabcola` envelope: a 32-byte big-endian header containing
  the 16-byte `COLABCOLA\0BUNDLE` magic plus manifest/archive lengths, followed by the canonical
  manifest bytes and deterministic ZIP bytes. Its SHA-256 covers the complete framed object.
- The writer streams those three segments into an atomic output without creating a base64 copy.
  The runtime streams the staged object into `/content`, verifies it before extraction, and streams
  individual ZIP entries rather than loading large checkpoints or datasets into RAM.
- Drive objects live beneath `colabcola/staging/<job_id>/` and are selected by exact parent plus
  app-private `job_id` and staged-SHA properties. A duplicate exact match is an ambiguity error.
  Upload adapters use resumable transfer for absent objects and return read-back metadata.
- Each controller binds one exact Drive file ID, relative path, portable job ID, staged/archive byte
  counts, and hashes. Filename-only lookup is forbidden.
- A receipt limits cleanup authority to named file IDs. Normal cleanup is allowed only after an
  expected result has been recovered and verified locally by size and SHA-256; abandoned jobs need
  a separate explicit authorization. Receipts use strict canonical JSON and can be reloaded across
  processes; duplicate, unknown, mistyped, unsafe, or noncanonical fields are rejected. Cleanup
  means moving to Drive trash, not permanent deletion.
- Desktop upload uses a same-project Desktop OAuth client, the non-sensitive `drive.file` scope,
  an OS-keyring user token, and Google Drive resumable media upload in fixed 8 MiB chunks. The
  controller uses a distinct same-project limited-input client obtained from Colab Secrets, device
  authorization for `drive.file`, metadata lookup by immutable file ID, and `files.get?alt=media`.
  It never mounts the user's Drive or opens a payload file chooser.
- Secondary embedded mode splits archive bytes into raw 768 KiB slices and standard base64.
  Since 768 KiB is divisible by three, the base64 strings can also be concatenated before one
  strict decode without introducing internal padding.
- Secondary mode embeds the archive completely in the notebook and needs no sidecar. It remains
  useful for small portable jobs, but is no longer the default because large notebook JSON makes
  the Colab frontend slow even when payload cells are hidden.
- Each payload code cell carries at most 768 KiB of raw archive input, producing at most 1 MiB of
  base64 text before Python and notebook JSON syntax.
- Each base64 chunk is represented as deterministic adjacent string literals of at most 8 KiB;
  Python concatenates them at compile time. The compiler must not emit a million-character source
  line or impose a runtime join/copy merely to obtain the original chunk.
- Payload cells are ordered and numbered from zero; concatenation must reproduce the archive hash.
- Every payload cell is a Colab Form hidden by default through `cellView: form` metadata and a
  first-line `@title` declaration. Its visible title and source warning tell users to keep the
  generated payload hidden because revealing or editing it can make Colab slow or unresponsive.
- Compiled code cells have `execution_count: null` and empty `outputs`.
- Notebook JSON uses stable key ordering, UTF-8, LF newlines, no cell IDs, and one terminal newline.
  Identical inputs and compiler version must produce byte-identical notebooks.

## 6. Controller and embedded notebook metadata

The notebook uses `nbformat=4`, pinned `nbformat_minor=4`, and therefore requires no cell IDs. Its
metadata contains:

- a Python 3 kernelspec;
- Colab GPU preference metadata when `resources.accelerator` is `gpu`;
- `metadata.colabcola` with contract version, job ID, manifest hash, archive hash, compiler version,
  payload cell count, hidden-display mode, literal line length, archive-warning threshold,
  structured warnings, and the count of currently implemented ordered cell groups; and
- no local paths or compile timestamp.

The compiler verifies all seven controller groups and its exact staging binding before atomically
replacing the requested `.ipynb` output. The controller contains metadata and runtime code, never
archive bytes or base64 payload cells.

Primary controller cells appear in this exact order:

1. **Human-readable warning and job summary (Markdown).** Identifies the exact staged object and
   explains the one-time Colab Secret setup, per-file authorization, exact download, and trusted
   code execution boundary.
2. **Canonical manifest and constants (code).** Binds the Drive file ID/path and every identity,
   length, and hash needed to reject substitution or truncation.
3. **Runtime preflight (code).** Establishes durable result paths and checks the runtime contract.
4. **Drive acquisition (code).** Reads the named Colab Secret, performs limited-input device OAuth
   for `drive.file`, verifies the exact ID/parent/name/size/MIME/app-properties metadata, downloads
   that immutable object through the Drive API, and stream-verifies the complete staged envelope.
5. **Safe extraction (code).** Rechecks archive/job identity and streams verified regular files to
   `/content/colabcola/jobs/<job_id>/`. When the manifest declares remote inputs, the same cell
   then checks free disk against their declared total plus a 1 GiB margin and downloads each file
   over HTTPS into the job root, with up to three attempts and HTTP Range resumption, accepting a
   file only when its byte count and SHA-256 match the declaration.
6. **Environment preparation (code).** Applies only the declared environment contract.
7. **Execution and result collection (code).** Runs without a shell and makes success or failure
   evidence durable. When the manifest declares durable checkpoints, this cell restores the newest
   unique committed pair before launch, publishes complete pairs while the process runs, and
   records restoration, publication, and retained-step facts in `result.json`.

The following seven-group structure is retained only for secondary embedded mode:

Cells appear in this exact order:

1. **Human-readable warning and job summary (Markdown).** States that the notebook executes bundled
   code, lists the job ID and declared resource, instructs the user to inspect trusted content, and
   repeats any archive-size warning before execution. It prominently warns users to keep payload
   cells hidden and use Run All instead of expanding or editing their base64 source.
2. **Canonical manifest and constants (code).** Contains the portable manifest as a literal and
   defines only deterministic constants and standard-library imports.
3. **Payload chunks (one or more hidden Colab Form code cells).** Each cell uses compile-time
   concatenation of bounded 8 KiB literals to append one numbered base64 string. No chunk is
   evaluated as Python source, and users are warned that revealing these generated cells can make
   the Colab front end unresponsive.
4. **Runtime preflight (code).** Records Python, platform, declared runtime-package versions, CUDA
   availability, GPU count/name/memory when visible, and resource-contract pass/fail without
   aborting later failure-evidence collection.
5. **Reassembly and safe extraction (code).** Decodes chunks, verifies the archive and every file
   hash, rejects unsafe archive entries, and extracts to `/content/colabcola/jobs/<job_id>/`; then
   acquires declared remote inputs exactly as the primary controller does.
6. **Environment preparation (code).** Verifies runtime requirements, installs only declared pinned
   dependencies or embedded wheels, records the resolved environment, and rechecks constraints.
7. **Execution and result collection (code).** Runs the argv without a shell, streams and stores
   stdout/stderr, enforces the timeout, validates safe regular-file outputs and per-pattern byte
   ceilings, writes the result manifest and deterministic-layout ZIP, prints their paths, and
   re-raises a terminal exception only after failure evidence is durable.

A cell may not clone the workload source or download undeclared job data. Primary controller mode
may authorize and download its exact staged Drive object. A job with an explicit
`durable_checkpoints` contract may additionally create and access only its app-property-bound
checkpoint folder beneath the staged job folder. A job with `remote_inputs` may download exactly
the declared HTTPS URLs into the declared targets, each verified by byte count and SHA-256 before
acceptance. Declared dependency installation is the other version 1 runtime network operation.

## 7. Runtime and result contract

CLI lifecycle states such as `awaiting_drive_staging` and `awaiting_manual_launch` describe only what the local client or launch
adapter has established. They are separate from notebook runtime terminal states and cannot prove
remote execution or success; only recovered runtime evidence in `result.json` can do that.

The reconstructed job root is `/content/colabcola/jobs/<job_id>/`. Results are written beneath
`/content/colabcola/results/<job_id>/`:

```text
result.json
stdout.log
stderr.log
artifacts.zip
```

`result.json` contains at least:

- job ID, contract/compiler versions, manifest/archive hashes, and a nullable notebook hash;
- terminal state: `success`, `preflight_failed`, `integrity_failed`, `remote_input_failed`,
  `environment_failed`, `job_failed`, or `output_validation_failed`;
- process exit code or `null` when execution never started;
- UTC start/end timestamps and duration, which are runtime evidence and are not written back into
  the compiled notebook source;
- primary-Drive authorization/metadata, download, staged-envelope reconstruction, total acquisition,
  and archive-extraction durations when that transport is used;
- for a job with remote inputs, `transport.remote_inputs` with the count, total bytes, and seconds,
  and per file the target, bytes, SHA-256, HTTP status, attempts, and seconds; a disk shortfall
  before the first download ends in `preflight_failed` and any download, size, or hash failure in
  `remote_input_failed`, both with failure stage `remote_inputs` and the partial file removed;
- observed Python, package, CUDA, and GPU facts;
- stdout/stderr hashes and truncation indicators;
- every output path, byte count, and SHA-256; and
- failure stage, exception type, and sanitized message when applicable.

The compiler receipt contains the verified final notebook SHA-256. The running kernel cannot
derive that same value from the notebook source without a self-reference or frontend/launcher
handoff, so `result.json.notebook_sha256` is `null` in the standalone UI workflow. A future launch
adapter may populate it only from an out-of-band verified compiler receipt; the kernel never
invents or approximates the value.

`artifacts.zip` contains only validated output matches and `result.json`. Its contents are safe-path
checked and deterministically ordered, although runtime-produced file bytes and timestamps are not
assumed reproducible. After packaging, the execution cell stream-hashes the complete ZIP and prints
its exact name, byte count, and SHA-256 so the local lifecycle client can verify a downloaded result
against evidence produced before transfer.

## 8. HiFi-GAN Phase 2-3 workload profile

### 8.1 Source and model identity

- Model: HiFi-GAN neural vocoder by Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae, arXiv
  `2010.05646`.
- Reference repository: `https://github.com/jik876/hifi-gan.git`.
- Pinned upstream commit: `4769534d45265d52a904b850da5a622601885777`.
- Generator profile: upstream `config_v1.json` (22,050 Hz, 80 mel bands, segment size 8,192) with
  only explicitly documented smoke-test overrides.
- The upstream MIT license file and source revision metadata are bundled with role `license`.
- The checked-in `src/colabcola/_hifigan/profile.json` records byte counts and SHA-256 for eleven
  upstream authorities: the license, README, V1 configuration, runtime source modules, historical
  requirements, original trainer, and both upstream LJ Speech split lists. The source planner
  rejects missing or changed bytes before creating payload declarations.

The upstream historical requirements file is provenance, not the Colab environment lock. The
selected lock targets the pinnable Colab runtime `2026.07`: Python `3.12.13`, PyTorch
`2.11.0+cu128`, NumPy `2.0.2`, librosa `0.11.0`, SciPy `1.16.3`, TensorBoard `2.20.0`, SoundFile
`0.14.0`, and Matplotlib `3.10.0`. These are exact preinstalled-runtime requirements from the
official GPU package snapshot at backend-info commit
`ec53f93fe44a9cab118dc5451527b18e36e1a643`; the workload requests no pip installation. The
notebook preflight must reject any mismatch. The snapshot and local compatibility tests define a
configured lock, not proof of a successful production Colab run. No training run may proceed
merely because `pip` accepted some dependency combination.

### 8.2 Deterministic LJ Speech subset

- Dataset family: LJ Speech 1.1, public dataset page
  `https://keithito.com/LJ-Speech-Dataset/`.
- Source split authority: the training and validation file lists at the pinned HiFi-GAN commit.
- Selection rule: the lexicographically first declared positive number of unique clip IDs within
  each pinned split. Phase 2 defaults to 64 training plus 8 validation; Phase 3 declares 512 plus 32.
- The two sets must be disjoint.
- Only the selected mono 16-bit PCM, 22,050 Hz WAV files and generated subset lists enter the
  notebook payload. Transcripts are not required by the upstream vocoder training path and are not
  bundled unless implementation evidence proves otherwise.
- The subset builder records every clip ID, relative path, WAV-header facts, duration, byte count,
  and SHA-256 plus aggregate counts, duration, raw bytes, archive bytes, and base64 bytes.
- The builder and manifest verifier accept only explicit positive, non-boolean counts and may not
  silently drop clips to reduce artifact size. Every published manifest carries the complete ID
  lists and measured split-local counts.

The implemented builder streams the official archive, retains only the selected files, rejects
changed split authorities, invalid WAV headers, or undeclared subset-tree entries, and writes
atomically. It emits a canonical manifest that the payload planner must verify. The Phase 2
selection contains 72 clips totaling 472.667210884 seconds; its 74 payload files occupy 20,848,584
raw bytes and deterministically compress to 17,919,360 ZIP bytes. The Phase 3 selection contains
544 clips totaling 3,559.823673469 seconds; its 546 payload files occupy 157,018,144 raw bytes and
deterministically compress to 134,195,543 ZIP bytes. Both remain below the 1,024 MiB warning.

The full 2,748,572,632-byte LJ Speech download is never a job payload. It was discarded after Phase
2 construction, reacquired with the same exact SHA-256 for Phase 3, and is retained only as a local
reproducible construction input until a separately recorded lifecycle decision removes it.

### 8.3 Training adapter

The pinned upstream trainer is wrapped or minimally patched because it exposes epoch limits but no
exact optimizer-step stop. The overlay is itself bundled source with a recorded patch/hash and must:

- accept `--maximum_steps` and stop after exactly that many completed generator/discriminator
  optimizer steps;
- force paired generator and optimizer/discriminator checkpoints at the final step;
- emit newline-delimited structured metrics with step, generator loss, discriminator loss,
  mel-spectrogram error, seconds per batch, and finite/non-finite status;
- preserve upstream checkpoint loading and record the resumed starting step;
- persist an exact versioned epoch/minibatch continuation cursor, ordered training-file identity,
  configuration identity, Python/Torch CPU/Torch CUDA RNG state, and both scheduler states;
- normalize every restored Torch RNG byte tensor back to CPU after the upstream device-mapped
  checkpoint load and before calling the CPU or CUDA RNG setters;
- reject a resume when any state version, dataset/configuration identity, batch geometry, seed,
  cursor equation, upstream revision, or checkpoint filename disagrees;
- make per-sample random cropping independent of DataLoader worker scheduling and resume prefetch;
- write each local checkpoint member through a same-directory temporary file, flush and fsync it,
  and atomically replace the final filename before it is eligible for publication;
- retain only the declared newest number of complete checkpoint pairs;
- report the exact retained checkpoint-step list in the durable `run_complete` metric so retention
  can be verified without returning every gigabyte-scale pair;
- reject CPU execution for this profile before training begins; and
- keep all outputs below `outputs/` in the reconstructed job root.

The implemented `train_colabcola.py` and `resume_state.py` overlays retain the upstream model,
losses, and optimizer path. They adapt librosa's keyword-only mel-filter call and PyTorch's complex
STFT API, use an explicit `completed_steps` checkpoint field while retaining the historical
zero-based `steps` field, validate paired checkpoint filenames, and derive a deterministic crop
seed from the run seed plus epoch/sample position/name. The loader starts at the recorded next
minibatch, so prefetched-but-uncommitted samples are regenerated rather than skipped. Checkpoints
store state version 1, the exact cursor and identities, all declared RNG/scheduler state, and prune
to `--checkpoint_retention` newest complete pairs (default 2). The trainer appends durable NDJSON
metrics, forces a final pair when needed, accepts exactly one visible compatible CUDA GPU, and
records the retained checkpoint-step list before constraining metrics/checkpoints beneath
`outputs/`. Because the retained upstream loader applies `map_location=device` to every checkpoint
tensor, the overlay explicitly moves the saved CPU and per-GPU RNG byte tensors back to CPU before
passing them to PyTorch's RNG setters. Both checked-in overlay byte counts and SHA-256 values are
part of the workload profile and are verified before payload planning.

This is exact continuation of the declared sample/crop/cursor and saved training state. It is not a
promise of bit-identical floating-point weights when moving between GPU architectures: cuDNN
benchmark selection remains enabled, and T4-to-A100 numerical kernels may differ. The actual GPU
identity therefore belongs in run evidence, not checkpoint compatibility identity.

### 8.3.1 Durable checkpoint transport

Durable checkpointing is an optional JobSpec contract containing the job-relative checkpoint
directory, exact generator/state filename prefixes, fixed decimal width, polling interval, and
remote retention count. It is valid only with Drive transport. The controller keeps refreshable
OAuth credentials in notebook memory only, never writes them into the job root or result evidence,
and clears them after execution.

For each newly complete local pair, the controller hashes both stable files, uploads and verifies
the generator and state objects, then uploads a canonical JSON manifest last. That manifest is the
only commit marker and binds job ID, step, member IDs, names, sizes, and SHA-256 values. Orphan
member uploads are never resumable state. Resume discovery rejects duplicate manifests for one
step, malformed manifests, metadata mismatches, missing members, and content-hash mismatches, then
atomically restores only the highest valid committed step before starting the workload.

Remote retention defaults by contract to at least two committed pairs. After a newer manifest is
committed and verified, pruning trashes an obsolete manifest first and only then its two members;
therefore interruption during pruning can produce harmless orphans but cannot make an obsolete
pair appear committed or delete the sole recovery pair.

### 8.4 Smoke and resume budgets

The completed Phase 2 target was:

1. train from initialization through exactly 100 completed optimizer steps;
2. force paired checkpoints at step 100;
3. start a second invocation from those checkpoints;
4. train exactly 10 additional steps, ending at step 110; and
5. force and collect paired checkpoints at step 110.

Checkpoint interval 50 may produce an intermediate pair, but the final-step pair is mandatory even
when the interval does not divide the stop boundary.

### 8.5 Acceptance evidence

Phase 2 required and now has the following evidence across its exact smoke/resume lifecycle:

- manifest, archive, notebook, source-overlay, dataset-file, checkpoint, and result hashes;
- observed Colab runtime, Python, CUDA, PyTorch, GPU model/count/memory, and dependency versions;
- exactly 64 training and 8 validation clips with measured aggregate duration and bytes;
- exactly 100 new steps followed by a verified resume for 10 more steps;
- finite generator, discriminator, and mel-spectrogram metrics at every recorded point;
- paired step-100 and step-110 generator/training-state checkpoints;
- proof that the resumed invocation began from step 100 rather than reinitializing;
- successful required-output validation and a result bundle; and
- notebook size, reconstruction time, environment time, training time, throughput, peak GPU memory
  when measurable, and output size.

This is evidence that colabcola transported and executed a real GPU training job. It is not evidence
of HiFi-GAN convergence, perceptual quality, or reproduction of published results.

### 8.6 Phase 3 sustained pilot

The pilot uses the verified 512-training/32-validation subset and two sequential invocations:

1. initialize at step 0 and stop exactly at completed step 2,500;
2. recover and independently verify the first result outside Colab;
3. create the second JobSpec from only the verified step-2,500 pair and continued metrics;
4. resume at the exact recorded cursor and stop at completed step 5,000; and
5. verify both results and clean each receipt-scoped staged input only after recovery.

Each segment has a 14,400-second entrypoint timeout, checkpoint and validation interval 500,
checkpoint retention 2, and exact final generator/state/configuration/metrics outputs. The first
JobSpec expands to 559 files: a 102,057-byte canonical manifest plus a 134,277,402-byte deterministic
archive framed as a 134,379,491-byte staged object. It has zero warnings and portable job ID
`6e15cecb1498661cd853a41c026f548b1bf627a7c1b970b3559a899e4eb5e929`. These local facts authorize
neither Drive transfer nor GPU execution by themselves.

Pilot acceptance additionally requires consecutive finite steps, run markers proving 0â†’2,500 and
2,500â†’5,000, state version/cursor/dataset identity agreement, a final retained-step list of no more
than two complete pairs, observed runtime/package/accelerator facts, final pair hashes, verified
result bundles, and receipt-scoped cleanup. T4 or A100 may satisfy the CUDA capability; the actual
allocation controls the measured throughput. TPU does not satisfy this pilot contract.

### 8.7 Phase 3 A100 calibration gate

The first full-corpus funded experiment is exactly 0â†’100,000 completed steps on a user-selected
A100. Its checked-in JobSpec uses the complete 12,950-training/150-validation LJ Speech split,
25,000-step checkpoint and validation intervals, three-pair local retention, two-pair committed
Drive retention, and a 36,000-second process timeout. It publishes checkpoints from
`outputs/cp_hifigan` using exact `g_########`/`do_########` pairing and requires the exact
step-100,000 pair plus configuration and metrics as terminal outputs.

Compilation, staging, A100 selection, authorization, execution, artifact recovery, compute-unit
measurement, and final Drive cleanup remain explicit user-operated actions. The calibration is a
budget gate rather than a paper-quality claim: its recovered timing and observed A100 compute-unit
rate determine whether the remaining 2,407,900 steps fit the available balance with reserve.

## 9. Phase 1 implementation sequence

1. Add the package layout, portable JSON schema, standard-library dataclasses, and strict parser.
2. Implement `.colajob` routing, the configuration-path argument boundary, effective-argument
   normalization, and command parsing tests without weakening the existing explicit `validate`
   operation.
3. Implement path normalization, file expansion, hashing, canonical manifest generation, and the
   deterministic ZIP writer.
4. Implement base64 chunking and deterministic notebook emission.
5. Implement notebook runtime helpers for preflight, safe extraction, environment validation,
   subprocess execution, failure capture, output validation, and result packaging.
6. Add unit tests for invalid paths, symlinks, duplicate targets, deterministic archives, chunk
   corruption, hash mismatch, missing runtime requirements, failure states, and output limits.
7. Add a small generated-audio CPU fixture and prove byte-identical compilation plus local notebook
   reconstruction before acquiring LJ Speech or running HiFi-GAN.

## 10. Change control

Changes to the CLI argument boundary, portable schema, content identity, archive normalization,
notebook cell order, security boundary, or HiFi-GAN acceptance contract require a contract-version
decision in the authoritative devbrain plan. Measurements and observed results belong in the
devbrain numbers ledger; generated notebooks, datasets, checkpoints, and logs remain outside
devbrain.

## Managed lifecycle extension (0.15)

The optional managed_recovery policy binds checkpoint/log patterns into job identity
without changing existing jobs or compiler contract 0.10. The supervisor and detached
worker implement the bounded protocol in LIFECYCLE.md. Local tests and the bounded
WSL/T4 deliberate replacement proof pass: checkpoint 25 restored on a distinct runtime,
continuation through step 2000, verified Drive/local artifacts and owned-compute release.
Native Windows provider support and public-product onboarding remain separate work.


## Contract 0.16: ordinary command packaging

`colabcola package` is an offline front end to the unchanged strict JSON JobSpec v1
and deterministic bundle contract 0.10. It accepts standalone scripts, explicitly
included project trees for `python -m`, and JSON arrays of ordered Python commands.
It snapshots inputs into a sibling `.files` directory and generates a product-owned
launcher without requiring workload imports, decorators, builders or a special layout.
The launcher invokes fresh Python processes without a shell, stops on the first
failure and shares the job filesystem. A sequence restarts from the beginning after
interruption; submission-time arguments are accepted only for a single command.
POSIX execution timeout kills the whole workload process group; managed execution
also propagates its outer deadline into the workload's timeout. Native Windows local
packaging is tested separately from the still-unverified native Windows provider.

The optional `python.pip_requirements` field contains ordinary PyPI constraints
installed remotely and recorded with resolved versions. The field is omitted from
the portable manifest when empty, preserving old job identities. It is deliberately
not hash-locked resolution. Existing `install_requirements` remains hash-checked;
`runtime_requirements` remains a preflight requirement on already-installed packages.
URL/local/VCS requirement references and pip-option lines are rejected. The schema
and parser expose the new opt-in field; credentials and host helper argv are never
part of the portable payload. See PACKAGING.md for the precise inclusion policy.

The packager can generate an adjacent bounded execution configuration when Drive
and token-helper settings are supplied. All generated jobs use managed supervision,
including print-only programs without checkpoints. Absent a declared checkpoint,
a lost runtime restarts commands from their beginning. For one command, an explicit
resume option can receive the restored integrity-bound checkpoint manifest path;
restoring application state remains the program's responsibility. Packaging itself
performs no network access, dependency installation, source execution or allocation.


## Contract 0.17: longer managed runs and explicit runtime selection

Managed plans now accept explicit 60-86400-second shared wall budgets and retain
the 1-3 attempt ceiling. Defaults stay bounded at 900 seconds. This is not a provider
lifetime or billing guarantee. New plans bind renewable-v1 authorization: the host
keeps refresh credentials, renews short-lived access independently of transfers,
and atomically delivers only access/expiry. Runtime execution has an independent
deadline/authorization watchdog; committed Drive snapshots remain the recovery
authority. See LIFECYCLE.md for the 600/120-second renewal/stop margins.

An explicit rest backend uses Google's documented Runtime.version and runtimeSpec
fields, asynchronous operations, stable runtime names and UUID4 request IDs. It
requires exact returned specifications and dated version, with no Latest fallback.
Uncertain creates are reconciled without issuing new creates during cleanup. An
unknown outcome is not absence; deletion must be confirmed. high_mem requests
host memory only; existing minimum_gpu_memory_bytes preflight still gates work.

JobSpec schema 1 and compiler contract 0.10 remain unchanged. Old prepared journals
keep their exact policy and frozen worker files. New provider/version policy requires
a new plan; no existing job or journal is silently migrated. In contract 0.17 snapshot/result caps were 1 GiB; contract 0.18 below extends
outputs. Prepared inputs remain 512 MiB. Long-run renewal and REST behavior have
offline tests; live REST/account/date and multi-hour execution are not yet verified.


## Contract 0.18: streamed five-GiB results and observable renewal

Managed final verification supports 5 GiB of expanded workload outputs plus 64 MiB
result metadata. Schema2 snapshots have a 6 GiB aggregate transfer cap for ZIP
overhead and terminal evidence; 8 MiB chunks, 4096 members and 4 MiB commit manifests
remain bounded. No increase to the 512 MiB prepared input limit or schema1 64 MiB
snapshots. Packager defaults are unchanged; output declarations remain explicit.

Runtime validation and ZIP64 packaging hash/write output streams, checking that
bytes still match validation. Publication rejects growth beyond the prechecked
size. Recovery retains hash-verified chunk objects across interrupted downloads
and only exposes a complete snapshot after member verification. Disk requirements
and cache retention are documented in LIFECYCLE.md.

Renewable workers record public evidence of actual token changes used for Drive
requests and successful immutable publication/readback after initial expiry. No
credential values or hashes leave the private runtime control channel. The new
ordinary CPU example prepares a bounded test; local tests do not assert that REST
dated images or sustained live authorization work for the account. Existing prepared
workers stay frozen. JobSpec schema 1 and compiler contract 0.10 are unchanged.


## Contract 0.19: primary workflow without beta Colab REST

Packaging, unattended execution, logs, checkpoint recovery and verified download
use the official Colab CLI session adapter (`legacy`) by default. REST is deferred
until non-beta availability is re-evaluated; it remains explicit only for historical
plans and offline adapter tests. No old job or approval hash is rewritten.
The failed REST probe created zero runtimes. Earlier Hyperplus, SIREN, Kodak and
GaussianAvatar managed runs already used the CLI session adapter.

Provider selection rejects unknown backends and dated legacy plans without creating
a provider. CLI release confirms exact endpoint absence before recording success.
The renewal example accepts legacy/Latest evidence, records observed Python, and
checks cold Drive bytes and absence through the same selected adapter. It never
claims dated-image acceptance for Latest. Genuine token rotation and verified
publication at least120 seconds after initial expiry remain mandatory.

The inspected official CLI cannot request a dated image. A separate job Python
interpreter on Latest is a possible future compatibility path, not equivalent to
a dated image and not yet implemented. LongCat compatibility remains unverified.
No runtime/workaround may evade account permissions, quotas or service lifetime.
JobSpec schema1, compiler contract0.10 and transfer limits are unchanged.


## Contract0.20: explicit isolated workload Python

JobSpec schema1 gains optional python.isolation (kind, exact3.12 version, HTTPS URL,
bytes, SHA-256). Its full descriptor participates in portable identity. The archive
is verified before bounded safe extraction and creation of a venv without system-site
packages. The notebook control interpreter remains responsible for transport and
supervision; dependency installation, version/marker probes and workload processes
use the declared interpreter. No fallback to control Python is permitted.

For isolated jobs, dependency checks precede remote-input downloads. Host Python/pip
configuration cannot redirect installs or inject packages. Managed logs include the
setup log. Result environment.isolation reports the actual version, prefixes, paths
and archive descriptor; runtime.python continues to describe the control interpreter.
Absent isolation preserves existing runtime/environment semantics. The historical
manifest serialization identifier0.10 is unchanged; the descriptor and compiler
version0.1.0.dev2 distinguish this implementation and its frozen notebooks.
