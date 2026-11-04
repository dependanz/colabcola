# Development installation (0.1.0.dev24)

This preview has not been published to PyPI. Python 3.12+ is required. Install the
single development wheel in a fresh environment; managed execution currently uses
Linux/Ubuntu WSL. Native Windows can package jobs and run local tests.

```bash
python3.12 -m venv "$HOME/.venvs/colabcola-dev24"
source "$HOME/.venvs/colabcola-dev24/bin/activate"
python -m pip install /path/to/colabcola-0.1.0.dev24-py3-none-any.whl
python -m pip check
colabcola lifecycle doctor
colabcola login --profile danzel
```

A normal install resolves public Python dependencies and includes privately
namespaced copies of the inspected Colab client and kernel fork, with upstream
licenses, source hashes and reproducible vendoring provenance. No provider
wheelhouse or consumer-written token helper is needed. The compatibility `[colab]`
extra remains accepted but adds no dependencies. `gcloud` is a separate prerequisite
for the automatic Cloud project setup route, not a Python dependency.

## Personal project setup through login

The canonical user walkthrough, illustrations and troubleshooting are in
[README: user-owned project setup](README.md#user-owned-project-setup).
Login prompts remain in the terminal; Google consent uses the browser.

The implementation separates Cloud management from workload authorization:

- The automatic path invokes `gcloud` with an isolated management configuration at
  `CONFIG_HOME/cloud-setup/gcloud` and an explicitly selected account. Discovery
  selects projects labeled `colabcola-managed=true`; an explicit `--project`
  selects another existing project the user controls.
- New-project intent and its exact ID are persisted before approval and creation.
  An ambiguous permission/not-found response is not treated as proof of absence.
  Resuming setup retains that ID and reconciles the approved operation.
- Automatic checks require project permissions, unlinked billing and Drive API
  enablement. Linked, enabled or unknown billing fails closed. The manual Console
  route records project/API/billing confirmations as `self_reported`.
- Desktop client import checks type, fixed Google endpoints and exact project
  binding. OAuth-client creation and terms acceptance remain manual Google steps.
- When a selected project needs a client, automatic discovery is bounded and
  nonrecursive within the current user's Downloads locations. Linux uses XDG
  Downloads or its home fallback; WSL additionally resolves the current Windows
  Downloads folder. The native Windows helper uses the Known Folder location;
  native Windows managed execution remains unsupported.
- Only Google's `client_secret_*.json` names, including numbered copies, are
  candidates. Files must be stable, regular, bounded and pass Desktop endpoint
  and exact-project validation. Equivalent validated clients coalesce; distinct
  matching clients require user selection. Discovery never chooses by recency.
- Each scan is capped across all supplied locations at 8 directories, 2,048
  directory entries and 64 matching filenames; each file is limited to 16 KiB.
  An incomplete initial result falls back to manual selection. Incomplete results
  during an active watch cannot establish uniqueness and are retried.
- Interactive Linux/WSL login polls once per second with no overall timeout.
  Auto-selection requires two matching complete scans. Submitted input takes
  priority; unfinished path text is discarded before advancing to authorization.
  `later`, blank input or Ctrl+C pauses. Ambiguity produces numbered choices or
  an explicit path, never a newest-file heuristic.
- Noninteractive login checks existing stable candidates without a background
  download wait. `--drive-client`, a reusable imported client and
  `--no-auto-client` bypass discovery. Restoration for a previously bound profile
  additionally requires its exact recorded digest of the normalized client.
  The source download is left untouched; selection logs an escaped path and
  project ID, not client-secret or token values.
- Colab authorization uses the pinned provider identity; Drive authorization uses
  the personal Desktop client with PKCE and a loopback callback. Login verifies
  matching stable Google identities and an owned, writable Drive workspace before
  committing the profile. It lists assignments without allocating compute.

The same `colabcola login --profile danzel` resumes setup with its saved project.
A matching stable download can be detected on resume; `--drive-client PATH` selects one explicitly
and `--no-auto-client` keeps manual path entry. `--status` reads local metadata
without credentials or network, while `--force` requests consent again.
A dev18 record stopped before creation intent was recorded may ask once whether
its saved ID should be created or reused; that ID remains unchanged.

The personal Drive client identifies the user's Cloud project for API use.
Google controls account storage, Colab compute, quotas and applicable charges.
Setup neither links billing nor buys a Colab subscription. Testing-token expiry,
manual Production decisions and account migration are documented in the README.

## Configuration and compatibility

Named profiles live at `~/.colabcola/configs/NAME`. For a fresh configuration,
choose a new, unused name; an existing name resumes its saved login:

```bash
colabcola login --profile danzel --status
# An unused name reports state: not_configured without network access.
colabcola login --profile danzel
colabcola package --profile danzel -- python train.py
colabcola --profile danzel train.colajob
```

Names are 1–64 lowercase letters, digits, underscores or hyphens, start with a
letter or digit, and exclude Windows reserved device names. The same `--profile`
option works after `login`, `package` and `run`, or before a direct job path.
Explicit `--profile` overrides environment selections; combining it with
`login --config-home` is an error. Login never changes a persistent active profile.

Without an explicit selector, resolution is `COLABCOLA_CONFIG_HOME`, then
`COLABCOLA_PROFILE`, then an existing `~/.config/colabcola`, then the new default
`~/.colabcola/configs/default`. Legacy files are never moved, copied or deleted.
Existing environment-selected directories remain supported. To use plain commands
in one shell, clear a custom root only when intended, then set the profile:

```bash
unset COLABCOLA_CONFIG_HOME
export COLABCOLA_PROFILE=danzel
colabcola login --status
```

`login --config-home PATH` selects a directory for that invocation; it does not
change future commands. Keep `COLABCOLA_CONFIG_HOME` set consistently when using
this custom-path workflow. Offline status reports the absolute `config_home` and
the profile name when using a named profile; it does not check Google credentials.
`--force` renews consent, not the profile's account/project/client binding.

Packaging does not read credentials or bind the job to an account. Its default
`.run.json` account selector is resolved when preparing/executing. Prepared helper
argv and existing explicit helper files keep their absolute configuration paths;
they are not migrated. Resume with the same profile and frozen journal.
Keep configuration, downloaded client JSON and user tokens outside workload
projects. Cloud-management credentials remain separate from the narrow Drive token
helper used by managed jobs.

Existing working personal profiles and dev17 shared-client profiles bypass fresh
project setup when no project/client change was requested; login explicitly states
which configuration it reuses. A client-only configuration without a completed
profile can be adopted through manual Drive-enabled/no-billing confirmations,
recorded as user-reported rather than machine-verified checks. A fresh dev24 setup
requires a personal client and never silently falls back to the bundled publisher
client. The `publisher_oauth.json`
resource remains only for compatibility with existing profiles. No new consumer
needs an invitation to the publisher's test-user list for the personal Drive app.
The separate Colab provider's eligibility and availability still depend on Google.

## Build and verification

From the development checkout, build one ordinary wheel:

```bash
python -m pip install build
python -m build
```

For the staged development build and provenance output, use a fresh output directory:

```bash
python tools/build_development.py --output /path/to/named-profiles-dev24
```

The build includes the vendored provider sources and their provenance. It does not
publish packages, read user credentials or launch compute. Local acceptance must
exercise named-profile isolation and validation, selector precedence, offline
status, fresh configuration, setup interruption/retry, client/account mismatch,
legacy-profile reuse without migration, and package/run behavior with fake cloud
services. Use temporary homes/configuration roots for all local checks; do not
read a real account's saved login or contact Google to validate profile selection.
Windows and WSL local checks are distinct from real Google acceptance. Earlier workload
runs and another account's successful setup do not establish a new user's Google
service eligibility, refresh-token lifetime or runtime availability.

References: [Desktop OAuth and PKCE](https://developers.google.com/identity/protocols/oauth2/native-app),
[Drive scopes](https://developers.google.com/workspace/drive/api/guides/api-specific-auth),
[Testing restrictions](https://support.google.com/cloud/answer/15549945?hl=en).

## Historical dev1-dev13 installation and evidence

The sections below preserve earlier development history. Their separate provider
wheelhouse, dependency and rebuild instructions do not describe the current dev24
installation. Use the commands above for a fresh installation.

### Historical development installation (0.1.0.dev10)

This preview is local only. No PyPI release or open-source license for colabcola has
been declared. The ordinary packager/compiler supports Python 3.11+. Unattended
provider operation uses Python 3.12+ on Linux/WSL; native Windows runtime control
is not yet accepted. Workload dependencies belong to the job, not this install.

## Install from the development wheelhouse

Create an empty environment, then install the wheel (no editable checkout):

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --find-links /path/to/wheelhouse 'colabcola[colab]==0.1.0.dev10'
.venv/bin/python -m pip check
.venv/bin/colabcola --version
.venv/bin/colabcola lifecycle doctor
```

For native Windows packaging only, install `colabcola==0.1.0.dev10` without `[colab]`
using the same `--find-links` directory in a fresh Python environment. Doctor checks
local dependency identity without reading credentials or contacting Google.
Installation and doctor do not authorize, upload, run workloads or allocate compute.
Google/Drive sign-in and host token-helper configuration remain separate one-time
steps documented in LIFECYCLE.md; a new environment does not carry those credentials.

## Why the provider wheels are included

The inspected Google CLI commit `79cd126383e72b1e911f20b7030e4247f7861704`
requires Jupyter kernel client0.9.0, but its uv lock uses Google's fork at
`f18e982c3265df5e923aa9def101ab3fd737e139`, which reports0.8.0. The live proofs
used that fork. Installing the CLI alone silently picks a different implementation;
forcing the fork with `--no-deps` leaves a broken environment.

The build tool exports those exact commits and builds **local modified distributions**:
`google-colab-cli==0.1.dev70+g79cd12638.colabcola1` and
`jupyter-kernel-client==0.8.0+colabcola.f18e982c`. Changes are limited to version
labels and correcting the CLI dependency to that fork's local version. Runtime code
and upstream Apache/BSD license files are preserved. These are not official upstream
releases. The colab extra pins both local versions, so an unrelated index wheel cannot
satisfy them. Doctor checks canonical LF provider source hashes, accepting Windows CRLF checkouts. No VCS dependency or Git executable
is required on the installing machine; pip still downloads normal public dependencies.
Retain all three wheels together. A public provider distribution strategy remains a
release gate; never publish these local provider builds under upstream project names.

## Rebuild

Use a disposable build environment with `build`, `setuptools>=68`, `wheel`,
`hatchling` and `hatch-vcs`. Clone the two official provider repositories, then:

```bash
python tools/build_development.py --cli-source /path/to/google-colab-cli --kernel-source /path/to/jupyter-kernel-client --output /new/wheelhouse
```

The tool exports pinned commits, leaves both checkouts untouched, stages an explicit
colabcola source allowlist, builds an sdist and its wheel, retains provider licenses,
and writes SHA-256 provenance. It neither installs nor publishes. Generated artifacts
belong outside the source tree. No jobs, datasets, model weights or credentials are
included. Tests/examples remain in the development checkout, outside distributions.

Release gates: user-selected license, reviewed/version-controlled source snapshot,
public provider dependency distribution, clean supported-platform testing, authentication
onboarding and explicit publication approval. This dev wheel does not close those gates.

Version0.1.0.dev2 adds explicit checksum-pinned Python3.12 workload isolation; the
0.1.0.dev1 wheel remains retained historical evidence. See PACKAGING.md for the
isolation option and LONGCAT_READINESS.md for the prepared managed consumer.

Version 0.1.0.dev4 fixes connection cleanup after kernel startup fails: cleanup never requests a client for an unstarted kernel, preserves the original error, and retains genuine teardown diagnostics. The compiler fix remains included.

Version 0.1.0.dev5 adds general packager options `--wheel` and `--remote-inputs`; the cleanup and compiler fixes are retained.

Version 0.1.0.dev6 adds command-first guided packaging, reusable TOML .colaml answers, source inspection and installed console entrypoints. See PACKAGING.md. Packaging remains local.

Version 0.1.0.dev7 adds verified remote-input byte progress, cached verified snapshot manifests and bounded root-error excerpts. The LongCat allocator correction is a consumer setting, not a dependency change.

Version 0.1.0.dev8 prefers --colaml and --save-colaml for configuration files. The previous --answers and --save-answers options remain compatible aliases; job contents and runtime behavior are unchanged.

Version 0.1.0.dev9 reduces repeated log transport,uses direct log commits and emits verified progress before slow provider checks. Renewal delivery skips unchanged leases and reconciles validity separately from retries using worker publication evidence. Local native/WSL regressions cover safety and process handshakes;new cloud latency remains unmeasured. Frozen prior controllers and journals are not upgraded in place.


Version0.1.0.dev10 adds optional completion-marker output groups,automatic late
catalogue recovery,and guarded runtime-loss confirmation. Local filesystem
transports and real Linux worker interruption test persistence without Colab.
The dev9 live assignment-loss cause remains unknown;no dev10 cloud acceptance is
claimed. Build/install into a fresh environment and prepare a fresh journal to
adopt the worker changes. Existing frozen controllers and journals remain evidence.

Version0.1.0.dev11 tracks genuine worker execution in the owned notebook kernel,
reconnects without resubmission,and adopts renewed Drive access outside its busy
execution queue. `kernel-v1` requires a fresh prepared journal. Local real IPC
kernel tests reproduce detached-worker idleness and verify busy execution,token
handoff,reconnect and actual compiled-worker artifact recovery. The dev9/dev10
service termination cause remains inferred,not confirmed;no new runtime was used.


A subsequent separately authorized dev11 live proof completed all 8 private
LongCat cases through the public `.colaml` packager on one A100 80 GB. Natural
renewal, post-expiry publication, all 8 pre-final clip commits, cold hashes/full
decoding and independent runtime absence pass in 9468.904660 controller seconds.
Independent checks finished at 9674.901152 seconds, within 10800. See
`NONREST_REVALIDATION.md`. Code and the installed controller were frozen; this
input-driven replay changes documentation and experiment artifacts only.
Residual sampled log delivery delay remains: 42.076150-second median and
90.132747-second maximum. This run does not establish a log-delay fix.
