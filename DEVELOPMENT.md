# Development installation (0.1.0.dev10)

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
