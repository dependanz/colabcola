# Colab Cola — Legacy manual controller

[← Back to the main README](../README.md)

This workflow is retained for compatibility with earlier `.colajob` jobs.
For new runs, prefer the managed CLI described in the main README.

## Legacy manual controller setup

The manual controller flow below is retained for existing jobs. It is not required
for the install, login, package and managed-run workflow above.

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
