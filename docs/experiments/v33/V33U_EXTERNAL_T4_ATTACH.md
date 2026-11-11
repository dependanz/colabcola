# v33u — user-owned, browser-provisioned T4 attachment

This is an opt-in, **non-owning** alternative to Colab Cola's blocked assignment POST. The bridge selects an existing Colab T4 only when exactly **one active assignment** exists for the selected Google profile, and requires exactly one existing Jupyter kernel. Any ambiguous inventory fails closed.

## Ownership and side effects

- The attachment path **never** invokes `provider.allocate`, `client.assign`, `provider.release`, `client.unassign`, or kernel creation/restart.
- An attachment must specify `--attach-existing --confirm-attach`, Comm H.264, 30 FPS, the capacity benchmark and a fresh run directory. `--confirm-allocation` is forbidden.
- The read-only `--preflight-only` checks assignment and kernel identity but does not upload files, execute notebook code, or create a journal.
- A **live confirmed attachment modifies the existing notebook kernel**. It uploads bridge Python sources under a unique `/content/colabcola-bridge/<run_id>` directory and executes the CUDA renderer and Jupyter Comm bootstrap. An independent `--install-remote-pyav` opt-in permits an additional package installation if necessary. Prefer a fresh, disposable notebook with no other user work running.
- The worker sends a best-effort `close` to its own Comm and closes the host WebSocket. This does **not** disconnect the Colab runtime or terminate its kernel. Remote-worker shutdown is not independently confirmed. The user must disconnect/delete the browser runtime when finished.
- The journal records `backend=google-colab-external`, `ownership=external`, and **no** allocation ID, endpoint or credential. `bridge cleanup` rejects this journal rather than attempting to unassign anything. A bootstrap failure records `needs_manual_review`, not an invented cleanup success.

## Read-only preflight (no allocation)

```powershell
& .\.venv\Scripts\python.exe -m colabcola --profile pypi-new-account-test bridge colab `
  --attach-existing --transport comm --video-codec h264 `
  --inflight 1 --fps 30 --poll-hz 30 --benchmark-h264-capacity `
  --h264-capacity-phase-seconds 10 --max-seconds 300 --preflight-only
```

For a live test, the same command requires `--run-dir <new-path> --confirm-attach`, and `--install-remote-pyav` if PyAV is not available inside that runtime.

## Verification and status

- No live external attachment has been attempted by this change. Its network/proxy and kernel-execution behavior are not yet verified.
- Run `tests/test_bridge_colab_external_attach.py`, the full `tests/test_bridge*.py` suite and GitHub CI before any live test.
- Retain the existing managed allocation path and its ownership enforcement unchanged.
- Do not merge into `dev` or `main` until the external ownership and failure-path tests have been reviewed.

Related tracking issue: https://github.com/dependanz/colabcola/issues/21
