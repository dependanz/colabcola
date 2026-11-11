# v33t — scoped Colab assignment POST timeout

## Evidence

Two Colab Cola T4 assignment POSTs failed on November 11, 2026 with 40-second read timeouts. Both owned-runtime journals were confirmed released; neither reached renderer startup. In contrast, a normal Colab browser notebook successfully connected to a Tesla T4 with CUDA available and `nvidia-smi` exit code 0.

The official Google Colab CLI session-management documentation describes an AuthorizedSession default timeout of 120 seconds when not overridden: https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md . Browser provisioning success does not prove a longer CLI timeout fixes these failures, but establishes a bounded experimental hypothesis.

## Change

The existing `COLAB_CLI_REQUEST_TIMEOUT=40` and `COLAB_CLI_ASSIGN_RETRIES=0` settings are unchanged. Only the bridge's `GoogleProvider.allocate` supplies `post_timeout=120.0`; this is forwarded to the vendored client's assignment POST. The initial assignment GET, preflight GET, reconciliation GET, unassign and all unrelated client operations retain their normal timeout behavior. The default vendored client call signature remains compatible.

No automatic retries, repeated ambiguous requests, changed allocation identifiers, codec changes, renderer changes, or new listeners are introduced. If the POST still times out, the durable allocation UUID, sanitized journal failure category and fail-closed owned cleanup remain active.

## Verification

Run `tests/test_bridge_colab_assignment_post_timeout.py`, `tests/test_bridge_colab_assignment_timeout.py`, then all `tests/test_bridge*.py`. Run the managed 30 FPS `--preflight-only` mode. After GitHub CI and local tests pass, verify the ordinary browser notebook has been disconnected and that Colab Cola sees no active assignments before contemplating at most one **explicitly authorized** supervised allocation. Never launch a paid trial automatically.

Keep this feature change isolated from `dev` and `main` pending evaluation. Tracking: https://github.com/dependanz/colabcola/issues/21 .
