# v33t — scoped Colab assignment POST timeout

## Evidence

Two Colab Cola T4 assignment POSTs failed on November 11, 2026 with 40-second read timeouts; cleanup confirmed both owned runs released, and neither reached GPU renderer startup. A normal Colab browser notebook using the same account successfully connected to a Tesla T4 with CUDA available and `nvidia-smi` exit code 0. This supports investigating the timeout but does not prove that longer waiting fixes allocation.

## Corrected implementation (integrity-preserving)

The first v33t commit (`b1953ad`) modified the vendored Colab CLI client. That violated the intentional pinned-source integrity check in `verify_provider()`, so the local 657-test regression passed but managed provider preflight failed with `Google provider differs from inspected commit 79cd126...`.

The corrective commit restores `src/colabcola/_vendor/colab_cli/client.py` byte-for-byte from the previously verified feature commit `917934e`, with **no** changes to its pinned upstream hashes or the provider provenance manifest.

The bridge's own `GoogleProvider.allocate` now uses the verified vendored client's GET/POST helpers directly. It performs the normal assignment GET, returns an existing owned assignment when present, or submits a **single** allocation POST using the 120-second scoped timeout. All other provider operations retain the normal 40-second timeout. The original `COLAB_CLI_ASSIGN_RETRIES=0` remains and the bridge does not retry any ambiguous POST.

This preserves the journaled allocation UUID, the timeout-sanitizing wrapper in `bridge_colab.run_colab`, and fail-closed owned-runtime reconciliation. No new listener, codec, or renderer behavior is introduced.

## Validation and next steps

Run `verify_provider()` and `_verify_bundle()` through offline tests, then `tests/test_bridge_colab_assignment_post_timeout.py`, `tests/test_bridge_colab_assignment_timeout.py`, full `tests/test_bridge*.py` regression and a managed 30 FPS `--preflight-only`. Only after CI and Windows preflight pass, after disconnecting the separate browser Colab notebook, consider one explicitly approved supervised T4 experiment. No automatic allocation or retry is part of this change.

Keep this feature branch isolated from `dev` and `main` pending validation. Issue: https://github.com/dependanz/colabcola/issues/21.
