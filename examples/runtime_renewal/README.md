# Drive renewal probe through the Colab CLI

This ordinary standalone Python script writes a fresh 1 MiB checkpoint each minute
for 65 minutes. It needs no extra packages, training data or GPU. Package it with
the general CLI; there is no custom job builder. The prepared example requests
Latest through the primary legacy (official Colab CLI) adapter, Python >=3.11, CPU only, one runtime, 4500 seconds total and
4200 seconds workload timeout. It does not test LongCat dependencies, CUDA or VRAM.

The runtime worker records token changes and authenticated snapshot readback using
those tokens in authorization.json, outside the workload ZIP. No token values or
refresh credentials are included. Acceptance requires a verified publication using
a changed token at least 120 seconds after the initial access token's expiry, plus
successful final local recovery and confirmed owned-runtime deletion. Repeatedly
copying the same token does not pass. There is no forced shortening of token expiry.
A token whose lifetime exceeds this bounded experiment will not pass; do not extend
the budget automatically. No artificial browser keepalive or quota workaround is used.

Prepare from WSL, replacing OUT and the exact host settings:

```bash
colabcola package -o OUT/probe.colajob --runtime-version latest --provider-backend legacy --gpu NONE --python '>=3.11' --max-runtimes 1 --max-seconds 4500 --timeout 4200 --artifact 'outputs/probe/**' --artifact-max-bytes 134217728 --checkpoint 'outputs/probe/state-*.bin.json' --log 'outputs/probe/events.ndjson' --drive-parent EXACT_FOLDER_ID --token-command HOST_HELPER_ARGV.json run.py --seconds 3900
colabcola --prepare-only OUT/probe.colajob
```

Review OUT/probe.run/plan.json and its printed approval hash. Only after explicit
launch authorization, submit `colabcola --approve PLAN_SHA256 OUT/probe.colajob`.
Keep the laptop/WSL awake and connected. The 75-minute bound includes allocation,
setup, checkpoint publication and final recovery. The host must remain available
for runtime deletion; an API outage can delay deletion beyond the work deadline.
A Python mismatch, expiry failure or rejected allocation fails the probe. There is
no replacement runtime. Latest is explicitly requested; this does not verify a dated image.

After the run, `python verify.py OUT/probe.run` checks recovered local evidence.
The separately explicit `--drive-readback` option downloads the final snapshot to
a fresh cache, verifies its hashes and checks the exact owned Colab CLI runtime is absent.
It never allocates, but it does use authenticated network access; it is not part of
offline preparation. Independent cold verification occurs after runtime release.
The probe's outputs are about 66 MiB, bounded at 128 MiB; large artifact transport is
verified separately by benchmarks/verify_large_artifacts.py with real 5 GiB data.

The old REST/2026.07 probe is preserved as historical evidence. It failed before
creation because the beta API was disabled for its consumer project; no dated
runtime or renewal acceptance was obtained. The verifier still supports that old
plan type, but selects its backend explicitly and never uses REST for legacy plans.
The new non-REST probe needs a fresh approval; old launch budgets are not renewed.
