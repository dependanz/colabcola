# Private-input managed acceptance (issue #40)

This fixture generates a new synthetic 1 GiB file (128 unique 8 MiB chunks) and two independent `.colaml` jobs. It does not use any existing research data. Both jobs are prepared before execution. Run A must publish the asset, restore it on a fresh CPU runtime, and verify the full-file SHA-256 inside its workload. The script then moves its own synthetic source to `original-synthetic.bin` inside the benchmark directory. Run B must use the same private Drive receipt, upload zero additional asset payload bytes, and restore and verify the original bytes on another allocation.

Run from a built installation of the private-input feature, with an authorized profile that uses an owned, unshared Drive folder:

```bash
python examples/private_inputs/benchmark.py "$HOME/colabcola-runs/private-input-proof" --profile default
```

The directory must not exist. This driver requests at most two CPU allocations, with a maximum managed wall budget of 1800 seconds each (including staging). It uses explicit read-only preflight checks, stops on the first error, does not override active-assignment safeguards, and does not retry with a new workspace. Use the printed `BENCH_DIR` to inspect a stopped run rather than repeatedly allocating. It retains local evidence and the synthetic source; the verified private Drive asset is retained intentionally for reuse. No permission mutations, public URLs, or automatic deletions are performed.

The verifier checks the frozen manifest, prepared files, job-bound admission report, disk and size bounds, exact workload digest, private-input restoration, staging evidence, final bundle and runtime release. Intermediate completed-output recovery is reported separately, not required for this input proof. A fast workload can make a final snapshot visible before its intermediate snapshot has been recovered. The final bundle is still mandatory.

Use `--prepare-fixture-only` to generate only the synthetic data and recipes without any controller calls. The integration tests execute the same fixtures and verifier with a simulated provider and a disk-backed Drive HTTP fake. The 1 GiB version is opt-in:

```bash
COLABCOLA_PRIVATE_ASSET_LARGE_PROOF=1 python -m pytest -q tests/test_private_input_managed.py -k real_1gib_managed_workflow -s
```

It requires at least 5 GiB of local free disk. The fake provider, host recovery-disk admission, and small run-snapshot store are simulated; private-asset request serialization, local transfer buffers, on-disk asset reconstruction, workload digest and runtime disk checks are real. This is not evidence of live Drive throughput or permissions. The ordinary regression suite runs a small version of the same two-job flow.
