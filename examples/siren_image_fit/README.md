# Fixed-image SIREN colabcola integration job

This is a small payload for exercising **colabcola**, not a model-selection benchmark.
It directly fits one fixed CIFAR-10 RGB image with a SIREN: two 64-wide sine hidden
layers, omega 30, a linear RGB output, 4,547 parameters and 2,000 full-image Adam steps.
Coordinates are pixel centers on [0,1]^2, remapped internally to [-1,1]^2. RGB targets
are similarly remapped. There is no conditioning encoder, hypernetwork or Fourier
encoding. Sine initialization follows the [official SIREN implementation](https://github.com/vsitzmann/siren/blob/master/modules.py).

`make_job.py` uses the first image in the existing hash-bound comparison dataset
(test_batch.bin record 32). It writes a portable `.colajob` whose inputs are the script,
the fixed image and its provenance manifest. The input image stays fixed throughout
training. Output metrics describe fitting that image, not held-out generalization.

```powershell
python make_job.py --source CIFAR_TUNING_ROOT --output OUTPUT_ROOT/siren-fixed.colajob
```

For local correctness only, create a separate four-step job with `--steps 4 --local-cpu`
and run `verify_local.py --job LOCAL_JOB --output NEW_CHECK_DIRECTORY`. This checks
compiled notebook reproducibility, exact artifact hashes and fresh-process checkpoint
continuation. A local check is never evidence that Colab execution succeeded.

Actual execution uses the existing unattended lifecycle through WSL on this laptop:

```text
colabcola lifecycle prepare siren-fixed.colajob --run-dir NEW_RUN_DIR --drive-parent EXACT_COLABCOLA_FOLDER_ID --token-command PRIVATE_HELPER_ARGV.json --gpu T4 --max-attempts 1 --max-seconds 900
colabcola lifecycle start NEW_RUN_DIR --approve EXACT_PREPARED_PLAN_SHA256
colabcola lifecycle status NEW_RUN_DIR
colabcola lifecycle logs NEW_RUN_DIR
```

The token-command file contains argv only, never credentials. Preparation is local;
start uploads immutable inputs and allocates compute. The exact current-runtime profile
is checked on Colab before training. An incompatible runtime fails preflight instead
of silently changing dependencies. This job publishes immutable model/optimizer/RNG
checkpoints every 250 steps, structured logs, final metrics, an unquantized reconstruction,
a reference/reconstruction PNG and the final checkpoint. The lifecycle retrieves verified
artifacts and releases owned compute. This one-runtime job does not repeat the earlier
intentional runtime-replacement proof.

`COLABCOLA_RESUME_MANIFEST` supplies a verified checkpoint if the workload is run under
a separately bounded recovery plan. Source, image, configuration and runtime identities
must match. Unclipped reconstruction values are used for MSE/PSNR/edge metrics; PNG
values are clipped and quantized only for display.

The current proof artifacts and live outcome are registered in devbrain under
`siren_colajob_proof`. See its `live-verification.json` for actual execution/Drive/cleanup
evidence, `local-verification/verification.json` for local packaging checks, and
`verified-preview/` for the downloaded outputs. No native Windows runtime-provider
support is claimed by this WSL run.

## Verified Colab execution, 2026-10-15

The actual `.colajob` completed through the WSL lifecycle on one Tesla T4. The
supervisor staged verified inputs, bootstrapped the runtime, retrieved logs/checkpoints,
verified final results and released the owned runtime in 160.640403 seconds of the
900-second cap. Independent provider readback found zero active assignments. Drive
retained two checkpoint snapshots; the final step-2,000 checkpoint and all final
members passed independent remote/local byte comparison. All seven declared ZIP
outputs passed validation. Artifact ZIP SHA-256:
`3cb7ece8a8af4da4e22760dc315eb414cb2309592b1ece693b3ac5821a0f319b`.

The payload fit the fixed image in 2,000 steps (7.506983 training/evaluation seconds),
with 51.385663 dB PSNR and RGB MSE 0.00000726831513. Recomputed pixel/edge errors agree;
a CPU reload of the downloaded CUDA checkpoint reproduces its output with maximum
absolute error 1.1920929e-7. All 2,000 consecutive gradient steps and final Adam counters
were checked. This is successful job execution and direct fitting of one image; it is
not a hypernetwork generalization result.

The accepted job is `siren_colajob_proof/siren-fixed.colajob`; exact plan identity is
`7e768838f342daeeef39202e90cdfbd82f3ab65b5ecfe0ad8641e7de8e82043e` and job identity is
`3f5356454e96b4f7b930a6067e5a85bdc0cae3a58208e13e0ce58597380bafd9`.
The proof validates the opt-in lifecycle command. Integration with the stable bare
`colabcola JOB.colajob` entry point, clearer CLI progress and native Windows provider
support remain distinct product work.
