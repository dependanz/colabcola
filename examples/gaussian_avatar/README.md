# GaussianAvatars as an ordinary Python module project

This example tests colabcola with the official [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars) source. A bounded real-data training pilot passed on the observed Colab stack on 2026-10-15. No FLAME models or subject images are included.

## Two explicit gates

1. **Environment pilot (verified on the observed Colab stack):** one T4, one runtime maximum, 900 seconds total; workload timeout 780 seconds, installation budget 660 seconds, MAX_JOBS=2, architecture 7.5. Build pinned simple-knn, diff-gaussian-rasterization and nvdiffrast. Import the actual training module, execute tiny CUDA nearest-neighbor and differentiable-rendering checks, recover logs/report/preview. No avatar fitting or restart test. Runtime incompatibility or an exhausted build budget is a failed pilot, not success. A T4 is not allocated during preparation.
2. **Training pilot (verified):** eight tracked views, width256, 200 iterations, no densification before the upstream default10000. The sequence first checks assets, then builds/tests the environment, then trains. Licensed FLAME assets, compatible tracked data and successful local FLAME deserialization are required. No new compute authorization is implied. The verified run completed 200 training-loop iterations (199 optimizer updates in upstream), saved a model and checkpoint, recovered all outputs, and released its one T4 within 758.064 seconds of the 900-second limit. This is a workflow pilot, not convergence or avatar-quality validation.

The last observed Colab stack was Python3.13.15 / Torch2.11.0+cu130 / NumPy2.1.3. Upstream documents Python3.10 and tests Torch2.0.1/CUDA11.7 and Torch2.2.0/CUDA12.1. Its NumPy1.22.3 pin cannot simply be used with Python3.13. This example uses a separate explicit headless compatibility candidate. GUI dependencies are omitted; nvdiffrast is required by the training import even without a viewer. Native packages are installed from included pinned source, not moving Git branches. Supporting PyPI dependencies are resolved at runtime and recorded in pip-freeze.txt; the whole environment is not hash-locked.

## 1. Assemble an ordinary project (Linux / WSL)

Use new directories; commands below do not overwrite an existing working copy. Set EXAMPLE to this example directory, WORK to a new source workspace, and JOBS to a new output directory outside WORK.

```bash
export EXAMPLE=/path/to/colabcola/examples/gaussian_avatar
export WORK=/path/to/gaussianavatars-project
export JOBS=/path/to/gaussianavatars-jobs
mkdir -p "$JOBS"
git clone --filter=blob:none --no-checkout https://github.com/ShenhanQian/GaussianAvatars.git "$WORK"
cd "$WORK"
git sparse-checkout set --no-cone '/*' '!/media/'
git checkout --detach 29b0021308d45c6dce0b522044a4893e189c7a23
git submodule update --init --recursive
git clone https://github.com/NVlabs/nvdiffrast.git vendor/nvdiffrast
git -C vendor/nvdiffrast checkout --detach 253ac4fcea7de5f396371124af597e6cc957bfae
cp -r "$EXAMPLE/gaussian_avatar" .
cp "$EXAMPLE/pilot-requirements.txt" "$EXAMPLE/source-lock.json" "$EXAMPLE/training-commands.json" .
```

The application commands are ordinary Python: `python -m gaussian_avatar.pilot --build`, `python -m gaussian_avatar.assets ...`, and upstream `python -m train ...`. The helper module has no colabcola imports and does no packaging. No make_job.py is needed. The training code remains unchanged. During setup, a hash-bound patch adds the missing `<cstdint>` include to the pinned rasterizer header; unknown source bytes are rejected. Patch hashes and actual compiler versions are recorded in report.json. Native pip builds use verbose output plus pip.log; keep `--log 'outputs/environment/*'` enabled so failure diagnostics survive even when training never starts. Source licenses remain in the project; see upstream LICENSE.md and LICENSE_GS.md.

## 2. Package the environment pilot

Use an existing credential-free Drive token-helper argv JSON and the exact Drive parent you configured for colabcola. Do not paste credentials into this example or job. First-time login is documented in ../../LIFECYCLE.md.

```bash
colabcola package -o "$JOBS/environment.colajob" --project "$WORK"   --exclude 'media/*' --exclude 'vendor/nvdiffrast/docs/*' --exclude 'vendor/nvdiffrast/samples/*'   --gpu T4 --python '>=3.13,<3.14' --runtime-requirement 'torch==2.11.0'   --artifact 'outputs/environment/**' --artifact-max-bytes 67108864   --log 'outputs/environment/*'   --timeout 780 --max-runtimes 1 --max-seconds 900   --drive-parent "$DRIVE_PARENT" --token-command "$DRIVE_HELPER"   -- python -m gaussian_avatar.pilot --build
colabcola --prepare-only "$JOBS/environment.colajob"
```

Inspect the returned plan hash, payload and bounds. **Only after approving a real run**, submit `colabcola --approve PLAN_SHA256 "$JOBS/environment.colajob"`. Keep the supervising laptop awake. This can consume the entire15-minute allowance compiling; a longer run is not silently authorized. Results must show all three CUDA operations and the real training-module import passing. A successful report explicitly has avatar_training=false. Failure reports and build.log explain the unresolved gate. The outer workload process-group timeout bounds native compiler children; the supervisor releases its owned runtime.

## 3. Obtain real training inputs

Follow the official [download guide](https://github.com/ShenhanQian/GaussianAvatars/blob/main/doc/download.md):

- Register and obtain the FLAME2023 version with jaw rotation and FLAME Vertex Masks through the [FLAME download site](https://flame.is.tue.mpg.de/download.php), under its access/license process. Place the files in a private local folder as flame2023.pkl and FLAME_masks.pkl. Do not redistribute these files with this example. There is no automatic login/license acceptance or fabricated substitute.
- Download one preprocessed NeRSemble subject via the official guide's LRZ release link or its OneDrive access process. Follow the source's dataset terms and citation requirements. Raw photographs alone are insufficient: the training sequence needs camera transforms and per-frame FLAME tracking parameters. For personal captures, upstream points to [VHAP](https://github.com/ShenhanQian/VHAP) for preprocessing; running that pipeline is a separate job.
- Select the extracted sequence directory containing transforms_train.json, transforms_val.json, transforms_test.json, referenced PNG images and flame_param_path NPZ files. Do not pass the outer download folder.

The model files are pickle-based trusted inputs. Only deserialize model files you obtained from the official source. The structural validator does not load pickle. FLAME2023 uses legacy Chumpy arrays; convert a private copy in an isolated environment before packaging it for current Colab. This preserves numeric values, records array/model hashes and keeps the original license and files. The conversion does not change the upstream model or trainer.

```bash
cd "$WORK"
python -m venv "$JOBS/flame-convert-env"
"$JOBS/flame-convert-env/bin/python" -m pip install numpy scipy chumpy==0.70 six
"$JOBS/flame-convert-env/bin/python" -m gaussian_avatar.convert_flame \
  --source "$FLAME_ORIGINAL/flame2023.pkl" --output "$FLAME_DIR/flame2023.pkl"
cp "$FLAME_ORIGINAL/FLAME_masks.pkl" "$FLAME_DIR/FLAME_masks.pkl"
```

FLAME_ORIGINAL is the extracted official model/masks directory. FLAME_DIR is a new private output directory. Do not globally downgrade the Colab Python/Torch stack. Training uses the converted numeric pickle and does not need Chumpy.

For a small real-data pilot, avoid downloading the complete subject archive:

```bash
python -m gaussian_avatar.acquire --output "$PILOT_DATA"
python -m gaussian_avatar.assets --source "$PILOT_DATA" --flame "$FLAME_DIR"
```

The acquisition helper reads eight tracked training views from the official public subject218 ZIP using bounded HTTP ranges (128 MiB transfer cap). It requires exact range responses and stable archive identity, verifies ZIP member CRCs, records retained SHA-256 hashes and refuses to overwrite an existing output. It does not claim to verify the undownloaded archive's full hash. The source's dataset terms still apply.

Alternatively, for an already downloaded sequence:

```bash
python -m gaussian_avatar.assets --source "$SEQUENCE" --flame "$FLAME_DIR" \
  --subset-output "$PILOT_DATA" --frames 8
```

Both paths retain canonical_flame_param.npz, retain eight views, remap timestep ids and write empty validation/test splits. Empty/missing canonical parameters and a competing sparse/ Colmap layout are rejected. At least two camera centers are required to avoid degenerate normalization. The prepared subject218 pilot contains eight 550x802 RGB views of one tracked pose. This is a real training setup test, not an evaluation dataset or a learned animation demonstration. Its converted FLAME model has passed a CPU forward/backward check against the actual tracking parameters; The bounded GPU pilot subsequently completed; this tiny subset does not establish held-out quality.

## 4. Training packaging recipe: only after the gates above

Before packaging, run `python -m gaussian_avatar.assets --source "$PILOT_DATA" --flame "$FLAME_DIR" --loader-check "$WORK"` in an environment with NumPy, Pillow and tqdm. The CPU check executes hash-pinned upstream loader selection, camera/mesh reader and geometry bodies without importing CUDA extensions. It requires DynamicNerf, nonempty zero-based pose meshes, matching camera timesteps, finite numeric parameter archives and positive camera radius. It does not replace full CUDA scene/training verification. Unknown upstream source hashes fail closed.

training-commands.json contains validate (including the loader check) -> environment/build -> upstream `python -m train`, preserving ordinary module execution and fail-fast sequence semantics. Use explicit data mappings:

```bash
colabcola package -o "$JOBS/training.colajob" --project "$WORK"   --commands "$WORK/training-commands.json"   --exclude 'media/*' --exclude 'vendor/nvdiffrast/docs/*' --exclude 'vendor/nvdiffrast/samples/*'   --data "$PILOT_DATA=data/pilot"   --data "$FLAME_DIR/flame2023.pkl=flame_model/assets/flame/flame2023.pkl"   --data "$FLAME_DIR/FLAME_masks.pkl=flame_model/assets/flame/FLAME_masks.pkl"   --gpu T4 --python '>=3.13,<3.14' --runtime-requirement 'torch==2.11.0'   --artifact 'outputs/environment/**' --artifact 'outputs/avatar/**' --artifact-max-bytes 134217728   --log 'outputs/environment/*'   --timeout 780 --max-runtimes 1 --max-seconds 900   --drive-parent "$DRIVE_PARENT" --token-command "$DRIVE_HELPER"
colabcola --prepare-only "$JOBS/training.colajob"
```

**Transport:** managed inputs above32MiB use8MiB Drive chunks, with identity/hash verification before runtime execution. Total prepared inputs are capped at512MiB; notebook compilation still needs memory for the embedded JSON. The actual model/data pilot has passed local packaging and preparation. Large-input remote transfer has passed live validation. With the header patch, all three native extensions build on the observed stack. A later pip check found that Colab IPython lacked jedi; the requirements now include it. CUDA gradient tests subsequently passed. Training failed because the subset omitted canonical_flame_param.npz, selecting the wrong upstream loader. Acquisition and subsetting now retain that genuine archive member and validation requires it. The corrected pilot then passed training and automatic artifact recovery. A larger dataset may exceed the input limit.

Training success requires actual CUDA execution, a finite-loss log, saved step200 model/checkpoint and verified recovered artifacts. Upstream final checkpoints are artifacts here; automatic exact mid-training continuation is not implemented by this example. Its sequence restarts from the beginning and no restart experiment is enabled. A separate data-bearing run and explicit budget approval come after preparation succeeds.

## Next example

After GaussianAvatar works, use gsplat as a separate ordinary library project. No gsplat installation, preparation or compute is included in this pilot.

## Verified bounded run (2026-10-15)

The general packager produced job `0cba2171bdf6b6dc7d49e0565102f9a4c440a3eb9f347f8395d64a4e665375bb` and plan `049c2ec6a1f8d336823967d37edba543c2d7c725042723a490c48e902c6286f4`. One public `colabcola --approve PLAN_SHA256 loader.colajob` command staged inputs, ran the module sequence, recovered final artifacts and released compute despite two transient status-channel failures.

The run used one Tesla T4 for a 758.064-second total lifecycle, with zero active assignments independently confirmed afterward. CUDA smoke tests and 200 training-loop iterations passed. The step-200 checkpoint contains 10,144 Gaussians, finite tensors and optimizer step199 (upstream does not update on the final iteration). The displayed EMA loss went from0.5899081 to0.0449687. The 2,438,159-byte result ZIP has SHA-256 `ef12587b4e826523d01d4860ea67caa0cd8bf12c3d2823e75c5d132ec14c6b38`; all11 outputs, including the checkpoint, PLY, FLAME parameters and logs, passed independent Drive/local readback. This does not verify restart continuation or held-out rendering quality.

## Render a recovered model without retraining

Copy the recovered `cfg_args`, `chkpnt200.pth`, and `point_cloud/iteration_200/{point_cloud.ply,flame_param.npz}` into a new ordinary project's `saved_model/` directory. Record their SHA-256 values in `model-hashes.json` (relative names as keys). Keep the matching tracked dataset and FLAME assets. `render-commands.json` runs validation, environment setup, upstream `python -m render` at iteration200, then `python -m gaussian_avatar.preview`. It contains no training command. Explicit `-s data/pilot` overrides the previous runtime path in cfg_args.

Package this sequence with the general CLI using `--commands render-commands.json`, and declare `--artifact 'outputs/preview/**' --artifact 'saved_model/train/ours_200/**' --artifact 'outputs/environment/**'`. Use the same environment and bounded execution flags as the training recipe. The preview step verifies the saved files still match their hashes, requires all eight input/render pairs, and saves labeled pairs plus overview.png. These are training-view comparisons at the pilot resolution; they are not held-out evaluation. Rebuilding native extensions on a fresh T4 is still part of the runtime budget.


The render-only job `b94f55d8991334deaf6cd3887422f022872d805c0a0a818ab935d3171a06fa3f` completed through the public CLI on2026-10-15: one Tesla T4,740.428 seconds total lifecycle within900, runtime released and zero active assignments independently confirmed. Eight256x373 input/render pairs, eight labeled pair PNGs and a1024x1636 overview were recovered from Drive. Original model hashes and exact preview pixels passed independent checks; no retraining occurred. The33-output ZIP is2,619,856 bytes, SHA-256 `e9e64020a6363bc2ad17f35196a03a96bc606485f86377c5036d59eddb2551e5`. One status ReadTimeout recovered on the same runtime. Reconstructions show recognizable head shape/pose with soft fine detail after the short200-step training pilot. Upstream also attempted unrequested MP4 exports; these are empty because yuv420p rejects the odd373-pixel height. The requested PNG comparisons are valid; no video result is claimed.

## Longer training with matched previews

`quality-commands.json` is an ordinary six-command module sequence: validate inputs,
build/check the environment, train to 5000 iterations, render the saved step-200 and
step-5000 models, then run `python -m gaussian_avatar.compare`. It keeps the same
eight training views and width256, and saves checkpoints at200/1000/5000. The
upstream trainer and its densification settings are unchanged; this endpoint is
below the default10000 densification start.

Copy `gaussian_avatar/compare.py` and `quality-commands.json` into the assembled
project. Use the general training packaging recipe with
`--commands "$WORK/quality-commands.json"`, add `--artifact 'outputs/comparison/**'`,
and choose an explicit longer execution bound. The current pilot uses one T4,
`--max-seconds 3600 --timeout 3300`, and512MiB per output pattern. These are maximum
budgets, not a requirement to keep the runtime allocated after completion.

The comparison module requires matching input pixels and complete render sets. It
writes eight input/step200/step5000 panels, an overview and per-view RGB MSE/PSNR.
It also records MSE on a simple non-white input mask (minimum RGB<0.98). Metrics
use quantized PNGs and training views; neither convergence nor held-out/animation
quality follows from a lower training error. The final checkpoint is recovered as
an artifact; this sequence does not implement automatic mid-training resume.

The initial quality run exposed delayed stdout/stderr and a Jupyter cleanup warning.
Both fixes passed a fresh live replay: the single command displayed 298 training
updates across 15 verified snapshots before completion, with no client-cleanup warning.
The one-T4 run completed in 963.540 seconds, recovered all outputs and independently
confirmed runtime release. The evidence is recorded in
[NONREST_REVALIDATION.md](../../NONREST_REVALIDATION.md).

The longer non-REST pilot passed on2026-10-15: one T4 released in1038.612 seconds
within3600,5000 iterations/4999 optimizer updates,63 outputs cold-readback verified.
All eight matched training views improved from step200 to5000: aggregate PNG
PSNR23.94 to36.39dB and mean RGB MSE down94.30%. The result includes eight labeled
comparisons and an overview. This improves training-view fitting; the dataset still
contains only one pose and does not establish held-out or animation quality.
