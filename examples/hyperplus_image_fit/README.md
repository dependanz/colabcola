# Hyperplus image-conditioned coordinate-network proof

This workload uses private dependanz/hyperplus at commit
24b446e9041724cb282763a90d724cf10edbcd24. The verified wheel is hyperplus 0.1.0;
HYPERPLUS_PROFILE.json binds its wheel and installed module hashes. It is not
redistributed by this example. Obtain the private source with your own authorized
GitHub access, build the pinned wheel, and keep credentials outside job artifacts.

The encoder generates all 1,763 parameters of an RGB coordinate MLP through
Hyperplus. Pixel centers use x right, y down on [0,1]^2. Four fixed Fourier
frequencies augment coordinates. A convolutional image encoder emits the complete
weight vector; its output bias starts at ordinary initialized decoder parameters,
with small nonzero output weights to preserve encoder gradient flow.

The default experiment uses a verified CIFAR-10 subset of 128 training and 32 held-out
images, selected from separate official batches. Labels are unused. Held-out images
are given to the encoder: this measures conditioned reconstruction, not unconditional
generation. Report each split separately and compare against each image's mean-color
baseline. Bounded reconstruction measurements are reported below; convergence has not
been established.

## Commands

Run these commands from this directory with a Python environment containing torch,
numpy, pillow, pytest, and the pinned Hyperplus wheel. Use absolute paths for the
dataset, wheel and output locations. The isolated verification environment on the
current laptop is recorded in the devbrain manifest.

    python -m pytest test_workload.py -q
    python prepare_data.py --output DATASET_ROOT
    python make_job.py --data DATASET_ROOT --wheel PINNED_WHEEL --output JOB.colajob
    colabcola validate JOB.colajob
    colabcola compile --transport embedded --output JOB.ipynb JOB.colajob

prepare_data.py downloads the official CIFAR-10 binary archive, checks its published
MD5, records SHA-256, and writes hash-bound NCHW uint8 arrays. Downloads can resume a
retained .partial file via HTTP Range. The official archive and 128/32 subset are now verified on this laptop after
bounded HTTP Range assembly. The preparer's own sequential resumption branch
remains unverified.

The GPU job defaults to 2026.07; use --runtime latest for the managed provider.
The dated job retains Python 3.12.13/PyTorch 2.11.0+cu128. Latest uses the explicit
COLAB_RUNTIME_20261015.json profile: Python 3.13.15, PyTorch 2.11.0+cu130,
NumPy 2.1.3 and Pillow 11.3.0, based on the observed Python and a pinned official
backend-info snapshot. The profile is included in the payload identity. GPU availability and actual runtime compatibility still need
preflight verification. Generation and compilation do not authorize a cloud run.

For a bounded CPU diagnostic through the actual generated notebook:

    python verify_local.py --output NEW_OUTPUT_ROOT --wheel PINNED_WHEEL

This creates explicitly synthetic 8x8 RGB inputs (4 train, 2 held out), compiles a
CPU JobSpec twice to verify reproducibility, executes all code cells locally,
trains four steps, and verifies seven output hashes. It does not allocate Colab,
use Drive, or demonstrate image quality. In the recorded run Hyperplus was removed
from the isolated environment first, proving installation from the embedded wheel.

To resume a trusted local checkpoint in a new attempt directory:

    python train.py --data DATASET_ROOT --output NEW_ATTEMPT --steps 200 --resume OLD_ATTEMPT/state-00000100.pt.json

Supply the same model, batch, pixel, learning-rate, seed, threads and device arguments
as the original attempt. The identity check also binds the data manifest, workload
source, installed Hyperplus code and PyTorch version. Only the target step and
checkpoint cadence may change. CPU/CUDA environment switching is rejected.

## Evidence and remaining work

- 52 upstream tests passed against the installed wheel. Two upstream bilinear vmap
  performance warnings were emitted.
- Five workload tests passed, including exact six-step uninterrupted versus
  three-step/fresh-process/three-step continuation. Model, optimizer, random states
  and loss history match exactly on CPU.
- The compiled four-step CPU notebook succeeded with seven verified outputs.
- Immutable local checkpoints contain model, optimizer, CPU/CUDA RNG, sampler RNG,
  cumulative loss history and run identity. Members are written before their
  hash-bound commit manifests. Restoring corrupt or mismatched evidence fails.
- Structured attempt events, split MSE/PSNR, conditioning-image mean-color baselines
  and reference/reconstruction PNGs are produced.

The managed_recovery policy now connects checkpoints and logs to the lifecycle worker.
COLABCOLA_RESUME_MANIFEST selects the verified restored checkpoint. Allocation/recovery,
publication and cleanup are verified live on T4. The bounded WSL proof restored step
25 on a distinct T4 and continued through step 2000, with all 1975 resumed sampling
batches and saved loss history verified. See ../../LIFECYCLE.md. The supervisor must run on a host
that remains online. The full lifecycle acceptance protocol lives in devbrain.

CIFAR reference: Alex Krizhevsky, Learning Multiple Layers of Features from Tiny
Images (2009), https://www.cs.toronto.edu/~kriz/cifar.html.

## Live current-runtime validation (2026-10-15)

Python 3.13.15, torch 2.11.0+cu130, NumPy 2.1.3 and Pillow 11.3.0 passed
exact preflight on Tesla T4. Adaptive CUDA pooling failed strict determinism; the
encoder now computes equivalent 4x4 bin means with matching CPU forward/gradients.
Strict determinism remains enabled. The corrected job completed 2000 GPU steps with
verified Drive checkpoints, terminal logs and seven final outputs. Train PSNR was
20.48 dB on 128 images; held-out PSNR 14.85 dB on 32 conditioning images. These are
bounded-workload results, not convergence. The corrected deliberate-recovery proof
restored step 25 on a fresh T4 and reached step 2000 with the same reported reconstruction
metrics. Drive/local checkpoint and artifact bytes were independently verified; both
runtimes were released within 261.246272 seconds of the shared 900-second budget.


## HOSC quality comparison (2026-10-15)

The generated coordinate MLP accepts `--activation hosc --hosc-beta B --hosc-omega W`.
It uses HOSC(x) = tanh(B * sin(W * x)), with finite positive B and W. See the original
[Serrano et al. paper](https://arxiv.org/abs/2401.10967) and the
[2026 follow-up](https://arxiv.org/abs/2601.07870). The encoder stays unchanged.
The initial comparisons retain Fourier inputs, ordinary decoder initialization and fixed
activation parameters, isolating the activation change; they do not reproduce the paper's
larger INR architecture or its frequency/initialization protocol.

The default remains tanh, preserving baseline numerical behavior. Activation settings are
bound to checkpoint identity. Changing activation, B or W requires a new run; historical
lifecycle notebooks/checkpoints remain immutable and retain their original source.
`make_job.py` forwards activation, B/W, width, latent size, learning rate and seed explicitly.
For example, after preparing the data and pinned wheel:

    python make_job.py --data DATASET_ROOT --wheel PINNED_WHEEL --output hosc.colajob --runtime latest --steps 8000 --activation hosc --hosc-beta 1 --hosc-omega 3

This only creates a job file; it does not allocate compute. The HOSC results below were
measured on CPU. A four-step compiled HOSC notebook also passed local execution with seven
verified outputs and reproducible notebook bytes. Tests cover the analytic derivative,
Hyperplus gradients, exact fresh-process continuation and mismatched activation rejection.

| Train images / steps | Activation | Train PSNR (dB) | Held-out PSNR (dB) |
|---|---|---:|---:|
| 128 / 2000 | tanh | 20.4815 | 14.8503 |
| 128 / 2000 | HOSC B=1 W=1 | 21.8736 | 14.5255 |
| 128 / 2000 | HOSC B=2 W=1 | 21.6263 | 14.8111 |
| 128 / 2000 | HOSC B=1 W=3 | 22.8843 | 14.6840 |
| 1024 / 8000 | tanh | 20.0856 | 18.0195 |
| 1024 / 8000 | HOSC B=1 W=3 | 20.8559 | 17.7378 |

All rows use seed 1234, 32-wide coordinate layers, latent width 64, batch 8, 256 sampled
pixels and learning rate 0.001. The same 32 held-out images are used throughout. Within
matched comparisons, initial parameter tensors and sampled image batches are identical.
B=1/W=3 was selected by training MSE from the first comparison. The larger-data follow-up
is exploratory: held-out images had already been inspected, and no multi-seed claim is made.

The expanded HOSC setup reduces held-out MSE by 48.5654% versus the original small-data
baseline, but more data and training contribute to that gain. HOSC improves training fit
in both matched comparisons; it does not beat matched tanh on held-out reconstruction.
All six checkpoints were reloaded and their train/held-out metrics independently reproduced.
No HOSC GPU run or new cloud allocation is claimed. Full evidence and images are mapped in
Devbrain as hyperplus_hosc_quality; the larger subset is hyperplus_cifar_quality_subset.

## Validation-selected HOSC tuning

`tune_quality.py` runs a bounded CPU experiment using the existing, checksum-verified
CIFAR binary archive and the installed pinned Hyperplus wheel. Supply new data/output
folders; existing experiment folders are never overwritten:

```powershell
python tune_quality.py --archive PATH/cifar-10-binary.tar.gz --data NEW_DATA_ROOT --output NEW_OUTPUT_ROOT --previous PREVIOUS_HOSC/state-00008000.pt.json
```

The training set stays fixed at records 0:1024 of `data_batch_1.bin`. Validation uses
records 1024:1280 of that member; final testing uses records 32:288 of `test_batch.bin`,
excluding the 32 test images inspected during the earlier experiment. Source ranges and
array hashes are retained in `splits.json`. During search, the workload's legacy
`heldout.npy` and `metrics.final.heldout` names explicitly mean **validation**.

The sweep crosses beta {0.5, 1, 2} with omega {1, 3, 6}, using coordinate width 64 and
encoder latent width 128. All nine candidates train for 4,000 steps. The two lowest
validation-MSE candidates continue from their complete checkpoints to 12,000 steps.
The experiment also trains width-32/latent-64 HOSC B=1/W=3, small tanh, and large tanh
controls for 12,000 steps. Seed 1234, Adam learning rate 0.001, batches of eight images
and 256 sampled pixels stay fixed. Equal-step controls receive identical sampled
image batches; a larger network consumes more compute per step.

Two additional controls use the selected HOSC endpoint's exact step count: tanh at
the same network size, and HOSC with the same beta/omega at the other network size.
These controls do not participate in selection.

The best validation-MSE HOSC endpoint among the nine screen endpoints, two continued
endpoints and small HOSC control is frozen in `selection.json` before test data are
loaded for model evaluation. No result from the fresh test set changes that choice.
RGB MSE/PSNR and horizontal/vertical RGB first-difference MSE are reported, with paired
image-bootstrap intervals. Those intervals quantify image variation, not uncertainty
across independently trained seeds. First-eight sequential test-image comparisons avoid
visual cherry-picking. Images remain 32x32; enlarged previews are nearest-neighbor views,
not higher-resolution reconstructions.

`protocol.json`, `promotion.json`, `search.json`, `selection.json`, `test-results.json`,
`verification.json`, per-run logs and immutable checkpoints provide the audit trail.
Every saved validation result is reproduced after checkpoint reload. The local experiment
has a 30-minute subprocess-work budget and a six-minute per-training-process timeout;
final verification/evaluation may extend the wall-clock interval. It allocates no cloud
runtime and makes no claim about Windows runtime-supervisor support.

### Verified fresh-test outcome (2026-10-15)

Validation selected **HOSC beta 2, omega 1, coordinate width 64, latent width 128,
checkpoint step 4,000**. The hypernetwork has 789,411 trainable parameters and generates
5,571 coordinate-network parameters (previously 152,515 and 1,763). Continuing either
of the top two screened HOSC configurations to step 12,000 worsened validation MSE,
so the earlier checkpoint was retained. This is a selection over the declared grid,
not a globally optimal setting.

The 256 fresh test images were excluded from selection. All rows use the same 1,024
training images and one training seed. Edge MSE compares horizontal and vertical RGB
first differences; lower is better.

| Model | Steps | Test PSNR (dB) | Test RGB MSE | Test edge MSE |
|---|---:|---:|---:|---:|
| Previous HOSC, B=1 W=3, width32 latent64 | 8,000 | 17.9777 | 0.01593045 | 0.00888446 |
| Selected HOSC, B=2 W=1, width64 latent128 | 4,000 | 18.3826 | 0.01451238 | 0.00838897 |
| Exact-step tanh, width64 latent128 | 4,000 | 18.3254 | 0.01470489 | 0.00820647 |
| Exact-step HOSC B=2 W=1, width32 latent64 | 4,000 | 18.1851 | 0.01518774 | 0.00847813 |
| Longer tanh, width64 latent128 | 12,000 | 18.4250 | 0.01437151 | 0.00852813 |

Selected HOSC reduces RGB MSE **8.90%** and edge MSE **5.58%** versus the previous HOSC
checkpoint on these same fresh images. Its larger capacity reduces RGB MSE 4.45% and
edge MSE 1.05% versus the exact-step smaller HOSC control. These comparisons have
paired image-bootstrap intervals excluding zero, but only one trained seed was tested.

HOSC does **not** establish an advantage over matched tanh: its 0.0572 dB PSNR gain
has an image-bootstrap MSE-difference interval spanning zero, and its edge MSE is 2.22%
higher. The larger tanh run at 12,000 steps also has slightly better PSNR. Reconstructions
remain visibly blurry; the measured gain is modest, not high-fidelity reconstruction.

The original run used the frozen `executed_tuner.py`, with exact-step controls declared
in `matched-control-amendment.json` before selection/test evaluation. The reusable
`tune_quality.py` now incorporates those controls. All 16 endpoints, 96,000 optimizer
updates, matched sampled-image sequences, restored loss prefixes and Adam counters
passed verification. Seven final model evaluations were reloaded in a fresh process;
NumPy independently reproduced pixel/edge aggregation. Tests: 23 passed, one skipped
for unavailable CUDA; the subsequently updated tuning utility passed its four tests.

Evidence is mapped as `hyperplus_hosc_tuning`. Use `verified-test-results.json` and
`independent-verification.json` for the final assessment, and `matched-test-comparison.png`
for the first eight sequential test examples. The frozen runner's raw report mislabeled
combined target-plus-hypernetwork slots as generated parameters; the verified report
corrects only that metadata and reproduces all quality metrics exactly. The raw report
and executed source remain preserved. All training was local CPU, with zero cloud
allocations; the main experiment interval was 1,331.36 seconds, including the concurrent
supplemental controls, within the declared 1,800-second allowance.
