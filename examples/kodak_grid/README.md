# Kodak paired image fitting

`run.py` is an ordinary standalone Python program. It imports PyTorch, NumPy and
Pillow; it does not import colabcola or require a packaging helper.

It fits each image independently with HOSC and SIREN, at hidden depths 1/2/3 and
widths 32/64/128/256. Both activations share initial weights, sampled coordinates,
Adam settings and step counts. The RGB output layer is excluded from hidden depth.
Defaults: seed 1234, HOSC beta 1, omega 30, Adam 1e-4, 8192 pixels per step.

## Run normally

Download the original Kodak images and their manifest once:

```bash
python run.py --prepare-data --data /path/to/kodak
```

Train locally on a GPU (or pass `--device cpu`):

```bash
python run.py --data /path/to/kodak --output outputs/grid --images kodim01.png --steps 250
```

Omit `--images` for all 24 images. `--depths` and `--widths` select subsets.
Outputs include reconstructed PNGs, model weights, losses and metrics in one ZIP
per activation pair, plus summary, progress and event files. Checkpoints contain
model, Adam and sampling state and references to completed pair ZIPs. Resume with
`--resume /path/to/state-000000050.pt.json` and a new output directory.

## Package with questions

From this directory, package the ordinary command:

```powershell
colabcola package -- python run.py --data /path/to/kodak --images kodim01.png --steps 250
```

The packager suggests mapping the dataset to `data`, detects NumPy/Pillow/PyTorch
imports, and asks for compute, output globs, checkpoints and time. Choose T4 for a
Colab GPU run. Recover `outputs/grid/**`; the checkpoint manifest glob is
`outputs/grid/state-*.pt.json`, accepted through `--resume`. Save answers as
`kodak.colaml` when asked; later use `colabcola package --answers kodak.colaml`
with a fresh --output. The .colaml file is optional TOML, not a Python builder.
For Colab's existing packages, answer `none` to the dependency question: this records
runtime package checks rather than installing guessed versions. Packaging does not
run training or allocate a runtime.

## Explicit general CLI options

Run from this directory; use a new output filename for each package:

```bash
colabcola package --output /path/to/jobs/kodak.colajob \
  --gpu T4 \
  --runtime-requirement torch --runtime-requirement numpy --runtime-requirement pillow \
  --data /path/to/kodak=data \
  --artifact 'outputs/grid/fits/*.zip' \
  --artifact outputs/grid/summary.json \
  --artifact outputs/grid/identity.json \
  --artifact outputs/grid/progress.json \
  --artifact outputs/grid/events.ndjson \
  --checkpoint 'outputs/grid/state-*.pt.json' --resume-arg=--resume \
  --log outputs/grid/progress.json --log outputs/grid/events.ndjson \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  --drive-parent YOUR_DRIVE_FOLDER_ID --token-command /path/to/drive-helper.json \
  run.py --data data --output outputs/grid --images kodim01.png --steps 250
```

Use PowerShell backticks instead of Bash backslashes for multiline Windows commands,
or put the command on one line. The portable `SOURCE=TARGET` mapping avoids ambiguity
with Windows drive letters. All colabcola options precede `run.py`; subsequent arguments
belong to the script.

Packaging is offline and creates the `.colajob`, a sibling `.files` source snapshot,
and (with the two host settings above) a `.run.json` execution configuration.
There is no `make_job.py`. The source can be anywhere, and these folder names are
not required by colabcola. Dependency versions can be constrained explicitly;
`--requirements requirements.txt` installs declared packages remotely instead of
requiring them to exist already.

On the configured WSL host, execute:

```bash
colabcola /path/to/jobs/kodak.colajob
```

The default configuration is one runtime and 900 seconds, not an estimate that the
full experiment will finish. Package with explicitly chosen `--max-runtimes`,
`--max-seconds` and `--timeout` values for a different bound. Repeating a command
reuses its journal without extending its budget. The general packaging route passed a fresh non-REST restart/recovery pilot on
2026-10-15: two T4 runtimes, step-50 restoration, 24 completed fits and verified
Drive/local artifacts, with both runtimes released in 353.4 seconds. This replay
used `--recovery-probe-wait 45` and `fault_after_checkpoint: true` in the execution
configuration. See [the revalidation evidence](../../NONREST_REVALIDATION.md).

The checkpoints are this program's own optional restart format. Other scripts can
run without any checkpoint code; after runtime loss they restart from the beginning.
See [the general packaging guide](../../PACKAGING.md) for modules and command sequences.
