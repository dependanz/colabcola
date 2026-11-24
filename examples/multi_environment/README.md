# Ordered steps in two Python environments

This example prepares a small JSON file in Python 3.10, then reads and evaluates it
in Python 3.12. Both steps use the same packaged project and output directories.
The controller's Python installation is independent of those workload versions.
The example requests a CPU runtime and has no third-party workload dependencies.

Use a colabcola build that includes named recipe environments. From this directory:

```sh
colabcola login
colabcola run --colaml pipeline.colaml --non-interactive \
  --run-dir "$HOME/colabcola-runs/multi-environment-1" --prepare-only
colabcola run --colaml pipeline.colaml --non-interactive \
  --run-dir "$HOME/colabcola-runs/multi-environment-1" --preflight-only
colabcola run --colaml pipeline.colaml --non-interactive \
  --run-dir "$HOME/colabcola-runs/multi-environment-1"
```

The managed command uses your selected login profile even during preparation.
The first command prepares a frozen job; the second checks admission without
allocating compute; the third runs it and recovers the results. Each repeat uses
the same prepared job and budget. To start another run, choose a new run directory.

Check the recovered result and artifacts:

- Environment setup is a separate stage. `environments.json` records requested
  specifications, resolved interpreters and installed package versions.
- `steps.json` records `prepare` followed by `evaluate` with their environment
  names, timestamps and exit codes. The same execution evidence is included in
  `result.json`.
- `outputs/evaluation.json` contains `sum: 17`, a Python 3.10 preparation version
  and a Python 3.12 evaluation version.
- `outputs/prepare/complete.json` protects the preparation data through a declared
  completed output group. Its `files` paths are relative to the marker's directory.

This example checks interpreter separation and ordered execution. CUDA compilation
needs a separate GPU acceptance test. Environment restoration across runs belongs
to the environment-cache feature; this implementation creates fresh environments.

See [the recipe guide](../../docs/RECIPES.md#named-environments-and-ordered-steps)
for conda packages, pip requirements, CUDA toolkits and per-step working directories.
