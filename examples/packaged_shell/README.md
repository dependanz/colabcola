# Packaged Bash scripts in two environments

This CPU example runs two regular `.sh` files without local execution during
packaging. The first uses Python 3.10 and the second uses Python 3.12, each through
its selected environment's activated `PATH`. Arguments with spaces, dollar signs,
command substitution syntax, and semicolons remain literal values. The second
step runs from a nested working directory.

From this directory, with a configured colabcola installation supporting packaged
Bash scripts, choose a fresh run directory:

```bash
run_dir="$HOME/colabcola-runs/packaged-shell-1"
TERM=dumb colabcola run --profile default --colaml pipeline.colaml \
  --run-dir "$run_dir" --non-interactive
```

The single command performs preparation, preflight, allocation, execution,
verification, recovery, and runtime release. The recipe requests one CPU runtime
and a 20-minute total budget.

Acceptance requires the `prepare stdout`, `prepare stderr`, `evaluate stdout`,
and `evaluate stderr` messages in the recovered logs; two successful ordered step
records; `outputs/acceptance.json` recording `state: passed`, Python 3.10 and 3.12,
the exact argument values, and working directory `scripts`; a verified completed
output group; and an owned runtime release. The sample intentionally uses no
external workload dependencies.

The scripts must be included project files. Bash is invoked without startup files
or shell interpolation of the recipe arguments. Script paths are normalized POSIX
paths relative to each step's `cwd`. This is a trusted-code execution model, not a
sandbox for shell code. Arbitrary `bash -c` strings and interpreter flags are not
accepted by the recipe.
