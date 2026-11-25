# Targeted environment wheel example (issue #43, M0)

From a disposable copy of this directory, run `python build_wheel.py` to create
an offline, deterministic `py3-none-any` fixture wheel. It requires only the
standard library and does not install anything into the controller's Python.
With an installed controller that supports named `pip.wheels`, prepare and run
`pipeline.colaml` through the managed lifecycle on one CPU runtime.

The `train` environment must import the packaged wheel and return 17. The
`evaluate` environment has the same Python version but must not have the wheel
installed. Verify `outputs/acceptance.json`, the wheel's path, size, and SHA-256
in `result.json` under `environment.environments.environments.train.embedded_wheels`,
both ordered step exit codes, completed outputs, and runtime release. A separate
automated Linux test also builds and installs an actual CPython C extension wheel.
This fixture is intentionally CPU-only and does not claim CUDA wheel ABI coverage.
The same recipe can opt in to persistent environment caching with `cache = true`
under `environments.train` or `environments.evaluate`. Live Colab cache acceptance
is still pending.
