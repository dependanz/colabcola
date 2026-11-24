# Build a CUDA extension with an environment-local toolkit

This example compiles a small PyTorch CUDA operator on a T4 runtime and checks its
output exactly. It uses Python 3.12, PyTorch 2.7.1 with CUDA 12.8, an explicitly
declared CUDA 12.8 compiler, and a matching conda C++ compiler. The allocated
runtime supplies the GPU driver; its system CUDA toolkit is recorded only as
comparison evidence.

The declared toolkit is deliberately minimal. It provides the compiler, CUDA
runtime development files, and CCCL headers. The `cublas`, `cusparse`, and
`cusolver` headers required by `ATen/cuda/CUDAContext.h` come from PyTorch's NVIDIA
wheel dependencies. The test checks that colabcola exposes those header paths
inside the selected environment before compiling. It also verifies that the
toolkit's `targets/x86_64-linux/include` directory is on the compiler search path
and contains its CUDA runtime header.

The extension uses PyTorch operator registration rather than Python bindings to
keep compilation small. `MAX_JOBS=1` limits compilation to one worker, and the
build targets the allocated GPU's actual architecture. The kernel adds one to
4,096 float32 values and checks every output with zero tolerance.

## Run the example

Use a colabcola build with named environments and a configured login profile. From
this directory, choose a new run directory that does not already exist:

```bash
run_dir="$HOME/colabcola-runs/cuda-environment-1"
TERM=dumb colabcola run --profile default --colaml pipeline.colaml \
  --run-dir "$run_dir" --non-interactive --prepare-only
TERM=dumb colabcola run --profile default --colaml pipeline.colaml \
  --run-dir "$run_dir" --non-interactive --preflight-only
TERM=dumb colabcola run --profile default --colaml pipeline.colaml \
  --run-dir "$run_dir" --non-interactive
python verify_run.py "$run_dir"
```

Run each command after the preceding one succeeds. `python` must refer to the same
installation that supplies the `colabcola` command. `TERM=dumb` requests readable
plain progress output for this command only.

The recipe requests one T4 runtime, sets a 30-minute total managed budget and a
10-minute workload timeout, and publishes a completed output group after the
kernel check succeeds. The first run downloads the declared dependencies and
builds its environment. This example does not claim persistent environment or
native-wheel caching.

## What constitutes a pass

`verify_run.py` verifies the recovered final artifact bundle, the completed output
snapshot, the successful setup and step records, the runtime release record, and
the extension's own checks. Its summary must end with:

```text
CUDA ACCEPTANCE PASSED
```

Two outcomes are recorded separately:

- **Extension build and execution:** the selected environment's compiler and
  headers produced a working CUDA operator, and all numerical checks passed.
- **Newer host toolkit exercised:** a system `nvcc` was observed with a newer
  major/minor version than the selected CUDA 12.8 compiler. This proves the run
  exercised toolkit isolation rather than merely using two matching installations.

If the host has no observable `nvcc`, or has the same or an older toolkit, the
extension can still succeed and its artifacts are retained. The verifier reports
`host mismatch not exercised` and refuses the full acceptance pass because that
specific compatibility scenario remains untested.

The recovered `artifacts.zip` contains `outputs/cuda/acceptance.json` with compiler,
framework, header, GPU, build, and numerical evidence. The compiled shared library
is hashed in that report but is not included as an output or reused by another run.

## Dependency references

- [PyTorch 2.7.1 CUDA 12.8 installation](https://pytorch.org/get-started/previous-versions/#v271)
- [Official CUDA 12.8 PyTorch wheel index](https://download.pytorch.org/whl/cu128/torch/)
- [NVIDIA CUDA 12.8.1 conda channel](https://anaconda.org/nvidia/repo?label=cuda-12.8.1)
- [CUDA 12.8.1 Linux installation guide](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/)
- [PyTorch C++ extension API](https://docs.pytorch.org/docs/stable/cpp_extension.html)
