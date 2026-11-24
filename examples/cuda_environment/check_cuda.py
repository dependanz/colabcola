"""Compile and execute a small CUDA operator using the declared environment."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time


EXPECTED_CUDA = "12.8"
EXPECTED_TORCH = "2.7.1+cu128"


def capture(argv: list[str]) -> str:
    return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT, timeout=20).strip()


def release_version(output: str) -> str:
    match = re.search(r"release\s+([0-9]+\.[0-9]+)", output)
    if match is None:
        raise RuntimeError("Cannot identify the CUDA compiler version")
    return match.group(1)


def inside(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def host_compiler(prefix: Path) -> dict | None:
    # Read host compiler identity explicitly. Never use it to build the extension.
    candidates = [Path("/usr/local/cuda/bin/nvcc"), Path("/usr/bin/nvcc")]
    candidates.extend(sorted(Path("/usr/local").glob("cuda*/bin/nvcc")))
    seen = set()
    for candidate in candidates:
        if not candidate.is_file() or inside(candidate, prefix):
            continue
        resolved = str(candidate.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            version = release_version(capture([str(candidate), "--version"]))
        except (OSError, subprocess.SubprocessError, RuntimeError):
            continue
        return {"path": resolved, "version": version}
    return None


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    prefix = Path(sys.prefix).resolve()
    nvcc = Path(shutil.which("nvcc") or "missing-nvcc").resolve()
    cuda_home = Path(CUDA_HOME or "missing-cuda-home").resolve()
    compiler_words = shlex.split(os.environ.get("CXX", "c++"))
    if len(compiler_words) != 1:
        raise RuntimeError("Expected one executable in the activated CXX variable")
    cxx = Path(shutil.which(compiler_words[0]) or "missing-cxx").resolve()
    checks = {
        "python_3_12": sys.version_info[:2] == (3, 12),
        "torch_build_matches": str(torch.__version__) == EXPECTED_TORCH,
        "torch_cuda_matches": torch.version.cuda == EXPECTED_CUDA,
        "cuda_available": torch.cuda.is_available(),
        "cuda_home_is_environment": cuda_home == prefix,
        "nvcc_is_environment_local": nvcc.is_file() and inside(nvcc, prefix),
        "torch_nvcc_is_environment_local": (
            (cuda_home / "bin/nvcc").is_file()
            and (cuda_home / "bin/nvcc").resolve() == nvcc
        ),
        "cxx_is_environment_local": cxx.is_file() and inside(cxx, prefix),
        "single_compile_worker": os.environ.get("MAX_JOBS") == "1",
    }
    if not all(checks.values()):
        raise RuntimeError("Environment checks failed: " + json.dumps(checks, sort_keys=True))

    nvcc_version = release_version(capture([str(nvcc), "--version"]))
    cxx_version = capture([str(cxx), "-dumpfullversion"])
    checks["nvcc_matches"] = nvcc_version == EXPECTED_CUDA
    checks["supported_cxx"] = cxx_version.split(".")[0] == "13"

    # These are the wheel-only headers required by ATen's CUDA context headers.
    include_dirs = [Path(item).resolve() for item in os.environ.get("CPATH", "").split(os.pathsep) if item]
    target_include = prefix / "targets/x86_64-linux/include"
    toolkit_header = target_include / "cuda_runtime.h"
    checks["target_include_search_path"] = target_include.resolve() in include_dirs
    checks["target_runtime_header"] = toolkit_header.is_file() and inside(toolkit_header, prefix)
    header_sources = {}
    for header in ("cublas_v2.h", "cusparse.h", "cusolverDn.h"):
        matches = [directory / header for directory in include_dirs if (directory / header).is_file()]
        wheel_matches = [path for path in matches if "site-packages/nvidia/" in path.as_posix() and inside(path, prefix)]
        header_sources[header] = [str(path) for path in wheel_matches]
        checks[header + "_from_environment_wheel"] = bool(wheel_matches)
        checks[header + "_requires_wheel_search_path"] = bool(matches) and all(
            "site-packages/nvidia/" in path.as_posix() for path in matches
        )
    if not all(checks.values()):
        raise RuntimeError("Toolchain checks failed: " + json.dumps(checks, sort_keys=True))

    host = host_compiler(prefix)
    newer_host = bool(host) and tuple(map(int, host["version"].split("."))) > tuple(
        map(int, EXPECTED_CUDA.split("."))
    )
    properties = torch.cuda.get_device_properties(0)
    architecture = f"{properties.major}.{properties.minor}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = architecture
    build = Path("build").resolve()
    build.mkdir(exist_ok=True)
    started = time.monotonic()
    print(f"Building with environment CUDA {nvcc_version}, C++ {cxx_version}, GPU architecture {architecture}", flush=True)
    load(
        name="colabcola_cuda_example",
        sources=[str(Path(__file__).with_name("extension.cu").resolve())],
        build_directory=str(build),
        extra_cflags=["-O0"],
        extra_cuda_cflags=["-O0"],
        with_cuda=True,
        is_python_module=False,
        verbose=True,
    )
    elapsed = time.monotonic() - started
    values = torch.arange(4096, device="cuda", dtype=torch.float32)
    actual = torch.ops.colabcola_cuda_example.add_one(values)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, values + 1, rtol=0, atol=0)
    checks["kernel_result_exact"] = True
    library = build / "colabcola_cuda_example.so"
    report = {
        "schema": 1,
        "extension_test": "passed",
        "checks": checks,
        "host_mismatch_exercised": newer_host,
        "host_mismatch_detail": "newer host toolkit observed" if newer_host else "host mismatch not exercised",
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "environment_prefix": str(prefix),
        "cuda_home": str(cuda_home),
        "nvcc": {"path": str(nvcc), "version": nvcc_version},
        "cxx": {"path": str(cxx), "version": cxx_version},
        "host_nvcc": host,
        "toolkit_runtime_header": str(toolkit_header),
        "wheel_headers": header_sources,
        "gpu": {"name": properties.name, "architecture": architecture},
        "build_seconds": elapsed,
        "compiled_library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "elements_checked": values.numel(),
        "maximum_absolute_error": float((actual - values - 1).abs().max().item()),
    }
    write_json(Path("outputs/cuda/acceptance.json"), report)
    write_json(Path("outputs/cuda/complete.json"), {"state": "complete"})
    print("CUDA extension build and kernel checks passed", flush=True)
    print(report["host_mismatch_detail"], flush=True)


if __name__ == "__main__":
    main()
