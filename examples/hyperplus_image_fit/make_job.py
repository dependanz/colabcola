"""Create an explicit JobSpec; this command never uploads or allocates compute."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform


def make_job(args):
    root = Path(__file__).parent.resolve()
    wheel, data, output = Path(args.wheel).resolve(), Path(args.data).resolve(), Path(args.output).resolve()
    profile = json.loads((root / "HYPERPLUS_PROFILE.json").read_text())
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != profile["wheel_sha256"]:
        raise ValueError("wheel differs from the verified pinned dependency")
    for name in ("train.npy", "heldout.npy", "manifest.json"):
        if not (data / name).is_file():
            raise ValueError("missing prepared dataset member: " + name)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    def include(role, source, target):
        payload.append({"role": role, "source": os.path.relpath(source, output.parent).replace("\\", "/"), "target": target})
    include("source", root / "train.py", "workload/train.py")
    include("config", root / "HYPERPLUS_PROFILE.json", "workload/HYPERPLUS_PROFILE.json")
    include("wheel", wheel, "wheels/" + wheel.name)
    for name in ("train.npy", "heldout.npy", "manifest.json"):
        include("data", data / name, "data/" + name)
    runtime = getattr(args, "runtime", "2026.07")
    runtime_profile = None
    if not args.local_cpu and runtime == "latest":
        profile_path = root / "COLAB_RUNTIME_20261015.json"
        runtime_profile = json.loads(profile_path.read_text())
        include("config", profile_path, "workload/COLAB_RUNTIME_20261015.json")
    packages = ["torch", "numpy", "pillow"]
    requirements = ([f"{name}=={importlib.metadata.version(name)}" for name in packages]
                    if args.local_cpu else ["torch==2.11.0+cu128", "numpy==2.0.2", "Pillow>=10,<13"])
    if runtime_profile:
        requirements = [f"{name}=={runtime_profile['packages'][name]}" for name in packages]
    python_version = (platform.python_version() if args.local_cpu else
                      runtime_profile["python"] if runtime_profile else "3.12.13")
    job = {
        "schema_version": 1, "name": args.name, "payload": payload,
        "entrypoint": {"kind": "python_script", "path": "workload/train.py", "cwd": ".",
                       "argv": ["--data", "data", "--output", "outputs/fit", "--steps", str(args.steps),
                                "--checkpoint-every", str(min(25, args.steps)),
                                "--device", "cpu" if args.local_cpu else "cuda"]},
        "python": {"requires_python": "==" + python_version,
                   "runtime_requirements": requirements, "install_requirements": [],
                   "embedded_wheels": ["wheels/" + wheel.name]},
        "runtime": {"provider": "google_colab", "version": getattr(args, "runtime", "2026.07")},
        "resources": {"accelerator": "cpu" if args.local_cpu else "gpu",
                      "gpu_count": 0 if args.local_cpu else 1, "cuda_required": not args.local_cpu,
                      "minimum_gpu_memory_bytes": 0 if args.local_cpu else 4 * 1024**3},
        "run": {"timeout_seconds": 1800, "seed": 1234,
                "environment": {"PYTHONUNBUFFERED": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
                "capture_stdout": True, "capture_stderr": True},
        "outputs": [{"pattern": "outputs/fit/" + name, "required": True, "maximum_bytes": size}
                    for name, size in [
                        ("identity.json", 1048576), ("metrics.json", 1048576),
                        ("events.ndjson", 16777216), ("train.png", 1048576), ("heldout.png", 1048576),
                        (f"state-{args.steps:08d}.pt", 33554432),
                        (f"state-{args.steps:08d}.pt.json", 1048576)]],
        "limits": {"maximum_file_count": 32},
        "managed_recovery": {
            "checkpoint_pattern": "outputs/fit/state-*.pt.json",
            "log_patterns": ["outputs/fit/events.ndjson", "outputs/fit/metrics.json", "outputs/fit/identity.json"],
        },
    }
    for key in ("activation", "hosc_beta", "hosc_omega", "width", "latent", "lr", "seed"):
        if hasattr(args, key):
            job["entrypoint"]["argv"].extend(("--" + key.replace("_", "-"), str(getattr(args, key))))
    if args.local_cpu:
        job["resources"]["minimum_gpu_memory_bytes"] = None
    output.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wheel", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--name", default="hyperplus-cifar10-proof")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--runtime", choices=("2026.07", "latest"), default="2026.07")
    p.add_argument("--activation", choices=("tanh", "hosc"), default="tanh")
    p.add_argument("--hosc-beta", type=float, default=1.0)
    p.add_argument("--hosc-omega", type=float, default=1.0)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--local-cpu", action="store_true", help="local diagnostic runtime, not Colab GPU proof")
    make_job(p.parse_args())
