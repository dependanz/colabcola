"""Bounded native build and CUDA acceptance probe, runnable with python -m."""
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def run_command(argv, deadline, output):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("pilot setup budget exhausted")
    print("Running: " + " ".join(argv), flush=True)
    with (output / "build.log").open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND " + json.dumps(argv) + "\n")
        log.flush()
        started = time.monotonic()
        try:
            result = subprocess.run(argv, check=True, timeout=remaining, stdout=log,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        except subprocess.CalledProcessError as exc:
            log.write(f"\nFAILED exit={exc.returncode} elapsed={time.monotonic()-started:.3f}s\n")
            raise
        else:
            log.write(f"\nPASSED exit={result.returncode} elapsed={time.monotonic()-started:.3f}s\n")


def smoke(output):
    import torch
    import nvdiffrast.torch as dr
    from simple_knn._C import distCUDA2
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from utils.graphics_utils import getProjectionMatrix
    # Import the real training module without starting training or loading licensed assets.
    importlib.import_module("train")
    torch.manual_seed(1234)
    xyz = torch.tensor([[-.1, -.1, 2.], [.1, -.1, 2.], [0., .1, 2.], [0., 0., 2.1]], device="cuda", requires_grad=True)
    distance = distCUDA2(xyz.detach())
    assert torch.isfinite(distance).all() and (distance > 0).all()
    settings = GaussianRasterizationSettings(
        image_height=32, image_width=32, tanfovx=1., tanfovy=1.,
        bg=torch.zeros(3, device="cuda"), scale_modifier=1.,
        viewmatrix=torch.eye(4, device="cuda"),
        projmatrix=getProjectionMatrix(.01, 100., 1.5707963267948966, 1.5707963267948966).T.contiguous().cuda(),
        sh_degree=0, campos=torch.zeros(3, device="cuda"), prefiltered=False, debug=False)
    colors = torch.full((4, 3), .6, device="cuda", requires_grad=True)
    screen = torch.zeros_like(xyz, requires_grad=True)
    image, radii = GaussianRasterizer(settings)(
        means3D=xyz, means2D=screen, opacities=torch.full((4, 1), .8, device="cuda"),
        colors_precomp=colors, scales=torch.full((4, 3), .1, device="cuda"),
        rotations=torch.tensor([[1., 0., 0., 0.]], device="cuda").repeat(4, 1))
    image.square().mean().backward()
    assert torch.isfinite(image).all() and (radii > 0).any()
    assert colors.grad is not None and torch.isfinite(colors.grad).all() and colors.grad.abs().sum() > 0
    pos = torch.tensor([[[-.5, -.5, 0., 1.], [.5, -.5, 0., 1.], [0., .5, 0., 1.]]], device="cuda", requires_grad=True)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
    rast, _ = dr.rasterize(dr.RasterizeCudaContext(), pos, tri, resolution=[32, 32])
    rast[..., :2].square().sum().backward()
    assert (rast[..., 3] > 0).any() and torch.isfinite(rast).all()
    assert pos.grad is not None and torch.isfinite(pos.grad).all() and pos.grad.abs().sum() > 0
    from PIL import Image
    Image.fromarray((image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).astype("uint8")).save(output / "raster-smoke.png")
    return {"training_module_import": True, "simple_knn_forward": True,
            "gaussian_forward_backward": True, "nvdiffrast_forward_backward": True,
            "avatar_training": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--build-seconds", type=int, default=660)
    parser.add_argument("--output", type=Path, default=Path("outputs/environment"))
    args = parser.parse_args()
    if not 1 <= args.build_seconds <= 660:
        parser.error("build-seconds must be between 1 and 660")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"state": "started", "avatar_training": False, "python": sys.version}
    started = time.monotonic()
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
        if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "Tesla T4":
            raise RuntimeError("This pilot requires exactly a Tesla T4")
        if torch.__version__.split("+")[0] != "2.11.0":
            raise RuntimeError("Pilot targets Torch 2.11.0; prepare a new job for a different stack")
        nvcc = Path(CUDA_HOME or "/missing") / "bin/nvcc"
        if not nvcc.is_file() or not shutil.which("g++"):
            raise RuntimeError("CUDA toolkit nvcc and g++ are required; this pilot does not install system toolchains")
        report.update(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, torch_cuda=torch.version.cuda,
                      cuda_home=CUDA_HOME, capability=list(torch.cuda.get_device_capability(0)))
        os.environ.update(TORCH_CUDA_ARCH_LIST="7.5", MAX_JOBS="2", PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_PROGRESS_BAR="off")
        deadline = started + args.build_seconds
        report["toolchain"] = {}
        for label, argv in (("nvcc", [str(nvcc), "--version"]), ("g++", ["g++", "--version"])):
            version = subprocess.run(argv, check=True, capture_output=True, text=True, timeout=10)
            report["toolchain"][label] = version.stdout.strip()
        if args.build:
            from .compat import apply
            report["compatibility_patch"] = apply(Path.cwd())
            (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            run_command([sys.executable, "-m", "pip", "install", "--verbose", "--log", str(args.output / "pip.log"), "-r", "pilot-requirements.txt"], deadline, args.output)
            for source in ("submodules/simple-knn", "submodules/diff-gaussian-rasterization", "vendor/nvdiffrast"):
                run_command([sys.executable, "-m", "pip", "install", "--verbose", "--log", str(args.output / "pip.log"), "--no-build-isolation", "--no-deps", "./" + source], deadline, args.output)
        run_command([sys.executable, "-m", "pip", "check"], deadline, args.output)
        report.update(smoke(args.output), state="passed")
    except Exception as exc:
        report.update(state="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        try:
            freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=10)
            (args.output / "pip-freeze.txt").write_text(freeze.stdout, encoding="utf-8")
        except Exception:
            pass
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
