"""Image-conditioned coordinate networks using the pinned external Hyperplus wheel."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import uuid

import hyperplus as hp
import numpy as np
from PIL import Image
import torch
from torch import nn

PIN = "24b446e9041724cb282763a90d724cf10edbcd24"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def library_identity():
    root = Path(hp.__file__).parent
    identity = {
        "version": importlib.metadata.version("hyperplus"),
        "source_commit": PIN,
        "modules": {p.name: digest(p) for p in sorted(root.glob("*.py"))},
    }
    profile = json.loads(Path(__file__).with_name("HYPERPLUS_PROFILE.json").read_text())
    if any(identity[key] != profile[key] for key in ("version", "source_commit", "modules")):
        raise ValueError("installed Hyperplus does not match the pinned source profile")
    return identity


def coordinates(height, width):
    """Row-major pixel centers; x grows right, y grows down."""
    y, x = torch.meshgrid((torch.arange(height) + 0.5) / height,
                          (torch.arange(width) + 0.5) / width, indexing="ij")
    return torch.stack((x, y), dim=-1).reshape(-1, 2)


class Fourier(nn.Module):
    def forward(self, xy):
        frequencies = xy.new_tensor([1, 2, 4, 8])
        angles = 2 * math.pi * xy.unsqueeze(-1) * frequencies
        return torch.cat((xy, angles.sin().flatten(-2), angles.cos().flatten(-2)), -1)


class SpatialMeanPool4(nn.Module):
    """Adaptive 4x4 bin averages using reductions with deterministic CUDA backward."""
    def forward(self, value):
        height, width = value.shape[-2:]
        bins = []
        for row in range(4):
            top, bottom = row * height // 4, ((row + 1) * height + 3) // 4
            for column in range(4):
                left, right = column * width // 4, ((column + 1) * width + 3) // 4
                bins.append(value[..., top:bottom, left:right].mean(dim=(-2, -1)))
        return torch.stack(bins, dim=-1).reshape(*value.shape[:-2], 4, 4)


class HOSC(nn.Module):
    """Hyperbolic oscillation: tanh(beta * sin(omega * x)).

    Serrano et al., arXiv:2401.10967; Wlodarczyk et al., arXiv:2601.07870.
    Fixed, explicit settings keep the first comparison an activation-only change.
    """
    def __init__(self, beta=1.0, omega=1.0):
        super().__init__()
        if not all(math.isfinite(v) and v > 0 for v in (beta, omega)):
            raise ValueError("HOSC beta and omega must be finite and positive")
        self.beta, self.omega = float(beta), float(omega)

    def forward(self, value):
        return torch.tanh(self.beta * torch.sin(self.omega * value))


def model(width=32, latent=64, *, activation="tanh", hosc_beta=1.0, hosc_omega=1.0):
    if activation not in ("tanh", "hosc"):
        raise ValueError("unknown coordinate activation")
    nonlinearity = nn.Tanh if activation == "tanh" else lambda: HOSC(hosc_beta, hosc_omega)
    target = nn.Sequential(Fourier(), nn.Linear(18, width), nonlinearity(),
                           nn.Linear(width, width), nonlinearity(), nn.Linear(width, 3),
                           nn.Sigmoid())
    # Hyperplus deliberately leaves generated-weight initialization to its caller.
    base = torch.cat([p.detach().flatten() for p in target.parameters()])
    encoder = nn.Sequential(
        nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.GELU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
        SpatialMeanPool4(), nn.Flatten(),
        nn.Linear(512, latent), nn.Tanh(), nn.Linear(latent, len(base)))
    nn.init.normal_(encoder[-1].weight, std=0.001)
    with torch.no_grad():
        encoder[-1].bias.copy_(base)
    return hp.HyperPlus(target, encoder)


def load_data(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    arrays = {}
    for split in ("train", "heldout"):
        path = root / (split + ".npy")
        if digest(path) != manifest["files"][path.name]["sha256"]:
            raise ValueError("dataset hash mismatch: " + split)
        array = np.load(path, allow_pickle=False)
        if (array.dtype != np.uint8 or array.ndim != 4 or array.shape[1] != 3
                or min(array.shape[0], array.shape[2], array.shape[3]) < 1):
            raise ValueError("expected nonempty uint8 NCHW RGB data")
        arrays[split] = torch.from_numpy(array.copy()).float() / 255
    if arrays["train"].shape[1:] != arrays["heldout"].shape[1:]:
        raise ValueError("split image shapes differ")
    return arrays, digest(root / "manifest.json")


def save_checkpoint(out, state):
    name = f"state-{state['step']:08d}.pt"
    path = out / name
    if path.exists() or (out / (name + ".json")).exists():
        raise ValueError("refusing to overwrite an immutable checkpoint")
    temporary = out / (name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    manifest = out / (name + ".json")
    atomic_json(manifest, {"schema": 1, "step": state["step"], "file": name,
                          "bytes": path.stat().st_size, "sha256": digest(path)})
    return manifest


def load_checkpoint(manifest_path):
    manifest_path = Path(manifest_path)
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    name = record["file"]
    if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("invalid checkpoint member path")
    path = manifest_path.parent / name
    if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
        raise ValueError("checkpoint integrity failure")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if record["schema"] != 1 or state["schema"] != 1 or state["step"] != record["step"]:
        raise ValueError("checkpoint schema/step mismatch")
    return state


@torch.no_grad()
def evaluate(net, images, grid, device):
    errors, baseline_errors, recon = [], [], []
    for start in range(0, len(images), 8):
        batch = images[start:start + 8].to(device)
        pred = net(batch, grid.expand(len(batch), -1, -1))
        truth = batch.permute(0, 2, 3, 1).reshape(len(batch), -1, 3)
        errors.extend((pred - truth).square().mean((1, 2)).cpu().tolist())
        baseline_errors.extend((truth - truth.mean(1, keepdim=True)).square().mean((1, 2)).cpu().tolist())
        if len(recon) < 8:
            recon.extend(pred.cpu())
    mse = sum(errors) / len(errors)
    return {"images": len(images), "rgb_values_per_image": int(images[0].numel()),
            "mse": mse, "psnr_db_from_mean_mse": -10 * math.log10(max(mse, 1e-12)),
            "conditioning_image_mean_color_mse": sum(baseline_errors) / len(baseline_errors)}, recon[:8]


def train(args):
    if min(args.steps, args.batch, args.pixels, args.width, args.latent, args.checkpoint_every) < 1:
        raise ValueError("training counts must be positive")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
    arrays, data_hash = load_data(args.data)
    height, width = arrays["train"].shape[-2:]
    grid = coordinates(height, width).to(device)
    net = model(args.width, args.latent, activation=args.activation,
                hosc_beta=args.hosc_beta, hosc_omega=args.hosc_omega).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    sampler = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    config = {key: getattr(args, key) for key in
              ("width", "latent", "batch", "pixels", "lr", "seed", "threads",
               "activation", "hosc_beta", "hosc_omega")}
    identity = {"config": config, "dataset_sha256": data_hash,
                "library": library_identity(), "workload_sha256": digest(__file__),
                "torch": str(torch.__version__), "device_type": device.type}
    start, losses = 0, []
    if args.resume:
        state = load_checkpoint(args.resume)
        if state["identity"] != identity:
            raise ValueError("resume identity/config/data/environment mismatch")
        start, losses = state["step"], state["losses"]
        if args.steps <= start:
            raise ValueError("target steps must exceed restored step")
        net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        sampler.set_state(state["sampler_rng"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    attempt = uuid.uuid4().hex
    atomic_json(out / "identity.json", {**identity, "attempt": attempt, "restored_step": start})
    initial = {}
    for split, images in arrays.items():
        initial[split] = evaluate(net, images, grid, device)[0]
    gradient_norm = None
    with (out / "events.ndjson").open("w", encoding="utf-8") as events:
        def emit(event, **values):
            row = {"event": event, "attempt": attempt, **values}
            line = json.dumps(row, sort_keys=True, allow_nan=False)
            events.write(line + "\n")
            events.flush()
            print(line, flush=True)
        emit("started", restored_step=start, target_steps=args.steps)
        for step in range(start + 1, args.steps + 1):
            indices = torch.randint(len(arrays["train"]), (args.batch,), generator=sampler)
            pixels = torch.randint(height * width, (args.pixels,), generator=sampler)
            batch = arrays["train"][indices].to(device)
            truth = batch.permute(0, 2, 3, 1).reshape(args.batch, -1, 3)[:, pixels].to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = net(batch, grid[pixels].expand(args.batch, -1, -1))
            loss = (predictions - truth).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            grads = [p.grad for p in net.parameters() if p.grad is not None]
            if not grads or any(not torch.isfinite(g).all() for g in grads):
                raise ValueError("missing or nonfinite Hyperplus gradient")
            gradient_norm = float(torch.stack([g.norm().square() for g in grads]).sum().sqrt())
            if gradient_norm <= 0:
                raise ValueError("zero Hyperplus gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
            emit("train_step", step=step, mse=losses[-1], gradient_norm=gradient_norm,
                 image_indices=indices.tolist())
            if step % args.checkpoint_every == 0 or step == args.steps:
                checkpoint = save_checkpoint(out, {
                    "schema": 1, "identity": identity, "step": step, "losses": losses,
                    "model": net.state_dict(), "optimizer": optimizer.state_dict(),
                    "sampler_rng": sampler.get_state(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                })
                emit("checkpoint_committed_local", step=step, manifest=checkpoint.name,
                     sha256=digest(checkpoint))
        result = {"step": args.steps, "restored_step": start, "initial": initial,
                  "final": {}, "gradient_norm_last": gradient_norm, "slots": net.slot_counts,
                  "lifecycle_scope": "local workload only; no Drive or runtime recovery claim"}
        for split, images in arrays.items():
            metrics, recon = evaluate(net, images, grid, device)
            result["final"][split] = metrics
            rows = []
            for i, predicted in enumerate(recon):
                reference = images[i].permute(1, 2, 0).numpy()
                rendered = predicted.reshape(height, width, 3).numpy()
                rows.append(np.concatenate((reference, rendered), axis=1))
            tile = (np.concatenate(rows, axis=0).clip(0, 1) * 255).round().astype(np.uint8)
            Image.fromarray(tile).resize((width * 8, height * 4 * len(rows)),
                                         Image.Resampling.NEAREST).save(out / (split + ".png"))
        atomic_json(out / "metrics.json", result)
        emit("completed_local", step=args.steps)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True, help="new directory for this attempt")
    p.add_argument("--resume", default=os.environ.get("COLABCOLA_RESUME_MANIFEST"), help="verified local checkpoint commit manifest")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--pixels", type=int, default=256)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--activation", choices=("tanh", "hosc"), default="tanh")
    p.add_argument("--hosc-beta", type=float, default=1.0, help="positive HOSC saturation/sharpness")
    p.add_argument("--hosc-omega", type=float, default=1.0, help="positive HOSC carrier frequency")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return p


if __name__ == "__main__":
    train(parser().parse_args())
