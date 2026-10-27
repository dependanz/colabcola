"""Paired per-image HOSC/SIREN fits: deterministic, checkpointable workflow workload."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
from PIL import Image
import torch
from torch import nn


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


BASE = "https://r0k.us/graphics/kodak/kodak/"


def prepare_data(destination):
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    previous = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else None
    def acquire(index):
        name = f"kodim{index:02d}.png"
        path = root / name
        if not path.exists():
            request = urllib.request.Request(BASE + name, headers={"User-Agent": "colabcola-kodak-workflow/1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read(2 * 1024 * 1024 + 1)
            if len(body) > 2 * 1024 * 1024:
                raise ValueError("Kodak download exceeds bound")
            temp = path.with_suffix(".download")
            temp.write_bytes(body)
            with Image.open(temp) as image:
                image.verify()
            temp.replace(path)
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size not in ((768, 512), (512, 768)):
                raise ValueError("unexpected Kodak image format/dimensions")
            shape = [image.height, image.width, 3]
        return {"name": name, "url": BASE + name, "bytes": path.stat().st_size,
                "sha256": digest(path), "shape": shape}
    with ThreadPoolExecutor(max_workers=4) as pool:
        images = list(pool.map(acquire, range(1, 25)))
    result = {"schema": 1, "dataset": "Kodak", "source": "https://r0k.us/graphics/kodak/",
              "transformation": "none; original RGB PNGs, native dimensions", "images": images}
    if previous and previous != result:
        raise ValueError("existing Kodak manifest no longer matches; preserve and inspect")
    atomic(root / "manifest.json", result)
    print(json.dumps({"path": str(root), "images": len(images), "bytes": sum(x["bytes"] for x in images),
                      "manifest_sha256": digest(root / "manifest.json")}, indent=2))


def tensor_hash(values):
    h = hashlib.sha256()
    for key, tensor in sorted(values.items()):
        h.update(key.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Pair(nn.Module):
    """Leading dimension has two independent networks; losses are summed, not averaged."""
    def __init__(self, depth, width, beta, omega, seed):
        super().__init__()
        self.beta, self.omega = beta, omega
        self.weights, self.biases = nn.ParameterList(), nn.ParameterList()
        generator = torch.Generator().manual_seed(seed)
        sizes = [2] + [width] * depth + [3]
        for index, (nin, nout) in enumerate(zip(sizes, sizes[1:])):
            bound = 1 / nin if index == 0 else math.sqrt(6 / nin) / omega
            w = torch.empty(nin, nout).uniform_(-bound, bound, generator=generator)
            b = torch.empty(1, nout).uniform_(-1 / math.sqrt(nin), 1 / math.sqrt(nin), generator=generator)
            self.weights.append(nn.Parameter(w.unsqueeze(0).repeat(2, 1, 1)))
            self.biases.append(nn.Parameter(b.unsqueeze(0).repeat(2, 1, 1)))

    def forward(self, xy):
        value = xy.unsqueeze(0).expand(2, -1, -1)
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            value = torch.bmm(value, weight) + bias
            if i < len(self.weights) - 1:
                sine = torch.sin(self.omega * value)
                value = torch.stack((torch.tanh(self.beta * sine[0]), sine[1]))
        return value

    def initial_hashes(self):
        return [tensor_hash({name: param[i] for name, param in self.named_parameters()}) for i in range(2)]


def coordinates(height, width):
    y, x = torch.meshgrid((torch.arange(height) + .5) / height,
                          (torch.arange(width) + .5) / width, indexing="ij")
    return torch.stack((x, y), -1).reshape(-1, 2) * 2 - 1


def load_resume(path):
    path = Path(path)
    meta = json.loads(path.read_text())
    name = meta["file"]
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("invalid checkpoint path")
    state_file = path.parent / name
    if state_file.stat().st_size != meta["bytes"] or digest(state_file) != meta["sha256"]:
        raise ValueError("checkpoint integrity mismatch")
    state = torch.load(state_file, map_location="cpu", weights_only=True)
    if meta["schema"] != 2 or state["global_step"] != meta["step"]:
        raise ValueError("checkpoint step/schema mismatch")
    for name, record in meta["artifacts"].items():
        if not name.startswith("fits/") or len(Path(name).parts) != 2 or "\\" in name or ".." in name:
            raise ValueError("invalid artifact path")
        artifact = path.parent / name
        if artifact.stat().st_size != record["bytes"] or digest(artifact) != record["sha256"]:
            raise ValueError("completed fit integrity mismatch")
    return state, meta


def save_checkpoint(out, state, artifacts):
    path = out / f"state-{state['global_step']:09d}.pt"
    if path.exists():
        raise ValueError("checkpoint identity already exists")
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    atomic(path.with_suffix(".pt.json"), {"schema": 2, "step": state["global_step"],
        "file": path.name, "bytes": path.stat().st_size, "sha256": digest(path), "artifacts": artifacts})


def pair_seed(seed, image, depth, width):
    return int.from_bytes(hashlib.sha256(f"{seed}:{image}:{depth}:{width}".encode()).digest()[:8], "little") % (2**63 - 1)


def run(args):
    if min(args.steps, args.pixels, args.checkpoint_every, args.beta, args.omega, args.lr) <= 0:
        raise ValueError("training settings must be positive")
    if not 0 <= args.recovery_probe_wait <= 60:
        raise ValueError("recovery probe wait must be within 0-60 seconds")
    if not math.isfinite(args.beta + args.omega + args.lr):
        raise ValueError("training settings must be finite")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "fits").mkdir()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    images = args.images or [r["name"] for r in manifest["images"]]
    records = {r["name"]: r for r in manifest["images"]}
    if len(images) != len(set(images)) or any(name not in records for name in images):
        raise ValueError("unknown or duplicate image")
    if any(d not in (1, 2, 3) for d in args.depths) or any(w not in (32, 64, 128, 256) for w in args.widths):
        raise ValueError("architecture outside declared grid")
    if len(set(args.depths)) != len(args.depths) or len(set(args.widths)) != len(args.widths):
        raise ValueError("duplicate architecture")
    pairs = [(name, d, w) for name in images for d in args.depths for w in args.widths]
    identity = {"schema": 1, "source_sha256": digest(__file__), "dataset_sha256": digest(data / "manifest.json"),
        "images": images, "depths": args.depths, "widths": args.widths, "steps": args.steps,
        "seed": args.seed, "pixels": args.pixels, "beta": args.beta, "omega": args.omega, "lr": args.lr,
        "torch": str(torch.__version__), "device": args.device,
        "coordinates": "pixel centers in [0,1]^2 mapped to [-1,1]^2",
        "output": "linear RGB mapped from [-1,1]; metrics unclipped; previews clipped"}
    state = {"identity": identity, "completed": [], "pair_index": 0, "step": 0, "global_step": 0,
             "model": None, "optimizer": None, "sampling_rng": None, "losses": []}
    artifacts = {}
    if args.resume:
        state, metadata = load_resume(args.resume)
        if state["identity"] != identity:
            raise ValueError("resume experiment identity differs")
        artifacts = metadata["artifacts"]
        for name in artifacts:
            shutil.copyfile(Path(args.resume).parent / name, out / name)
    restored_step = state["global_step"]
    atomic(out / "identity.json", {**identity, "restored_step": restored_step})
    started = time.monotonic()
    device = torch.device(args.device)
    events = (out / "events.ndjson").open("w", encoding="utf-8")

    def emit(event, **values):
        record = {"event": event, "elapsed_seconds": time.monotonic() - started, **values}
        line = json.dumps(record, sort_keys=True, allow_nan=False)
        events.write(line + "\n"); events.flush()
        print(line, flush=True)

    emit("started", restored_step=restored_step, completed_fits=len(state["completed"]) * 2,
         total_fits=len(pairs) * 2, seed=args.seed,
         gpu=torch.cuda.get_device_name() if device.type == "cuda" else None)
    for index in range(state["pair_index"], len(pairs)):
        name, depth, width = pairs[index]
        path = data / name
        if digest(path) != records[name]["sha256"]:
            raise ValueError("dataset image integrity mismatch")
        with Image.open(path) as img:
            array = np.asarray(img.convert("RGB")).copy()
        height, image_width = array.shape[:2]
        if [height, image_width] != records[name]["shape"][:2]:
            raise ValueError("dataset image dimensions differ")
        xy = coordinates(height, image_width).to(device)
        truth = torch.from_numpy(array).to(device).float().reshape(-1, 3) / 255
        target = truth * 2 - 1
        seed = pair_seed(args.seed, name, depth, width)
        pair = Pair(depth, width, args.beta, args.omega, seed).to(device)
        initial_hashes = pair.initial_hashes()
        if initial_hashes[0] != initial_hashes[1]:
            raise ValueError("activation initialization differs")
        optimizer = torch.optim.Adam(pair.parameters(), lr=args.lr)
        generator = torch.Generator().manual_seed(seed)
        first = state["step"] if index == state["pair_index"] else 0
        losses = state["losses"] if first else []
        if first:
            pair.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            generator.set_state(state["sampling_rng"])
        fit_id = f"{Path(name).stem}-d{depth}-w{width}-s{args.seed}"
        pair_started = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        emit("pair_started", pair=fit_id, initial_sha256=initial_hashes, restored_step=first)
        for step in range(first + 1, args.steps + 1):
            sample = torch.randint(len(xy), (args.pixels,), generator=generator).to(device)
            optimizer.zero_grad(set_to_none=True)
            per_fit = (pair(xy[sample]) - target[sample].unsqueeze(0)).square().mean(dim=(1, 2))
            if not torch.isfinite(per_fit).all():
                raise ValueError("nonfinite loss")
            per_fit.sum().backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in pair.parameters()):
                raise ValueError("nonfinite gradient")
            optimizer.step()
            losses.append(per_fit.detach().cpu().tolist())
            state.update(pair_index=index, step=step, global_step=index * (args.steps + 1) + step,
                         model=pair.state_dict(), optimizer=optimizer.state_dict(),
                         sampling_rng=generator.get_state(), losses=losses)
            if step % 25 == 0 or step == 1 or step == args.steps:
                progress = {"image": name, "depth": depth, "width": width, "step": step, "steps": args.steps,
                            "completed_fits": index * 2, "total_fits": len(pairs) * 2,
                            "hosc_rgb_mse": losses[-1][0] / 4, "siren_rgb_mse": losses[-1][1] / 4}
                atomic(out / "progress.json", progress)
                emit("progress", **progress)
            if step % args.checkpoint_every == 0 and step < args.steps:
                save_checkpoint(out, state, artifacts)
                emit("checkpoint_committed_local", global_step=state["global_step"])
                if args.recovery_probe_wait and not args.resume and index == 0 and step == args.checkpoint_every:
                    emit("recovery_probe_wait", seconds=args.recovery_probe_wait)
                    time.sleep(args.recovery_probe_wait)
                if args.stop_after and state["global_step"] >= args.stop_after:
                    events.close()
                    return
        with torch.no_grad():
            pred = torch.cat([(pair(chunk) + 1) / 2 for chunk in xy.split(16384)], dim=1)
        mse = (pred - truth.unsqueeze(0)).square().mean(dim=(1, 2)).cpu().tolist()
        metrics = {"pair": fit_id, "image": name, "depth": depth, "width": width, "seed": args.seed,
                   "pair_seed": seed, "initial_sha256": initial_hashes, "steps": args.steps,
                   "parameters_per_fit": sum(p.numel() for p in pair.parameters()) // 2,
                   "training_steps_this_attempt": args.steps - first,
                   "seconds_this_attempt": time.monotonic() - pair_started,
                   "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                   "hosc": {"rgb_mse": mse[0], "psnr_db": -10 * math.log10(max(mse[0], 1e-12))},
                   "siren": {"rgb_mse": mse[1], "psnr_db": -10 * math.log10(max(mse[1], 1e-12))}}
        archive_path = out / "fits" / (fit_id + ".zip")
        temporary = archive_path.with_suffix(".tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("metrics.json", json.dumps(metrics, sort_keys=True))
            archive.writestr("losses.json", json.dumps(losses))
            model_data = io.BytesIO()
            torch.save({k: v.detach().cpu() for k, v in pair.state_dict().items()}, model_data)
            archive.writestr("models.pt", model_data.getvalue())
            for i, activation in enumerate(("hosc", "siren")):
                png = io.BytesIO()
                pixels = (pred[i].cpu().numpy().reshape(height, image_width, 3).clip(0, 1) * 255).round().astype(np.uint8)
                Image.fromarray(pixels).save(png, format="PNG")
                archive.writestr(activation + ".png", png.getvalue())
        os.replace(temporary, archive_path)
        relative = archive_path.relative_to(out).as_posix()
        artifacts[relative] = {"bytes": archive_path.stat().st_size, "sha256": digest(archive_path)}
        state["completed"].append(metrics)
        state.update(pair_index=index + 1, step=0, global_step=(index + 1) * (args.steps + 1),
                     model=None, optimizer=None, sampling_rng=None, losses=[])
        save_checkpoint(out, state, artifacts)
        emit("pair_completed", **metrics, completed_fits=(index + 1) * 2, total_fits=len(pairs) * 2)
        if args.stop_after and state["global_step"] >= args.stop_after:
            events.close()
            return
    atomic(out / "summary.json", {"identity": identity, "completed_fits": len(state["completed"]) * 2,
        "total_fits": len(pairs) * 2, "restored_step": restored_step, "pairs": state["completed"],
        "elapsed_seconds": time.monotonic() - started, "artifacts": artifacts})
    atomic(out / "progress.json", {"completed_fits": len(pairs) * 2, "total_fits": len(pairs) * 2,
                                  "step": args.steps, "steps": args.steps, "state": "completed"})
    emit("completed_local", completed_fits=len(pairs) * 2)
    events.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True); p.add_argument("--output", default="outputs/grid")
    p.add_argument("--prepare-data", action="store_true", help="download native Kodak images and write their manifest, then exit")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--images", nargs="+")
    p.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--widths", nargs="+", type=int, default=[32, 64, 128, 256])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--pixels", type=int, default=8192)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--beta", type=float, default=1)
    p.add_argument("--omega", type=float, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--resume", help="resume from a previously written checkpoint JSON manifest")
    p.add_argument("--recovery-probe-wait", type=int, default=0, help="bounded first-checkpoint pause for deliberate runtime-loss testing")
    p.add_argument("--stop-after", type=int, help="local interruption-test hook; not a result endpoint")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.prepare_data:
        prepare_data(args.data)
    else:
        run(args)

