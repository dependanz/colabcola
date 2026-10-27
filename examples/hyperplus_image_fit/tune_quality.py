"""Bounded CPU HOSC search with validation-only selection and a sealed test stage.

The workload's heldout.npy is explicitly the VALIDATION split during search.
No test array is loaded until selection.json has been committed.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import train as workload
from prepare_data import MD5, URL, digest as archive_digest

SPLITS = {
    "train": ("data_batch_1.bin", 0, 1024),
    "validation": ("data_batch_1.bin", 1024, 256),
    "test": ("test_batch.bin", 32, 256),
}


def validate_splits(splits):
    for name, (member, start, count) in splits.items():
        if start < 0 or count < 1 or start + count > 10000:
            raise ValueError("invalid source range: " + name)
        for other, (m, s, n) in splits.items():
            if name != other and member == m and max(start, s) < min(start + count, s + n):
                raise ValueError("overlapping source records")


def prepare(archive, root):
    validate_splits(SPLITS)
    if archive_digest(archive, "md5") != MD5:
        raise ValueError("official CIFAR archive checksum mismatch")
    root.mkdir(parents=True, exist_ok=False)
    search = root / "search"
    search.mkdir()
    manifest = {"schema": 1, "archive_sha256": workload.digest(archive),
                "source_url": URL, "selection": "validation, never test, supplied as workload heldout", "files": {}}
    inventory = {}
    with tarfile.open(archive, "r:gz") as tar:
        for split, (member, start, count) in SPLITS.items():
            entry = tar.getmember("cifar-10-batches-bin/" + member)
            if not entry.isfile() or entry.size != 30_730_000:
                raise ValueError("unexpected archive member")
            with tar.extractfile(entry) as stream:
                stream.read(start * 3073)
                raw = stream.read(count * 3073)
            array = np.frombuffer(raw, dtype=np.uint8).reshape(count, 3073)[:, 1:].reshape(count, 3, 32, 32)
            path = root / "test.npy" if split == "test" else search / ("train.npy" if split == "train" else "heldout.npy")
            np.save(path, array, allow_pickle=False)
            record = {"sha256": workload.digest(path), "images": count, "source_member": member,
                      "start_record": start, "role": split}
            inventory[split] = {**record, "path": str(path)}
            if split != "test":
                manifest["files"][path.name] = record
    workload.atomic_json(search / "manifest.json", manifest)
    workload.atomic_json(root / "splits.json", {"archive_sha256": manifest["archive_sha256"], "files": inventory})
    return inventory


def select(rows, count=1):
    """Only validation MSE participates; deterministic run-name tie break."""
    if any(not math.isfinite(r["validation_mse"]) for r in rows):
        raise ValueError("nonfinite validation metric")
    return sorted(rows, key=lambda r: (r["validation_mse"], r["name"]))[:count]


def image_errors(pred, truth):
    mse = (pred - truth).square().mean((1, 2, 3))
    # First differences in RGB, equally weighting horizontal and vertical edges.
    edge = sum((p.diff(dim=d) - truth.diff(dim=d)).square().mean((1, 2, 3))
               for d, p in ((2, pred), (3, pred))) / 2
    return mse, edge


@torch.no_grad()
def assess(checkpoint, images):
    state = workload.load_checkpoint(checkpoint)
    cfg = state["identity"]["config"]
    net = workload.model(cfg["width"], cfg["latent"], activation=cfg["activation"],
                         hosc_beta=cfg["hosc_beta"], hosc_omega=cfg["hosc_omega"])
    net.load_state_dict(state["model"])
    net.eval()
    grid = workload.coordinates(*images.shape[-2:])
    predictions = []
    for batch in images.split(8):
        p = net(batch, grid.expand(len(batch), -1, -1))
        predictions.append(p.reshape(len(batch), *images.shape[-2:], 3).permute(0, 3, 1, 2))
    pred = torch.cat(predictions)
    mse, edge = image_errors(pred, images)
    mean = float(mse.double().mean())
    return {"mse": mean, "psnr_db": -10 * math.log10(max(mean, 1e-12)),
            "edge_mse": float(edge.double().mean()), "per_image_mse": mse.tolist(),
            "per_image_edge_mse": edge.tolist(), "images": len(images),
            "trainable_parameters": sum(p.numel() for p in net.parameters()),
            "generated_parameters": net.slot_counts[0], "config": cfg,
            "step": state["step"], "checkpoint_manifest": str(checkpoint),
            "checkpoint_manifest_sha256": workload.digest(checkpoint)}, pred


def paired_interval(candidate, baseline, key):
    delta = np.asarray(candidate[key]) - np.asarray(baseline[key])
    rng = np.random.default_rng(4567)
    means = delta[rng.integers(0, len(delta), (5000, len(delta)))].mean(1)
    return {"mean_candidate_minus_baseline": float(delta.mean()),
            "image_bootstrap_95_percent": np.quantile(means, [.025, .975]).tolist(),
            "resamples": 5000, "scope": "image variation only; one training seed"}


def run(args):
    out, data = Path(args.output), Path(args.data)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    candidates = [{"activation": "hosc", "hosc_beta": b, "hosc_omega": w,
                   "width": 64, "latent": 128}
                  for b in (.5, 1., 2.) for w in (1., 3., 6.)]
    protocol = {"schema": 1, "splits": SPLITS, "candidates": candidates,
                "screen_steps": 4000, "final_steps": 12000, "promote": 2,
                "controls": ["tanh width32 latent64", "tanh width64 latent128", "HOSC beta1 omega3 width32 latent64",
                             "tanh at winner size/steps", "HOSC at alternative size with winner beta/omega/steps"],
                "selection": "lowest validation MSE, name tie break; final pool all nine screen endpoints, two continued endpoints and small HOSC control",
                "fixed": {"seed": 1234, "lr": .001, "batch": 8, "pixels": 256, "threads": 2},
                "budget_seconds": 1800, "per_process_seconds": 360,
                "comparison": "equal optimizer steps and samples for controls, not equal FLOPs; screen endpoints also eligible",
                "test_rule": "commit selection.json before loading test.npy; no retraining after test metrics",
                "workload_sha256": workload.digest(Path(workload.__file__)), "tuner_sha256": workload.digest(__file__),
                "historical_checkpoint": str(Path(args.previous)), "historical_checkpoint_manifest_sha256": workload.digest(args.previous),
                "metrics": ["RGB MSE/PSNR", "horizontal/vertical first-difference MSE", "paired image bootstrap CI"],
                "limits": "32x32 CIFAR, one seed; no high-resolution, perceptual or paper-reproduction claim"}
    workload.atomic_json(out / "protocol.json", protocol)
    (out / "executed_tuner.py").write_bytes(Path(__file__).read_bytes())
    inventory = prepare(Path(args.archive), data)
    workload.atomic_json(out / "data.json", inventory)
    rows = []

    def execute(name, cfg, steps, resume=None):
        remaining = protocol["budget_seconds"] - (time.monotonic() - started)
        if remaining < 10:
            raise TimeoutError("local experiment budget exhausted")
        command = [sys.executable, str(Path(workload.__file__)), "--data", str(data / "search"),
                   "--output", str(out / name), "--steps", str(steps), "--checkpoint-every", str(steps)]
        for key, value in {**protocol["fixed"], **cfg}.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        if resume:
            command.extend(["--resume", resume])
        before = time.monotonic()
        with (out / (name + ".stdout.log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=min(360, remaining))
        metrics = json.loads((out / name / "metrics.json").read_text())
        row = {"name": name, "config": cfg, "steps": steps, "restored_step": metrics["restored_step"],
               "seconds": time.monotonic() - before, "validation_mse": metrics["final"]["heldout"]["mse"],
               "validation_psnr_db": metrics["final"]["heldout"]["psnr_db_from_mean_mse"],
               "train_mse": metrics["final"]["train"]["mse"],
               "checkpoint": str(out / name / f"state-{steps:08d}.pt.json")}
        rows.append(row)
        workload.atomic_json(out / "search.json", rows)
        print(json.dumps(row), flush=True)
        return row

    screen = [execute(f"screen-b{c['hosc_beta']:g}-w{c['hosc_omega']:g}", c, 4000) for c in candidates]
    promoted = select(screen, 2)
    workload.atomic_json(out / "promotion.json", promoted)
    finalists = [execute(r["name"].replace("screen", "final"), r["config"], 12000, r["checkpoint"]) for r in promoted]
    small = execute("small-hosc-b1-w3", {"activation": "hosc", "hosc_beta": 1., "hosc_omega": 3., "width": 32, "latent": 64}, 12000)
    controls = [execute("tanh-" + size, {"activation": "tanh", "hosc_beta": 1., "hosc_omega": 1., "width": width, "latent": latent}, 12000)
                for size, width, latent in (("small", 32, 64), ("large", 64, 128))]
    winner = select(screen + finalists + [small])[0]
    cfg = winner["config"]
    controls.extend([
        execute("matched-tanh", {**cfg, "activation": "tanh", "hosc_beta": 1., "hosc_omega": 1.}, winner["steps"]),
        execute("matched-capacity-hosc", {**cfg, "width": 32 if cfg["width"] == 64 else 64,
                                          "latent": 64 if cfg["width"] == 64 else 128}, winner["steps"]),
    ])
    selection = {"winner": winner, "selected_before_test": True, "search_sha256": workload.digest(out / "search.json"),
                 "protocol_sha256": workload.digest(out / "protocol.json")}
    workload.atomic_json(out / "selection.json", selection)
    print("Selection frozen; beginning fresh test evaluation.", flush=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    arrays, _ = workload.load_data(data / "search")
    # Reproduce every saved validation result and verify every training event.
    verification = []
    for r in rows:
        measured, _ = assess(r["checkpoint"], arrays["heldout"])
        if not math.isclose(measured["mse"], r["validation_mse"], abs_tol=1e-8, rel_tol=1e-6):
            raise AssertionError("reloaded validation mismatch")
        events = [json.loads(line) for line in (out / r["name"] / "events.ndjson").read_text().splitlines()]
        steps = [e for e in events if e["event"] == "train_step"]
        if [e["step"] for e in steps] != list(range(r["restored_step"] + 1, r["steps"] + 1)):
            raise AssertionError("nonconsecutive steps")
        if any(not math.isfinite(e["gradient_norm"]) or e["gradient_norm"] <= 0 for e in steps):
            raise AssertionError("invalid gradient")
        verification.append({"name": r["name"], "validation_reproduced": True, "verified_steps": len(steps)})
    if workload.digest(data / "test.npy") != inventory["test"]["sha256"]:
        raise ValueError("test data hash mismatch")
    test = torch.from_numpy(np.load(data / "test.npy", allow_pickle=False).copy()).float() / 255
    assessments, predictions = {}, {}
    for label, checkpoint in [("selected_hosc", winner["checkpoint"]), ("previous_hosc", args.previous),
                              ("small_hosc_12000", small["checkpoint"])] + [(r["name"], r["checkpoint"]) for r in controls]:
        assessments[label], predictions[label] = assess(checkpoint, test)
    comparisons = {label: {metric: paired_interval(assessments["selected_hosc"], a, metric)
                           for metric in ("per_image_mse", "per_image_edge_mse")}
                   for label, a in assessments.items() if label != "selected_hosc"}
    workload.atomic_json(out / "test-results.json", {"selection_sha256": workload.digest(out / "selection.json"),
                         "test_sha256": inventory["test"]["sha256"], "models": assessments, "paired_comparisons": comparisons})
    # First eight sequential test records, never chosen by appearance or model score.
    labels = ["Reference", "Previous HOSC", "Tanh (matched)", "Selected HOSC"]
    panels = [test, predictions["previous_hosc"], predictions["matched-tanh"], predictions["selected_hosc"]]
    canvas = Image.new("RGB", (4 * 160, 32 + 8 * 136), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (label, panel) in enumerate(zip(labels, panels)):
        draw.text((column * 160 + 4, 10), label, fill="black")
        for row in range(8):
            pixels = (panel[row].permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8)
            tile = Image.fromarray(pixels).resize((128, 128), Image.Resampling.NEAREST)
            canvas.paste(tile, (column * 160, 32 + row * 136))
    canvas.save(out / "test-comparison.png")
    workload.atomic_json(out / "verification.json", {"runs": verification, "elapsed_seconds": time.monotonic() - started,
                         "test_results_sha256": workload.digest(out / "test-results.json"),
                         "comparison_png_sha256": workload.digest(out / "test-comparison.png"),
                         "cloud_allocations": 0, "status": "passed"})
    print(json.dumps({k: {m: v[m] for m in ("mse", "psnr_db", "edge_mse", "trainable_parameters")} for k, v in assessments.items()}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True)
    p.add_argument("--data", required=True, help="new dataset directory")
    p.add_argument("--output", required=True, help="new experiment directory")
    p.add_argument("--previous", required=True, help="trusted prior HOSC checkpoint commit manifest")
    run(p.parse_args())
