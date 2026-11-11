"""Bounded 40-minute real-T4 lifecycle/control-plane acceptance workload.

Requires preflight-provided CUDA Torch in live mode. A --cpu-smoke run is
restricted to <=10 seconds and exists only for offline regression tests.
This program never requests Colab allocation or reads account credentials.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import os
from pathlib import Path
import time


def _utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".next")
    with staged.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staged, path)


def _gpu_pulse(device):
    torch, lhs, rhs, result = device
    until = time.perf_counter() + 0.15
    while time.perf_counter() < until:
        torch.mm(lhs, rhs, out=result)
        torch.cuda.synchronize()


def _prepare_gpu(cpu_smoke):
    if cpu_smoke:
        return None, "CPU-SMOKE"
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("T4 acceptance requires Torch in the Colab runtime") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("T4 acceptance requires an available CUDA device")
    name = torch.cuda.get_device_name(0)
    if "T4" not in name.upper():
        raise RuntimeError("T4 acceptance received a non-T4 CUDA device")
    lhs = torch.rand((256, 256), device="cuda")
    rhs = torch.rand((256, 256), device="cuda")
    result = torch.empty_like(lhs)
    _gpu_pulse((torch, lhs, rhs, result))
    return (torch, lhs, rhs, result), name


def _checkpoint(directory: Path, number: int, elapsed: float, pulses: int):
    directory.mkdir(parents=True, exist_ok=True)
    name = f"state-{number:04d}.bin"
    data = (json.dumps(
        {"schema": 1, "step": number, "elapsed_seconds": elapsed,
         "gpu_pulses": pulses}, sort_keys=True, separators=(",", ":")
    ) + "\n").encode("utf-8")
    _atomic(directory / name, data)
    manifest = {
        "schema": 1, "step": number, "file": name, "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    _atomic(directory / f"state-{number:04d}.json",
            (json.dumps(manifest, sort_keys=True) + "\n").encode("utf-8"))


def _load_resume(manifest_path, duration):
    manifest_path = Path(manifest_path)
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    name = record.get("file")
    if (not isinstance(name, str) or re.fullmatch(r"state-[0-9]{4}.bin", name) is None
            or record.get("schema") != 1 or type(record.get("step")) is not int
            or record["step"] <= 0 or type(record.get("bytes")) is not int):
        raise RuntimeError("invalid resume manifest")
    data = (manifest_path.parent / name).read_bytes()
    if (len(data) != record["bytes"]
            or hashlib.sha256(data).hexdigest() != record.get("sha256")):
        raise RuntimeError("resume payload hash/length mismatch")
    value = json.loads(data)
    elapsed = value.get("elapsed_seconds")
    pulses = value.get("gpu_pulses")
    if (value.get("schema") != 1 or value.get("step") != record["step"]
            or type(elapsed) not in (float, int) or not math.isfinite(elapsed)
            or not 0 <= elapsed < duration
            or type(pulses) is not int or pulses < 0):
        raise RuntimeError("resume payload contents invalid")
    return float(elapsed), pulses, record["step"]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--pulse-seconds", type=float, default=10.0)
    parser.add_argument("--progress-seconds", type=float, default=30.0)
    parser.add_argument("--checkpoint-seconds", type=float, default=300.0)
    parser.add_argument("--resume-manifest")
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args(argv)
    if (not 1 <= args.duration_seconds <= 3600
            or not 0 < args.pulse_seconds <= 60
            or not 0 < args.progress_seconds <= 120
            or not 0 < args.checkpoint_seconds <= 600):
        parser.error("invalid bounded duration or reporting intervals")
    if args.cpu_smoke and args.duration_seconds > 10:
        parser.error("CPU smoke is restricted to 10 seconds")
    if not args.cpu_smoke and args.duration_seconds < 2400:
        parser.error("live acceptance requires at least 2400 seconds")

    base_elapsed, prior_pulses, prior_checkpoints = (
        _load_resume(args.resume_manifest, args.duration_seconds)
        if args.resume_manifest else (0.0, 0, 0))
    output = Path("outputs")
    if output.exists():
        raise RuntimeError("refusing to reuse an existing output directory")
    output.mkdir()
    device, name = _prepare_gpu(args.cpu_smoke)

    start_wall = _utc()
    started = time.monotonic()
    pulses, reports, checkpoints = prior_pulses, 0, prior_checkpoints
    next_pulse = next_report = base_elapsed
    next_checkpoint = base_elapsed + args.checkpoint_seconds
    progress = output / "progress.ndjson"
    while True:
        elapsed = base_elapsed + time.monotonic() - started
        if elapsed >= args.duration_seconds:
            break
        if elapsed >= next_pulse:
            if device is not None:
                _gpu_pulse(device)
            pulses += 1
            next_pulse = elapsed + args.pulse_seconds
        if elapsed >= next_report:
            event = {"at": _utc(), "elapsed_seconds": elapsed,
                     "gpu_pulses": pulses, "checkpoint_count": checkpoints}
            with progress.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")
                stream.flush()
            reports += 1
            print(f"T4 probe elapsed={elapsed:.1f}s pulses={pulses} "
                  f"checkpoints={checkpoints}", flush=True)
            next_report = elapsed + args.progress_seconds
        if elapsed >= next_checkpoint:
            checkpoints += 1
            _checkpoint(output / "checkpoints", checkpoints, elapsed, pulses)
            next_checkpoint = elapsed + args.checkpoint_seconds
        time.sleep(0.20)

    segment_seconds = time.monotonic() - started
    elapsed = base_elapsed + segment_seconds
    if elapsed < args.duration_seconds:
        raise RuntimeError("acceptance clock ended early")
    # A final immutable state snapshot is useful even when the exact interval
    # did not coincide with the deadline.
    checkpoints += 1
    _checkpoint(output / "checkpoints", checkpoints, elapsed, pulses)
    summary = {
        "schema": 1, "probe": "colabcola-t4-lifecycle-40min",
        "mode": ("cpu-smoke" if args.cpu_smoke else
                 "t4-recovery" if args.resume_manifest else "t4-live"),
        "requested_seconds": args.duration_seconds,
        "elapsed_seconds": elapsed,
        "segment_elapsed_seconds": segment_seconds,
        "resumed_from_step": prior_checkpoints if args.resume_manifest else None,
        "started_at": start_wall, "finished_at": _utc(),
        "cuda_device": name, "gpu_pulses": pulses,
        "progress_observations": reports, "checkpoint_count": checkpoints,
        "state": "success",
    }
    _atomic(output / "acceptance.json",
            (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print("T4 acceptance finished " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
