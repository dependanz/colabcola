"""CPU-only post-reconstruction measurement for the full LJ Speech transport probe."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import stat
import sys
import time
from pathlib import Path


_CLIP_ID = re.compile(r"LJ[0-9]{3}-[0-9]{4}")


def _split_ids(path: Path, label: str) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        clip_id = line.split("|", 1)[0]
        if not _CLIP_ID.fullmatch(clip_id):
            raise RuntimeError(f"{label}:{line_number} has an invalid clip ID")
        if clip_id in seen:
            raise RuntimeError(f"{label} repeats clip ID {clip_id}")
        seen.add(clip_id)
        values.append(clip_id)
    if not values:
        raise RuntimeError(f"{label} contains no clip IDs")
    return tuple(values)


def measure_dataset(
    dataset_root: Path,
    *,
    expected_training_count: int,
    expected_validation_count: int,
    expected_wav_bytes: int,
    expected_payload_bytes: int,
) -> dict[str, object]:
    started = time.monotonic()
    root = dataset_root.resolve(strict=True)
    training_path = root / "training.txt"
    validation_path = root / "validation.txt"
    wav_root = root / "wavs"
    training = _split_ids(training_path, "training split")
    validation = _split_ids(validation_path, "validation split")
    if len(training) != expected_training_count:
        raise RuntimeError("training clip count does not match the probe contract")
    if len(validation) != expected_validation_count:
        raise RuntimeError("validation clip count does not match the probe contract")
    if set(training) & set(validation):
        raise RuntimeError("training and validation splits overlap")

    observed: dict[str, int] = {}
    with os.scandir(wav_root) as entries:
        for entry in entries:
            metadata = entry.stat(follow_symlinks=False)
            if entry.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(f"WAV tree contains a non-regular entry: {entry.name}")
            path = Path(entry.name)
            if path.suffix != ".wav" or not _CLIP_ID.fullmatch(path.stem):
                raise RuntimeError(f"WAV tree contains an unexpected file: {entry.name}")
            observed[path.stem] = metadata.st_size

    expected_ids = set(training) | set(validation)
    if set(observed) != expected_ids:
        missing = len(expected_ids - set(observed))
        extra = len(set(observed) - expected_ids)
        raise RuntimeError(f"WAV identity mismatch: missing={missing}, extra={extra}")
    wav_bytes = sum(observed.values())
    payload_bytes = wav_bytes + training_path.stat().st_size + validation_path.stat().st_size
    if wav_bytes != expected_wav_bytes:
        raise RuntimeError("WAV byte count does not match the probe contract")
    if payload_bytes != expected_payload_bytes:
        raise RuntimeError("dataset payload byte count does not match the probe contract")

    return {
        "dataset_payload_bytes": payload_bytes,
        "free_disk_bytes_after_reconstruction": shutil.disk_usage(root).free,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "scan_seconds": time.monotonic() - started,
        "state": "transport_probe_complete",
        "training_clip_count": len(training),
        "validation_clip_count": len(validation),
        "wav_bytes": wav_bytes,
        "wav_file_count": len(observed),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected_training_count", required=True, type=int)
    parser.add_argument("--expected_validation_count", required=True, type=int)
    parser.add_argument("--expected_wav_bytes", required=True, type=int)
    parser.add_argument("--expected_payload_bytes", required=True, type=int)
    args = parser.parse_args()
    value = measure_dataset(
        args.dataset_root,
        expected_training_count=args.expected_training_count,
        expected_validation_count=args.expected_validation_count,
        expected_wav_bytes=args.expected_wav_bytes,
        expected_payload_bytes=args.expected_payload_bytes,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_bytes(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )
    os.replace(temporary, output)
    print(json.dumps(value, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
