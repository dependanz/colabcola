from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence


STATE_VERSION = 1
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT_RE = re.compile(r"(g|do)_([0-9]{8})")


class ResumeStateError(ValueError):
    """The checkpoint cannot prove an exact continuation boundary."""


@dataclass(frozen=True)
class ContinuationCursor:
    completed_steps: int
    epoch: int
    next_batch_in_epoch: int


def normalize_torch_rng_state_devices(
    torch_rng_state: object,
    torch_cuda_rng_state_all: object,
) -> tuple[object, list[object]]:
    """Return CPU-backed RNG tensors after a device-mapped checkpoint load."""

    if not isinstance(torch_cuda_rng_state_all, (list, tuple)):
        raise ResumeStateError("checkpoint torch_cuda_rng_state_all must be a sequence")

    def to_cpu(value: object, label: str) -> object:
        cpu = getattr(value, "cpu", None)
        if not callable(cpu):
            raise ResumeStateError(f"checkpoint {label} must be a Torch RNG tensor")
        return cpu()

    return (
        to_cpu(torch_rng_state, "torch_rng_state"),
        [
            to_cpu(value, f"torch_cuda_rng_state_all[{index}]")
            for index, value in enumerate(torch_cuda_rng_state_all)
        ],
    )


def ordered_file_identity(paths: Sequence[str | os.PathLike[str]]) -> str:
    """Hash the portable ordered sample identity without runtime-root paths."""

    names = tuple(Path(path).name for path in paths)
    if not names or any(not name or name in {".", ".."} for name in names):
        raise ResumeStateError("training file identity requires nonempty portable names")
    if len(names) != len(set(names)):
        raise ResumeStateError("training file identity contains duplicate basenames")
    content = "".join(f"{name}\n" for name in names).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def deterministic_crop_seed(
    *, seed: int, epoch: int, sample_position: int, sample_name: str
) -> int:
    """Derive a worker/order-independent crop seed for one training sample."""

    for label, value in (
        ("seed", seed),
        ("epoch", epoch),
        ("sample_position", sample_position),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ResumeStateError(f"{label} must be a nonnegative integer")
    if not sample_name or Path(sample_name).name != sample_name:
        raise ResumeStateError("sample_name must be one portable basename")
    material = f"{seed}\0{epoch}\0{sample_position}\0{sample_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def remaining_epoch_sample_keys(
    audio_files: Sequence[str | os.PathLike[str]],
    *,
    batch_size: int,
    epoch: int,
    start_batch: int,
    seed: int,
) -> Iterator[tuple[int, int]]:
    """Yield deterministic sample keys after an exact minibatch boundary."""

    for label, value, minimum in (
        ("batch_size", batch_size, 1),
        ("epoch", epoch, 0),
        ("start_batch", start_batch, 0),
        ("seed", seed, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            qualifier = "positive" if minimum == 1 else "nonnegative"
            raise ResumeStateError(f"{label} must be a {qualifier} integer")
    stop = (len(audio_files) // batch_size) * batch_size
    start = start_batch * batch_size
    if start > stop:
        raise ResumeStateError("resume batch cursor is outside the training epoch")
    for position in range(start, stop):
        name = Path(audio_files[position]).name
        yield (
            position,
            deterministic_crop_seed(
                seed=seed,
                epoch=epoch,
                sample_position=position,
                sample_name=name,
            ),
        )


def validate_continuation_state(
    state: Mapping[str, object],
    *,
    expected_training_files_sha256: str,
    expected_config_sha256: str,
    expected_batch_size: int,
    expected_batches_per_epoch: int,
    expected_seed: int,
) -> ContinuationCursor:
    """Validate the portable and positional portion of a versioned checkpoint."""

    state_version = state.get("colabcola_state_version")
    if (
        isinstance(state_version, bool)
        or not isinstance(state_version, int)
        or state_version != STATE_VERSION
    ):
        raise ResumeStateError(
            "checkpoint lacks the supported exact-continuation state version"
        )
    for label, expected in (
        ("training_files_sha256", expected_training_files_sha256),
        ("config_sha256", expected_config_sha256),
    ):
        if not _SHA256_RE.fullmatch(expected):
            raise ResumeStateError(f"expected {label} is not a lowercase SHA-256")
        if state.get(label) != expected:
            raise ResumeStateError(f"checkpoint {label} does not match this run")
    for label, expected in (
        ("batch_size", expected_batch_size),
        ("batches_per_epoch", expected_batches_per_epoch),
        ("seed", expected_seed),
    ):
        minimum = 0 if label == "seed" else 1
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < minimum:
            qualifier = "nonnegative" if label == "seed" else "positive"
            raise ResumeStateError(f"expected {label} must be a {qualifier} integer")
        actual = state.get(label)
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
            raise ResumeStateError(f"checkpoint {label} does not match this run")

    values: dict[str, int] = {}
    for label in ("completed_steps", "epoch", "next_batch_in_epoch"):
        value = state.get(label)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ResumeStateError(f"checkpoint {label} must be a nonnegative integer")
        values[label] = value
    if values["next_batch_in_epoch"] > expected_batches_per_epoch:
        raise ResumeStateError("checkpoint batch cursor exceeds the epoch")
    expected_steps = (
        values["epoch"] * expected_batches_per_epoch
        + values["next_batch_in_epoch"]
    )
    if values["completed_steps"] != expected_steps:
        raise ResumeStateError(
            "checkpoint completed_steps does not match its epoch/batch cursor"
        )
    for label in (
        "python_random_state",
        "torch_rng_state",
        "torch_cuda_rng_state_all",
        "scheduler_g",
        "scheduler_d",
    ):
        if label not in state:
            raise ResumeStateError(f"checkpoint is missing {label}")
    return ContinuationCursor(
        completed_steps=values["completed_steps"],
        epoch=values["epoch"],
        next_batch_in_epoch=values["next_batch_in_epoch"],
    )


def prune_checkpoint_pairs(checkpoint_root: Path | str, keep_pairs: int) -> tuple[int, ...]:
    """Retain only the newest complete checkpoint pairs beneath one output directory."""

    if isinstance(keep_pairs, bool) or not isinstance(keep_pairs, int) or keep_pairs <= 0:
        raise ResumeStateError("checkpoint retention must be a positive integer")
    root = Path(checkpoint_root)
    if not root.is_dir() or root.is_symlink():
        raise ResumeStateError("checkpoint root must be a real directory")

    by_kind: dict[str, dict[int, Path]] = {"g": {}, "do": {}}
    for item in root.iterdir():
        match = _CHECKPOINT_RE.fullmatch(item.name)
        if match is None:
            continue
        if item.is_symlink() or not item.is_file():
            raise ResumeStateError("checkpoint pair member must be a regular file")
        kind, step_text = match.groups()
        by_kind[kind][int(step_text)] = item

    generator_steps = set(by_kind["g"])
    state_steps = set(by_kind["do"])
    if generator_steps != state_steps:
        raise ResumeStateError("checkpoint directory contains an unpaired checkpoint")
    kept = tuple(sorted(generator_steps)[-keep_pairs:])
    keep_set = set(kept)
    for step in sorted(generator_steps - keep_set):
        by_kind["g"][step].unlink()
        by_kind["do"][step].unlink()
    return kept
