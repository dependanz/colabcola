"""Deterministic LJ Speech subset construction for bounded HiFi-GAN jobs."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import stat
import tarfile
import uuid
import wave
import zipfile
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .bundle import ZIP_COMPRESSION_LEVEL, ZIP_FIXED_DATETIME, canonical_json_bytes
from .errors import SpecError
from .hifigan import HiFiGanProfile, load_hifigan_profile
from .spec import PayloadEntry

LJSPEECH_OFFICIAL_PAGE = "https://keithito.com/LJ-Speech-Dataset/"
LJSPEECH_ARCHIVE_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
LJSPEECH_ARCHIVE_NAME = "LJSpeech-1.1.tar.bz2"
LJSPEECH_DATASET_DIRECTORY = "LJSpeech-1.1"
LJSPEECH_MANIFEST_NAME = "LJSPEECH_SUBSET_MANIFEST.json"
LJSPEECH_TRAINING_COUNT = 64
LJSPEECH_VALIDATION_COUNT = 8
LJSPEECH_PILOT_TRAINING_COUNT = 512
LJSPEECH_PILOT_VALIDATION_COUNT = 32
LJSPEECH_SAMPLE_RATE = 22_050
LJSPEECH_CHANNELS = 1
LJSPEECH_SAMPLE_WIDTH_BYTES = 2

_COPY_BUFFER_BYTES = 1024 * 1024
_REPARSE_FLAG = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


@dataclass(frozen=True, slots=True)
class LJSpeechSelection:
    """The exact split-local clip IDs selected for one bounded subset."""

    training_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]

    @property
    def all_ids(self) -> tuple[str, ...]:
        return self.training_ids + self.validation_ids


@dataclass(frozen=True, slots=True)
class LJSpeechSubset:
    """A verified on-disk subset and its canonical manifest."""

    root: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    selection: LJSpeechSelection


def select_ljspeech_ids(
    training_split: Path | str,
    validation_split: Path | str,
    *,
    training_count: int = LJSPEECH_TRAINING_COUNT,
    validation_count: int = LJSPEECH_VALIDATION_COUNT,
) -> LJSpeechSelection:
    """Select the lexicographically first unique IDs from disjoint split files."""

    for label, value in (
        ("training_count", training_count),
        ("validation_count", validation_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SpecError(f"LJ Speech {label} must be a positive integer")
    training = _read_split_ids(Path(training_split), "training split")
    validation = _read_split_ids(Path(validation_split), "validation split")
    overlap = sorted(set(training) & set(validation))
    if overlap:
        raise SpecError(
            "HiFi-GAN training and validation split authorities overlap: "
            + ", ".join(overlap[:10])
        )
    if len(training) < training_count:
        raise SpecError(
            f"training split has {len(training)} unique IDs; {training_count} are required"
        )
    if len(validation) < validation_count:
        raise SpecError(
            f"validation split has {len(validation)} unique IDs; {validation_count} are required"
        )
    return LJSpeechSelection(
        training_ids=tuple(sorted(training)[:training_count]),
        validation_ids=tuple(sorted(validation)[:validation_count]),
    )


def build_ljspeech_subset(
    archive_path: Path | str,
    training_split: Path | str,
    validation_split: Path | str,
    output_root: Path | str,
    *,
    profile: HiFiGanProfile | None = None,
    training_count: int = LJSPEECH_TRAINING_COUNT,
    validation_count: int = LJSPEECH_VALIDATION_COUNT,
) -> LJSpeechSubset:
    """Construct and atomically publish a measured subset from the official archive."""

    selected_profile = profile or load_hifigan_profile()
    archive = _regular_file(Path(archive_path), "LJ Speech archive")
    training_source = _regular_file(Path(training_split), "HiFi-GAN training split")
    validation_source = _regular_file(Path(validation_split), "HiFi-GAN validation split")
    _verify_profile_split(
        selected_profile,
        "LJSpeech-1.1/training.txt",
        training_source,
    )
    _verify_profile_split(
        selected_profile,
        "LJSpeech-1.1/validation.txt",
        validation_source,
    )
    selection = select_ljspeech_ids(
        training_source,
        validation_source,
        training_count=training_count,
        validation_count=validation_count,
    )

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise SpecError(f"LJ Speech subset destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(
        f".{destination.name}.staging-{uuid.uuid4().hex}"
    )
    staging.mkdir()
    try:
        dataset_root = staging / LJSPEECH_DATASET_DIRECTORY
        wav_root = dataset_root / "wavs"
        wav_root.mkdir(parents=True)
        extracted = _extract_selected_wavs(archive, wav_root, selection)

        training_bytes = _split_bytes(selection.training_ids)
        validation_bytes = _split_bytes(selection.validation_ids)
        training_output = dataset_root / "training.txt"
        validation_output = dataset_root / "validation.txt"
        training_output.write_bytes(training_bytes)
        validation_output.write_bytes(validation_bytes)

        clip_records = tuple(
            _wave_record(
                wav_root / f"{clip_id}.wav",
                clip_id,
                "training" if clip_id in set(selection.training_ids) else "validation",
                extracted[clip_id],
            )
            for clip_id in selection.all_ids
        )
        payload_paths = tuple(
            [record["path"] for record in clip_records]
            + [
                f"{LJSPEECH_DATASET_DIRECTORY}/training.txt",
                f"{LJSPEECH_DATASET_DIRECTORY}/validation.txt",
            ]
        )
        transport = _measure_transport(staging, payload_paths)
        total_frames = sum(int(record["frames"]) for record in clip_records)
        total_wav_bytes = sum(int(record["file_bytes"]) for record in clip_records)
        total_audio_bytes = sum(int(record["audio_data_bytes"]) for record in clip_records)
        manifest: dict[str, object] = {
            "aggregates": {
                "audio_data_bytes": total_audio_bytes,
                "clip_count": len(clip_records),
                "frames": total_frames,
                "payload_file_count": len(payload_paths),
                "payload_raw_bytes": transport["raw_bytes"],
                "total_duration_seconds": _duration_seconds(total_frames),
                "training_clip_count": len(selection.training_ids),
                "transport_archive_bytes": transport["archive_bytes"],
                "transport_archive_sha256": transport["archive_sha256"],
                "transport_base64_bytes": transport["base64_bytes"],
                "transport_scope": (
                    "selected WAV files plus generated split lists; "
                    "excludes this manifest"
                ),
                "validation_clip_count": len(selection.validation_ids),
                "wav_file_bytes": total_wav_bytes,
            },
            "audio_contract": {
                "channels": LJSPEECH_CHANNELS,
                "compression": "NONE",
                "sample_rate_hz": LJSPEECH_SAMPLE_RATE,
                "sample_width_bytes": LJSPEECH_SAMPLE_WIDTH_BYTES,
            },
            "clips": list(clip_records),
            "dataset": {
                "archive_bytes": archive.stat().st_size,
                "archive_filename": LJSPEECH_ARCHIVE_NAME,
                "archive_sha256": _sha256_file(archive),
                "archive_url": LJSPEECH_ARCHIVE_URL,
                "license": "public domain",
                "official_page": LJSPEECH_OFFICIAL_PAGE,
                "release": "1.1",
            },
            "schema_version": 1,
            "selection": {
                "rule": "lexicographically first unique IDs within each pinned split",
                "training_ids": list(selection.training_ids),
                "training_split_sha256": _sha256_file(training_source),
                "validation_ids": list(selection.validation_ids),
                "validation_split_sha256": _sha256_file(validation_source),
            },
            "upstream": {
                "commit": selected_profile.commit,
                "repository": selected_profile.repository,
            },
        }
        manifest_path = staging / LJSPEECH_MANIFEST_NAME
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        verified = _verify_ljspeech_subset(staging, selected_profile)
        os.replace(staging, destination)
        return LJSpeechSubset(
            root=destination,
            manifest_path=destination / LJSPEECH_MANIFEST_NAME,
            manifest=verified.manifest,
            selection=verified.selection,
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def verify_ljspeech_subset(
    root: Path | str, profile: HiFiGanProfile | None = None
) -> LJSpeechSubset:
    """Recompute every selected-file, audio, split, aggregate, and transport fact."""

    return _verify_ljspeech_subset(
        Path(root).resolve(), profile or load_hifigan_profile()
    )


def ljspeech_subset_payload(
    root: Path | str, profile: HiFiGanProfile | None = None
) -> tuple[PayloadEntry, ...]:
    """Return verified data/config/provenance entries for the HiFi-GAN job bundle."""

    subset = verify_ljspeech_subset(root, profile)
    dataset_root = subset.root / LJSPEECH_DATASET_DIRECTORY
    return (
        PayloadEntry(
            role="license",
            source=subset.manifest_path,
            target=f"workload/hifi_gan/{LJSPEECH_MANIFEST_NAME}",
        ),
        PayloadEntry(
            role="config",
            source=dataset_root / "training.txt",
            target="workload/hifi_gan/LJSpeech-1.1/training.txt",
        ),
        PayloadEntry(
            role="config",
            source=dataset_root / "validation.txt",
            target="workload/hifi_gan/LJSpeech-1.1/validation.txt",
        ),
        PayloadEntry(
            role="data",
            source=dataset_root / "wavs",
            target="workload/hifi_gan/LJSpeech-1.1/wavs",
        ),
    )


def _verify_ljspeech_subset(root: Path, profile: HiFiGanProfile) -> LJSpeechSubset:
    manifest_path = _regular_file(root / LJSPEECH_MANIFEST_NAME, "subset manifest")
    try:
        manifest_bytes = manifest_path.read_bytes()
        value = json.loads(
            manifest_bytes.decode("utf-8"), object_pairs_hook=_unique_object
        )
    except UnicodeDecodeError as exc:
        raise SpecError("LJ Speech subset manifest must be UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(f"invalid LJ Speech subset manifest JSON: {exc.msg}") from exc
    manifest = _mapping(value, "LJ Speech subset manifest")
    if canonical_json_bytes(manifest) != manifest_bytes:
        raise SpecError("LJ Speech subset manifest is not canonical JSON")
    _exact_keys(
        manifest,
        {
            "aggregates",
            "audio_contract",
            "clips",
            "dataset",
            "schema_version",
            "selection",
            "upstream",
        },
        "LJ Speech subset manifest",
    )
    if manifest["schema_version"] != 1:
        raise SpecError("LJ Speech subset manifest schema_version must equal 1")

    upstream = _mapping(manifest["upstream"], "manifest.upstream")
    if upstream.get("commit") != profile.commit or upstream.get("repository") != profile.repository:
        raise SpecError("LJ Speech subset manifest upstream identity does not match the profile")
    dataset = _mapping(manifest["dataset"], "manifest.dataset")
    expected_dataset = {
        "archive_filename": LJSPEECH_ARCHIVE_NAME,
        "archive_url": LJSPEECH_ARCHIVE_URL,
        "license": "public domain",
        "official_page": LJSPEECH_OFFICIAL_PAGE,
        "release": "1.1",
    }
    for key, expected in expected_dataset.items():
        if dataset.get(key) != expected:
            raise SpecError(f"manifest.dataset.{key} does not match the LJ Speech profile")
    _sha256_text(dataset.get("archive_sha256"), "manifest.dataset.archive_sha256")
    _nonnegative_integer(dataset.get("archive_bytes"), "manifest.dataset.archive_bytes")

    selection_value = _mapping(manifest["selection"], "manifest.selection")
    training_ids = _manifest_ids(selection_value.get("training_ids"), "training_ids")
    validation_ids = _manifest_ids(selection_value.get("validation_ids"), "validation_ids")
    selection = LJSpeechSelection(training_ids, validation_ids)
    if set(training_ids) & set(validation_ids):
        raise SpecError("subset manifest training and validation IDs must be disjoint")
    training_is_sorted = tuple(sorted(training_ids)) == training_ids
    validation_is_sorted = tuple(sorted(validation_ids)) == validation_ids
    if not training_is_sorted or not validation_is_sorted:
        raise SpecError("subset manifest IDs must be lexicographically ordered")
    if selection_value.get("rule") != (
        "lexicographically first unique IDs within each pinned split"
    ):
        raise SpecError("subset manifest selection rule is invalid")
    expected_training = _profile_source(profile, "LJSpeech-1.1/training.txt")
    expected_validation = _profile_source(profile, "LJSpeech-1.1/validation.txt")
    if selection_value.get("training_split_sha256") != expected_training.sha256:
        raise SpecError("subset manifest training split SHA-256 is not pinned")
    if selection_value.get("validation_split_sha256") != expected_validation.sha256:
        raise SpecError("subset manifest validation split SHA-256 is not pinned")

    _verify_subset_tree(root, selection)

    audio_contract = _mapping(manifest["audio_contract"], "manifest.audio_contract")
    if audio_contract != {
        "channels": LJSPEECH_CHANNELS,
        "compression": "NONE",
        "sample_rate_hz": LJSPEECH_SAMPLE_RATE,
        "sample_width_bytes": LJSPEECH_SAMPLE_WIDTH_BYTES,
    }:
        raise SpecError("subset manifest audio contract is invalid")

    clips_value = _array(manifest["clips"], "manifest.clips")
    expected_ids = selection.all_ids
    if len(clips_value) != len(expected_ids):
        raise SpecError("subset manifest clip count does not match its selected IDs")
    records: list[Mapping[str, Any]] = []
    for index, (value, expected_id) in enumerate(zip(clips_value, expected_ids, strict=True)):
        record = _mapping(value, f"manifest.clips[{index}]")
        expected_split = "training" if index < len(training_ids) else "validation"
        expected_path = f"{LJSPEECH_DATASET_DIRECTORY}/wavs/{expected_id}.wav"
        if record.get("id") != expected_id or record.get("split") != expected_split:
            raise SpecError(f"manifest.clips[{index}] ID or split is invalid")
        if record.get("path") != expected_path:
            raise SpecError(f"manifest.clips[{index}] path is invalid")
        expected_record = _wave_record(
            root.joinpath(*PurePosixPath(expected_path).parts),
            expected_id,
            expected_split,
            {
                "file_bytes": record.get("file_bytes"),
                "sha256": record.get("sha256"),
            },
        )
        if record != expected_record:
            raise SpecError(f"manifest.clips[{index}] does not match the WAV bytes")
        records.append(record)

    dataset_root = root / LJSPEECH_DATASET_DIRECTORY
    training_bytes = _split_bytes(training_ids)
    validation_bytes = _split_bytes(validation_ids)
    training_path = _regular_file(dataset_root / "training.txt", "generated training split")
    validation_path = _regular_file(dataset_root / "validation.txt", "generated validation split")
    if training_path.read_bytes() != training_bytes:
        raise SpecError("generated training split does not match selected IDs")
    if validation_path.read_bytes() != validation_bytes:
        raise SpecError("generated validation split does not match selected IDs")

    payload_paths = tuple(
        [str(record["path"]) for record in records]
        + [
            f"{LJSPEECH_DATASET_DIRECTORY}/training.txt",
            f"{LJSPEECH_DATASET_DIRECTORY}/validation.txt",
        ]
    )
    transport = _measure_transport(root, payload_paths)
    total_frames = sum(int(record["frames"]) for record in records)
    total_wav_bytes = sum(int(record["file_bytes"]) for record in records)
    total_audio_bytes = sum(int(record["audio_data_bytes"]) for record in records)
    expected_aggregates = {
        "audio_data_bytes": total_audio_bytes,
        "clip_count": len(records),
        "frames": total_frames,
        "payload_file_count": len(payload_paths),
        "payload_raw_bytes": transport["raw_bytes"],
        "total_duration_seconds": _duration_seconds(total_frames),
        "training_clip_count": len(training_ids),
        "transport_archive_bytes": transport["archive_bytes"],
        "transport_archive_sha256": transport["archive_sha256"],
        "transport_base64_bytes": transport["base64_bytes"],
        "transport_scope": (
            "selected WAV files plus generated split lists; excludes this manifest"
        ),
        "validation_clip_count": len(validation_ids),
        "wav_file_bytes": total_wav_bytes,
    }
    if manifest["aggregates"] != expected_aggregates:
        raise SpecError("subset manifest aggregate or transport facts do not match its files")
    return LJSpeechSubset(root, manifest_path, manifest, selection)


def _extract_selected_wavs(
    archive: Path,
    wav_root: Path,
    selection: LJSpeechSelection,
) -> dict[str, dict[str, object]]:
    desired = {
        f"{LJSPEECH_DATASET_DIRECTORY}/wavs/{clip_id}.wav": clip_id
        for clip_id in selection.all_ids
    }
    extracted: dict[str, dict[str, object]] = {}
    try:
        with tarfile.open(archive, mode="r:bz2") as source:
            for member in source:
                member_name = (
                    member.name[:-1]
                    if member.isdir() and member.name.endswith("/")
                    else member.name
                )
                normalized = _portable_member_name(member_name)
                clip_id = desired.get(normalized)
                if clip_id is None:
                    continue
                if clip_id in extracted:
                    raise SpecError(f"LJ Speech archive repeats selected clip {clip_id}")
                if not member.isfile():
                    raise SpecError(f"selected archive member is not a regular file: {normalized}")
                stream = source.extractfile(member)
                if stream is None:
                    raise SpecError(f"selected archive member cannot be read: {normalized}")
                target = wav_root / f"{clip_id}.wav"
                digest = hashlib.sha256()
                size = 0
                with stream, target.open("xb") as output:
                    while chunk := stream.read(_COPY_BUFFER_BYTES):
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                if size != member.size:
                    raise SpecError(
                        f"selected archive member size changed: {normalized}: "
                        f"expected {member.size}, observed {size}"
                    )
                extracted[clip_id] = {"file_bytes": size, "sha256": digest.hexdigest()}
    except SpecError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise SpecError(f"LJ Speech archive cannot be read: {archive}: {exc}") from exc
    missing = sorted(set(selection.all_ids) - set(extracted))
    if missing:
        raise SpecError("LJ Speech archive is missing selected clips: " + ", ".join(missing))
    return extracted


def _verify_subset_tree(root: Path, selection: LJSpeechSelection) -> None:
    expected_directories = {
        LJSPEECH_DATASET_DIRECTORY,
        f"{LJSPEECH_DATASET_DIRECTORY}/wavs",
    }
    expected_files = {
        LJSPEECH_MANIFEST_NAME,
        f"{LJSPEECH_DATASET_DIRECTORY}/training.txt",
        f"{LJSPEECH_DATASET_DIRECTORY}/validation.txt",
        *(
            f"{LJSPEECH_DATASET_DIRECTORY}/wavs/{clip_id}.wav"
            for clip_id in selection.all_ids
        ),
    }
    observed_files: set[str] = set()
    try:
        for current, directories, files in os.walk(root, topdown=True, followlinks=False):
            current_path = Path(current)
            for name in directories:
                path = current_path / name
                relative = path.relative_to(root).as_posix()
                metadata = os.lstat(path)
                attributes = getattr(metadata, "st_file_attributes", 0)
                if (
                    stat.S_ISLNK(metadata.st_mode)
                    or attributes & _REPARSE_FLAG
                    or not stat.S_ISDIR(metadata.st_mode)
                ):
                    raise SpecError(f"subset contains an unsafe directory: {relative}")
                if relative not in expected_directories:
                    raise SpecError(f"subset contains an undeclared directory: {relative}")
            for name in files:
                path = current_path / name
                relative = path.relative_to(root).as_posix()
                _regular_file(path, f"subset file {relative}")
                if relative not in expected_files:
                    raise SpecError(f"subset contains an undeclared file: {relative}")
                observed_files.add(relative)
    except SpecError:
        raise
    except OSError as exc:
        raise SpecError(f"LJ Speech subset tree cannot be inspected: {root}: {exc}") from exc
    missing = sorted(expected_files - observed_files)
    if missing:
        raise SpecError("LJ Speech subset is missing declared files: " + ", ".join(missing))


def _wave_record(
    path: Path,
    clip_id: str,
    split: str,
    extracted: Mapping[str, object],
) -> dict[str, object]:
    source = _regular_file(path, f"selected WAV {clip_id}")
    file_bytes = source.stat().st_size
    sha256 = _sha256_file(source)
    if extracted.get("file_bytes") != file_bytes or extracted.get("sha256") != sha256:
        raise SpecError(f"selected WAV {clip_id} changed after extraction")
    try:
        with wave.open(os.fspath(source), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
            compression = wav.getcomptype()
            audio = wav.readframes(frames)
            trailing = wav.readframes(1)
    except (EOFError, OSError, wave.Error) as exc:
        raise SpecError(f"selected WAV {clip_id} is invalid: {exc}") from exc
    if (
        channels != LJSPEECH_CHANNELS
        or sample_width != LJSPEECH_SAMPLE_WIDTH_BYTES
        or sample_rate != LJSPEECH_SAMPLE_RATE
        or compression != "NONE"
        or frames <= 0
    ):
        raise SpecError(
            f"selected WAV {clip_id} must be mono 16-bit PCM at 22050 Hz; "
            f"observed channels={channels}, width={sample_width}, "
            f"rate={sample_rate}, compression={compression}, frames={frames}"
        )
    expected_audio_bytes = frames * channels * sample_width
    if len(audio) != expected_audio_bytes or trailing:
        raise SpecError(f"selected WAV {clip_id} frame data is truncated or inconsistent")
    return {
        "audio_data_bytes": expected_audio_bytes,
        "channels": channels,
        "duration_seconds": _duration_seconds(frames),
        "file_bytes": file_bytes,
        "frames": frames,
        "id": clip_id,
        "path": f"{LJSPEECH_DATASET_DIRECTORY}/wavs/{clip_id}.wav",
        "sample_rate_hz": sample_rate,
        "sample_width_bytes": sample_width,
        "sha256": sha256,
        "split": split,
    }


def _measure_transport(root: Path, paths: Sequence[str]) -> dict[str, object]:
    archive_path = root / f".transport-{uuid.uuid4().hex}.zip"
    raw_bytes = 0
    try:
        with zipfile.ZipFile(
            archive_path,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=ZIP_COMPRESSION_LEVEL,
        ) as archive:
            archive.comment = b""
            for relative in sorted(paths):
                source = _regular_file(
                    root.joinpath(*PurePosixPath(relative).parts),
                    f"transport payload {relative}",
                )
                info = _zip_info(relative)
                with source.open("rb") as input_stream, archive.open(info, "w") as output:
                    while chunk := input_stream.read(_COPY_BUFFER_BYTES):
                        output.write(chunk)
                        raw_bytes += len(chunk)
        archive_bytes = archive_path.stat().st_size
        archive_sha256 = _sha256_file(archive_path)
        return {
            "archive_bytes": archive_bytes,
            "archive_sha256": archive_sha256,
            "base64_bytes": len(base64.b64encode(b"\0" * (archive_bytes % 3)))
            + 4 * (archive_bytes // 3),
            "raw_bytes": raw_bytes,
        }
    finally:
        archive_path.unlink(missing_ok=True)


def _zip_info(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, ZIP_FIXED_DATETIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.create_version = 20
    info.extract_version = 20
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.internal_attr = 0
    info.extra = b""
    info.comment = b""
    return info


def _read_split_ids(path: Path, label: str) -> tuple[str, ...]:
    source = _regular_file(path, label)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise SpecError(f"{label} must be UTF-8: {source}") from exc
    ids: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, 1):
        if not line:
            continue
        clip_id = line.split("|", 1)[0]
        _validate_clip_id(clip_id, f"{label}:{line_number}")
        if clip_id in seen:
            raise SpecError(f"{label} repeats clip ID {clip_id}")
        seen.add(clip_id)
        ids.append(clip_id)
    if not ids:
        raise SpecError(f"{label} contains no clip IDs")
    return tuple(ids)


def _manifest_ids(value: Any, label: str) -> tuple[str, ...]:
    ids = tuple(
        _validate_clip_id(item, f"manifest.selection.{label}[{index}]")
        for index, item in enumerate(_array(value, f"manifest.selection.{label}"))
    )
    if not ids:
        raise SpecError(f"manifest.selection.{label} must not be empty")
    if len(ids) != len(set(ids)):
        raise SpecError(f"manifest.selection.{label} contains duplicate IDs")
    return ids


def _validate_clip_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 10:
        raise SpecError(f"{where} must be an LJ Speech clip ID")
    if (
        not value.startswith("LJ")
        or value[5] != "-"
        or not value[2:5].isdigit()
        or not value[6:].isdigit()
    ):
        raise SpecError(f"{where} must match LJ999-9999")
    return value


def _verify_profile_split(profile: HiFiGanProfile, relative: str, path: Path) -> None:
    expected = _profile_source(profile, relative)
    if path.stat().st_size != expected.size:
        raise SpecError(
            f"{relative} size mismatch: expected {expected.size}, observed {path.stat().st_size}"
        )
    digest = _sha256_file(path)
    if digest != expected.sha256:
        raise SpecError(
            f"{relative} SHA-256 mismatch: expected {expected.sha256}, observed {digest}"
        )


def _profile_source(profile: HiFiGanProfile, relative: str):
    matches = [record for record in profile.source_files if record.path == relative]
    if len(matches) != 1:
        raise SpecError(f"HiFi-GAN profile must declare exactly one {relative}")
    return matches[0]


def _portable_member_name(value: str) -> str:
    path = PurePosixPath(value)
    normalized = path.as_posix()
    if (
        value.startswith("/")
        or "\\" in value
        or "\x00" in value
        or normalized != value
        or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
    ):
        raise SpecError(f"LJ Speech archive contains an unsafe path: {value!r}")
    return normalized


def _split_bytes(ids: Iterable[str]) -> bytes:
    return ("".join(f"{clip_id}\n" for clip_id in ids)).encode("utf-8")


def _duration_seconds(frames: int) -> str:
    seconds = Decimal(frames) / Decimal(LJSPEECH_SAMPLE_RATE)
    return format(seconds.quantize(Decimal("0.000000001"), rounding=ROUND_HALF_UP), "f")


def _regular_file(path: Path, where: str) -> Path:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise SpecError(f"{where} is unavailable: {path}: {exc}") from exc
    attributes = getattr(metadata, "st_file_attributes", 0)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or attributes & _REPARSE_FLAG
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise SpecError(f"{where} must be a regular non-link file: {path}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(_COPY_BUFFER_BYTES):
                digest.update(chunk)
    except OSError as exc:
        raise SpecError(f"file cannot be hashed: {path}: {exc}") from exc
    return digest.hexdigest()


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SpecError(f"duplicate LJ Speech subset manifest key: {key!r}")
        result[key] = value
    return result


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise SpecError(f"{where} must be an object with string keys")
    return value


def _array(value: Any, where: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise SpecError(f"{where} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise SpecError(f"{where} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise SpecError(f"{where} has unknown fields: {', '.join(unknown)}")


def _sha256_text(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SpecError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_integer(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SpecError(f"{where} must be a nonnegative integer")
    return value
