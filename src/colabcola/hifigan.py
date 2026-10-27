"""Pinned HiFi-GAN workload profile and source-provenance verification."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .errors import SpecError
from .spec import PayloadEntry

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ALLOWED_ROLES = frozenset({"source", "config", "license"})
_REPARSE_FLAG = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


@dataclass(frozen=True, slots=True)
class HiFiGanSourceFile:
    """One byte-exact file at the pinned upstream revision."""

    path: str
    size: int
    sha256: str
    role: str
    bundle_target: str | None


@dataclass(frozen=True, slots=True)
class HiFiGanOverlay:
    """One local compatibility/training adapter whose bytes are profile-locked."""

    path: str
    size: int
    sha256: str
    role: str
    bundle_target: str


@dataclass(frozen=True, slots=True)
class HiFiGanRuntimeLock:
    """Exact package facts expected from one selectable Colab runtime image."""

    provider: str
    runtime_version: str
    backend_info_commit: str
    gpu_package_snapshot_url: str
    requires_python: str
    runtime_requirements: tuple[str, ...]
    install_requirements: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HiFiGanProfile:
    """Validated source, overlay, runtime, and smoke-budget profile."""

    profile_path: Path
    repository: str
    commit: str
    license_spdx: str
    license_notice: str
    source_files: tuple[HiFiGanSourceFile, ...]
    overlays: tuple[HiFiGanOverlay, ...]
    runtime: HiFiGanRuntimeLock
    initial_completed_steps: int
    resumed_completed_steps: int
    checkpoint_interval: int


def default_hifigan_profile_path() -> Path:
    """Return the checked-in profile shipped beside the HiFi-GAN overlay."""

    return Path(__file__).with_name("_hifigan") / "profile.json"


def load_hifigan_profile(path: Path | str | None = None) -> HiFiGanProfile:
    """Load and strictly validate a HiFi-GAN profile without touching its checkout."""

    source = Path(path) if path is not None else default_hifigan_profile_path()
    try:
        value = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except UnicodeDecodeError as exc:
        raise SpecError(f"{source}: HiFi-GAN profile must be UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"{source}:{exc.lineno}:{exc.colno}: invalid HiFi-GAN profile JSON: {exc.msg}"
        ) from exc

    root = _mapping(value, "HiFi-GAN profile")
    _exact_keys(
        root,
        {"schema_version", "workload", "upstream", "overlays", "runtime", "smoke"},
        "HiFi-GAN profile",
    )
    if _integer(root["schema_version"], "HiFi-GAN profile.schema_version") != 1:
        raise SpecError("HiFi-GAN profile.schema_version must equal 1")
    if _string(root["workload"], "HiFi-GAN profile.workload") != "hifi-gan-v1":
        raise SpecError("HiFi-GAN profile.workload must equal 'hifi-gan-v1'")

    upstream = _mapping(root["upstream"], "HiFi-GAN profile.upstream")
    _exact_keys(
        upstream,
        {"repository", "commit", "license_spdx", "license_notice", "files"},
        "HiFi-GAN profile.upstream",
    )
    repository = _string(upstream["repository"], "HiFi-GAN profile.upstream.repository")
    commit = _string(upstream["commit"], "HiFi-GAN profile.upstream.commit")
    if not repository.startswith("https://") or not repository.endswith(".git"):
        raise SpecError("HiFi-GAN profile upstream repository must be an HTTPS Git URL")
    if not _COMMIT_RE.fullmatch(commit):
        raise SpecError("HiFi-GAN profile upstream commit must be a lowercase 40-hex revision")

    source_files = tuple(
        _parse_source_file(item, index)
        for index, item in enumerate(_array(upstream["files"], "HiFi-GAN profile.upstream.files"))
    )
    if not source_files:
        raise SpecError("HiFi-GAN profile must declare upstream files")
    _reject_duplicates((item.path for item in source_files), "upstream file path")
    _reject_duplicates(
        (item.bundle_target for item in source_files if item.bundle_target is not None),
        "upstream bundle target",
    )

    overlays = tuple(
        _parse_overlay(item, index)
        for index, item in enumerate(_array(root["overlays"], "HiFi-GAN profile.overlays"))
    )
    if not overlays:
        raise SpecError("HiFi-GAN profile must declare at least one overlay")
    _reject_duplicates((item.path for item in overlays), "overlay path")
    _reject_duplicates((item.bundle_target for item in overlays), "overlay bundle target")

    all_targets = [item.bundle_target for item in source_files if item.bundle_target is not None]
    all_targets.extend(item.bundle_target for item in overlays)
    all_targets.append("workload/hifi_gan/HIFIGAN_PROFILE.json")
    _reject_duplicates(all_targets, "HiFi-GAN payload target")

    runtime_value = _mapping(root["runtime"], "HiFi-GAN profile.runtime")
    _exact_keys(
        runtime_value,
        {
            "provider",
            "runtime_version",
            "backend_info_commit",
            "gpu_package_snapshot_url",
            "requires_python",
            "runtime_requirements",
            "install_requirements",
        },
        "HiFi-GAN profile.runtime",
    )
    backend_commit = _string(
        runtime_value["backend_info_commit"], "HiFi-GAN profile.runtime.backend_info_commit"
    )
    if not _COMMIT_RE.fullmatch(backend_commit):
        raise SpecError("HiFi-GAN runtime backend-info commit must be lowercase 40-hex")
    runtime_requirements = tuple(
        _string(item, f"HiFi-GAN profile.runtime.runtime_requirements[{index}]")
        for index, item in enumerate(
            _array(
                runtime_value["runtime_requirements"],
                "HiFi-GAN profile.runtime.runtime_requirements",
            )
        )
    )
    if not runtime_requirements or any("==" not in item for item in runtime_requirements):
        raise SpecError("HiFi-GAN runtime requirements must be nonempty exact pins")
    _reject_duplicates(runtime_requirements, "runtime requirement")
    install_requirements = tuple(
        _string(item, f"HiFi-GAN profile.runtime.install_requirements[{index}]")
        for index, item in enumerate(
            _array(
                runtime_value["install_requirements"],
                "HiFi-GAN profile.runtime.install_requirements",
            )
        )
    )
    runtime = HiFiGanRuntimeLock(
        provider=_string(runtime_value["provider"], "HiFi-GAN profile.runtime.provider"),
        runtime_version=_string(
            runtime_value["runtime_version"], "HiFi-GAN profile.runtime.runtime_version"
        ),
        backend_info_commit=backend_commit,
        gpu_package_snapshot_url=_https_url(
            runtime_value["gpu_package_snapshot_url"],
            "HiFi-GAN profile.runtime.gpu_package_snapshot_url",
        ),
        requires_python=_string(
            runtime_value["requires_python"], "HiFi-GAN profile.runtime.requires_python"
        ),
        runtime_requirements=runtime_requirements,
        install_requirements=install_requirements,
    )

    smoke = _mapping(root["smoke"], "HiFi-GAN profile.smoke")
    _exact_keys(
        smoke,
        {"initial_completed_steps", "resumed_completed_steps", "checkpoint_interval"},
        "HiFi-GAN profile.smoke",
    )
    initial = _positive_integer(
        smoke["initial_completed_steps"], "HiFi-GAN profile.smoke.initial_completed_steps"
    )
    resumed = _positive_integer(
        smoke["resumed_completed_steps"], "HiFi-GAN profile.smoke.resumed_completed_steps"
    )
    interval = _positive_integer(
        smoke["checkpoint_interval"], "HiFi-GAN profile.smoke.checkpoint_interval"
    )
    if resumed <= initial:
        raise SpecError("HiFi-GAN resumed step target must exceed the initial target")

    return HiFiGanProfile(
        profile_path=source.resolve(),
        repository=repository,
        commit=commit,
        license_spdx=_string(upstream["license_spdx"], "HiFi-GAN profile.upstream.license_spdx"),
        license_notice=_string(
            upstream["license_notice"], "HiFi-GAN profile.upstream.license_notice"
        ),
        source_files=source_files,
        overlays=overlays,
        runtime=runtime,
        initial_completed_steps=initial,
        resumed_completed_steps=resumed,
        checkpoint_interval=interval,
    )


def verify_hifigan_checkout(
    checkout: Path | str, profile: HiFiGanProfile | None = None
) -> tuple[Path, ...]:
    """Verify every profile-declared upstream file by type, size, and SHA-256."""

    selected = profile or load_hifigan_profile()
    root = Path(checkout).resolve()
    verified: list[Path] = []
    for record in selected.source_files:
        path = root.joinpath(*PurePosixPath(record.path).parts)
        _verify_regular_file(path, record.size, record.sha256, f"upstream file {record.path}")
        verified.append(path)
    return tuple(verified)


def verify_hifigan_profile_assets(
    profile: HiFiGanProfile | None = None,
) -> tuple[Path, ...]:
    """Verify all local overlay bytes declared by the selected profile."""

    selected = profile or load_hifigan_profile()
    root = selected.profile_path.parent
    verified: list[Path] = []
    for record in selected.overlays:
        path = root.joinpath(*PurePosixPath(record.path).parts)
        _verify_regular_file(path, record.size, record.sha256, f"overlay {record.path}")
        verified.append(path)
    return tuple(verified)


def hifigan_source_payload(
    checkout: Path | str, profile_path: Path | str | None = None
) -> tuple[PayloadEntry, ...]:
    """Build verified source/config/license payload declarations for Phase 2.

    Dataset entries are deliberately absent; Phase 2 step 2 supplies the measured
    LJ Speech subset separately.
    """

    profile = load_hifigan_profile(profile_path)
    root = Path(checkout).resolve()
    verified_upstream = verify_hifigan_checkout(root, profile)
    upstream_by_path = {
        record.path: path
        for record, path in zip(profile.source_files, verified_upstream, strict=True)
    }
    verified_overlays = verify_hifigan_profile_assets(profile)
    overlay_by_path = {
        record.path: path
        for record, path in zip(profile.overlays, verified_overlays, strict=True)
    }

    entries: list[PayloadEntry] = []
    for record in profile.source_files:
        if record.bundle_target is not None:
            entries.append(
                PayloadEntry(
                    role=record.role,
                    source=upstream_by_path[record.path],
                    target=record.bundle_target,
                )
            )
    for record in profile.overlays:
        entries.append(
            PayloadEntry(
                role=record.role,
                source=overlay_by_path[record.path],
                target=record.bundle_target,
            )
        )
    entries.append(
        PayloadEntry(
            role="license",
            source=profile.profile_path,
            target="workload/hifi_gan/HIFIGAN_PROFILE.json",
        )
    )
    return tuple(sorted(entries, key=lambda item: item.target))


def _parse_source_file(value: Any, index: int) -> HiFiGanSourceFile:
    where = f"HiFi-GAN profile.upstream.files[{index}]"
    item = _mapping(value, where)
    _exact_keys(item, {"path", "size", "sha256", "role", "bundle_target"}, where)
    role = _role(item["role"], f"{where}.role")
    target_value = item["bundle_target"]
    target = (
        None
        if target_value is None
        else _portable_path(target_value, f"{where}.bundle_target")
    )
    return HiFiGanSourceFile(
        path=_portable_path(item["path"], f"{where}.path"),
        size=_nonnegative_integer(item["size"], f"{where}.size"),
        sha256=_sha256(item["sha256"], f"{where}.sha256"),
        role=role,
        bundle_target=target,
    )


def _parse_overlay(value: Any, index: int) -> HiFiGanOverlay:
    where = f"HiFi-GAN profile.overlays[{index}]"
    item = _mapping(value, where)
    _exact_keys(item, {"path", "size", "sha256", "role", "bundle_target"}, where)
    return HiFiGanOverlay(
        path=_portable_path(item["path"], f"{where}.path"),
        size=_nonnegative_integer(item["size"], f"{where}.size"),
        sha256=_sha256(item["sha256"], f"{where}.sha256"),
        role=_role(item["role"], f"{where}.role"),
        bundle_target=_portable_path(item["bundle_target"], f"{where}.bundle_target"),
    )


def _verify_regular_file(path: Path, size: int, sha256: str, where: str) -> None:
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
    if metadata.st_size != size:
        raise SpecError(f"{where} size mismatch: expected {size}, observed {metadata.st_size}")
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise SpecError(f"{where} cannot be read: {path}: {exc}") from exc
    if digest != sha256:
        raise SpecError(f"{where} SHA-256 mismatch: expected {sha256}, observed {digest}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SpecError(f"duplicate HiFi-GAN profile key: {key!r}")
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


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise SpecError(f"{where} must be a nonempty string")
    return value


def _https_url(value: Any, where: str) -> str:
    text = _string(value, where)
    if not text.startswith("https://"):
        raise SpecError(f"{where} must be an HTTPS URL")
    return text


def _integer(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecError(f"{where} must be an integer")
    return value


def _nonnegative_integer(value: Any, where: str) -> int:
    result = _integer(value, where)
    if result < 0:
        raise SpecError(f"{where} must be nonnegative")
    return result


def _positive_integer(value: Any, where: str) -> int:
    result = _integer(value, where)
    if result <= 0:
        raise SpecError(f"{where} must be greater than zero")
    return result


def _portable_path(value: Any, where: str) -> str:
    text = _string(value, where)
    path = PurePosixPath(text)
    if (
        text.startswith("/")
        or "\\" in text
        or "\x00" in text
        or path.as_posix() != text
        or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
    ):
        raise SpecError(f"{where} must be a normalized relative POSIX path")
    return text


def _sha256(value: Any, where: str) -> str:
    text = _string(value, where)
    if not _SHA256_RE.fullmatch(text):
        raise SpecError(f"{where} must be a lowercase SHA-256 digest")
    return text


def _role(value: Any, where: str) -> str:
    text = _string(value, where)
    if text not in _ALLOWED_ROLES:
        raise SpecError(f"{where} must be one of {sorted(_ALLOWED_ROLES)}")
    return text


def _reject_duplicates(values: Iterable[str], description: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise SpecError(f"duplicate HiFi-GAN {description}: {value}")
        seen.add(value)
