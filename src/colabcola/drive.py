"""Provider-neutral Google Drive staging and cleanup lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Mapping, Protocol, Sequence

from .errors import DriveTransportError
from .transport import StagedBundleArtifact, verify_staged_bundle_file


DRIVE_ROOT_FOLDER_NAME = "colabcola"
DRIVE_STAGING_FOLDER_NAME = "staging"
DRIVE_RECEIPT_SCHEMA_VERSION = 1
DRIVE_BINARY_MIME_TYPE = "application/octet-stream"

_DRIVE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_REPARSE_FLAG = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_COPY_BUFFER_BYTES = 8 * 1024 * 1024
_RECEIPT_KEYS = frozenset(
    {
        "cleaned_file_ids",
        "cleanup_file_ids",
        "job_id",
        "remote_file_id",
        "remote_name",
        "remote_parent_id",
        "remote_relative_path",
        "reused",
        "schema_version",
        "staged_bytes",
        "staged_sha256",
        "state",
        "verified_results",
    }
)


@dataclass(frozen=True, slots=True)
class DriveFolder:
    """One exact Drive folder returned by an adapter."""

    file_id: str
    name: str
    parent_id: str


@dataclass(frozen=True, slots=True)
class DriveFile:
    """One exact staged binary object returned by an adapter."""

    file_id: str
    name: str
    parent_id: str
    size: int
    mime_type: str
    app_properties: tuple[tuple[str, str], ...]

    def property_value(self, key: str) -> str | None:
        return dict(self.app_properties).get(key)


@dataclass(frozen=True, slots=True)
class VerifiedResult:
    """Local result evidence that is safe to retain after remote cleanup."""

    name: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class DriveStagingReceipt:
    """Receipt-scoped authority for one staged object and its cleanup."""

    schema_version: int
    state: str
    job_id: str
    staged_sha256: str
    staged_bytes: int
    remote_file_id: str
    remote_parent_id: str
    remote_relative_path: str
    remote_name: str
    reused: bool
    cleanup_file_ids: tuple[str, ...]
    verified_results: tuple[VerifiedResult, ...] = ()
    cleaned_file_ids: tuple[str, ...] = ()

    def as_value(self) -> dict[str, object]:
        return {
            "cleaned_file_ids": list(self.cleaned_file_ids),
            "cleanup_file_ids": list(self.cleanup_file_ids),
            "job_id": self.job_id,
            "remote_file_id": self.remote_file_id,
            "remote_name": self.remote_name,
            "remote_parent_id": self.remote_parent_id,
            "remote_relative_path": self.remote_relative_path,
            "reused": self.reused,
            "schema_version": self.schema_version,
            "staged_bytes": self.staged_bytes,
            "staged_sha256": self.staged_sha256,
            "state": self.state,
            "verified_results": [
                {"name": item.name, "sha256": item.sha256, "size": item.size}
                for item in self.verified_results
            ],
        }


class DriveBackend(Protocol):
    """Minimal adapter seam; credentials and HTTP libraries stay outside the core."""

    def ensure_folder(self, *, name: str, parent_id: str) -> DriveFolder: ...

    def find_payload(
        self,
        *,
        name: str,
        parent_id: str,
        app_properties: Mapping[str, str],
    ) -> Sequence[DriveFile]: ...

    def upload_resumable(
        self,
        *,
        local_path: Path,
        name: str,
        parent_id: str,
        mime_type: str,
        app_properties: Mapping[str, str],
    ) -> DriveFile: ...

    def trash(self, *, file_id: str) -> None: ...


def stage_drive_bundle(
    artifact: StagedBundleArtifact,
    local_path: Path | str,
    backend: DriveBackend,
) -> DriveStagingReceipt:
    """Reuse one exact staged object or resumably upload it through an adapter."""

    source = verify_staged_bundle_file(local_path, artifact)
    root = backend.ensure_folder(name=DRIVE_ROOT_FOLDER_NAME, parent_id="root")
    _verify_folder(root, name=DRIVE_ROOT_FOLDER_NAME, parent_id="root")
    staging = backend.ensure_folder(name=DRIVE_STAGING_FOLDER_NAME, parent_id=root.file_id)
    _verify_folder(staging, name=DRIVE_STAGING_FOLDER_NAME, parent_id=root.file_id)
    job_folder = backend.ensure_folder(name=artifact.bundle.job_id, parent_id=staging.file_id)
    _verify_folder(job_folder, name=artifact.bundle.job_id, parent_id=staging.file_id)

    properties = {
        "colabcola_job_id": artifact.bundle.job_id,
        "colabcola_staged_sha256": artifact.staged_sha256,
    }
    matches = tuple(
        backend.find_payload(
            name=artifact.file_name,
            parent_id=job_folder.file_id,
            app_properties=properties,
        )
    )
    if len(matches) > 1:
        raise DriveTransportError(
            "Drive adapter returned multiple exact staged payloads; file identity is ambiguous"
        )
    if matches:
        remote = matches[0]
        reused = True
    else:
        remote = backend.upload_resumable(
            local_path=source,
            name=artifact.file_name,
            parent_id=job_folder.file_id,
            mime_type=DRIVE_BINARY_MIME_TYPE,
            app_properties=properties,
        )
        reused = False
    _verify_remote_payload(remote, artifact, job_folder.file_id, properties)

    relative_path = PurePosixPath(
        DRIVE_ROOT_FOLDER_NAME,
        DRIVE_STAGING_FOLDER_NAME,
        artifact.bundle.job_id,
        artifact.file_name,
    ).as_posix()
    receipt = DriveStagingReceipt(
        schema_version=DRIVE_RECEIPT_SCHEMA_VERSION,
        state="staged",
        job_id=artifact.bundle.job_id,
        staged_sha256=artifact.staged_sha256,
        staged_bytes=artifact.staged_bytes,
        remote_file_id=remote.file_id,
        remote_parent_id=remote.parent_id,
        remote_relative_path=relative_path,
        remote_name=remote.name,
        reused=reused,
        cleanup_file_ids=(remote.file_id,),
    )
    verify_drive_receipt(receipt)
    return receipt


def drive_receipt_for_observed_upload(
    artifact: StagedBundleArtifact,
    *,
    remote_file_id: str,
    remote_parent_id: str,
    remote_relative_path: str,
    reused: bool = False,
) -> DriveStagingReceipt:
    """Record a connector-assisted upload after independent metadata readback."""

    receipt = DriveStagingReceipt(
        schema_version=DRIVE_RECEIPT_SCHEMA_VERSION,
        state="staged",
        job_id=artifact.bundle.job_id,
        staged_sha256=artifact.staged_sha256,
        staged_bytes=artifact.staged_bytes,
        remote_file_id=remote_file_id,
        remote_parent_id=remote_parent_id,
        remote_relative_path=remote_relative_path,
        remote_name=artifact.file_name,
        reused=reused,
        cleanup_file_ids=(remote_file_id,),
    )
    verify_drive_receipt(receipt)
    return receipt


def record_verified_result(
    receipt: DriveStagingReceipt,
    result_path: Path | str,
    *,
    expected_size: int,
    expected_sha256: str,
) -> DriveStagingReceipt:
    """Authorize normal cleanup only after exact local result verification."""

    verify_drive_receipt(receipt)
    if receipt.state not in {"staged", "results_verified"}:
        raise DriveTransportError(
            f"cannot record a verified result from receipt state {receipt.state!r}"
        )
    result = _verify_local_file(
        result_path, expected_size=expected_size, expected_sha256=expected_sha256
    )
    by_name = {item.name: item for item in receipt.verified_results}
    by_name[result.name] = result
    updated = replace(
        receipt,
        state="results_verified",
        verified_results=tuple(by_name[name] for name in sorted(by_name)),
    )
    verify_drive_receipt(updated)
    return updated


def authorize_abandoned_cleanup(receipt: DriveStagingReceipt) -> DriveStagingReceipt:
    """Explicitly authorize cleanup when no result is expected or recoverable."""

    verify_drive_receipt(receipt)
    if receipt.state == "cleaned":
        return receipt
    if receipt.state not in {"staged", "results_verified", "cleanup_authorized"}:
        raise DriveTransportError(
            f"cannot authorize abandoned cleanup from state {receipt.state!r}"
        )
    updated = replace(receipt, state="cleanup_authorized")
    verify_drive_receipt(updated)
    return updated


def cleanup_drive_staging(
    receipt: DriveStagingReceipt, backend: DriveBackend
) -> DriveStagingReceipt:
    """Move only receipt-scoped objects to trash after an authorized transition."""

    verify_drive_receipt(receipt)
    if receipt.state == "cleaned":
        return receipt
    if receipt.state not in {"results_verified", "cleanup_authorized"}:
        raise DriveTransportError(
            "remote staging cleanup requires verified results or explicit abandoned cleanup"
        )
    cleaned = list(receipt.cleaned_file_ids)
    for file_id in receipt.cleanup_file_ids:
        if file_id in cleaned:
            continue
        backend.trash(file_id=file_id)
        cleaned.append(file_id)
    updated = replace(receipt, state="cleaned", cleaned_file_ids=tuple(cleaned))
    verify_drive_receipt(updated)
    return updated


def verify_drive_receipt(receipt: DriveStagingReceipt) -> None:
    """Validate identity, safe path, lifecycle state, and cleanup scope."""

    if receipt.schema_version != DRIVE_RECEIPT_SCHEMA_VERSION:
        raise DriveTransportError("unsupported Drive receipt schema version")
    if not re.fullmatch(r"[0-9a-f]{64}", receipt.job_id):
        raise DriveTransportError("Drive receipt job ID is not a SHA-256 digest")
    if not re.fullmatch(r"[0-9a-f]{64}", receipt.staged_sha256):
        raise DriveTransportError("Drive receipt staged SHA-256 is invalid")
    if isinstance(receipt.staged_bytes, bool) or receipt.staged_bytes <= 0:
        raise DriveTransportError("Drive receipt staged byte count must be positive")
    _verify_drive_id(receipt.remote_file_id, "remote file ID")
    _verify_drive_id(receipt.remote_parent_id, "remote parent ID")
    relative = PurePosixPath(receipt.remote_relative_path)
    if (
        relative.is_absolute()
        or relative.as_posix() != receipt.remote_relative_path
        or any(part in {"", ".", ".."} or ":" in part for part in relative.parts)
        or tuple(relative.parts[:2])
        != (DRIVE_ROOT_FOLDER_NAME, DRIVE_STAGING_FOLDER_NAME)
    ):
        raise DriveTransportError("Drive receipt relative path is unsafe or outside staging")
    if relative.name != receipt.remote_name:
        raise DriveTransportError("Drive receipt remote name and relative path disagree")
    if receipt.state not in {
        "staged",
        "results_verified",
        "cleanup_authorized",
        "cleaned",
    }:
        raise DriveTransportError(f"unsupported Drive receipt state: {receipt.state!r}")
    if receipt.state == "results_verified" and not receipt.verified_results:
        raise DriveTransportError(
            "results-verified Drive receipt must contain local result evidence"
        )
    if not receipt.cleanup_file_ids or len(set(receipt.cleanup_file_ids)) != len(
        receipt.cleanup_file_ids
    ):
        raise DriveTransportError("Drive cleanup scope must contain unique file IDs")
    for file_id in (*receipt.cleanup_file_ids, *receipt.cleaned_file_ids):
        _verify_drive_id(file_id, "cleanup file ID")
    if not set(receipt.cleaned_file_ids).issubset(receipt.cleanup_file_ids):
        raise DriveTransportError("cleaned Drive IDs exceed the receipt cleanup scope")
    if receipt.state == "cleaned" and set(receipt.cleaned_file_ids) != set(
        receipt.cleanup_file_ids
    ):
        raise DriveTransportError("cleaned receipt has an incomplete cleanup scope")
    for item in receipt.verified_results:
        if not item.name or "/" in item.name or "\\" in item.name:
            raise DriveTransportError("verified result name must be one safe basename")
        if isinstance(item.size, bool) or item.size < 0:
            raise DriveTransportError("verified result byte count is invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", item.sha256):
            raise DriveTransportError("verified result SHA-256 is invalid")


def drive_receipt_json_bytes(receipt: DriveStagingReceipt) -> bytes:
    verify_drive_receipt(receipt)
    return (
        json.dumps(
            receipt.as_value(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def load_drive_receipt(path: Path | str) -> DriveStagingReceipt:
    """Strictly load one canonical, credential-free lifecycle receipt."""

    source = Path(path)
    metadata = os.lstat(source)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or getattr(metadata, "st_file_attributes", 0) & _REPARSE_FLAG
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise DriveTransportError(f"Drive receipt is unavailable or unsafe: {source}")
    content = source.read_bytes()
    try:
        value = json.loads(content.decode("utf-8"), object_pairs_hook=_unique_json_object)
    except UnicodeDecodeError as exc:
        raise DriveTransportError("Drive receipt must be UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise DriveTransportError(f"Drive receipt is invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict) or set(value) != _RECEIPT_KEYS:
        raise DriveTransportError("Drive receipt has missing or unknown fields")

    string_fields = (
        "state",
        "job_id",
        "staged_sha256",
        "remote_file_id",
        "remote_parent_id",
        "remote_relative_path",
        "remote_name",
    )
    if any(not isinstance(value[field], str) for field in string_fields):
        raise DriveTransportError("Drive receipt string field has the wrong type")
    if isinstance(value["schema_version"], bool) or not isinstance(
        value["schema_version"], int
    ):
        raise DriveTransportError("Drive receipt schema version has the wrong type")
    if isinstance(value["staged_bytes"], bool) or not isinstance(
        value["staged_bytes"], int
    ):
        raise DriveTransportError("Drive receipt byte count has the wrong type")
    if not isinstance(value["reused"], bool):
        raise DriveTransportError("Drive receipt reused field has the wrong type")

    cleanup = _receipt_string_list(value["cleanup_file_ids"], "cleanup_file_ids")
    cleaned = _receipt_string_list(value["cleaned_file_ids"], "cleaned_file_ids")
    result_values = value["verified_results"]
    if not isinstance(result_values, list):
        raise DriveTransportError("Drive receipt verified_results must be an array")
    results: list[VerifiedResult] = []
    for index, item in enumerate(result_values):
        if not isinstance(item, dict) or set(item) != {"name", "sha256", "size"}:
            raise DriveTransportError(
                f"Drive receipt verified_results[{index}] has invalid fields"
            )
        if not isinstance(item["name"], str) or not isinstance(item["sha256"], str):
            raise DriveTransportError(
                f"Drive receipt verified_results[{index}] string field is invalid"
            )
        if isinstance(item["size"], bool) or not isinstance(item["size"], int):
            raise DriveTransportError(
                f"Drive receipt verified_results[{index}] size is invalid"
            )
        results.append(
            VerifiedResult(name=item["name"], size=item["size"], sha256=item["sha256"])
        )

    receipt = DriveStagingReceipt(
        schema_version=value["schema_version"],
        state=value["state"],
        job_id=value["job_id"],
        staged_sha256=value["staged_sha256"],
        staged_bytes=value["staged_bytes"],
        remote_file_id=value["remote_file_id"],
        remote_parent_id=value["remote_parent_id"],
        remote_relative_path=value["remote_relative_path"],
        remote_name=value["remote_name"],
        reused=value["reused"],
        cleanup_file_ids=cleanup,
        verified_results=tuple(results),
        cleaned_file_ids=cleaned,
    )
    verify_drive_receipt(receipt)
    if drive_receipt_json_bytes(receipt) != content:
        raise DriveTransportError("Drive receipt is not in canonical persisted form")
    return receipt


def write_drive_receipt(
    receipt: DriveStagingReceipt, output_path: Path | str
) -> Path:
    """Atomically persist a verified, credential-free lifecycle receipt."""

    destination = Path(output_path)
    if destination.suffix.lower() != ".json":
        raise DriveTransportError(f"Drive receipt must use the .json extension: {destination}")
    if destination.exists() and destination.is_dir():
        raise DriveTransportError(f"Drive receipt output is a directory: {destination}")
    if destination.is_symlink():
        raise DriveTransportError(f"Drive receipt output may not be a symlink: {destination}")
    content = drive_receipt_json_bytes(receipt)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    if destination.read_bytes() != content:
        raise DriveTransportError("persisted Drive receipt bytes do not match")
    return destination.resolve(strict=True)


def _verify_folder(folder: DriveFolder, *, name: str, parent_id: str) -> None:
    _verify_drive_id(folder.file_id, "folder ID")
    if folder.name != name or folder.parent_id != parent_id:
        raise DriveTransportError("Drive adapter returned an unexpected folder identity")


def _verify_remote_payload(
    remote: DriveFile,
    artifact: StagedBundleArtifact,
    parent_id: str,
    properties: Mapping[str, str],
) -> None:
    _verify_drive_id(remote.file_id, "remote file ID")
    if remote.name != artifact.file_name or remote.parent_id != parent_id:
        raise DriveTransportError("Drive adapter returned an unexpected staged object identity")
    if remote.size != artifact.staged_bytes:
        raise DriveTransportError("Drive staged object byte count does not match")
    if remote.mime_type != DRIVE_BINARY_MIME_TYPE:
        raise DriveTransportError("Drive staged object MIME type does not match")
    actual_properties = dict(remote.app_properties)
    if any(actual_properties.get(key) != value for key, value in properties.items()):
        raise DriveTransportError("Drive staged object app properties do not match")


def _verify_drive_id(value: str, label: str) -> None:
    if not isinstance(value, str) or not _DRIVE_ID.fullmatch(value):
        raise DriveTransportError(f"{label} is invalid")


def _verify_local_file(
    path: Path | str, *, expected_size: int, expected_sha256: str
) -> VerifiedResult:
    if isinstance(expected_size, bool) or expected_size < 0:
        raise DriveTransportError("expected result byte count is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise DriveTransportError("expected result SHA-256 is invalid")
    source = Path(path)
    metadata = os.lstat(source)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or getattr(metadata, "st_file_attributes", 0) & _REPARSE_FLAG
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise DriveTransportError(f"result file is unavailable or unsafe: {source}")
    if metadata.st_size != expected_size:
        raise DriveTransportError("downloaded result byte count does not match")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while block := stream.read(_COPY_BUFFER_BYTES):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise DriveTransportError("downloaded result SHA-256 does not match")
    return VerifiedResult(name=source.name, size=metadata.st_size, sha256=expected_sha256)


def _receipt_string_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise DriveTransportError(f"Drive receipt {field} must be an array of strings")
    return tuple(value)


def _unique_json_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DriveTransportError(f"Drive receipt contains duplicate field {key!r}")
        result[key] = value
    return result
