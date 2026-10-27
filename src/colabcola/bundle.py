"""Deterministic payload expansion, manifest generation, and ZIP packaging."""

from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import os
import stat
import struct
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import BundleError, SpecError
from .invocation import JobInvocation
from .spec import JobSpec, PayloadEntry

COMPILER_CONTRACT_VERSION = "0.10"
ARCHIVE_WARNING_BYTES = 1024 * 1024 * 1024
ZIP_COMPRESSION_LEVEL = 9
ZIP_FIXED_DATETIME = (1980, 1, 1, 0, 0, 0)

_COPY_BUFFER_BYTES = 1024 * 1024
_REPARSE_FLAG = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


@dataclass(frozen=True, slots=True)
class FileRecord:
    """One portable file and its content identity."""

    role: str
    path: str
    size: int
    sha256: str
    mode: str = "0644"

    def as_manifest_value(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "path": self.path,
            "role": self.role,
            "sha256": self.sha256,
            "size": self.size,
        }


@dataclass(frozen=True, slots=True)
class Bundle:
    """A portable manifest plus the exact deterministic archive it describes."""

    name: str
    files: tuple[FileRecord, ...]
    canonical_manifest_bytes: bytes
    archive_bytes: bytes
    manifest_sha256: str
    archive_sha256: str
    job_id: str
    raw_input_bytes: int
    warnings: tuple[str, ...]

    def verify(self) -> None:
        """Verify hashes, canonical encoding, ZIP metadata, and every archived file."""

        verify_bundle(self)


@dataclass(frozen=True, slots=True)
class _ExpandedFile:
    role: str
    path: str
    source: Path


def build_bundle(invocation: JobInvocation) -> Bundle:
    """Compile one validated invocation into a deterministic in-memory bundle."""

    expanded = _expand_payload(invocation.spec)
    _validate_runtime_targets(invocation.spec, expanded)
    _reject_remote_input_conflicts(invocation.spec, expanded)
    archive_bytes, records = _write_archive(expanded)
    bundle_warnings = archive_size_warnings(
        len(archive_bytes), warning_bytes=ARCHIVE_WARNING_BYTES
    )
    manifest = _portable_manifest(invocation, records)
    canonical_manifest = canonical_json_bytes(manifest)
    manifest_sha256 = hashlib.sha256(canonical_manifest).hexdigest()
    archive_digest = hashlib.sha256(archive_bytes).digest()
    archive_sha256 = archive_digest.hex()
    job_id = _compute_job_id(canonical_manifest, archive_digest)
    bundle = Bundle(
        name=invocation.spec.name,
        files=records,
        canonical_manifest_bytes=canonical_manifest,
        archive_bytes=archive_bytes,
        manifest_sha256=manifest_sha256,
        archive_sha256=archive_sha256,
        job_id=job_id,
        raw_input_bytes=sum(record.size for record in records),
        warnings=bundle_warnings,
    )
    verify_bundle(bundle)
    for message in bundle_warnings:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return bundle


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Encode a mapping as canonical UTF-8 JSON with one terminal newline."""

    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise BundleError(f"manifest cannot be encoded as canonical JSON: {exc}") from exc
    return text.encode("utf-8") + b"\n"


def archive_size_warnings(
    archive_bytes: int, *, warning_bytes: int = ARCHIVE_WARNING_BYTES
) -> tuple[str, ...]:
    """Describe non-blocking notebook-size risk at the compiler warning threshold."""

    if (
        isinstance(archive_bytes, bool)
        or not isinstance(archive_bytes, int)
        or archive_bytes < 0
    ):
        raise ValueError("archive_bytes must be a nonnegative integer")
    if (
        isinstance(warning_bytes, bool)
        or not isinstance(warning_bytes, int)
        or warning_bytes <= 0
    ):
        raise ValueError("warning_bytes must be a positive integer")
    if archive_bytes < warning_bytes:
        return ()
    base64_bytes = 4 * ((archive_bytes + 2) // 3)
    warning_mib = warning_bytes / (1024 * 1024)
    return (
        f"deterministic archive is {archive_bytes} bytes, meeting or exceeding the "
        f"{warning_bytes}-byte ({warning_mib:g} MiB) warning threshold; Drive staging must "
        f"transfer at least the archive bytes, and secondary embedded-base64 mode would use "
        f"{base64_bytes} bytes before notebook JSON overhead",
    )


def verify_bundle(bundle: Bundle) -> None:
    """Reject a bundle whose identity, contents, or ZIP metadata have changed."""

    if hashlib.sha256(bundle.canonical_manifest_bytes).hexdigest() != bundle.manifest_sha256:
        raise BundleError("canonical manifest SHA-256 does not match")
    if hashlib.sha256(bundle.archive_bytes).hexdigest() != bundle.archive_sha256:
        raise BundleError("archive SHA-256 does not match")

    try:
        archive_digest = bytes.fromhex(bundle.archive_sha256)
    except ValueError as exc:
        raise BundleError("archive SHA-256 is not a hexadecimal digest") from exc
    expected_job_id = _compute_job_id(bundle.canonical_manifest_bytes, archive_digest)
    if expected_job_id != bundle.job_id:
        raise BundleError("job ID does not match the manifest and archive identities")

    manifest = _decode_canonical_manifest(bundle.canonical_manifest_bytes)
    try:
        manifest_files = manifest["files"]
    except KeyError as exc:
        raise BundleError("canonical manifest is missing files") from exc
    expected_files = [record.as_manifest_value() for record in bundle.files]
    if manifest_files != expected_files:
        raise BundleError("canonical manifest file records do not match the bundle")
    if bundle.raw_input_bytes != sum(record.size for record in bundle.files):
        raise BundleError("raw input byte count does not match the file records")
    expected_warnings = archive_size_warnings(
        len(bundle.archive_bytes), warning_bytes=ARCHIVE_WARNING_BYTES
    )
    if bundle.warnings != expected_warnings:
        raise BundleError("bundle warnings do not match the archive size")

    try:
        with zipfile.ZipFile(io.BytesIO(bundle.archive_bytes), "r") as archive:
            if archive.comment:
                raise BundleError("archive comment is not empty")
            infos = archive.infolist()
            if [info.filename for info in infos] != [record.path for record in bundle.files]:
                raise BundleError("archive entry order or paths do not match the manifest")
            for info, record in zip(infos, bundle.files, strict=True):
                _verify_zip_info(info, record)
                content = archive.read(info)
                if len(content) != record.size:
                    raise BundleError(f"archive size does not match for {record.path}")
                if hashlib.sha256(content).hexdigest() != record.sha256:
                    raise BundleError(f"archive content SHA-256 does not match for {record.path}")
    except BundleError:
        raise
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        raise BundleError(f"archive is not a valid deterministic ZIP: {exc}") from exc


def _expand_payload(spec: JobSpec) -> tuple[_ExpandedFile, ...]:
    expanded: list[_ExpandedFile] = []
    for index, entry in enumerate(spec.payload):
        where = f"JobSpec.payload[{index}]"
        metadata = _safe_lstat(entry.source, where)
        if _is_forbidden(metadata):
            raise SpecError(
                f"{where}.source is a forbidden symlink or reparse point: {entry.source}"
            )
        if stat.S_ISREG(metadata.st_mode):
            if entry.exclude:
                raise SpecError(f"{where}.exclude is only valid for directory sources")
            expanded.append(_ExpandedFile(entry.role, entry.target, entry.source))
        elif stat.S_ISDIR(metadata.st_mode):
            patterns = tuple(
                _validate_exclude_pattern(pattern, f"{where}.exclude[{pattern_index}]")
                for pattern_index, pattern in enumerate(entry.exclude)
            )
            before = len(expanded)
            _expand_directory(
                entry,
                patterns,
                expanded,
                where,
                maximum_file_count=spec.limits.maximum_file_count,
            )
            if len(expanded) == before:
                raise SpecError(f"{where} expands to no files after exclusions")
        else:
            raise SpecError(f"{where}.source is not a regular file or directory: {entry.source}")

        if len(expanded) > spec.limits.maximum_file_count:
            raise SpecError(
                "expanded payload file count exceeds "
                f"limits.maximum_file_count={spec.limits.maximum_file_count}"
            )

    expanded.sort(key=lambda item: item.path)
    _reject_target_collisions(expanded)
    return tuple(expanded)


def _expand_directory(
    entry: PayloadEntry,
    patterns: tuple[tuple[str, ...], ...],
    expanded: list[_ExpandedFile],
    where: str,
    *,
    maximum_file_count: int,
) -> None:
    def visit(directory: Path, relative_parts: tuple[str, ...]) -> None:
        try:
            with os.scandir(directory) as iterator:
                children = sorted(iterator, key=lambda child: child.name)
        except OSError as exc:
            raise SpecError(f"{where}.source cannot be enumerated: {directory}: {exc}") from exc

        for child in children:
            source = directory / child.name
            child_parts = relative_parts + (child.name,)
            relative = _portable_relative_path(child_parts, where)
            metadata = _safe_lstat(source, where)
            if _is_forbidden(metadata):
                raise SpecError(
                    f"{where}.source contains a forbidden symlink or reparse point: {source}"
                )
            excluded = any(_glob_parts_match(child_parts, pattern) for pattern in patterns)
            if stat.S_ISDIR(metadata.st_mode):
                if not excluded:
                    visit(source, child_parts)
            elif stat.S_ISREG(metadata.st_mode):
                if not excluded:
                    expanded.append(
                        _ExpandedFile(
                            role=entry.role,
                            path=f"{entry.target}/{relative}",
                            source=source,
                        )
                    )
                    if len(expanded) > maximum_file_count:
                        raise SpecError(
                            "expanded payload file count exceeds "
                            f"limits.maximum_file_count={maximum_file_count}"
                        )
            else:
                raise SpecError(f"{where}.source contains a forbidden special file: {source}")

    visit(entry.source, ())


def _validate_exclude_pattern(pattern: str, where: str) -> tuple[str, ...]:
    if "\x00" in pattern or "\\" in pattern or pattern.startswith("/"):
        raise SpecError(f"{where} must be a relative POSIX glob")
    parts = tuple(pattern.split("/"))
    if any(
        not part
        or part in {".", ".."}
        or ":" in part
        or ("**" in part and part != "**")
        for part in parts
    ):
        raise SpecError(f"{where} must be a normalized relative POSIX glob")
    return parts


def _glob_parts_match(path_parts: Sequence[str], pattern_parts: Sequence[str]) -> bool:
    if not pattern_parts:
        return not path_parts
    pattern = pattern_parts[0]
    if pattern == "**":
        return _glob_parts_match(path_parts, pattern_parts[1:]) or (
            bool(path_parts) and _glob_parts_match(path_parts[1:], pattern_parts)
        )
    return bool(path_parts) and fnmatch.fnmatchcase(path_parts[0], pattern) and _glob_parts_match(
        path_parts[1:], pattern_parts[1:]
    )


def _portable_relative_path(parts: Sequence[str], where: str) -> str:
    for part in parts:
        if not part or part in {".", ".."} or ":" in part or "\\" in part or "\x00" in part:
            raise SpecError(
                f"{where}.source contains a name that is not portable to a POSIX job path: {part!r}"
            )
    return "/".join(parts)


def _reject_target_collisions(files: Sequence[_ExpandedFile]) -> None:
    previous: str | None = None
    for item in files:
        if item.path == previous:
            raise SpecError(f"expanded payload contains duplicate target: {item.path}")
        previous = item.path

    paths = {item.path for item in files}
    for item in files:
        parts = item.path.split("/")
        for end in range(1, len(parts)):
            ancestor = "/".join(parts[:end])
            if ancestor in paths:
                raise SpecError(
                    "expanded payload contains a file/directory target conflict: "
                    f"{ancestor} and {item.path}"
                )


def _reject_remote_input_conflicts(spec: JobSpec, files: Sequence[_ExpandedFile]) -> None:
    """Remote inputs are files of the job tree: they obey the file-count limit and the no-nesting rule."""

    if not spec.remote_inputs:
        return
    if len(files) + len(spec.remote_inputs) > spec.limits.maximum_file_count:
        raise SpecError(
            "expanded payload plus remote inputs exceed "
            f"limits.maximum_file_count={spec.limits.maximum_file_count}"
        )
    remote_targets = {entry.target for entry in spec.remote_inputs}
    all_paths = {item.path for item in files} | remote_targets
    for target in sorted(remote_targets):
        parts = target.split("/")
        for end in range(1, len(parts)):
            if "/".join(parts[:end]) in all_paths:
                raise SpecError(
                    f"remote input target lies beneath another target: {target}"
                )
    for item in files:
        parts = item.path.split("/")
        for end in range(1, len(parts)):
            if "/".join(parts[:end]) in remote_targets:
                raise SpecError(
                    f"payload file lies beneath a remote input target: {item.path}"
                )


def _validate_runtime_targets(spec: JobSpec, files: Sequence[_ExpandedFile]) -> None:
    roles_by_path = {item.path: item.role for item in files}
    entrypoint_role = roles_by_path.get(spec.entrypoint.path)
    if entrypoint_role != "source":
        raise SpecError(
            "entrypoint.path must resolve to an expanded payload file with role 'source': "
            f"{spec.entrypoint.path}"
        )

    if spec.entrypoint.cwd != ".":
        prefix = spec.entrypoint.cwd + "/"
        if spec.entrypoint.cwd in roles_by_path or not any(
            path.startswith(prefix) for path in roles_by_path
        ):
            raise SpecError(
                "entrypoint.cwd must resolve to a reconstructed payload directory: "
                f"{spec.entrypoint.cwd}"
            )

    for wheel in spec.python.embedded_wheels:
        if roles_by_path.get(wheel) != "wheel":
            raise SpecError(
                "python.embedded_wheels must resolve to expanded payload files with role 'wheel': "
                f"{wheel}"
            )


def _write_archive(
    files: Sequence[_ExpandedFile],
) -> tuple[bytes, tuple[FileRecord, ...]]:
    buffer = io.BytesIO()
    records: list[FileRecord] = []
    try:
        with zipfile.ZipFile(
            buffer,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=ZIP_COMPRESSION_LEVEL,
        ) as archive:
            archive.comment = b""
            for item in files:
                _assert_regular_file(item.source)
                digest = hashlib.sha256()
                size = 0
                info = _normalized_zip_info(item.path)
                try:
                    with item.source.open("rb") as source, archive.open(info, "w") as target:
                        while chunk := source.read(_COPY_BUFFER_BYTES):
                            digest.update(chunk)
                            size += len(chunk)
                            target.write(chunk)
                except OSError as exc:
                    raise SpecError(f"payload file cannot be read: {item.source}: {exc}") from exc
                records.append(
                    FileRecord(
                        role=item.role,
                        path=item.path,
                        size=size,
                        sha256=digest.hexdigest(),
                    )
                )
    except (RuntimeError, NotImplementedError, zipfile.LargeZipFile) as exc:
        raise SpecError(f"payload cannot be encoded as a deterministic ZIP: {exc}") from exc

    archive_bytes = buffer.getvalue()
    return archive_bytes, tuple(records)


def _normalized_zip_info(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=path, date_time=ZIP_FIXED_DATETIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.create_version = 20
    info.extract_version = 20
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.internal_attr = 0
    info.extra = b""
    info.comment = b""
    return info


def _portable_manifest(
    invocation: JobInvocation, files: Sequence[FileRecord]
) -> dict[str, object]:
    spec = invocation.spec
    manifest: dict[str, object] = {
        "archive": {
            "compression": "deflate",
            "compression_level": ZIP_COMPRESSION_LEVEL,
            "format": "zip",
        },
        "compiler_contract_version": COMPILER_CONTRACT_VERSION,
        "entrypoint": {
            "argv": list(invocation.effective_argv),
            "cwd": spec.entrypoint.cwd,
            "kind": spec.entrypoint.kind,
            "path": spec.entrypoint.path,
        },
        "files": [record.as_manifest_value() for record in files],
        "job_spec_schema_version": spec.schema_version,
        "limits": {
            "maximum_file_count": spec.limits.maximum_file_count,
        },
        "outputs": [
            {
                "maximum_bytes": output.maximum_bytes,
                "pattern": output.pattern,
                "required": output.required,
            }
            for output in spec.outputs
        ],
        "python": {
            "embedded_wheels": list(spec.python.embedded_wheels),
            "install_requirements": list(spec.python.install_requirements),
            "requires_python": spec.python.requires_python,
            "runtime_requirements": list(spec.python.runtime_requirements),
        },
        "resources": {
            "accelerator": spec.resources.accelerator,
            "cuda_required": spec.resources.cuda_required,
            "gpu_count": spec.resources.gpu_count,
            "minimum_gpu_memory_bytes": spec.resources.minimum_gpu_memory_bytes,
        },
        "runtime": {
            "provider": spec.runtime.provider,
            "version": spec.runtime.version,
        },
        "run": {
            "capture_stderr": spec.run.capture_stderr,
            "capture_stdout": spec.run.capture_stdout,
            "environment": dict(spec.run.environment),
            "seed": spec.run.seed,
            "timeout_seconds": spec.run.timeout_seconds,
        },
    }
    if spec.python.isolation is not None:
        from dataclasses import asdict
        manifest["python"]["isolation"] = asdict(spec.python.isolation)
    if spec.python.pip_requirements:
        manifest["python"]["pip_requirements"] = list(spec.python.pip_requirements)
    if spec.durable_checkpoints is not None:
        checkpoints = spec.durable_checkpoints
        manifest["durable_checkpoints"] = {
            "directory": checkpoints.directory,
            "filename_digits": checkpoints.filename_digits,
            "generator_prefix": checkpoints.generator_prefix,
            "poll_seconds": checkpoints.poll_seconds,
            "remote_retention": checkpoints.remote_retention,
            "state_prefix": checkpoints.state_prefix,
        }
    if spec.managed_recovery is not None:
        manifest["managed_recovery"] = {
            "checkpoint_pattern": spec.managed_recovery.checkpoint_pattern,
            "log_patterns": list(spec.managed_recovery.log_patterns),
        }
    if spec.managed_recovery is not None and spec.managed_recovery.artifact_groups:
        manifest["managed_recovery"]["artifact_groups"] = [g.mapping() for g in spec.managed_recovery.artifact_groups]
    if spec.remote_inputs:
        # Present only when declared, so jobs without remote inputs keep their identities.
        manifest["remote_inputs"] = [
            {
                "bytes": entry.bytes,
                "sha256": entry.sha256,
                "target": entry.target,
                "url": entry.url,
            }
            for entry in sorted(spec.remote_inputs, key=lambda entry: entry.target)
        ]
    return manifest


def _compute_job_id(canonical_manifest: bytes, archive_digest: bytes) -> str:
    framed = (
        len(canonical_manifest).to_bytes(8, "big")
        + canonical_manifest
        + len(archive_digest).to_bytes(8, "big")
        + archive_digest
    )
    return hashlib.sha256(framed).hexdigest()


def _decode_canonical_manifest(data: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError, BundleError) as exc:
        raise BundleError(f"canonical manifest is invalid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise BundleError("canonical manifest must contain a JSON object")
    if canonical_json_bytes(value) != data:
        raise BundleError("manifest bytes are not canonical JSON")
    return value


def _unique_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"duplicate manifest key: {key!r}")
        result[key] = value
    return result


def _verify_zip_info(info: zipfile.ZipInfo, record: FileRecord) -> None:
    if info.is_dir() or info.compress_type != zipfile.ZIP_DEFLATED:
        raise BundleError(f"archive entry type or compression is invalid: {record.path}")
    if info.date_time != ZIP_FIXED_DATETIME:
        raise BundleError(f"archive timestamp is not normalized: {record.path}")
    expected_extra = _expected_zip_extra(info)
    expected_version = zipfile.ZIP64_VERSION if expected_extra else 20
    if (
        info.create_system != 3
        or info.create_version != expected_version
        or info.extract_version != expected_version
        or info.extra != expected_extra
        or info.comment
    ):
        raise BundleError(f"archive metadata is not normalized: {record.path}")
    archived_mode = info.external_attr >> 16
    if not stat.S_ISREG(archived_mode) or f"{stat.S_IMODE(archived_mode):04o}" != record.mode:
        raise BundleError(f"archive mode is not normalized: {record.path}")
    if info.file_size != record.size:
        raise BundleError(f"archive metadata size does not match: {record.path}")


def _expected_zip_extra(info: zipfile.ZipInfo) -> bytes:
    """Return the one canonical ZIP64 extra that CPython emits when required."""

    values: list[int] = []
    if info.file_size > zipfile.ZIP64_LIMIT:
        values.append(info.file_size)
    if info.compress_size > zipfile.ZIP64_LIMIT:
        values.append(info.compress_size)
    if info.header_offset > zipfile.ZIP64_LIMIT:
        values.append(info.header_offset)
    if not values:
        return b""
    payload = struct.pack("<" + "Q" * len(values), *values)
    return struct.pack("<HH", 0x0001, len(payload)) + payload


def _safe_lstat(path: Path, where: str) -> os.stat_result:
    try:
        return os.lstat(path)
    except OSError as exc:
        raise SpecError(f"{where}.source cannot be inspected: {path}: {exc}") from exc


def _assert_regular_file(path: Path) -> None:
    metadata = _safe_lstat(path, "payload")
    if _is_forbidden(metadata) or not stat.S_ISREG(metadata.st_mode):
        raise SpecError(f"payload source changed or is no longer a regular file: {path}")


def _is_forbidden(metadata: os.stat_result) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & _REPARSE_FLAG)
