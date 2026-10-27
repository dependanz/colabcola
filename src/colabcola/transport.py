"""Deterministic binary staging envelope for provider transports."""

from __future__ import annotations

import hashlib
import os
import stat
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .bundle import Bundle, verify_bundle
from .errors import DriveTransportError


STAGED_BUNDLE_SUFFIX = ".colabcola"
STAGED_BUNDLE_MAGIC = b"COLABCOLA\x00BUNDLE"
STAGED_BUNDLE_HEADER_BYTES = 32
STAGED_COPY_BUFFER_BYTES = 8 * 1024 * 1024

_HEADER = struct.Struct(">16sQQ")
_REPARSE_FLAG = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


@dataclass(frozen=True, slots=True)
class StagedBundleArtifact:
    """One complete manifest-plus-archive file prepared for remote staging."""

    bundle: Bundle
    header_bytes: bytes
    staged_bytes: int
    staged_sha256: str

    @property
    def file_name(self) -> str:
        return f"{self.bundle.job_id}{STAGED_BUNDLE_SUFFIX}"

    def segments(self) -> Iterator[bytes]:
        yield self.header_bytes
        yield self.bundle.canonical_manifest_bytes
        yield self.bundle.archive_bytes

    def verify(self) -> None:
        verify_staged_bundle(self)


def build_staged_bundle(bundle: Bundle) -> StagedBundleArtifact:
    """Frame a verified bundle without base64 or another archive copy."""

    verify_bundle(bundle)
    header = _HEADER.pack(
        STAGED_BUNDLE_MAGIC,
        len(bundle.canonical_manifest_bytes),
        len(bundle.archive_bytes),
    )
    digest = hashlib.sha256()
    for segment in (header, bundle.canonical_manifest_bytes, bundle.archive_bytes):
        digest.update(segment)
    artifact = StagedBundleArtifact(
        bundle=bundle,
        header_bytes=header,
        staged_bytes=(
            len(header)
            + len(bundle.canonical_manifest_bytes)
            + len(bundle.archive_bytes)
        ),
        staged_sha256=digest.hexdigest(),
    )
    verify_staged_bundle(artifact)
    return artifact


def verify_staged_bundle(artifact: StagedBundleArtifact) -> None:
    """Reject changed framing, lengths, identity, or staged-file hash."""

    verify_bundle(artifact.bundle)
    expected_header = _HEADER.pack(
        STAGED_BUNDLE_MAGIC,
        len(artifact.bundle.canonical_manifest_bytes),
        len(artifact.bundle.archive_bytes),
    )
    if artifact.header_bytes != expected_header:
        raise DriveTransportError("staged bundle header does not match the bundle lengths")
    expected_bytes = sum(
        len(segment)
        for segment in (
            expected_header,
            artifact.bundle.canonical_manifest_bytes,
            artifact.bundle.archive_bytes,
        )
    )
    if artifact.staged_bytes != expected_bytes:
        raise DriveTransportError("staged bundle byte count does not match")
    digest = hashlib.sha256()
    for segment in artifact.segments():
        digest.update(segment)
    if digest.hexdigest() != artifact.staged_sha256:
        raise DriveTransportError("staged bundle SHA-256 does not match")


def write_staged_bundle(
    artifact: StagedBundleArtifact, output_path: Path | str
) -> Path:
    """Atomically write a verified staged bundle without concatenating it in memory."""

    verify_staged_bundle(artifact)
    destination = Path(output_path)
    if destination.suffix.lower() != STAGED_BUNDLE_SUFFIX:
        raise DriveTransportError(
            f"staged bundle output must use {STAGED_BUNDLE_SUFFIX}: {destination}"
        )
    if destination.exists() and destination.is_dir():
        raise DriveTransportError(f"staged bundle output is a directory: {destination}")
    if destination.is_symlink():
        raise DriveTransportError(f"staged bundle output may not be a symlink: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            for segment in artifact.segments():
                stream.write(segment)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    finally:
        temporary.unlink(missing_ok=True)

    verify_staged_bundle_file(destination, artifact)
    return destination.resolve(strict=True)


def verify_staged_bundle_file(
    path: Path | str, artifact: StagedBundleArtifact
) -> Path:
    """Stream-verify a persisted envelope against its compiler-known artifact."""

    verify_staged_bundle(artifact)
    source = Path(path)
    metadata = os.lstat(source)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or getattr(metadata, "st_file_attributes", 0) & _REPARSE_FLAG
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise DriveTransportError(f"staged bundle file is unavailable or unsafe: {source}")
    if metadata.st_size != artifact.staged_bytes:
        raise DriveTransportError("persisted staged bundle byte count does not match")

    whole_digest = hashlib.sha256()
    archive_digest = hashlib.sha256()
    with source.open("rb") as stream:
        header = stream.read(STAGED_BUNDLE_HEADER_BYTES)
        whole_digest.update(header)
        if header != artifact.header_bytes:
            raise DriveTransportError("persisted staged bundle header does not match")

        manifest = stream.read(len(artifact.bundle.canonical_manifest_bytes))
        whole_digest.update(manifest)
        if manifest != artifact.bundle.canonical_manifest_bytes:
            raise DriveTransportError("persisted staged bundle manifest does not match")

        remaining = len(artifact.bundle.archive_bytes)
        while remaining:
            block = stream.read(min(STAGED_COPY_BUFFER_BYTES, remaining))
            if not block:
                raise DriveTransportError("persisted staged bundle archive is truncated")
            whole_digest.update(block)
            archive_digest.update(block)
            remaining -= len(block)
        if stream.read(1):
            raise DriveTransportError("persisted staged bundle has trailing bytes")

    if archive_digest.hexdigest() != artifact.bundle.archive_sha256:
        raise DriveTransportError("persisted staged archive SHA-256 does not match")
    if whole_digest.hexdigest() != artifact.staged_sha256:
        raise DriveTransportError("persisted staged bundle SHA-256 does not match")
    return source.resolve(strict=True)
