"""High-level deterministic compilation and atomic notebook output."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .bundle import build_bundle
from .controller import (
    DriveControllerAuth,
    DriveControllerArtifact,
    build_drive_controller,
    write_drive_controller,
)
from .drive import DriveStagingReceipt
from .errors import NotebookError
from .invocation import JobInvocation, bind_job_arguments
from .notebook import NotebookArtifact, build_notebook
from .spec import JobSpec
from .transport import (
    StagedBundleArtifact,
    build_staged_bundle,
    write_staged_bundle,
)


@dataclass(frozen=True, slots=True)
class CompilationResult:
    """One verified notebook artifact persisted at an explicit path."""

    artifact: NotebookArtifact
    output_path: Path

    @property
    def job_id(self) -> str:
        return self.artifact.bundle.job_id

    @property
    def notebook_sha256(self) -> str:
        return self.artifact.notebook_sha256

    @property
    def notebook_bytes(self) -> bytes:
        return self.artifact.notebook_bytes

    @property
    def warnings(self) -> tuple[str, ...]:
        return self.artifact.bundle.warnings


@dataclass(frozen=True, slots=True)
class StagedBundleResult:
    """One verified raw bundle persisted for Drive upload."""

    artifact: StagedBundleArtifact
    output_path: Path

    @property
    def job_id(self) -> str:
        return self.artifact.bundle.job_id

    @property
    def staged_bytes(self) -> int:
        return self.artifact.staged_bytes

    @property
    def staged_sha256(self) -> str:
        return self.artifact.staged_sha256

    @property
    def warnings(self) -> tuple[str, ...]:
        return self.artifact.bundle.warnings


@dataclass(frozen=True, slots=True)
class DriveControllerResult:
    """One verified Drive controller persisted after remote object identity is known."""

    artifact: DriveControllerArtifact
    output_path: Path

    @property
    def job_id(self) -> str:
        return self.artifact.bundle.job_id

    @property
    def notebook_sha256(self) -> str:
        return self.artifact.notebook_sha256

    @property
    def notebook_bytes(self) -> bytes:
        return self.artifact.notebook_bytes


def write_notebook(artifact: NotebookArtifact, output_path: Path | str) -> Path:
    """Verify and atomically replace one ``.ipynb`` destination."""

    artifact.verify()
    destination = Path(output_path)
    if destination.suffix.lower() != ".ipynb":
        raise NotebookError(f"notebook output must use the .ipynb extension: {destination}")
    if destination.exists() and destination.is_dir():
        raise NotebookError(f"notebook output is a directory: {destination}")
    if destination.is_symlink():
        raise NotebookError(f"notebook output may not be a symlink: {destination}")

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(artifact.notebook_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return destination.resolve(strict=True)


def compile_invocation(
    invocation: JobInvocation, output_path: Path | str
) -> CompilationResult:
    """Compile a bound invocation and atomically persist its verified notebook."""

    artifact = build_notebook(build_bundle(invocation))
    persisted = write_notebook(artifact, output_path)
    return CompilationResult(artifact=artifact, output_path=persisted)


def compile_notebook(
    spec: JobSpec,
    output_path: Path | str,
    *,
    job_args: Sequence[str] = (),
) -> CompilationResult:
    """Compile a JobSpec through the public dependency-free Python API."""

    return compile_invocation(bind_job_arguments(spec, job_args), output_path)


def prepare_drive_invocation(
    invocation: JobInvocation, output_path: Path | str
) -> StagedBundleResult:
    """Compile and persist the deterministic binary artifact used for Drive staging."""

    artifact = build_staged_bundle(build_bundle(invocation))
    persisted = write_staged_bundle(artifact, output_path)
    return StagedBundleResult(artifact=artifact, output_path=persisted)


def prepare_drive_bundle(
    spec: JobSpec,
    output_path: Path | str,
    *,
    job_args: Sequence[str] = (),
) -> StagedBundleResult:
    """Prepare a Drive-staged bundle through the public dependency-free Python API."""

    return prepare_drive_invocation(bind_job_arguments(spec, job_args), output_path)


def compile_drive_controller_invocation(
    invocation: JobInvocation,
    receipt: DriveStagingReceipt,
    auth: DriveControllerAuth,
    output_path: Path | str,
) -> DriveControllerResult:
    """Rebuild the exact bundle and bind a controller to an observed Drive receipt."""

    staged = build_staged_bundle(build_bundle(invocation))
    artifact = build_drive_controller(staged, receipt, auth)
    persisted = write_drive_controller(artifact, output_path)
    return DriveControllerResult(artifact=artifact, output_path=persisted)


def compile_drive_controller(
    spec: JobSpec,
    receipt: DriveStagingReceipt,
    auth: DriveControllerAuth,
    output_path: Path | str,
    *,
    job_args: Sequence[str] = (),
) -> DriveControllerResult:
    """Compile a tiny controller through the public dependency-free Python API."""

    return compile_drive_controller_invocation(
        bind_job_arguments(spec, job_args), receipt, auth, output_path
    )
