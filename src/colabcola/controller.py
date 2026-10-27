"""Deterministic tiny notebook for a verified Google Drive staged bundle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ._version import __version__
from .bundle import COMPILER_CONTRACT_VERSION, Bundle, verify_bundle
from .drive import DriveStagingReceipt, verify_drive_receipt
from .errors import DriveTransportError
from .google_drive import GOOGLE_DRIVE_SCOPE, GoogleDriveSettings
from .notebook import remote_inputs_summary
from .runtime import drive_runtime_cell_sources
from .transport import (
    STAGED_BUNDLE_HEADER_BYTES,
    STAGED_BUNDLE_MAGIC,
    StagedBundleArtifact,
    verify_staged_bundle,
)


CONTROLLER_NOTEBOOK_FORMAT = 4
CONTROLLER_NOTEBOOK_FORMAT_MINOR = 4
CONTROLLER_CELL_GROUPS = 7


@dataclass(frozen=True, slots=True)
class DriveControllerAuth:
    """Credential-free OAuth identity expected from the user's Colab Secret."""

    google_cloud_project_id: str
    oauth_client_id: str
    colab_secret_name: str


def controller_auth_from_settings(settings: GoogleDriveSettings) -> DriveControllerAuth:
    return DriveControllerAuth(
        google_cloud_project_id=settings.google_cloud_project_id,
        oauth_client_id=settings.colab_oauth_client_id,
        colab_secret_name=settings.colab_secret_name,
    )


@dataclass(frozen=True, slots=True)
class DriveControllerArtifact:
    """One deterministic controller bound to one exact remote Drive object."""

    bundle: Bundle
    staged_bundle: StagedBundleArtifact
    receipt: DriveStagingReceipt
    auth: DriveControllerAuth
    notebook_bytes: bytes
    notebook_sha256: str

    def verify(self) -> None:
        verify_drive_controller(self)


def build_drive_controller(
    staged_bundle: StagedBundleArtifact,
    receipt: DriveStagingReceipt,
    auth: DriveControllerAuth,
) -> DriveControllerArtifact:
    """Build a small notebook containing metadata and runtime code, not payload bytes."""

    verify_staged_bundle(staged_bundle)
    verify_drive_receipt(receipt)
    _verify_auth(auth)
    _verify_binding(staged_bundle, receipt)
    notebook_bytes = _stable_json_bytes(_controller_document(staged_bundle, receipt, auth))
    artifact = DriveControllerArtifact(
        bundle=staged_bundle.bundle,
        staged_bundle=staged_bundle,
        receipt=receipt,
        auth=auth,
        notebook_bytes=notebook_bytes,
        notebook_sha256=hashlib.sha256(notebook_bytes).hexdigest(),
    )
    verify_drive_controller(artifact)
    return artifact


def verify_drive_controller(artifact: DriveControllerArtifact) -> None:
    """Reject changed controller JSON, hashes, or staged-object binding."""

    verify_bundle(artifact.bundle)
    verify_staged_bundle(artifact.staged_bundle)
    verify_drive_receipt(artifact.receipt)
    _verify_auth(artifact.auth)
    if artifact.staged_bundle.bundle != artifact.bundle:
        raise DriveTransportError("controller bundle and staged bundle disagree")
    _verify_binding(artifact.staged_bundle, artifact.receipt)
    expected = _stable_json_bytes(
        _controller_document(artifact.staged_bundle, artifact.receipt, artifact.auth)
    )
    if artifact.notebook_bytes != expected:
        raise DriveTransportError(
            "controller bytes do not match the deterministic notebook document"
        )
    if hashlib.sha256(expected).hexdigest() != artifact.notebook_sha256:
        raise DriveTransportError("controller notebook SHA-256 does not match")


def write_drive_controller(
    artifact: DriveControllerArtifact, output_path: Path | str
) -> Path:
    """Atomically persist one verified controller notebook."""

    verify_drive_controller(artifact)
    destination = Path(output_path)
    if destination.suffix.lower() != ".ipynb":
        raise DriveTransportError(
            f"controller notebook output must use the .ipynb extension: {destination}"
        )
    if destination.exists() and destination.is_dir():
        raise DriveTransportError(f"controller output is a directory: {destination}")
    if destination.is_symlink():
        raise DriveTransportError(f"controller output may not be a symlink: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
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
    if destination.read_bytes() != artifact.notebook_bytes:
        raise DriveTransportError("persisted controller notebook bytes do not match")
    return destination.resolve(strict=True)


def _controller_document(
    staged: StagedBundleArtifact,
    receipt: DriveStagingReceipt,
    auth: DriveControllerAuth,
) -> dict[str, object]:
    manifest = json.loads(staged.bundle.canonical_manifest_bytes)
    metadata: dict[str, object] = {
        "colabcola": {
            "archive_sha256": staged.bundle.archive_sha256,
            "compiler_contract_version": COMPILER_CONTRACT_VERSION,
            "compiler_version": __version__,
            "drive_file_id": receipt.remote_file_id,
            "drive_oauth_scope": GOOGLE_DRIVE_SCOPE,
            "implemented_cell_groups": CONTROLLER_CELL_GROUPS,
            "job_id": staged.bundle.job_id,
            "manifest_sha256": staged.bundle.manifest_sha256,
            "staged_bytes": staged.staged_bytes,
            "staged_sha256": staged.staged_sha256,
            "transport": "google_drive",
            "runtime_provider": manifest["runtime"]["provider"],
            "runtime_version": manifest["runtime"]["version"],
            "warnings": list(staged.bundle.warnings),
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    if manifest["resources"]["accelerator"] == "gpu":
        metadata["accelerator"] = "GPU"

    cells: list[dict[str, object]] = [
        _markdown_cell(_summary_markdown(staged, receipt, auth)),
        _code_cell(_constants_source(staged, receipt, auth), group="manifest"),
    ]
    cells.extend(
        _code_cell(source, group=group)
        for group, source in drive_runtime_cell_sources(isolated=bool(manifest["python"].get("isolation")))
    )
    return {
        "cells": cells,
        "metadata": metadata,
        "nbformat": CONTROLLER_NOTEBOOK_FORMAT,
        "nbformat_minor": CONTROLLER_NOTEBOOK_FORMAT_MINOR,
    }


def _summary_markdown(
    staged: StagedBundleArtifact,
    receipt: DriveStagingReceipt,
    auth: DriveControllerAuth,
) -> str:
    manifest = json.loads(staged.bundle.canonical_manifest_bytes)
    warnings = "".join(
        f"\n\n> **Size warning:** {message}" for message in staged.bundle.warnings
    )
    remote_warning, remote_bullet = remote_inputs_summary(manifest)
    return (
        f"# colabcola Drive job: {staged.bundle.name}\n\n"
        "> **Security warning:** This controller obtains and executes a trusted staged bundle. "
        "Verify the job identity and payload provenance before running it.\n\n"
        "> **Drive staging:** This notebook contains no base64 payload and does not mount Drive. "
        "It requests only per-file `drive.file` access, downloads one immutable file ID into "
        "`/content`, and verifies every byte before extraction. Remote cleanup occurs only after "
        "results are recovered and verified."
        f"{remote_warning}\n\n"
        f"> **One-time setup:** Store the limited-input OAuth client JSON for project "
        f"`{auth.google_cloud_project_id}` in the Colab Secret "
        f"`{auth.colab_secret_name}` and grant this notebook access. No user token or client "
        "credential is embedded in this notebook.\n\n"
        f"- Job ID: `{staged.bundle.job_id}`\n"
        f"- Drive file ID: `{receipt.remote_file_id}`\n"
        f"- Staged file: `{receipt.remote_relative_path}`\n"
        f"- Required Colab runtime: `{manifest['runtime']['version']}`\n"
        f"- Requested accelerator: `{manifest['resources']['accelerator']}`\n"
        f"- Payload: {len(staged.bundle.files)} files, "
        f"{len(staged.bundle.archive_bytes)} ZIP bytes"
        f"{remote_bullet}\n"
        f"- Staged bundle: {staged.staged_bytes} bytes, SHA-256 "
        f"`{staged.staged_sha256}`"
        f"{warnings}\n\n"
        "Use **Run all** to verify the runtime, acquire the exact staged object, reconstruct the "
        "job beneath `/content`, execute without a shell, and collect durable result evidence."
    )


def _constants_source(
    staged: StagedBundleArtifact,
    receipt: DriveStagingReceipt,
    auth: DriveControllerAuth,
) -> str:
    manifest_text = staged.bundle.canonical_manifest_bytes.decode("utf-8")
    return (
        "# Canonical manifest and exact Google Drive staging identity.\n"
        "import hashlib\n"
        "import json\n"
        "import os\n"
        "import stat\n"
        "import sys\n"
        "import zipfile\n\n"
        f"COLABCOLA_COMPILER_VERSION = {json.dumps(__version__)}\n"
        f"COLABCOLA_CONTRACT_VERSION = {json.dumps(COMPILER_CONTRACT_VERSION)}\n"
        'COLABCOLA_TRANSPORT_MODE = "google_drive"\n'
        f"COLABCOLA_JOB_ID = {json.dumps(staged.bundle.job_id)}\n"
        f"COLABCOLA_MANIFEST_SHA256 = {json.dumps(staged.bundle.manifest_sha256)}\n"
        f"COLABCOLA_ARCHIVE_SHA256 = {json.dumps(staged.bundle.archive_sha256)}\n"
        f"COLABCOLA_ARCHIVE_BYTES = {len(staged.bundle.archive_bytes)}\n"
        f"COLABCOLA_STAGED_MAGIC_HEX = {json.dumps(STAGED_BUNDLE_MAGIC.hex())}\n"
        f"COLABCOLA_STAGED_HEADER_BYTES = {STAGED_BUNDLE_HEADER_BYTES}\n"
        f"COLABCOLA_STAGED_BYTES = {staged.staged_bytes}\n"
        f"COLABCOLA_STAGED_SHA256 = {json.dumps(staged.staged_sha256)}\n"
        f"COLABCOLA_DRIVE_FILE_ID = {json.dumps(receipt.remote_file_id)}\n"
        f"COLABCOLA_DRIVE_FILE_NAME = {json.dumps(receipt.remote_name)}\n"
        f"COLABCOLA_DRIVE_PARENT_ID = {json.dumps(receipt.remote_parent_id)}\n"
        f"COLABCOLA_DRIVE_RELATIVE_PATH = {json.dumps(receipt.remote_relative_path)}\n"
        f"COLABCOLA_GOOGLE_PROJECT_ID = {json.dumps(auth.google_cloud_project_id)}\n"
        f"COLABCOLA_GOOGLE_OAUTH_CLIENT_ID = {json.dumps(auth.oauth_client_id)}\n"
        f"COLABCOLA_GOOGLE_OAUTH_SCOPE = {json.dumps(GOOGLE_DRIVE_SCOPE)}\n"
        f"COLABCOLA_GOOGLE_OAUTH_SECRET_NAME = {json.dumps(auth.colab_secret_name)}\n"
        f"COLABCOLA_MANIFEST_JSON = {json.dumps(manifest_text, ensure_ascii=False)}\n"
        "COLABCOLA_MANIFEST = json.loads(COLABCOLA_MANIFEST_JSON)"
    )


def _markdown_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {"colabcola": {"group": "summary"}},
        "source": source,
    }


def _code_cell(source: str, *, group: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"colabcola": {"group": group}},
        "outputs": [],
        "source": source,
    }


def _stable_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise DriveTransportError(
            f"controller notebook cannot be encoded as stable JSON: {exc}"
        ) from exc
    return text.encode("utf-8") + b"\n"


def _verify_binding(
    staged: StagedBundleArtifact, receipt: DriveStagingReceipt
) -> None:
    if receipt.job_id != staged.bundle.job_id:
        raise DriveTransportError("Drive receipt job ID does not match the staged bundle")
    if receipt.staged_sha256 != staged.staged_sha256:
        raise DriveTransportError("Drive receipt SHA-256 does not match the staged bundle")
    if receipt.staged_bytes != staged.staged_bytes:
        raise DriveTransportError("Drive receipt byte count does not match the staged bundle")
    if receipt.remote_name != staged.file_name:
        raise DriveTransportError("Drive receipt filename does not match the staged bundle")


def _verify_auth(auth: DriveControllerAuth) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", auth.google_cloud_project_id):
        raise DriveTransportError("controller Google Cloud project ID is invalid")
    if not re.fullmatch(
        r"[A-Za-z0-9._-]+\.apps\.googleusercontent\.com", auth.oauth_client_id
    ):
        raise DriveTransportError("controller OAuth client ID is invalid")
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", auth.colab_secret_name):
        raise DriveTransportError("controller Colab secret name is invalid")
