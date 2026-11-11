"""Deterministic notebook JSON and hard-coded base64 payload cells."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import urllib.parse
from dataclasses import dataclass
from typing import Any, Mapping

from ._version import __version__
from .bundle import (
    ARCHIVE_WARNING_BYTES,
    Bundle,
    COMPILER_CONTRACT_VERSION,
    verify_bundle,
)
from .errors import NotebookError
from .runtime import runtime_cell_sources

RAW_PAYLOAD_CHUNK_BYTES = 768 * 1024
PAYLOAD_LITERAL_LINE_CHARS = 8 * 1024
NOTEBOOK_FORMAT = 4
NOTEBOOK_FORMAT_MINOR = 4
IMPLEMENTED_CELL_GROUPS = 7


@dataclass(frozen=True, slots=True)
class NotebookArtifact:
    """Verified bytes for one complete self-contained job notebook."""

    bundle: Bundle
    payload_chunks: tuple[str, ...]
    notebook_bytes: bytes
    notebook_sha256: str

    @property
    def payload_cell_count(self) -> int:
        return len(self.payload_chunks)

    def verify(self) -> None:
        """Verify deterministic JSON and complete archive reconstruction from cells."""

        verify_notebook(self)


def encode_archive_chunks(
    archive_bytes: bytes, *, raw_chunk_bytes: int = RAW_PAYLOAD_CHUNK_BYTES
) -> tuple[str, ...]:
    """Encode complete raw-archive slices as concatenable standard-base64 strings."""

    if not archive_bytes:
        raise NotebookError("archive bytes must not be empty")
    if raw_chunk_bytes <= 0 or raw_chunk_bytes % 3:
        raise NotebookError("raw payload chunk size must be a positive multiple of 3")
    return tuple(
        base64.b64encode(archive_bytes[start : start + raw_chunk_bytes]).decode("ascii")
        for start in range(0, len(archive_bytes), raw_chunk_bytes)
    )


def build_notebook(bundle: Bundle) -> NotebookArtifact:
    """Build deterministic notebook bytes containing all seven cell groups."""

    verify_bundle(bundle)
    chunks = encode_archive_chunks(bundle.archive_bytes)
    notebook_bytes = _stable_notebook_json_bytes(_notebook_document(bundle, chunks))
    artifact = NotebookArtifact(
        bundle=bundle,
        payload_chunks=chunks,
        notebook_bytes=notebook_bytes,
        notebook_sha256=hashlib.sha256(notebook_bytes).hexdigest(),
    )
    verify_notebook(artifact)
    return artifact


def verify_notebook(artifact: NotebookArtifact) -> None:
    """Reject changed notebook JSON, metadata, chunk text, or reconstructed bytes."""

    verify_bundle(artifact.bundle)
    expected_chunks = encode_archive_chunks(artifact.bundle.archive_bytes)
    if artifact.payload_chunks != expected_chunks:
        raise NotebookError("payload chunks do not match the bundle archive")

    try:
        reconstructed = b"".join(
            base64.b64decode(chunk, validate=True) for chunk in artifact.payload_chunks
        )
    except (ValueError, binascii.Error) as exc:
        raise NotebookError(f"payload chunk is not valid standard base64: {exc}") from exc
    if reconstructed != artifact.bundle.archive_bytes:
        raise NotebookError("decoded payload chunks do not reconstruct the archive")

    expected_bytes = _stable_notebook_json_bytes(
        _notebook_document(artifact.bundle, artifact.payload_chunks)
    )
    if artifact.notebook_bytes != expected_bytes:
        raise NotebookError("notebook bytes do not match the deterministic notebook document")
    if hashlib.sha256(artifact.notebook_bytes).hexdigest() != artifact.notebook_sha256:
        raise NotebookError("notebook SHA-256 does not match")


def _notebook_document(bundle: Bundle, chunks: tuple[str, ...]) -> dict[str, object]:
    metadata: dict[str, object] = {
        "colabcola": {
            "archive_warning_bytes": ARCHIVE_WARNING_BYTES,
            "archive_sha256": bundle.archive_sha256,
            "compiler_contract_version": COMPILER_CONTRACT_VERSION,
            "compiler_version": __version__,
            "implemented_cell_groups": IMPLEMENTED_CELL_GROUPS,
            "job_id": bundle.job_id,
            "manifest_sha256": bundle.manifest_sha256,
            "payload_cell_count": len(chunks),
            "payload_display": "hidden_form",
            "payload_literal_line_chars": PAYLOAD_LITERAL_LINE_CHARS,
            "warnings": list(bundle.warnings),
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    manifest = json.loads(bundle.canonical_manifest_bytes)
    if manifest["resources"]["accelerator"] == "gpu":
        metadata["accelerator"] = "GPU"

    cells: list[dict[str, object]] = [
        _markdown_cell(_summary_markdown(bundle)),
        _code_cell(_manifest_cell_source(bundle), group="manifest"),
    ]
    for index, chunk in enumerate(chunks):
        raw_size = min(
            RAW_PAYLOAD_CHUNK_BYTES,
            len(bundle.archive_bytes) - index * RAW_PAYLOAD_CHUNK_BYTES,
        )
        cells.append(_payload_cell(chunk, index=index, total=len(chunks), raw_size=raw_size))
    cells.extend(
        _code_cell(source, group=group)
        for group, source in runtime_cell_sources(
            isolated=bool(manifest["python"].get("isolation")),
            remote_inputs=bool(manifest.get("remote_inputs")),
        )
    )

    return {
        "cells": cells,
        "metadata": metadata,
        "nbformat": NOTEBOOK_FORMAT,
        "nbformat_minor": NOTEBOOK_FORMAT_MINOR,
    }


def remote_inputs_summary(manifest: Mapping[str, Any]) -> tuple[str, str]:
    """Return a warning paragraph and a bullet line for declared remote inputs, or two empty strings."""

    entries = manifest.get("remote_inputs") or []
    if not entries:
        return "", ""
    hosts = sorted(
        {urllib.parse.urlsplit(str(entry["url"])).hostname or "?" for entry in entries}
    )
    host_text = ", ".join(f"`{host}`" for host in hosts)
    total = sum(int(entry["bytes"]) for entry in entries)
    plural = "s" if len(entries) != 1 else ""
    warning = (
        f"\n\n> **Remote inputs:** This job downloads {len(entries)} declared file{plural} over "
        f"HTTPS from {host_text} and verifies each against its declared byte count and SHA-256 "
        "before anything runs. Nothing undeclared is fetched."
    )
    bullet = f"\n- Remote inputs: {len(entries)} file{plural}, {total} bytes from {host_text}"
    return warning, bullet


def _summary_markdown(bundle: Bundle) -> str:
    manifest = json.loads(bundle.canonical_manifest_bytes)
    accelerator = manifest["resources"]["accelerator"]
    size_warnings = "".join(
        f"\n\n> **Size warning:** {message}" for message in bundle.warnings
    )
    remote_warning, remote_bullet = remote_inputs_summary(manifest)
    return (
        f"# colabcola job: {bundle.name}\n\n"
        "> **Security warning:** This notebook is designed to reconstruct and execute "
        "trusted bundled code. Inspect the manifest and payload provenance before running it.\n\n"
        "> **Payload visibility warning:** Keep every embedded payload cell hidden. Showing or "
        "editing the generated base64 source can make the Colab interface slow or unresponsive. "
        "Use **Run all**; do not expand the payload cells."
        f"{remote_warning}\n\n"
        f"- Job ID: `{bundle.job_id}`\n"
        f"- Requested accelerator: `{accelerator}`\n"
        f"- Payload: {len(bundle.files)} files, {bundle.raw_input_bytes} raw bytes, "
        f"{len(bundle.archive_bytes)} ZIP bytes"
        f"{remote_bullet}"
        f"{size_warnings}\n\n"
        "Run all cells to verify the runtime, reconstruct the payload beneath `/content`, "
        "execute without a shell, and collect durable result evidence."
    )


def _manifest_cell_source(bundle: Bundle) -> str:
    manifest_text = bundle.canonical_manifest_bytes.decode("utf-8")
    manifest_literal = json.dumps(manifest_text, ensure_ascii=False)
    return (
        "# Canonical bundle manifest and deterministic constants.\n"
        "import base64\n"
        "import hashlib\n"
        "import io\n"
        "import json\n"
        "import os\n"
        "import stat\n"
        "import sys\n"
        "import zipfile\n\n"
        f"COLABCOLA_COMPILER_VERSION = {json.dumps(__version__)}\n"
        f"COLABCOLA_CONTRACT_VERSION = {json.dumps(COMPILER_CONTRACT_VERSION)}\n"
        'COLABCOLA_TRANSPORT_MODE = "embedded"\n'
        f"COLABCOLA_JOB_ID = {json.dumps(bundle.job_id)}\n"
        f"COLABCOLA_MANIFEST_SHA256 = {json.dumps(bundle.manifest_sha256)}\n"
        f"COLABCOLA_ARCHIVE_SHA256 = {json.dumps(bundle.archive_sha256)}\n"
        f"COLABCOLA_ARCHIVE_BYTES = {len(bundle.archive_bytes)}\n"
        f"COLABCOLA_RAW_CHUNK_BYTES = {RAW_PAYLOAD_CHUNK_BYTES}\n"
        f"COLABCOLA_MANIFEST_JSON = {manifest_literal}\n"
        "COLABCOLA_MANIFEST = json.loads(COLABCOLA_MANIFEST_JSON)\n"
        "COLABCOLA_PAYLOAD_CHUNKS = []"
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


def _payload_cell(chunk: str, *, index: int, total: int, raw_size: int) -> dict[str, object]:
    literal_lines = "\n".join(
        f"    {json.dumps(chunk[start : start + PAYLOAD_LITERAL_LINE_CHARS])}"
        for start in range(0, len(chunk), PAYLOAD_LITERAL_LINE_CHARS)
    )
    source = (
        f"# @title WARNING: Embedded payload {index + 1}/{total} - keep hidden "
        "{ display-mode: \"form\" }\n"
        "# Do not reveal or edit this generated base64 payload; showing it can freeze Colab.\n"
        f"# Chunk {index} contains {raw_size} raw archive bytes.\n"
        "COLABCOLA_PAYLOAD_CHUNKS.append(\n"
        f"{literal_lines}\n"
        ")"
    )
    cell = _code_cell(source, group="payload")
    cell["metadata"] = {
        "cellView": "form",
        "colabcola": {
            "display": "hidden_form",
            "group": "payload",
            "index": index,
            "literal_line_chars": PAYLOAD_LITERAL_LINE_CHARS,
            "raw_bytes": raw_size,
        }
    }
    return cell


def _stable_notebook_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise NotebookError(f"notebook cannot be encoded as stable JSON: {exc}") from exc
    return text.encode("utf-8") + b"\n"
