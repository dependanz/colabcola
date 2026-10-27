from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from colabcola import (
    DriveFile,
    DriveFolder,
    DriveTransportError,
    JobSpec,
    authorize_abandoned_cleanup,
    bind_job_arguments,
    build_bundle,
    build_staged_bundle,
    cleanup_drive_staging,
    drive_receipt_json_bytes,
    load_drive_receipt,
    record_verified_result,
    stage_drive_bundle,
    write_drive_receipt,
    write_staged_bundle,
)
from colabcola.drive import DRIVE_BINARY_MIME_TYPE

from test_notebook import notebook_job


class FakeDriveBackend:
    def __init__(self) -> None:
        self.folders: dict[tuple[str, str], DriveFolder] = {}
        self.matches: tuple[DriveFile, ...] = ()
        self.uploads: list[Path] = []
        self.trashed: list[str] = []

    def ensure_folder(self, *, name: str, parent_id: str) -> DriveFolder:
        key = (parent_id, name)
        if key not in self.folders:
            self.folders[key] = DriveFolder(
                file_id=f"folder_{len(self.folders) + 1}", name=name, parent_id=parent_id
            )
        return self.folders[key]

    def find_payload(
        self,
        *,
        name: str,
        parent_id: str,
        app_properties: Mapping[str, str],
    ) -> Sequence[DriveFile]:
        return self.matches

    def upload_resumable(
        self,
        *,
        local_path: Path,
        name: str,
        parent_id: str,
        mime_type: str,
        app_properties: Mapping[str, str],
    ) -> DriveFile:
        self.uploads.append(local_path)
        return DriveFile(
            file_id="payload_1",
            name=name,
            parent_id=parent_id,
            size=local_path.stat().st_size,
            mime_type=mime_type,
            app_properties=tuple(sorted(app_properties.items())),
        )

    def trash(self, *, file_id: str) -> None:
        self.trashed.append(file_id)


class DriveLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = self.root / "main.py"
        source.write_text("print('ok')\n", encoding="utf-8")
        spec = JobSpec.from_mapping(notebook_job(source), base_dir=self.root)
        self.artifact = build_staged_bundle(build_bundle(bind_job_arguments(spec)))
        self.path = self.root / self.artifact.file_name
        write_staged_bundle(self.artifact, self.path)
        self.backend = FakeDriveBackend()

    def test_absent_exact_object_uploads_once_and_receipt_is_canonical(self) -> None:
        receipt = stage_drive_bundle(self.artifact, self.path, self.backend)

        self.assertEqual(self.backend.uploads, [self.path.resolve()])
        self.assertFalse(receipt.reused)
        self.assertEqual(receipt.remote_file_id, "payload_1")
        self.assertEqual(receipt.cleanup_file_ids, ("payload_1",))
        self.assertIn(self.artifact.bundle.job_id.encode(), drive_receipt_json_bytes(receipt))

    def test_one_exact_match_is_reused_without_upload(self) -> None:
        first = stage_drive_bundle(self.artifact, self.path, self.backend)
        job_parent = first.remote_parent_id
        properties = (
            ("colabcola_job_id", self.artifact.bundle.job_id),
            ("colabcola_staged_sha256", self.artifact.staged_sha256),
        )
        self.backend.uploads.clear()
        self.backend.matches = (
            DriveFile(
                file_id="existing_payload",
                name=self.artifact.file_name,
                parent_id=job_parent,
                size=self.artifact.staged_bytes,
                mime_type=DRIVE_BINARY_MIME_TYPE,
                app_properties=properties,
            ),
        )

        reused = stage_drive_bundle(self.artifact, self.path, self.backend)
        self.assertTrue(reused.reused)
        self.assertEqual(reused.remote_file_id, "existing_payload")
        self.assertEqual(self.backend.uploads, [])

    def test_ambiguous_or_changed_remote_identity_is_rejected(self) -> None:
        valid = self.backend.upload_resumable(
            local_path=self.path,
            name=self.artifact.file_name,
            parent_id="folder_3",
            mime_type=DRIVE_BINARY_MIME_TYPE,
            app_properties={
                "colabcola_job_id": self.artifact.bundle.job_id,
                "colabcola_staged_sha256": self.artifact.staged_sha256,
            },
        )
        self.backend.matches = (valid, valid)
        with self.assertRaisesRegex(DriveTransportError, "ambiguous"):
            stage_drive_bundle(self.artifact, self.path, self.backend)

        self.backend.matches = (replace(valid, size=valid.size + 1),)
        with self.assertRaisesRegex(DriveTransportError, "byte count"):
            stage_drive_bundle(self.artifact, self.path, self.backend)

    def test_cleanup_requires_verified_result_or_explicit_abandonment(self) -> None:
        receipt = stage_drive_bundle(self.artifact, self.path, self.backend)
        with self.assertRaisesRegex(DriveTransportError, "verified results"):
            cleanup_drive_staging(receipt, self.backend)

        result = self.root / "artifacts.zip"
        result.write_bytes(b"verified result")
        verified = record_verified_result(
            receipt,
            result,
            expected_size=result.stat().st_size,
            expected_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),
        )
        cleaned = cleanup_drive_staging(verified, self.backend)
        self.assertEqual(cleaned.state, "cleaned")
        self.assertEqual(self.backend.trashed, [receipt.remote_file_id])
        self.assertEqual(cleanup_drive_staging(cleaned, self.backend), cleaned)
        self.assertEqual(self.backend.trashed, [receipt.remote_file_id])

        second = stage_drive_bundle(self.artifact, self.path, FakeDriveBackend())
        authorized = authorize_abandoned_cleanup(second)
        self.assertEqual(authorized.state, "cleanup_authorized")

    def test_persisted_receipt_round_trips_and_noncanonical_json_is_rejected(self) -> None:
        receipt = stage_drive_bundle(self.artifact, self.path, self.backend)
        path = self.root / "receipt.json"
        write_drive_receipt(receipt, path)
        self.assertEqual(load_drive_receipt(path), receipt)

        path.write_text(
            __import__("json").dumps(receipt.as_value(), indent=2), encoding="utf-8"
        )
        with self.assertRaisesRegex(DriveTransportError, "canonical persisted form"):
            load_drive_receipt(path)


if __name__ == "__main__":
    unittest.main()
