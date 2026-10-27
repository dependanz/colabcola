from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from colabcola import (
    DriveTransportError,
    JobSpec,
    bind_job_arguments,
    build_bundle,
    build_staged_bundle,
    verify_staged_bundle,
    verify_staged_bundle_file,
    write_staged_bundle,
)
from colabcola.transport import STAGED_BUNDLE_HEADER_BYTES, STAGED_BUNDLE_MAGIC

from test_notebook import notebook_job


class StagedBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "main.py"
        self.source.write_text("print('drive transport')\n", encoding="utf-8")
        spec = JobSpec.from_mapping(notebook_job(self.source), base_dir=self.root)
        self.bundle = build_bundle(bind_job_arguments(spec, ("snowman-☃",)))
        self.artifact = build_staged_bundle(self.bundle)

    def test_frame_is_exact_deterministic_and_contains_no_base64_copy(self) -> None:
        second = build_staged_bundle(self.bundle)
        content = b"".join(self.artifact.segments())

        self.assertEqual(self.artifact, second)
        self.assertEqual(content[:16], STAGED_BUNDLE_MAGIC)
        self.assertEqual(
            self.artifact.staged_bytes,
            STAGED_BUNDLE_HEADER_BYTES
            + len(self.bundle.canonical_manifest_bytes)
            + len(self.bundle.archive_bytes),
        )
        self.assertEqual(self.artifact.staged_sha256, hashlib.sha256(content).hexdigest())
        self.assertEqual(content[-len(self.bundle.archive_bytes) :], self.bundle.archive_bytes)
        self.assertNotIn(__import__("base64").b64encode(self.bundle.archive_bytes), content)

    def test_atomic_write_and_stream_verification(self) -> None:
        destination = self.root / "nested" / self.artifact.file_name
        resolved = write_staged_bundle(self.artifact, destination)

        self.assertEqual(resolved, destination.resolve())
        self.assertEqual(destination.stat().st_size, self.artifact.staged_bytes)
        self.assertEqual(verify_staged_bundle_file(destination, self.artifact), resolved)
        self.assertFalse(any(path.suffix == ".tmp" for path in destination.parent.iterdir()))

    def test_changed_metadata_and_persisted_bytes_are_rejected(self) -> None:
        with self.assertRaisesRegex(DriveTransportError, "byte count"):
            verify_staged_bundle(replace(self.artifact, staged_bytes=1))

        destination = self.root / self.artifact.file_name
        write_staged_bundle(self.artifact, destination)
        changed = bytearray(destination.read_bytes())
        changed[-1] ^= 1
        destination.write_bytes(changed)
        with self.assertRaisesRegex(DriveTransportError, "archive SHA-256"):
            verify_staged_bundle_file(destination, self.artifact)

    def test_wrong_output_extension_is_rejected(self) -> None:
        with self.assertRaisesRegex(DriveTransportError, r"must use \.colabcola"):
            write_staged_bundle(self.artifact, self.root / "job.zip")


if __name__ == "__main__":
    unittest.main()
