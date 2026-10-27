from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from colabcola import (
    DriveControllerAuth,
    DriveTransportError,
    JobSpec,
    NotebookError,
    compile_drive_controller,
    compile_notebook,
    drive_receipt_for_observed_upload,
    prepare_drive_bundle,
)

from test_notebook import notebook_job


class CompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "main.py"
        self.source.write_text("print('ok')\n", encoding="utf-8")
        self.spec = JobSpec.from_mapping(notebook_job(self.source), base_dir=self.root)

    def test_public_api_atomically_writes_verified_repeatable_notebook(self) -> None:
        output = self.root / "nested" / "job.ipynb"
        first = compile_notebook(self.spec, output, job_args=("--extra", "snowman-☃"))
        first_bytes = output.read_bytes()
        second = compile_notebook(self.spec, output, job_args=("--extra", "snowman-☃"))

        self.assertEqual(first.output_path, output.resolve())
        self.assertEqual(first.notebook_bytes, first_bytes)
        self.assertEqual(second.notebook_bytes, first_bytes)
        self.assertEqual(first.job_id, second.job_id)
        self.assertEqual(first.notebook_sha256, hashlib.sha256(first_bytes).hexdigest())
        self.assertEqual(first.warnings, ())
        self.assertFalse(any(path.suffix == ".tmp" for path in output.parent.iterdir()))

    def test_public_api_rejects_non_notebook_output(self) -> None:
        with self.assertRaisesRegex(NotebookError, r"must use the \.ipynb extension"):
            compile_notebook(self.spec, self.root / "job.json")
        self.assertFalse((self.root / "job.json").exists())

    def test_drive_public_api_writes_bundle_then_bound_controller(self) -> None:
        bundle_path = self.root / "nested" / "job.colabcola"
        prepared = prepare_drive_bundle(self.spec, bundle_path, job_args=("--drive",))
        receipt = drive_receipt_for_observed_upload(
            prepared.artifact,
            remote_file_id="payload_abc",
            remote_parent_id="folder_abc",
            remote_relative_path=(
                f"colabcola/staging/{prepared.job_id}/{prepared.artifact.file_name}"
            ),
        )
        controller_path = self.root / "nested" / "controller.ipynb"
        auth = DriveControllerAuth(
            google_cloud_project_id="colabcola-fixture",
            oauth_client_id="controller-fixture.apps.googleusercontent.com",
            colab_secret_name="COLABCOLA_GOOGLE_DRIVE_OAUTH",
        )
        controller = compile_drive_controller(
            self.spec,
            receipt,
            auth,
            controller_path,
            job_args=("--drive",),
        )

        self.assertEqual(prepared.staged_bytes, bundle_path.stat().st_size)
        self.assertEqual(controller.output_path, controller_path.resolve())
        self.assertEqual(controller.job_id, prepared.job_id)
        self.assertLess(len(controller.notebook_bytes), 100_000)

    def test_drive_public_api_enforces_output_extensions(self) -> None:
        with self.assertRaisesRegex(DriveTransportError, r"must use \.colabcola"):
            prepare_drive_bundle(self.spec, self.root / "job.zip")


if __name__ == "__main__":
    unittest.main()
