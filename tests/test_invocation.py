from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

from colabcola import (
    GoogleDriveSettings,
    JobSpec,
    SpecError,
    bind_job_arguments,
    drive_receipt_for_observed_upload,
    load_job_invocation,
)
from colabcola.cli import SubmissionCommand, ValidateCommand, _parse_command, main


def valid_spec(source: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "fixture-job",
        "payload": [
            {
                "role": "source",
                "source": os.fspath(source),
                "target": "workload/main.py",
            }
        ],
        "entrypoint": {
            "kind": "python_script",
            "path": "workload/main.py",
            "argv": ["--config", "workload/config.json"],
            "cwd": ".",
        },
        "python": {
            "requires_python": ">=3.11,<3.13",
            "runtime_requirements": ["torch>=2.6"],
            "install_requirements": [],
            "embedded_wheels": [],
        },
        "runtime": {"provider": "google_colab", "version": "2026.07"},
        "resources": {
            "accelerator": "gpu",
            "gpu_count": 1,
            "cuda_required": True,
            "minimum_gpu_memory_bytes": None,
        },
        "run": {
            "timeout_seconds": 3600,
            "seed": 1234,
            "environment": {"COLABCOLA_MODE": "test"},
            "capture_stdout": True,
            "capture_stderr": True,
        },
        "outputs": [
            {
                "pattern": "outputs/**",
                "required": True,
                "maximum_bytes": 1024,
            }
        ],
        "limits": {
            "maximum_file_count": 100,
        },
    }


class InvocationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "main.py"
        self.source.write_text("print('ok')\n", encoding="utf-8")
        self.job_file = self.root / "fixture.colajob"
        self.job_file.write_text(
            json.dumps(valid_spec(self.source)), encoding="utf-8"
        )

    def test_load_requires_colajob_suffix(self) -> None:
        wrong = self.root / "fixture.json"
        wrong.write_text(self.job_file.read_text(encoding="utf-8"), encoding="utf-8")
        with self.assertRaisesRegex(SpecError, r"must use the \.colajob extension"):
            load_job_invocation(wrong)

    def test_binding_appends_arguments_without_reinterpretation(self) -> None:
        invocation = load_job_invocation(
            self.job_file,
            ("--epochs", "20", "--output", "remote", "", "snowman-☃"),
        )
        self.assertEqual(
            invocation.effective_argv,
            (
                "--config",
                "workload/config.json",
                "--epochs",
                "20",
                "--output",
                "remote",
                "",
                "snowman-☃",
            ),
        )

    def test_binding_is_immutable_and_rejects_invalid_arguments(self) -> None:
        spec = JobSpec.from_mapping(valid_spec(self.source), base_dir=self.root)
        invocation = bind_job_arguments(spec, ["--epochs", "20"])
        with self.assertRaises(FrozenInstanceError):
            invocation.job_args = ()  # type: ignore[misc]
        with self.assertRaisesRegex(SpecError, "not one string"):
            bind_job_arguments(spec, "--epochs")
        with self.assertRaisesRegex(SpecError, "argument 1 must be a string"):
            bind_job_arguments(spec, ["--epochs", 20])  # type: ignore[list-item]
        with self.assertRaisesRegex(SpecError, "forbidden NUL"):
            bind_job_arguments(spec, ["bad\x00argument"])

    def test_primary_parser_uses_job_path_as_argument_boundary(self) -> None:
        command = _parse_command(
            [
                "--output",
                "dist/local.colabcola",
                os.fspath(self.job_file),
                "--output",
                "remote-output",
                "--epochs",
                "20",
            ]
        )
        self.assertIsInstance(command, SubmissionCommand)
        assert isinstance(command, SubmissionCommand)
        self.assertEqual(command.output, Path("dist/local.colabcola"))
        self.assertEqual(command.transport, "drive")
        self.assertEqual(
            command.job_args,
            ("--output", "remote-output", "--epochs", "20"),
        )

    def test_validate_command_remains_explicit(self) -> None:
        command = _parse_command(["validate", os.fspath(self.job_file)])
        self.assertEqual(command, ValidateCommand(job_file=self.job_file))

    def test_google_drive_auth_command_reports_only_public_setup_identity(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        settings_path = self.root / "google-drive.json"
        settings = GoogleDriveSettings(
            schema_version=1,
            google_cloud_project_id="colabcola-fixture",
            desktop_client_secrets=self.root / "desktop-private.json",
            colab_oauth_client_id="controller-fixture.apps.googleusercontent.com",
            colab_secret_name="COLABCOLA_GOOGLE_DRIVE_OAUTH",
        )
        with (
            mock.patch(
                "colabcola.cli.google_drive_settings_from_clients", return_value=settings
            ),
            mock.patch("colabcola.cli.create_google_drive_backend") as backend,
            mock.patch(
                "colabcola.cli.write_google_drive_settings", return_value=settings_path
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = main(
                [
                    "auth",
                    "google-drive",
                    "--desktop-client-secrets",
                    os.fspath(self.root / "desktop-private.json"),
                    "--colab-client-secrets",
                    os.fspath(self.root / "colab-private.json"),
                    "--drive-config",
                    os.fspath(settings_path),
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(stderr.getvalue(), "")
        backend.assert_called_once_with(settings, force_authorize=True)
        response = json.loads(stdout.getvalue())
        self.assertEqual(response["state"], "google_drive_configured")
        self.assertEqual(
            response["drive_scope"],
            "https://www.googleapis.com/auth/drive.file",
        )
        self.assertIn("COLABCOLA_GOOGLE_DRIVE_OAUTH", response["next_action"])
        self.assertNotIn("private", stdout.getvalue())

    def test_primary_main_automatically_stages_and_writes_bound_controller(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        settings = GoogleDriveSettings(
            schema_version=1,
            google_cloud_project_id="colabcola-fixture",
            desktop_client_secrets=self.root / "desktop.json",
            colab_oauth_client_id="controller-fixture.apps.googleusercontent.com",
            colab_secret_name="COLABCOLA_GOOGLE_DRIVE_OAUTH",
        )

        def stage(artifact, path, backend):
            self.assertTrue(Path(path).is_file())
            self.assertIs(backend, fake_backend)
            return drive_receipt_for_observed_upload(
                artifact,
                remote_file_id="payload_auto_1",
                remote_parent_id="folder_auto_1",
                remote_relative_path=(
                    f"colabcola/staging/{artifact.bundle.job_id}/{artifact.file_name}"
                ),
            )

        fake_backend = object()
        with (
            mock.patch("colabcola.cli.load_google_drive_settings", return_value=settings),
            mock.patch("colabcola.cli.create_google_drive_backend", return_value=fake_backend),
            mock.patch("colabcola.cli.stage_drive_bundle", side_effect=stage),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = main([os.fspath(self.job_file), "--epochs", "20"])
        self.assertEqual(result, 0)
        self.assertEqual(stderr.getvalue(), "")
        receipt = json.loads(stdout.getvalue())
        self.assertEqual(receipt["state"], "awaiting_manual_launch")
        self.assertEqual(receipt["transport"], "google_drive")
        self.assertEqual(receipt["drive_file_id"], "payload_auto_1")
        self.assertEqual(receipt["bundle_path"], self.job_file.with_suffix(".colabcola").as_posix())
        bundle = self.job_file.with_suffix(".colabcola").read_bytes()
        self.assertEqual(receipt["bundle_bytes"], len(bundle))
        self.assertEqual(receipt["bundle_sha256"], __import__("hashlib").sha256(bundle).hexdigest())
        self.assertGreaterEqual(receipt["compile_seconds"], 0.0)
        self.assertGreaterEqual(receipt["drive_staging_seconds"], 0.0)
        self.assertGreaterEqual(receipt["controller_seconds"], 0.0)
        self.assertEqual(receipt["warnings"], [])
        self.assertTrue(Path(receipt["controller_path"]).is_file())
        self.assertTrue(Path(receipt["receipt_path"]).is_file())

    def test_explicit_compile_command_uses_drive_transport_by_default(self) -> None:
        output = self.root / "compiled.colabcola"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = main(["compile", "--output", os.fspath(output), os.fspath(self.job_file)])
        self.assertEqual(result, 0)
        self.assertEqual(stderr.getvalue(), "")
        receipt = json.loads(stdout.getvalue())
        self.assertEqual(receipt["state"], "awaiting_drive_staging")
        self.assertEqual(receipt["transport"], "google_drive")
        self.assertGreaterEqual(receipt["compile_seconds"], 0.0)
        self.assertTrue(output.is_file())

    def test_explicit_embedded_transport_retains_self_contained_notebook(self) -> None:
        output = self.root / "embedded.ipynb"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = main(
                [
                    "compile",
                    "--transport",
                    "embedded",
                    "--output",
                    os.fspath(output),
                    os.fspath(self.job_file),
                ]
            )
        self.assertEqual(result, 0)
        self.assertEqual(stderr.getvalue(), "")
        receipt = json.loads(stdout.getvalue())
        self.assertEqual(receipt["state"], "compiled")
        self.assertEqual(receipt["transport"], "embedded")
        self.assertTrue(output.is_file())

    def test_primary_main_reports_nonblocking_archive_warning(self) -> None:
        output = self.root / "warning.ipynb"
        stdout = io.StringIO()
        with (
            mock.patch("colabcola.bundle.ARCHIVE_WARNING_BYTES", 1),
            mock.patch("colabcola.notebook.ARCHIVE_WARNING_BYTES", 1),
            self.assertWarnsRegex(RuntimeWarning, "warning threshold"),
            contextlib.redirect_stdout(stdout),
        ):
            result = main(
                [
                    "compile",
                    "--transport",
                    "embedded",
                    "--output",
                    os.fspath(output),
                    os.fspath(self.job_file),
                ]
            )

        self.assertEqual(result, 0)
        receipt = json.loads(stdout.getvalue())
        self.assertEqual(len(receipt["warnings"]), 1)
        self.assertIn("warning threshold", receipt["warnings"][0])
        notebook = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(
            notebook["metadata"]["colabcola"]["warnings"], receipt["warnings"]
        )

    def test_validate_cli_accepts_colajob(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = main(["validate", os.fspath(self.job_file)])
        self.assertEqual(result, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(json.loads(stdout.getvalue())["name"], "fixture-job")

    def test_controller_cli_writes_tiny_notebook_and_receipt(self) -> None:
        prepared_stdout = io.StringIO()
        with contextlib.redirect_stdout(prepared_stdout):
            self.assertEqual(main(["compile", os.fspath(self.job_file)]), 0)
        prepared = json.loads(prepared_stdout.getvalue())
        output = self.root / "controller.ipynb"
        receipt = self.root / "controller.receipt.json"
        remote_path = (
            f"colabcola/staging/{prepared['job_id']}/{prepared['job_id']}.colabcola"
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        settings = GoogleDriveSettings(
            schema_version=1,
            google_cloud_project_id="colabcola-fixture",
            desktop_client_secrets=self.root / "desktop.json",
            colab_oauth_client_id="controller-fixture.apps.googleusercontent.com",
            colab_secret_name="COLABCOLA_GOOGLE_DRIVE_OAUTH",
        )
        with (
            mock.patch("colabcola.cli.load_google_drive_settings", return_value=settings),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = main(
                [
                    "controller",
                    "--drive-file-id",
                    "payload_cli_1",
                    "--drive-parent-id",
                    "folder_cli_1",
                    "--drive-relative-path",
                    remote_path,
                    "--output",
                    os.fspath(output),
                    "--receipt",
                    os.fspath(receipt),
                    os.fspath(self.job_file),
                ]
            )
        self.assertEqual(result, 0)
        self.assertEqual(stderr.getvalue(), "")
        response = json.loads(stdout.getvalue())
        self.assertEqual(response["state"], "awaiting_manual_launch")
        self.assertEqual(response["drive_file_id"], "payload_cli_1")
        self.assertTrue(output.is_file())
        self.assertTrue(receipt.is_file())

    def test_validate_cli_rejects_wrong_extension(self) -> None:
        wrong = self.root / "fixture.json"
        wrong.write_text(self.job_file.read_text(encoding="utf-8"), encoding="utf-8")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = main(["validate", os.fspath(wrong)])
        self.assertEqual(result, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("must use the .colajob extension", stderr.getvalue())

    def test_unknown_colabcola_option_before_job_is_rejected(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            _parse_command(["--unknown", os.fspath(self.job_file)])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("unrecognized arguments: --unknown", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
