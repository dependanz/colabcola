from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from colabcola import (
    DriveControllerAuth,
    JobSpec,
    bind_job_arguments,
    build_bundle,
    build_drive_controller,
    build_staged_bundle,
    drive_receipt_for_observed_upload,
    write_staged_bundle,
)

from test_notebook import notebook_job


class DriveControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "main.py"
        self.source.write_text(
            "from pathlib import Path\n"
            "Path('outputs').mkdir()\n"
            "Path('outputs/value.txt').write_text('controller ok', encoding='utf-8')\n",
            encoding="utf-8",
        )
        mapping = notebook_job(self.source)
        mapping["outputs"] = [
            {"pattern": "outputs/value.txt", "required": True, "maximum_bytes": 100}
        ]
        spec = JobSpec.from_mapping(mapping, base_dir=self.root)
        self.staged = build_staged_bundle(build_bundle(bind_job_arguments(spec)))
        self.staged_path = self.root / self.staged.file_name
        write_staged_bundle(self.staged, self.staged_path)
        self.receipt = drive_receipt_for_observed_upload(
            self.staged,
            remote_file_id="drive_payload_123",
            remote_parent_id="drive_job_folder_123",
            remote_relative_path=(
                f"colabcola/staging/{self.staged.bundle.job_id}/{self.staged.file_name}"
            ),
        )
        self.auth = DriveControllerAuth(
            google_cloud_project_id="colabcola-fixture",
            oauth_client_id="controller-fixture.apps.googleusercontent.com",
            colab_secret_name="COLABCOLA_GOOGLE_DRIVE_OAUTH",
        )

    def test_controller_is_small_deterministic_and_has_no_payload_or_host_path(self) -> None:
        first = build_drive_controller(self.staged, self.receipt, self.auth)
        second = build_drive_controller(self.staged, self.receipt, self.auth)
        document = json.loads(first.notebook_bytes)

        self.assertEqual(first.notebook_bytes, second.notebook_bytes)
        self.assertLess(len(first.notebook_bytes), 100_000)
        self.assertNotIn(self.staged.bundle.archive_bytes, first.notebook_bytes)
        self.assertNotIn(os.fsencode(self.root), first.notebook_bytes)
        self.assertEqual(document["metadata"]["colabcola"]["transport"], "google_drive")
        self.assertEqual(document["metadata"]["colabcola"]["compiler_contract_version"], "0.10")
        self.assertEqual(document["metadata"]["colabcola"]["runtime_version"], "latest")
        self.assertEqual(
            [cell["metadata"]["colabcola"]["group"] for cell in document["cells"]],
            [
                "summary",
                "manifest",
                "preflight",
                "acquisition",
                "extraction",
                "environment",
                "execution",
            ],
        )
        self.assertIn("contains no base64 payload", document["cells"][0]["source"])
        self.assertIn("does not mount Drive", document["cells"][0]["source"])
        self.assertNotIn("drive.mount", first.notebook_bytes.decode("utf-8"))
        for cell in document["cells"][1:]:
            compile(cell["source"], "<colabcola-drive-controller>", "exec")

    def test_controller_streams_local_override_and_executes_complete_job(self) -> None:
        artifact = build_drive_controller(self.staged, self.receipt, self.auth)
        document = json.loads(artifact.notebook_bytes)
        content_root = self.root / "content"
        namespace = {
            "COLABCOLA_CONTENT_ROOT_OVERRIDE": os.fspath(content_root),
            "COLABCOLA_STAGED_BUNDLE_PATH_OVERRIDE": os.fspath(self.staged_path),
        }
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            for cell in document["cells"]:
                if cell["cell_type"] == "code":
                    exec(cell["source"], namespace)

        result_root = content_root / "colabcola" / "results" / self.staged.bundle.job_id
        result = json.loads((result_root / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["terminal_state"], "success")
        self.assertEqual(result["transport"]["mode"], "google_drive")
        self.assertEqual(result["transport"]["drive_file_id"], "drive_payload_123")
        self.assertEqual(result["transport"]["staged_sha256"], self.staged.staged_sha256)
        self.assertGreaterEqual(result["transport"]["acquisition_seconds"], 0.0)
        self.assertEqual(result["transport"]["authorization_and_metadata_seconds"], 0.0)
        self.assertEqual(result["transport"]["download_seconds"], 0.0)
        self.assertGreaterEqual(
            result["transport"]["staged_reconstruction_seconds"], 0.0
        )
        self.assertGreaterEqual(result["transport"]["extraction_seconds"], 0.0)
        self.assertEqual(result["outputs"][0]["path"], "outputs/value.txt")
        evidence_line = next(
            line
            for line in stdout.getvalue().splitlines()
            if line.startswith("colabcola artifacts evidence: ")
        )
        evidence = json.loads(evidence_line.partition(": ")[2])
        artifacts_path = result_root / "artifacts.zip"
        self.assertEqual(evidence["name"], "artifacts.zip")
        self.assertEqual(evidence["size"], artifacts_path.stat().st_size)
        self.assertEqual(
            evidence["sha256"], hashlib.sha256(artifacts_path.read_bytes()).hexdigest()
        )

    def test_corrupt_staged_object_produces_integrity_failure_evidence(self) -> None:
        corrupt = self.root / "corrupt.colabcola"
        content = bytearray(self.staged_path.read_bytes())
        content[-1] ^= 1
        corrupt.write_bytes(content)
        artifact = build_drive_controller(self.staged, self.receipt, self.auth)
        document = json.loads(artifact.notebook_bytes)
        content_root = self.root / "corrupt-content"
        namespace = {
            "COLABCOLA_CONTENT_ROOT_OVERRIDE": os.fspath(content_root),
            "COLABCOLA_STAGED_BUNDLE_PATH_OVERRIDE": os.fspath(corrupt),
        }
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "integrity_failed"):
                for cell in document["cells"]:
                    if cell["cell_type"] == "code":
                        exec(cell["source"], namespace)

        result_path = (
            content_root
            / "colabcola"
            / "results"
            / self.staged.bundle.job_id
            / "result.json"
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(result["terminal_state"], "integrity_failed")
        self.assertEqual(result["failure"]["stage"], "integrity")
        self.assertNotEqual(hashlib.sha256(corrupt.read_bytes()).hexdigest(), self.staged.staged_sha256)

    def test_controller_device_oauth_downloads_and_verifies_exact_drive_object(self) -> None:
        artifact = build_drive_controller(self.staged, self.receipt, self.auth)
        document = json.loads(artifact.notebook_bytes)
        cells = {
            cell["metadata"]["colabcola"]["group"]: cell["source"]
            for cell in document["cells"]
            if cell["cell_type"] == "code"
        }
        secret_value = "fixture-device-client-secret"
        access_token = "fixture-user-access-token"
        secret_document = json.dumps(
            {
                "device": {
                    "client_id": self.auth.oauth_client_id,
                    "client_secret": secret_value,
                    "project_id": self.auth.google_cloud_project_id,
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            }
        )

        class UserData:
            @staticmethod
            def get(name: str) -> str:
                if name != self.auth.colab_secret_name:
                    raise AssertionError("unexpected Colab Secret name")
                return secret_document

        class Response:
            def __init__(self, content: bytes, status: int = 200) -> None:
                self.content = content
                self.status = status
                self.offset = 0

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self, size: int = -1) -> bytes:
                if size < 0:
                    result = self.content[self.offset :]
                    self.offset = len(self.content)
                    return result
                result = self.content[self.offset : self.offset + size]
                self.offset += len(result)
                return result

        requests: list[object] = []

        def urlopen(request, timeout: int):
            self.assertEqual(timeout, 120)
            requests.append(request)
            url = request.full_url
            if url == "https://oauth2.googleapis.com/device/code":
                return Response(
                    json.dumps(
                        {
                            "device_code": "fixture-device-code",
                            "expires_in": 300,
                            "interval": 1,
                            "user_code": "ABCD-EFGH",
                            "verification_url": "https://www.google.com/device",
                        }
                    ).encode("utf-8")
                )
            if url == "https://oauth2.googleapis.com/token":
                return Response(
                    json.dumps(
                        {
                            "access_token": access_token,
                            "scope": "https://www.googleapis.com/auth/drive.file",
                            "token_type": "Bearer",
                        }
                    ).encode("utf-8")
                )
            if "alt=media" in url:
                return Response(self.staged_path.read_bytes())
            if "/drive/v3/files/drive_payload_123?" in url:
                return Response(
                    json.dumps(
                        {
                            "appProperties": {
                                "colabcola_job_id": self.staged.bundle.job_id,
                                "colabcola_staged_sha256": self.staged.staged_sha256,
                            },
                            "id": self.receipt.remote_file_id,
                            "mimeType": "application/octet-stream",
                            "name": self.receipt.remote_name,
                            "parents": [self.receipt.remote_parent_id],
                            "size": str(self.staged.staged_bytes),
                            "trashed": False,
                        }
                    ).encode("utf-8")
                )
            raise AssertionError(f"unexpected URL: {url}")

        google = types.ModuleType("google")
        colab = types.ModuleType("google.colab")
        colab.userdata = UserData
        google.colab = colab
        content_root = self.root / "remote-content"
        namespace = {"COLABCOLA_CONTENT_ROOT_OVERRIDE": os.fspath(content_root)}
        exec(cells["manifest"], namespace)
        namespace.update(
            {
                "COLABCOLA_CONTENT_ROOT": content_root,
                "_COLABCOLA_TERMINAL_STATE": None,
                "_colabcola_fail": lambda stage, state, exc: (_ for _ in ()).throw(exc),
                "pathlib": __import__("pathlib"),
                "shutil": __import__("shutil"),
                "time": __import__("time"),
            }
        )
        with (
            mock.patch.dict("sys.modules", {"google": google, "google.colab": colab}),
            mock.patch("urllib.request.urlopen", side_effect=urlopen),
            mock.patch("time.sleep", return_value=None),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            exec(cells["acquisition"], namespace)

        self.assertEqual(namespace["_COLABCOLA_TRANSPORT_FACTS"]["drive_file_id"], "drive_payload_123")
        self.assertEqual(
            namespace["_COLABCOLA_TRANSPORT_FACTS"]["oauth_scope"],
            "https://www.googleapis.com/auth/drive.file",
        )
        self.assertEqual(namespace["COLABCOLA_ARCHIVE_PATH"].read_bytes(), self.staged.bundle.archive_bytes)
        self.assertEqual(len(requests), 4)
        self.assertTrue(all("drive_payload_123" in request.full_url for request in requests[2:]))
        self.assertEqual(requests[2].headers["Authorization"], "Bearer " + access_token)
        self.assertNotIn(secret_value, artifact.notebook_bytes.decode("utf-8"))
        self.assertNotIn(access_token, artifact.notebook_bytes.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
