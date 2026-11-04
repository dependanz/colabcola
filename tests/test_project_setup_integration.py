"""Wizard/adapter regression for Google's ambiguous missing-project diagnostic."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from colabcola import project_setup
from colabcola.cloud_project import PERMISSIONS


class ProjectSetupIntegrationTests(unittest.TestCase):
    def setUp(self):
        discovery = mock.patch("colabcola.client_downloads.download_directories", return_value=[])
        discovery.start(); self.addCleanup(discovery.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "config"
        self.download = self.base / "desktop.json"
        self.project = "colabcola-aaaaaaaaaaaa"
        self.email = "person@example.test"
        self.download.write_text(json.dumps({"installed": {
            "project_id": self.project, "client_id": "123-personal.apps.googleusercontent.com",
            "client_secret": "PRIVATE-CLIENT-SECRET", "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token", "redirect_uris": ["http://localhost"]
        }}), encoding="utf-8")
        self.created = False
        self.enabled = False
        self.calls = []
        self.messages = io.StringIO()
        for patch in (
            mock.patch("colabcola.cloud_project.shutil.which", return_value="gcloud"),
            mock.patch("colabcola.cloud_project.subprocess.run", side_effect=self.gcloud),
            mock.patch("colabcola.cloud_project.requests.Session", side_effect=AssertionError("unexpected HTTP")),
            mock.patch("colabcola.cloud_project.CloudProjectClient.check_permissions", return_value={
                "required": list(PERMISSIONS), "granted": list(PERMISSIONS), "missing": []}),
            mock.patch("colabcola.project_setup.uuid.uuid4", return_value=SimpleNamespace(hex="a" * 32)),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def result(self, value):
        return subprocess.CompletedProcess([], 0, json.dumps(value), "")

    def gcloud(self, argv, **kwargs):
        command = argv[1:3]
        self.calls.append(command)
        self.assertFalse(kwargs["shell"])
        if command == ["auth", "list"]:
            return self.result([{"account": self.email, "status": ""}])
        if command == ["auth", "print-access-token"]:
            return subprocess.CompletedProcess([], 0, "PRIVATE-MANAGEMENT-TOKEN", "")
        if command == ["projects", "list"]:
            return self.result([])
        if command == ["projects", "describe"]:
            self.assertEqual(argv[3], self.project)
            if not self.created:
                return subprocess.CompletedProcess([], 1, "",
                    "ERROR: (gcloud.projects.describe) [person@example.test] does not have permission "
                    "to access projects instance [" + self.project + "] (or it may not exist): "
                    "The caller does not have permission. This command is authenticated as "
                    "person@example.test which is the active account specified by the [core/account] property")
            return self.result({"projectId": self.project, "projectNumber": "123456789",
                                "lifecycleState": "ACTIVE", "labels": {"colabcola-managed": "true"}})
        if command == ["projects", "create"]:
            saved = json.loads((self.root / "project-setup.json").read_text())
            self.assertEqual(argv[3], self.project)
            self.assertEqual(saved["project_id"], self.project)
            self.assertTrue(saved["setup_approved"])
            self.assertTrue(saved["create_requested"])
            self.created = True
            return subprocess.CompletedProcess([], 0, "", "")
        if command == ["billing", "projects"]:
            return self.result({"projectId": self.project, "billingEnabled": False, "billingAccountName": ""})
        if command == ["services", "list"]:
            return self.result([{"config": {"name": "drive.googleapis.com"}, "state": "ENABLED"}] if self.enabled else [])
        if command == ["services", "enable"]:
            self.assertIn("drive.googleapis.com", argv)
            self.enabled = True
            return subprocess.CompletedProcess([], 0, "", "")
        raise AssertionError("unexpected Google operation: " + str(command))

    def run_setup(self, answers, **kwargs):
        iterator = iter(answers)
        with contextlib.redirect_stderr(self.messages):
            return project_setup.ensure_personal_client(self.root, input_fn=lambda question: next(iterator), **kwargs)

    def test_fresh_new_creates_only_after_approval_without_permission_lookup_gate(self):
        value = self.run_setup(["new", "yes", str(self.download)])
        self.assertEqual(value["stage"], "ready")
        self.assertEqual(value["project_id"], self.project)
        self.assertTrue(self.enabled)
        self.assertLess(self.calls.index(["projects", "create"]), self.calls.index(["projects", "describe"]))
        self.assertNotIn("PRIVATE-", self.messages.getvalue())

    def test_saved_dev18_selection_resumes_same_id_with_explicit_create_choice(self):
        self.root.mkdir()
        old = {"schema": 1, "mode": "user_project", "project_id": self.project,
               "project_number": None, "stage": "selected", "method": "gcloud",
               "cloud_checks": "not_checked", "setup_approved": False,
               "create_requested": False, "billing_unlinked": False, "drive_enabled": False,
               "client_sha256": None, "cloud_account_email": self.email}
        path = self.root / "project-setup.json"
        path.write_text(json.dumps(old), encoding="utf-8")
        before = path.read_bytes()
        self.assertEqual(project_setup.status(self.root)["project_id"], self.project)
        self.assertEqual(path.read_bytes(), before)
        value = self.run_setup(["create", "yes", str(self.download)])
        self.assertEqual(value["stage"], "ready")
        self.assertEqual(value["project_id"], self.project)
        self.assertEqual(value["schema"], 2)
        self.assertEqual(value["project_mode"], "new")
        self.assertNotIn(["projects", "list"], self.calls)
        self.assertLess(self.calls.index(["projects", "create"]), self.calls.index(["projects", "describe"]))

    def test_explicit_reuse_does_not_interpret_ambiguous_access_as_absence(self):
        with self.assertRaises(project_setup.ProjectSetupError):
            self.run_setup([], project=self.project)
        self.assertFalse(self.created)
        self.assertFalse(self.enabled)
        self.assertNotIn(["projects", "create"], self.calls)
        saved = json.loads((self.root / "project-setup.json").read_text())
        self.assertFalse(saved["setup_approved"])

    def test_declined_new_project_has_no_project_or_api_mutation(self):
        with self.assertRaises(project_setup.ProjectSetupError):
            self.run_setup(["new", "no"])
        self.assertFalse(self.created)
        self.assertFalse(self.enabled)
        self.assertNotIn(["projects", "describe"], self.calls)
        self.assertNotIn(["projects", "create"], self.calls)


if __name__ == "__main__":
    unittest.main()
