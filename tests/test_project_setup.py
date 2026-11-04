"""Personal-project guide acceptance with fake Cloud operations and temporary clients only."""
import contextlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlsplit

from colabcola import project_setup as setup
from colabcola.cloud_project import CloudProjectError


class ProjectSetupTests(unittest.TestCase):
    def setUp(self):
        discovery = mock.patch("colabcola.client_downloads.download_directories", return_value=[])
        discovery.start(); self.addCleanup(discovery.stop)
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name); self.root = self.base / "isolated-config"
        self.download = self.base / "download.json"; self.write_client(self.download)
        self.cloud = mock.Mock()
        self.cloud.available.return_value = True
        self.cloud.authenticate.return_value = "user@example.test"
        self.cloud.projects.return_value = []
        self.cloud.project.return_value = self.project()
        self.cloud.create.side_effect = lambda identity: self.project(identity)
        self.cloud.check_permissions.return_value = {"required": ["serviceusage.services.enable"],
            "granted": ["serviceusage.services.enable"], "missing": []}
        self.cloud.billing.return_value = {"linked": False, "enabled": False}
        self.cloud.ensure_drive.return_value = {"enabled": True, "changed": True}
        patch = mock.patch("colabcola.cloud_project.CloudProjectClient", return_value=self.cloud)
        self.factory = patch.start(); self.addCleanup(patch.stop)
        self.messages = io.StringIO(); self.questions = []

    @staticmethod
    def project(identity="fixture-project", marked=True):
        return {"project_id": identity, "project_number": "123456", "state": "ACTIVE",
                "labels": {"colabcola-managed": "true"} if marked else {}}

    def write_client(self, path, project="fixture-project", identity="123456-personal.apps.googleusercontent.com"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"installed": {"project_id": project, "client_id": identity,
            "client_secret": "PRIVATE-DESKTOP-SECRET", "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token", "redirect_uris": ["http://localhost"]}}), encoding="utf-8")

    def run_setup(self, answers=(), **kwargs):
        answers = iter(answers)
        def respond(question):
            self.questions.append(question)
            try:
                value = next(answers)
            except StopIteration:
                raise EOFError()
            if isinstance(value, BaseException):
                raise value
            return value
        with contextlib.redirect_stderr(self.messages):
            return setup.ensure_personal_client(self.root, input_fn=respond, **kwargs)

    def ready(self):
        return self.run_setup(["yes"], project="fixture-project", client_json=self.download)

    def legacy_record(self, *, stage="selected", approved=False, create=False):
        self.root.mkdir(parents=True, exist_ok=True)
        value = {"schema": 1, "mode": "user_project", "project_id": "fixture-project", "project_number": None,
                 "stage": stage, "method": "gcloud", "cloud_checks": "not_checked", "setup_approved": approved,
                 "create_requested": create, "billing_unlinked": False, "drive_enabled": False,
                 "client_sha256": None, "cloud_account_email": "user@example.test"}
        (self.root / setup.SETUP_FILE).write_text(json.dumps(value), encoding="utf-8")
        return value

    def assert_no_leaks(self, error=None):
        text = self.messages.getvalue() + (str(error) if error else "")
        if (self.root / setup.SETUP_FILE).exists():
            text += (self.root / setup.SETUP_FILE).read_text()
        self.assertNotIn("PRIVATE-", text)
        # Google's legitimate download filename includes client_secret_; the
        # credential field and all synthetic secret values must stay private.
        self.assertNotRegex(text, r'["\']client_secret["\']\s*:')

    def test_existing_project_one_concrete_approval_enables_drive_and_imports_client(self):
        def enable(identity):
            saved = json.loads((self.root / setup.SETUP_FILE).read_text())
            self.assertEqual(saved["project_id"], identity)
            self.assertTrue(saved["setup_approved"]); self.assertTrue(saved["billing_unlinked"])
            return {"enabled": True, "changed": True}
        self.cloud.ensure_drive.side_effect = enable
        value = self.ready()
        self.assertEqual(value["stage"], "ready"); self.assertEqual(value["cloud_checks"], "online_verified")
        self.assertEqual(value["project_number"], "123456"); self.assertEqual(value["cloud_account_email"], "user@example.test")
        self.assertEqual(len(self.questions), 1)
        self.cloud.create.assert_not_called(); self.cloud.ensure_drive.assert_called_once_with("fixture-project")
        self.assertIn("No billing changes or paid services", self.messages.getvalue())
        self.assertIn("seven days", self.messages.getvalue())
        if os.name != "nt":
            for name in (setup.SETUP_FILE, setup.CLIENT_FILE):
                self.assertEqual((self.root / name).stat().st_mode & 0o777, 0o600)
        self.assert_no_leaks()

    def test_ready_reuse_is_offline_with_no_management_signin_or_sidecar_writes(self):
        value = self.ready()
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        self.factory.reset_mock(); self.cloud.reset_mock()
        self.assertEqual(self.run_setup(), value)
        self.factory.assert_not_called(); self.cloud.authenticate.assert_not_called()
        self.assertEqual({p.name: p.read_bytes() for p in self.root.iterdir()}, before)

    def test_create_persists_approved_identity_before_mutation_and_resume_keeps_it(self):
        self.cloud.project.return_value = None
        def uncertain(identity):
            saved = json.loads((self.root / setup.SETUP_FILE).read_text())
            self.assertEqual(saved["project_id"], identity); self.assertEqual(saved["stage"], "approved")
            self.assertTrue(saved["create_requested"])
            self.cloud.project.return_value = self.project(identity)
            raise CloudProjectError("Creation could not be confirmed; retry the selected project")
        self.cloud.create.side_effect = uncertain
        with self.assertRaises(setup.ProjectSetupError):
            self.run_setup(["new", "yes"])
        saved = json.loads((self.root / setup.SETUP_FILE).read_text()); identity = saved["project_id"]
        self.write_client(self.download, project=identity)
        self.cloud.create.side_effect = lambda value: self.project(value)
        result = self.run_setup(client_json=self.download)
        self.assertEqual(result["project_id"], identity)
        self.assertEqual(self.cloud.create.call_args_list, [mock.call(identity), mock.call(identity)])
        self.cloud.projects.assert_called_once()  # Resume never picks or invents another project.
        self.assert_no_leaks()

    def test_resume_reconciles_managed_label_before_enabling(self):
        self.cloud.create.side_effect = CloudProjectError("Creation uncertain")
        with self.assertRaises(setup.ProjectSetupError):
            self.run_setup(["new", "yes"])
        identity = json.loads((self.root / setup.SETUP_FILE).read_text())["project_id"]
        self.write_client(self.download, project=identity)
        self.cloud.create.side_effect = CloudProjectError("Existing project is not Colabcola-managed")
        with self.assertRaisesRegex(setup.ProjectSetupError, "not Colabcola-managed"):
            self.run_setup(client_json=self.download)
        self.cloud.ensure_drive.assert_not_called()

    def test_fresh_new_intent_does_not_pre_describe_ambiguous_project_id(self):
        self.cloud.project.side_effect = CloudProjectError("Project lookup failed (FAILED); permission denied or project may not exist")
        def create(identity):
            saved = json.loads((self.root / setup.SETUP_FILE).read_text())
            self.assertEqual(saved["schema"], 2); self.assertEqual(saved["project_mode"], "new")
            self.assertEqual(saved["project_id"], identity)
            self.assertTrue(saved["setup_approved"]); self.assertTrue(saved["create_requested"])
            self.write_client(self.download, project=identity)
            return self.project(identity)
        self.cloud.create.side_effect = create
        result = self.run_setup(["new", "yes", str(self.download)])
        self.assertEqual(result["project_mode"], "new")
        self.cloud.project.assert_not_called(); self.cloud.create.assert_called_once_with(result["project_id"])
        self.assertEqual(len([q for q in self.questions if "Approve" in q]), 1)

    def test_existing_denial_or_not_found_never_becomes_creation_authority(self):
        for error in (CloudProjectError("Project lookup failed (FAILED); permission denied or project may not exist"), None):
            with self.subTest(outcome=type(error).__name__):
                self.root = self.base / ("denied" if error else "not-found")
                self.cloud.project.side_effect = error
                self.cloud.project.return_value = None
                with self.assertRaises(setup.ProjectSetupError):
                    self.run_setup(project="colabcola-existing-id")
                self.assertEqual(setup.status(self.root)["project_mode"], "existing")
                self.cloud.create.assert_not_called(); self.cloud.ensure_drive.assert_not_called()
        self.assertEqual(self.questions, [])

    def test_declining_new_creation_preserves_identity_without_mutation(self):
        with self.assertRaisesRegex(setup.ProjectSetupError, "not approved"):
            self.run_setup(["new", "no"])
        saved = json.loads((self.root / setup.SETUP_FILE).read_text())
        self.assertEqual(saved["project_mode"], "new"); self.assertFalse(saved["setup_approved"])
        self.assertFalse(saved["create_requested"])
        self.cloud.project.assert_not_called(); self.cloud.create.assert_not_called(); self.cloud.ensure_drive.assert_not_called()
        self.write_client(self.download, project=saved["project_id"])
        result = self.run_setup(["yes"], client_json=self.download)
        self.assertEqual(result["project_id"], saved["project_id"])
        self.cloud.create.assert_called_once_with(saved["project_id"])

    def test_new_intent_survives_interruption_before_creation_approval(self):
        with self.assertRaisesRegex(setup.ProjectSetupError, "paused at selected"):
            self.run_setup(["new", KeyboardInterrupt()])
        saved = json.loads((self.root / setup.SETUP_FILE).read_text())
        self.assertEqual(saved["schema"], 2); self.assertEqual(saved["project_mode"], "new")
        self.cloud.create.assert_not_called(); self.cloud.project.assert_not_called()
        self.write_client(self.download, project=saved["project_id"])
        result = self.run_setup(["yes"], client_json=self.download)
        self.assertEqual(result["project_id"], saved["project_id"])
        self.cloud.projects.assert_called_once()

    def test_legacy_selected_record_requires_explicit_intent_before_exact_id_create(self):
        self.legacy_record()
        self.cloud.project.side_effect = AssertionError("new intent must not describe before creation")
        result = self.run_setup(["create", "yes"], client_json=self.download)
        self.assertEqual(result["schema"], 2); self.assertEqual(result["project_id"], "fixture-project")
        self.assertEqual(result["project_mode"], "new")
        self.assertIn("create or reuse", self.questions[0]); self.assertIn("Approve", self.questions[1])
        self.cloud.create.assert_called_once_with("fixture-project"); self.cloud.project.assert_not_called()
        self.cloud.projects.assert_not_called()

    def test_legacy_selected_record_can_explicitly_reuse_without_creation(self):
        self.legacy_record()
        result = self.run_setup(["reuse", "yes"], client_json=self.download)
        self.assertEqual(result["project_mode"], "existing")
        self.cloud.project.assert_called_once_with("fixture-project"); self.cloud.create.assert_not_called()

    def test_legacy_intent_prompt_interrupt_keeps_old_bytes_and_choice_is_asked_once(self):
        self.legacy_record()
        original = (self.root / setup.SETUP_FILE).read_bytes()
        with self.assertRaisesRegex(setup.ProjectSetupError, "paused"):
            self.run_setup([EOFError()])
        self.assertEqual((self.root / setup.SETUP_FILE).read_bytes(), original)
        self.factory.assert_not_called()
        with self.assertRaisesRegex(setup.ProjectSetupError, "paused"):
            self.run_setup(["create", KeyboardInterrupt()])
        selected = json.loads((self.root / setup.SETUP_FILE).read_text())
        self.assertEqual(selected["schema"], 2); self.assertEqual(selected["project_mode"], "new")
        count = len(self.questions)
        self.assertEqual(self.run_setup(["yes"], client_json=self.download)["stage"], "ready")
        self.assertNotIn("create or reuse", " ".join(self.questions[count:]))

    def test_legacy_approved_create_preserves_approval_and_exact_identity(self):
        self.legacy_record(stage="approved", approved=True, create=True)
        result = self.run_setup(client_json=self.download)
        self.assertEqual(result["project_mode"], "new"); self.assertEqual(result["project_id"], "fixture-project")
        self.assertEqual(self.questions, [])
        self.cloud.create.assert_called_once_with("fixture-project"); self.cloud.project.assert_not_called()

    def test_legacy_status_normalizes_unknown_intent_without_reading_client_or_writing(self):
        self.legacy_record()
        original = (self.root / setup.SETUP_FILE).read_bytes()
        with mock.patch.object(setup, "_client", side_effect=AssertionError("status reads no clients")):
            result = setup.status(self.root)
        self.assertEqual(result["project_mode"], "unknown")
        self.assertEqual((self.root / setup.SETUP_FILE).read_bytes(), original)
        self.factory.assert_not_called()

    def test_legacy_ready_binding_remains_offline_and_byte_preserved(self):
        current = self.ready(); current["schema"] = 1; current.pop("project_mode")
        (self.root / setup.SETUP_FILE).write_text(json.dumps(current), encoding="utf-8")
        original = {p.name: p.read_bytes() for p in self.root.iterdir()}
        self.factory.reset_mock()
        result = self.run_setup()
        self.assertEqual(result["project_mode"], "existing"); self.assertEqual(result["stage"], "ready")
        self.assertEqual({p.name: p.read_bytes() for p in self.root.iterdir()}, original)
        self.factory.assert_not_called()

    def test_single_marked_discovered_project_is_the_selection_default(self):
        self.cloud.projects.return_value = [self.project(), self.project("another-project", marked=False)]
        value = self.run_setup(["", "yes", str(self.download)])
        self.assertEqual(value["project_id"], "fixture-project")
        self.assertIn("[fixture-project]", self.questions[0])
        self.cloud.create.assert_not_called()

    def test_eof_and_interrupt_leave_exact_selected_project_for_resume(self):
        for interruption in (EOFError(), KeyboardInterrupt()):
            with self.subTest(interruption=type(interruption).__name__):
                self.root = self.base / type(interruption).__name__
                with self.assertRaisesRegex(setup.ProjectSetupError, "paused"):
                    self.run_setup([interruption], project="fixture-project")
                value = setup.status(self.root)
                self.assertEqual(value["project_id"], "fixture-project")
                self.assertEqual(value["stage"], "project_ready")
        self.cloud.create.assert_not_called(); self.cloud.ensure_drive.assert_not_called()

    def test_oauth_handoff_resumes_without_repeating_cloud_setup(self):
        with self.assertRaisesRegex(setup.ProjectSetupError, "Cloud setup is saved"):
            self.run_setup(["yes", ""], project="fixture-project")
        self.assertEqual(setup.status(self.root)["stage"], "awaiting_client")
        for route in ("auth/audience", "auth/scopes", "auth/clients", "auth/overview"):
            self.assertIn(route + "?project=fixture-project", self.messages.getvalue())
        self.assertIn("External", self.messages.getvalue()); self.assertIn("test user", self.messages.getvalue().lower())
        self.assertIn("does not publish", self.messages.getvalue())
        self.factory.reset_mock()
        result = self.run_setup(client_json=self.download)
        self.assertEqual(result["stage"], "ready"); self.factory.assert_not_called()

    def test_console_links_encode_exact_project_and_known_account_as_navigation_hint(self):
        account = "person+cloud@example.test"
        with contextlib.redirect_stderr(self.messages):
            setup._oauth_instructions("fixture-project", account)
        message = self.messages.getvalue()
        links = re.findall(r"https://console\.cloud\.google\.com/\S+", message)
        self.assertEqual({urlsplit(link).path for link in links}, {"/auth/overview", "/auth/audience", "/auth/scopes", "/auth/clients"})
        for link in links:
            self.assertEqual(parse_qs(urlsplit(link).query), {"project": ["fixture-project"], "authuser": [account]})
            self.assertIn("%2B", link); self.assertIn("%40", link)
            self.assertNotIn(account, link)
        self.assertIn("Selected Cloud project ID: fixture-project", message)
        self.assertIn("Verified Cloud setup account: " + account, message)
        self.assertIn("best-effort account hint", message)
        self.assertIn("reopen the exact project link", message)
        self.assertIn("support-email dropdown", message)

    def test_manual_console_guidance_has_no_invented_account_or_authuser_hint(self):
        with contextlib.redirect_stderr(self.messages):
            setup._oauth_instructions("fixture-project")
        message = self.messages.getvalue()
        links = re.findall(r"https://console\.cloud\.google\.com/\S+", message)
        self.assertEqual(len(links), 4)
        for link in links:
            self.assertEqual(parse_qs(urlsplit(link).query), {"project": ["fixture-project"]})
        self.assertNotIn("Verified Cloud setup account:", message)
        self.assertIn("not independently verified in manual setup", message)
        self.assertIn("separate/private browser session", message)

    def test_blank_client_path_is_an_intentional_pause_and_bare_login_resumes_guide_offline(self):
        with self.assertRaisesRegex(setup.ProjectSetupError, "intentionally paused") as error:
            self.run_setup(["yes", ""], project="fixture-project")
        self.assertIn("no setup progress was lost", str(error.exception))
        self.assertIn("rerun colabcola login to resume", str(error.exception))
        original = (self.root / setup.SETUP_FILE).read_bytes()
        self.messages = io.StringIO(); self.questions = []
        self.factory.reset_mock()
        self.factory.side_effect = AssertionError("pending Desktop JSON must not access Cloud management")
        with self.assertRaisesRegex(setup.ProjectSetupError, "intentionally paused"):
            self.run_setup([""])
        self.assertEqual((self.root / setup.SETUP_FILE).read_bytes(), original)
        self.assertIn("Get started", self.messages.getvalue())
        self.assertIn("Verified Cloud setup account: user@example.test", self.messages.getvalue())
        self.assertIn("authuser=user%40example.test", self.messages.getvalue())
        self.assertEqual(len(self.questions), 1)
        self.assertIn("downloaded Desktop OAuth JSON", self.questions[0])
        result = self.run_setup([str(self.download)])
        self.assertEqual(result["stage"], "ready")
        self.assertEqual(result["project_id"], "fixture-project")
        self.factory.assert_not_called()

    def test_missing_gcloud_manual_flow_is_explicitly_self_reported(self):
        self.cloud.available.return_value = False
        value = self.run_setup(["manual", "yes"], project="fixture-project", client_json=self.download)
        self.assertEqual(value["cloud_checks"], "self_reported")
        self.assertIsNone(value["project_number"]); self.assertIsNone(value["cloud_account_email"])
        self.cloud.authenticate.assert_not_called(); self.cloud.ensure_drive.assert_not_called()
        self.assertIn("winget install --exact --id Google.CloudSDK", self.messages.getvalue())
        self.assertIn("not independently verified", self.messages.getvalue())
        self.assertIn("drive.googleapis.com?project=fixture-project", self.messages.getvalue())

    def test_missing_gcloud_install_retry_and_stop_do_not_create_projects(self):
        self.cloud.available.side_effect = [False, True]
        self.assertEqual(self.run_setup(["retry", "yes"], project="fixture-project", client_json=self.download)["stage"], "ready")
        self.root = self.base / "stopped"
        self.cloud.available.side_effect = None; self.cloud.available.return_value = False
        self.cloud.reset_mock()
        with self.assertRaisesRegex(setup.ProjectSetupError, "paused before project changes"):
            self.run_setup(["stop"])
        self.cloud.create.assert_not_called(); self.cloud.ensure_drive.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_permissions_or_linked_unknown_billing_stop_before_api_enable(self):
        cases = ("permissions", "linked", "unknown", "enabled")
        for case in cases:
            with self.subTest(case=case):
                self.root = self.base / case
                self.cloud.check_permissions.return_value = {"required": ["needed"], "granted": ["needed"], "missing": []}
                self.cloud.billing.return_value = {"linked": False, "enabled": False}
                self.cloud.billing.side_effect = None
                if case == "permissions": self.cloud.check_permissions.return_value["missing"] = ["needed"]
                elif case == "linked": self.cloud.billing.return_value["linked"] = True
                elif case == "enabled": self.cloud.billing.return_value["enabled"] = True
                else: self.cloud.billing.side_effect = CloudProjectError("Billing state could not be verified")
                with self.assertRaises(setup.ProjectSetupError):
                    self.run_setup(["yes"], project="fixture-project", client_json=self.download)
                self.cloud.ensure_drive.assert_not_called(); self.cloud.create.assert_not_called()
                self.assertFalse((self.root / setup.CLIENT_FILE).exists())
                self.assert_no_leaks()

    def test_raw_cloud_error_is_never_disclosed(self):
        self.cloud.project.side_effect = RuntimeError("PRIVATE-ACCESS-TOKEN PRIVATE-RESPONSE")
        with self.assertRaises(setup.ProjectSetupError) as error:
            self.run_setup(project="fixture-project")
        self.assert_no_leaks(error.exception)
        self.cloud.ensure_drive.assert_not_called()

    def test_client_mismatch_or_malformed_file_never_replaces_existing_profile(self):
        self.ready()
        (self.root / "profile.json").write_text("PRIVATE-OLD-PROFILE")
        (self.root / "drive-oauth.json").write_text("PRIVATE-OLD-TOKEN")
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        changed = self.base / "changed.json"; self.write_client(changed, identity="123456-different.apps.googleusercontent.com")
        self.factory.reset_mock()
        with self.assertRaisesRegex(setup.ProjectSetupError, "new --config-home"):
            self.run_setup(client_json=changed)
        with self.assertRaisesRegex(setup.ProjectSetupError, "new --config-home"):
            self.run_setup(project="another-project")
        self.assertEqual({p.name: p.read_bytes() for p in self.root.iterdir()}, before)
        self.factory.assert_not_called()

    def test_missing_local_client_can_restore_only_exact_recorded_binding(self):
        self.ready(); (self.root / "profile.json").write_text("PRIVATE-PROFILE")
        (self.root / "drive-oauth.json").write_text("PRIVATE-TOKEN")
        (self.root / setup.CLIENT_FILE).unlink()
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        changed = self.base / "changed.json"; self.write_client(changed, identity="123456-different.apps.googleusercontent.com")
        self.factory.reset_mock()
        for kwargs, answers in (({"client_json": changed}, []), ({}, [str(changed)])):
            with self.subTest(explicit=bool(kwargs)), self.assertRaisesRegex(setup.ProjectSetupError, "new --config-home"):
                self.run_setup(answers, **kwargs)
            self.assertFalse((self.root / setup.CLIENT_FILE).exists())
            self.assertEqual((self.root / "profile.json").read_bytes(), before["profile.json"])
            self.assertEqual((self.root / "drive-oauth.json").read_bytes(), before["drive-oauth.json"])
        restored = self.run_setup(client_json=self.download)
        self.assertEqual(restored["stage"], "ready"); self.factory.assert_not_called()

    def test_wrong_project_json_rejected_before_mutation_or_import(self):
        other = self.base / "other.json"; self.write_client(other, project="another-project")
        with self.assertRaisesRegex(setup.ProjectSetupError, "another project"):
            self.run_setup(project="fixture-project", client_json=other)
        self.factory.assert_not_called(); self.assertFalse(self.root.exists())
        with self.assertRaisesRegex(setup.ProjectSetupError, "another project"):
            self.run_setup(["yes", str(other)], project="fixture-project")
        self.assertEqual(setup.status(self.root)["stage"], "awaiting_client")
        self.assertFalse((self.root / setup.CLIENT_FILE).exists())

    def test_local_import_wins_without_cloud_signin_after_manual_confirmation(self):
        self.write_client(self.root / setup.CLIENT_FILE)
        value = self.run_setup(["yes"])
        self.assertEqual(value["method"], "imported"); self.assertEqual(value["cloud_checks"], "self_reported")
        self.factory.assert_not_called()
        self.assertEqual(self.run_setup(), value)

    def test_status_reads_sidecar_without_client_or_credentials(self):
        self.ready()
        self.factory.reset_mock()
        for name in (setup.CLIENT_FILE, "colab-oauth.json", "drive-oauth.json"):
            (self.root / name).write_text("PRIVATE-INVALID-CONTENT")
        with mock.patch.object(setup, "_client", side_effect=AssertionError("status must not read clients")):
            value = setup.status(self.root)
        self.assertEqual(value, {"mode": "user_project", "project_id": "fixture-project", "project_mode": "existing",
                                 "stage": "ready", "cloud_checks": "online_verified"})
        self.factory.assert_not_called()

    def test_malformed_oversized_and_non_desktop_files_are_rejected_safely(self):
        for body in ("PRIVATE-NOT-JSON", "x" * 16385, json.dumps({"web": {"client_secret": "PRIVATE"}})):
            self.download.write_text(body)
            with self.assertRaises(setup.ProjectSetupError) as error:
                self.run_setup(client_json=self.download)
            self.assert_no_leaks(error.exception)
        self.factory.assert_not_called()

    def test_tampered_sidecar_and_client_binding_refuse_without_cloud(self):
        self.ready(); self.factory.reset_mock()
        self.write_client(self.root / setup.CLIENT_FILE, identity="123456-tampered.apps.googleusercontent.com")
        with self.assertRaisesRegex(setup.ProjectSetupError, "recorded setup binding"):
            self.run_setup()
        record = json.loads((self.root / setup.SETUP_FILE).read_text()); record["stage"] = {"PRIVATE": "bad"}
        (self.root / setup.SETUP_FILE).write_text(json.dumps(record))
        with self.assertRaises(setup.ProjectSetupError) as error:
            self.run_setup()
        self.assertNotIn("PRIVATE", str(error.exception)); self.factory.assert_not_called()

    def test_symlink_download_is_rejected_before_cloud_calls(self):
        link = self.base / "linked.json"
        try:
            link.symlink_to(self.download)
        except OSError:
            self.skipTest("symlink creation unavailable on this host")
        with self.assertRaisesRegex(setup.ProjectSetupError, "symbolic links"):
            self.run_setup(client_json=link)
        self.factory.assert_not_called()

    def test_windows_pasted_paths_translate_for_wsl_and_strip_matching_quotes(self):
        with mock.patch.object(setup.sys, "platform", "linux"):
            self.assertEqual(setup._path_text('"C:\\Users\\Example\\Downloads\\client.json"'), "/mnt/c/Users/Example/Downloads/client.json")
            self.assertEqual(setup._path_text("'/home/user/client.json'"), "/home/user/client.json")




    def detected_directory(self):
        directory = self.base / "Downloads"
        directory.mkdir(exist_ok=True)
        return directory

    def test_unique_download_imports_without_a_path_question_and_preserves_download(self):
        directory = self.detected_directory()
        path = directory / "client_secret_fixture.apps.googleusercontent.com.json"
        self.write_client(path)
        before = path.read_bytes()
        with mock.patch("colabcola.client_downloads.download_directories", return_value=[directory]), \
             mock.patch("colabcola.client_handoff.time.sleep"):
            result = self.run_setup(["yes"], project="fixture-project")
        self.assertEqual(result["stage"], "ready")
        self.assertEqual(len(self.questions), 1)  # Cloud change approval only.
        self.assertIn("Detected Desktop OAuth JSON", self.messages.getvalue())
        self.assertEqual(path.read_bytes(), before)
        self.assert_no_leaks()

    def test_identical_duplicate_downloads_do_not_require_a_choice(self):
        directory = self.detected_directory()
        for name in ("client_secret_fixture.json", "client_secret_fixture (1).json"):
            self.write_client(directory / name)
        with mock.patch("colabcola.client_downloads.download_directories", return_value=[directory]), \
             mock.patch("colabcola.client_handoff.time.sleep"):
            result = self.run_setup(["yes"], project="fixture-project")
        self.assertEqual(result["stage"], "ready")
        self.assertEqual(len(self.questions), 1)
        self.assert_no_leaks()

    def test_different_matching_downloads_require_an_explicit_choice(self):
        directory = self.detected_directory()
        self.write_client(directory / "client_secret_a.json")
        second = directory / "client_secret_b.json"
        self.write_client(second, identity="123456-second.apps.googleusercontent.com")
        with mock.patch("colabcola.client_downloads.download_directories", return_value=[directory]):
            result = self.run_setup(["yes", "2"], project="fixture-project")
        self.assertEqual(result["stage"], "ready")
        self.assertEqual(setup._client(self.root / setup.CLIENT_FILE)[1], setup._client(second)[1])
        self.assertIn("More than one different Desktop client", self.messages.getvalue())
        self.assert_no_leaks()

    def test_partial_matching_download_prevents_automatic_import(self):
        directory = self.detected_directory()
        self.write_client(directory / "client_secret_valid.json")
        (directory / "client_secret_partial.json").write_text('{"installed":', encoding="utf-8")
        with mock.patch("colabcola.client_downloads.download_directories", return_value=[directory]):
            with self.assertRaisesRegex(setup.ProjectSetupError, "intentionally paused"):
                self.run_setup(["yes", ""], project="fixture-project")
        self.assertEqual(setup.status(self.root)["stage"], "awaiting_client")
        self.assertFalse((self.root / setup.CLIENT_FILE).exists())
        self.assertIn("check was incomplete", self.messages.getvalue())
        self.assert_no_leaks()

    def test_bound_missing_client_auto_restores_only_exact_saved_client(self):
        saved = self.ready()
        (self.root / "profile.json").write_text('{"schema":1,"drive_parent":"fixture-parent"}', encoding="utf-8")
        profile = (self.root / "profile.json").read_bytes()
        (self.root / setup.CLIENT_FILE).unlink()
        directory = self.detected_directory()
        self.write_client(directory / "client_secret_original.json")
        self.write_client(directory / "client_secret_other.json", identity="123456-other.apps.googleusercontent.com")
        self.factory.reset_mock(); self.questions = []
        with mock.patch("colabcola.client_downloads.download_directories", return_value=[directory]), \
             mock.patch("colabcola.client_handoff.time.sleep"):
            restored = self.run_setup()
        self.assertEqual(restored, saved)
        self.assertEqual(self.questions, [])
        self.factory.assert_not_called()
        self.assertEqual((self.root / "profile.json").read_bytes(), profile)
        self.assert_no_leaks()

    def test_bound_missing_client_is_not_replaced_by_same_project_download(self):
        saved = self.ready()
        (self.root / "profile.json").write_text('{"schema":1,"drive_parent":"fixture-parent"}', encoding="utf-8")
        (self.root / setup.CLIENT_FILE).unlink()
        directory = self.detected_directory()
        self.write_client(directory / "client_secret_other.json", identity="123456-other.apps.googleusercontent.com")
        with mock.patch("colabcola.client_downloads.download_directories", return_value=[directory]):
            with self.assertRaisesRegex(setup.ProjectSetupError, "intentionally paused"):
                self.run_setup(["later"])
        self.assertFalse((self.root / setup.CLIENT_FILE).exists())
        self.assertEqual(json.loads((self.root / setup.SETUP_FILE).read_text())["client_sha256"], saved["client_sha256"])
        self.assert_no_leaks()

    def test_existing_client_explicit_client_and_opt_out_never_discover_downloads(self):
        with mock.patch("colabcola.client_downloads.download_directories", side_effect=AssertionError("no Downloads access")) as discover:
            self.ready()
            self.run_setup()
            self.root = self.base / "manual-config"
            result = self.run_setup(["yes", str(self.download)], project="fixture-project", auto_client=False)
            self.assertEqual(result["stage"], "ready")
            discover.assert_not_called()
        self.assert_no_leaks()


if __name__ == "__main__":
    unittest.main()
