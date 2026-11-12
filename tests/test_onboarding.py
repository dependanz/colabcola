"""Isolated first-login acceptance; no user configuration, OAuth or Drive calls."""
import contextlib
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

from googleapiclient.errors import HttpError

from colabcola import onboarding
from colabcola import drive_oauth
from colabcola import lifecycle_provider as provider
from colabcola.google_drive import GOOGLE_DRIVE_SCOPE
from colabcola.lifecycle_io import LifecycleError, atomic


_REAL_LOGIN_CLIENT_CONFIGURATION = onboarding._login_client_configuration


class OnboardingTests(unittest.TestCase):
    def setUp(self):
        discovery = mock.patch("colabcola.client_downloads.download_directories", return_value=[])
        discovery.start(); self.addCleanup(discovery.stop)
        # Ordinary login tests are noninteractive regardless of the test host.
        # The dedicated terminal-mode tests override this explicitly.
        terminal = mock.patch("sys.stdin", io.StringIO())
        terminal.start(); self.addCleanup(terminal.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.home = self.root / "empty-profile"
        patch = mock.patch.dict(os.environ, {"COLABCOLA_CONFIG_HOME": str(self.home)})
        patch.start(); self.addCleanup(patch.stop)
        self.scopes = list(provider.SCOPES)
        self.credential = SimpleNamespace(valid=True, token="PRIVATE-ACCESS-TOKEN", refresh_token="PRIVATE-REFRESH-TOKEN",
            expiry=datetime.now(timezone.utc) + timedelta(hours=1), granted_scopes=self.scopes,
            has_scopes=lambda scopes: set(scopes) <= set(self.scopes), refresh=mock.Mock(),
            to_json=lambda: json.dumps({"token": "PRIVATE-ACCESS-TOKEN", "refresh_token": "PRIVATE-REFRESH-TOKEN"}))
        self.drive_credential = SimpleNamespace(token="PRIVATE-DRIVE-TOKEN",
            expiry=datetime.now(timezone.utc) + timedelta(hours=1))
        self.account = {"sub": "test-google-subject", "email": "account@example.test"}
        self.folder = {"id": "owned-folder", "name": "colabcola", "mimeType": "application/vnd.google-apps.folder",
                       "ownedByMe": True, "trashed": False, "capabilities": {"canAddChildren": True},
                       "appProperties": {"colabcola.role": "workspace", "colabcola.schema": "1"}}
        self.files = mock.Mock()
        self.files.list.return_value.execute.return_value = {"files": []}
        self.files.create.return_value.execute.return_value = {"id": "owned-folder"}
        self.files.get.return_value.execute.return_value = self.folder
        self.client = SimpleNamespace(assignments=mock.Mock(return_value=[]), close=mock.Mock(),
                                      allocate=mock.Mock(), release=mock.Mock())
        self.patches = {}
        for name, value in (("colabcola.onboarding._check_host", None),
                            ("colabcola.onboarding._login_client_configuration", {"mode": "user_project", "project_id": "fixture-project", "cloud_checks": "verified"}),
                            ("colabcola.lifecycle_provider.dependency_report", {}),
                            ("colabcola.lifecycle_provider.login", None),
                            ("colabcola.lifecycle_provider.credentials", self.credential),
                            ("colabcola.lifecycle_provider.GoogleProvider", self.client),
                            ("colabcola.drive_oauth.preflight_client", {}),
                            ("colabcola.drive_oauth.credentials", self.drive_credential),
                            ("colabcola.drive_oauth.login", self.drive_credential),
                            ("colabcola.drive_oauth.identity", self.account),
                            ("colabcola.onboarding._drive_files", self.files),
                            ("colabcola.lifecycle_provider.auth_path", self.home / "colab-oauth.json")):
            patch = mock.patch(name, return_value=value)
            self.patches[name] = patch.start(); self.addCleanup(patch.stop)
        self.stdout, self.stderr = io.StringIO(), io.StringIO()

    def run_login(self, *args):
        with contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            return onboarding.main(list(args))

    def assert_no_secrets(self):
        text = self.stdout.getvalue() + self.stderr.getvalue()
        for path in self.home.glob("*.json"):
            if path.name not in {"colab-oauth.json", "drive-oauth.json"}:
                text += path.read_text()
        self.assertNotIn("PRIVATE-", text)
        self.client.allocate.assert_not_called(); self.client.release.assert_not_called()

    def http_error(self, body, status=403):
        # Use the SDK's actual exception, whose normal string includes URI and private messages.
        error = HttpError(SimpleNamespace(status=status, reason="PRIVATE-HTTP-REASON"), b"{}",
                          uri="https://example.invalid/PRIVATE-URI?access_token=PRIVATE-ACCESS-TOKEN")
        error.content = json.dumps(body).encode() if not isinstance(body, bytes) else body
        return error

    def test_project_setup_precedes_google_consent_and_gets_optional_arguments(self):
        project = self.patches["colabcola.onboarding._login_client_configuration"]
        def configure(root, profile, selected_project, client_json):
            self.patches["colabcola.lifecycle_provider.login"].assert_not_called()
            self.patches["colabcola.drive_oauth.login"].assert_not_called()
            return {"mode": "user_project", "project_id": selected_project, "cloud_checks": "verified"}
        project.side_effect = configure
        client = self.root / "downloaded-desktop.json"
        self.assertEqual(self.run_login("--project", "fixture-project", "--drive-client", str(client)), 0)
        project.assert_called_once_with(self.home, None, "fixture-project", client)
        self.assertEqual(json.loads(self.stdout.getvalue())["project_setup"]["project_id"], "fixture-project")

    def test_setup_failure_stops_before_google_consent_or_folder_creation(self):
        self.patches["colabcola.onboarding._login_client_configuration"].side_effect = onboarding.OnboardingError("Setup paused; rerun colabcola login")
        self.assertEqual(self.run_login(), 2)
        self.assertIn("Setup paused", self.stderr.getvalue())
        self.patches["colabcola.lifecycle_provider.credentials"].assert_not_called()
        self.patches["colabcola.drive_oauth.login"].assert_not_called()
        self.files.create.assert_not_called()

    def test_management_account_mismatch_stops_before_drive_authorization(self):
        self.patches["colabcola.onboarding._login_client_configuration"].return_value = {
            "mode": "user_project", "cloud_account_email": "different@example.test"}
        self.assertEqual(self.run_login(), 2)
        self.assertIn("account used for Google Cloud project setup", self.stderr.getvalue())
        self.patches["colabcola.drive_oauth.credentials"].assert_not_called()
        self.files.create.assert_not_called()
        self.assertFalse((self.home / "profile.json").exists())

    def test_existing_shared_profile_reuses_client_without_project_setup(self):
        atomic(self.home / "profile.json", {"schema": 1, "drive_parent": "owned-folder"})
        self.patches["colabcola.onboarding._login_client_configuration"].side_effect = _REAL_LOGIN_CLIENT_CONFIGURATION
        before = (self.home / "profile.json").read_bytes()
        with mock.patch("colabcola.project_setup.ensure_personal_client", side_effect=AssertionError("unexpected setup")):
            self.assertEqual(self.run_login(), 0, self.stderr.getvalue())
        self.assertEqual(json.loads(self.stdout.getvalue())["project_setup"]["mode"], "legacy_shared_client")
        self.assertIn("legacy", self.stderr.getvalue())
        self.assertEqual((self.home / "profile.json").read_bytes(), before)
        self.assertFalse((self.home / "project-setup.json").exists())
        self.files.create.assert_not_called()

    def test_existing_local_client_profile_reuses_without_project_setup(self):
        atomic(self.home / "profile.json", {"schema": 1, "drive_parent": "owned-folder"})
        (self.home / "drive-client.json").write_text("local client fixture", encoding="utf-8")
        self.patches["colabcola.drive_oauth.preflight_client"].return_value = {"installed": {"project_id": "fixture-project"}}
        self.patches["colabcola.onboarding._login_client_configuration"].side_effect = _REAL_LOGIN_CLIENT_CONFIGURATION
        before = (self.home / "drive-client.json").read_bytes()
        with mock.patch("colabcola.project_setup.ensure_personal_client", side_effect=AssertionError("unexpected setup")):
            self.assertEqual(self.run_login(), 0, self.stderr.getvalue())
        self.assertEqual(json.loads(self.stdout.getvalue())["project_setup"]["mode"], "existing_user_client")
        self.assertEqual((self.home / "drive-client.json").read_bytes(), before)
        self.assertFalse((self.home / "project-setup.json").exists())
        self.files.create.assert_not_called()

    def test_fresh_login_creates_profile_helper_and_folder_after_consent(self):
        self.patches["colabcola.lifecycle_provider.credentials"].side_effect = [LifecycleError("absent"), self.credential]
        self.patches["colabcola.drive_oauth.credentials"].side_effect = drive_oauth.DriveOAuthError("Drive login required")
        self.assertEqual(self.run_login(), 0, self.stderr.getvalue())
        self.patches["colabcola.lifecycle_provider.login"].assert_called_once_with(scopes=self.scopes)
        self.patches["colabcola.drive_oauth.login"].assert_called_once_with(login_hint=self.account["email"], expected_subject=self.account["sub"])
        self.patches["colabcola.onboarding._drive_files"].assert_called_once_with(self.drive_credential)
        self.assertEqual(self.patches["colabcola.drive_oauth.identity"].call_args_list,
                         [mock.call(self.credential), mock.call(self.drive_credential)])
        result = json.loads(self.stdout.getvalue())
        self.assertEqual(result["account_email"], self.account["email"])
        self.assertTrue(result["same_google_account_verified"])
        self.assertEqual(onboarding.load_profile(), {"schema": 1, "drive_parent": "owned-folder"})
        self.assertEqual(json.loads((self.home / "drive-helper.json").read_text()),
                         [str(Path(sys.executable).absolute()), "-m", "colabcola", "_drive-token", "--config-home", str(self.home)])
        self.files.create.assert_called_once()
        self.client.assignments.assert_called_once(); self.client.close.assert_called_once()
        self.assert_no_secrets()

    def test_repeated_login_checks_exact_folder_without_consent_or_creation(self):
        self.files.list.return_value.execute.return_value = {"files": [self.folder]}
        self.assertEqual(self.run_login(), 0)
        before = (self.home / "profile.json").read_bytes()
        self.files.reset_mock()
        self.files.get.return_value.execute.return_value = self.folder
        self.assertEqual(self.run_login(), 0)
        self.files.list.assert_not_called(); self.files.create.assert_not_called()
        self.files.get.assert_called_once_with(fileId="owned-folder", fields=onboarding._FOLDER_FIELDS)
        self.patches["colabcola.lifecycle_provider.login"].assert_not_called()
        self.patches["colabcola.drive_oauth.login"].assert_not_called()
        self.assertEqual((self.home / "profile.json").read_bytes(), before)
        self.assert_no_secrets()

    def test_force_reauthorizes(self):
        self.assertEqual(self.run_login("--force"), 0)
        self.patches["colabcola.lifecycle_provider.login"].assert_called_once_with(scopes=self.scopes)
        self.patches["colabcola.drive_oauth.login"].assert_called_once_with(login_hint=self.account["email"], expected_subject=self.account["sub"])
        self.patches["colabcola.drive_oauth.credentials"].assert_not_called()

    def test_missing_publisher_client_refuses_before_any_consent_or_credentials(self):
        self.patches["colabcola.drive_oauth.preflight_client"].side_effect = drive_oauth.DriveOAuthError("Publisher Drive OAuth client is not configured")
        self.assertEqual(self.run_login(), 2)
        self.assertIn("Drive OAuth client configuration", self.stderr.getvalue())
        self.assertIn("Publisher Drive OAuth client is not configured", self.stderr.getvalue())
        for name in ("colabcola.lifecycle_provider.credentials", "colabcola.lifecycle_provider.login",
                     "colabcola.drive_oauth.credentials", "colabcola.drive_oauth.login", "colabcola.drive_oauth.identity"):
            self.patches[name].assert_not_called()
        self.client.assignments.assert_not_called()
        self.files.create.assert_not_called()
        self.assertFalse(self.home.exists())
        self.assert_no_secrets()

    def test_account_mismatch_refuses_before_folder_or_profile_mutations(self):
        self.patches["colabcola.drive_oauth.identity"].side_effect = [self.account,
            {"sub": "different-subject", "email": self.account["email"]}]
        self.assertEqual(self.run_login(), 2)
        self.assertIn("different Google accounts", self.stderr.getvalue())
        self.patches["colabcola.onboarding._drive_files"].assert_not_called()
        self.files.list.assert_not_called(); self.files.create.assert_not_called()
        self.assertFalse((self.home / "profile.json").exists())
        self.assertFalse((self.home / "drive-helper.json").exists())
        self.assert_no_secrets()

    def test_account_mismatch_preserves_previous_profile_helper_and_destination(self):
        self.assertEqual(self.run_login(), 0)
        before = {p.name: p.read_bytes() for p in self.home.iterdir()}
        self.files.reset_mock()
        self.patches["colabcola.onboarding._drive_files"].reset_mock()
        self.patches["colabcola.drive_oauth.identity"].side_effect = [self.account,
            {"sub": "different-subject", "email": "another@example.test"}]
        self.assertEqual(self.run_login("--force", "--drive-parent", "replacement-folder"), 2)
        self.assertEqual({p.name: p.read_bytes() for p in self.home.iterdir()}, before)
        self.patches["colabcola.onboarding._drive_files"].assert_not_called()
        self.files.get.assert_not_called(); self.files.create.assert_not_called()
        self.assert_no_secrets()

    def test_legacy_combined_credential_is_reused_only_for_colab(self):
        self.scopes.append(GOOGLE_DRIVE_SCOPE)
        self.patches["colabcola.drive_oauth.credentials"].side_effect = drive_oauth.DriveOAuthError("Drive login required")
        self.assertEqual(self.run_login(), 0)
        self.patches["colabcola.lifecycle_provider.login"].assert_not_called()
        self.patches["colabcola.drive_oauth.login"].assert_called_once_with(login_hint=self.account["email"], expected_subject=self.account["sub"])
        self.patches["colabcola.onboarding._drive_files"].assert_called_once_with(self.drive_credential)
        self.assert_no_secrets()

    def test_drive_consent_failure_preserves_previous_profile_and_helper(self):
        self.assertEqual(self.run_login(), 0)
        before = {p.name: p.read_bytes() for p in self.home.iterdir()}
        self.files.reset_mock()
        self.patches["colabcola.drive_oauth.login"].side_effect = drive_oauth.DriveOAuthError("Drive permissions were not granted; repeat Drive consent")
        self.assertEqual(self.run_login("--force"), 2)
        self.assertIn("Drive authorization", self.stderr.getvalue())
        self.assertIn("Drive permissions were not granted", self.stderr.getvalue())
        self.assertEqual({p.name: p.read_bytes() for p in self.home.iterdir()}, before)
        self.files.get.assert_not_called(); self.files.create.assert_not_called()
        self.assert_no_secrets()

    def test_missing_or_partial_scope_refuses_profile(self):
        self.credential.granted_scopes = provider.SCOPES[:-1]
        self.assertEqual(self.run_login(), 2)
        self.assertFalse((self.home / "profile.json").exists())
        self.files.create.assert_not_called()
        self.assert_no_secrets()

    def test_provider_partial_consent_uses_authored_guidance_without_publishing_profile(self):
        self.patches["colabcola.lifecycle_provider.login"].side_effect = provider.AuthorizationScopeError()
        self.assertEqual(self.run_login("--force"), 2)
        self.assertIn("Google did not grant all requested permissions", self.stderr.getvalue())
        self.assertIn("colabcola login --force", self.stderr.getvalue())
        self.assertFalse((self.home / "profile.json").exists())
        self.files.create.assert_not_called()
        self.assert_no_secrets()

    def test_missing_refresh_token_refuses_profile(self):
        self.credential.refresh_token = None
        self.assertEqual(self.run_login(), 2)
        self.assertFalse((self.home / "profile.json").exists())
        self.assert_no_secrets()

    def test_ambiguous_folders_refuse_without_creation(self):
        self.files.list.return_value.execute.return_value = {"files": [self.folder, {**self.folder, "id": "another"}]}
        self.assertEqual(self.run_login(), 2)
        self.files.create.assert_not_called()
        self.assertFalse((self.home / "profile.json").exists())
        self.assert_no_secrets()

    def test_bad_saved_folder_preserves_existing_profile_and_helper(self):
        self.assertEqual(self.run_login(), 0)
        before = {p.name: p.read_bytes() for p in self.home.iterdir()}
        self.files.get.return_value.execute.return_value = {**self.folder, "trashed": True}
        self.assertEqual(self.run_login(), 2)
        self.assertEqual({p.name: p.read_bytes() for p in self.home.iterdir()}, before)
        self.assert_no_secrets()

    def test_colab_error_does_not_create_folder_or_disclose_provider_message(self):
        self.client.assignments.side_effect = LifecycleError("Bearer PRIVATE-ACCESS-TOKEN")
        self.assertEqual(self.run_login(), 2)
        self.client.close.assert_called_once()
        self.files.create.assert_not_called()
        self.assertFalse((self.home / "profile.json").exists())
        self.assert_no_secrets()

    def test_drive_failure_preserves_absent_profile(self):
        self.files.create.return_value.execute.side_effect = RuntimeError("access_token=PRIVATE-ACCESS-TOKEN")
        self.assertEqual(self.run_login(), 2)
        self.assertFalse((self.home / "profile.json").exists())
        self.assertFalse((self.home / "drive-helper.json").exists())
        self.assert_no_secrets()

    def test_drive_disabled_api_reports_machine_fields_and_action_without_reconsent(self):
        error = self.http_error({"error": {
            "code": 403, "message": "PRIVATE-ACCOUNT PRIVATE-PROJECT PRIVATE-TOKEN",
            "errors": [{"reason": "accessNotConfigured", "domain": "usageLimits", "message": "PRIVATE-ERROR"}],
            "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "SERVICE_DISABLED",
                         "domain": "googleapis.com", "metadata": {"service": "drive.googleapis.com",
                         "consumer": "projects/PRIVATE-CONSUMER", "activationUrl": "https://PRIVATE-URL"}}]}})
        self.files.list.return_value.execute.side_effect = error
        for _ in range(2):
            self.assertEqual(self.run_login(), 2)
        text = self.stderr.getvalue()
        for expected in ("Drive folder lookup", "HTTP 403", "SERVICE_DISABLED", "accessNotConfigured",
                         "domain=googleapis.com,usageLimits", "service=drive.googleapis.com",
                         "Drive API is disabled", "selected Desktop OAuth client", "repeating Google consent"):
            self.assertIn(expected, text)
        self.assertNotIn("--force", text)
        self.patches["colabcola.lifecycle_provider.login"].assert_not_called()
        self.files.create.assert_not_called()
        self.assertFalse((self.home / "profile.json").exists())
        self.assertFalse((self.home / "drive-helper.json").exists())
        self.assert_no_secrets()

    def test_each_drive_operation_has_its_own_failure_stage(self):
        cases = ((self.patches["colabcola.onboarding._drive_files"], "Drive client initialization"),
                 (self.files.list, "Drive folder lookup"),
                 (self.files.list.return_value.execute, "Drive folder lookup"),
                 (self.files.create, "Drive folder creation"),
                 (self.files.create.return_value.execute, "Drive folder creation"),
                 (self.files.get, "Drive folder verification"),
                 (self.files.get.return_value.execute, "Drive folder verification"))
        for operation, stage in cases:
            with self.subTest(stage=stage, operation=operation):
                self.stdout, self.stderr = io.StringIO(), io.StringIO()
                operation.side_effect = self.http_error({"error": {"message": "PRIVATE-MESSAGE"}})
                try:
                    self.assertEqual(self.run_login(), 2)
                    self.assertEqual(self.stderr.getvalue(), "[colabcola] Error: login: setup failed at " + stage + "; HTTP 403\n")
                    self.assertFalse((self.home / "profile.json").exists())
                    self.assertFalse((self.home / "drive-helper.json").exists())
                    self.assert_no_secrets()
                finally:
                    operation.side_effect = None

    def test_saved_folder_http_error_preserves_profile_and_helper_without_replacement(self):
        self.assertEqual(self.run_login(), 0)
        before = {p.name: p.read_bytes() for p in self.home.iterdir()}
        self.files.reset_mock()
        self.files.get.return_value.execute.side_effect = self.http_error({"error": {"errors": [{"reason": "notFound"}]}}, 404)
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        self.assertEqual(self.run_login(), 2)
        self.assertIn("Drive folder verification; HTTP 404; reason=notFound", self.stderr.getvalue())
        self.assertIn("--config-home NEW_DIRECTORY", self.stderr.getvalue())
        self.assertEqual({p.name: p.read_bytes() for p in self.home.iterdir()}, before)
        self.files.list.assert_not_called(); self.files.create.assert_not_called()
        self.assert_no_secrets()

    def test_google_reason_guidance_distinguishes_scopes_policy_and_quotas(self):
        for reason, status, expected, absent in (
                ("ACCESS_TOKEN_SCOPE_INSUFFICIENT", 403, "login --force", "administrator"),
                ("insufficientPermissions", 403, "login --force", "administrator"),
                ("domainPolicy", 403, "administrator", "--force"),
                ("rateLimitExceeded", 403, "wait for it to reset", "--force"),
                ("QUOTA_EXCEEDED", 429, "wait for it to reset", "--force"),
                ("storageQuotaExceeded", 403, "free storage", "--force")):
            with self.subTest(reason=reason):
                self.files.list.return_value.execute.side_effect = self.http_error({"error": {"errors": [{"reason": reason}]}}, status)
                self.stdout, self.stderr = io.StringIO(), io.StringIO()
                self.assertEqual(self.run_login(), 2)
                self.assertIn(expected, self.stderr.getvalue())
                self.assertNotIn(absent, self.stderr.getvalue())
                self.files.create.assert_not_called()
                self.assert_no_secrets()

    def test_unrecognized_http_payloads_are_redacted_and_bounded(self):
        hostile = {"error": {"message": "PRIVATE-MESSAGE", "errors": [None, {"reason": "PRIVATE-REASON", "domain": "PRIVATE-DOMAIN"}],
                              "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": ["PRIVATE-REASON"],
                                           "domain": {"PRIVATE-DOMAIN": "PRIVATE-SECRET"},
                                           "metadata": {"service": "PRIVATE-SERVICE", "consumer": "PRIVATE-CONSUMER"}},
                                          {"@type": "PRIVATE-TYPE", "reason": "SERVICE_DISABLED"}]}}
        bodies = (b"PRIVATE-NOT-JSON", b"\xffPRIVATE-INVALID-UTF8", b"[]", b"null", b'{"error":[]}',
                  b'{"error":{"errors":{},"details":"PRIVATE-DETAILS"}}', hostile,
                  {"error": {"details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "PRIVATE-REASON",
                                           "domain": "googleapis.com", "metadata": {"service": "drive.googleapis.com"}}]}},
                  b'{"error":{"errors":[{"reason":"SERVICE_DISABLED"}],"message":"' + b"x" * 65536 + b'"}}',
                  b"[" * 2000 + b"]" * 2000)
        for body in bodies:
            with self.subTest(body_type=type(body).__name__):
                self.files.list.return_value.execute.side_effect = self.http_error(body)
                self.stdout, self.stderr = io.StringIO(), io.StringIO()
                self.assertEqual(self.run_login(), 2)
                self.assertEqual(self.stderr.getvalue(), "[colabcola] Error: login: setup failed at Drive folder lookup; HTTP 403\n")
                self.files.create.assert_not_called()
                self.assert_no_secrets()

    def test_only_allowlisted_domains_and_services_are_printed_for_known_reason(self):
        self.files.list.return_value.execute.side_effect = self.http_error({"error": {
            "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "SERVICE_DISABLED",
                         "domain": "PRIVATE-DOMAIN", "metadata": {"service": "PRIVATE-SERVICE", "consumer": "PRIVATE-CONSUMER"}}]}})
        self.assertEqual(self.run_login(), 2)
        self.assertIn("reason=SERVICE_DISABLED", self.stderr.getvalue())
        self.assertNotIn("domain=", self.stderr.getvalue())
        self.assertNotIn("service=", self.stderr.getvalue())
        self.assert_no_secrets()

    def test_invalid_http_status_is_not_displayed(self):
        for status in (True, "403 PRIVATE-SECRET", 999, None):
            with self.subTest(status=status):
                self.files.list.return_value.execute.side_effect = self.http_error(b"{}", status)
                self.stdout, self.stderr = io.StringIO(), io.StringIO()
                self.assertEqual(self.run_login(), 2)
                self.assertEqual(self.stderr.getvalue(), "[colabcola] Error: login: setup failed at Drive folder lookup\n")
                self.assert_no_secrets()

    def test_authorization_and_colab_failures_report_the_stage_without_exception_text(self):
        self.patches["colabcola.lifecycle_provider.login"].side_effect = RuntimeError("PRIVATE-AUTHORIZATION")
        self.assertEqual(self.run_login("--force"), 2)
        self.assertIn("setup failed at Colab authorization", self.stderr.getvalue())
        self.patches["colabcola.lifecycle_provider.login"].side_effect = None
        self.client.assignments.side_effect = LifecycleError("PRIVATE-COLAB")
        self.assertEqual(self.run_login(), 2)
        self.assertIn("setup failed at Colab access verification", self.stderr.getvalue())
        self.files.create.assert_not_called()
        self.assert_no_secrets()

    def test_status_empty_and_configured_is_offline_without_credentials(self):
        self.assertEqual(self.run_login("--status"), 0)
        self.assertEqual(json.loads(self.stdout.getvalue())["state"], "not_configured")
        self.assertFalse(self.home.exists())
        self.patches["colabcola.lifecycle_provider.credentials"].assert_not_called()
        self.patches["colabcola.lifecycle_provider.dependency_report"].assert_not_called()
        self.patches["colabcola.drive_oauth.preflight_client"].assert_not_called()
        self.patches["colabcola.onboarding._login_client_configuration"].assert_not_called()
        self.patches["colabcola.drive_oauth.credentials"].assert_not_called()
        self.assertEqual(self.run_login(), 0)
        self.patches["colabcola.lifecycle_provider.credentials"].reset_mock()
        self.patches["colabcola.drive_oauth.credentials"].reset_mock()
        self.patches["colabcola.drive_oauth.identity"].reset_mock()
        self.stdout = io.StringIO()
        self.assertEqual(self.run_login("--status"), 0)
        self.assertEqual(json.loads(self.stdout.getvalue())["state"], "configured")
        self.patches["colabcola.lifecycle_provider.credentials"].assert_not_called()
        self.patches["colabcola.drive_oauth.credentials"].assert_not_called()
        self.patches["colabcola.drive_oauth.identity"].assert_not_called()
        self.assert_no_secrets()

    def test_incomplete_or_tampered_helper_is_not_configured(self):
        atomic(self.home / "profile.json", {"schema": 1, "drive_parent": "owned-folder"})
        self.assertEqual(self.run_login("--status"), 2)
        with self.assertRaises(LifecycleError): onboarding.execution_defaults()
        atomic(self.home / "drive-helper.json", ["malicious", "PRIVATE-ACCESS-TOKEN"])
        self.stdout = io.StringIO(); self.stderr = io.StringIO()
        self.assertEqual(self.run_login("--status"), 2)
        self.assertNotIn("PRIVATE-", self.stdout.getvalue() + self.stderr.getvalue())

    def test_config_home_override_does_not_read_ambient_profile(self):
        self.home.mkdir()
        (self.home / "profile.json").write_text("THIS MUST NOT BE READ")
        isolated = self.root / "override"
        self.assertEqual(self.run_login("--config-home", str(isolated), "--status"), 0)
        self.assertEqual(json.loads(self.stdout.getvalue())["state"], "not_configured")
        self.assertEqual(os.environ["COLABCOLA_CONFIG_HOME"], str(self.home))
        self.patches["colabcola.lifecycle_provider.credentials"].assert_not_called()
        self.assertFalse(isolated.exists())

    def test_defaults_are_credential_free(self):
        self.assertEqual(self.run_login(), 0)
        self.patches["colabcola.lifecycle_provider.credentials"].reset_mock()
        value = onboarding.execution_defaults()
        self.assertEqual(value, {"drive_parent": "owned-folder", "token_command": self.home / "drive-helper.json"})
        self.patches["colabcola.lifecycle_provider.credentials"].assert_not_called()

    def test_two_installations_share_one_profile_and_preserve_legacy_helper(self):
        first = self.root / "env-first" / "python"
        second = self.root / "env-second" / "python"
        for path in (first, second):
            path.parent.mkdir()
            path.write_text("fixture", encoding="utf-8")
        with mock.patch.object(onboarding.sys, "executable", str(first)):
            self.assertEqual(self.run_login(), 0, self.stderr.getvalue())
            original = (self.home / "drive-helper.json").read_bytes()
            initial = onboarding.execution_defaults()
            self.assertEqual(initial["token_command"], self.home / "drive-helper.json")
        with mock.patch.object(onboarding.sys, "executable", str(second)):
            switched = onboarding.execution_defaults()
            self.assertNotEqual(switched["token_command"], initial["token_command"])
            self.assertEqual(json.loads(switched["token_command"].read_text()), onboarding._helper_argv())
            self.assertEqual(self.run_login(), 0, self.stderr.getvalue())
            self.assertEqual((self.home / "drive-helper.json").read_bytes(), original)
            again = onboarding.execution_defaults()
            self.assertEqual(again, switched)
        with mock.patch.object(onboarding.sys, "executable", str(first)):
            self.assertEqual(onboarding.execution_defaults(), initial)
        self.assertEqual((self.home / "drive-helper.json").read_bytes(), original)
        self.assert_no_secrets()

    def test_status_on_second_installation_does_not_create_helper_or_read_tokens(self):
        first = self.root / "first-python"
        second = self.root / "second-python"
        first.touch(); second.touch()
        with mock.patch.object(onboarding.sys, "executable", str(first)):
            self.assertEqual(self.run_login(), 0)
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(onboarding.sys, "executable", str(second)):
            candidate = onboarding._install_helper_path()
            self.assertFalse(candidate.exists())
            self.patches["colabcola.drive_oauth.credentials"].reset_mock()
            self.patches["colabcola.lifecycle_provider.credentials"].reset_mock()
            self.assertEqual(self.run_login("--status"), 0, self.stderr.getvalue())
            self.assertFalse(candidate.exists())
            self.assertEqual(json.loads(self.stdout.getvalue())["credentials_checked"], False)
            self.patches["colabcola.drive_oauth.credentials"].assert_not_called()
            self.patches["colabcola.lifecycle_provider.credentials"].assert_not_called()
            onboarding.execution_defaults()
            self.assertTrue(candidate.exists())

    def test_modified_install_helper_is_rejected_without_replacing_it(self):
        first = self.root / "one-python"
        second = self.root / "two-python"
        first.touch(); second.touch()
        with mock.patch.object(onboarding.sys, "executable", str(first)):
            self.assertEqual(self.run_login(), 0)
        with mock.patch.object(onboarding.sys, "executable", str(second)):
            helper = onboarding.execution_defaults()["token_command"]
            atomic(helper, ["unexpected", "-m", "colabcola", "_drive-token",
                            "--config-home", str(self.home)])
            before = helper.read_bytes()
            with self.assertRaisesRegex(onboarding.OnboardingError, "Drive helper changed"):
                onboarding.execution_defaults()
            self.assertEqual(helper.read_bytes(), before)
        with mock.patch.object(onboarding.sys, "executable", str(first)):
            self.assertEqual(onboarding.execution_defaults()["token_command"], self.home / "drive-helper.json")

    def test_missing_current_interpreter_fails_before_creating_sidecar(self):
        first = self.root / "valid-python"
        missing = self.root / "deleted-python"
        first.touch()
        with mock.patch.object(onboarding.sys, "executable", str(first)):
            self.assertEqual(self.run_login(), 0)
        with mock.patch.object(onboarding.sys, "executable", str(missing)):
            candidate = onboarding._install_helper_path()
            with self.assertRaisesRegex(onboarding.OnboardingError, "interpreter is missing or moved"):
                onboarding.execution_defaults()
            self.assertFalse(candidate.exists())

    def test_profile_and_helper_schema_mismatches_name_both_versions(self):
        self.assertEqual(self.run_login(), 0)
        atomic(self.home / "profile.json", {"schema": 2, "drive_parent": "owned-folder"})
        with self.assertRaisesRegex(onboarding.OnboardingError, "profile schema 2; supported schema: 1"):
            onboarding.execution_defaults()
        atomic(self.home / "profile.json", {"schema": 1, "drive_parent": "owned-folder"})
        atomic(self.home / "drive-helper.json", {"schema": 2, "argv": onboarding._helper_argv()})
        with self.assertRaisesRegex(onboarding.OnboardingError, "Drive helper schema 2; supported schema: 1"):
            onboarding.execution_defaults()

    def test_legacy_helper_wrong_profile_is_rejected(self):
        self.assertEqual(self.run_login(), 0)
        argv = onboarding._helper_argv()
        argv[-1] = str(self.root / "other-account")
        atomic(self.home / "drive-helper.json", argv)
        with self.assertRaisesRegex(onboarding.OnboardingError, "Invalid Colabcola Drive helper"):
            onboarding.execution_defaults()

    def test_host_rejected_before_authorization(self):
        self.patches["colabcola.onboarding._check_host"].side_effect = onboarding.OnboardingError("Use Ubuntu WSL")
        self.assertEqual(self.run_login(), 2)
        self.patches["colabcola.lifecycle_provider.credentials"].assert_not_called()
        self.patches["colabcola.lifecycle_provider.login"].assert_not_called()
        self.assertFalse(self.home.exists())

    def test_token_protocol_refreshes_early_without_prompting(self):
        self.assertEqual(self.run_login(), 0)
        drive_credentials = self.patches["colabcola.drive_oauth.credentials"]
        drive_credentials.reset_mock()
        self.patches["colabcola.lifecycle_provider.credentials"].reset_mock()
        self.patches["colabcola.lifecycle_provider.credentials"].side_effect = AssertionError("Colab credential read by Drive helper")
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            code = onboarding.token_main([])
        self.assertEqual(code, 0, self.stderr.getvalue())
        drive_credentials.assert_called_once_with(minimum_seconds=1860)
        self.patches["colabcola.lifecycle_provider.credentials"].assert_not_called()
        self.patches["colabcola.lifecycle_provider.login"].assert_not_called()
        self.patches["colabcola.drive_oauth.login"].assert_not_called()
        self.assertEqual(json.loads(self.stdout.getvalue())["access_token"], "PRIVATE-DRIVE-TOKEN")
        self.assertEqual(self.stderr.getvalue(), "")
        self.assertFalse((self.home / "colab-oauth.json").exists())

    def test_colab_renewal_remains_separate_from_drive_credentials(self):
        self.credential.expiry = datetime.now(timezone.utc) + timedelta(seconds=300)
        def refresh(request):
            self.credential.expiry = datetime.now(timezone.utc) + timedelta(hours=1)
        self.credential.refresh.side_effect = refresh
        self.assertEqual(self.run_login(), 0)
        self.credential.refresh.assert_called_once()
        self.patches["colabcola.drive_oauth.credentials"].assert_called_once_with(minimum_seconds=600)
        self.patches["colabcola.lifecycle_provider.login"].assert_not_called()
        self.patches["colabcola.drive_oauth.login"].assert_not_called()
        self.assertEqual(json.loads((self.home / "colab-oauth.json").read_text())["token"], "PRIVATE-ACCESS-TOKEN")
        self.assertFalse((self.home / "drive-oauth.json").exists())
        self.assert_no_secrets()

    def test_token_failure_does_not_emit_credentials_or_prompt(self):
        self.assertEqual(self.run_login(), 0)
        self.patches["colabcola.drive_oauth.credentials"].side_effect = RuntimeError("PRIVATE-DRIVE-TOKEN")
        self.patches["colabcola.lifecycle_provider.credentials"].reset_mock()
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            self.assertEqual(onboarding.token_main([]), 2)
        self.assertEqual(self.stdout.getvalue(), "")
        self.patches["colabcola.lifecycle_provider.login"].assert_not_called()
        self.patches["colabcola.lifecycle_provider.credentials"].assert_not_called()
        self.patches["colabcola.drive_oauth.login"].assert_not_called()
        self.assert_no_secrets()

    def test_interrupted_login_does_not_publish_profile(self):
        self.patches["colabcola.lifecycle_provider.credentials"].side_effect = KeyboardInterrupt
        self.assertEqual(self.run_login(), 130)
        self.assertFalse((self.home / "profile.json").exists())
        self.assert_no_secrets()

    def test_explicit_folder_selects_owned_destination_without_discovery_or_creation(self):
        self.files.get.return_value.execute.return_value = {**self.folder, "id": "selected-folder", "appProperties": {}}
        self.assertEqual(self.run_login("--drive-parent", "selected-folder"), 0)
        self.assertEqual(onboarding.load_profile()["drive_parent"], "selected-folder")
        self.files.list.assert_not_called(); self.files.create.assert_not_called()
        self.assertEqual(self.run_login(), 0)
        self.assertEqual(onboarding.load_profile()["drive_parent"], "selected-folder")
        self.assert_no_secrets()

    def test_explicit_invalid_parent_is_rejected_before_credentials(self):
        self.assertEqual(self.run_login("--drive-parent", "folder/../../other"), 2)
        self.patches["colabcola.lifecycle_provider.credentials"].assert_not_called()
        self.files.create.assert_not_called()

    def test_colab_auth_link_is_refused_before_login_credential_read(self):
        self.assertEqual(self.run_login(), 0)
        credentials = self.patches["colabcola.lifecycle_provider.credentials"]
        credentials.reset_mock()
        safe_check = onboarding._no_links
        def reject_auth(path):
            if path == self.home / "colab-oauth.json":
                raise onboarding.OnboardingError("Colabcola configuration cannot use links or junctions")
            return safe_check(path)
        with mock.patch.object(onboarding, "_no_links", side_effect=reject_auth):
            self.assertEqual(self.run_login(), 2)
            self.assertEqual(self.run_login("--force"), 2)
        credentials.assert_not_called()
        self.patches["colabcola.lifecycle_provider.login"].assert_not_called()
        self.assert_no_secrets()

    def test_drive_auth_link_is_refused_before_login_or_renewal_credential_read(self):
        self.assertEqual(self.run_login(), 0)
        credentials = self.patches["colabcola.drive_oauth.credentials"]
        credentials.reset_mock()
        safe_check = onboarding._no_links
        def reject_auth(path):
            if path == self.home / "drive-oauth.json":
                raise onboarding.OnboardingError("Colabcola configuration cannot use links or junctions")
            return safe_check(path)
        with mock.patch.object(onboarding, "_no_links", side_effect=reject_auth):
            self.assertEqual(self.run_login(), 2)
            self.assertEqual(self.run_login("--force"), 2)
            with contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
                self.assertEqual(onboarding.token_main([]), 2)
        credentials.assert_not_called()
        self.patches["colabcola.drive_oauth.login"].assert_not_called()
        self.assert_no_secrets()

    def test_real_auth_symlink_cannot_read_outside_configuration(self):
        self.assertEqual(self.run_login(), 0)
        outside = self.root / "outside-credential.json"
        outside.write_text("PRIVATE-OUTSIDE-CREDENTIAL")
        try:
            (self.home / "colab-oauth.json").symlink_to(outside)
        except OSError as exc:
            self.skipTest("host cannot create test symlink: " + type(exc).__name__)
        credentials = self.patches["colabcola.lifecycle_provider.credentials"]
        credentials.reset_mock()
        self.assertEqual(self.run_login("--force"), 2)
        credentials.assert_not_called()
        self.patches["colabcola.lifecycle_provider.login"].assert_not_called()
        self.assertEqual(outside.read_text(), "PRIVATE-OUTSIDE-CREDENTIAL")
        self.assert_no_secrets()

    def test_real_drive_auth_symlink_cannot_read_outside_configuration(self):
        self.assertEqual(self.run_login(), 0)
        outside = self.root / "outside-drive-credential.json"
        outside.write_text("PRIVATE-OUTSIDE-CREDENTIAL")
        try:
            (self.home / "drive-oauth.json").symlink_to(outside)
        except OSError as exc:
            self.skipTest("host cannot create test symlink: " + type(exc).__name__)
        credentials = self.patches["colabcola.drive_oauth.credentials"]
        credentials.reset_mock()
        self.assertEqual(self.run_login("--force"), 2)
        with contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            self.assertEqual(onboarding.token_main([]), 2)
        credentials.assert_not_called()
        self.patches["colabcola.drive_oauth.login"].assert_not_called()
        self.assertEqual(outside.read_text(), "PRIVATE-OUTSIDE-CREDENTIAL")
        self.assert_no_secrets()

    def test_malformed_profile_requires_explicit_force_and_preserves_bytes(self):
        self.home.mkdir()
        invalid = b'{"schema": 1, INVALID PRIVATE-SECRET'
        (self.home / "profile.json").write_bytes(invalid)
        self.assertEqual(self.run_login(), 2)
        self.assertIn("colabcola login --force", self.stderr.getvalue())
        self.assertEqual((self.home / "profile.json").read_bytes(), invalid)
        self.patches["colabcola.lifecycle_provider.credentials"].assert_not_called()
        self.patches["colabcola.lifecycle_provider.login"].assert_not_called()
        self.assertNotIn("PRIVATE-", self.stdout.getvalue() + self.stderr.getvalue())

    def test_force_repairs_malformed_profile_after_verified_setup(self):
        self.patches["colabcola.onboarding._login_client_configuration"].side_effect = _REAL_LOGIN_CLIENT_CONFIGURATION
        self.home.mkdir()
        (self.home / "profile.json").write_text("invalid profile")
        self.files.list.return_value.execute.return_value = {"files": [self.folder]}
        self.assertEqual(self.run_login("--force"), 0, self.stderr.getvalue())
        self.assertEqual(onboarding.load_profile(), {"schema": 1, "drive_parent": "owned-folder"})
        self.files.create.assert_not_called()
        self.patches["colabcola.lifecycle_provider.login"].assert_called_once_with(scopes=self.scopes)
        self.assert_no_secrets()

    def test_failed_force_repair_retains_original_profile(self):
        self.home.mkdir()
        invalid = b"invalid profile"
        (self.home / "profile.json").write_bytes(invalid)
        self.files.list.return_value.execute.side_effect = RuntimeError("PRIVATE-DRIVE-SECRET")
        self.assertEqual(self.run_login("--force"), 2)
        self.assertEqual((self.home / "profile.json").read_bytes(), invalid)
        self.assertFalse((self.home / "drive-helper.json").exists())
        self.assert_no_secrets()

    def test_force_cannot_replace_profile_symlink(self):
        self.home.mkdir()
        outside = self.root / "outside-profile.json"
        outside.write_text("invalid profile")
        try:
            (self.home / "profile.json").symlink_to(outside)
        except OSError as exc:
            self.skipTest("host cannot create test symlink: " + type(exc).__name__)
        self.assertEqual(self.run_login("--force"), 2)
        self.assertEqual(outside.read_text(), "invalid profile")
        self.patches["colabcola.lifecycle_provider.credentials"].assert_not_called()
        self.patches["colabcola.lifecycle_provider.login"].assert_not_called()

    def test_configuration_rejects_extra_fields_and_non_integer_schema(self):
        for value in ({"schema": True, "drive_parent": "folder"},
                      {"schema": 1, "drive_parent": "folder", "access_token": "PRIVATE-ACCESS-TOKEN"}):
            atomic(self.home / "profile.json", value)
            self.stdout, self.stderr = io.StringIO(), io.StringIO()
            self.assertEqual(self.run_login("--status"), 2)
            self.assertNotIn("PRIVATE-", self.stdout.getvalue() + self.stderr.getvalue())


    def test_login_does_not_start_documentation_services(self):
        for interactive in (False, True):
            with self.subTest(interactive=interactive), \
                 mock.patch("sys.stdin", SimpleNamespace(isatty=lambda: interactive)), \
                 mock.patch("socketserver.TCPServer.server_bind", side_effect=AssertionError("documentation server forbidden")) as bind, \
                 mock.patch("webbrowser.open", side_effect=AssertionError("documentation browser forbidden")) as browser:
                self.stdout, self.stderr = io.StringIO(), io.StringIO()
                self.assertEqual(self.run_login(), 0)
                self.assertEqual(json.loads(self.stdout.getvalue())["state"], "configured")
                self.assertNotIn("Local setup guide", self.stderr.getvalue())
                self.assertNotIn("127.0.0.1:", self.stderr.getvalue())
                bind.assert_not_called(); browser.assert_not_called()
                self.assert_no_secrets()

    def test_paused_setup_does_not_start_documentation_services(self):
        self.patches["colabcola.onboarding._login_client_configuration"].side_effect = onboarding.OnboardingError("Setup paused; rerun colabcola login")
        for interactive in (False, True):
            with self.subTest(interactive=interactive), \
                 mock.patch("sys.stdin", SimpleNamespace(isatty=lambda: interactive)), \
                 mock.patch("socketserver.TCPServer.server_bind", side_effect=AssertionError("documentation server forbidden")) as bind, \
                 mock.patch("webbrowser.open", side_effect=AssertionError("documentation browser forbidden")) as browser:
                self.stdout, self.stderr = io.StringIO(), io.StringIO()
                self.assertEqual(self.run_login(), 2)
                self.assertEqual(self.stderr.getvalue(), "[colabcola] Error: login: Setup paused; rerun colabcola login\n")
                bind.assert_not_called(); browser.assert_not_called()
        self.assertFalse(self.home.exists())
        self.patches["colabcola.lifecycle_provider.credentials"].assert_not_called()

    def test_removed_documentation_flags_and_help_are_offline(self):
        with mock.patch("colabcola.onboarding._selected_home", side_effect=AssertionError("no configuration access")):
            for flag in ("--guide", "--no-browser"):
                with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    onboarding.main([flag])
                self.assertEqual(caught.exception.code, 2)
            output = io.StringIO()
            with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as caught:
                onboarding.main(["--help"])
            self.assertEqual(caught.exception.code, 0)
            self.assertNotIn("--guide", output.getvalue()); self.assertNotIn("--no-browser", output.getvalue())
        self.assertFalse(self.home.exists())
        self.patches["colabcola.onboarding._check_host"].assert_not_called()


    def test_no_auto_client_flag_reaches_personal_setup(self):
        self.assertEqual(self.run_login("--no-auto-client"), 0)
        self.patches["colabcola.onboarding._login_client_configuration"].assert_called_once_with(self.home, None, None, None, auto_client=False)
        self.assert_no_secrets()

    def test_status_refuses_client_discovery_option_before_account_access(self):
        with mock.patch("colabcola.onboarding._selected_home", side_effect=AssertionError("no account access")), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            onboarding.main(["--status", "--no-auto-client"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()


