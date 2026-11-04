"""Cloud project setup with synthetic gcloud/HTTPS responses; no Google calls."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from colabcola.cloud_project import CloudProjectClient, CloudProjectError, LABEL, PERMISSIONS
from colabcola.lifecycle_io import LifecycleError


class CloudProjectTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "selected-profile"
        self.client = CloudProjectClient(self.root)
        self.email = "person@example.test"
        self.project_id = "colabcola-test-project"
        self.project = {"projectId": self.project_id, "projectNumber": "123456789",
                        "labels": {LABEL: "true"}, "lifecycleState": "ACTIVE"}
        self.accounts = [{"account": self.email, "status": "ACTIVE"}]
        patches = {
            "which": mock.patch("colabcola.cloud_project.shutil.which", return_value="gcloud"),
            "run": mock.patch("colabcola.cloud_project.subprocess.run", side_effect=AssertionError("unexpected subprocess")),
            "session": mock.patch("colabcola.cloud_project.requests.Session", side_effect=AssertionError("unexpected network")),
            "home": mock.patch.object(Path, "home", side_effect=AssertionError("ambient home forbidden")),
        }
        for name, patch in patches.items():
            setattr(self, name, patch.start())
            self.addCleanup(patch.stop)

    def ok(self, value):
        return subprocess.CompletedProcess([], 0, json.dumps(value), "")

    def raw(self, text="PRIVATE-MANAGEMENT-TOKEN"):
        return subprocess.CompletedProcess([], 0, text + "\n", "")

    def fail(self, code="PERMISSION_DENIED"):
        return subprocess.CompletedProcess([], 1, "PRIVATE-OUTPUT", "ERROR: " + code + ": PRIVATE-DETAILS")

    def bind(self):
        self.client.account = self.email

    @contextlib.contextmanager
    def permission_response(self, value=None, *, status=200, body=None):
        value = {"permissions": list(PERMISSIONS)} if value is None else value
        response = mock.Mock(status_code=status)
        response.iter_content.return_value = [json.dumps(value).encode() if body is None else body]
        factory = mock.MagicMock()
        session = factory.return_value.__enter__.return_value
        session.post.return_value.__enter__.return_value = response
        with mock.patch("colabcola.cloud_project.requests.Session", factory):
            yield session, response

    def assert_safe(self, error):
        self.assertNotIn("PRIVATE-", str(error.exception))
        self.assertNotIn("PRIVATE-", repr(vars(error.exception)))

    def test_availability_does_not_execute_or_create_config(self):
        self.assertTrue(self.client.available())
        self.run.assert_not_called()
        self.assertFalse(self.root.exists())
        self.which.return_value = None
        self.assertFalse(self.client.available())
        with self.assertRaises(CloudProjectError) as error:
            self.client.authenticate()
        self.assertIn("Install gcloud", str(error.exception))
        self.run.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_explicit_account_and_isolated_config_override_ambient_environment(self):
        self.bind()
        self.run.side_effect = [self.ok(self.project)]
        ambient = {"CLOUDSDK_CONFIG": "PRIVATE-AMBIENT-CONFIG", "CLOUDSDK_AUTH_ACCESS_TOKEN": "PRIVATE-AMBIENT-TOKEN",
                   "CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT": "PRIVATE-IMPERSONATION",
                   "CLOUDSDK_CORE_PROJECT": "wrong-project", "CLOUDSDK_API_ENDPOINT_OVERRIDES_CLOUDRESOURCEMANAGER": "PRIVATE-ENDPOINT",
                   "GOOGLE_APPLICATION_CREDENTIALS": "PRIVATE-ADC", "GCLOUD_PROJECT": "wrong-project"}
        with mock.patch.dict(os.environ, ambient):
            result = self.client.project(self.project_id)
        self.assertEqual(result["project_id"], self.project_id)
        argv, = self.run.call_args.args
        kwargs = self.run.call_args.kwargs
        self.assertIsInstance(argv, list)
        self.assertIn("--account=" + self.email, argv)
        self.assertFalse(kwargs["shell"])
        self.assertEqual(kwargs["timeout"], 60)
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertTrue(kwargs["capture_output"])
        self.assertEqual(kwargs["env"]["CLOUDSDK_CONFIG"], str(self.root / "cloud-setup" / "gcloud"))
        for key in ambient:
            if key != "CLOUDSDK_CONFIG":
                self.assertNotIn(key, kwargs["env"])
        self.assertEqual(kwargs["env"]["CLOUDSDK_CORE_LOG_HTTP"], "false")
        if os.name != "nt":
            for directory in (self.root, self.client.config.parent, self.client.config):
                self.assertEqual(directory.stat().st_mode & 0o777, 0o700)

    def test_project_operations_require_explicit_authentication(self):
        for operation in (lambda: self.client.projects(), lambda: self.client.project(self.project_id),
                          lambda: self.client.billing(self.project_id), lambda: self.client.ensure_drive(self.project_id)):
            with self.assertRaisesRegex(CloudProjectError, "Authenticate"):
                operation()
        self.run.assert_not_called()

    def test_reuses_active_account_without_interactive_login_or_config_change(self):
        self.run.side_effect = [self.ok(self.accounts), self.raw()]
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            self.assertEqual(self.client.authenticate(), self.email)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(self.client.account, self.email)
        self.assertEqual(self.run.call_count, 2)
        commands = [call.args[0] for call in self.run.call_args_list]
        self.assertEqual(commands[0][1:3], ["auth", "list"])
        self.assertEqual(commands[1][1:3], ["auth", "print-access-token"])
        self.assertIn("--account=" + self.email, commands[1])
        self.assertFalse(any("login" in command or "config" in command for command in commands))

    def test_explicit_inactive_account_does_not_switch_global_configuration(self):
        other = {"account": "other@example.test", "status": "ACTIVE"}
        self.run.side_effect = [self.ok([other, {"account": self.email, "status": ""}]), self.raw()]
        self.assertEqual(self.client.authenticate(account=self.email), self.email)
        for call in self.run.call_args_list:
            self.assertIn("--account=" + self.email, call.args[0])
        self.assertFalse(any("login" in call.args[0] for call in self.run.call_args_list))

    def test_fresh_login_is_the_only_interactive_command_and_has_timeout(self):
        self.run.side_effect = [self.ok([]), self.raw(""), self.ok([{ "account": self.email, "status": ""}]), self.raw()]
        self.assertEqual(self.client.authenticate(), self.email)
        interactive = [call for call in self.run.call_args_list if not call.kwargs["capture_output"]]
        self.assertEqual(len(interactive), 1)
        command = interactive[0]
        self.assertEqual(command.args[0][1:3], ["auth", "login"])
        for flag in ("--no-launch-browser", "--brief", "--no-activate"):
            self.assertIn(flag, command.args[0])
        self.assertEqual(command.kwargs["timeout"], 300)
        self.assertIsNone(command.kwargs["stdin"])
        self.assertIn("--account=" + self.email, self.run.call_args_list[-1].args[0])

    def test_revoked_account_uses_force_login_but_network_failure_does_not(self):
        self.run.side_effect = [self.ok(self.accounts), self.fail("invalid_grant"), self.raw(""),
                               self.ok(self.accounts), self.raw()]
        self.assertEqual(self.client.authenticate(), self.email)
        self.assertIn("--force", self.run.call_args_list[2].args[0])
        self.run.reset_mock()
        self.run.side_effect = [self.ok(self.accounts), self.fail("UNAVAILABLE")]
        with self.assertRaises(CloudProjectError) as error:
            self.client.authenticate()
        self.assert_safe(error)
        self.assertFalse(any("login" in call.args[0] for call in self.run.call_args_list))

    def test_ambiguous_or_wrong_login_account_refuses_without_project_operations(self):
        self.run.side_effect = [self.ok([{ "account": self.email, "status": ""},
                                        {"account": "other@example.test", "status": ""}])]
        with self.assertRaisesRegex(CloudProjectError, "Several"):
            self.client.authenticate()
        self.run.reset_mock()
        self.run.side_effect = [self.ok([]), self.raw(""), self.ok([{ "account": "other@example.test", "status": "ACTIVE"}])]
        with self.assertRaisesRegex(CloudProjectError, "selected account"):
            self.client.authenticate(account=self.email)
        self.assertFalse(any("projects" in call.args[0] for call in self.run.call_args_list))

    def test_invalid_user_arguments_cannot_reach_gcloud_or_a_shell(self):
        self.bind()
        for value in ("project & PRIVATE-COMMAND", "project\nPRIVATE", "--help", "a" * 31, "Uppercase"):
            with self.assertRaises(CloudProjectError):
                self.client.project(value)
        for value in ("user&whoami@example.test", "$(PRIVATE)@example.test", "user\n@example.test", "service@project.iam.gserviceaccount.com"):
            with self.assertRaises(CloudProjectError):
                self.client.authenticate(value)
        self.run.assert_not_called()

    def test_project_discovery_filters_labels_and_state_not_display_name(self):
        self.bind()
        unmarked = {**self.project, "projectId": "colabcola-other", "labels": {}, "name": "Colabcola"}
        inactive = {**self.project, "projectId": "colabcola-deleted", "lifecycleState": "DELETE_REQUESTED"}
        self.run.side_effect = [self.ok([unmarked, self.project, inactive])]
        result = self.client.projects()
        self.assertEqual([row["project_id"] for row in result], [self.project_id])
        self.assertIn("--filter=labels.colabcola-managed=true", self.run.call_args.args[0])
        self.assertNotIn("name", result[0])

    def test_direct_project_not_found_is_distinct_from_denied_or_unknown(self):
        self.bind()
        self.run.side_effect = [self.fail("NOT_FOUND"), self.fail("PERMISSION_DENIED"), self.fail("FAILED")]
        self.assertIsNone(self.client.project(self.project_id))
        for _ in range(2):
            with self.assertRaises(CloudProjectError) as error:
                self.client.project(self.project_id)
            self.assert_safe(error)

    def test_mismatched_or_malformed_project_metadata_is_rejected(self):
        self.bind()
        cases = [[], {**self.project, "projectId": "different-project"}, {**self.project, "projectNumber": True},
                 {**self.project, "labels": []}, {**self.project, "lifecycleState": None}]
        self.run.side_effect = [self.ok(value) for value in cases]
        for _ in cases:
            with self.assertRaises(CloudProjectError):
                self.client.project(self.project_id)

    def test_existing_managed_project_is_reused_only_after_create_conflict_and_verified_permissions(self):
        self.bind()
        self.run.side_effect = [self.fail("ALREADY_EXISTS"), self.ok(self.project), self.raw()]
        with self.permission_response():
            result = self.client.create(self.project_id)
        self.assertEqual(result["project_id"], self.project_id)
        self.assertEqual(self.run.call_args_list[0].args[0][1:4], ["projects", "create", self.project_id])
        self.assertEqual(self.run.call_args_list[1].args[0][1:4], ["projects", "describe", self.project_id])
        self.assertEqual(sum("create" in call.args[0] for call in self.run.call_args_list), 1)

    def test_create_uses_saved_id_label_no_billing_or_default_project_change(self):
        self.bind()
        self.run.side_effect = [self.raw(""), self.ok(self.project), self.raw()]
        with self.permission_response():
            self.assertEqual(self.client.create(self.project_id)["project_id"], self.project_id)
        commands = [call.args[0] for call in self.run.call_args_list]
        create = commands[0]
        self.assertEqual(create[1:4], ["projects", "create", self.project_id])
        self.assertIn("--labels=colabcola-managed=true", create)
        self.assertIn("--no-enable-cloud-apis", create)
        for command in commands:
            self.assertNotIn("--set-as-default", command)
            self.assertNotIn("billing", command)
        self.assertEqual(commands[1][1:4], ["projects", "describe", self.project_id])

    def test_uncertain_create_reconciles_direct_get_without_creating_a_second_project(self):
        self.bind()
        self.run.side_effect = [subprocess.TimeoutExpired("PRIVATE-COMMAND", 120,
                               output="PRIVATE-OUTPUT", stderr="PRIVATE-ERROR"), self.ok(self.project), self.raw()]
        with self.permission_response():
            self.assertEqual(self.client.create(self.project_id)["project_id"], self.project_id)
        commands = [call.args[0] for call in self.run.call_args_list]
        self.assertEqual(commands[0][1:4], ["projects", "create", self.project_id])
        self.assertEqual(sum("create" in command for command in commands), 1)
        self.assertFalse(any("list" in command for command in commands))

    def test_unconfirmed_create_preserves_stable_selection_and_actual_failure_cause(self):
        self.bind()
        self.run.side_effect = [self.fail("RESOURCE_EXHAUSTED"), self.fail("NOT_FOUND")]
        with self.assertRaises(CloudProjectError) as error:
            self.client.create(self.project_id)
        self.assert_safe(error)
        self.assertIn("same saved project ID", str(error.exception))
        self.assertIn("quota or project limit", str(error.exception))
        self.assertEqual(error.exception.code, "RESOURCE_EXHAUSTED")
        self.assertEqual(sum("create" in call.args[0] for call in self.run.call_args_list), 1)

    def test_name_collision_and_missing_permissions_are_never_adopted(self):
        self.bind()
        self.run.side_effect = [self.fail("ALREADY_EXISTS"), self.ok({**self.project, "labels": {}})]
        with self.assertRaisesRegex(CloudProjectError, "not an active Colabcola-managed"):
            self.client.create(self.project_id)
        self.session.assert_not_called()
        self.run.side_effect = [self.fail("ALREADY_EXISTS"), self.ok(self.project), self.raw()]
        with self.permission_response({"permissions": [PERMISSIONS[0]]}):
            with self.assertRaisesRegex(CloudProjectError, "lacks required"):
                self.client.create(self.project_id)
        for call in self.run.call_args_list:
            self.assertNotIn("update", call.args[0])
            self.assertNotIn("enable", call.args[0])
            if call.args[0][1] == "projects":
                self.assertEqual(call.args[0][3], self.project_id)

    def ambiguous_read_failure(self):
        return subprocess.CompletedProcess([], 1, "", "ERROR: (gcloud.projects.describe) "
            "[PRIVATE-ACCOUNT] does not have permission to access projects instance [PRIVATE-PROJECT] "
            "(or it may not exist): The caller does not have permission. PRIVATE-CONTEXT")

    def test_actual_gcloud_ambiguous_read_is_never_classified_as_absent(self):
        self.bind()
        self.run.side_effect = [self.ambiguous_read_failure()]
        with self.assertRaises(CloudProjectError) as error:
            self.client.project(self.project_id)
        self.assertEqual(error.exception.code, "ACCESS_DENIED_OR_NOT_FOUND")
        self.assertIn("absent or inaccessible", str(error.exception))
        self.assert_safe(error)
        self.assertEqual(self.run.call_count, 1)
        self.assertNotIn("create", self.run.call_args.args[0])

    def test_create_failures_classify_causes_without_echoing_google_details(self):
        self.bind()
        cases = [
            ("Caller must accept Terms of Service", "TERMS_REQUIRED", "Complete that step yourself"),
            ("Request violates organization policy constraints/resourcemanager.projects", "ORG_POLICY", "organization administrator"),
            ("Your project quota has been exhausted", "RESOURCE_EXHAUSTED", "quota or project limit"),
            ("invalid_grant", "UNAUTHENTICATED", "needs sign-in again"),
            ("The caller does not have permission", "PERMISSION_DENIED", "not allowed"),
            ("The project ID you specified is already in use", "ALREADY_EXISTS", "already in use"),
            ("Unrecognized Google service failure", "FAILED", "could not be confirmed"),
        ]
        for message, code, guidance in cases:
            self.run.reset_mock()
            self.run.side_effect = [self.fail(message), self.ambiguous_read_failure()]
            with self.subTest(code=code):
                with self.assertRaises(CloudProjectError) as error:
                    self.client.create(self.project_id)
                self.assertEqual(error.exception.code, code)
                self.assertIn(guidance, str(error.exception))
                self.assertIn("same saved project ID", str(error.exception))
                self.assert_safe(error)
                self.assertEqual(self.run.call_count, 2)
                self.assertEqual(self.run.call_args_list[0].args[0][1:4], ["projects", "create", self.project_id])
                self.assertEqual(self.run.call_args_list[1].args[0][1:4], ["projects", "describe", self.project_id])

    def test_timed_out_create_retry_uses_the_same_id_and_verifies_existing_project(self):
        self.bind()
        self.run.side_effect = [subprocess.TimeoutExpired("PRIVATE-COMMAND", 120), self.ambiguous_read_failure(),
                               self.fail("ALREADY_EXISTS"), self.ok(self.project), self.raw()]
        with self.assertRaises(CloudProjectError) as error:
            self.client.create(self.project_id)
        self.assertEqual(error.exception.code, "TIMEOUT")
        self.assert_safe(error)
        with self.permission_response():
            self.assertEqual(self.client.create(self.project_id)["project_id"], self.project_id)
        creates = [call.args[0] for call in self.run.call_args_list if "create" in call.args[0]]
        self.assertEqual(len(creates), 2)
        self.assertEqual({command[3] for command in creates}, {self.project_id})
        self.assertFalse(any("list" in call.args[0] for call in self.run.call_args_list))

    def test_success_without_readable_project_remains_pending_and_cannot_enable_drive(self):
        self.bind()
        self.run.side_effect = [self.raw(""), self.ambiguous_read_failure()]
        with self.assertRaises(CloudProjectError) as error:
            self.client.create(self.project_id)
        self.assertEqual(error.exception.code, "PENDING")
        self.assertIn("same saved project ID", str(error.exception))
        self.assert_safe(error)
        self.assertEqual(self.run.call_count, 2)
        self.session.assert_not_called()
        self.assertFalse(any("enable" in call.args[0] for call in self.run.call_args_list))

    def test_permission_request_uses_only_selected_token_fixed_endpoint_and_no_redirects(self):
        self.bind()
        self.run.side_effect = [self.raw()]
        with self.permission_response({"permissions": list(PERMISSIONS[:-1])}) as (session, _):
            result = self.client.check_permissions(self.project_id)
        self.assertEqual(result["missing"], [PERMISSIONS[-1]])
        self.assertEqual(result["required"], list(PERMISSIONS))
        self.assertFalse(session.trust_env)
        self.assertEqual(session.post.call_args.args,
                         ("https://cloudresourcemanager.googleapis.com/v1/projects/" + self.project_id + ":testIamPermissions",))
        arguments = session.post.call_args.kwargs
        self.assertEqual(arguments["headers"], {"Authorization": "Bearer PRIVATE-MANAGEMENT-TOKEN"})
        self.assertFalse(arguments["allow_redirects"])
        self.assertEqual(arguments["timeout"], 30)
        self.assertTrue(arguments["stream"])
        self.assertNotIn("PRIVATE-", json.dumps(result))

    def test_permission_response_is_bounded_fail_closed_and_redacted(self):
        self.bind()
        cases = [(302, b"{}"), (403, b"PRIVATE-ERROR"), (200, b"X" * 16385),
                 (200, b"PRIVATE-INVALID-JSON"), (200, b'{"permissions":["unexpected.permission"]}')]
        for status, body in cases:
            self.run.side_effect = [self.raw()]
            with self.subTest(status=status, size=len(body)), self.permission_response(status=status, body=body):
                with self.assertRaises(CloudProjectError) as error:
                    self.client.check_permissions(self.project_id)
            self.assert_safe(error)

    def test_billing_returns_linked_and_enabled_separately_without_mutation(self):
        self.bind()
        for account, enabled in (("", False), ("billingAccounts/123-ABC", False), ("billingAccounts/123-ABC", True)):
            self.run.side_effect = [self.ok({"projectId": self.project_id, "billingAccountName": account, "billingEnabled": enabled})]
            with self.subTest(linked=bool(account), enabled=enabled):
                self.assertEqual(self.client.billing(self.project_id), {"linked": bool(account), "enabled": enabled})
            self.assertEqual(self.run.call_args.args[0][1:5], ["billing", "projects", "describe", self.project_id])

    def test_unknown_billing_never_defaults_to_unlinked(self):
        self.bind()
        valid = {"projectId": self.project_id, "billingAccountName": "", "billingEnabled": False}
        cases = [{}, {**valid, "projectId": "different-project"}, {**valid, "billingEnabled": None},
                 {**valid, "billingAccountName": None}, {**valid, "billingEnabled": True}]
        self.run.side_effect = [self.ok(value) for value in cases]
        for _ in cases:
            with self.assertRaisesRegex(CloudProjectError, "billing status is unknown"):
                self.client.billing(self.project_id)

    def test_already_enabled_drive_is_read_only_and_enable_is_verified_idempotently(self):
        self.bind()
        enabled = [{"config": {"name": "drive.googleapis.com"}, "state": "ENABLED"}]
        self.run.side_effect = [self.ok(enabled)]
        self.assertEqual(self.client.ensure_drive(self.project_id), {"enabled": True, "changed": False})
        self.assertFalse(any("enable" in call.args[0] for call in self.run.call_args_list))
        self.run.reset_mock()
        self.run.side_effect = [self.ok([]), self.ok({}), self.ok(enabled), self.ok(enabled)]
        self.assertEqual(self.client.ensure_drive(self.project_id), {"enabled": True, "changed": True})
        self.assertEqual(self.client.ensure_drive(self.project_id), {"enabled": True, "changed": False})
        commands = [call.args[0] for call in self.run.call_args_list]
        self.assertEqual(sum("enable" in command for command in commands), 1)
        self.assertEqual(commands[1][1:4], ["services", "enable", "drive.googleapis.com"])
        self.assertTrue(all("--project=" + self.project_id in command for command in commands))

    def test_drive_beyond_first_two_enabled_services_is_found_without_api_iteration_limit(self):
        self.bind()
        enabled = [{"config": {"name": name}, "state": "ENABLED"} for name in
                   ("cloudapis.googleapis.com", "serviceusage.googleapis.com", "drive.googleapis.com")]
        def gcloud_services(argv, **kwargs):
            self.assertEqual(argv[1:3], ["services", "list"])
            # Match the installed SDK: --limit caps API iteration before its
            # local config.name filter. Drive is deliberately not an early row.
            limit = next((int(value.split("=", 1)[1]) for value in argv if value.startswith("--limit=")), None)
            visible = enabled[:limit] if limit is not None else enabled
            filtered = [row for row in visible if row["config"]["name"] == "drive.googleapis.com"]
            return self.ok(filtered)
        self.run.side_effect = gcloud_services
        self.assertEqual(self.client.ensure_drive(self.project_id), {"enabled": True, "changed": False})
        self.run.assert_called_once()
        self.assertFalse(any(value.startswith("--limit") for value in self.run.call_args.args[0]))

    def test_enable_timeout_requires_status_confirmation_not_command_success(self):
        self.bind()
        enabled = [{"config": {"name": "drive.googleapis.com"}, "state": "ENABLED"}]
        self.run.side_effect = [self.ok([]), self.fail("UNAVAILABLE"), self.ok(enabled), self.ok(enabled)]
        self.assertEqual(self.client.ensure_drive(self.project_id), {"enabled": True, "changed": True})
        self.run.side_effect = [self.ok([]), self.ok({}), self.ok([])]
        with self.assertRaisesRegex(CloudProjectError, "pending"):
            self.client.ensure_drive(self.project_id)

    def test_subprocess_timeouts_errors_and_invalid_output_are_sanitized(self):
        self.bind()
        cases = [subprocess.TimeoutExpired("PRIVATE-COMMAND", 60, stderr="PRIVATE-ERROR"),
                 OSError("PRIVATE-OS-ERROR"), self.fail(), self.raw("PRIVATE-INVALID-JSON"), self.raw("X" * (1024 * 1024 + 1))]
        for result in cases:
            self.run.side_effect = [result]
            with self.subTest(result_type=type(result).__name__):
                with self.assertRaises(CloudProjectError) as error:
                    self.client.project(self.project_id)
            self.assert_safe(error)

    def test_linked_configuration_is_refused_before_running_gcloud(self):
        target = self.root.parent / "outside"
        target.mkdir()
        self.root.mkdir()
        link = self.root / "cloud-setup"
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            self.skipTest("symlink creation unavailable: " + type(exc).__name__)
        with self.assertRaises(LifecycleError):
            CloudProjectClient(self.root)
        self.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
