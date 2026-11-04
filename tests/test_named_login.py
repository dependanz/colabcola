"""Named-profile login integration using only synthetic accounts and local files."""
import contextlib
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from colabcola import onboarding, drive_oauth, lifecycle_provider, profiles
from colabcola.cloud_project import CloudProjectClient
from colabcola.google_drive import default_google_drive_settings_path
from colabcola.lifecycle_io import LifecycleError, atomic


class NamedLoginTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="colabcola-named-login-")
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name)
        self.legacy = self.home / ".config" / "colabcola"
        self.legacy.mkdir(parents=True)
        (self.legacy / "preserve.txt").write_bytes(b"existing-login-must-remain-byte-identical")
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        environment = {k: v for k, v in os.environ.items() if not k.startswith(("COLABCOLA_", "GOOGLE_", "GCLOUD_"))}
        environment.update(HOME=str(self.home), USERPROFILE=str(self.home), COLABCOLA_CONFIG_HOME=str(self.legacy))
        self.stack.enter_context(mock.patch.dict(os.environ, environment, clear=True))
        self.stack.enter_context(mock.patch("pathlib.Path.home", return_value=self.home))
        self.stack.enter_context(mock.patch("socket.socket.connect", side_effect=AssertionError("no network in profile tests")))
        self.stack.enter_context(mock.patch("keyring.get_password", side_effect=AssertionError("no ambient keyring")))
        self.stack.enter_context(mock.patch("colabcola.client_downloads.download_directories", return_value=[]))
        self.authorizations = []
        self.drive_authorizations = []

    def invoke(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            result = onboarding.main(list(args))
        self.assertNotIn("FIXTURE-TOKEN", out.getvalue() + err.getvalue())
        self.assertNotIn("FIXTURE-SECRET", out.getvalue() + err.getvalue())
        return result, out.getvalue(), err.getvalue()

    def snapshot(self, root):
        return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}

    def fake_google(self):
        def account():
            return profiles.profile_name(onboarding.config_home()) or "legacy"

        def credential(name, service):
            value = SimpleNamespace(valid=True, token="FIXTURE-TOKEN-" + service + "-" + name,
                refresh_token="FIXTURE-TOKEN-REFRESH", expiry=datetime.now(timezone.utc) + timedelta(hours=2),
                granted_scopes=None, has_scopes=lambda scopes: True, account=name)
            value.to_json = lambda: json.dumps({"token": value.token, "account": name, "fixture": True})
            return value

        def setup(root, profile, project=None, client_json=None, **kwargs):
            atomic(root / "drive-client.json", {"installed": {
                "client_id": account() + ".apps.googleusercontent.com", "client_secret": "FIXTURE-SECRET",
                "project_id": "fixture-" + account(), "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"}})
            return {"mode": "user_project", "cloud_account_email": account() + "@example.invalid"}

        def colab_credentials():
            path = lifecycle_provider.auth_path()
            if not path.exists():
                raise LifecycleError("fixture login needed")
            name = json.loads(path.read_text())["account"]
            return credential(name, "colab")

        def colab_login(scopes=None):
            path = lifecycle_provider.auth_path()
            self.authorizations.append(path)
            atomic(path, json.loads(credential(account(), "colab").to_json()))

        def drive_credentials(**kwargs):
            path = drive_oauth.auth_path()
            if not path.exists():
                raise drive_oauth.DriveOAuthError("fixture Drive login needed")
            return credential(json.loads(path.read_text())["account"], "drive")

        def drive_login(login_hint, expected_subject):
            self.assertEqual(login_hint, account() + "@example.invalid")
            self.assertEqual(expected_subject, account())
            result = credential(account(), "drive")
            self.drive_authorizations.append(drive_oauth.auth_path())
            drive_oauth._save(result)
            return result

        def folder(identity):
            return {"id": identity, "mimeType": "application/vnd.google-apps.folder", "ownedByMe": True,
                    "trashed": False, "capabilities": {"canAddChildren": True},
                    "appProperties": {"colabcola.role": "workspace", "colabcola.schema": "1"}}

        files = mock.Mock()
        files.list.return_value.execute.return_value = {"files": []}
        files.create.side_effect = lambda **kwargs: SimpleNamespace(execute=lambda **options: folder("drive-" + account()))
        files.get.side_effect = lambda fileId, **kwargs: SimpleNamespace(execute=lambda **options: folder(fileId))
        provider = SimpleNamespace(assignments=mock.Mock(return_value=[]), close=mock.Mock(),
                                   allocate=mock.Mock(side_effect=AssertionError("no allocation")))
        for target, replacement in {
            "colabcola.onboarding._check_host": mock.Mock(),
            "colabcola.onboarding._login_client_configuration": mock.Mock(side_effect=setup),
            "colabcola.lifecycle_provider.dependency_report": mock.Mock(return_value={}),
            "colabcola.lifecycle_provider.credentials": mock.Mock(side_effect=colab_credentials),
            "colabcola.lifecycle_provider.login": mock.Mock(side_effect=colab_login),
            "colabcola.lifecycle_provider.GoogleProvider": mock.Mock(return_value=provider),
            "colabcola.drive_oauth.credentials": mock.Mock(side_effect=drive_credentials),
            "colabcola.drive_oauth.login": mock.Mock(side_effect=drive_login),
            "colabcola.drive_oauth.identity": mock.Mock(side_effect=lambda cred: {"email": cred.account + "@example.invalid", "sub": cred.account}),
            "colabcola.onboarding._drive_files": mock.Mock(return_value=files),
        }.items():
            self.stack.enter_context(mock.patch(target, replacement))
        return provider

    def test_status_selects_named_root_without_reading_ambient_or_creating_files(self):
        (self.legacy / "profile.json").write_text("must-not-be-read")
        before = self.snapshot(self.home)
        code, out, err = self.invoke("--profile", "danzel", "--status")
        self.assertEqual(code, 0, err)
        value = json.loads(out)
        self.assertEqual(value["profile"], "danzel")
        self.assertEqual(value["state"], "not_configured")
        self.assertEqual(Path(value["config_home"]), self.home / ".colabcola/configs/danzel")
        self.assertFalse(value["network_accessed"])
        self.assertEqual(self.snapshot(self.home), before)
        self.assertEqual(os.environ["COLABCOLA_CONFIG_HOME"], str(self.legacy))

    def test_two_logins_are_isolated_reused_and_legacy_is_unchanged(self):
        before = self.snapshot(self.legacy)
        provider = self.fake_google()
        for name in ("danzel", "dezioxe", "danzel"):
            code, out, err = self.invoke("--profile", name)
            self.assertEqual(code, 0, err)
            value = json.loads(out)
            self.assertEqual(value["profile"], name)
            self.assertEqual(value["account_email"], name + "@example.invalid")
            self.assertFalse(value["runtime_allocated"])
            root = profiles.named_home(name)
            self.assertEqual(json.loads((root / "colab-oauth.json").read_text())["account"], name)
            self.assertEqual(json.loads((root / "drive-oauth.json").read_text())["account"], name)
            helper = json.loads((root / "drive-helper.json").read_text())
            self.assertEqual(helper[-2:], ["--config-home", str(root)])
        self.assertEqual(len(self.authorizations), 2)
        self.assertEqual(len(self.drive_authorizations), 2)
        self.assertEqual(self.snapshot(self.legacy), before)
        self.assertEqual(os.environ["COLABCOLA_CONFIG_HOME"], str(self.legacy))
        provider.allocate.assert_not_called()

    def test_internal_helper_keeps_exact_profile_when_session_profile_changes(self):
        self.fake_google()
        self.assertEqual(self.invoke("--profile", "danzel")[0], 0)
        danzel = profiles.named_home("danzel")
        self.assertEqual(self.invoke("--profile", "dezioxe")[0], 0)
        other_before = self.snapshot(profiles.named_home("dezioxe"))
        os.environ["COLABCOLA_CONFIG_HOME"] = str(profiles.named_home("dezioxe"))
        os.environ["COLABCOLA_PROFILE"] = "dezioxe"
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = onboarding.token_main(["--config-home", str(danzel)])
        self.assertEqual(code, 0, err.getvalue())
        self.assertEqual(json.loads(out.getvalue())["access_token"], "FIXTURE-TOKEN-drive-danzel")
        self.assertEqual(self.snapshot(profiles.named_home("dezioxe")), other_before)
        self.assertEqual(os.environ["COLABCOLA_PROFILE"], "dezioxe")
        self.assertEqual(os.environ["COLABCOLA_CONFIG_HOME"], str(profiles.named_home("dezioxe")))

    def test_invalid_and_conflicting_selectors_do_not_modify_profiles(self):
        before = self.snapshot(self.home)
        for name in ("../escape", "Danzel", "con", "a/b", ""):
            with self.subTest(name=name):
                try:
                    code, out, err = self.invoke("--profile", name, "--status")
                except SystemExit as exc:
                    code = exc.code
                self.assertEqual(code, 2)
        with self.assertRaises(SystemExit) as conflict:
            self.invoke("--profile", "danzel", "--config-home", str(self.legacy), "--status")
        self.assertEqual(conflict.exception.code, 2)
        self.assertEqual(self.snapshot(self.home), before)

    def test_explicit_helper_for_another_profile_refuses_before_journal_or_prepare(self):
        from colabcola import managed
        self.fake_google()
        self.assertEqual(self.invoke("--profile", "danzel")[0], 0)
        job = self.home / "work.colajob"
        settings = job.with_suffix(".run.json")
        atomic(settings, {"drive_parent": "drive-danzel", "token_command": str(profiles.named_home("danzel") / "drive-helper.json"),
                          "gpu": "NONE", "max_attempts": 1, "max_seconds": 300})
        command = SimpleNamespace(job_file=job, execution_config=None)
        with profiles.selected(profile="dezioxe"), mock.patch("colabcola.managed.prepare") as prepare:
            with self.assertRaisesRegex(LifecycleError, "different configuration"):
                managed.run_managed(command, None)
        prepare.assert_not_called()
        self.assertFalse(job.with_suffix(".run").exists())

    def test_all_managed_services_resolve_same_named_profile(self):
        with profiles.selected(profile="danzel") as root:
            self.assertEqual(onboarding.config_home(), root)
            self.assertEqual(lifecycle_provider.auth_path(), root / "colab-oauth.json")
            self.assertEqual(drive_oauth.auth_path(), root / "drive-oauth.json")
            self.assertEqual(CloudProjectClient().config, root / "cloud-setup/gcloud")
            self.assertEqual(default_google_drive_settings_path(), root / "google-drive.json")
        self.assertFalse((self.home / ".colabcola").exists())

    def test_env_profile_is_shared_by_provider_and_legacy_drive_settings(self):
        os.environ.pop("COLABCOLA_CONFIG_HOME")
        os.environ["COLABCOLA_PROFILE"] = "danzel"
        root = profiles.named_home("danzel")
        self.assertEqual(lifecycle_provider.auth_path(), root / "colab-oauth.json")
        self.assertEqual(default_google_drive_settings_path(), root / "google-drive.json")
        self.assertEqual(json.loads(self.invoke("--status")[1])["profile"], "danzel")


if __name__ == "__main__":
    unittest.main()
