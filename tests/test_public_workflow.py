"""Public install/login/package/run acceptance with isolated identity and local fakes.

No real OAuth, Drive or Colab requests are permitted. Workloads execute as ordinary
CPU subprocesses through the compiled notebook and real recovery verifier.
"""
from __future__ import annotations

import contextlib
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

from colabcola.cli import main
from colabcola.bundle import build_bundle
from colabcola.invocation import load_job_invocation
from colabcola.lifecycle_io import LifecycleError, atomic, publish
from colabcola.package_guide import save_answers
from test_lifecycle import MemoryStore, Provider


class _Request:
    def __init__(self, value):
        self.value = value

    def execute(self, **kwargs):
        return self.value


class _DriveFiles:
    def __init__(self):
        self.folders = {}
        self.creates = []

    def _folder(self, identity):
        return {"id": identity, "name": "colabcola",
                "mimeType": "application/vnd.google-apps.folder", "ownedByMe": True,
                "trashed": False, "capabilities": {"canAddChildren": True},
                "appProperties": {"colabcola.role": "workspace", "colabcola.schema": "1"}}

    def list(self, **kwargs):
        return _Request({"files": list(self.folders.values())})

    def create(self, **kwargs):
        self.creates.append(kwargs)
        folder = self._folder("fixture-drive-parent")
        self.folders[folder["id"]] = folder
        return _Request(folder)

    def get(self, *, fileId, **kwargs):
        return _Request(self.folders[fileId])


class PublicWorkflowTests(unittest.TestCase):
    interactive_login = False

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="colabcola-public-fixture-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.config = self.root / "empty-config"
        self.home = self.root / "empty-home"
        self.home.mkdir()
        isolated = {key: value for key, value in os.environ.items()
                    if not key.startswith(("COLABCOLA_", "GOOGLE_", "GCLOUD_"))}
        isolated.update(COLABCOLA_CONFIG_HOME=str(self.config), HOME=str(self.home),
                        USERPROFILE=str(self.home), APPDATA=str(self.home / "AppData"),
                        XDG_CONFIG_HOME=str(self.home / ".config"), PYTHONDONTWRITEBYTECODE="1")
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self.stack = stack
        stack.enter_context(mock.patch("colabcola.client_downloads.download_directories", return_value=[]))
        stack.enter_context(mock.patch("sys.stdin", SimpleNamespace(isatty=lambda: self.interactive_login)))
        stack.enter_context(mock.patch("webbrowser.open", return_value=False))
        stack.enter_context(mock.patch.dict(os.environ, isolated, clear=True))
        # Any missed auth or network seam fails instead of reaching the user's account.
        stack.enter_context(mock.patch("keyring.get_password", side_effect=AssertionError("ambient keyring read")))
        stack.enter_context(mock.patch("keyring.set_password", side_effect=AssertionError("ambient keyring write")))
        stack.enter_context(mock.patch("requests.sessions.Session.request", side_effect=AssertionError("unexpected HTTP")))
        stack.enter_context(mock.patch("urllib.request.urlopen", side_effect=AssertionError("unexpected HTTP")))
        self.files = _DriveFiles()
        self.login_provider = SimpleNamespace(assignments=mock.Mock(return_value=[]), close=mock.Mock())
        self.credential = SimpleNamespace(valid=True, expired=False, token="FIXTURE-ACCESS-TOKEN",
             refresh_token="FIXTURE-REFRESH-TOKEN", granted_scopes=None,
             expiry=datetime.now(timezone.utc) + timedelta(hours=1),
             has_scopes=lambda scopes: True)
        self.credential.refresh = mock.Mock()
        self.credential.to_json = lambda: json.dumps({"token": self.credential.token,
                    "refresh_token": self.credential.refresh_token, "scopes": ["fixture"]})
        self.drive_credential = SimpleNamespace(valid=True, expired=False, token="FIXTURE-DRIVE-ACCESS-TOKEN",
             refresh_token="FIXTURE-DRIVE-REFRESH-TOKEN", granted_scopes=None,
             expiry=datetime.now(timezone.utc) + timedelta(hours=1),
             has_scopes=lambda scopes: True)
        self.drive_credential.refresh = mock.Mock()
        self.drive_credential.to_json = lambda: json.dumps({"token": self.drive_credential.token,
                    "refresh_token": self.drive_credential.refresh_token, "scopes": ["fixture-drive"]})
        self.logged_in = False
        self.drive_logged_in = False
        self.oauth_calls = []
        self.drive_oauth_calls = []

    def invoke(self, *arguments):
        self.out, self.err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(self.out), contextlib.redirect_stderr(self.err):
            result = main(list(arguments))
        self.assertNotIn("FIXTURE-ACCESS-TOKEN", self.out.getvalue() + self.err.getvalue())
        self.assertNotIn("FIXTURE-REFRESH-TOKEN", self.out.getvalue() + self.err.getvalue())
        self.assertNotIn("FIXTURE-DRIVE-ACCESS-TOKEN", self.out.getvalue() + self.err.getvalue())
        self.assertNotIn("FIXTURE-DRIVE-REFRESH-TOKEN", self.out.getvalue() + self.err.getvalue())
        return result

    def login(self):
        def credentials():
            if not self.logged_in:
                raise LifecycleError("Colab login required")
            return self.credential

        def authorize(*, scopes=None):
            self.oauth_calls.append(scopes)
            self.config.mkdir(parents=True, exist_ok=True)
            atomic(self.config / "colab-oauth.json", {"fixture": "ONLY-FIXTURE-CREDENTIALS"})
            self.logged_in = True

        def drive_credentials(minimum_seconds=600, force_refresh=False):
            if not self.drive_logged_in:
                from colabcola.drive_oauth import DriveOAuthError
                raise DriveOAuthError("Drive login required")
            return self.drive_credential

        def authorize_drive(login_hint=None, timeout_seconds=300, expected_subject=None):
            self.assertEqual(login_hint, "fixture@example.invalid")
            self.assertEqual(expected_subject, "fixture-google-subject")
            self.drive_oauth_calls.append(login_hint)
            self.drive_logged_in = True
            return self.drive_credential

        def identity(credential):
            self.assertTrue(credential is self.credential or credential is self.drive_credential)
            return {"sub": "fixture-google-subject", "email": "fixture@example.invalid"}

        def drive_files(credential):
            self.assertIs(credential, self.drive_credential, "Drive must use the dedicated Drive authorization")
            return self.files

        for name, value in {
            "colabcola.onboarding._check_host": mock.Mock(),
            "colabcola.lifecycle_provider.dependency_report": mock.Mock(return_value={"provider_source_verified": True}),
            "colabcola.lifecycle_provider.credentials": mock.Mock(side_effect=credentials),
            "colabcola.lifecycle_provider.login": mock.Mock(side_effect=authorize),
            "colabcola.lifecycle_provider.GoogleProvider": mock.Mock(return_value=self.login_provider),
            "colabcola.drive_oauth.preflight_client": mock.Mock(),
            "colabcola.drive_oauth.credentials": mock.Mock(side_effect=drive_credentials),
            "colabcola.drive_oauth.login": mock.Mock(side_effect=authorize_drive),
            "colabcola.drive_oauth.identity": mock.Mock(side_effect=identity),
            "colabcola.onboarding._drive_files": mock.Mock(side_effect=drive_files),
        }.items():
            self.stack.enter_context(mock.patch(name, value))
        # Exercise the manual-client project guide with a synthetic
        # downloaded Desktop client and manual Console confirmations. No gcloud
        # executable, Cloud account, network or user configuration is consulted.
        downloaded = self.root / "downloaded-desktop.json"
        atomic(downloaded, {"installed": {
            "client_id": "fixture-personal.apps.googleusercontent.com",
            "client_secret": "FIXTURE-DESKTOP-SECRET", "project_id": "fixture-project",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token"}})
        administrator = mock.Mock()
        administrator.available.return_value = False
        self.stack.enter_context(mock.patch("colabcola.cloud_project.CloudProjectClient", return_value=administrator))
        with mock.patch("builtins.input", side_effect=["manual", "fixture-project", "yes", str(downloaded)]):
            self.assertEqual(self.invoke("login", "--no-auto-client"), 0, self.err.getvalue())
        self.assertIn("self-reported", self.err.getvalue())
        self.assertIn("https://console.cloud.google.com/auth/clients?project=fixture-project", self.err.getvalue())
        administrator.authenticate.assert_not_called()
        administrator.create.assert_not_called()
        setup = json.loads((self.config / "project-setup.json").read_text())
        self.assertEqual(setup["project_id"], "fixture-project")
        self.assertEqual(setup["cloud_checks"], "self_reported")
        self.assertEqual(setup["stage"], "ready")
        self.assertNotIn("FIXTURE-DESKTOP-SECRET", json.dumps(setup) + self.out.getvalue() + self.err.getvalue())
        self.assertTrue((self.config / "profile.json").is_file())
        self.assertEqual(len(self.oauth_calls), 1)
        self.assertEqual(len(self.drive_oauth_calls), 1)
        from colabcola.lifecycle_provider import SCOPES
        from colabcola.google_drive import GOOGLE_DRIVE_SCOPE
        self.assertEqual(self.oauth_calls, [SCOPES])
        self.assertNotIn(GOOGLE_DRIVE_SCOPE, self.oauth_calls[0])
        self.assertEqual(len(self.files.creates), 1)
        with mock.patch("builtins.input", side_effect=AssertionError("completed login must not ask setup again")), \
                mock.patch("colabcola.cloud_project.CloudProjectClient", side_effect=AssertionError("no Cloud admin after completed setup")):
            self.assertEqual(self.invoke("login"), 0, self.err.getvalue())
        self.assertEqual(len(self.oauth_calls), 1, "repeat login must reuse valid credentials")
        self.assertEqual(len(self.drive_oauth_calls), 1, "repeat login must reuse valid Drive credentials")
        self.assertEqual(len(self.files.creates), 1, "repeat login must reuse its folder")

    def package(self, commands, *, name="experiment", guided=False):
        job = self.root / (name + ".colajob")
        if guided:
            recipe = self.root / (name + ".colaml")
            arguments = ["package", "--output", str(job), "--project", str(self.project),
                         "--interactive", "--save-colaml", str(recipe)]
            for command in commands:
                arguments += ["--command", " ".join(command)]
            (self.project / "requirements.txt").write_text("# fixture has no dependencies\n")
            with mock.patch("builtins.input", side_effect=["none", "NONE", "outputs/**", "none", "1", "y"]):
                self.assertEqual(self.invoke(*arguments), 0, self.err.getvalue())
        else:
            arguments = ["package", "--output", str(job), "--project", str(self.project),
                         "--gpu", "NONE", "--timeout", "60", "--max-seconds", "900",
                         "--artifact", "outputs/**"]
            if len(commands) == 1:
                arguments += ["--", *commands[0]]
            else:
                sequence = self.root / (name + "-commands.json")
                atomic(sequence, commands)
                arguments += ["--commands", str(sequence)]
            self.assertEqual(self.invoke(*arguments), 0, self.err.getvalue())
        return job

    def remote(self):
        self.provider = Provider()
        self.provider.assignments = mock.Mock(return_value=[])
        self.provider.close = mock.Mock()
        self.store = MemoryStore()
        import time
        for name, value in {
            "colabcola.preflight.installation_integrity": {"files_verified": 1},
            "colabcola.lifecycle_provider.dependency_report": {"provider_source_verified": True},
            "colabcola.lifecycle_provider.provider_for_plan": self.provider,
            "colabcola.lifecycle.get_token": {"access_token": "FIXTURE-ACCESS-TOKEN", "expires_at": time.time() + 3600},
            "colabcola.preflight.drive_parent": {"owned": True},
            "colabcola.preflight.shutil.disk_usage": SimpleNamespace(free=100 * 1024**3),
            "colabcola.lifecycle.DriveStore": self.store,
        }.items():
            self.stack.enter_context(mock.patch(name, return_value=value))
        self.stack.enter_context(mock.patch("colabcola.lifecycle.time.sleep"))
        def boot(endpoint, plan, attempt, token):
            self.provider.boots.append(endpoint)
            runtime = self.root / ("local-cpu-" + plan["job_id"])
            namespace = {"COLABCOLA_CONTENT_ROOT_OVERRIDE": str(runtime),
                         "COLABCOLA_REQUESTED_GPU_TYPE": plan["gpu"]}
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                for cell in json.loads((Path(plan["directory"]) / "notebook.ipynb").read_text())["cells"]:
                    if cell["cell_type"] == "code":
                        exec(cell["source"], namespace)
            results = runtime / "colabcola/results" / plan["job_id"]
            from colabcola.lifecycle_worker import log_members
            members = {name: (results / name).read_bytes()
                       for name in ("result.json", "artifacts.zip", "hardware.json")}
            members.update(log_members(results, runtime / "colabcola/jobs" / plan["job_id"], []))
            publish(self.store, plan, "final", members, attempt=attempt["number"])
        self.provider.boot = boot

    def assert_result(self, job, expected):
        self.assertEqual(self.invoke(str(job)), 0, self.err.getvalue())
        state = json.loads(job.with_suffix(".run").joinpath("state.json").read_text())
        self.assertEqual(state["phase"], "completed")
        self.assertEqual(state["preflight"]["state"], "passed")
        self.assertEqual(len(self.provider.allocations), 1)
        self.assertEqual(self.provider.releases, ["vm-1"])
        self.assertFalse(self.provider.active)
        with zipfile.ZipFile(Path(state["final"]["local_directory"]) / "artifacts.zip") as archive:
            for name, body in expected.items():
                self.assertEqual(archive.read(name), body)
        return state

    def test_package_before_login_is_offline_and_run_has_actionable_login_requirement(self):
        (self.project / "train.py").write_text("print('fixture')")
        with mock.patch("colabcola.onboarding.execution_defaults", side_effect=AssertionError("package must not read login")):
            job = self.package([["python", "train.py"]])
        config = json.loads(job.with_suffix(".run.json").read_text())
        self.assertEqual(config["profile"], "default")
        self.assertNotIn("drive_parent", config)
        self.assertNotIn("token_command", config)
        self.assertFalse(self.config.exists())
        self.assertEqual(self.invoke(str(job)), 2)
        self.assertIn("colabcola login", self.err.getvalue())
        self.assertFalse(job.with_suffix(".run").exists())

    def test_fresh_login_guided_script_package_and_bare_run_recovers_real_output(self):
        self.login()
        (self.project / "train.py").write_text('from pathlib import Path\nPath("outputs").mkdir()\nPath("outputs/result.txt").write_text("script complete")\nprint("live fixture progress",flush=True)\n')
        job = self.package([["python", "train.py"]], guided=True)
        portable = job.read_text() + job.with_suffix(".run.json").read_text() + job.with_suffix(".colaml").read_text()
        self.assertNotIn(str(self.config), portable)
        self.assertNotIn("drive-helper", portable)
        self.assertNotIn("token_command", portable)
        self.assertNotIn("colab-oauth", portable)
        self.assertNotIn("drive-oauth", portable)
        self.assertNotIn("drive-client", portable)
        self.remote()
        self.assert_result(job, {"outputs/result.txt": b"script complete"})
        self.assertIn("live fixture progress", self.err.getvalue())

    def test_module_job_uses_login_defaults_and_reconnect_does_not_reallocate(self):
        self.login()
        pkg = self.project / "model"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "values.py").write_text("VALUE=42")
        (pkg / "train.py").write_text('from .values import VALUE\nfrom pathlib import Path\nPath("outputs").mkdir()\nPath("outputs/model.txt").write_text(str(VALUE))\n')
        job = self.package([["python", "-m", "model.train"]])
        self.remote()
        self.assert_result(job, {"outputs/model.txt": b"42"})
        plan = job.with_suffix(".run").joinpath("plan.json").read_bytes()
        (pkg / "train.py").write_text("raise RuntimeError('changed source must not run')")
        self.assertEqual(self.invoke(str(job)), 0, self.err.getvalue())
        self.assertEqual(len(self.provider.allocations), 1)
        self.assertEqual(job.with_suffix(".run").joinpath("plan.json").read_bytes(), plan)

    def test_train_evaluate_sequence_runs_in_order_and_recovers_both_outputs(self):
        self.login()
        pkg = self.project / "model"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "train.py").write_text('from pathlib import Path\nPath("outputs").mkdir()\nPath("outputs/model.txt").write_text("42")\n')
        (pkg / "evaluate.py").write_text('from pathlib import Path\nassert Path("outputs/model.txt").read_text()=="42"\nPath("outputs/evaluation.txt").write_text("passed")\n')
        job = self.package([["python", "-m", "model.train"], ["python", "-m", "model.evaluate"]])
        self.remote()
        self.assert_result(job, {"outputs/model.txt": b"42", "outputs/evaluation.txt": b"passed"})

    def test_account_configuration_inside_project_is_not_in_payload(self):
        self.config = self.project / "private-account-configuration"
        os.environ["COLABCOLA_CONFIG_HOME"] = str(self.config)
        self.login()
        (self.config / "unrecognised-private-cache.txt").write_text("DO-NOT-PACKAGE-ACCOUNT-DATA")
        management = self.config / "cloud-setup" / "gcloud"
        management.mkdir(parents=True)
        (management / "credentials.db").write_bytes(b"DO-NOT-PACKAGE-CLOUD-MANAGEMENT-CREDENTIALS")
        (self.project / "train.py").write_text("print('fixture')")
        job = self.package([["python", "train.py"]])
        bundle = build_bundle(load_job_invocation(job))
        self.assertFalse(any("private-account-configuration" in item.path for item in bundle.files))
        self.assertNotIn(b"DO-NOT-PACKAGE-ACCOUNT-DATA", bundle.archive_bytes)

    def test_changed_profile_cannot_redirect_a_prepared_job(self):
        self.login()
        (self.project / "train.py").write_text("print('fixture')")
        job = self.package([["python", "train.py"]])
        self.assertEqual(self.invoke("--prepare-only", str(job)), 0, self.err.getvalue())
        plan = job.with_suffix(".run").joinpath("plan.json").read_bytes()
        profile = json.loads((self.config / "profile.json").read_text())
        profile["drive_parent"] = "different-parent"
        atomic(self.config / "profile.json", profile)
        self.remote()
        self.assertEqual(self.invoke(str(job)), 2)
        self.assertEqual(self.provider.allocations, {})
        self.assertEqual(job.with_suffix(".run").joinpath("plan.json").read_bytes(), plan)
        self.assertEqual(json.loads(job.with_suffix(".run").joinpath("state.json").read_text())["phase"], "prepared")

    def test_generated_private_helper_runs_in_separate_process_with_fixture_credentials(self):
        self.login()
        from colabcola.drive_oauth import DRIVE_SCOPES, auth_path
        scopes = list(DRIVE_SCOPES)
        token_file = auth_path()
        client = {"client_id": "fixture-drive.apps.googleusercontent.com",
                  "client_secret": "FIXTURE-DRIVE-CLIENT-SECRET", "project_id": "fixture-project",
                  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                  "token_uri": "https://oauth2.googleapis.com/token", "redirect_uris": ["http://localhost"]}
        atomic(self.config / "drive-client.json", {"installed": client})
        atomic(token_file, {"token": "FIXTURE-DRIVE-ACCESS-TOKEN", "refresh_token": "FIXTURE-DRIVE-REFRESH-TOKEN",
             "token_uri": client["token_uri"], "client_id": client["client_id"],
             "client_secret": client["client_secret"], "scopes": scopes,
             "expiry": (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.000000Z")})
        token_file.chmod(0o600)
        # A malformed Colab cache makes accidental reuse fail; Drive renewal must
        # use its own authorization and must leave the Colab cache untouched.
        colab_file = self.config / "colab-oauth.json"
        colab_file.write_text("intentionally unusable Colab credential fixture")
        colab_before = colab_file.read_bytes()
        guard = self.root / "child-network-guard"
        guard.mkdir()
        (guard / "sitecustomize.py").write_text(
            "import socket\ndef blocked(*a,**k): raise RuntimeError('network forbidden in fixture')\n"
            "socket.create_connection=blocked\nsocket.socket.connect=blocked\nsocket.socket.connect_ex=blocked\n")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(guard) + os.pathsep + env.get("PYTHONPATH", "")
        argv = json.loads((self.config / "drive-helper.json").read_text())
        result = subprocess.run(argv, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, "private helper process failed")
        value = json.loads(result.stdout)
        self.assertEqual(value["access_token"], "FIXTURE-DRIVE-ACCESS-TOKEN")
        self.assertGreater(value["expires_at"], datetime.now(timezone.utc).timestamp() + 1860)
        self.assertEqual(result.stderr, "")
        self.assertEqual(set(json.loads(token_file.read_text())["scopes"]), set(scopes))
        self.assertEqual(colab_file.read_bytes(), colab_before)

    def test_legacy_explicit_helper_prepares_without_new_login_or_profile_changes(self):
        (self.project / "train.py").write_text("print('fixture')")
        helper = self.root / "legacy-helper.json"
        atomic(helper, ["fixture-helper-must-not-execute"])
        job = self.root / "legacy.colajob"
        self.assertEqual(self.invoke("package", "-o", str(job), "--project", str(self.project),
                  "--drive-parent", "legacy-parent", "--token-command", str(helper),
                  "--", "python", "train.py"), 0, self.err.getvalue())
        frozen = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (helper, job, job.with_suffix(".run.json"))}
        with mock.patch("colabcola.onboarding.execution_defaults", side_effect=AssertionError("legacy must not read profile")):
            self.assertEqual(self.invoke("--prepare-only", str(job)), 0, self.err.getvalue())
        self.assertEqual(frozen, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen})
        self.assertFalse(self.config.exists())

    def test_recipe_run_uses_profile_without_execution_credentials_in_recipe(self):
        self.login()
        (self.project / "train.py").write_text('from pathlib import Path\nPath("outputs").mkdir()\nPath("outputs/result.txt").write_text("recipe complete")\n')
        recipe = self.root / "recipe.colaml"
        save_answers(recipe, {"project": str(self.project), "commands": [["python", "train.py"]],
             "inputs": [], "environment": {"dependencies": []}, "compute": {"gpu": "NONE"},
             "outputs": {"include": ["outputs/**"], "checkpoints": []},
             "limits": {"timeout_seconds": 60, "max_seconds": 900, "max_runtimes": 1}})
        self.remote()
        workspace = self.root / "recipe-run"
        self.assertEqual(self.invoke("run", "--colaml", str(recipe), "--run-dir", str(workspace), "--non-interactive"), 0, self.err.getvalue())
        state = json.loads((workspace / "recipe.run/state.json").read_text())
        self.assertEqual(state["phase"], "completed")
        self.assertEqual(self.provider.releases, ["vm-1"])
        with zipfile.ZipFile(Path(state["final"]["local_directory"]) / "artifacts.zip") as archive:
            self.assertEqual(archive.read("outputs/result.txt"), b"recipe complete")
        self.assertEqual(json.loads((workspace / "cli-exit.json").read_text())["exit_code"], 0)


if __name__ == "__main__":
    unittest.main()
