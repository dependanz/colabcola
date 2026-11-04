"""Publisher Drive OAuth boundary tests; synthetic configuration and no network."""
import contextlib
import copy
from datetime import datetime, timedelta, timezone
import io
import json
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from google.oauth2.credentials import Credentials

from colabcola import drive_oauth
from colabcola.lifecycle_io import LifecycleError


class DriveOAuthTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.home = self.root / "selected-profile"
        self.home.mkdir(mode=0o700)
        self.client = {"installed": {
            "client_id": "synthetic-desktop.apps.googleusercontent.com",
            "client_secret": "PRIVATE-DESKTOP-CLIENT-SECRET",
            "project_id": "synthetic-publisher",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }}
        self.client_path = self.home / drive_oauth.CLIENT_FILE
        self.client_path.write_text(json.dumps(self.client), encoding="utf-8")
        self.token_path = self.home / drive_oauth.TOKEN_FILE
        self.colab_path = self.home / "colab-oauth.json"
        self.colab_path.write_bytes(b"PRIVATE-COLAB-AUTH-MUST-STAY-UNCHANGED")
        self.colab_before = self.colab_path.read_bytes()
        self.identity = {"sub": "synthetic-subject", "email": "person@example.test"}
        for patch in (
            mock.patch.dict(os.environ, {"COLABCOLA_CONFIG_HOME": str(self.home)}),
            mock.patch.object(Path, "home", side_effect=AssertionError("ambient home must not be consulted")),
            mock.patch("requests.sessions.Session.request", side_effect=AssertionError("network is forbidden")),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self.assert_colab_unchanged)

    def assert_colab_unchanged(self):
        self.assertEqual(self.colab_path.read_bytes(), self.colab_before)

    def credential(self, *, granted="all", requested=None, seconds=3600, refresh="PRIVATE-REFRESH-TOKEN", **overrides):
        scopes = list(drive_oauth.DRIVE_SCOPES)
        args = dict(token="PRIVATE-ACCESS-TOKEN", refresh_token=refresh,
                    token_uri=self.client["installed"]["token_uri"],
                    client_id=self.client["installed"]["client_id"],
                    client_secret=self.client["installed"]["client_secret"],
                    scopes=scopes if requested is None else requested,
                    granted_scopes=scopes if granted == "all" else granted,
                    expiry=(datetime.now(timezone.utc) + timedelta(seconds=seconds)).replace(tzinfo=None))
        args.update(overrides)
        return Credentials(**args)

    def save(self, credential):
        self.token_path.write_text(credential.to_json(), encoding="utf-8")
        os.chmod(self.token_path, 0o600)
        return self.token_path.read_bytes()

    @contextlib.contextmanager
    def flow(self, credential=None, side_effect=None):
        flow = SimpleNamespace(run_local_server=mock.Mock(return_value=credential, side_effect=side_effect))
        with mock.patch("google_auth_oauthlib.flow.InstalledAppFlow.from_client_config", return_value=flow) as factory:
            yield flow, factory

    def assert_safe(self, error):
        self.assertNotIn("PRIVATE-", str(error.exception))

    def test_desktop_client_uses_fixed_google_endpoints_and_constructed_loopback(self):
        for endpoint in ("https://accounts.google.com/o/oauth2/auth", "https://accounts.google.com/o/oauth2/v2/auth"):
            with self.subTest(endpoint=endpoint):
                value = copy.deepcopy(self.client)
                value["installed"]["auth_uri"] = endpoint
                value["installed"]["redirect_uris"] = ["https://untrusted.invalid/PRIVATE-CALLBACK"]
                validated = drive_oauth.validate_client(value)
                self.assertEqual(validated["installed"]["redirect_uris"], ["http://localhost"])
                self.assertEqual(value["installed"]["redirect_uris"], ["https://untrusted.invalid/PRIVATE-CALLBACK"])

    def test_rejects_web_client_untrusted_endpoints_and_malformed_identity(self):
        invalid = [None, [], {"web": self.client["installed"]}, {**self.client, "web": {}}, {"installed": []}]
        for key, value in (("auth_uri", "https://attacker.invalid/PRIVATE-AUTH"),
                           ("token_uri", "https://attacker.invalid/PRIVATE-TOKEN"),
                           ("client_id", "PRIVATE-UNEXPECTED-CLIENT"), ("project_id", "INVALID PROJECT"),
                           ("client_secret", ""), ("client_secret", "X" * 1025),
                           ("unexpected", "PRIVATE-EXTRA")):
            candidate = copy.deepcopy(self.client)
            candidate["installed"][key] = value
            invalid.append(candidate)
        for candidate in invalid:
            with self.subTest(candidate_type=type(candidate).__name__):
                with self.assertRaises(drive_oauth.DriveOAuthError) as error:
                    drive_oauth.validate_client(candidate)
                self.assert_safe(error)

    def test_explicit_profile_uses_only_its_client(self):
        with mock.patch.object(drive_oauth.importlib.resources, "files") as resources:
            result = drive_oauth.preflight_client()
        self.assertEqual(result["installed"]["client_id"], self.client["installed"]["client_id"])
        resources.assert_not_called()
        self.assertEqual(drive_oauth.auth_path(), self.token_path)

    def test_missing_publisher_client_is_actionable_without_account_access(self):
        self.client_path.unlink()
        resource = mock.Mock()
        resource.is_file.return_value = False
        with mock.patch.object(drive_oauth.importlib.resources, "files") as resources:
            resources.return_value.joinpath.return_value = resource
            with self.assertRaises(drive_oauth.DriveOAuthError) as error:
                drive_oauth.preflight_client()
        self.assertIn("publisher", str(error.exception))
        self.assertIn("set up your own Google Cloud project", str(error.exception))
        resource.read_bytes.assert_not_called()

    def test_personal_setup_never_falls_back_to_publisher_when_client_missing(self):
        self.client_path.unlink()
        (self.home / "project-setup.json").write_text("{}", encoding="utf-8")
        with mock.patch.object(drive_oauth.importlib.resources, "files") as resources:
            with self.assertRaisesRegex(drive_oauth.DriveOAuthError, "shared publisher credentials will not be substituted"):
                drive_oauth.preflight_client()
        resources.assert_not_called()

    def test_oversized_client_is_rejected_before_json_file_read(self):
        self.client_path.write_bytes(b"X" * 16385)
        with mock.patch.object(Path, "read_text") as read:
            with self.assertRaises(drive_oauth.DriveOAuthError):
                drive_oauth.preflight_client()
        read.assert_not_called()

    def test_malformed_client_json_is_redacted(self):
        self.client_path.write_text("PRIVATE-INVALID-JSON", encoding="utf-8")
        with self.assertRaises(drive_oauth.DriveOAuthError) as error:
            drive_oauth.preflight_client()
        self.assert_safe(error)

    def test_client_directory_is_rejected(self):
        self.client_path.unlink()
        self.client_path.mkdir()
        with self.assertRaises(drive_oauth.DriveOAuthError):
            drive_oauth.preflight_client()

    @unittest.skipIf(os.name == "nt", "POSIX permissions are enforced on Linux/WSL")
    def test_world_readable_tokens_are_refused_without_refresh(self):
        self.save(self.credential())
        os.chmod(self.token_path, 0o644)
        with mock.patch.object(Credentials, "refresh") as refresh:
            with self.assertRaisesRegex(drive_oauth.DriveOAuthError, "0600"):
                drive_oauth.credentials()
        refresh.assert_not_called()

    def _symlink(self, link, target, *, directory=False):
        try:
            link.symlink_to(target, target_is_directory=directory)
        except OSError as exc:
            self.skipTest("symlink creation unavailable: " + type(exc).__name__)

    def test_client_symlink_refused_before_target_read(self):
        target = self.root / "outside-client.json"
        target.write_text("PRIVATE-OUTSIDE-CLIENT", encoding="utf-8")
        self.client_path.unlink()
        self._symlink(self.client_path, target)
        with mock.patch.object(Path, "read_text") as read:
            with self.assertRaises(LifecycleError):
                drive_oauth.preflight_client()
        read.assert_not_called()
        self.assertEqual(target.read_text(), "PRIVATE-OUTSIDE-CLIENT")

    def test_token_symlink_refused_before_auth_or_read(self):
        target = self.root / "outside-token.json"
        target.write_text("PRIVATE-OUTSIDE-TOKEN", encoding="utf-8")
        self._symlink(self.token_path, target)
        with mock.patch("google_auth_oauthlib.flow.InstalledAppFlow.from_client_config") as consent:
            with self.assertRaises(LifecycleError):
                drive_oauth.login()
        consent.assert_not_called()
        with mock.patch.object(Credentials, "from_authorized_user_info") as deserialize:
            with self.assertRaises(LifecycleError):
                drive_oauth.credentials()
        deserialize.assert_not_called()
        self.assertEqual(target.read_text(), "PRIVATE-OUTSIDE-TOKEN")

    def test_config_parent_symlink_refused(self):
        alias = self.root / "profile-alias"
        self._symlink(alias, self.home, directory=True)
        with mock.patch.dict(os.environ, {"COLABCOLA_CONFIG_HOME": str(alias)}):
            with self.assertRaises(LifecycleError):
                drive_oauth.preflight_client()

    def test_login_roundtrip_uses_real_credentials_and_does_not_touch_colab(self):
        for granted in (drive_oauth.DRIVE_SCOPES, " ".join(drive_oauth.DRIVE_SCOPES), None):
            with self.subTest(granted_type=type(granted).__name__):
                credential = self.credential(granted=granted)
                with self.flow(credential), mock.patch.object(drive_oauth, "identity", return_value=self.identity) as identity:
                    self.assertIs(drive_oauth.login(expected_subject=self.identity["sub"]), credential)
                identity.assert_called_once_with(credential)
                with mock.patch.object(Credentials, "refresh") as refresh:
                    restored = drive_oauth.credentials()
                refresh.assert_not_called()
                self.assertTrue(restored.has_scopes(drive_oauth.DRIVE_SCOPES))
                self.assertEqual(restored.token, credential.token)
                self.assertEqual(restored.refresh_token, credential.refresh_token)
                if os.name != "nt":
                    self.assertEqual(self.token_path.stat().st_mode & 0o777, 0o600)

    def test_partial_consent_is_refused_before_lossy_serialization(self):
        previous = self.save(self.credential())
        credential = self.credential(granted=drive_oauth.DRIVE_SCOPES[:-1])
        # Real Google serialization loses granted_scopes; check before this boundary.
        restored = Credentials.from_authorized_user_info(json.loads(credential.to_json()))
        self.assertIsNone(restored.granted_scopes)
        self.assertTrue(restored.has_scopes(drive_oauth.DRIVE_SCOPES))
        with self.flow(credential), mock.patch.object(credential, "to_json", wraps=credential.to_json) as serialize:
            with self.assertRaises(drive_oauth.DriveOAuthError) as error:
                drive_oauth.login()
        self.assert_safe(error)
        serialize.assert_not_called()
        self.assertEqual(self.token_path.read_bytes(), previous)

    def test_missing_requested_scope_and_missing_refresh_are_refused(self):
        for credential in (self.credential(requested=drive_oauth.DRIVE_SCOPES[:-1]), self.credential(refresh=None)):
            with self.subTest(missing_refresh=credential.refresh_token is None), self.flow(credential):
                with self.assertRaises(drive_oauth.DriveOAuthError):
                    drive_oauth.login()
        self.assertFalse(self.token_path.exists())

    def test_cached_credentials_from_wrong_publisher_are_refused_before_refresh(self):
        for override in ({"client_id": "other.apps.googleusercontent.com"}, {"client_secret": "PRIVATE-OTHER-SECRET"}):
            previous = self.save(self.credential(seconds=1, **override))
            with self.subTest(field=next(iter(override))), mock.patch.object(Credentials, "refresh") as refresh:
                with self.assertRaises(drive_oauth.DriveOAuthError) as error:
                    drive_oauth.credentials()
            self.assert_safe(error)
            refresh.assert_not_called()
            self.assertEqual(self.token_path.read_bytes(), previous)

    def test_oversized_token_refused_before_deserialization(self):
        self.token_path.write_bytes(b"X" * 16385)
        os.chmod(self.token_path, 0o600)
        with mock.patch.object(Credentials, "from_authorized_user_info") as deserialize:
            with self.assertRaises(drive_oauth.DriveOAuthError):
                drive_oauth.credentials()
        deserialize.assert_not_called()

    def _renew(self, credential, request):
        credential.token = "PRIVATE-RENEWED-TOKEN"
        credential.expiry = (datetime.now(timezone.utc) + timedelta(hours=1)).replace(tzinfo=None)

    def test_near_expiry_and_forced_renewal_write_only_drive_credentials(self):
        for seconds, force in ((1000, False), (3600, True)):
            with self.subTest(seconds=seconds, force=force):
                previous = self.save(self.credential(seconds=seconds))
                with mock.patch.object(Credentials, "refresh", autospec=True, side_effect=self._renew) as refresh:
                    result = drive_oauth.credentials(minimum_seconds=1860, force_refresh=force)
                refresh.assert_called_once()
                self.assertEqual(result.token, "PRIVATE-RENEWED-TOKEN")
                self.assertNotEqual(self.token_path.read_bytes(), previous)
                self.assertEqual(json.loads(self.token_path.read_text())["token"], result.token)
                self.assertEqual(sorted(path.name for path in self.home.iterdir()),
                                 sorted([drive_oauth.CLIENT_FILE, drive_oauth.TOKEN_FILE, "colab-oauth.json"]))
                self.assert_colab_unchanged()

    def test_failed_refresh_does_not_replace_cached_tokens_or_disclose_errors(self):
        for failure in (RuntimeError("PRIVATE-REFRESH-FAILURE"), None):
            previous = self.save(self.credential(seconds=100))
            with self.subTest(failure_type=type(failure).__name__), \
                    mock.patch.object(Credentials, "refresh", side_effect=failure):
                with self.assertRaises(drive_oauth.DriveOAuthError) as error:
                    drive_oauth.credentials(minimum_seconds=1860)
            self.assert_safe(error)
            self.assertEqual(self.token_path.read_bytes(), previous)

    def test_refresh_scope_loss_does_not_replace_cached_tokens(self):
        previous = self.save(self.credential(seconds=100))
        def lose_scope(credential, request):
            self._renew(credential, request)
            credential._granted_scopes = drive_oauth.DRIVE_SCOPES[:-1]
        with mock.patch.object(Credentials, "refresh", autospec=True, side_effect=lose_scope):
            with self.assertRaises(drive_oauth.DriveOAuthError):
                drive_oauth.credentials()
        self.assertEqual(self.token_path.read_bytes(), previous)

    def test_loopback_uses_pkce_random_local_port_and_bounded_timeout(self):
        with self.flow(self.credential()) as (flow, factory):
            drive_oauth.login(login_hint="person@example.test")
        self.assertEqual(factory.call_args.args[1], drive_oauth.DRIVE_SCOPES)
        self.assertTrue(factory.call_args.kwargs["autogenerate_code_verifier"])
        arguments = flow.run_local_server.call_args.kwargs
        for key, value in {"host": "127.0.0.1", "bind_addr": "127.0.0.1", "port": 0,
                           "open_browser": False, "timeout_seconds": 300, "prompt": "consent",
                           "access_type": "offline", "login_hint": "person@example.test"}.items():
            self.assertEqual(arguments[key], value, key)

    def test_callback_and_oauth_debug_logs_are_suppressed_and_logging_restored(self):
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        root_logger = logging.getLogger()
        old_level, old_disabled = root_logger.level, logging.root.manager.disable
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.DEBUG)
        logging.disable(logging.NOTSET)
        self.addCleanup(root_logger.removeHandler, handler)
        self.addCleanup(root_logger.setLevel, old_level)
        self.addCleanup(logging.disable, old_disabled)
        def exchange(**kwargs):
            logging.getLogger("google_auth_oauthlib.flow").info("GET /?code=PRIVATE-CONSENT-CODE")
            logging.getLogger("oauthlib.oauth2").debug("PRIVATE-ACCESS-TOKEN")
            return self.credential()
        with self.flow(side_effect=exchange):
            drive_oauth.login()
        self.assertEqual(logging.root.manager.disable, logging.NOTSET)
        root_logger.info("logging restored")
        self.assertNotIn("PRIVATE-", output.getvalue())
        self.assertIn("logging restored", output.getvalue())

    def test_timeout_and_state_failure_preserve_credentials_with_safe_errors(self):
        previous = self.save(self.credential())
        for failure in (TimeoutError("PRIVATE-CALLBACK"), ValueError("PRIVATE-STATE-MISMATCH")):
            with self.subTest(error_type=type(failure).__name__), self.flow(side_effect=failure):
                before = logging.root.manager.disable
                with self.assertRaises(drive_oauth.DriveOAuthError) as error:
                    drive_oauth.login()
                self.assertEqual(logging.root.manager.disable, before)
            self.assert_safe(error)
            if isinstance(failure, TimeoutError):
                self.assertIn("timed out", str(error.exception))
            self.assertEqual(self.token_path.read_bytes(), previous)

    def test_different_google_account_preserves_cached_drive_credentials(self):
        previous = self.save(self.credential())
        credential = self.credential(token="PRIVATE-NEW-ACCOUNT-TOKEN")
        with self.flow(credential), mock.patch.object(drive_oauth, "identity", return_value=self.identity), \
                mock.patch.object(credential, "to_json", wraps=credential.to_json) as serialize:
            with self.assertRaises(drive_oauth.DriveOAuthError) as error:
                drive_oauth.login(expected_subject="different-colab-subject")
        self.assert_safe(error)
        serialize.assert_not_called()
        self.assertEqual(self.token_path.read_bytes(), previous)

    def test_identity_uses_fixed_endpoint_no_redirects_and_verified_sub(self):
        credential = self.credential()
        response = mock.Mock(status_code=200, content=b"{}")
        response.json.return_value = {**self.identity, "email_verified": True, "ignored": "PRIVATE-UNNEEDED"}
        with mock.patch("google.auth.transport.requests.AuthorizedSession") as factory:
            session = factory.return_value.__enter__.return_value
            session.get.return_value = response
            self.assertEqual(drive_oauth.identity(credential), self.identity)
        factory.assert_called_once_with(credential)
        session.get.assert_called_once_with("https://openidconnect.googleapis.com/v1/userinfo", timeout=30,
                                            allow_redirects=False)

    def test_identity_rejects_unverified_malformed_oversized_and_redirect_responses(self):
        valid = {**self.identity, "email_verified": True}
        cases = [(302, b"{}", valid), (403, b"PRIVATE-SERVER-ERROR", valid),
                 (200, b"X" * 16385, valid), (200, b"{}", []),
                 (200, b"{}", {**valid, "sub": ""}), (200, b"{}", {**valid, "sub": "X" * 256}),
                 (200, b"{}", {**valid, "email": "PRIVATE-INVALID-EMAIL"}),
                 (200, b"{}", {**valid, "email_verified": False}),
                 (200, b"{}", {**valid, "email_verified": "true"})]
        for status, content, payload in cases:
            response = mock.Mock(status_code=status, content=content)
            response.json.return_value = payload
            with self.subTest(status=status, size=len(content)), \
                    mock.patch("google.auth.transport.requests.AuthorizedSession") as factory:
                factory.return_value.__enter__.return_value.get.return_value = response
                with self.assertRaises(drive_oauth.DriveOAuthError) as error:
                    drive_oauth.identity(self.credential())
            self.assert_safe(error)
            if status != 200 or len(content) > 16384:
                response.json.assert_not_called()


if __name__ == "__main__":
    unittest.main()
