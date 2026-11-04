"""Mocked consent flows with real Google credential serialization; no network or user config."""
import contextlib
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from google.oauth2.credentials import Credentials

from colabcola import lifecycle_provider as provider
from colabcola.google_drive import GOOGLE_DRIVE_SCOPE


class ProviderLoginTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "isolated" / "colab-oauth.json"
        self.scopes = [*provider.SCOPES, GOOGLE_DRIVE_SCOPE]
        self.stdout = io.StringIO()

    def credential(self, granted, requested=None):
        return Credentials(token="PRIVATE-ACCESS-TOKEN", refresh_token="PRIVATE-REFRESH-TOKEN",
                           token_uri="https://example.invalid/token", client_id="PRIVATE-CLIENT-ID",
                           client_secret="PRIVATE-CLIENT-SECRET", scopes=self.scopes if requested is None else requested,
                           granted_scopes=granted, expiry=(datetime.now(timezone.utc) + timedelta(hours=1)).replace(tzinfo=None))

    @contextlib.contextmanager
    def flow(self, credential):
        flow = SimpleNamespace(credentials=credential, redirect_uri=None, fetch_token=mock.Mock(),
                               authorization_url=mock.Mock(return_value=("https://example.invalid/consent", "test-state")))
        with mock.patch.object(provider, "dependency_report"), \
                mock.patch.object(provider, "auth_path", return_value=self.path), \
                mock.patch("google_auth_oauthlib.flow.InstalledAppFlow.from_client_config", return_value=flow), \
                mock.patch("builtins.input", return_value="PRIVATE-CONSENT-CODE"), \
                contextlib.redirect_stdout(self.stdout):
            yield flow
        self.assertNotIn("PRIVATE-", self.stdout.getvalue())

    def test_partial_consent_is_rejected_before_lossy_serialization_or_initial_write(self):
        credential = self.credential(provider.SCOPES)
        reloaded = Credentials.from_authorized_user_info(json.loads(credential.to_json()))
        self.assertIsNone(reloaded.granted_scopes)
        self.assertTrue(reloaded.has_scopes(self.scopes))
        with self.flow(credential), mock.patch.object(credential, "to_json", wraps=credential.to_json) as serialize, \
                mock.patch.object(provider, "atomic") as write:
            with self.assertRaises(provider.AuthorizationScopeError) as error:
                provider.login(scopes=self.scopes)
            serialize.assert_not_called()
            write.assert_not_called()
        self.assertIn("did not grant all requested permissions", str(error.exception))
        self.assertNotIn("PRIVATE-", str(error.exception))
        self.assertFalse(self.path.parent.exists())

    def test_partial_consent_preserves_existing_credentials_byte_for_byte(self):
        self.path.parent.mkdir()
        previous = b"PRIVATE-EXISTING-CREDENTIAL-FILE"
        self.path.write_bytes(previous)
        with self.flow(self.credential(" ".join(provider.SCOPES))), mock.patch.object(provider, "atomic") as write:
            with self.assertRaises(provider.AuthorizationScopeError):
                provider.login(scopes=self.scopes)
            write.assert_not_called()
        self.assertEqual(self.path.read_bytes(), previous)

    def test_missing_requested_scope_is_rejected_even_if_granted_scope_lists_it(self):
        with self.flow(self.credential(self.scopes, requested=provider.SCOPES)):
            with self.assertRaises(provider.AuthorizationScopeError):
                provider.login(scopes=self.scopes)
        self.assertFalse(self.path.exists())

    def test_full_consent_and_omitted_unchanged_scope_response_can_be_saved(self):
        # OAuth may omit the granted scope field when it is identical to the request.
        for granted in (self.scopes, " ".join(self.scopes), None):
            with self.subTest(granted_type=type(granted).__name__), self.flow(self.credential(granted)):
                provider.login(scopes=self.scopes)
                saved = json.loads(self.path.read_text())
                self.assertEqual(saved["scopes"], self.scopes)
                self.assertEqual(saved["token"], "PRIVATE-ACCESS-TOKEN")


if __name__ == "__main__":
    unittest.main()
