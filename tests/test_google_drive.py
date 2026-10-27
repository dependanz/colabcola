from __future__ import annotations

import json
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from colabcola import (
    DriveTransportError,
    GoogleDriveSettings,
    google_drive_settings_from_clients,
    load_google_drive_settings,
    load_google_oauth_client,
    write_google_drive_settings,
)
from colabcola.google_drive import GoogleDriveApiBackend


AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"


def _client_document(
    *, project_id: str, client_id: str, client_secret: str, client_type: str = "installed"
) -> dict[str, object]:
    return {
        client_type: {
            "auth_uri": AUTH_URI,
            "client_id": client_id,
            "client_secret": client_secret,
            "project_id": project_id,
            "token_uri": TOKEN_URI,
        }
    }


class _Request:
    def __init__(self, value: object) -> None:
        self.value = value

    def execute(self, *, num_retries: int) -> object:
        if num_retries != 3:
            raise AssertionError("unexpected retry policy")
        return self.value


class _Files:
    def __init__(self, values: list[dict[str, object]]) -> None:
        self.values = values
        self.list_calls: list[dict[str, object]] = []
        self.update_calls: list[dict[str, object]] = []

    def list(self, **kwargs: object) -> _Request:
        self.list_calls.append(kwargs)
        return _Request({"files": self.values})

    def update(self, **kwargs: object) -> _Request:
        self.update_calls.append(kwargs)
        return _Request({"id": kwargs["fileId"], "trashed": True})


class GoogleDriveSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.desktop_secret = "desktop-private-value"
        self.colab_secret = "colab-private-value"
        self.desktop = self.root / "desktop.json"
        self.colab = self.root / "colab.json"
        self.desktop.write_text(
            json.dumps(
                _client_document(
                    project_id="colabcola-fixture",
                    client_id="desktop-fixture.apps.googleusercontent.com",
                    client_secret=self.desktop_secret,
                )
            ),
            encoding="utf-8",
        )
        self.colab.write_text(
            json.dumps(
                _client_document(
                    project_id="colabcola-fixture",
                    client_id="device-fixture.apps.googleusercontent.com",
                    client_secret=self.colab_secret,
                    client_type="device",
                )
            ),
            encoding="utf-8",
        )

    def test_same_project_clients_write_canonical_credential_free_settings(self) -> None:
        settings = google_drive_settings_from_clients(self.desktop, self.colab)
        output = self.root / "settings.json"
        write_google_drive_settings(settings, output)

        persisted = output.read_text(encoding="utf-8")
        self.assertEqual(load_google_drive_settings(output), settings)
        self.assertNotIn(self.desktop_secret, persisted)
        self.assertNotIn(self.colab_secret, persisted)
        self.assertNotIn("refresh_token", persisted)
        self.assertEqual(persisted.count("\n"), 1)
        self.assertTrue(persisted.endswith("\n"))

    def test_clients_must_be_separate_and_belong_to_one_project(self) -> None:
        other = self.root / "other.json"
        other.write_text(
            json.dumps(
                _client_document(
                    project_id="another-project",
                    client_id="other-fixture.apps.googleusercontent.com",
                    client_secret="other-private-value",
                    client_type="device",
                )
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(DriveTransportError, "same Google Cloud project"):
            google_drive_settings_from_clients(self.desktop, other)

        same = self.root / "same.json"
        same.write_text(
            json.dumps(
                _client_document(
                    project_id="colabcola-fixture",
                    client_id="desktop-fixture.apps.googleusercontent.com",
                    client_secret="same-private-value",
                    client_type="device",
                )
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(DriveTransportError, "separate platform clients"):
            google_drive_settings_from_clients(self.desktop, same)

    def test_duplicate_json_keys_and_unexpected_endpoints_are_rejected(self) -> None:
        duplicate = self.root / "duplicate.json"
        duplicate.write_text(
            '{"installed":{"project_id":"colabcola-fixture",'
            '"client_id":"desktop-fixture.apps.googleusercontent.com",'
            '"client_secret":"one","client_secret":"two",'
            f'"auth_uri":"{AUTH_URI}","token_uri":"{TOKEN_URI}"}}',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(DriveTransportError, "duplicate JSON object key"):
            load_google_oauth_client(duplicate)

        unexpected = _client_document(
            project_id="colabcola-fixture",
            client_id="desktop-fixture.apps.googleusercontent.com",
            client_secret="private-value",
        )
        unexpected["installed"]["token_uri"] = "https://example.invalid/token"  # type: ignore[index]
        self.desktop.write_text(json.dumps(unexpected), encoding="utf-8")
        with self.assertRaisesRegex(DriveTransportError, "token endpoint"):
            load_google_oauth_client(self.desktop)

    def test_oauth_client_link_is_rejected_before_resolution(self) -> None:
        metadata = SimpleNamespace(st_mode=stat.S_IFLNK | 0o777, st_file_attributes=0)
        with mock.patch("colabcola.google_drive.os.lstat", return_value=metadata):
            with self.assertRaisesRegex(DriveTransportError, "non-link"):
                load_google_oauth_client(self.desktop)

    def test_api_adapter_queries_exact_app_visible_object_and_trashes_only_by_id(self) -> None:
        values = [
            {
                "appProperties": {
                    "colabcola_job_id": "job_123",
                    "colabcola_staged_sha256": "a" * 64,
                },
                "id": "payload_123",
                "mimeType": "application/octet-stream",
                "name": "job_123.colabcola",
                "parents": ["job_folder_123"],
                "size": "42",
                "trashed": False,
            }
        ]
        files = _Files(values)
        backend = object.__new__(GoogleDriveApiBackend)
        backend.settings = GoogleDriveSettings(
            schema_version=1,
            google_cloud_project_id="colabcola-fixture",
            desktop_client_secrets=self.desktop,
            colab_oauth_client_id="device-fixture.apps.googleusercontent.com",
        )
        backend._files = files

        matches = backend.find_payload(
            name="job_123.colabcola",
            parent_id="job_folder_123",
            app_properties={
                "colabcola_job_id": "job_123",
                "colabcola_staged_sha256": "a" * 64,
            },
        )
        self.assertEqual(matches[0].file_id, "payload_123")
        query = str(files.list_calls[0]["q"])
        self.assertIn("'job_folder_123' in parents", query)
        self.assertIn("trashed = false", query)
        self.assertIn("colabcola_job_id", query)
        self.assertIn("colabcola_staged_sha256", query)

        backend.trash(file_id="payload_123")
        self.assertEqual(
            files.update_calls,
            [{"fileId": "payload_123", "body": {"trashed": True}, "fields": "id,trashed"}],
        )


if __name__ == "__main__":
    unittest.main()
