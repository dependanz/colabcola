"""Download detection using temporary files and mocked platform discovery only."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from colabcola import client_downloads as downloads
from colabcola.drive_oauth import validate_client
from colabcola.lifecycle_io import digest, encoded


class ClientDownloadsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="client-download-fixtures-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folder = self.root / "Downloads"
        self.folder.mkdir()
        for target in ("colabcola.client_downloads._known_folder_downloads",
                       "colabcola.client_downloads._windows_downloads_from_wsl",
                       "colabcola.client_downloads.subprocess.run"):
            patch = mock.patch(target, side_effect=AssertionError("real discovery forbidden"))
            patch.start(); self.addCleanup(patch.stop)
        patch = mock.patch.object(Path, "home", side_effect=AssertionError("real home forbidden"))
        patch.start(); self.addCleanup(patch.stop)

    @staticmethod
    def client(project="fixture-project", identity="123-fixture.apps.googleusercontent.com", secret="PRIVATE-FIXTURE-SECRET"):
        return {"installed": {"client_id": identity, "client_secret": secret, "project_id": project,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth", "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"]}}

    def write(self, name="client_secret_fixture.json", value=None, folder=None):
        path = (folder or self.folder) / name
        path.write_text(json.dumps(self.client() if value is None else value), encoding="utf-8")
        return path

    def scan(self, directories=None, **kwargs):
        return downloads.scan_downloads([self.folder] if directories is None else directories,
                                        project_id="fixture-project", **kwargs)

    def test_desktop_snapshot_is_normalized_bound_and_secret_safe_in_repr(self):
        path = self.write()
        before = path.read_bytes()
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            result = self.scan()
        self.assertFalse(result.incomplete)
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertEqual(candidate.path, path)
        self.assertEqual(candidate.client, validate_client(self.client()))
        self.assertEqual(candidate.sha256, digest(encoded(candidate.client)))
        self.assertEqual(path.read_bytes(), before)
        self.assertNotIn("PRIVATE-", repr(result) + repr(candidate) + output.getvalue())
        self.assertNotIn("client_secret", repr(candidate).replace(path.name, ""))
        self.assertEqual(output.getvalue(), "")

    def test_duplicate_downloads_deduplicate_normalized_client(self):
        original = self.write()
        duplicate = self.client()
        duplicate["installed"]["redirect_uris"] = ["https://unused.example"]
        duplicate["installed"]["auth_provider_x509_cert_url"] = "ignored metadata"
        self.write("client_secret_fixture (1).json", duplicate)
        self.write("client_secret_fixture(2).json")
        result = self.scan()
        self.assertFalse(result.incomplete)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].sha256, digest(encoded(validate_client(self.client()))))
        self.assertIn(result.candidates[0].path, [original, self.folder / "client_secret_fixture (1).json", self.folder / "client_secret_fixture(2).json"])

    def test_distinct_clients_and_changed_secret_are_ambiguous(self):
        self.write()
        self.write("client_secret_second.json", self.client(identity="456-other.apps.googleusercontent.com"))
        self.write("client_secret_changed.json", self.client(secret="PRIVATE-DIFFERENT-SECRET"))
        result = self.scan()
        self.assertFalse(result.incomplete)
        self.assertEqual(len(result.candidates), 3)
        self.assertEqual(len({candidate.sha256 for candidate in result.candidates}), 3)

    def test_expected_binding_returns_only_exact_client_without_rebinding(self):
        expected = digest(encoded(validate_client(self.client())))
        self.write("client_secret_changed.json", self.client(secret="PRIVATE-DIFFERENT-SECRET"))
        self.assertEqual(self.scan(expected_sha256=expected).candidates, ())
        path = self.write()
        result = self.scan(expected_sha256=expected)
        self.assertEqual([candidate.path for candidate in result.candidates], [path])
        self.assertFalse(result.incomplete)

    def test_wrong_project_web_device_tokens_service_account_and_endpoints_ignored(self):
        values = [self.client(project="another-project"), {"web": self.client()["installed"]},
                  {"type": "service_account", "private_key": "PRIVATE-KEY"}, {"type": "authorized_user", "refresh_token": "PRIVATE-TOKEN"},
                  {"device_code": "PRIVATE-DEVICE"}, {"installed": {"client_id": "missing"}}]
        bad_auth = self.client(); bad_auth["installed"]["auth_uri"] = "https://attacker.example/auth"
        bad_token = self.client(); bad_token["installed"]["token_uri"] = "https://attacker.example/token"
        for index, value in enumerate([*values, bad_auth, bad_token]):
            self.write(f"client_secret_{index}.json", value)
        self.assertEqual(self.scan(), downloads.ScanResult((), False))

    def test_only_matching_names_are_read_and_subdirectories_are_not_scanned(self):
        for name in ("arbitrary.json", "client_secret_fixture.json.crdownload", "client_secret_fixture.json.tmp", "token.json", "client_secret_.json"):
            self.write(name)
        nested = self.folder / "nested"; nested.mkdir()
        self.write(folder=nested)
        with mock.patch.object(downloads, "_read_snapshot", side_effect=AssertionError("unrelated file read")):
            self.assertEqual(self.scan(), downloads.ScanResult((), False))

    def test_malformed_partial_and_invalid_encoding_prevent_unique_selection(self):
        self.write()
        malformed = self.folder / "client_secret_partial.json"
        for body in (b'{"installed":', b"", b"\xff\xfeINVALID"):
            with self.subTest(body=body):
                malformed.write_bytes(body)
                result = self.scan()
                self.assertEqual(len(result.candidates), 1)
                self.assertTrue(result.incomplete)

    def test_oversized_matching_file_is_not_read_and_prevents_unique_selection(self):
        path = self.folder / "client_secret_huge.json"
        path.write_bytes(b"x" * (downloads.MAX_FILE_BYTES + 1))
        with mock.patch.object(downloads.os, "open", side_effect=AssertionError("oversized file opened")):
            result = self.scan()
        self.assertEqual(result.candidates, ())
        self.assertTrue(result.incomplete)

    def test_unreadable_matching_file_marks_incomplete_without_error_text(self):
        self.write()
        with mock.patch.object(downloads.os, "open", side_effect=PermissionError("PRIVATE-PATH")):
            self.assertEqual(self.scan(), downloads.ScanResult((), True))

    def test_changed_during_read_is_incomplete_and_reader_stays_bounded(self):
        path = self.write()
        original_read = os.read
        requested = []
        changed = False
        def grow(descriptor, size):
            nonlocal changed
            requested.append(size)
            body = original_read(descriptor, size)
            if not changed:
                changed = True
                with path.open("ab") as stream:
                    stream.write(b"x" * (downloads.MAX_FILE_BYTES * 2))
            return body
        with mock.patch.object(downloads.os, "read", side_effect=grow):
            result = self.scan()
        self.assertEqual(result.candidates, ())
        self.assertTrue(result.incomplete)
        self.assertLessEqual(sum(requested), downloads.MAX_FILE_BYTES * 2 + 2)
        self.assertLessEqual(max(requested), downloads.MAX_FILE_BYTES + 1)

    def test_replaced_file_between_lstat_and_open_is_rejected(self):
        self.write()
        original_fstat = os.fstat
        def changed(descriptor):
            value = original_fstat(descriptor)
            return SimpleNamespace(st_dev=value.st_dev, st_ino=value.st_ino + 1, st_mode=value.st_mode,
                st_size=value.st_size, st_mtime_ns=value.st_mtime_ns, st_ctime_ns=value.st_ctime_ns)
        with mock.patch.object(downloads.os, "fstat", side_effect=changed):
            self.assertEqual(self.scan(), downloads.ScanResult((), True))

    def test_symlink_file_and_symlink_directory_are_not_followed(self):
        outside = self.root / "outside"; outside.mkdir()
        target = self.write(folder=outside)
        try:
            (self.folder / "client_secret_link.json").symlink_to(target)
            linked = self.root / "linked"; linked.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            self.skipTest("host cannot create fixture symlinks: " + type(exc).__name__)
        with mock.patch.object(downloads.os, "open", side_effect=AssertionError("link target opened")):
            self.assertEqual(self.scan([self.folder, linked]), downloads.ScanResult((), False))

    def test_junction_marked_file_is_not_opened(self):
        target = self.write()
        with mock.patch.object(Path, "is_junction", side_effect=lambda: True, create=True), \
             mock.patch.object(downloads.os, "open", side_effect=AssertionError("junction opened")):
            self.assertEqual(self.scan(), downloads.ScanResult((), False))

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO fixture requires POSIX")
    def test_special_file_does_not_block_or_open(self):
        os.mkfifo(self.folder / "client_secret_fifo.json")
        with mock.patch.object(downloads.os, "open", side_effect=AssertionError("FIFO opened")):
            self.assertEqual(self.scan(), downloads.ScanResult((), False))

    def test_directory_enumeration_failure_or_overflow_is_incomplete(self):
        with mock.patch.object(downloads.os, "scandir", side_effect=PermissionError("PRIVATE-DIRECTORY")):
            self.assertEqual(self.scan(), downloads.ScanResult((), True))
        for index in range(4):
            self.write(f"ordinary{index}.json")
        with mock.patch.object(downloads, "MAX_ENTRIES", 2):
            self.assertTrue(self.scan().incomplete)

    def test_matching_file_and_directory_bounds_stop_scanning(self):
        for index in range(3):
            self.write(f"client_secret_{index}.json")
        with mock.patch.object(downloads, "MAX_MATCHING_FILES", 2):
            self.assertTrue(self.scan().incomplete)
        with mock.patch.object(downloads, "MAX_DIRECTORIES", 1):
            self.assertTrue(self.scan([self.folder, self.root]).incomplete)

    def test_missing_downloads_directory_is_empty_and_duplicate_paths_scan_once(self):
        self.write()
        self.assertEqual(self.scan([self.root / "missing"]), downloads.ScanResult((), False))
        original = os.scandir
        with mock.patch.object(downloads.os, "scandir", wraps=original) as scan:
            self.assertEqual(len(self.scan([self.folder, self.folder]).candidates), 1)
        scan.assert_called_once_with(self.folder)

    def test_invalid_project_and_digest_fail_before_any_directory_access(self):
        with mock.patch.object(downloads.os, "scandir", side_effect=AssertionError("must validate first")):
            with self.assertRaises(ValueError):
                self.scan(expected_sha256="not-a-digest")
            with self.assertRaises(Exception) as caught:
                downloads.scan_downloads([self.folder], project_id="../PRIVATE")
            self.assertNotIn("PRIVATE", str(caught.exception))

    def linux(self, **environment):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(downloads.sys, "platform", "linux"))
        stack.enter_context(mock.patch.object(downloads.platform, "release", return_value="generic-linux"))
        stack.enter_context(mock.patch.object(Path, "home", return_value=self.root))
        stack.enter_context(mock.patch.dict(os.environ, environment, clear=True))
        return stack

    def test_linux_default_and_xdg_paths_are_current_user_only(self):
        self.linux()
        self.assertEqual(downloads.download_directories(), [self.folder])
        config = self.root / ".config"; config.mkdir()
        settings = config / "user-dirs.dirs"
        settings.write_text('XDG_DOWNLOAD_DIR="$HOME/Received files"\n', encoding="utf-8")
        self.assertEqual(downloads.download_directories(), [self.root / "Received files"])
        settings.write_text('XDG_DOWNLOAD_DIR="$HOME/"\n', encoding="utf-8")
        self.assertEqual(downloads.download_directories(), [])

    def test_xdg_config_home_and_shell_expressions_never_execute(self):
        config = self.root / "config elsewhere"; config.mkdir()
        self.linux(XDG_CONFIG_HOME=str(config))
        settings = config / "user-dirs.dirs"
        settings.write_text('XDG_DOWNLOAD_DIR="${HOME}/Transfers"\n', encoding="utf-8")
        self.assertEqual(downloads.download_directories(), [self.root / "Transfers"])
        settings.write_text('XDG_DOWNLOAD_DIR="$(touch PRIVATE)"\n', encoding="utf-8")
        self.assertEqual(downloads.download_directories(), [self.folder])
        self.assertFalse((self.root / "PRIVATE").exists())

    def test_windows_uses_current_user_known_folder_without_powershell(self):
        with mock.patch.object(downloads.sys, "platform", "win32"), \
             mock.patch.object(downloads, "_known_folder_downloads", return_value=self.folder) as known:
            self.assertEqual(downloads.download_directories(), [self.folder])
        known.assert_called_once_with()

    def test_unavailable_platform_discovery_gracefully_skips(self):
        with mock.patch.object(downloads.sys, "platform", "win32"), \
             mock.patch.object(downloads, "_known_folder_downloads", side_effect=OSError("PRIVATE-ERROR")):
            self.assertEqual(downloads.download_directories(), [])
        with mock.patch.object(downloads.sys, "platform", "unsupported"):
            self.assertEqual(downloads.download_directories(), [])

    def test_wsl_adds_one_current_windows_location_and_deduplicates(self):
        self.linux(WSL_DISTRO_NAME="FixtureUbuntu")
        windows = self.root / "Windows Downloads"
        with mock.patch.object(downloads, "_windows_downloads_from_wsl", return_value=windows) as query:
            self.assertEqual(downloads.download_directories(), [self.folder, windows])
        query.assert_called_once_with()
        with mock.patch.object(downloads, "_windows_downloads_from_wsl", return_value=self.folder):
            self.assertEqual(downloads.download_directories(), [self.folder])

    def test_wsl_query_is_static_bounded_and_maps_only_returned_drive(self):
        # Save the real function despite the safety patch installed in setUp.
        function = _REAL_WSL_QUERY
        completed = subprocess.CompletedProcess([], 0, json.dumps(r"D:\Current User\Redirected Downloads"), "")
        with mock.patch.object(downloads.subprocess, "run", return_value=completed) as run:
            self.assertEqual(function(), Path("/mnt/d/Current User/Redirected Downloads"))
        run.assert_called_once()
        argv = run.call_args.args[0]
        self.assertEqual(argv[:4], ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"])
        self.assertIn(downloads._DOWNLOADS_GUID, argv[4])
        self.assertNotIn("Get-ChildItem", argv[4])
        self.assertEqual(run.call_args.kwargs["timeout"], 5)
        self.assertEqual(run.call_args.kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(run.call_args.kwargs["stderr"], subprocess.DEVNULL)

    def test_wsl_query_timeout_bad_output_and_unc_never_guess_paths(self):
        for stdout in ("PRIVATE-INVALID", json.dumps(r"\\server\share\Downloads"), json.dumps("relative"), json.dumps({"path": "private"}), "x" * 16385):
            with self.subTest(stdout=stdout[:50]), mock.patch.object(downloads.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout, "")):
                self.assertIsNone(_REAL_WSL_QUERY())
        with mock.patch.object(downloads.subprocess, "run", side_effect=subprocess.TimeoutExpired("fixture", 5)):
            self.assertIsNone(_REAL_WSL_QUERY())


_REAL_WSL_QUERY = downloads._windows_downloads_from_wsl


if __name__ == "__main__":
    unittest.main()
