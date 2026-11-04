"""Prompt/automatic-selection behavior with synthetic scanner snapshots only."""
import io
from pathlib import Path
import unittest
from unittest import mock

from colabcola import client_handoff as handoff
from colabcola.client_downloads import Candidate, ScanResult


class ClientHandoffTests(unittest.TestCase):
    def setUp(self):
        self.messages = []
        self.a = Candidate(Path("/fixture/Downloads/client_secret_a.json"), {"installed": {"client_secret": "PRIVATE-A"}}, "a" * 64)
        self.b = Candidate(Path("/fixture/Downloads/client_secret_b.json"), {"installed": {"client_secret": "PRIVATE-B"}}, "b" * 64)
        patch = mock.patch.object(handoff.client_downloads, "download_directories", return_value=[Path("/fixture/Downloads")])
        patch.start(); self.addCleanup(patch.stop)

    def choose(self, answers=(), **kwargs):
        ask = mock.Mock(side_effect=list(answers))
        result = handoff.choose_client("fixture-project", input_fn=ask, interactive=False, say=self.messages.append, **kwargs)
        self.assertNotIn("PRIVATE-", " ".join(self.messages))
        return result, ask

    def test_unique_stable_snapshot_is_returned_without_reopening_or_prompting(self):
        snapshot = ScanResult((self.a,), False)
        with mock.patch.object(handoff.client_downloads, "scan_downloads", return_value=snapshot) as scan, \
             mock.patch.object(handoff.time, "sleep") as delay:
            result, ask = self.choose(expected_sha256="a" * 64)
        self.assertIs(result, self.a)
        ask.assert_not_called()
        self.assertEqual(scan.call_count, 2)
        self.assertEqual(scan.call_args.kwargs["expected_sha256"], "a" * 64)
        delay.assert_called_once_with(handoff.POLL_SECONDS)

    def test_changed_or_incomplete_second_snapshot_requires_manual_input(self):
        for second in (ScanResult((self.b,), False), ScanResult((self.a,), True), ScanResult((), False)):
            with self.subTest(second=second), \
                 mock.patch.object(handoff.client_downloads, "scan_downloads", side_effect=[ScanResult((self.a,), False), second]), \
                 mock.patch.object(handoff.time, "sleep"):
                result, ask = self.choose(["later"])
                self.assertIsNone(result); ask.assert_called_once()

    def test_multiple_clients_are_not_selected_by_recency(self):
        with mock.patch.object(handoff.client_downloads, "scan_downloads", return_value=ScanResult((self.a, self.b), False)), \
             mock.patch.object(handoff.time, "sleep") as delay:
            result, ask = self.choose(["2"])
        self.assertEqual(result, str(self.b.path))
        delay.assert_not_called(); ask.assert_called_once()
        self.assertIn("More than one different", " ".join(self.messages))

    def test_no_file_does_not_wait_in_noninteractive_mode(self):
        with mock.patch.object(handoff.client_downloads, "scan_downloads", return_value=ScanResult((), False)) as scan, \
             mock.patch.object(handoff.time, "sleep") as delay:
            result, ask = self.choose(["/elsewhere/desktop.json"])
        self.assertEqual(result, "/elsewhere/desktop.json")
        self.assertEqual(scan.call_count, 1); delay.assert_not_called(); ask.assert_called_once()

    def test_opt_out_does_not_discover_or_scan(self):
        with mock.patch.object(handoff.client_downloads, "download_directories", side_effect=AssertionError("no discovery")), \
             mock.patch.object(handoff.client_downloads, "scan_downloads", side_effect=AssertionError("no scan")):
            result, _ = self.choose(["manual.json"], automatic=False)
        self.assertEqual(result, "manual.json")

    def test_empty_later_eof_and_interrupt_remain_explicit_pause_paths(self):
        with mock.patch.object(handoff.client_downloads, "download_directories", return_value=[]):
            for answer in ("", " later ", "LATER"):
                self.assertIsNone(self.choose([answer])[0])
            for error in (EOFError(), KeyboardInterrupt()):
                with self.assertRaises(type(error)):
                    self.choose([error])

    def test_console_paths_are_escaped_and_client_contents_are_not_displayed(self):
        candidate = Candidate(Path("/fixture/Downloads\nnot-a-terminal-command/client_secret_a.json"), self.a.client, self.a.sha256)
        with mock.patch.object(handoff.client_downloads, "scan_downloads", return_value=ScanResult((candidate,), False)), \
             mock.patch.object(handoff.time, "sleep"):
            self.choose()
        self.assertTrue(any("\\nnot-a-terminal-command" in line for line in self.messages))
        self.assertFalse(any("\n" in line for line in self.messages))


if __name__ == "__main__":
    unittest.main()
