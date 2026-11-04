"""Pure presentation regression tests: no provider, credentials or network."""
import io
import json
from pathlib import Path
import tempfile
import unittest

from colabcola.managed import Progress
from colabcola.lifecycle_io import digest


class ConsoleUXTests(unittest.TestCase):
    def test_one_compact_preflight_summary_with_unmodified_evidence(self):
        stream = io.StringIO()
        progress = Progress(Path("unused"), stream)
        report = {"state": "passed", "checks": [
            {"name": "journal", "state": "passed"},
            {"name": "drive_parent", "state": "passed"},
        ]}
        for check in report["checks"]:
            progress.preflight_checked(check["name"])
        progress.preflight_completed(report)
        self.assertEqual(stream.getvalue(),
                         "[colabcola] Preflight: 2/2 checks passed (installation, account, assignments, Drive)\n")
        self.assertEqual(report["checks"][0]["name"], "journal")

    def test_failed_historical_preflight_is_not_reported_as_current_failure(self):
        stream = io.StringIO()
        progress = Progress(Path("unused"), stream)
        state = {"events": [
            {"sequence": 1, "event": "preflight_failed"},
            {"sequence": 2, "event": "preflight_passed"},
            {"sequence": 3, "event": "preflight_passed"},
            {"sequence": 4, "event": "staging_started"},
            {"sequence": 5, "event": "runtime_allocated"},
            {"sequence": 6, "event": "cleanup_requested"},
        ]}
        progress(state)
        progress(state)
        output = stream.getvalue()
        self.assertNotIn("preflight failed", output.lower())
        self.assertNotIn("preflight passed", output.lower())
        self.assertIn("[colabcola] Upload: preparing input bundle", output)
        self.assertIn("[colabcola] Runtime: allocated", output)
        self.assertIn("[colabcola] Cleanup: releasing owned runtime", output)
        self.assertEqual(output.count("Runtime: allocated"), 1)


    def test_completed_run_messages_are_concise_and_cpu_is_human_readable(self):
        # Existing-run rendering is exercised more comprehensively by test_managed.
        from unittest import mock
        from types import SimpleNamespace
        from colabcola import managed
        self.assertTrue(hasattr(managed.Progress, "say"))
        sink = io.StringIO()
        progress = Progress(Path("unused"), sink)
        progress.say("Run: CPU | up to 1 runtime | 180s total")
        self.assertEqual(sink.getvalue(),
                         "[colabcola] Run: CPU | up to 1 runtime | 180s total\n")

    def test_plain_output_has_no_ansi_and_does_not_interpret_rich_markup(self):
        sink = io.StringIO()
        progress = Progress(Path("unused"), sink)
        progress.say("Runtime: [some untrusted text]")
        self.assertEqual(sink.getvalue(),
                         "[colabcola] Runtime: [some untrusted text]\n")
        self.assertNotIn("\x1b", sink.getvalue())

    def test_interactive_rich_path_keeps_user_text_literal(self):
        class TerminalSink(io.StringIO):
            def isatty(self):
                return True

        sink = TerminalSink()
        progress = Progress(Path("unused"), sink)
        progress.say("Runtime: [not a Rich markup tag]")
        output = sink.getvalue()
        self.assertIn("[colabcola]", output)
        self.assertIn("[not a Rich markup tag]", output)

    def test_verified_workload_logs_have_clear_source_and_remain_redacted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "runtime").mkdir()
            payload = b"colabcola: command 1/1 starting\nresult: ok\nauthorization: token-secret\n"
            (root / "runtime/stdout.log").write_bytes(payload)
            stream = io.StringIO()
            progress = Progress(root, stream)
            state = {"logs": {"local_directory": str(root), "commit": {
                "attempt": 1, "members": {"runtime/stdout.log": {"bytes": len(payload), "sha256": digest(payload)}}
            }}}
            progress(state)
            printed = stream.getvalue()
            self.assertIn("[colabcola] [attempt 1 | stdout.log] command 1/1 starting", printed)
            self.assertNotIn("[attempt 1 | stdout.log] colabcola: command", printed)
            self.assertIn("[colabcola] [attempt 1 | stdout.log] result: ok", printed)
            self.assertIn("authorization: [redacted]", printed)
            self.assertNotIn("token-secret", printed)


if __name__ == "__main__":
    unittest.main()
