"""Contract tests for the shared public CLI output, entirely offline."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from colabcola.console import ConsoleUI, emit_json
from colabcola.managed import Progress


class SharedConsoleTests(unittest.TestCase):
    def test_plain_progress_and_error_use_one_prefix_without_stdout(self):
        err, out = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out):
            ui = ConsoleUI(err)
            ui.say("Package: ready")
            ui.error("invalid source", command="package")
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(err.getvalue(),
                         "[colabcola] Package: ready\n"
                         "[colabcola] Error: package: invalid source\n")

    def test_control_sequences_cannot_rewrite_terminal(self):
        err = io.StringIO()
        ui = ConsoleUI(err)
        ui.say("Runtime: token\x1b[31m\rsecret\x00")
        self.assertNotIn("\x1b", err.getvalue())
        self.assertNotIn("\r", err.getvalue())
        self.assertIn("[colabcola] Runtime: token", err.getvalue())
        self.assertIn("secret", err.getvalue())

    def test_rich_terminal_keeps_untrusted_text_literal(self):
        class Interactive(io.StringIO):
            def isatty(self):
                return True
        err = Interactive()
        ConsoleUI(err).say("Runtime: [dangerous tag]")
        self.assertIn("[dangerous tag]", err.getvalue())
        self.assertIn("[colabcola]", err.getvalue())

    def test_json_writer_never_emits_progress_to_stdout(self):
        out, err = io.StringIO(), io.StringIO()
        payload = {"state": "prepared", "job_id": "id"}
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            emit_json(payload, sort_keys=True, separators=(",", ":"))
        self.assertEqual(out.getvalue(), '{"job_id":"id","state":"prepared"}\n')
        self.assertEqual(err.getvalue(), "")
        self.assertEqual(json.loads(out.getvalue()), payload)

    def test_managed_progress_uses_shared_renderer(self):
        with mock.patch.object(ConsoleUI, "say") as say:
            Progress(Path("unused"), io.StringIO()).say("Artifacts: verified")
        say.assert_called_once_with("Artifacts: verified")

    def test_validate_failure_is_consistent_and_stdout_stays_empty(self):
        from colabcola.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            job = str(Path(tmp) / "does-not-exist.colajob")
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = main(["validate", job])
        self.assertEqual(code, 2)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("[colabcola] Error: ", err.getvalue())
        self.assertNotIn("\x1b", err.getvalue())


if __name__ == "__main__":
    unittest.main()
