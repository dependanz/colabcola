"""Offline regression for Windows' nonexistent terminal-globstar prefix.

The test executes the actual generated notebook output-validation block, not
a substitute algorithm. A mocked glob result reproduces Windows behavior on
Linux CI without depending on particular platform-specific glob semantics.
"""
from __future__ import annotations

import glob
import hashlib
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

from colabcola.runtime import runtime_cell_sources


def _output_validation_source():
    execution = dict(runtime_cell_sources())["execution"]
    anchor = execution.index("        output_records = {}")
    start = execution.rfind("if _COLABCOLA_TERMINAL_STATE is None:", 0, anchor)
    end = execution.index("\n\ndef _colabcola_file_evidence", anchor)
    if start < 0:
        raise AssertionError("output validation block missing")
    return execution[start:end]


class OptionalOutputGlobTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.job = Path(temp.name) / "job"
        self.job.mkdir()

    def validate(self, *, required=False, matched=None):
        errors = []
        namespace = {
            "COLABCOLA_JOB_ROOT": self.job,
            "COLABCOLA_MANIFEST": {"outputs": [
                {"pattern": "outputs/**", "required": required,
                 "maximum_bytes": 1024 * 1024}
            ]},
            "_COLABCOLA_TERMINAL_STATE": None,
            "_COLABCOLA_OUTPUTS": [],
            "glob": glob,
            "hashlib": hashlib,
            "os": os,
            "pathlib": __import__("pathlib"),
            "stat": stat,
        }

        def failure(stage, state, exception):
            errors.append((stage, type(exception).__name__, str(exception)))
            namespace["_COLABCOLA_TERMINAL_STATE"] = state

        namespace["_colabcola_fail"] = failure
        if matched is None:
            exec(_output_validation_source(), namespace)
        else:
            with mock.patch("glob.glob", return_value=[str(matched)]):
                exec(_output_validation_source(), namespace)
        return namespace, errors

    def test_optional_outputs_directory_may_not_exist(self):
        result, errors = self.validate()
        self.assertEqual(result["_COLABCOLA_TERMINAL_STATE"], "success")
        self.assertEqual(result["_COLABCOLA_OUTPUTS"], [])
        self.assertFalse(errors)

    def test_windows_nonexistent_prefix_placeholder_is_not_an_output(self):
        result, errors = self.validate(matched=self.job / "outputs")
        self.assertEqual(result["_COLABCOLA_TERMINAL_STATE"], "success")
        self.assertEqual(result["_COLABCOLA_OUTPUTS"], [])
        self.assertFalse(errors)

    def test_required_outputs_still_fail_when_directory_missing(self):
        result, errors = self.validate(required=True, matched=self.job / "outputs")
        self.assertEqual(result["_COLABCOLA_TERMINAL_STATE"], "output_validation_failed")
        self.assertEqual(errors[0][0], "output_validation")
        self.assertIn("required output pattern matched no files", errors[0][2])

    def test_other_disappeared_glob_match_is_not_ignored(self):
        result, errors = self.validate(matched=self.job / "outputs" / "gone.txt")
        self.assertEqual(result["_COLABCOLA_TERMINAL_STATE"], "output_validation_failed")
        self.assertEqual(errors[0][1], "FileNotFoundError")

    def test_real_file_is_still_verified_and_hashed(self):
        directory = self.job / "outputs"
        directory.mkdir()
        target = directory / "actual.txt"
        target.write_bytes(b"verified")
        result, errors = self.validate(matched=target)
        self.assertFalse(errors)
        self.assertEqual(result["_COLABCOLA_TERMINAL_STATE"], "success")
        self.assertEqual(result["_COLABCOLA_OUTPUTS"], [{
            "path": "outputs/actual.txt",
            "size": 8,
            "sha256": hashlib.sha256(b"verified").hexdigest()
        }])

    def test_dangling_symlink_at_prefix_is_not_ignored(self):
        link = self.job / "outputs"
        try:
            link.symlink_to(self.job / "nonexistent-dir", target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlink creation unavailable: {type(exc).__name__}")
        result, errors = self.validate(matched=link)
        self.assertEqual(result["_COLABCOLA_TERMINAL_STATE"], "output_validation_failed")
        self.assertEqual(errors[0][0], "output_validation")
        self.assertIn("unsafe output match", errors[0][2])


if __name__ == "__main__":
    unittest.main()
