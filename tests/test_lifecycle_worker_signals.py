"""Linux regression: interrupted worker cannot orphan its owned child session.

No Google Colab, Drive credentials, GPU allocation, or external network needed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from colabcola.lifecycle_io import atomic
from colabcola import lifecycle_worker as worker


@unittest.skipUnless(sys.platform == "linux", "real Linux process groups required")
class WorkerSignalCleanupTests(unittest.TestCase):
    def exercise(self, source):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_id = "a" * 64
            job = root / "runtime" / "colabcola" / "jobs" / job_id
            job.mkdir(parents=True)
            config = {
                "directory": str(root), "content_root": str(root / "runtime"),
                "run_id": "r" * 32, "job_id": job_id,
                "parent": "fixture", "attempt": 1, "files": {},
                "deadline": time.time() + 180, "resume": None,
                "checkpoint_pattern": "outputs/no-checkpoints-*.json",
                "log_patterns": [], "poll_seconds": 0.01,
            }
            configuration = root / "worker.json"
            atomic(configuration, config)
            original_popen = subprocess.Popen
            children = []

            def spawn(_argv, **kwargs):
                self.assertTrue(kwargs["start_new_session"])
                child = original_popen(
                    [sys.executable, "-c", "import time;time.sleep(120)"],
                    **kwargs)
                children.append(child)
                return child

            def inject(_store):
                self.assertEqual(len(children), 1)
                self.assertIsNone(children[0].poll())
                if source == "keyboard":
                    raise KeyboardInterrupt()
                if source == "sigterm":
                    signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                    self.fail("SIGTERM handler did not interrupt")
                self.fail("unknown test injection")

            prior_sigterm = signal.getsignal(signal.SIGTERM)
            prior_sigint = signal.getsignal(signal.SIGINT)
            try:
                with (
                    mock.patch.object(worker, "DriveStore", return_value=object()),
                    mock.patch.object(worker, "acquire_inputs"),
                    mock.patch.object(worker, "verified_inputs"),
                    mock.patch.object(worker, "stop_reason", return_value=None),
                    mock.patch.object(worker.subprocess, "Popen", side_effect=spawn),
                    mock.patch.object(worker.CompletedArtifacts, "publish",
                                      side_effect=inject),
                ):
                    if source == "keyboard":
                        with self.assertRaises(KeyboardInterrupt):
                            worker.run(configuration)
                    else:
                        with self.assertRaises(SystemExit) as caught:
                            worker.run(configuration)
                        self.assertEqual(caught.exception.code, 128 + signal.SIGTERM)
                self.assertEqual(signal.getsignal(signal.SIGTERM), prior_sigterm)
                self.assertEqual(signal.getsignal(signal.SIGINT), prior_sigint)
                status = json.loads((root / "status.json").read_text())
                self.assertEqual(status["state"], "failed")
                if source == "keyboard":
                    self.assertEqual(status["reason"], "KeyboardInterrupt")
                    self.assertNotIn("termination_signal", status)
                else:
                    self.assertEqual(status["reason"], "SystemExit")
                    self.assertEqual(status["termination_signal"]["name"], "SIGTERM")
                    self.assertGreater(status["termination_signal"]["at"], 0)
                self.assertEqual(len(children), 1)
                self.assertIsNotNone(children[0].poll())
                with self.assertRaises(ProcessLookupError):
                    os.killpg(children[0].pid, 0)
            finally:
                for child in children:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait(timeout=5)

    def test_keyboard_interrupt_stops_owned_child_group(self):
        self.exercise("keyboard")

    def test_sigterm_handler_records_signal_and_stops_owned_group(self):
        self.exercise("sigterm")


if __name__ == "__main__":
    unittest.main()
