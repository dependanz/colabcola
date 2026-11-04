import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile
from colabcola import cli
from colabcola.lifecycle_io import (publish, publish_files, recover, snapshots, LifecycleError, digest, atomic, encoded)
from colabcola.lifecycle_worker import checkpoint_members
from colabcola.managed import Progress
from test_lifecycle import MemoryStore
from test_notebook import notebook_job


class ManagedTests(unittest.TestCase):
    def test_chunk_roundtrip_reuse_and_corrupt_chunk_never_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "large"; source.write_bytes(b"abcdefgh" * 10)
            store = MemoryStore(); identity = {"run_id": "a" * 32, "job_id": "b" * 64}
            with mock.patch("colabcola.lifecycle_io.CHUNK_BYTES", 16):
                snap = publish_files(store, identity, "checkpoint", {"fits/a.zip": source}, step=5)
                self.assertEqual(len(snap["commit"]["members"]["fits/a.zip"]["chunks"]), 5)
                count = len(store.objects)
                publish_files(store, identity, "checkpoint", {"fits/a.zip": source}, step=6)
                self.assertEqual(len(store.objects), count + 1)
                self.assertEqual(len(snapshots(store, identity, "checkpoint")), 2)
                target = recover(store, snap, root / "recovered")
                self.assertEqual((target / "fits/a.zip").read_bytes(), source.read_bytes())
                ref = snap["commit"]["members"]["fits/a.zip"]["chunks"][0]
                store.objects[ref["id"]] = ("member", "bad", b"wrong")
                with self.assertRaises(LifecycleError):
                    recover(store, snap, root / "cold" / "bad")
                self.assertFalse((root / "cold" / "bad").exists())
                self.assertFalse(list((root / "cold").glob("bad.*.partial")))

    def test_v2_checkpoint_requires_completed_artifact_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "fits").mkdir()
            (root / "state.pt").write_bytes(b"state")
            (root / "fits/a.zip").write_bytes(b"artifact")
            atomic(root / "state.pt.json", {"schema": 2, "step": 3, "file": "state.pt", "bytes": 5,
                   "sha256": digest(b"state"), "artifacts": {"fits/a.zip": {"bytes": 8, "sha256": digest(b"artifact")}}})
            members = checkpoint_members(root, "state*.json")[0][1]
            self.assertEqual(set(members), {"state.pt.json", "state.pt", "fits/a.zip"})
            (root / "fits/a.zip").write_bytes(b"bad")
            with self.assertRaises(LifecycleError):
                checkpoint_members(root, "state*.json")

    def test_bare_managed_job_prepare_is_offline_and_forwards_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); script = root / "work.py"; script.write_text("print('ok')")
            value = notebook_job(script)
            value["runtime"]["version"] = "latest"
            value["managed_recovery"] = {"checkpoint_pattern": "outputs/state*.json", "log_patterns": []}
            job = root / "work.colajob"; atomic(job, value)
            atomic(root / "token.json", ["never-call-token"])
            atomic(job.with_suffix(".run.json"), {"drive_parent": "test", "token_command": "token.json",
                   "gpu": "NONE", "max_attempts": 2, "max_seconds": 90})
            with mock.patch("colabcola.managed.supervise", side_effect=AssertionError("must be offline")), mock.patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(cli.main(["--prepare-only", str(job), "--x", "a b"]), 0)
                self.assertEqual(cli.main(["--prepare-only", str(job), "--x", "a b"]), 0)
                self.assertEqual(cli.main(["--prepare-only", str(job), "--x", "changed"]), 2)

    def test_human_progress_only_emits_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); atomic(root / "progress.json", {"completed_fits": 2, "total_fits": 24,
                                    "image": "kodim01.png", "depth": 1, "width": 32, "step": 5, "steps": 10})
            stream = io.StringIO(); reporter = Progress(root, stream)
            state = {"events": [{"sequence": 1, "event": "checkpoint_verified"}],
                     "logs": {"local_directory": str(root)}}
            reporter(state); reporter(state)
            self.assertEqual(stream.getvalue().count("Fits 2/24"), 1)
            self.assertEqual(stream.getvalue().count("checkpoint verified"), 1)

    def test_waiting_authorization_is_neutral_and_real_expiry_warning_still_displays(self):
        stream = io.StringIO(); reporter = Progress(Path("unused"), stream)
        pending = {"drive_authorization": {"state": "waiting", "expires_at": 0}}
        reporter(pending); reporter(pending)
        self.assertEqual(stream.getvalue().count("Drive access pending runtime bootstrap"), 1)
        self.assertNotIn("expiry protection", stream.getvalue())
        self.assertNotIn("unavailable", stream.getvalue())
        reporter({"drive_authorization": {"state": "expiring", "expires_at": 1}})
        self.assertIn("Drive access expiring; expiry protection remains active", stream.getvalue())


    def test_snapshot_bound_is_checked_before_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "file"; path.write_bytes(b"12345")
            store = MemoryStore()
            with mock.patch("colabcola.lifecycle_io.SNAPSHOT_BYTES", 4):
                with self.assertRaises(LifecycleError):
                    publish_files(store, {"run_id": "a"*32, "job_id": "b"*64}, "checkpoint", {"x": path})
            self.assertEqual(store.objects, {})

    def test_terminal_reconnect_is_offline_and_new_segment_binds_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); script = root / "work.py"; script.write_text("print('ok')")
            mapping = notebook_job(script); mapping["runtime"]["version"] = "latest"
            mapping["managed_recovery"] = {"checkpoint_pattern": "outputs/state*.json", "log_patterns": []}
            job = root / "work.colajob"; atomic(job, mapping)
            atomic(root / "token.json", ["must-not-run"])
            atomic(job.with_suffix(".run.json"), {"drive_parent": "test", "token_command": "token.json",
                   "gpu": "NONE", "max_attempts": 2, "max_seconds": 90})
            with mock.patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(cli.main(["--prepare-only", str(job)]), 0)
                previous = job.with_suffix(".run")
                plan = json.loads((previous / "plan.json").read_text())
                checkpoint = {"sha256": "c"*64, "commit": {"job_id": plan["job_id"], "step": 7}}
                atomic(previous / "state.json", {"phase": "budget_exhausted", "attempts": [], "events": [],
                                                "checkpoint": checkpoint})
                with mock.patch("colabcola.managed.supervise", side_effect=AssertionError("no allocation")):
                    self.assertEqual(cli.main([str(job)]), 3)
                fresh = root / "segment2"
                self.assertEqual(cli.main(["--prepare-only", "--run-dir", str(fresh),
                    "--resume-from", str(previous), str(job)]), 0)
                new = json.loads((fresh / "plan.json").read_text())
                self.assertEqual(new["initial_checkpoint"], checkpoint)
                self.assertNotEqual(new["run_id"], plan["run_id"])
                self.assertEqual(json.loads((fresh / "state.json").read_text())["checkpoint"], checkpoint)


class TerminalReportingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.script = self.root / "work.py"; self.script.write_text("print('ok')")
        mapping = notebook_job(self.script); mapping["runtime"]["version"] = "latest"
        mapping["managed_recovery"] = {"checkpoint_pattern": "outputs/state*.json", "log_patterns": []}
        self.job = self.root / "work.colajob"; atomic(self.job, mapping)
        atomic(self.root / "token.json", ["must-not-run"])
        atomic(self.job.with_suffix(".run.json"), {"drive_parent": "test", "token_command": "token.json",
               "gpu": "NONE", "max_attempts": 1, "max_seconds": 90})
        with mock.patch("sys.stdout", new_callable=io.StringIO), mock.patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(cli.main(["--prepare-only", str(self.job)]), 0)
        self.directory = self.job.with_suffix(".run")
        self.plan = json.loads((self.directory / "plan.json").read_text())
        self.identity = {key: self.plan[key] for key in ("run_id", "job_id")}
        self.store = MemoryStore()
        self.state = {"phase": "completed", "attempts": [{"number": 1, "endpoint": "fixture", "released": True}],
                      "deadline": 12345, "events": [{"sequence": index + 1, "event": event}
                      for index, event in enumerate(("inputs_staged", "runtime_allocated", "cleanup_requested", "owned_runtime_released"))]}

    def final_snapshot(self, *, local=True, terminal="success"):
        payload = b"verified output"
        result = {"job_id": self.plan["job_id"], "terminal_state": terminal,
                  "outputs": [{"path": "outputs/result.txt", "size": len(payload), "sha256": digest(payload)}]}
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as target:
            target.writestr("result.json", encoded(result)); target.writestr("outputs/result.txt", payload)
        snapshot = publish(self.store, self.identity, "final", {
            "result.json": encoded(result), "artifacts.zip": archive.getvalue(),
            "runtime/stdout.log": b"HISTORICAL_WORKLOAD_LINE\n",
            "progress.json": encoded({"state": "completed", "completed_fits": 2, "total_fits": 2})}, attempt=1)
        if local:
            path = recover(self.store, snapshot, self.directory / "artifacts" / snapshot["sha256"])
            snapshot["local_directory"] = str(path)
        return snapshot

    def intermediate_snapshot(self):
        source = self.root / "clip.txt"; source.write_bytes(b"completed output")
        snapshot = publish_files(self.store, self.identity, "artifacts", {"outputs/clip.txt": source}, attempt=1)
        path = recover(self.store, snapshot, self.directory / "artifacts" / snapshot["sha256"])
        snapshot["local_directory"] = str(path)
        return snapshot

    def invoke(self, *, write_state=True):
        if write_state:
            atomic(self.directory / "state.json", self.state)
        output, messages = io.StringIO(), io.StringIO()
        with mock.patch("sys.stdout", output), mock.patch("sys.stderr", messages), \
                mock.patch("colabcola.managed.supervise", side_effect=AssertionError("must not supervise a terminal run")), \
                mock.patch("colabcola.lifecycle_provider.provider_for_plan", side_effect=AssertionError("must not construct a provider")), \
                mock.patch("colabcola.lifecycle_provider.GoogleProvider", side_effect=AssertionError("must not allocate")), \
                mock.patch("colabcola.lifecycle.DriveStore", return_value=self.store), \
                mock.patch.object(self.store, "put", side_effect=AssertionError("late recovery must not upload")):
            code = cli.main([str(self.job)])
        return code, output.getvalue(), messages.getvalue()

    def test_terminal_repeat_rechecks_final_without_replaying_history_or_allocating(self):
        self.state["final"] = self.final_snapshot()
        before = encoded(self.state)
        code, output, messages = self.invoke()
        self.assertEqual(code, 0, messages)
        self.assertIn("Run: viewing existing completed journal (no new runtime)", messages)
        self.assertIn("Run: CPU | up to 1 runtime | 90s total", messages)
        for historical in ("inputs staged", "runtime allocated", "cleanup requested", "owned runtime released",
                           "HISTORICAL_WORKLOAD_LINE", "Fits 2/2", "Live logs:"):
            self.assertNotIn(historical, messages)
        summary = json.loads(output)
        self.assertEqual(summary["final_artifacts_status"], "verified")
        self.assertEqual(summary["completed_outputs_status"], "not_configured")
        self.assertIsNone(summary["completed_artifact_directory"])
        self.assertTrue(summary["viewing_existing_run"]); self.assertTrue(summary["run_success"])
        self.assertNotIn("Optional intermediate", messages)
        self.assertEqual((self.directory / "state.json").read_bytes(), before)

    def test_late_recovery_reports_new_saved_material_and_preserves_failed_outcome(self):
        self.state.update(phase="failed", reason="runtime_lost")
        old = publish(self.store, self.identity, "logs", {"runtime/stdout.log": b"OLD_SAVED_LINE\n"}, attempt=1)
        old["local_directory"] = str(recover(self.store, old, self.directory / "logs" / old["sha256"]))
        self.state["logs"] = old
        publish(self.store, self.identity, "logs", {"runtime/stdout.log": b"NEW_SAVED_LINE\n"}, attempt=1, step=1)
        self.final_snapshot(local=False)
        attempts = json.loads(json.dumps(self.state["attempts"]))
        code, output, messages = self.invoke()
        self.assertEqual(code, 3, messages)
        self.assertIn("Run: viewing existing failed journal", messages)
        self.assertIn("Checking Drive for late artifacts and saved logs", messages)
        self.assertIn("Recovered additional saved logs from Drive for this existing run", messages)
        self.assertIn("Recovered final artifact bundle from Drive for this existing run", messages)
        for text in ("OLD_SAVED_LINE", "NEW_SAVED_LINE", "HISTORICAL_WORKLOAD_LINE", "runtime allocated", "Live logs:"):
            self.assertNotIn(text, messages)
        after = json.loads((self.directory / "state.json").read_text())
        self.assertEqual(after["attempts"], attempts); self.assertEqual(after["deadline"], 12345)
        summary = json.loads(output)
        self.assertEqual(summary["state"], "failed"); self.assertEqual(summary["reason"], "runtime_lost")
        self.assertEqual(summary["final_artifacts_status"], "verified"); self.assertFalse(summary["run_success"])
        again, _, repeated = self.invoke(write_state=False)
        self.assertEqual(again, 3); self.assertNotIn("Recovered additional", repeated)
        self.assertNotIn("Recovered final", repeated)

    def test_intermediate_only_is_distinct_from_missing_final(self):
        self.state.update(phase="failed", artifacts=self.intermediate_snapshot())
        code, output, messages = self.invoke()
        self.assertEqual(code, 3)
        summary = json.loads(output)
        self.assertEqual(summary["final_artifacts_status"], "missing")
        self.assertEqual(summary["completed_outputs_status"], "verified")
        self.assertIsNone(summary["result_directory"])
        self.assertTrue(summary["completed_artifact_directory"])
        self.assertIn("Final artifact bundle is not available locally", messages)
        self.assertIn("Intermediate completed-output commits verified", messages)
        self.assertFalse(summary["run_success"])

    def test_recorded_completion_without_final_does_not_return_success(self):
        code, output, messages = self.invoke()
        self.assertEqual(code, 3)
        summary = json.loads(output)
        self.assertEqual(summary["state"], "completed")
        self.assertEqual(summary["final_artifacts_status"], "missing")
        self.assertFalse(summary["run_success"])
        self.assertIn("artifact recovery is incomplete", messages)

    def test_configured_intermediate_outputs_can_be_absent_while_final_is_verified(self):
        self.state["final"] = self.final_snapshot()
        self.plan["artifact_groups"] = [{"name": "optional"}]
        atomic(self.directory / "plan.json", self.plan)
        code, output, messages = self.invoke()
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["completed_outputs_status"], "not_available")
        self.assertIn("Artifacts: configured intermediate outputs were not recovered", messages)

    def test_corrupt_or_missing_saved_artifacts_are_not_reported_as_success(self):
        self.state["final"] = self.final_snapshot()
        archive = Path(self.state["final"]["local_directory"]) / "artifacts.zip"
        archive.write_bytes(b"corrupt")
        code, output, _ = self.invoke()
        self.assertEqual(code, 2); self.assertNotIn('"run_success": true', output)
        archive.unlink()
        code, output, _ = self.invoke()
        self.assertEqual(code, 2); self.assertNotIn('"run_success": true', output)

    def test_corrupt_intermediate_commit_is_not_claimed_verified(self):
        self.state.update(phase="failed", artifacts=self.intermediate_snapshot())
        (Path(self.state["artifacts"]["local_directory"]) / "outputs/clip.txt").write_bytes(b"tampered")
        code, output, messages = self.invoke()
        self.assertEqual(code, 2)
        self.assertNotIn("Intermediate completed-output commits verified:", messages)
        self.assertNotIn('"completed_outputs_status": "verified"', output)

    def test_terminal_reopen_still_rejects_changed_frozen_input_before_recovery(self):
        self.state["final"] = self.final_snapshot()
        (self.directory / "notebook.ipynb").write_text("tampered")
        with mock.patch("colabcola.managed.recover_terminal_final", side_effect=AssertionError("validation must come first")):
            code, _, messages = self.invoke()
        self.assertEqual(code, 2); self.assertIn("prepared source changed", messages)

    def test_fresh_and_reopened_completion_use_same_artifact_summary(self):
        self.state["final"] = self.final_snapshot()
        stdout, stderr = io.StringIO(), io.StringIO()
        def finish(directory, approval, **kwargs):
            kwargs["validator"](self.plan)
            return self.state
        with mock.patch("sys.stdout", stdout), mock.patch("sys.stderr", stderr), \
                mock.patch("colabcola.managed.supervise", side_effect=finish):
            self.assertEqual(cli.main([str(self.job)]), 0)
        fresh = json.loads(stdout.getvalue())
        code, output, _ = self.invoke()
        self.assertEqual(code, 0)
        reopened = json.loads(output)
        self.assertFalse(fresh.pop("viewing_existing_run")); self.assertTrue(reopened.pop("viewing_existing_run"))
        self.assertEqual(fresh, reopened)



if __name__ == "__main__":
    unittest.main()

