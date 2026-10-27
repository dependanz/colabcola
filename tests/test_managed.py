import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from colabcola import cli
from colabcola.lifecycle_io import (publish_files, recover, snapshots, LifecycleError, digest, atomic)
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



if __name__ == "__main__":
    unittest.main()

