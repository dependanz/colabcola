"""Managed private assets: real storage protocol, no external services."""
from __future__ import annotations
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
import zipfile
from unittest import mock

from colabcola import lifecycle_io as lio, lifecycle_worker as worker, lifecycle
from colabcola import private_inputs, preflight
from colabcola.lifecycle_io import PrivateAssetStorage
from colabcola.lifecycle_provider import GoogleProvider
from colabcola.runtime import runtime_cell_sources
from test_lifecycle import MemoryStore, Provider
from test_notebook import notebook_job
from test_private_assets import AssetDriveHTTP


class ManagedPrivateInputTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        remote = self.root / "remote"
        remote.mkdir()
        self.drive = AssetDriveHTTP(remote)
        self.asset = self.root / "private-model.bin"
        self.body = b"private-model-fixture-" * 100
        self.asset.write_bytes(self.body)
        self.entry = {"target": "models/weights.bin", "bytes": len(self.body), "sha256": lio.digest(self.body)}
        self.source = self.root / "check.py"
        self.source.write_text('''import pathlib,hashlib,json
p = pathlib.Path('models/weights.bin')
report = {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
pathlib.Path('outputs').mkdir(exist_ok=True)
pathlib.Path('outputs/check.json').write_text(json.dumps(report))
''')
        self.helper = self.root / "helper.json"
        lio.atomic(self.helper, ["fixture-token-helper"])
        self.provider = Provider()
        self.token = lambda: {"access_token": "test-lease", "expires_at": time.time() + 3600}
        self.patch = mock.patch.object(lio, "CHUNK_BYTES", 64)
        self.patch.start(); self.addCleanup(self.patch.stop)

    def prepare(self, name="first", *, entries=None):
        value = notebook_job(self.source, name=name)
        value["run"]["seed"] = 1234 if name == "first" else 1235
        value["entrypoint"]["argv"] = []
        value["outputs"] = [{"pattern": "outputs/check.json", "required": True, "maximum_bytes": 4096}]
        value["managed_recovery"] = {"checkpoint_pattern": "outputs/state-*.json", "log_patterns": []}
        value["private_inputs"] = [{"source": str(self.asset), **self.entry}] if entries is None else entries
        job = self.root / (name + ".colajob")
        lio.atomic(job, value)
        directory = self.root / (name + ".run")
        result = lifecycle.prepare(SimpleNamespace(job=job, run_dir=directory, token_command=self.helper,
            drive_parent="private-parent", gpu="NONE", max_attempts=1, max_seconds=900,
            fault_after_checkpoint=False))
        plan = json.loads((directory / "plan.json").read_text())
        return directory, plan, result["approval_sha256"]

    def engine(self, directory):
        return lifecycle.Supervisor(directory, self.provider, MemoryStore(), self.token)

    def storage(self, *args, **kwargs):
        return PrivateAssetStorage("private-parent", lambda: "test-lease", store_factory=self.drive.factory)

    def stage(self, engine):
        with mock.patch.object(private_inputs, "PrivateAssetStorage", side_effect=self.storage):
            engine.tick()
        return engine.state

    def config(self, engine, name):
        # Exercise real provider configuration serialization without a real kernel.
        plan = {k: v for k, v in engine.plan.items() if k != "execution_mode"}
        attempt = dict(engine.state["attempts"][0], private_input_receipts=engine.state["private_input_receipts"])
        google = object.__new__(GoogleProvider)
        google._session = mock.Mock(return_value=object())
        google._execute = mock.Mock()
        google._upload = mock.Mock()
        google.boot("owned", plan, attempt, self.token())
        uploads = {call.args[1].rsplit("/", 1)[-1]: call.args[2] for call in google._upload.call_args_list}
        config = json.loads(uploads["worker.json"])
        self.assertNotIn(str(self.asset), uploads["worker.json"].decode())
        self.assertNotIn("private_input_sources", config)
        self.assertEqual(config["private_inputs"], [self.entry])
        control = self.root / (name + "-control")
        control.mkdir()
        for filename in ("notebook.ipynb", "lifecycle_worker.py", "lifecycle_io.py"):
            (control / filename).write_bytes(uploads[filename])
        config.update(directory=str(control), content_root=str(self.root / (name + "-runtime")))
        lio.atomic(control / "worker.json", config)
        return control / "worker.json", config

    def execute(self, config_path, config, *, failure=False):
        with mock.patch.object(worker, "PrivateAssetStorage", side_effect=self.storage), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if failure:
                with self.assertRaises(RuntimeError):
                    worker.execute_notebook(config_path)
            else:
                worker.execute_notebook(config_path)
        runtime = Path(config["content_root"]) / "colabcola"
        result = lifecycle.verify_final(runtime / "results" / config["job_id"], config["job_id"])
        return result, runtime / "jobs" / config["job_id"]

    def preflight_run(self, directory, approval):
        provider = SimpleNamespace(assignments=mock.Mock(return_value=[]), close=mock.Mock())
        with mock.patch.object(preflight, "installation_integrity", return_value={"files_verified": 1}), \
             mock.patch("colabcola.lifecycle_provider.dependency_report", return_value={"verified": True}), \
             mock.patch("colabcola.lifecycle_provider.provider_for_plan", return_value=provider), \
             mock.patch.object(lifecycle, "get_token", side_effect=lambda *_: self.token()), \
             mock.patch.object(preflight, "drive_parent", return_value={"owned": True}), \
             mock.patch.object(preflight.shutil, "disk_usage", return_value=SimpleNamespace(free=100*1024**3)), \
             mock.patch.object(lio, "PrivateAssetStorage", side_effect=self.storage):
            report, _, _ = preflight.run_preflight(directory, approval, close_provider=True)
        return report

    def test_two_distinct_jobs_reuse_verified_bytes(self):
        d, plan, approval = self.prepare()
        self.assertEqual(plan["private_inputs"], [self.entry])
        self.assertEqual(plan["private_input_sources"], {self.entry["target"]: str(self.asset)})
        self.assertNotIn(str(self.asset).encode(), (d / "notebook.ipynb").read_bytes())
        self.assertNotIn(self.body, (d / "notebook.ipynb").read_bytes())
        report = self.preflight_run(d, approval)
        self.assertFalse(report["upload_performed"])
        self.assertFalse(report["allocation_performed"])
        self.assertFalse(any(e[0] == "put" for e in self.drive.events))
        self.assertEqual(report["private_inputs"]["total_bytes"], len(self.body))
        self.assertEqual(report["private_inputs"]["total_limit_bytes"], 8*1024**3)
        self.assertIsNone(report["private_inputs"]["runtime_free_bytes"])
        self.assertGreater(report["private_inputs"]["local_recovery_headroom_bytes"], 0)
        self.assertNotIn(str(self.asset), json.dumps(report))
        a = self.engine(d)
        self.stage(a)
        self.assertEqual(a.state["phase"], "allocating")
        self.assertFalse(self.provider.allocations)
        self.assertEqual(a.state["private_input_staging"][self.entry["target"]]["state"], "published")
        self.assertGreater(a.state["private_input_staging"][self.entry["target"]]["confirmed_uploaded_bytes"], 0)
        config_path, config = self.config(a, "a")
        # Prove exact receipts work even when a newly uploaded commit is not indexed.
        self.drive.hide_catalog = True
        result, job_root = self.execute(config_path, config)
        self.drive.hide_catalog = False
        self.assertEqual(result["terminal_state"], "success")
        lio.verify_private_input_result(plan["private_inputs"], result)
        self.assertEqual((job_root / self.entry["target"]).read_bytes(), self.body)
        self.assertEqual(json.loads((job_root / "outputs/check.json").read_text()),
                         {k: self.entry[k] for k in ("bytes", "sha256")})
        posts = sum(e[0] == "put" for e in self.drive.events)
        # Independent job and absent local file: only the committed frozen bytes are available.
        self.asset.unlink()
        d2, plan2, approval2 = self.prepare("second")
        self.assertNotEqual(plan2["job_id"], plan["job_id"])
        self.assertEqual(self.preflight_run(d2, approval2)["private_inputs"]["assets"][0]["state"], "committed")
        b = self.engine(d2)
        self.stage(b)
        self.assertEqual(b.state["private_input_staging"][self.entry["target"]]["state"], "reused")
        self.assertEqual(b.state["private_input_staging"][self.entry["target"]]["confirmed_uploaded_bytes"], 0)
        self.assertEqual(b.state["private_input_receipts"], a.state["private_input_receipts"])
        self.assertEqual(sum(e[0] == "put" for e in self.drive.events), posts)
        config_path2, config2 = self.config(b, "b")
        result2, job_root2 = self.execute(config_path2, config2)
        lio.verify_private_input_result(plan2["private_inputs"], result2)
        self.assertEqual((job_root2 / self.entry["target"]).read_bytes(), self.body)
        self.assertEqual(sum(e[0] == "put" for e in self.drive.events), posts)

    def test_restart_binds_receipts_before_boot_and_final_evidence(self):
        d, plan, _ = self.prepare()
        a = self.engine(d)
        with mock.patch.object(private_inputs, "PrivateAssetStorage", side_effect=self.storage):
            # Simulate controller interruption after the asset has been journaled
            # but before the small input bundle is published.
            with mock.patch.object(lifecycle, "publish", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    a.tick()
        self.assertEqual(a.state["phase"], "staging")
        self.assertEqual(a.state["attempts"], [])
        posts = sum(e[0] == "put" for e in self.drive.events)
        self.drive.hide_catalog = True
        restarted = self.engine(d)
        self.stage(restarted)
        self.assertEqual(restarted.state["phase"], "allocating")
        self.assertEqual(sum(e[0] == "put" for e in self.drive.events), posts)
        restarted.tick()
        with mock.patch.object(self.provider, "boot", wraps=self.provider.boot) as boot:
            restarted.tick()
        self.assertEqual(boot.call_args.args[2]["private_input_receipts"], restarted.state["private_input_receipts"])
        self.assertEqual(restarted.state["phase"], "monitoring")
        config_path, config = self.config(restarted, "restart")
        result, _ = self.execute(config_path, config)
        lio.verify_private_input_result(plan["private_inputs"], result)
        # Actual result verification, final commit recovery, and owned release.
        runtime_result = Path(config["content_root"]) / "colabcola/results" / config["job_id"]
        lio.publish_files(restarted.store, plan, "final", {name: runtime_result / name
                          for name in ("result.json", "artifacts.zip")}, attempt=1)
        restarted.tick()
        self.assertEqual(restarted.state["phase"], "completed")
        self.assertTrue(restarted.state["attempts"][0]["released"])
        self.assertFalse(self.provider.active)

    def test_final_result_missing_private_evidence_cannot_succeed(self):
        d, plan, _ = self.prepare()
        a = self.engine(d)
        self.stage(a); a.tick(); a.tick()
        config_path, config = self.config(a, "missing-evidence")
        result, _ = self.execute(config_path, config)
        del result["transport"]["private_inputs"]
        root = Path(config["content_root"]) / "colabcola/results" / config["job_id"]
        lio.atomic(root / "result.json", result)
        archive = root / "artifacts.zip"
        with zipfile.ZipFile(archive) as original:
            members = {name: original.read(name) for name in original.namelist() if name != "result.json"}
        with zipfile.ZipFile(archive, "w") as modified:
            for name, body in members.items():
                modified.writestr(name, body)
            modified.writestr("result.json", lio.encoded(result))
        lio.publish_files(a.store, plan, "final", {name: root / name for name in ("result.json", "artifacts.zip")}, attempt=1)
        a.tick()
        self.assertEqual(a.state["phase"], "failed")
        self.assertEqual(a.state["reason"], "private_input_evidence_invalid")
        self.assertTrue(a.state["attempts"][0]["released"])

    def test_missing_or_mutated_source_fails_before_allocation(self):
        d, _, _ = self.prepare()
        self.asset.write_bytes(b"x" * len(self.body))
        a = self.engine(d)
        self.stage(a)
        self.assertEqual(a.state["phase"], "failed")
        self.assertEqual(a.state["reason"], "private_input_staging_failed")
        self.assertEqual(a.state["attempts"], [])
        self.assertFalse(any(e[0] == "put" for e in self.drive.events))
        self.assertFalse(self.provider.allocations)
        self.asset.unlink()
        d2, _, approval2 = self.prepare("missing")
        with self.assertRaisesRegex(preflight.PreflightError, "private_inputs failed"):
            self.preflight_run(d2, approval2)
        self.assertEqual(json.loads((d2 / "state.json").read_text())["phase"], "prepared")

    def test_receipt_scope_corruption_and_disk_fail_closed(self):
        d, plan, _ = self.prepare()
        a = self.engine(d)
        self.stage(a)
        config_path, config = self.config(a, "bad")
        manifest = preflight.frozen_job(d, plan)
        root = self.root / "scratch"; root.mkdir()
        for bad in ({}, {self.entry["target"]: {"bytes": self.entry["bytes"], "sha256": "0"*64,
                                                 "reference": a.state["private_input_receipts"][self.entry["target"]]["reference"]}}):
            with self.subTest(bad=bad), self.assertRaises(lio.LifecycleError):
                worker.restore_private_inputs({**config, "private_input_receipts": bad}, manifest, root, self.storage())
        facts = {}
        with mock.patch.object(worker.shutil, "disk_usage", return_value=SimpleNamespace(free=1)), \
             self.assertRaisesRegex(lio.LifecycleError, "insufficient runtime disk"):
            worker.restore_private_inputs(config, manifest, root, self.storage(), facts)
        self.assertEqual(facts["runtime_free_bytes"], 1)
        self.assertLess(facts["runtime_headroom_bytes"], 0)
        self.assertFalse(list(root.iterdir()))
        receipt = a.state["private_input_receipts"][self.entry["target"]]["reference"]
        commit = json.loads((self.drive.directory / receipt["id"]).read_text())
        self.drive.corrupt.add(commit["chunks"][0]["id"])
        result, job_root = self.execute(config_path, config, failure=True)
        self.assertEqual(result["terminal_state"], "integrity_failed")
        self.assertEqual(result["failure"]["stage"], "private_inputs")
        self.assertFalse((job_root / self.entry["target"]).exists())
        self.assertFalse((job_root / "outputs/check.json").exists())

    def test_runtime_no_callback_and_legacy_variants(self):
        for named in (False, True):
            for isolated in (False, True):
                for remote in (False, True):
                    opts = dict(named_environments=named, isolated=isolated, remote_inputs=remote)
                    before = runtime_cell_sources(**opts)
                    private = runtime_cell_sources(**opts, private_inputs=True)
                    self.assertEqual([x[0] for x in before], [x[0] for x in private])
                    extraction = dict(private)["extraction"]
                    self.assertIn("COLABCOLA_PRIVATE_INPUT_ACQUIRE", extraction)
                    self.assertNotIn("COLABCOLA_PRIVATE_INPUT_ACQUIRE", dict(before)["extraction"])
        d, plan, _ = self.prepare()
        # The managed notebook still fails closed if opened and run standalone.
        control = self.root / "standalone"; control.mkdir()
        (control / "notebook.ipynb").write_bytes((d / "notebook.ipynb").read_bytes())
        config = {"directory": str(control), "files": plan["files"], "deadline": time.time()+900,
                  "content_root": str(self.root / "standalone-runtime"), "gpu": "NONE", "job_id": plan["job_id"]}
        lio.atomic(control / "worker.json", config)
        result, root = self.execute(control / "worker.json", config, failure=True)
        self.assertEqual(result["terminal_state"], "integrity_failed")
        self.assertFalse((root / "outputs/check.json").exists())

    def test_preflight_plan_mismatch_and_unshared_parent(self):
        d, plan, approval = self.prepare()
        bad = copy.deepcopy(plan)
        bad["private_inputs"][0]["sha256"] = "a"*64
        with self.assertRaisesRegex(lio.LifecycleError, "differ from the frozen job"):
            preflight._manifest_identity(d, bad)
        self.drive.shared_parent = True
        with self.assertRaisesRegex(preflight.PreflightError, "private_inputs failed"):
            self.preflight_run(d, approval)
        self.assertFalse(any(e[0] == "put" for e in self.drive.events))
