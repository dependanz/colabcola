"""Public one-command workflow. All remote/provider interactions are local fakes."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock
from colabcola.cli import main
from colabcola.lifecycle_io import atomic,publish
from colabcola.package_guide import save_answers
from test_lifecycle import MemoryStore,Provider


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.project=self.root/"project";self.project.mkdir()
        self.source=self.project/"work.py"
        self.source.write_text('from pathlib import Path; Path("outputs").mkdir(); Path("outputs/result.txt").write_text(Path("data/input.txt").read_text()); print("work complete",flush=True)')
        (self.root/"input.txt").write_text("real fixture data")
        (self.root/"requirements.txt").write_text("# no external packages\n")
        atomic(self.root/"helper.json",["must-not-run"])
        self.recipe=self.root/"recipe.colaml"
        self.mapping={"name":"workflow-fixture","project":"project","output":"must-not-write-here.colajob",
            "include":["work.py"],"commands":[["python","work.py"]],
            "inputs":[{"source":"input.txt","target":"data/input.txt"}],
            "environment":{"python":">=3.11","requirements":["requirements.txt"]},
            "compute":{"gpu":"NONE"},"outputs":{"include":["outputs/**"],"checkpoints":[]},
            "limits":{"timeout_seconds":60,"max_seconds":900,"max_runtimes":1},
            "execution":{"drive_parent":"fixture","token_command":"helper.json","provider_backend":"legacy","runtime_version":"latest"}}
        save_answers(self.recipe,self.mapping)
        self.workspace=self.root/"new-run"

    def call(self,*options,workspace=None):
        self.out,self.err=io.StringIO(),io.StringIO()
        with contextlib.redirect_stdout(self.out),contextlib.redirect_stderr(self.err):
            return main(["run","--colaml",str(self.recipe),"--run-dir",str(workspace or self.workspace),"--non-interactive",*options])

    def state(self):return json.loads((self.workspace/"recipe.run/state.json").read_text())
    def report(self):return json.loads((self.workspace/"cli-exit.json").read_text())
    def before(self):
        return {p.relative_to(self.root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.root.rglob("*") if p.is_file() and self.workspace not in p.parents}

    def remote(self):
        stack=contextlib.ExitStack();self.addCleanup(stack.close)
        self.provider=Provider();self.provider.assignments=mock.Mock(return_value=[]);self.provider.close=mock.Mock()
        self.store=MemoryStore()
        patches={
            "colabcola.preflight.installation_integrity":dict(files_verified=40),
            "colabcola.lifecycle_provider.dependency_report":dict(provider_source_verified=True),
            "colabcola.lifecycle_provider.provider_for_plan":self.provider,
            "colabcola.lifecycle.get_token":dict(access_token="FAKE-TOKEN",expires_at=time.time()+3600),
            "colabcola.preflight.drive_parent":dict(owned=True),
            "colabcola.preflight.shutil.disk_usage":type("Disk",(),dict(free=100*1024**3))(),
            "colabcola.lifecycle.DriveStore":self.store}
        self.patches={name:stack.enter_context(mock.patch(name,return_value=value)) for name,value in patches.items()}
        stack.enter_context(mock.patch("colabcola.lifecycle.time.sleep"))
        def boot(endpoint,plan,attempt,token):
            self.provider.boots.append(endpoint)
            namespace={"COLABCOLA_CONTENT_ROOT_OVERRIDE":str(self.root/"local-runtime"),
                       "COLABCOLA_REQUESTED_GPU_TYPE":plan["gpu"]}
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                for cell in json.loads((Path(plan["directory"])/"notebook.ipynb").read_text())["cells"]:
                    if cell["cell_type"]=="code":exec(cell["source"],namespace)
            result_root=self.root/"local-runtime/colabcola/results"/plan["job_id"]
            publish(self.store,plan,"final",{name:(result_root/name).read_bytes()
                for name in ("result.json","artifacts.zip","hardware.json")},attempt=attempt["number"])
        self.provider.boot=boot

    def test_prepare_only_uses_original_relative_paths_and_preserves_source_tree(self):
        before=self.before()
        with mock.patch("colabcola.lifecycle.get_token",side_effect=AssertionError("offline")),mock.patch("colabcola.lifecycle_provider.provider_for_plan",side_effect=AssertionError("offline")):
            self.assertEqual(self.call("--prepare-only"),0,self.err.getvalue())
        self.assertEqual(self.before(),before)
        self.assertFalse((self.root/"must-not-write-here.colajob").exists())
        self.assertEqual(self.state()["phase"],"prepared");self.assertEqual(self.state()["attempts"],[])
        self.assertEqual((self.workspace/"recipe.files/data/input.txt").read_text(),"real fixture data")
        self.assertEqual(self.report()["exit_code"],0)
        self.assertIn("Packaging recipe",(self.workspace/"launch.log").read_text())
        self.assertEqual(json.loads((self.workspace/"workflow.json").read_text())["phase"],"ready")

    def test_default_command_packages_preflights_executes_and_recovers_with_fake_provider(self):
        self.remote()
        self.assertEqual(self.call(),0,self.err.getvalue())
        state=self.state();self.assertEqual(state["phase"],"completed")
        self.assertEqual(len(self.provider.allocations),1);self.assertEqual(self.provider.releases,["vm-1"])
        self.assertEqual(state["preflight"]["state"],"passed")
        import zipfile
        with zipfile.ZipFile(Path(state["final"]["local_directory"])/"artifacts.zip") as archive:
            self.assertEqual(archive.read("outputs/result.txt"),b"real fixture data")
        self.assertEqual(self.report()["attempts"],1)
        self.assertIn("Runtime hardware",(self.workspace/"launch.log").read_text())
        self.assertEqual(state["hardware"]["requested_gpu"],"NONE")

    def test_terminal_repeat_does_not_repackage_preflight_or_allocate(self):
        self.remote();self.assertEqual(self.call(),0,self.err.getvalue())
        plan=(self.workspace/"recipe.run/plan.json").read_bytes()
        self.source.write_text("raise AssertionError('changed source must not run')")
        with mock.patch("colabcola.workflow._package",side_effect=AssertionError("no repackage")),mock.patch("colabcola.preflight.run_preflight",side_effect=AssertionError("no new run")):
            self.assertEqual(self.call(),0,self.err.getvalue())
        self.assertEqual((self.workspace/"recipe.run/plan.json").read_bytes(),plan)
        self.assertEqual(len(self.provider.allocations),1)

    def test_prepare_then_repeat_runs_same_plan_once(self):
        self.assertEqual(self.call("--prepare-only"),0,self.err.getvalue())
        plan=(self.workspace/"recipe.run/plan.json").read_bytes()
        self.remote()
        with mock.patch("colabcola.workflow._package",side_effect=AssertionError("already packaged")):
            self.assertEqual(self.call(),0,self.err.getvalue())
        self.assertEqual((self.workspace/"recipe.run/plan.json").read_bytes(),plan)
        self.assertEqual(len(self.provider.allocations),1)

    def test_failed_preflight_retains_prepared_budget_and_automatic_exit_evidence(self):
        self.remote();self.provider.assignments.side_effect=TimeoutError("SECRET")
        self.assertEqual(self.call(),2)
        self.assertEqual(self.state()["phase"],"prepared");self.assertEqual(self.state()["attempts"],[])
        self.assertNotIn("deadline",self.state());self.assertEqual(self.provider.allocations,{})
        self.assertEqual(self.report()["exit_code"],2)
        self.assertNotIn("SECRET",(self.workspace/"launch.log").read_text())

    def test_preflight_only_creates_no_allocation_or_staging(self):
        self.remote();self.assertEqual(self.call("--preflight-only"),0,self.err.getvalue())
        self.assertEqual(self.provider.allocations,{})
        self.assertEqual(self.store.objects,{}) # prepare may construct a locally validated store; no put/upload
        self.assertEqual(self.state()["phase"],"prepared")

    def test_unrelated_existing_folder_is_not_modified(self):
        self.workspace.mkdir();(self.workspace/"old.txt").write_text("preserve")
        self.assertEqual(self.call("--prepare-only"),2)
        self.assertEqual([p.name for p in self.workspace.iterdir()],["old.txt"])
        self.assertEqual((self.workspace/"old.txt").read_text(),"preserve")

    def test_old_managed_folder_is_not_modified(self):
        self.workspace.mkdir();(self.workspace/"old.run").mkdir();atomic(self.workspace/"old.run/plan.json",{"legacy":"untouched"})
        self.assertEqual(self.call("--prepare-only"),2)
        self.assertFalse((self.workspace/"launch.log").exists());self.assertFalse((self.workspace/"workflow.json").exists())

    def test_nested_workspace_cannot_write_inside_an_existing_managed_job(self):
        protected=self.root/"protected";protected.mkdir();(protected/"old.run").mkdir();atomic(protected/"old.run/plan.json",{})
        self.assertEqual(self.call("--prepare-only",workspace=protected/"new"),2)
        self.assertFalse((protected/"new").exists())

    def test_repeat_prepare_retains_plan_budget_and_snapshot(self):
        self.assertEqual(self.call("--prepare-only"),0,self.err.getvalue())
        plan=(self.workspace/"recipe.run/plan.json").read_bytes()
        job=(self.workspace/"recipe.colajob").read_bytes()
        with mock.patch("colabcola.workflow._package",side_effect=AssertionError("no repackage")):
            self.assertEqual(self.call("--prepare-only"),0,self.err.getvalue())
        self.assertEqual((self.workspace/"recipe.run/plan.json").read_bytes(),plan)
        self.assertEqual((self.workspace/"recipe.colajob").read_bytes(),job)

    def test_recipe_edits_do_not_silently_change_a_prepared_run(self):
        self.assertEqual(self.call("--prepare-only"),0,self.err.getvalue())
        self.recipe.write_text("this is no longer valid TOML")
        self.assertEqual(self.call("--prepare-only"),0,self.err.getvalue())
        self.assertIn("Recipe changed",self.err.getvalue())

    def test_missing_recipe_can_still_reconnect_to_frozen_workspace(self):
        self.assertEqual(self.call("--prepare-only"),0,self.err.getvalue())
        self.recipe.unlink()
        self.assertEqual(self.call("--prepare-only"),0,self.err.getvalue())

    def test_expected_job_mismatch_stops_before_preparation_or_credentials(self):
        self.assertEqual(self.call("--prepare-only","--expect-job-id","a"*64),2)
        self.assertFalse((self.workspace/"recipe.run").exists())
        self.assertEqual(self.report()["exit_code"],2)
        self.assertEqual(self.call("--prepare-only"),2) # retained guard is not silently dropped

    def test_missing_login_keeps_frozen_package_for_retry_after_setup(self):
        import os
        data=dict(self.mapping);del data["execution"];self.recipe.unlink();save_answers(self.recipe,data)
        with mock.patch.dict(os.environ, {"COLABCOLA_CONFIG_HOME":str(self.root/"empty-account")}):
            self.assertEqual(self.call("--prepare-only"),2)
            self.assertIn("colabcola login",self.err.getvalue())
            record=json.loads((self.workspace/"workflow.json").read_text())
            self.assertEqual(record["phase"],"ready")
            frozen=(self.workspace/"recipe.colajob").read_bytes()
            self.assertEqual(self.call("--prepare-only"),2)
            self.assertIn("colabcola login",self.err.getvalue())
            self.assertEqual((self.workspace/"recipe.colajob").read_bytes(),frozen)

    def test_changed_frozen_job_is_rejected(self):
        self.assertEqual(self.call("--prepare-only"),0,self.err.getvalue())
        (self.workspace/"recipe.files/work.py").write_text("print('tampered')")
        self.assertEqual(self.call("--prepare-only"),2)
        self.assertEqual(self.state()["attempts"],[])

    def test_logs_capture_fragments_and_redact_credentials(self):
        def fake(*a):
            import sys
            sys.stdout.write("Bearer ");sys.stdout.write("SECRET\n")
            print("access_token=SECRET",file=sys.stderr)
            return 3
        with mock.patch("colabcola.managed.run_managed",side_effect=fake):
            self.assertEqual(self.call(),3,self.err.getvalue())
        log=(self.workspace/"launch.log").read_text()
        self.assertIn("[redacted]",log);self.assertNotIn("SECRET",log)
        self.assertRegex(log,r"\[\d{4}-\d\d-\d\dT.*Z stdout\]")
        self.assertEqual(self.report()["exit_code"],3)

    def test_interruption_gets_exit_report_without_allocating(self):
        with mock.patch("colabcola.workflow._package",side_effect=KeyboardInterrupt()):
            self.assertEqual(self.call(),130)
        self.assertEqual(self.report()["exit_code"],130)

    def test_other_recipe_cannot_reuse_workspace(self):
        self.assertEqual(self.call("--prepare-only"),0,self.err.getvalue())
        other=self.root/"other.colaml";other.write_bytes(self.recipe.read_bytes());self.recipe=other
        self.assertEqual(self.call("--prepare-only"),2)

    def test_retry_after_failed_admission_reuses_prepared_job_and_single_budget(self):
        self.remote();self.provider.assignments.side_effect=TimeoutError("offline fixture")
        self.assertEqual(self.call(),2)
        plan=(self.workspace/"recipe.run/plan.json").read_bytes()
        self.provider.assignments.side_effect=None
        with mock.patch("colabcola.workflow._package",side_effect=AssertionError("no repackage")):
            self.assertEqual(self.call(),0,self.err.getvalue())
        self.assertEqual((self.workspace/"recipe.run/plan.json").read_bytes(),plan)
        self.assertEqual(len(self.provider.allocations),1)

    def test_concurrent_wrapper_cannot_overwrite_log_or_exit_report(self):
        from colabcola.lifecycle import run_lock
        self.assertEqual(self.call("--prepare-only"),0,self.err.getvalue())
        before=(self.workspace/"cli-exit.json").read_bytes()
        log=(self.workspace/"launch.log").read_bytes()
        with run_lock(self.workspace):
            self.assertEqual(self.call("--prepare-only"),2)
        self.assertEqual((self.workspace/"cli-exit.json").read_bytes(),before)
        self.assertEqual((self.workspace/"launch.log").read_bytes(),log)
