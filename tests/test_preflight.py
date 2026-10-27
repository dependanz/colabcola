"""No allocation or live Google calls: shared CLI admission and integrity acceptance."""
import base64
import contextlib
import hashlib
import io
import json
from importlib.metadata import PackagePath, FileHash
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest import mock
from colabcola import cli
from colabcola.lifecycle import supervise
from colabcola.lifecycle_io import atomic, digest, encoded, LifecycleError
from colabcola import preflight
from colabcola.gpu_check import _colabcola_gpu_check
from test_notebook import notebook_job


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.script=self.root/"work.py";self.script.write_text("print('ok')")
        value=notebook_job(self.script);value["runtime"]["version"]="latest"
        value["managed_recovery"]={"checkpoint_pattern":"outputs/state*.json","log_patterns":[]}
        self.job=self.root/"work.colajob";atomic(self.job,value)
        atomic(self.root/"helper.json",["never-actually-run"])
        self.config={"drive_parent":"owned-folder","token_command":"helper.json","gpu":"NONE","max_attempts":1,"max_seconds":18000}
        atomic(self.job.with_suffix(".run.json"),self.config)
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["--prepare-only",str(self.job)]),0)
        self.run=self.job.with_suffix(".run")
        self.plan=json.loads((self.run/"plan.json").read_text());self.approval=digest(encoded(self.plan))
        self.provider=SimpleNamespace(assignments=mock.Mock(return_value=[]),close=mock.Mock(),allocate=mock.Mock(),release=mock.Mock())
        self.patches={}
        for name,value in (
            ("colabcola.preflight.installation_integrity",{"files_verified":39}),
            ("colabcola.lifecycle_provider.dependency_report",{"provider_source_verified":True}),
            ("colabcola.lifecycle_provider.provider_for_plan",self.provider),
            ("colabcola.lifecycle.get_token",{"access_token":"SECRET-TOKEN","expires_at":time.time()+3600}),
            ("colabcola.preflight.drive_parent",{"owned":True}),
            ("colabcola.preflight.shutil.disk_usage",SimpleNamespace(free=100*1024**3))):
            patch=mock.patch(name,return_value=value);self.patches[name]=patch.start();self.addCleanup(patch.stop)

    def run_cli(self,*options):
        self.stdout=io.StringIO();self.stderr=io.StringIO()
        with contextlib.redirect_stdout(self.stdout),contextlib.redirect_stderr(self.stderr):
            return cli.main([*options,str(self.job)])

    def state(self):return json.loads((self.run/"state.json").read_text())
    def report(self):
        state=self.state();return json.loads(Path(state["preflight"]["path"]).read_text())

    def assert_refused(self,code):
        with mock.patch("colabcola.lifecycle.Supervisor") as engine,mock.patch("colabcola.lifecycle.DriveStore") as store:
            self.assertEqual(self.run_cli("--preflight-only"),2)
            engine.assert_not_called();store.assert_not_called()
        state=self.state();self.assertEqual(state["phase"],"prepared");self.assertEqual(state["attempts"],[])
        self.assertNotIn("deadline",state)
        report=self.report();self.assertEqual(report["failed_check"],code)
        self.assertNotIn("SECRET",json.dumps(report)+self.stdout.getvalue()+self.stderr.getvalue())
        self.provider.allocate.assert_not_called();self.provider.release.assert_not_called()
        return report

    def test_standalone_success_is_repeatable_and_never_stages(self):
        before=(self.run/"plan.json").read_bytes()
        with mock.patch("colabcola.lifecycle.Supervisor") as engine,mock.patch("colabcola.lifecycle.DriveStore") as store:
            self.assertEqual(self.run_cli("--preflight-only","--approve",self.approval),0)
            self.assertEqual(self.run_cli("--preflight-only"),0)
            engine.assert_not_called();store.assert_not_called()
        self.assertEqual((self.run/"plan.json").read_bytes(),before)
        self.assertEqual(self.state()["phase"],"prepared");self.assertEqual(self.state()["attempts"],[])
        self.assertEqual(len(list((self.run/"preflight").glob("*.json"))),2)
        self.assertEqual(self.report()["state"],"passed")
        self.assertNotIn("SECRET",self.stdout.getvalue()+self.stderr.getvalue()+json.dumps(self.report()))
        self.assertEqual(self.provider.close.call_count,2)

    def test_bare_launch_automatically_checks_before_engine_exists(self):
        with mock.patch("colabcola.lifecycle.Supervisor") as engine,mock.patch("colabcola.lifecycle.RenewalPump",create=True):
            engine.return_value.state={"phase":"completed","events":[],"attempts":[]}
            self.assertEqual(self.run_cli(),0)
            engine.assert_called_once()
            self.assertEqual(self.report()["state"],"passed")
        self.provider.assignments.assert_called_once()
        self.provider.close.assert_called_once()

    def test_failure_on_normal_launch_also_never_constructs_engine(self):
        self.provider.assignments.return_value=[SimpleNamespace(endpoint="existing-vm")]
        with mock.patch("colabcola.lifecycle.Supervisor") as engine:
            self.assertEqual(self.run_cli(),2);engine.assert_not_called()
        self.assertEqual(self.state()["phase"],"prepared")

    def test_controller_tamper_rejected_before_credentials(self):
        self.patches["colabcola.preflight.installation_integrity"].side_effect=RuntimeError("SECRET")
        self.assert_refused("controller_integrity")
        self.patches["colabcola.lifecycle_provider.provider_for_plan"].assert_not_called()

    def test_provider_dependency_tamper_rejected(self):
        self.patches["colabcola.lifecycle_provider.dependency_report"].side_effect=RuntimeError("SECRET")
        self.assert_refused("provider_dependencies")

    def test_wrong_approval_has_durable_failure(self):
        self.assertEqual(self.run_cli("--preflight-only","--approve","bad"),2)
        self.assertEqual(self.report()["failed_check"],"approval")
        self.patches["colabcola.lifecycle_provider.provider_for_plan"].assert_not_called()

    def test_prepared_file_tamper_rejected(self):
        (self.run/"lifecycle_worker.py").write_text("changed")
        self.assert_refused("prepared_files")

    def test_rehashed_payload_tamper_still_fails_job_identity(self):
        document=json.loads((self.run/"notebook.ipynb").read_text())
        for cell in document["cells"]:
            if cell.get("metadata",{}).get("colabcola",{}).get("group")=="payload":
                cell["source"]='COLABCOLA_PAYLOAD_CHUNKS.append("YWR2ZXJzYXJ5")';break
        atomic(self.run/"notebook.ipynb",document)
        self.plan["files"]["notebook.ipynb"]=digest((self.run/"notebook.ipynb").read_bytes());atomic(self.run/"plan.json",self.plan)
        self.approval=digest(encoded(self.plan));self.assert_refused("frozen_job")

    def test_original_script_tamper_is_consumer_binding_failure(self):
        self.script.write_text("print('changed')")
        self.assert_refused("consumer_binding")

    def test_execution_config_tamper_is_consumer_binding_failure(self):
        atomic(self.job.with_suffix(".run.json"),{**self.config,"max_seconds":19000})
        self.assert_refused("consumer_binding")

    def test_disk_failure_occurs_before_online_checks(self):
        self.patches["colabcola.preflight.shutil.disk_usage"].return_value=SimpleNamespace(free=1)
        report=self.assert_refused("recovery_disk");self.assertFalse(report["network_accessed"])

    def test_failed_sign_in_redacts_exception(self):
        self.patches["colabcola.lifecycle_provider.provider_for_plan"].side_effect=RuntimeError("SECRET")
        self.assert_refused("colab_sign_in")

    def test_failed_listing_closes_client(self):
        self.provider.assignments.side_effect=RuntimeError("SECRET")
        self.assert_refused("active_assignments");self.provider.close.assert_called_once()

    def test_existing_assignments_need_exact_override_and_are_not_cancelled(self):
        self.provider.assignments.return_value=[SimpleNamespace(endpoint="other-1"),SimpleNamespace(endpoint="other-2")]
        report=self.assert_refused("active_assignments")
        self.assertEqual(report["assignments"]["observed"],["other-1","other-2"])
        self.assertEqual(self.run_cli("--preflight-only","--allow-active-assignment","other-1","--allow-active-assignment","other-2"),0)
        self.provider.release.assert_not_called()
        self.assertEqual(self.run_cli("--preflight-only","--allow-active-assignment","stale"),2)

    def test_short_missing_invalid_tokens_refused_without_secret(self):
        for value in ({"access_token":"SECRET","expires_at":time.time()+120},
                      {"access_token":"SECRET","expires_at":float('nan')},
                      {"access_token":"","expires_at":time.time()+3600}):
            self.patches["colabcola.lifecycle.get_token"].return_value=value
            self.assert_refused("drive_access")

    def test_drive_failure_closes_provider_and_keeps_prepared(self):
        self.patches["colabcola.preflight.drive_parent"].side_effect=RuntimeError("SECRET")
        self.assert_refused("drive_parent");self.provider.close.assert_called_once()

    def test_stale_success_cannot_bypass_next_launch_checks(self):
        self.assertEqual(self.run_cli("--preflight-only"),0)
        self.provider.assignments.side_effect=TimeoutError("SECRET")
        self.assert_refused("active_assignments")
        self.assertEqual(len(list((self.run/"preflight").glob("*.json"))),2)

    def test_nonprepared_is_not_reset(self):
        atomic(self.run/"state.json",{"phase":"monitoring","attempts":[{"number":1}],"events":[]})
        self.assertEqual(self.run_cli("--preflight-only"),2)
        self.assertEqual(self.state()["phase"],"monitoring");self.assertEqual(len(self.state()["attempts"]),1)

    def test_interrupt_persists_report_without_cancel_request(self):
        self.provider.assignments.side_effect=KeyboardInterrupt()
        self.assertEqual(self.run_cli("--preflight-only"),2)
        self.assertEqual(self.report()["state"],"interrupted")
        self.assertEqual(self.state()["phase"],"prepared");self.assertFalse((self.run/"cancel.request").exists())

    def test_cleanup_and_reconnect_bypass_fresh_launch_gate(self):
        atomic(self.run/"state.json",{"phase":"cleanup","attempts":[{"number":1}],"events":[]})
        with mock.patch("colabcola.preflight.run_preflight",side_effect=AssertionError("must not block cleanup")),mock.patch("colabcola.lifecycle.Supervisor") as engine:
            engine.return_value.state={"phase":"cancelled","events":[]}
            state=supervise(self.run,self.approval)
            self.assertEqual(state["phase"],"cancelled")
        self.provider.assignments.assert_not_called()

    def test_legacy_plan_is_not_rewritten(self):
        del self.plan["hardware_policy"];atomic(self.run/"plan.json",self.plan)
        before=(self.run/"plan.json").read_bytes()
        self.assertEqual(self.run_cli("--preflight-only"),0)
        self.assertEqual((self.run/"plan.json").read_bytes(),before)
        self.assertEqual(self.report()["hardware_policy"],"legacy_model_check_unavailable")

    def test_standalone_client_close_failure_is_a_failed_report(self):
        self.provider.close.side_effect=RuntimeError("SECRET")
        self.assert_refused("client_cleanup")

    def test_plan_parse_failure_is_journaled(self):
        (self.run/"plan.json").write_text("{")
        with self.assertRaises(preflight.PreflightError):
            preflight.run_preflight(self.run,self.approval)
        self.assertEqual(self.report()["failed_check"],"plan_read")

    def test_five_hour_virtual_renewal(self):
        from colabcola.lifecycle_auth import TokenLease
        now=[0];calls=[]
        def loader():
            calls.append(now[0]);return {"access_token":"SECRET-"+str(now[0]),"expires_at":now[0]+3600,"refresh_token":"SECRET"}
        lease=TokenLease(loader,clock=lambda:now[0])
        for tick in range(0,18001,60):
            now[0]=tick;value=lease()
            self.assertGreater(value["expires_at"],tick+600)
            self.assertEqual(set(value),{"access_token","expires_at"})
        self.assertEqual(calls,[0,3000,6000,9000,12000,15000,18000])


class IntegrityTests(unittest.TestCase):
    def test_record_verifies_missing_modified_extra_and_editable_installs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);package=root/"colabcola";package.mkdir()
            (package/"__init__.py").write_bytes(b"# fixture")
            entry=PackagePath("colabcola/__init__.py")
            entry.hash=FileHash("sha256="+base64.urlsafe_b64encode(hashlib.sha256(b"# fixture").digest()).rstrip(b"=").decode())
            dist=SimpleNamespace(files=[entry],locate_file=lambda p:root/str(p))
            self.assertEqual(preflight.installation_integrity(package,dist)["files_verified"],1)
            (package/"extra.py").write_bytes(b"# extra")
            with self.assertRaises(LifecycleError):preflight.installation_integrity(package,dist)
            (package/"extra.py").unlink();(package/"__init__.py").write_bytes(b"# changed")
            with self.assertRaises(LifecycleError):preflight.installation_integrity(package,dist)
            (package/"__init__.py").unlink()
            with self.assertRaises(LifecycleError):preflight.installation_integrity(package,dist)
            with self.assertRaises(LifecycleError):preflight.installation_integrity(package,SimpleNamespace(files=[]))


class FolderTests(unittest.TestCase):
    def test_metadata_get_is_bounded_read_only_and_requires_exact_owned_folder(self):
        good={"id":"parent","mimeType":"application/vnd.google-apps.folder","ownedByMe":True,"trashed":False,"capabilities":{"canAddChildren":True}}
        requests=[]
        def fetch(req,timeout):
            requests.append(req);self.assertEqual(timeout,30)
            return io.BytesIO(encoded(good))
        self.assertTrue(preflight.drive_parent("parent",{"access_token":"SECRET"},fetch)["owned"])
        self.assertEqual(requests[0].method,"GET")
        for key,bad in (("ownedByMe",False),("trashed",True),("mimeType","file"),("id","other"),("capabilities",{})):
            value={**good,key:bad}
            with self.subTest(key=key),self.assertRaises(LifecycleError):
                preflight.drive_parent("parent",{"access_token":"SECRET"},lambda *a,**k:io.BytesIO(encoded(value)))


class GpuTests(unittest.TestCase):
    resources={"accelerator":"gpu","gpu_count":1,"cuda_required":True,"minimum_gpu_memory_bytes":60000000000}
    def check(self,request,name,memory=80*1024**3):
        return _colabcola_gpu_check(request,self.resources,[{"index":0,"name":name,"memory_bytes":memory}],True)
    def test_aliases_and_mismatches(self):
        for name in ("NVIDIA A100-SXM4-80GB","NVIDIA A100 80GB PCIe","A100-SXM4-80GB"):
            self.assertEqual(self.check("A100",name)["state"],"passed")
        for name in ("NVIDIA H100 80GB HBM3","NVIDIA A1000","unknown"):
            self.assertEqual(self.check("A100",name)["state"],"failed")
        self.assertEqual(self.check("A100","A100-SXM4-40GB",40*1024**3)["state"],"failed")
    def test_no_cross_device_memory_model_combination(self):
        value=_colabcola_gpu_check("A100",self.resources,[{"index":0,"name":"A100","memory_bytes":40*1024**3},{"index":1,"name":"H100","memory_bytes":80*1024**3}],True)
        self.assertEqual(value["state"],"failed")
    def test_other_supported_families_and_no_fuzzy_prefixes(self):
        resources={**self.resources,"minimum_gpu_memory_bytes":1}
        for selector,name in (("T4","Tesla T4"),("L4","NVIDIA L4"),("H100","NVIDIA H100 80GB HBM3"),("G4","NVIDIA RTX PRO 6000 Blackwell Server Edition")):
            value=_colabcola_gpu_check(selector,resources,[{"name":name,"memory_bytes":1}],True)
            self.assertEqual(value["state"],"passed")
        for name in ("NVIDIA T40","NVIDIA L40","NVIDIA RTX PRO 6000 Blackwell Workstation Edition"):
            self.assertEqual(_colabcola_gpu_check("T4",resources,[{"name":name,"memory_bytes":1}],True)["state"],"failed")
    def test_missing_cuda_count_and_cpu_conflict(self):
        self.assertEqual(_colabcola_gpu_check("A100",self.resources,[],False)["state"],"failed")
        self.assertEqual(_colabcola_gpu_check("A100",{**self.resources,"accelerator":"cpu"},[],False)["state"],"failed")


class CompiledHardwareTests(unittest.TestCase):
    def test_compiled_preflight_records_match_and_rejects_wrong_gpu_before_extraction(self):
        import sys
        from colabcola import JobSpec, build_bundle, bind_job_arguments
        from colabcola.notebook import build_notebook
        for name,memory,expected in (("NVIDIA A100-SXM4-80GB",80*1024**3,"passed"),
                                     ("NVIDIA H100 80GB HBM3",80*1024**3,"failed"),
                                     ("NVIDIA A100-SXM4-40GB",40*1024**3,"failed")):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);source=root/"train.py";source.write_text("raise AssertionError('must not execute')")
                mapping=notebook_job(source,accelerator="gpu")
                mapping["resources"]["minimum_gpu_memory_bytes"]=60000000000
                mapping["python"]["runtime_requirements"]=[]
                bundle=build_bundle(bind_job_arguments(JobSpec.from_mapping(mapping,base_dir=root)))
                document=json.loads(build_notebook(bundle).notebook_bytes)
                namespace={"COLABCOLA_CONTENT_ROOT_OVERRIDE":str(root/"content"),"COLABCOLA_REQUESTED_GPU_TYPE":"A100"}
                torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,device_count=lambda:1,
                    get_device_properties=lambda index:SimpleNamespace(name=name,total_memory=memory)))
                with mock.patch.dict(sys.modules,{"torch":torch}):
                    for cell in document["cells"]:
                        if cell.get("metadata",{}).get("colabcola",{}).get("group") in ("manifest","preflight"):
                            exec(cell["source"],namespace)
                hardware=json.loads((root/"content/colabcola/results"/bundle.job_id/"hardware.json").read_text())
                self.assertEqual(hardware["state"],expected)
                self.assertEqual(hardware["gpus"][0]["name"],name)
                self.assertFalse((root/"content/colabcola/jobs"/bundle.job_id).exists())
                self.assertEqual(namespace["_COLABCOLA_RUNTIME_FACTS"]["resource_check"],hardware)
                self.assertEqual(namespace["_COLABCOLA_TERMINAL_STATE"],None if expected=="passed" else "preflight_failed")

    def test_gpu_failure_final_is_recovered_reported_and_releases_exact_owned_runtime(self):
        from test_lifecycle import LifecycleTests
        from colabcola.lifecycle_io import publish
        from colabcola.managed import Progress
        fixture=LifecycleTests();fixture.setUp()
        try:
            engine=fixture.running()
            report={"job_id":engine.plan["job_id"],"requested_gpu":"A100","gpus":[{"name":"H100","memory_bytes":80*1024**3}],"state":"failed"}
            result={"job_id":engine.plan["job_id"],"terminal_state":"preflight_failed","outputs":[]}
            import zipfile
            buffer=io.BytesIO()
            with zipfile.ZipFile(buffer,"w") as archive:archive.writestr("result.json",encoded(result))
            publish(fixture.store,fixture.identity,"final",{"result.json":encoded(result),"artifacts.zip":buffer.getvalue(),"hardware.json":encoded(report)},attempt=1)
            stream=io.StringIO();engine.observer=Progress(fixture.root,stream)
            engine.advance()
            self.assertEqual(engine.state["phase"],"failed")
            self.assertEqual(engine.state["hardware"],report)
            self.assertEqual(fixture.provider.releases,["vm-1"])
            self.assertIn("received H100",stream.getvalue())
        finally:fixture.doCleanups()
