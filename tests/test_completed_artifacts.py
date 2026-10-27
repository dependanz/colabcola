"""Offline runtime-loss,late-log and completed-output recovery regressions."""
import contextlib
import io
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

from colabcola import JobSpec, bind_job_arguments, build_bundle
from colabcola.compiler import compile_invocation
from colabcola.invocation import load_job_invocation
from colabcola.lifecycle import recover_terminal_final, recover_progress
from colabcola.lifecycle_io import LifecycleError, atomic, digest, publish, recover, snapshots
from colabcola.lifecycle_provider import GoogleProvider
from colabcola.lifecycle_worker import CompletedArtifacts
from colabcola.managed import Progress
from colabcola.package_guide import save_answers
from colabcola.cli import main
from test_notebook import notebook_job
import test_lifecycle as fixtures

GROUP = {'marker':'outputs/clips/*/manifest.json','files':['video.mp4'], 'ready_when':{'status':'completed'}}

class Store(fixtures.MemoryStore):
    def list(self, kind):
        values = super().list(kind)
        for item in values:
            item['createdTime'] = f'{list(self.objects).index(item["id"]):010d}'
        return values

class CompletedArtifactsTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.job=self.root/'job';self.job.mkdir()
        self.store=Store();self.identity={'run_id':'a'*32,'job_id':'b'*64}
        self.config={**self.identity,'attempt':1,'artifact_groups':[GROUP],
                     'artifact_output_rules':[{'pattern':'outputs/**','maximum_bytes':10000}]}
        self.worker=CompletedArtifacts(self.config,self.job,self.root/'control')
    def clip(self, name='one', status='completed', body=b'complete-video'):
        root=self.job/'outputs/clips'/name;root.mkdir(parents=True,exist_ok=True)
        (root/'video.mp4').write_bytes(body);atomic(root/'manifest.json',{'status':status})
        return root
    def test_incomplete_and_failed_markers_never_publish(self):
        root=self.clip(status='running');self.assertIsNone(self.worker.publish(self.store))
        atomic(root/'manifest.json',{'status':'failed'});self.assertIsNone(self.worker.publish(self.store))
        (root/'manifest.json').write_text('{');self.assertIsNone(self.worker.publish(self.store))
        self.assertEqual(self.store.list('artifacts'),[])
    def test_complete_group_cold_recovers_video_and_marker_before_final(self):
        self.clip();snap=self.worker.publish(self.store)
        self.assertEqual(snap['commit']['kind'],'artifacts');self.assertEqual(snap['commit']['schema'],2)
        target=recover(self.store,snap,self.root/'cold')
        self.assertEqual((target/'outputs/clips/one/video.mp4').read_bytes(),b'complete-video')
        self.assertEqual(len(snap['commit']['members']),2);self.assertEqual(self.store.list('final'),[])
    def test_partial_upload_has_no_visible_commit_and_retries_staged_bytes(self):
        self.clip();self.store.fail_kind='member'
        with self.assertRaises(LifecycleError):self.worker.publish(self.store)
        self.assertEqual(self.store.list('artifacts'),[])
        self.store.fail_kind=None;snap=self.worker.publish(self.store)
        self.assertEqual(len(snap['commit']['members']),2)
    def test_unchanged_groups_are_not_reuploaded_and_new_groups_are_cumulative(self):
        self.clip();first=self.worker.publish(self.store)
        with mock.patch.object(self.store,'put',wraps=self.store.put) as put:
            self.assertIsNone(self.worker.publish(self.store));put.assert_not_called()
            self.clip('two',body=b'second');second=self.worker.publish(self.store)
        self.assertEqual(second['commit']['step'],2);self.assertEqual(len(second['commit']['members']),4)
        self.assertEqual(second['commit']['members']['outputs/clips/one/video.mp4'],first['commit']['members']['outputs/clips/one/video.mp4'])
    def test_mutating_original_file_cannot_change_the_protected_snapshot(self):
        root=self.clip();snap=self.worker.publish(self.store);(root/'video.mp4').write_bytes(b'changed')
        cold=recover(self.store,snap,self.root/'cold');self.assertEqual((cold/'outputs/clips/one/video.mp4').read_bytes(),b'complete-video')
    def test_changed_completion_marker_is_rejected(self):
        root=self.clip();self.worker.publish(self.store);atomic(root/'manifest.json',{'status':'failed'})
        with self.assertRaisesRegex(LifecycleError,'marker changed'):self.worker.publish(self.store)
    def test_declared_output_bound_precedes_any_transfer(self):
        self.clip();self.config['artifact_output_rules'][0]['maximum_bytes']=1
        with self.assertRaisesRegex(LifecycleError,'bound'):self.worker.publish(self.store)
        self.assertFalse(self.store.objects)
    def test_missing_file_and_output_escape_fail_without_commit(self):
        root=self.clip();(root/'video.mp4').unlink()
        with self.assertRaisesRegex(LifecycleError,'missing'):self.worker.publish(self.store)
        self.assertEqual(self.store.list('artifacts'),[])
        self.clip();self.config['artifact_output_rules'][0]['pattern']='outputs/other/**'
        with self.assertRaisesRegex(LifecycleError,'outside declared'):self.worker.publish(self.store)
    @unittest.skipUnless(os.name=='posix','symlink fixture requires POSIX')
    def test_symlink_member_is_rejected(self):
        root=self.clip();(root/'video.mp4').unlink();(self.root/'secret').write_bytes(b'secret')
        (root/'video.mp4').symlink_to(self.root/'secret')
        with self.assertRaisesRegex(LifecycleError,'symlink'):self.worker.publish(self.store)

class LossAndLateRecoveryTests(unittest.TestCase):
    setUp=fixtures.LifecycleTests.setUp
    engine=fixtures.LifecycleTests.engine
    running=fixtures.LifecycleTests.running
    reload=fixtures.LifecycleTests.reload
    final_snapshot=fixtures.LifecycleTests.final_snapshot
    def long_running(self):
        self.store=Store();engine=self.running();engine.plan['loss_grace_seconds']=90
        engine.state['deadline']=1500;atomic(self.root/'plan.json',engine.plan);engine.save();return engine
    def test_two_six_second_list_misses_do_not_confirm_loss(self):
        engine=self.long_running();self.provider.active.clear();engine.advance();self.now+=6;engine.advance()
        self.assertEqual(engine.state['phase'],'monitoring');self.assertFalse(self.provider.releases)
        self.now+=85;engine.advance();self.assertEqual(engine.state['phase'],'recovering')
        self.assertEqual(engine.state['last_failure']['cause'],'runtime_lost')
    def test_direct_assignment_presence_overrules_missing_list(self):
        engine=self.long_running()
        with mock.patch.object(self.provider,'lookup',return_value=None):
            engine.advance();self.now+=100;engine.advance()
        self.assertEqual(engine.state['phase'],'monitoring');self.assertFalse(self.provider.releases)
        self.assertEqual(engine.state['attempts'][0]['missing_observations'],0)
    def test_direct_presence_with_stale_worker_ref_still_recovers_new_catalogue_logs(self):
        engine=self.long_running()
        old=publish(self.store,self.identity,'logs',{'runtime/stdout.log':b'old\n'},attempt=1)
        engine.state['worker']={'attempt':1,'logs_commit':{k:old[k] for k in ('id','bytes','sha256')}}
        engine.advance()
        publish(self.store,self.identity,'logs',{'runtime/stdout.log':b'new while list absent\n'},attempt=1)
        with mock.patch.object(self.provider,'lookup',return_value=None):engine.advance()
        self.assertEqual(engine.state['phase'],'monitoring');self.assertFalse(self.provider.releases)
        self.assertIn('new while list absent',(Path(engine.state['logs']['local_directory'])/'runtime/stdout.log').read_text())
    def test_late_drive_progress_extends_quiet_grace_and_is_recovered(self):
        engine=self.long_running();self.provider.active.clear();engine.advance();self.now+=90
        publish(self.store,self.identity,'logs',{'runtime/stdout.log':b'late completed clip\n'},attempt=1)
        engine.advance();self.assertEqual(engine.state['phase'],'monitoring')
        self.assertIn('late completed clip',(Path(engine.state['logs']['local_directory'])/'runtime/stdout.log').read_text())
        self.now+=91;engine.advance();self.assertEqual(engine.state['phase'],'recovering')
    def test_direct_lookup_timeout_never_releases_or_allocates_again(self):
        engine=self.long_running();self.provider.active.clear()
        with mock.patch.object(self.provider,'reconcile',side_effect=TimeoutError('private-token')):
            engine.advance();self.now+=100;engine.advance()
        self.assertEqual(engine.state['phase'],'monitoring');self.assertFalse(self.provider.releases)
        self.assertEqual(len(self.provider.allocations),1);self.assertNotIn('private-token',json.dumps(engine.state))
    def test_release_failure_keeps_owned_attempt_unreleased(self):
        engine=self.long_running();self.provider.active.clear();engine.advance();self.now+=100
        with mock.patch.object(self.provider,'release',side_effect=TimeoutError('private')):engine.advance()
        self.assertEqual(engine.state['phase'],'monitoring');self.assertFalse(engine.state['attempts'][0].get('released'))
    def test_terminal_catalogue_finds_late_logs_and_artifacts_without_provider(self):
        engine=self.long_running();engine.plan.update(parent='fixture',token_command=['must-not-run'],
            artifact_groups=[GROUP],artifact_output_rules=[{'pattern':'outputs/**','maximum_bytes':10000}]);atomic(self.root/'plan.json',engine.plan)
        old=publish(self.store,self.identity,'logs',{'runtime/stdout.log':b'old\n'},attempt=1)
        engine.state['worker']={'attempt':1,'logs_commit':{k:old[k] for k in ('id','sha256','bytes')}}
        engine.finish('failed','runtime_lost_attempt_budget_exhausted');before=dict(engine.state['attempts'][0]);deadline=engine.state['deadline']
        publish(self.store,self.identity,'logs',{'runtime/stdout.log':b'completed clip access_token=private\nlate final tail'},attempt=1)
        job=self.root/'job';clip=job/'outputs/clips/one';clip.mkdir(parents=True);(clip/'video.mp4').write_bytes(b'saved');atomic(clip/'manifest.json',{'status':'completed'})
        CompletedArtifacts({**engine.plan,'attempt':1},job,self.root/'control').publish(self.store)
        output=io.StringIO();progress=Progress(self.root,output)
        with mock.patch('colabcola.lifecycle_provider.provider_for_plan',side_effect=AssertionError('no provider')):
            recovered=recover_terminal_final(self.root,store=self.store,observer=progress)
            events=len(recovered['events']);again=recover_terminal_final(self.root,store=self.store,observer=progress)
        self.assertEqual(again,recovered);self.assertEqual(len(again['events']),events)
        self.assertEqual(recovered['phase'],'failed');self.assertEqual(recovered['reason'],'runtime_lost_attempt_budget_exhausted')
        self.assertEqual(recovered['attempts'][0],before);self.assertEqual(recovered['deadline'],deadline)
        self.assertIn('late final tail',output.getvalue());self.assertNotIn('=private',output.getvalue())
        self.assertEqual((Path(recovered['artifacts']['local_directory'])/'outputs/clips/one/video.mp4').read_bytes(),b'saved')
    def test_unknown_late_final_is_recovered_without_promoting_failure_to_success(self):
        engine=self.long_running();engine.plan.update(parent='fixture',token_command=['must-not-run']);atomic(self.root/'plan.json',engine.plan)
        engine.finish('failed','runtime_lost');self.final_snapshot(1,0)
        recovered=recover_terminal_final(self.root,store=self.store)
        self.assertEqual(recovered['phase'],'failed');self.assertIn('final',recovered)
    def test_corrupt_late_log_is_rejected(self):
        engine=self.long_running();engine.plan.update(parent='fixture',token_command=['must-not-run']);atomic(self.root/'plan.json',engine.plan);engine.finish('failed','lost')
        snap=publish(self.store,self.identity,'logs',{'runtime/stdout.log':b'valid'},attempt=1)
        key=snap['commit']['members']['runtime/stdout.log']['id'];self.store.objects[key]=('member','stdout',b'corrupt')
        with self.assertRaises(LifecycleError):recover_terminal_final(self.root,store=self.store)
        self.assertNotIn('logs',json.loads((self.root/'state.json').read_text()))
    def test_other_attempt_logs_are_not_recovered(self):
        engine=self.long_running();engine.plan.update(parent='fixture',token_command=['must-not-run']);atomic(self.root/'plan.json',engine.plan);engine.finish('failed','lost')
        publish(self.store,self.identity,'logs',{'runtime/stdout.log':b'wrong attempt'},attempt=2)
        self.assertNotIn('logs',recover_terminal_final(self.root,store=self.store))

class CleanupTests(unittest.TestCase):
    def test_missing_listing_still_sends_exact_unassign_and_accepts404(self):
        provider=GoogleProvider.__new__(GoogleProvider);provider.client=mock.Mock();error=RuntimeError('private URL');error.response=mock.Mock(status_code=404);provider.client.unassign.side_effect=error
        with mock.patch.object(provider,'lookup',return_value=None):result=provider.release('owned-endpoint')
        provider.client.unassign.assert_called_once_with('owned-endpoint');self.assertEqual(result['unassign_status'],404)
    def test_denied_cleanup_does_not_claim_absence_or_expose_exception(self):
        provider=GoogleProvider.__new__(GoogleProvider);provider.client=mock.Mock();error=RuntimeError('private URL');error.response=mock.Mock(status_code=403);provider.client.unassign.side_effect=error
        with self.assertRaisesRegex(LifecycleError,'RuntimeError') as caught:provider.release('owned')
        self.assertNotIn('private',str(caught.exception))

class PackagingPolicyTests(unittest.TestCase):
    def test_colaml_public_packager_and_saved_replay_preserve_completion_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'experiment.py').write_text('pass');config=root/'job.colaml'
            save_answers(config,{'commands':[['python','experiment.py']],'project':str(root),'inputs':[],
                'environment':{'dependencies':[]},'compute':{'gpu':False},'outputs':{'include':['outputs/**'],'checkpoints':[],'completed':[GROUP]},'limits':{'timeout_seconds':60}})
            out=io.StringIO();err=io.StringIO()
            with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
                status=main(['package','--colaml',str(config),'--non-interactive','--output',str(root/'one.colajob'),'--save-colaml',str(root/'saved.colaml')])
            self.assertEqual(status,0,err.getvalue());job=json.loads((root/'one.colajob').read_text())
            self.assertEqual(job['managed_recovery']['artifact_groups'],[GROUP])
            first=json.loads(out.getvalue())['job_id'];out=io.StringIO()
            with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
                status=main(['package','--colaml',str(root/'saved.colaml'),'--non-interactive','--output',str(root/'two.colajob')])
            self.assertEqual(status,0,err.getvalue());self.assertEqual(json.loads(out.getvalue())['job_id'],first)
    def test_invalid_completion_paths_and_conditions_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            script=Path(tmp)/'work.py';script.write_text('pass');value=notebook_job(script)
            for group in [{**GROUP,'marker':'secrets.json'},{**GROUP,'files':['../secret']},{**GROUP,'ready_when':{}},{**GROUP,'ready_when':{'status':True}}]:
                value['managed_recovery']={'checkpoint_pattern':'outputs/off.json','log_patterns':[],'artifact_groups':[group]}
                with self.subTest(group=group),self.assertRaises(ValueError):JobSpec.from_mapping(value)

@unittest.skipUnless(sys.platform=='linux','real worker interruption requires Linux')
class WorkerInterruptionTests(unittest.TestCase):
    def test_completed_artifact_is_committed_before_workload_finishes_and_survives_worker_death(self):
        from test_lifecycle_worker import FileStore
        import colabcola.lifecycle_worker as worker
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'produce.py'
            source.write_text('import pathlib,json,time\np=pathlib.Path("outputs/clips/one");p.mkdir(parents=True)\n(p/"video.mp4").write_bytes(b"complete-output")\n(p/"manifest.tmp").write_text(json.dumps({"status":"completed"}))\n(p/"manifest.tmp").replace(p/"manifest.json")\nprint("completed one;second clip still running",flush=True)\ntime.sleep(60)\n')
            value=notebook_job(source);value['entrypoint']['argv']=[];value['outputs']=[{'pattern':'outputs/**','required':True,'maximum_bytes':10000}]
            value['managed_recovery']={'checkpoint_pattern':'outputs/off.json','log_patterns':[],'artifact_groups':[GROUP]}
            job=root/'work.colajob';job.write_text(json.dumps(value));compiled=compile_invocation(load_job_invocation(job),root/'notebook.ipynb')
            control=root/'control';control.mkdir()
            for name,path in [('notebook.ipynb',root/'notebook.ipynb'),('lifecycle_worker.py',Path(worker.__file__)),('lifecycle_io.py',Path(worker.__file__).with_name('lifecycle_io.py'))]:(control/name).write_bytes(path.read_bytes())
            config={'run_id':'a'*32,'job_id':compiled.job_id,'directory':str(control),'content_root':str(root/'runtime'),'files':{p.name:digest(p.read_bytes()) for p in control.iterdir()},'parent':'fixture','attempt':1,'deadline':time.time()+35,'resume':None,'checkpoint_pattern':'outputs/off.json','log_patterns':[],'artifact_groups':[GROUP],'artifact_output_rules':[{'pattern':'outputs/**','maximum_bytes':10000}],'poll_seconds':.05}
            atomic(control/'worker.json',config)
            script='import sys;sys.path.insert(0,'+repr(str(Path(__file__).parent))+');from test_lifecycle_worker import FileStore;import colabcola.lifecycle_worker as w;w.DriveStore=lambda *args:FileStore('+repr(str(root/'store'))+');w.run('+repr(str(control/'worker.json'))+')'
            process=subprocess.Popen([sys.executable,'-c',script],stdout=subprocess.PIPE,stderr=subprocess.PIPE);pid=None
            try:
                store=FileStore(root/'store');limit=time.monotonic()+20;committed=None
                while time.monotonic()<limit:
                    values=snapshots(store,config,'artifacts')
                    if values:committed=values[-1];break
                    if process.poll() is not None:self.fail(process.communicate()[1].decode())
                    time.sleep(.05)
                self.assertIsNotNone(committed);self.assertIsNone(process.poll());self.assertEqual(store.list('final'),[])
                status=json.loads((control/'status.json').read_text());pid=status['execution_pid']
                os.killpg(pid,signal.SIGKILL);process.kill();process.communicate(timeout=5)
                cold=recover(FileStore(root/'store'),committed,root/'cold')
                self.assertEqual((cold/'outputs/clips/one/video.mp4').read_bytes(),b'complete-output')
                self.assertEqual(json.loads((cold/'outputs/clips/one/manifest.json').read_text())['status'],'completed')
                self.assertEqual(store.list('final'),[])
            finally:
                if process.poll() is None:process.kill();process.communicate(timeout=5)
                if pid:
                    try:os.killpg(pid,signal.SIGKILL)
                    except ProcessLookupError:pass