"""Offline tracked-execution regressions. No Colab credentials or allocation."""
import base64
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from colabcola.lifecycle_io import LifecycleError, atomic, digest
from colabcola.lifecycle_kernel import WorkloadExecution, workload_code
from colabcola.lifecycle_provider import GoogleProvider
from colabcola.lifecycle_worker import current_token, stop_reason, token_reader


class ExecutionTests(unittest.TestCase):
    def connection(self, failure=None):
        ready=mock.Mock();ready.is_set.return_value=True
        wire=SimpleNamespace(connection_ready=ready,execute=mock.Mock(return_value='message'),
            iopub_channel=queue.Queue(),shell_channel=queue.Queue())
        if failure:wire.execute.side_effect=failure
        # Queue.get_msg matches the installed channel API,including nonblocking timeout.
        for name in ('iopub_channel','shell_channel'):
            q=getattr(wire,name);q.get_msg=lambda timeout,q=q:q.get(timeout=timeout)
        c=WorkloadExecution.__new__(WorkloadExecution)
        c.client=SimpleNamespace(_manager=SimpleNamespace(client=wire),stop=mock.Mock())
        c.request_id=None;c.state='attached';c.kernel_id='kernel'
        return c,wire

    def test_submit_once_then_track_busy_and_reply_without_leaking_outputs(self):
        c,w=self.connection();c.submit('real work')
        w.iopub_channel.put({'parent_header':{'msg_id':'message'},'header':{'msg_type':'status'},'content':{'execution_state':'busy'}})
        self.assertEqual(c.poll()['request_state'],'busy')
        w.shell_channel.put({'parent_header':{'msg_id':'message'},'header':{'msg_type':'execute_reply'},'content':{'status':'error','evalue':'secret URL'}})
        value=c.poll();self.assertEqual(value['request_state'],'failed');self.assertNotIn('secret',json.dumps(value))
        with self.assertRaisesRegex(LifecycleError,'not be replayed'):c.submit('again')
        w.execute.assert_called_once()

    def test_failed_kernel_reply_keeps_exception_class_not_private_message(self):
        c, wire = self.connection()
        c.submit('real work')
        wire.shell_channel.put({
            'parent_header': {'msg_id': 'message'},
            'header': {'msg_type': 'execute_reply'},
            'content': {'status': 'error', 'ename': 'KeyboardInterrupt',
                        'evalue': 'PRIVATE-ACCESS-TOKEN',
                        'traceback': ['PRIVATE-TRACEBACK']}})
        observed = c.poll()
        self.assertEqual(observed['request_state'], 'failed')
        self.assertEqual(observed['reply_error_type'], 'KeyboardInterrupt')
        self.assertNotIn('PRIVATE-', json.dumps(observed))
        self.assertEqual(c.poll()['reply_error_type'], 'KeyboardInterrupt')

    def test_untrusted_reply_exception_names_are_omitted(self):
        for value in ('PRIVATE-ACCESS-TOKEN', 'bad.word', 'x' * 200, 5, None):
            with self.subTest(value=value):
                c, wire = self.connection()
                c.submit('real work')
                wire.shell_channel.put({
                    'parent_header': {'msg_id': 'message'},
                    'header': {'msg_type': 'execute_reply'},
                    'content': {'status': 'error', 'ename': value,
                                'evalue': 'PRIVATE-ACCESS-TOKEN'}})
                observed = c.poll()
                self.assertEqual(observed['request_state'], 'failed')
                self.assertNotIn('reply_error_type', observed)
                self.assertNotIn('PRIVATE-', json.dumps(observed))

    def test_uncertain_submission_and_reconnect_never_reexecute(self):
        c,w=self.connection(TimeoutError('secret'));self.assertRaises(LifecycleError,c.submit,'real work')
        self.assertEqual(c.state,'submission_unknown')
        self.assertRaises(LifecycleError,c.submit,'again');w.execute.assert_called_once()
        c,w=self.connection();self.assertEqual(c.poll()['request_state'],'attached');w.execute.assert_not_called()
        c.close();c.client.stop.assert_called_once()

    def test_connection_loss_is_visible_and_redacted(self):
        c,w=self.connection();w.connection_ready.is_set.return_value=False
        with self.assertRaisesRegex(LifecycleError,'connection lost'):c.poll()


class TokenHandoffTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.now=time.time();atomic(self.root/'drive-token.json',{'access_token':'old','expires_at':self.now+300})
    def test_busy_kernel_delivery_needs_contents_upload_only(self):
        p=GoogleProvider.__new__(GoogleProvider);p._session=mock.Mock(return_value=object());p._execute=mock.Mock(side_effect=AssertionError('busy kernel must not receive token commands'))
        def upload(session,path,body):(self.root/Path(path).name).write_bytes(body)
        p._upload=upload
        p.refresh_token('owned',{'run_id':'run','execution_mode':'kernel-v1'},{'number':1},{'access_token':'new','expires_at':self.now+3600,'unexpected':'discard'})
        self.assertEqual(token_reader(self.root)(),'new');self.assertIsNone(stop_reason({'deadline':self.now+2000,'authorization_policy':'renewable-v1'},self.root,self.now))
        p._execute.assert_not_called()
    def test_partial_invalid_expired_or_older_pending_token_preserves_active(self):
        bad=['{',json.dumps({'access_token':'x','expires_at':self.now+200}),json.dumps({'access_token':'x','expires_at':self.now-1}),json.dumps({'access_token':'','expires_at':self.now+5000}),json.dumps({'access_token':'x','expires_at':float('inf')}),json.dumps({'access_token':'x','expires_at':self.now+4000,'refresh_token':'never'})]
        for value in bad:
            (self.root/'drive-token.next.json').write_text(value)
            self.assertEqual(current_token(self.root)['access_token'],'old')
    def test_expiry_guard_still_stops_without_valid_replacement(self):
        atomic(self.root/'drive-token.json',{'access_token':'old','expires_at':self.now+60})
        self.assertEqual(stop_reason({'deadline':self.now+2000,'authorization_policy':'renewable-v1'},self.root,self.now),'drive_authorization_expiring')
    def test_later_upload_is_not_removed_when_prior_token_is_adopted(self):
        atomic(self.root/'drive-token.next.json',{'access_token':'new','expires_at':self.now+3600})
        self.assertEqual(current_token(self.root)['access_token'],'new')
        atomic(self.root/'drive-token.next.json',{'access_token':'newest','expires_at':self.now+7200})
        self.assertEqual(current_token(self.root)['access_token'],'newest')
        self.assertTrue((self.root/'drive-token.next.json').exists())


class ProviderExecutionTests(unittest.TestCase):
    def setUp(self):
        self.p=GoogleProvider.__new__(GoogleProvider);self.p._session=mock.Mock(return_value=object())
        self.p._kernel=mock.Mock(return_value={'id':'kernel','execution_state':'busy'})
        self.p._workload_connection=mock.Mock(return_value=mock.Mock())
        self.p._execute=mock.Mock();self.p._upload=mock.Mock()
        self.plan={'run_id':'run','job_id':'job','parent':'folder','execution_mode':'kernel-v1','authorization_policy':'renewable-v1','checkpoint_pattern':'none','log_patterns':[],'files':{}}
        self.attempt={'number':1,'deadline':time.time()+1000}
        self.identity={'run_id':'run','job_id':'job','attempt':1,'kernel_id':'kernel'}
    def test_new_boot_submits_real_worker_and_writes_intent(self):
        self.p._json_file=mock.Mock(return_value=None)
        with tempfile.TemporaryDirectory() as tmp:
            self.plan['directory']=tmp
            for name in ['notebook.ipynb','lifecycle_worker.py','lifecycle_io.py']:
                body=b'prepared';(Path(tmp)/name).write_bytes(body);self.plan['files'][name]=digest(body)
            self.p.boot('owned',self.plan,self.attempt,{'access_token':'secret','expires_at':time.time()+1000})
        connection=self.p._workload_connection.return_value
        code=connection.submit.call_args.args[0];self.assertIn('runpy.run_path',code);self.assertNotIn('subprocess.Popen',code);self.assertNotIn('secret',code)
        self.assertEqual(self.p._execute.call_count,1) # mkdir only;worker is real tracked execution.
        self.assertTrue(any(c.args[1].endswith('/kernel-request.json') for c in self.p._upload.call_args_list))
    def test_restart_attaches_to_recorded_kernel_without_submitting_again(self):
        self.p._json_file=mock.Mock(side_effect=[{**self.identity,'state':'running'},self.identity])
        self.p.boot('owned',self.plan,self.attempt,{})
        self.p._workload_connection.return_value.submit.assert_not_called();self.p._execute.assert_not_called();self.p._upload.assert_not_called()
    def test_uncertain_or_wrong_identity_never_replays(self):
        for values in ([None,self.identity],[{**self.identity,'run_id':'other'},self.identity]):
            self.p._json_file=mock.Mock(side_effect=values)
            with self.assertRaises(LifecycleError):self.p.boot('owned',self.plan,self.attempt,{})
        self.p._workload_connection.assert_not_called()
    def test_observer_reattaches_without_replaying(self):
        self.p._contents=mock.Mock(return_value={'format':'text','content':json.dumps({'state':'running','attempt':1})})
        self.p._json_file=mock.Mock(side_effect=[{**self.identity,'state':'running'},self.identity])
        first=mock.Mock();first.poll.side_effect=LifecycleError('lost');second=mock.Mock();second.poll.return_value={'mode':'kernel-v1','request_state':'attached'}
        self.p._workloads={'owned':first};self.p._workload_connection.side_effect=[first,second]
        result=self.p.observe('owned',self.plan,self.attempt,None)
        self.assertEqual(result['kernel_execution']['execution_state'],'busy');first.close.assert_called_once();first.submit.assert_not_called();second.submit.assert_not_called()
    def test_observe_preserves_owned_worker_failure_receipt_without_secrets(self):
        self.p._contents = mock.Mock(return_value={
            'format': 'text', 'content': json.dumps({'state': 'running', 'attempt': 1})})
        self.p._json_file = mock.Mock(side_effect=[
            {**self.identity, 'state': 'failed', 'finished_at': 1790000000.25,
             'error_type': 'KeyboardInterrupt', 'evalue': 'PRIVATE-ACCESS-TOKEN'},
            self.identity])
        self.p._workload_connection.return_value.poll.return_value = {
            'mode': 'kernel-v1', 'request_state': 'failed',
            'reply_error_type': 'KeyboardInterrupt'}
        result = self.p.observe('owned', self.plan, self.attempt, None)
        details = result['kernel_execution']
        self.assertEqual(details['worker_execution_state'], 'failed')
        self.assertEqual(details['worker_error_type'], 'KeyboardInterrupt')
        self.assertEqual(details['worker_finished_at'], 1790000000.25)
        self.assertEqual(details['reply_error_type'], 'KeyboardInterrupt')
        self.assertNotIn('PRIVATE-', json.dumps(result))

    def test_observe_omits_malformed_worker_diagnostics(self):
        self.p._contents = mock.Mock(return_value={
            'format': 'text', 'content': json.dumps({'state': 'running', 'attempt': 1})})
        self.p._json_file = mock.Mock(side_effect=[
            {**self.identity, 'state': 'failed', 'finished_at': float('nan'),
             'error_type': 'PRIVATE-ACCESS-TOKEN'},
            self.identity])
        self.p._workload_connection.return_value.poll.return_value = {
            'mode': 'kernel-v1', 'request_state': 'failed'}
        details = self.p.observe('owned', self.plan, self.attempt, None)['kernel_execution']
        self.assertNotIn('worker_error_type', details)
        self.assertNotIn('worker_finished_at', details)
        self.assertNotIn('PRIVATE-', json.dumps(details))

    def test_missing_receipt_and_intent_while_running_is_unknown_not_worker_death(self):
        self.p._contents = mock.Mock(return_value={
            'format': 'text', 'content': json.dumps({'state':'running', 'attempt':1})})
        self.p._json_file = mock.Mock(return_value=None)
        with self.assertRaisesRegex(LifecycleError, 'receipt missing'):
            self.p.observe('owned', self.plan, self.attempt, None)
        self.p._kernel.assert_not_called()
        self.p._workload_connection.assert_not_called()
        self.p._execute.assert_not_called()

    def test_terminal_failed_status_without_execution_artifacts_is_still_actionable(self):
        for status in ('failed', 'persistence_failed'):
            with self.subTest(status=status):
                self.p._contents = mock.Mock(return_value={
                    'format': 'text',
                    'content': json.dumps({'state': status, 'attempt': 1})})
                self.p._json_file = mock.Mock(return_value=None)
                self.p._kernel.reset_mock()
                result = self.p.observe('owned', self.plan, self.attempt, None)
                self.assertEqual(result['state'], status)
                self.assertEqual(result['kernel_execution']['request_state'],
                                 'receipt_missing')
                self.assertEqual(result['kernel_execution']['worker_execution_state'],
                                 'unknown')
                self.p._kernel.assert_not_called()
                self.p._workload_connection.assert_not_called()
                self.p._execute.assert_not_called()

    def test_terminal_status_from_wrong_attempt_cannot_bypass_receipt_identity(self):
        self.p._contents = mock.Mock(return_value={
            'format': 'text', 'content': json.dumps({'state': 'failed', 'attempt': 2})})
        self.p._json_file = mock.Mock(return_value=None)
        with self.assertRaisesRegex(LifecycleError, 'receipt missing'):
            self.p.observe('owned', self.plan, self.attempt, None)
        self.p._kernel.assert_not_called()

    def test_legacy_observation_has_no_kernel_receipt_dependency(self):
        self.plan.pop('execution_mode')
        self.p._contents = mock.Mock(return_value={
            'format': 'text', 'content': json.dumps({'state':'failed', 'attempt':1})})
        self.p._json_file = mock.Mock(side_effect=AssertionError(
            'legacy must not read tracked-execution artifacts'))
        result = self.p.observe('owned', self.plan, self.attempt, None)
        self.assertEqual(result, {'state':'failed', 'attempt':1})
        self.p._json_file.assert_not_called()
        self.p._kernel.assert_not_called()
        self.p._workload_connection.assert_not_called()

    def test_missing_owned_kernel_returns_observation_without_reexecuting(self):
        from colabcola.lifecycle_provider import OwnedKernelAbsent
        self.p._contents = mock.Mock(return_value={
            'format': 'text', 'content': json.dumps({'state': 'running', 'attempt': 1})})
        self.p._json_file = mock.Mock(side_effect=[
            {**self.identity, 'state': 'running'}, self.identity])
        self.p._kernel.side_effect = OwnedKernelAbsent('owned workload kernel is absent')
        result = self.p.observe('owned', self.plan, self.attempt, None)
        evidence = result['kernel_execution']
        self.assertEqual(evidence['request_state'], 'kernel_absent')
        self.assertEqual(evidence['worker_execution_state'], 'running')
        self.assertEqual(evidence['kernel_id'], 'kernel')
        self.p._workload_connection.assert_not_called()
        self.p._execute.assert_not_called()

    def test_kernel_listing_failure_is_not_claimed_as_absence(self):
        self.p._contents = mock.Mock(return_value={
            'format': 'text', 'content': json.dumps({'state': 'running', 'attempt': 1})})
        self.p._json_file = mock.Mock(side_effect=[
            {**self.identity, 'state': 'running'}, self.identity])
        self.p._kernel.side_effect = LifecycleError('runtime kernel listing HTTP 503')
        with self.assertRaisesRegex(LifecycleError, '503'):
            self.p.observe('owned', self.plan, self.attempt, None)
        self.p._workload_connection.assert_not_called()

    def test_connection_cleanup_does_not_release_compute(self):
        self.p.client=mock.Mock();one=mock.Mock();two=mock.Mock();self.p._workloads={'a':one,'b':two}
        self.p.close();one.close.assert_called_once();two.close.assert_called_once();self.p.client.unassign.assert_not_called()

    def test_changed_kernel_receipt_is_rejected_before_connection(self):
        self.p._json_file=mock.Mock(side_effect=[{**self.identity,'kernel_id':'other'},self.identity])
        with self.assertRaisesRegex(LifecycleError,'identity mismatch'):
            self.p.boot('owned',self.plan,self.attempt,{})
        self.p._workload_connection.assert_not_called()
    def test_cleanup_attempts_all_connections_and_redacts_failures(self):
        first=mock.Mock();first.close.side_effect=RuntimeError('secret');second=mock.Mock()
        self.p._workloads={'a':first,'b':second}
        with self.assertRaisesRegex(LifecycleError,'cleanup failed: RuntimeError') as caught:self.p.close()
        second.close.assert_called_once();self.assertNotIn('secret',str(caught.exception))



@unittest.skipUnless(os.name=='posix' and importlib.util.find_spec('ipykernel'),'local IPC kernel dependency unavailable')
class RealKernelTests(unittest.TestCase):
    def test_real_kernel_tracks_work_and_rotates_token_without_duplicate_execution(self):
        from jupyter_client import KernelManager
        from colabcola import lifecycle_worker
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);km=KernelManager(transport='ipc',connection_file=str(root/'kernel.json'))
            km.start_kernel(cwd=tmp,env={**os.environ,'PYTHONPATH':str(Path(lifecycle_worker.__file__).parents[1])})
            client=km.blocking_client();client.start_channels();client.wait_for_ready(timeout=30)
            try:
                # Reproduce the original defect on a genuine notebook kernel.
                child="import time;from pathlib import Path;time.sleep(1.5);Path('detached-done').write_text('yes')"
                old=client.execute('import subprocess,sys\nsubprocess.Popen([sys.executable,"-c",'+repr(child)+'],start_new_session=True)')
                deadline=time.monotonic()+10;idle=False
                while time.monotonic()<deadline:
                    msg=client.get_iopub_msg(timeout=5)
                    if msg.get('parent_header',{}).get('msg_id')==old and msg['header']['msg_type']=='status' and msg['content']['execution_state']=='idle':idle=True;break
                self.assertTrue(idle);self.assertFalse((root/'detached-done').exists(),'detached child must still be running after kernel goes idle')
                atomic(root/'drive-token.json',{'access_token':'old','expires_at':time.time()+300})
                fixture="""from pathlib import Path
import time,json
from colabcola.lifecycle_worker import current_token
p=Path(__file__).parent
def publish_text(name, value):
 temporary=p/(name+'.next')
 temporary.write_text(value)
 temporary.replace(p/name)
publish_text('entered','one execution')
while not (p/'finish').exists():
 publish_text('work-progress',str(time.time()))
 publish_text('token-used',current_token(p)['access_token'])
 time.sleep(.05)
(p/'completed').write_text('real work completed')
"""
                (root/'lifecycle_worker.py').write_text(fixture)
                handle=WorkloadExecution.__new__(WorkloadExecution);handle.client=SimpleNamespace(_manager=SimpleNamespace(client=client),stop=lambda:client.stop_channels());handle.kernel_id='local';handle.state='attached';handle.request_id=None
                client.connection_ready=mock.Mock();client.connection_ready.is_set.return_value=True
                identity={'run_id':'local','job_id':'local','attempt':1,'kernel_id':'local'}
                handle.submit(workload_code(str(root),identity))
                deadline=time.monotonic()+10
                while not (root/'entered').exists() and time.monotonic()<deadline:time.sleep(.05)
                self.assertTrue((root/'entered').exists())
                self.assertEqual(handle.poll()['request_state'],'busy');self.assertEqual(json.loads((root/'kernel-execution.json').read_text())['state'],'running')
                provider=GoogleProvider.__new__(GoogleProvider);provider._session=lambda e:None;provider._execute=mock.Mock(side_effect=AssertionError('busy kernel execution forbidden'))
                provider._upload=lambda session,path,body:(root/Path(path).name).write_bytes(body)
                provider.refresh_token('owned',{'run_id':'local','execution_mode':'kernel-v1'},{'number':1},{'access_token':'rotated','expires_at':time.time()+3600})
                deadline=time.monotonic()+5
                while (not (root/'token-used').exists() or (root/'token-used').read_text()!='rotated') and time.monotonic()<deadline:time.sleep(.05)
                self.assertEqual((root/'token-used').read_text(),'rotated')
                # Closing and reconnecting control channels does not restart or stop work.
                before=float((root/'work-progress').read_text());handle.close();time.sleep(.15)
                self.assertGreater(float((root/'work-progress').read_text()),before)
                client=km.blocking_client();client.start_channels();client.connection_ready=mock.Mock();client.connection_ready.is_set.return_value=True
                handle.client=SimpleNamespace(_manager=SimpleNamespace(client=client),stop=lambda:client.stop_channels());handle.request_id=None;handle.state='attached'
                self.assertEqual(handle.poll()['request_state'],'attached')
                (root/'finish').write_text('finish')
                deadline=time.monotonic()+10
                while json.loads((root/'kernel-execution.json').read_text())['state']=='running' and time.monotonic()<deadline:time.sleep(.05)
                self.assertEqual(json.loads((root/'kernel-execution.json').read_text())['state'],'finished');self.assertTrue((root/'completed').exists())
                self.assertEqual((root/'entered').read_text(),'one execution');provider._execute.assert_not_called()
                if os.environ.get('COLABCOLA_KERNEL_PROOF'):
                    atomic(Path(os.environ['COLABCOLA_KERNEL_PROOF'])/'local-kernel-proof.json',
                        {'transport':'ipc','old_bootstrap_idle_while_child_running':True,
                         'tracked_worker_busy_until_completion':True,'token_adopted_while_busy':True,
                         'disconnect_did_not_stop_work':True,'reconnected_without_resubmission':True,
                         'worker_execution_count':1,'cloud_runtime_attempts':0})
            finally:
                limit=time.monotonic()+5
                while not (root/'detached-done').exists() and time.monotonic()<limit:time.sleep(.05)
                client.stop_channels();km.shutdown_kernel(now=True)

    def test_actual_worker_publishes_completed_group_and_final_from_busy_kernel(self):
        from jupyter_client import KernelManager
        from test_lifecycle_worker import FileStore
        from test_notebook import notebook_job
        from colabcola.compiler import compile_invocation
        from colabcola.invocation import load_job_invocation
        from colabcola.lifecycle import verify_final
        from colabcola.lifecycle_io import snapshots,recover
        import colabcola.lifecycle_worker as worker
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);control=root/'control';control.mkdir()
            source=root/'produce.py'
            source.write_text("from pathlib import Path\nimport json,time\np=Path('outputs/clips/one');p.mkdir(parents=True)\n(p/'video.mp4').write_bytes(b'completed before final')\n(p/'manifest.json').write_text(json.dumps({'status':'completed'}))\nwhile not Path("+repr(str(control/'finish'))+").exists():time.sleep(.05)\n")
            value=notebook_job(source);value['entrypoint']['argv']=[]
            value['outputs']=[{'pattern':'outputs/**','required':True,'maximum_bytes':10000}]
            value['managed_recovery']={'checkpoint_pattern':'outputs/off.json','log_patterns':[],
                'artifact_groups':[{'marker':'outputs/clips/*/manifest.json','files':['video.mp4'],'ready_when':{'status':'completed'}}]}
            job=root/'work.colajob';job.write_text(json.dumps(value))
            compiled=compile_invocation(load_job_invocation(job),control/'notebook.ipynb')
            wrapper="import sys\nfrom test_lifecycle_worker import FileStore\nimport colabcola.lifecycle_worker as w\nw.DriveStore=lambda *args:FileStore("+repr(str(root/'store'))+")\nif len(sys.argv)==3 and sys.argv[2]=='--execute':w.execute_notebook(sys.argv[1])\nelse:w.run(sys.argv[1])\n"
            (control/'lifecycle_worker.py').write_text(wrapper)
            (control/'lifecycle_io.py').write_bytes(Path(worker.__file__).with_name('lifecycle_io.py').read_bytes())
            config={'run_id':'a'*32,'job_id':compiled.job_id,'directory':str(control),'content_root':str(root/'runtime'),
                'files':{p.name:digest(p.read_bytes()) for p in control.iterdir()},'parent':'fixture','attempt':1,
                'deadline':time.time()+30,'resume':None,'checkpoint_pattern':'outputs/off.json','log_patterns':[],
                'artifact_groups':value['managed_recovery']['artifact_groups'],
                'artifact_output_rules':[{'pattern':'outputs/**','maximum_bytes':10000}],
                'poll_seconds':.05,'authorization_policy':'renewable-v1'}
            atomic(control/'worker.json',config);atomic(control/'drive-token.json',{'access_token':'old','expires_at':time.time()+300})
            km=KernelManager(transport='ipc',connection_file=str(root/'kernel.json'))
            km.start_kernel(cwd=tmp,env={**os.environ,'PYTHONPATH':str(Path(worker.__file__).parents[1])+os.pathsep+str(Path(__file__).parent)})
            client=km.blocking_client();client.start_channels();client.wait_for_ready(timeout=30)
            try:
                request=client.execute(workload_code(str(control),{'run_id':config['run_id'],'job_id':config['job_id'],'attempt':1,'kernel_id':'local'}))
                busy=False;limit=time.monotonic()+10
                while time.monotonic()<limit:
                    msg=client.get_iopub_msg(timeout=5)
                    if msg.get('parent_header',{}).get('msg_id')==request and msg['header']['msg_type']=='status' and msg['content']['execution_state']=='busy':busy=True;break
                self.assertTrue(busy)
                store=FileStore(root/'store');limit=time.monotonic()+15;commits=[]
                while time.monotonic()<limit:
                    commits=snapshots(store,config,'artifacts')
                    if commits:break
                    time.sleep(.05)
                self.assertEqual(len(commits),1);self.assertEqual(len(commits[0]['commit']['members']),2)
                self.assertEqual(store.list('final'),[])
                self.assertEqual(json.loads((control/'kernel-execution.json').read_text())['state'],'running')
                atomic(control/'drive-token.next.json',{'access_token':'rotated','expires_at':time.time()+3600})
                limit=time.monotonic()+5
                while json.loads((control/'drive-token.json').read_text())['access_token']!='rotated' and time.monotonic()<limit:time.sleep(.05)
                self.assertEqual(json.loads((control/'drive-token.json').read_text())['access_token'],'rotated')
                (control/'finish').write_text('finish');limit=time.monotonic()+10;finals=[]
                while time.monotonic()<limit:
                    finals=snapshots(store,config,'final')
                    if finals:break
                    time.sleep(.05)
                self.assertEqual(len(finals),1)
                cold=recover(store,finals[0],root/'cold');self.assertEqual(verify_final(cold,compiled.job_id)['terminal_state'],'success')
                if os.environ.get('COLABCOLA_KERNEL_PROOF'):
                    atomic(Path(os.environ['COLABCOLA_KERNEL_PROOF'])/'actual-worker-kernel-proof.json',
                        {'transport':'ipc','genuine_worker_in_kernel':True,'completed_group_commits':1,
                         'completed_members':2,'no_final_while_second_stage_pending':True,
                         'pending_token_adopted_while_busy':True,'final_cold_hash_verification':'passed',
                         'cloud_runtime_attempts':0,'drive_backend':'injected filesystem'})
            finally:
                client.stop_channels();km.shutdown_kernel(now=True)

if __name__=='__main__':unittest.main()
