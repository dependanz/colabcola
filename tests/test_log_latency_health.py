"""Offline log latency and renewable-health regressions. No cloud providers."""
import io
import json
from pathlib import Path
import tempfile
import threading
import os
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from colabcola.lifecycle_io import publish, recover, digest, LifecycleError
from colabcola.lifecycle_auth import RenewalPump, authorization_health
from colabcola.lifecycle import reported_snapshot
from colabcola.managed import Progress
import test_lifecycle as fixtures


class LogTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = fixtures.MemoryStore()
        self.identity = {'run_id':'a'*32,'job_id':'b'*64}
        self.logs = {f'outputs/clip{i}/stderr.log':f'old clip {i}\n'.encode() for i in range(16)}

    def test_only_changed_log_is_published_and_cache_cannot_be_mutated_by_caller(self):
        with mock.patch.object(self.store,'put',wraps=self.store.put) as put:
            first=publish(self.store,self.identity,'logs',self.logs,attempt=1)
            self.assertEqual(put.call_count,17)
            first['commit']['members']['outputs/clip0/stderr.log']['id']='caller mutation'
            changed={**self.logs,'outputs/clip15/stderr.log':b'new progress\n'}
            put.reset_mock();second=publish(self.store,self.identity,'logs',changed,attempt=1)
            self.assertEqual(put.call_count,2)  # One changed member plus its commit.
            self.assertNotEqual(second['commit']['members']['outputs/clip0/stderr.log']['id'],'caller mutation')
            put.reset_mock();publish(self.store,self.identity,'logs',changed,attempt=2)
            self.assertEqual(put.call_count,17)
            put.reset_mock();publish(self.store,{**self.identity,'job_id':'c'*64},'logs',changed,attempt=2)
            self.assertEqual(put.call_count,17)

    def pair(self):
        first=publish(self.store,self.identity,'logs',self.logs,attempt=1)
        path=recover(self.store,first,self.root/'first')
        before={**first,'local_directory':str(path)}
        second=publish(self.store,self.identity,'logs',{**self.logs,'outputs/clip15/stderr.log':b'new\n'},attempt=1)
        return before,second

    def test_unchanged_local_members_are_rehashed_and_only_changed_member_is_downloaded(self):
        before,second=self.pair()
        with mock.patch.object(self.store,'get',wraps=self.store.get) as get:
            result=recover(self.store,second,self.root/'second',previous=before)
            self.assertEqual(get.call_count,1)
            self.assertEqual((result/'outputs/clip15/stderr.log').read_bytes(),b'new\n')
            self.assertEqual((result/'outputs/clip0/stderr.log').read_bytes(),self.logs['outputs/clip0/stderr.log'])
            get.reset_mock();recover(self.store,second,self.root/'cold')
            self.assertEqual(get.call_count,16,'cold recovery must still verify remote bytes')

    def test_corrupt_cache_is_rejected_before_snapshot_is_visible(self):
        before,second=self.pair()
        (Path(before['local_directory'])/'outputs/clip0/stderr.log').write_bytes(b'corrupt')
        with self.assertRaisesRegex(LifecycleError,'cached log member integrity'):
            recover(self.store,second,self.root/'bad',previous=before)
        self.assertFalse((self.root/'bad').exists())

    def test_missing_local_member_is_fetched_and_foreign_attempt_is_not_reused(self):
        before,second=self.pair()
        (Path(before['local_directory'])/'outputs/clip0/stderr.log').unlink()
        with mock.patch.object(self.store,'get',wraps=self.store.get) as get:
            recover(self.store,second,self.root/'missing',previous=before)
            self.assertEqual(get.call_count,2)
            before['commit']['attempt']=2
            get.reset_mock();recover(self.store,second,self.root/'foreign',previous=before)
            self.assertEqual(get.call_count,16)


class SupervisorLogTests(unittest.TestCase):
    setUp=fixtures.LifecycleTests.setUp
    engine=fixtures.LifecycleTests.engine
    reload=fixtures.LifecycleTests.reload
    running=fixtures.LifecycleTests.running

    def test_cli_receives_verified_logs_before_slow_provider_call_returns(self):
        engine=self.running();output=io.StringIO();engine.observer=Progress(self.root,output)
        publish(self.store,self.identity,'logs',{'runtime/stdout.log':b'live progress now\n'},attempt=1)
        entered=threading.Event();release=threading.Event()
        def lookup(endpoint):
            entered.set();release.wait(3);return endpoint
        self.provider.lookup=lookup
        thread=threading.Thread(target=engine.advance);thread.start()
        try:
            self.assertTrue(entered.wait(2))
            self.assertIn('live progress now',output.getvalue())
            self.assertTrue(thread.is_alive(),'provider is still blocked after progress is visible')
        finally:release.set();thread.join(3)
        self.assertFalse(thread.is_alive())

    def test_worker_log_commit_bypasses_drive_search_lag_and_is_not_refetched(self):
        engine=self.running()
        snap=publish(self.store,self.identity,'logs',{'runtime/stdout.log':b'index not ready\n'},attempt=1)
        ref={k:snap[k] for k in ('id','sha256','bytes')}
        observed={'state':'running','attempt':1,'logs_commit':ref}
        engine.state['worker']=observed
        with mock.patch.object(self.store,'list',return_value=[]),mock.patch.object(self.provider,'observe',return_value=observed):
            engine.advance()
            self.assertEqual((Path(engine.state['logs']['local_directory'])/'runtime/stdout.log').read_bytes(),b'index not ready\n')
            with mock.patch.object(self.store,'get',wraps=self.store.get) as get:
                engine.advance();get.assert_not_called()
        self.assertEqual(len(self.provider.allocations),1)

    def test_malformed_direct_reference_is_rejected_before_transport(self):
        engine=self.running()
        good={'id':'fixture','sha256':'a'*64,'bytes':10}
        with mock.patch.object(self.store,'get',side_effect=AssertionError('no transfer')):
            for change in ({'id':'../escape'},{'bytes':True},{'sha256':'invalid'}):
                with self.assertRaisesRegex(LifecycleError,'reference invalid'):
                    reported_snapshot(self.store,engine.plan,engine.state['attempts'][0],{**good,**change},'logs')

    def test_worker_log_commit_cannot_cross_job_or_attempt(self):
        engine=self.running()
        for identity,attempt in [(self.identity,2),({**self.identity,'job_id':'c'*64},1)]:
            snap=publish(self.store,identity,'logs',{'stdout.log':b'foreign'},attempt=attempt)
            with self.assertRaisesRegex(LifecycleError,'identity mismatch'):
                reported_snapshot(self.store,engine.plan,engine.state['attempts'][0],snap,'logs')


class RenewalHealthTests(unittest.TestCase):
    def state(self):
        return {'phase':'monitoring','attempts':[{'number':1,'endpoint':'owned'}],
                'drive_authorization':{'state':'renewed','attempt':1,'expires_at':3600}}

    def worker(self,expiry=6600):
        return {'attempt':1,'authorization':{'latest_expires_at':expiry,'verified_publications':1,
                                           'last_verified_publication':{'kind':'logs','at':3200}}}

    def retry(self):
        return {'attempt':1,'credential_expires_at':6600,
                'delivery':{'state':'retrying','stage':'delivery','error_type':'ReadTimeout','at':3250}}

    def test_delivery_retry_does_not_claim_valid_access_is_unavailable(self):
        result=authorization_health(self.state(),self.retry(),now=3300)
        self.assertEqual(result['state'],'valid');self.assertEqual(result['expires_at'],3600)
        self.assertEqual(result['delivery']['state'],'retrying')

    def test_worker_confirms_lost_ack_only_for_a_known_lease(self):
        state=self.state();state['worker']=self.worker()
        state['drive_authorization']['error_type']='old error'
        result=authorization_health(state,self.retry(),now=3300)
        self.assertEqual(result['expires_at'],6600);self.assertEqual(result['state'],'valid')
        self.assertEqual(result['delivery']['source'],'worker');self.assertNotIn('error_type',result)
        self.assertEqual(result['last_delivery_error']['error_type'],'ReadTimeout')

    def test_unrecognized_or_malformed_worker_evidence_cannot_extend_expiry(self):
        values=[self.worker(9999),{**self.worker(),'attempt':2},{**self.worker(),'attempt':True},
                {'attempt':1,'authorization':[]}, {'attempt':1,'authorization':{'last_verified_publication':[]}},
                {'attempt':1,'authorization':{'latest_expires_at':float('inf')}},
                {'attempt':1,'authorization':{'latest_expires_at':6600,'verified_publications':1,'last_verified_publication':{'at':5000}}}]
        for worker in values:
            with self.subTest(worker=worker):
                state=self.state();state['worker']=worker
                result=authorization_health(state,self.retry(),now=3480)
                self.assertEqual(result['expires_at'],3600);self.assertEqual(result['state'],'expiring')

    def test_success_replaces_old_active_error_and_another_attempt_is_ignored(self):
        state=self.state();state['drive_authorization'].update(error_type='old',delivery={'state':'retrying','stage':'delivery','error_type':'old'})
        success={'attempt':1,'expires_at':6600,'credential_expires_at':6600,'delivery':{'state':'confirmed','source':'delivery'}}
        result=authorization_health(state,success,now=3300)
        self.assertNotIn('error_type',result);self.assertNotIn('error_type',result['delivery'])
        result=authorization_health(self.state(),{**success,'attempt':2},now=3300)
        self.assertEqual(result['expires_at'],3600)

    def test_invalid_known_expiry_fails_closed(self):
        for value in (True,'secret',float('nan'),float('inf'),None):
            state=self.state();state['drive_authorization']['expires_at']=value
            result=authorization_health(state,now=3300)
            self.assertEqual(result['state'],'unavailable');self.assertEqual(result['expires_at'],0)

    def test_unchanged_token_is_not_delivered_every_minute(self):
        engine=SimpleNamespace(state=self.state(),plan={});provider=mock.Mock()
        pump=RenewalPump(engine,provider,lambda:{'access_token':'secret','expires_at':3600},clock=lambda:1000)
        for _ in range(20):pump.once()
        self.assertEqual(provider.refresh_token.call_count,1)
        self.assertNotIn('secret',json.dumps(pump.health))

    def test_lost_ack_is_not_retried_after_worker_proves_new_access(self):
        engine=SimpleNamespace(state=self.state(),plan={});provider=mock.Mock()
        provider.refresh_token.side_effect=TimeoutError('private URL/token')
        pump=RenewalPump(engine,provider,lambda:{'access_token':'secret','expires_at':6600},clock=lambda:3300)
        try:pump.once()
        except TimeoutError as exc:pump.failed(exc)
        self.assertEqual(pump.health['delivery']['state'],'retrying')
        engine.state['worker']=self.worker();pump.once()
        self.assertEqual(provider.refresh_token.call_count,1)
        self.assertEqual(pump.health['delivery']['source'],'worker')
        self.assertNotIn('error_type',pump.health['delivery']);self.assertNotIn('private',json.dumps(pump.health))

    def test_refresh_failure_keeps_last_expiry_and_does_not_hide_retry(self):
        engine=SimpleNamespace(state=self.state(),plan={});provider=mock.Mock()
        loader=mock.Mock(return_value={'access_token':'secret','expires_at':3600})
        pump=RenewalPump(engine,provider,loader,clock=lambda:3300);pump.once()
        loader.side_effect=RuntimeError('private refresh output')
        try:pump.once()
        except RuntimeError as exc:pump.failed(exc)
        engine.state['worker']=self.worker(3600)
        result=authorization_health(engine.state,pump.health,now=3300)
        self.assertEqual(result['state'],'valid');self.assertEqual(result['delivery']['stage'],'credentials')
        self.assertEqual(authorization_health(engine.state,pump.health,now=3480)['state'],'expiring')
        self.assertNotIn('private',json.dumps(result))
        loader.side_effect=None;pump.once()
        self.assertEqual(provider.refresh_token.call_count,1)
        recovered=authorization_health(engine.state,pump.health,now=3300)
        self.assertEqual(recovered['delivery']['state'],'confirmed')
        self.assertNotIn('error_type',recovered['delivery'])

    def test_cli_distinguishes_valid_access_from_retry_and_clears_it_on_recovery(self):
        output=io.StringIO();progress=Progress(Path('.'),output)
        state=self.state();state['drive_authorization']=authorization_health(state,self.retry(),now=3300)
        progress(state);progress(state)
        self.assertEqual(output.getvalue().count('Drive access valid; token delivery retrying (ReadTimeout)'),1)
        state['worker']=self.worker();state['drive_authorization']=authorization_health(state,self.retry(),now=3300)
        progress(state)
        self.assertTrue(output.getvalue().rstrip().endswith('Drive access valid'))


@unittest.skipUnless(sys.platform == "linux", "real runtime worker requires Linux")
class WorkerProgressIntegrationTests(unittest.TestCase):
    def test_real_worker_publishes_direct_log_ref_before_workload_handshake(self):
        from colabcola.compiler import compile_invocation
        from colabcola.invocation import load_job_invocation
        from colabcola.lifecycle import verify_final
        from colabcola.lifecycle_io import atomic
        from test_notebook import notebook_job
        from test_lifecycle_worker import FileStore
        import colabcola.lifecycle_worker as worker
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);marker=root/'continue';source=root/'work.py'
            source.write_text('from pathlib import Path\nimport time\nprint("local worker step 1",flush=True)\nwhile not Path('+repr(str(marker))+').exists():time.sleep(.01)\nprint("local worker finished",flush=True)\n')
            mapping=notebook_job(source);mapping['entrypoint']['argv']=[]
            job=root/'work.colajob';atomic(job,mapping)
            compiled=compile_invocation(load_job_invocation(job),root/'notebook.ipynb')
            control=root/'control';control.mkdir()
            for name,body in [('notebook.ipynb',compiled.notebook_bytes),('lifecycle_worker.py',Path(worker.__file__).read_bytes()),('lifecycle_io.py',Path(worker.__file__).with_name('lifecycle_io.py').read_bytes())]:
                (control/name).write_bytes(body)
            identity={'run_id':'a'*32,'job_id':compiled.job_id}
            config={**identity,'directory':str(control),'content_root':str(root/'runtime'),
                    'files':{p.name:digest(p.read_bytes()) for p in control.iterdir()},
                    'parent':'fixture','attempt':1,'deadline':time.time()+35,'resume':None,
                    'checkpoint_pattern':'outputs/no-checkpoint-*.json','log_patterns':[], 'poll_seconds':.05}
            atomic(control/'worker.json',config)
            script=('import sys;sys.path.insert(0,'+repr(str(Path(__file__).parent))+');'
                    'from test_lifecycle_worker import FileStore;import colabcola.lifecycle_worker as w;'
                    'w.DriveStore=lambda *args:FileStore('+repr(str(root/'store'))+');'
                    'w.run('+repr(str(control/'worker.json'))+')')
            process=subprocess.Popen([sys.executable,'-c',script],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            output=io.StringIO();progress=Progress(root,output);store=FileStore(root/'store')
            try:
                limit=time.monotonic()+20
                while time.monotonic()<limit:
                    status=json.loads((control/'status.json').read_text()) if (control/'status.json').exists() else {}
                    if status.get('logs_commit'):
                        snap=reported_snapshot(store,identity,{'number':1},status['logs_commit'],'logs')
                        path=recover(store,snap,root/'logs'/snap['sha256'])
                        progress({'logs':{**snap,'local_directory':str(path)}})
                        if 'local worker step 1' in output.getvalue():break
                    if process.poll() is not None:self.fail(process.communicate()[1].decode())
                    time.sleep(.05)
                self.assertIn('local worker step 1',output.getvalue())
                self.assertNotIn('local worker finished',output.getvalue())
                self.assertIsNone(process.poll(),'worker still waiting for local handshake')
                marker.touch();stdout,stderr=process.communicate(timeout=12)
                self.assertEqual(process.returncode,0,stderr.decode())
                status=json.loads((control/'status.json').read_text());self.assertEqual(status['state'],'completed')
                final=reported_snapshot(store,identity,{'number':1},status['final_commit'],'final')
                path=recover(store,final,root/'final');self.assertEqual(verify_final(path,compiled.job_id)['terminal_state'],'success')
            finally:
                marker.touch()
                try:process.communicate(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.communicate(timeout=3)
                if (control/'status.json').exists():
                    pid=json.loads((control/'status.json').read_text()).get('execution_pid')
                    if pid:
                        try:os.killpg(pid,9)
                        except ProcessLookupError:pass


if __name__=='__main__':unittest.main()