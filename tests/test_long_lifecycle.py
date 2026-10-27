"""Offline long-run, renewal and documented REST contracts; never allocate compute."""
import contextlib
import io
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
import uuid

from colabcola import cli
from colabcola.lifecycle import Supervisor
from colabcola.lifecycle_auth import TokenLease, RenewalPump
from colabcola.lifecycle_io import atomic, LifecycleError, AllocationRejected
from colabcola.lifecycle_rest import RestGoogleProvider
from colabcola.lifecycle_worker import stop_reason, ExecutionWatchdog
from colabcola.packager import parser, package
import test_lifecycle as fixtures
from test_notebook import notebook_job


class TokenTests(unittest.TestCase):
    def test_three_hour_renewal_uses_short_tokens_and_never_returns_refresh_secret(self):
        now=[0]; calls=[]
        def load():
            calls.append(now[0])
            return {'access_token': 'token-'+str(now[0]), 'expires_at':now[0]+3600,
                    'refresh_token':'must-stay-host-only'}
        lease=TokenLease(load, clock=lambda:now[0])
        for tick in range(0,10801,60):
            now[0]=tick
            value=lease()
            self.assertGreater(value['expires_at'],tick+600)
            self.assertEqual(set(value),{'access_token','expires_at'})
        self.assertEqual(calls,[0,3000,6000,9000])

    def test_short_or_invalid_tokens_fail_without_exposing_values(self):
        for expiry in (100, float('nan'), float('inf'), '3600'):
            with self.subTest(expiry=expiry):
                lease=TokenLease(lambda:{'access_token':'secret','expires_at':expiry},clock=lambda:0)
                with self.assertRaises(LifecycleError) as caught:lease()
                self.assertNotIn('secret',str(caught.exception))

    def test_renewal_failure_does_not_extend_old_lease(self):
        now=[0]
        loader=mock.Mock(return_value={'access_token':'first','expires_at':3600})
        lease=TokenLease(loader,clock=lambda:now[0]);lease()
        now[0]=3000;loader.side_effect=RuntimeError('private-helper-output')
        with self.assertRaises(RuntimeError):lease()
        loader.side_effect=None;loader.return_value={'access_token':'next','expires_at':6600}
        self.assertEqual(lease()['expires_at'],6600)

    def test_worker_renews_and_stops_before_expiry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config={'deadline':10800,'authorization_policy':'renewable-v1'}
            atomic(root/'drive-token.json',{'access_token':'first','expires_at':3600})
            self.assertIsNone(stop_reason(config,root,3400))
            self.assertEqual(stop_reason(config,root,3480),'drive_authorization_expiring')
            atomic(root/'drive-token.json',{'access_token':'second','expires_at':7200})
            self.assertIsNone(stop_reason(config,root,3480))
            self.assertEqual(stop_reason(config,root,10800),'wall_budget_exhausted')
            (root/'drive-token.json').write_text('broken')
            self.assertEqual(stop_reason(config,root,4000),'drive_authorization_unavailable')

    def test_watchdog_kills_independently_of_worker_transfer_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            process=mock.Mock(pid=123);process.poll.return_value=None
            guard=ExecutionWatchdog({'deadline':0},Path(tmp),process)
            with mock.patch('colabcola.lifecycle_worker.os.killpg', create=True) as kill, mock.patch('colabcola.lifecycle_worker.signal.SIGKILL', 9, create=True):
                guard.start();guard.thread.join(timeout=3);guard.stop()
                kill.assert_called_once()
            self.assertEqual(json.loads((Path(tmp)/'watchdog.json').read_text())['reason'],'wall_budget_exhausted')

    def test_malformed_runtime_lease_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config={'deadline':10800,'authorization_policy':'renewable-v1'}
            for value in ([], None, {'expires_at':float('inf')}, {'expires_at':5000},
                          {'expires_at':5000,'access_token':''}):
                (root/'drive-token.json').write_text(json.dumps(value))
                self.assertIsNotNone(stop_reason(config,root,1000))
            self.assertIsNone(stop_reason({'deadline':10800},root,1000))

    @unittest.skipUnless(os.name=='posix', 'runtime process groups require Linux/macOS')
    def test_real_process_group_stops_after_lease_rotation_then_expiry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);now=time.time()
            atomic(root/'drive-token.json',{'access_token':'initial','expires_at':now+120.5})
            process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
            guard=ExecutionWatchdog({'deadline':now+30,'authorization_policy':'renewable-v1'},root,process)
            try:
                guard.start()
                atomic(root/'drive-token.json',{'access_token':'renewed','expires_at':now+3600})
                time.sleep(1.2)
                self.assertIsNone(process.poll())
                atomic(root/'drive-token.json',{'access_token':'expired','expires_at':time.time()+119})
                self.assertEqual(process.wait(timeout=5),-9)
                guard.thread.join(timeout=2)
                self.assertEqual(json.loads((root/'watchdog.json').read_text())['reason'],'drive_authorization_expiring')
            finally:
                guard.stop()
                if process.poll() is None:process.kill()
                process.wait()

    def test_renewal_runs_without_supervisor_tick(self):
        engine=mock.Mock(state={'phase':'monitoring','attempts':[{'number':1,'endpoint':'owned'}]},plan={})
        provider=mock.Mock();event=threading.Event();provider.refresh_token.side_effect=lambda *a:event.set()
        with RenewalPump(engine,provider,lambda:{'access_token':'private','expires_at':3600},interval=.01) as pump:
            self.assertTrue(event.wait(2))
        self.assertNotIn('private',json.dumps(pump.health))
        self.assertEqual(provider.refresh_token.call_args.args[0],'owned')


class FakeHTTP:
    def __init__(self):self.calls=[];self.queue=[]
    def add(self,status,value):self.queue.append((status,value))
    def request(self,method,url,**kwargs):
        self.calls.append((method,url,kwargs))
        item=self.queue.pop(0)
        if isinstance(item,Exception):raise item
        status,value=item
        return mock.Mock(status_code=status,json=lambda:value)


class RestTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.id=str(uuid.uuid4());self.http=FakeHTTP()
        self.provider=RestGoogleProvider({'directory':self.tmp.name,'runtime_version':'2026.07','high_mem':True},self.http)
        self.name=self.provider.name(self.id)
        self.value={'name':self.name,'runtimeSpec':self.provider.expected('A100'),'version':'2026.07',
                    'connectionInfo':{'token':'secret','url':'https://example.invalid','expireTime':'2099-01-01T00:00:00Z'}}

    def test_dated_highmem_create_reconciles_asynchronous_operation(self):
        self.http.add(404,{})
        self.http.add(200,{'name':'operations/op-1','done':False})
        self.http.add(404,{})
        with self.assertRaisesRegex(LifecycleError,'pending'):self.provider.allocate(self.id,'A100')
        post=self.http.calls[1]
        self.assertEqual(post[2]['json'],{'runtimeSpec':self.provider.expected('A100'),'version':'2026.07'})
        self.assertEqual(post[2]['params']['requestId'],self.id)
        self.http.add(200,self.value)
        self.assertEqual(self.provider.allocate(self.id,'A100'),self.name)
        self.assertEqual(sum(c[0]=='POST' for c in self.http.calls),1)
        self.assertNotIn('secret',self.provider.record_path(self.id).read_text())

    def test_timeout_reuses_request_id_and_cleanup_never_assumes_absence(self):
        self.http.add(404,{});self.http.queue.append(TimeoutError('sensitive'))
        with self.assertRaises(LifecycleError):self.provider.allocate(self.id,'A100')
        self.http.add(404,{})
        with self.assertRaisesRegex(LifecycleError,'unresolved'):self.provider.reconcile(self.id,'A100')
        self.http.add(404,{})
        self.http.add(200,{'name':'operations/one','done':True,'response':self.value})
        self.http.add(200,self.value)
        self.assertEqual(self.provider.allocate(self.id,'A100'),self.name)
        posts=[c for c in self.http.calls if c[0]=='POST']
        self.assertEqual(posts[0][2]['params'],posts[1][2]['params'])

    def test_rejected_operation_stops_and_cleanup_has_no_runtime(self):
        self.http.add(404,{});self.http.add(200,{'name':'operations/refused','done':True,'error':{'code':9,'message':'sensitive'}})
        with self.assertRaises(AllocationRejected) as caught:self.provider.allocate(self.id,'A100')
        self.assertNotIn('sensitive',str(caught.exception))
        self.http.add(404,{})
        self.assertIsNone(self.provider.reconcile(self.id,'A100'))

    def test_wrong_version_is_rejected_but_owned_runtime_can_be_released(self):
        self.http.add(200,{**self.value,'version':'latest'})
        with self.assertRaises(AllocationRejected):self.provider.allocate(self.id,'A100')
        self.http.add(200,self.value)
        self.assertEqual(self.provider.reconcile(self.id,'A100'),self.name)
        self.http.add(200,self.value);self.http.add(200,{});self.http.add(404,{})
        self.provider.release(self.name)
        self.assertEqual(self.http.calls[-2][0],'DELETE')
        self.assertTrue(self.http.calls[-2][1].endswith(self.name))

    def test_pending_operation_error_is_reconciled_after_restart(self):
        atomic(self.provider.record_path(self.id),{'name':self.name,'state':'pending','operation':'operations/pending'})
        self.http.add(404,{})
        self.http.add(200,{'done':True,'error':{'code':9}})
        self.assertIsNone(self.provider.reconcile(self.id,'A100'))

    def test_api_denial_never_falls_back_to_latest_or_logs_response(self):
        self.http.add(404,{});self.http.add(403,{'token':'private'})
        with self.assertRaisesRegex(AllocationRejected,'403'):self.provider.allocate(self.id,'A100')
        self.assertEqual(len(self.http.calls),2)
        self.assertNotIn('private',self.provider.record_path(self.id).read_text())

    def test_permission_failure_before_any_create_stops_without_pending_cleanup(self):
        self.http.add(403,{'message':'private'})
        with self.assertRaises(AllocationRejected):self.provider.allocate(self.id,'A100')
        self.assertIsNone(self.provider.reconcile(self.id,'A100'))
        self.assertEqual([c[0] for c in self.http.calls],['GET'])

    def test_replay_permission_failure_does_not_erase_uncertain_creation(self):
        self.http.add(404,{});self.http.queue.append(TimeoutError())
        with self.assertRaises(LifecycleError):self.provider.allocate(self.id,'A100')
        self.http.add(404,{});self.http.add(403,{})
        with self.assertRaises(LifecycleError) as caught:self.provider.allocate(self.id,'A100')
        self.assertNotIsInstance(caught.exception,AllocationRejected)
        self.http.add(404,{})
        with self.assertRaisesRegex(LifecycleError,'unresolved'):self.provider.reconcile(self.id,'A100')
        self.http.add(200,self.value)
        self.assertEqual(self.provider.reconcile(self.id,'A100'),self.name)

    def test_already_exists_without_visible_resource_never_confirms_cleanup(self):
        self.http.add(404,{})
        self.http.add(200,{'done':True,'error':{'code':6}})
        self.http.add(404,{})
        with self.assertRaisesRegex(LifecycleError,'not yet reconciled'):self.provider.allocate(self.id,'A100')
        self.http.add(404,{})
        with self.assertRaisesRegex(LifecycleError,'unresolved'):self.provider.reconcile(self.id,'A100')

    def test_connection_token_is_fetched_fresh_and_not_persisted(self):
        self.http.add(200,self.value)
        self.assertEqual(self.provider._session(self.name).token,'secret')
        self.http.add(200,{**self.value,'connectionInfo':{**self.value['connectionInfo'],'token':'renewed'}})
        self.assertEqual(self.provider._session(self.name).token,'renewed')
        self.assertFalse(self.provider.root.exists())


class PreparationTests(unittest.TestCase):
    def test_longcat_shaped_preparation_is_offline_and_preserves_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);script=root/'work.py';script.write_text("print('fixture only')")
            value=notebook_job(script,accelerator='gpu')
            value['runtime']['version']='2026.07'
            value['python']['requires_python']='>=3.12,<3.13'
            value['resources']['minimum_gpu_memory_bytes']=60000000000
            value['run']['timeout_seconds']=8900
            value['managed_recovery']={'checkpoint_pattern':'outputs/disabled-*.json','log_patterns':[]}
            job=root/'fixture.colajob';atomic(job,value)
            atomic(root/'helper.json',['must-not-execute'])
            config={'drive_parent':'placeholder','token_command':'helper.json','gpu':'A100','max_attempts':1,
                    'max_seconds':10800,'provider_backend':'rest','high_mem':True}
            atomic(job.with_suffix('.run.json'),config)
            with mock.patch('colabcola.managed.supervise',side_effect=AssertionError('no launch')),mock.patch('sys.stdout',new_callable=io.StringIO):
                self.assertEqual(cli.main(['--prepare-only',str(job)]),0)
                plan=json.loads((job.with_suffix('.run')/'plan.json').read_text())
                self.assertEqual(plan['runtime_version'],'2026.07')
                self.assertEqual(plan['max_seconds'],10800)
                self.assertEqual(plan['authorization_policy'],'renewable-v1')
                self.assertEqual(json.loads(job.read_text()),value)
                config['high_mem']=False;atomic(job.with_suffix('.run.json'),config)
                self.assertEqual(cli.main(['--prepare-only',str(job)]),2)

    def test_packager_accepts_explicit_date_and_long_budget_but_keeps_caps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);script=root/'work.py';script.write_text('pass')
            args=parser().parse_args(['-o',str(root/'job.colajob'),'--runtime-version','2026.07',
                '--provider-backend','rest','--gpu','A100','--minimum-gpu-memory-bytes','60000000000',
                '--max-seconds','10800','--timeout','8900',str(script)])
            package(args)
            value=json.loads((root/'job.colajob').read_text())
            self.assertEqual(value['runtime']['version'],'2026.07')
            self.assertEqual(value['resources']['minimum_gpu_memory_bytes'],60000000000)
            args.output=root/'too-long.colajob';args.max_seconds=86401
            with self.assertRaisesRegex(ValueError,'60-86400'):package(args)


class LongSupervisorTests(unittest.TestCase):
    setUp = fixtures.LifecycleTests.setUp
    engine = fixtures.LifecycleTests.engine
    running = fixtures.LifecycleTests.running
    reload = fixtures.LifecycleTests.reload
    def test_long_deadline_and_cleanup_do_not_require_drive_token(self):
        engine=self.running();engine.plan['max_seconds']=10800;engine.state['deadline']=11800
        engine.plan['authorization_policy']='renewable-v1'
        engine.state['drive_authorization']={'expires_at':6000}
        engine.token=mock.Mock(side_effect=LifecycleError('revoked'))
        self.now=5900;engine.advance()
        self.assertEqual(engine.state['phase'],'failed')
        self.assertEqual(engine.state['reason'],'drive_authorization_expiring')
        self.assertEqual(len(self.provider.releases),1)
        engine.token.assert_not_called()

    def test_three_hour_shared_deadline_survives_reload(self):
        engine=self.running();engine.state['deadline']=11800;engine.save()
        self.now=11799;engine=self.reload();engine.advance()
        self.assertEqual(engine.state['phase'],'monitoring')
        self.now=11800;engine.advance()
        self.assertEqual(engine.state['phase'],'budget_exhausted')
        self.assertEqual(len(self.provider.allocations),1)

    def test_permanent_allocation_rejection_cleans_up_owned_mismatch(self):
        engine=self.engine();engine.advance()
        self.provider.allocate=mock.Mock(side_effect=AllocationRejected('wrong runtime version'))
        self.provider.reconcile=mock.Mock(return_value='owned-mismatch')
        engine.advance()
        self.assertEqual(engine.state['phase'],'failed')
        self.assertEqual(self.provider.releases,['owned-mismatch'])
    def test_rejected_allocation_cleanup_outage_remains_pending(self):
        engine=self.engine();engine.advance()
        self.provider.allocate=mock.Mock(side_effect=AllocationRejected('wrong runtime version'))
        self.provider.reconcile=mock.Mock(side_effect=LifecycleError('offline'))
        engine.advance()
        self.assertEqual(engine.state['phase'],'cleanup')
        self.assertFalse(engine.state['attempts'][-1].get('released',False))
        self.provider.reconcile=mock.Mock(return_value='owned')
        engine.advance()
        self.assertEqual(engine.state['phase'],'failed')
        self.assertEqual(self.provider.releases,['owned'])



if __name__=='__main__':unittest.main()
