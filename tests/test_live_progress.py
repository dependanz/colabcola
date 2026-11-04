"""Local live-output and kernel cleanup regression tests; never allocate compute."""
import ast
import contextlib
import gc
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from colabcola.runtime import _EXECUTION_SOURCE
from colabcola.packager import _LAUNCHER
from colabcola.managed import Progress
from colabcola.lifecycle_kernel import safe_client_class, execute
from colabcola.lifecycle_io import LifecycleError


def drain():
    tree = ast.parse(_EXECUTION_SOURCE)
    ns = {}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
        and n.name in ('_colabcola_emit', '_colabcola_drain')], type_ignores=[]), '<runtime>', 'exec'), ns)
    return ns['_colabcola_drain']


class StreamingTests(unittest.TestCase):
    def test_module_sequence_logs_reach_cli_before_child_can_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'runtime').mkdir()
            (root/'launcher.py').write_text(_LAUNCHER, encoding='utf-8')
            (root/'commands.json').write_text(json.dumps({'commands':[['-m','train'],['-m','evaluate']], 'resume_arg':None}))
            (root/'train.py').write_text('import sys,time\nfrom pathlib import Path\nprint("training step 1")\nprint("\\rprogress 1/2\\r",file=sys.stderr)\nwhile not Path("continue").exists(): time.sleep(0.01)\nprint("training finished")\n')
            (root/'evaluate.py').write_text('print("evaluation finished")\n')
            env = dict(os.environ, PYTHONUNBUFFERED='1')
            p = subprocess.Popen([sys.executable, str(root/'launcher.py')], cwd=root, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            threads=[]; files=[]
            stream=io.StringIO(); reporter=Progress(root,stream)
            state={'logs':{'local_directory':str(root),'commit':{'attempt':1,'members':{'runtime/stdout.log':{},'runtime/stderr.log':{}}}}}
            try:
                for name, pipe in [('stdout',p.stdout),('stderr',p.stderr)]:
                    f=(root/f'runtime/{name}.log').open('wb');files.append(f)
                    t=threading.Thread(target=drain(),args=(pipe,io.StringIO(),f,True),daemon=True)
                    t.start();threads.append(t)
                deadline=time.monotonic()+8
                while time.monotonic()<deadline:
                    reporter(state)
                    if 'training step 1' in stream.getvalue() and 'progress 1/2' in stream.getvalue():break
                    time.sleep(0.02)
                self.assertIn('training step 1',stream.getvalue())
                self.assertIn('progress 1/2',stream.getvalue())
                self.assertIsNone(p.poll(), 'child must still be waiting for our handshake')
            finally:
                (root/'continue').touch()
                try:p.wait(timeout=8)
                except subprocess.TimeoutExpired:p.kill();p.wait()
                for t in threads:t.join(timeout=3)
                for f in files:f.close()
                p.stdout.close();p.stderr.close()
            self.assertEqual(p.returncode,0)
            self.assertIn(b'evaluation finished',(root/'runtime/stdout.log').read_bytes())
            reporter(state); reporter(state)
            self.assertEqual(stream.getvalue().count('training step 1'),1)
            self.assertEqual(stream.getvalue().count('evaluation finished'),1)

    def test_split_utf8_and_final_tail_exact_capture(self):
        class Pipe:
            def __init__(self):self.blocks=iter([b'\xe2',b'\x82\xac\rstep',b'\nlast\xe2',b''])
            def read1(self,n):return next(self.blocks)
        for capture in (True,False):
            display=io.StringIO();dest=io.BytesIO()
            drain()(Pipe(),display,dest,capture)
            self.assertEqual(display.getvalue(),'\u20ac\rstep\nlast\ufffd')
            self.assertEqual(dest.getvalue(),b'\xe2\x82\xac\rstep\nlast\xe2' if capture else b'')

    def test_snapshot_dedup_attempt_reset_redaction_bounds_and_final_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'runtime').mkdir();path=root/'runtime/stderr.log'
            out=io.StringIO();p=Progress(root,out)
            snap={'local_directory':str(root),'commit':{'attempt':1,'members':{'runtime/stderr.log':{}}}}
            path.write_bytes(b'one\npartial \xe2')
            p({'logs':snap});p({'logs':snap})
            path.write_bytes(b'one\npartial \xe2\x82\xac\rAuthorization: Bearer secret\naccess_token=hidden\n\x1b[31mend')
            p({'logs':snap});p({'final':snap});p({'final':snap})
            text=out.getvalue();self.assertEqual(text.count('one'),1);self.assertIn('partial \u20ac',text)
            self.assertNotIn('secret',text);self.assertNotIn('hidden',text);self.assertNotIn('\x1b',text)
            self.assertEqual(text.count('[31mend'),1)
            path.write_bytes(b'two\n');p({'logs':snap})
            snap['commit']['attempt']=2;p({'logs':snap})
            self.assertEqual(out.getvalue().count('two'),2)
            path.write_text('\n'.join(str(i) for i in range(100))+'\n');p({'logs':snap})
            self.assertIn('80 earlier updates omitted',out.getvalue())
            self.assertIn('99',out.getvalue())

    def test_root_oom_survives_long_launcher_tail_with_redaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'runtime').mkdir()
            lines=['[rank0]: torch.OutOfMemoryError: CUDA out of memory Bearer hidden']
            lines += ['launcher wrapper '+str(i) for i in range(100)]
            (root/'runtime/stderr.log').write_text('\n'.join(lines)+'\n')
            out=io.StringIO(); p=Progress(root,out)
            snapshot={'local_directory':str(root),'commit':{'attempt':1,'members':{'runtime/stderr.log':{}}}}
            p({'logs':snapshot}); p({'final':snapshot})
            self.assertIn('torch.OutOfMemoryError: CUDA out of memory',out.getvalue())
            self.assertNotIn('hidden',out.getvalue())
            self.assertIn('launcher wrapper 99',out.getvalue())
            self.assertEqual(out.getvalue().count('torch.OutOfMemoryError:'),1)
            self.assertLess(len(out.getvalue().splitlines()),30)

    def test_sequence_stops_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'launcher.py').write_text(_LAUNCHER)
            (root/'commands.json').write_text(json.dumps({'commands':[['-c','raise SystemExit(7)'],['-c','print("MUST NOT RUN")']]}))
            result=subprocess.run([sys.executable,str(root/'launcher.py')],capture_output=True,text=True)
            self.assertEqual(result.returncode,7);self.assertNotIn('MUST NOT RUN',result.stdout)


class KernelTests(unittest.TestCase):
    def test_partial_constructor_destructor_and_explicit_stop_are_safe(self):
        class Broken:
            def __init__(self,**kwargs):raise RuntimeError('construction')
            def stop(self,**kwargs):raise AssertionError('no manager exists')
        with self.assertNoLogs('colabcola.lifecycle_kernel',level='WARNING'):
            with self.assertRaisesRegex(RuntimeError,'construction'):safe_client_class(Broken)()
            gc.collect()

    def test_disconnect_idempotent_never_shuts_down_kernel(self):
        manager=SimpleNamespace(client=SimpleNamespace(kernel_socket=mock.Mock()))
        calls=[]
        class Base:
            def __init__(self,**kwargs):self._manager=manager;self._own_kernel=True
            def stop(self,**kwargs):calls.append(kwargs)
        client=safe_client_class(Base)();self.assertFalse(client._own_kernel)
        client.stop(shutdown_kernel=True);client.stop();del client;gc.collect()
        self.assertEqual(calls,[{'shutdown_kernel':False}]);manager.client.kernel_socket.close.assert_called_once()

    def run_fake(self, start_error=None, reply=None, close_error=None, execute_error=None):
        self.instances=[]
        outer=self
        class Client:
            def __init__(self,**kwargs):
                self._manager=SimpleNamespace(client=SimpleNamespace(kernel_socket=mock.Mock()))
                outer.instances.append(self)
            def start(self):
                if start_error:raise start_error
            def execute(self,*args,**kwargs):
                if execute_error:raise execute_error
                return reply
            def stop(self,**kwargs):
                if close_error:raise close_error
        module=SimpleNamespace(KernelClient=Client,JupyterSubprotocol=SimpleNamespace(DEFAULT='default'))
        with mock.patch('colabcola._vendor.jupyter_kernel_client',module,create=True),mock.patch('colabcola.lifecycle_kernel.time.sleep'):
            execute(SimpleNamespace(url='https://invalid',token='secret'),'pass','owned')

    def test_execution_failure_is_not_retried_and_is_redacted(self):
        with self.assertRaisesRegex(LifecycleError,'execution failed: RuntimeError'):
            self.run_fake(execute_error=RuntimeError('secret'))
        self.assertEqual(len(self.instances),1)
        for reply in (None,{'status':'error'},{'status':'ok','outputs':[{'output_type':'error'}]}):
            with self.assertRaisesRegex(LifecycleError,'bootstrap failed'):self.run_fake(reply=reply)

    def test_cleanup_failure_is_visible(self):
        with self.assertRaisesRegex(LifecycleError,'cleanup failed: RuntimeError'):
            self.run_fake(reply={'status':'ok'},close_error=RuntimeError('secret'))
        for client in self.instances:client._manager=None  # Explicitly failed cleanup was already asserted.

    def test_execution_and_cleanup_failures_both_remain_visible(self):
        with self.assertRaisesRegex(LifecycleError,'bootstrap failed; runtime kernel cleanup failed: RuntimeError'):
            self.run_fake(reply={'status':'error'},close_error=RuntimeError('secret'))
        for client in self.instances:client._manager=None

    def test_connection_timeouts_retry_with_cleanup(self):
        import requests
        with self.assertRaisesRegex(LifecycleError,'connection failed: ReadTimeout'):
            self.run_fake(start_error=requests.exceptions.ReadTimeout('secret'))
        self.assertEqual(len(self.instances),3)
        self.assertTrue(all(c._manager is None for c in self.instances))

    def test_stop_before_kernel_exists_never_creates_a_connection(self):
        class Manager:
            has_kernel = False
            @property
            def client(self):
                raise AssertionError('cleanup must not create a client')
        class Base:
            def __init__(self, **kwargs): self._manager = Manager()
            def stop(self, **kwargs):
                if self._manager.has_kernel: self._manager.client.stop_channels()
        client = safe_client_class(Base)()
        with self.assertNoLogs('colabcola.lifecycle_kernel', level='WARNING'):
            client.stop(); client.stop(); del client; gc.collect()

    def test_connection_and_cleanup_failures_both_remain_visible(self):
        import requests
        with self.assertRaisesRegex(LifecycleError,
                'connection failed: ReadTimeout; runtime kernel cleanup failed: RuntimeError'):
            self.run_fake(start_error=requests.exceptions.ReadTimeout('secret'),
                          close_error=RuntimeError('secret'))
        self.assertEqual(len(self.instances), 1)
        for client in self.instances: client._manager = None

    def test_installed_jupyter_start_timeout_without_kernel_cleans_up(self):
        try:
            from colabcola._vendor import jupyter_kernel_client
            from colabcola._vendor.jupyter_kernel_client.manager import KernelHttpManager
        except ImportError:
            self.skipTest('optional installed Jupyter dependency unavailable')
        import requests
        # Real installed manager/client, no remote traffic or kernel creation.
        with mock.patch.object(KernelHttpManager, 'start_kernel',
                side_effect=requests.exceptions.ReadTimeout('secret')) as start, \
                mock.patch.object(KernelHttpManager, 'client', new_callable=mock.PropertyMock) as factory, \
                mock.patch('colabcola.lifecycle_kernel.time.sleep'), \
                self.assertNoLogs('colabcola.lifecycle_kernel', level='WARNING'):
            factory.side_effect = AssertionError('cleanup must not request an unstarted client')
            with self.assertRaisesRegex(LifecycleError, 'connection failed: ReadTimeout'):
                execute(SimpleNamespace(url='https://invalid', token='secret'), 'pass', None)
            self.assertEqual(start.call_count, 3)
            factory.assert_not_called()
            gc.collect()

    def test_installed_jupyter_constructor_failure_no_missing_manager_warning(self):
        try:
            from colabcola._vendor import jupyter_kernel_client as jkc
        except ImportError:
            self.skipTest('optional installed Jupyter dependency unavailable')
        base=getattr(jkc,'KernelClient',getattr(jkc,'JupyterKernelClient',None))
        class BrokenManager:
            def __init__(self,**kwargs):raise RuntimeError('simulated manager failure')
        with mock.patch.object(base,'kernel_manager_class',BrokenManager):
            with self.assertNoLogs('colabcola.lifecycle_kernel',level='WARNING'):
                with self.assertRaisesRegex(RuntimeError,'simulated manager failure'):
                    safe_client_class(base)(server_url='https://invalid',token='secret')
                gc.collect()


if __name__=='__main__':unittest.main()
