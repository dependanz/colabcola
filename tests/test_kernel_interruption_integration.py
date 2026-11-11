"""Real local Jupyter interruption acceptance for issue #18 (no Colab service)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from colabcola.compiler import compile_invocation
from colabcola.invocation import load_job_invocation
from colabcola.lifecycle import Supervisor
from colabcola.lifecycle_io import atomic, digest
from colabcola.lifecycle_kernel import WorkloadExecution, workload_code
from test_lifecycle import MemoryStore, Provider
from test_notebook import notebook_job


@unittest.skipUnless(
    sys.platform == "linux" and importlib.util.find_spec("ipykernel"),
    "real Linux IPC kernel and ipykernel required",
)
class InterruptedDownloadKernelTests(unittest.TestCase):
    def test_interrupt_during_download_journals_and_cleans_up_in_one_tick(self):
        from jupyter_client import KernelManager
        import colabcola.lifecycle_worker as real_worker

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control = root / "control"
            control.mkdir()
            payload = b"z" * (4 * 1024 * 1024)
            source = root / "main.py"
            source.write_text("print('should not run after interrupt')\n")
            mapping = notebook_job(source)
            mapping["entrypoint"]["argv"] = []
            mapping["remote_inputs"] = [{
                "target": "weights/large.bin",
                "url": "https://models.example.invalid/large.bin",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }]
            mapping["managed_recovery"] = {
                "checkpoint_pattern": "outputs/state-*.json",
                "log_patterns": [],
            }
            job = root / "job.colajob"
            job.write_text(json.dumps(mapping))
            compiled = compile_invocation(
                load_job_invocation(job), control / "notebook.ipynb")

            # The kernel runs the tracked worker; that worker spawns a separate
            # execution subprocess which streams a synthetic remote input.
            # No network, Google Drive account, or GPU is involved.
            wrapper = """import sys,time,urllib.request
from pathlib import Path
sys.path.insert(0, __TESTS__)
from test_lifecycle_worker import FileStore
import colabcola.lifecycle_worker as w
w.DriveStore=lambda *args:FileStore(__STORE__)
if len(sys.argv)==3 and sys.argv[2]=='--execute':
 class SlowResponse:
  status=200
  def __init__(self): self.remaining=4*1024*1024;self.started=False
  def __enter__(self): return self
  def __exit__(self,*args): return False
  def read(self,n=-1):
   if self.remaining==0: return b''
   if not self.started:
    self.started=True
    Path(__MARKER__).write_text('streaming')
   time.sleep(0.04)
   size=min(32768,self.remaining,n if n>0 else 32768)
   self.remaining-=size
   return b'z'*size
 urllib.request.urlopen=lambda *args,**kwargs:SlowResponse()
 w.execute_notebook(sys.argv[1])
else:
 w.run(sys.argv[1])
"""
            wrapper = (wrapper.replace("__TESTS__", repr(str(Path(__file__).parent)))
                      .replace("__STORE__", repr(str(root / "store")))
                      .replace("__MARKER__", repr(str(control / "downloading"))))
            (control / "lifecycle_worker.py").write_text(wrapper)
            (control / "lifecycle_io.py").write_bytes(
                Path(real_worker.__file__).with_name("lifecycle_io.py").read_bytes())

            run_id = "a" * 32
            config = {
                "run_id": run_id, "job_id": compiled.job_id,
                "directory": str(control), "content_root": str(root / "runtime"),
                "files": {p.name: digest(p.read_bytes()) for p in (
                    control / "notebook.ipynb",
                    control / "lifecycle_worker.py",
                    control / "lifecycle_io.py",
                )},
                "parent": "fixture", "attempt": 1,
                "deadline": time.time() + 80, "resume": None,
                "checkpoint_pattern": "outputs/state-*.json",
                "log_patterns": [], "poll_seconds": 0.05,
            }
            atomic(control / "worker.json", config)
            km = KernelManager(transport="ipc", connection_file=str(root / "kernel.json"))
            km.start_kernel(
                cwd=tmp,
                env={**os.environ,
                     "PYTHONPATH": str(Path(real_worker.__file__).parents[1])
                     + os.pathsep + str(Path(__file__).parent)},
            )
            client = km.blocking_client()
            client.start_channels()
            client.wait_for_ready(timeout=30)
            child_pid = None
            try:
                client.connection_ready = mock.Mock()
                client.connection_ready.is_set.return_value = True
                handle = WorkloadExecution.__new__(WorkloadExecution)
                handle.client = SimpleNamespace(
                    _manager=SimpleNamespace(client=client),
                    stop=lambda: client.stop_channels())
                handle.kernel_id = "local"
                handle.request_id = None
                handle.reply_error_type = None
                handle.state = "attached"
                identity = {"run_id": run_id, "job_id": compiled.job_id,
                            "attempt": 1, "kernel_id": "local"}
                handle.submit(workload_code(str(control), identity))

                limit = time.monotonic() + 25
                status = None
                while time.monotonic() < limit:
                    path = control / "status.json"
                    if path.exists():
                        try:
                            status = json.loads(path.read_text())
                        except (OSError, ValueError):
                            pass
                    if ((control / "downloading").is_file()
                            and status is not None and status.get("execution_pid")):
                        break
                    time.sleep(0.04)
                self.assertTrue((control / "downloading").exists(),
                                "actual child did not enter synthetic download")
                self.assertIsNotNone(status)
                child_pid = status["execution_pid"]
                self.assertEqual(json.loads((control / "kernel-execution.json").read_text())
                                 ["state"], "running")

                # Interrupt the *actual* kernel process during the child stream.
                km.interrupt_kernel()
                receipt = None
                request_state = None
                limit = time.monotonic() + 20
                while time.monotonic() < limit:
                    path = control / "kernel-execution.json"
                    if path.exists():
                        try:
                            receipt = json.loads(path.read_text())
                        except (OSError, ValueError):
                            pass
                    request_state = handle.poll()["request_state"]
                    if receipt and receipt.get("state") == "failed":
                        break
                    time.sleep(0.05)
                self.assertIsNotNone(receipt)
                self.assertEqual(receipt["state"], "failed", receipt)
                self.assertEqual(receipt["error_type"], "KeyboardInterrupt")
                self.assertIsInstance(receipt["finished_at"], float)
                self.assertGreater(receipt["finished_at"], receipt["started_at"])
                self.assertIn(request_state, ("failed", "busy", "submitted"))

                # The execution child and its entire owned process group are gone.
                limit = time.monotonic() + 5
                while time.monotonic() < limit:
                    try:
                        os.killpg(child_pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.05)
                else:
                    self.fail("execution child process group survived kernel interrupt")

                # Feed the actual durable worker evidence into one supervisor tick.
                host = root / "supervisor"
                host.mkdir()
                started = time.time()
                plan = {
                    "run_id": run_id, "job_id": compiled.job_id,
                    "gpu": "NONE", "execution_mode": "kernel-v1",
                    "max_attempts": 1, "max_seconds": 80,
                    "fault_after_checkpoint": False,
                }
                atomic(host / "plan.json", plan)
                atomic(host / "state.json", {
                    "phase": "monitoring", "deadline": started + 70,
                    "attempts": [{
                        "number": 1, "allocation_id": "alloc-local",
                        "endpoint": "vm-1", "allocated_at": started,
                        "bootstrapped_at": started,
                    }],
                    "events": [],
                })
                provider = Provider()
                provider.allocations["alloc-local"] = "vm-1"
                provider.active.add("vm-1")

                def observed(*_):
                    durable = json.loads((control / "kernel-execution.json").read_text())
                    worker_status = json.loads((control / "status.json").read_text())
                    return {
                        "state": "running", "attempt": 1,
                        "heartbeat_at": worker_status.get("heartbeat_at"),
                        "kernel_execution": {
                            "request_state": handle.poll()["request_state"],
                            "worker_execution_state": durable["state"],
                            "worker_error_type": durable["error_type"],
                            "worker_finished_at": durable["finished_at"],
                        },
                    }
                provider.observe = observed
                engine = Supervisor(host, provider, MemoryStore(),
                                    lambda: {"access_token": "fixture"})
                engine.advance()  # One observed failure, one cleanup tick.
                self.assertEqual(engine.state["phase"], "failed")
                self.assertEqual(engine.state["reason"],
                                 "worker_failed_attempt_budget_exhausted")
                self.assertEqual(provider.releases, ["vm-1"])
                self.assertEqual(provider.active, set())
                self.assertEqual(len(provider.allocations), 1)
                failure = engine.state["last_failure"]
                self.assertEqual(failure["error_type"], "KeyboardInterrupt")
                self.assertEqual(failure["finished_at"], receipt["finished_at"])
                self.assertLessEqual(failure["observed_at"] - receipt["finished_at"], 20)
                self.assertEqual(len([x for x in engine.state["events"]
                                      if x["event"] == "worker_failed"]), 1)
                self.assertNotIn("https://", json.dumps(engine.state))
            finally:
                client.stop_channels()
                try:
                    km.shutdown_kernel(now=True)
                finally:
                    if child_pid is not None:
                        try:
                            os.killpg(child_pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass


if __name__ == "__main__":
    unittest.main()
