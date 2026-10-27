"""Linux-only real-process worker recovery with an injected filesystem transport."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from colabcola.lifecycle import verify_final
from colabcola.lifecycle_io import atomic, digest, encoded, publish, snapshots
from colabcola.compiler import compile_invocation
from colabcola.invocation import load_job_invocation
from test_notebook import notebook_job


class FileStore:
    def __init__(self, directory):
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
    def put(self,kind,name,body):
        key=digest(body)
        (self.directory/(key+".bin")).write_bytes(body)
        atomic(self.directory/(key+".json"),{"kind":kind,"size":len(body)})
        return {"id":key,"sha256":key,"bytes":len(body)}
    def get(self,ref):
        body=(self.directory/(ref["id"]+".bin")).read_bytes()
        if len(body)!=ref["bytes"] or digest(body)!=ref["sha256"]:
            raise ValueError("corrupt fixture")
        return body
    def list(self,kind):
        records=[]
        for p in self.directory.glob("*.json"):
            value=json.loads(p.read_text())
            if value["kind"]==kind:
                records.append({"id":p.stem,"size":value["size"],"appProperties":{"cc_sha":p.stem}})
        return records


@unittest.skipUnless(sys.platform=="linux","real runtime process test requires Linux")
class WorkerIntegrationTests(unittest.TestCase):
    def test_runtime_loss_recovers_in_distinct_process_and_publishes_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/"train.py"
            source.write_text('''import os,json,time,hashlib,pathlib,shutil
out=pathlib.Path("outputs"); out.mkdir()
resume=os.environ.get("COLABCOLA_RESUME_MANIFEST")
start=json.loads(pathlib.Path(resume).read_text())["step"] if resume else 0
(out/"fits").mkdir()
artifact=out/"fits/completed.zip"
if resume:
    shutil.copyfile(pathlib.Path(resume).parent/"fits/completed.zip",artifact)
else:
    artifact.write_bytes(b"saved-independent-fit")
catalog={"fits/completed.zip":{"bytes":artifact.stat().st_size,"sha256":hashlib.sha256(artifact.read_bytes()).hexdigest()}}
for step in range(start+1,21):
    state=str(step).encode(); name=f"state-{step}.bin"
    (out/name).write_bytes(state)
    value={"schema":2,"step":step,"file":name,"bytes":len(state),"sha256":hashlib.sha256(state).hexdigest(),"artifacts":catalog}
    temp=out/f"state-{step}.tmp"; temp.write_text(json.dumps(value))
    temp.replace(out/f"state-{step}.json")
    print(step,flush=True); time.sleep(.1)
(out/"result.txt").write_text(json.dumps({"restored":start,"step":20,"completed_artifact":artifact.read_text()}))
''')
            mapping=notebook_job(source)
            mapping["entrypoint"]["argv"]=[]
            mapping["outputs"]=[{"pattern":"outputs/result.txt","required":True,"maximum_bytes":4096}]
            mapping["managed_recovery"]={"checkpoint_pattern":"outputs/state-*.json","log_patterns":[]}
            job=root/"test.colajob"; job.write_text(json.dumps(mapping))
            compiled=compile_invocation(load_job_invocation(job),root/"notebook.ipynb")
            store=FileStore(root/"store")
            identity={"run_id":"a"*32,"job_id":compiled.job_id}
            import colabcola.lifecycle_worker as worker
            worker_source=Path(worker.__file__)
            processes=[]
            children=[]
            def start(number,resume=None):
                control=root/f"control-{number}"; control.mkdir()
                for name,path in (("notebook.ipynb",root/"notebook.ipynb"),
                                  ("lifecycle_worker.py",worker_source),
                                  ("lifecycle_io.py",worker_source.with_name("lifecycle_io.py"))):
                    (control/name).write_bytes(path.read_bytes())
                config={**identity,"directory":str(control),"content_root":str(root/f"runtime-{number}"),
                        "files":{p.name:digest(p.read_bytes()) for p in control.iterdir()},
                        "parent":"fixture","attempt":number,"deadline":time.time()+35,"resume":resume,
                        "checkpoint_pattern":"outputs/state-*.json","log_patterns":[],"poll_seconds":.1}
                atomic(control/"worker.json",config)
                script=("import sys;sys.path.insert(0,"+repr(str(Path(__file__).parent))+");"
                        "from test_lifecycle_worker import FileStore;"
                        "import colabcola.lifecycle_worker as w;"
                        "w.DriveStore=lambda *args: FileStore("+repr(str(root/"store"))+");"
                        "w.run("+repr(str(control/"worker.json"))+")")
                proc=subprocess.Popen([sys.executable,"-c",script],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                processes.append(proc)
                return proc,control
            try:
                first,control=start(1)
                limit=time.time()+20
                checkpoint=None
                while time.time()<limit:
                    values=snapshots(store,identity,"checkpoint")
                    if values:
                        checkpoint=values[-1]; break
                    if first.poll() is not None:
                        self.fail(first.communicate()[1].decode())
                    time.sleep(.05)
                self.assertIsNotNone(checkpoint)
                self.assertLess(checkpoint["commit"]["step"],20)
                status=json.loads((control/"status.json").read_text())
                pid=status["execution_pid"]; children.append(pid)
                os.killpg(pid,signal.SIGKILL)
                first.kill(); first.communicate(timeout=5)
                second,control2=start(2,checkpoint)
                stdout,stderr=second.communicate(timeout=35)
                self.assertEqual(second.returncode,0,stderr.decode())
                status=json.loads((control2/"status.json").read_text())
                self.assertEqual(status["state"],"completed",status)
                self.assertGreater(status["restored_step"],0)
                final=snapshots(store,identity,"final")[-1]
                from colabcola.lifecycle_io import recover
                restored=recover(store,final,root/"recovered")
                result=verify_final(restored,compiled.job_id)
                self.assertEqual(result["terminal_state"],"success")
                self.assertIn("20", (restored/"runtime/stdout.log").read_text())
                self.assertTrue((restored/"runtime/stderr.log").is_file())
                self.assertNotEqual(first.pid,second.pid)
                import zipfile
                with zipfile.ZipFile(restored/"artifacts.zip") as archive:
                    value=json.loads(archive.read("outputs/result.txt"))
                self.assertEqual(value["step"],20)
                self.assertEqual(value["completed_artifact"],"saved-independent-fit")
                self.assertEqual(value["restored"],checkpoint["commit"]["step"])
            finally:
                for proc in processes:
                    if proc.poll() is None:
                        proc.kill(); proc.communicate(timeout=5)
                for control in root.glob("control-*"):
                    status_path = control / "status.json"
                    if status_path.exists():
                        pid = json.loads(status_path.read_text()).get("execution_pid")
                        if pid:
                            try: os.killpg(pid, signal.SIGKILL)
                            except ProcessLookupError: pass
