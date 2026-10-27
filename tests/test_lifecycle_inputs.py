import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from colabcola.lifecycle_io import digest, encoded, atomic, verified_inputs, LifecycleError
from colabcola.lifecycle import Supervisor
from colabcola.lifecycle_worker import acquire_inputs
from colabcola.lifecycle_provider import GoogleProvider
from test_lifecycle import MemoryStore, Provider

class LargeInputTests(unittest.TestCase):
    def test_large_input_roundtrip_and_small_bootstrap(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);local=root/"local";local.mkdir();remote=root/"remote";remote.mkdir()
            # Exceeds both the former32MiB input and64MiB single-object/read limits.
            with (local/"notebook.ipynb").open("wb") as f:
                for _ in range(65): f.write(b"x"*1024*1024)
            for name in ("lifecycle_worker.py","lifecycle_io.py"):(local/name).write_bytes(b"fixture")
            hashes={p.name:digest(p.read_bytes()) for p in local.iterdir()}
            plan=dict(run_id="a"*32,job_id="b"*64,directory=str(local),files=hashes,
                input_transport="drive_chunks",gpu="T4",max_attempts=1,max_seconds=900,
                fault_after_checkpoint=False,parent="parent",checkpoint_pattern="outputs/state*.json",log_patterns=[])
            atomic(local/"plan.json",plan);atomic(local/"state.json",dict(phase="prepared",attempts=[],events=[]))
            store=MemoryStore();provider=Provider()
            engine=Supervisor(local,provider,store,lambda:{"access_token":"fixture"},clock=lambda:1000.)
            engine.advance();snapshot=engine.state["input_commit"]
            self.assertEqual(snapshot["commit"]["schema"],2)
            self.assertTrue(all(len(body)<=8*1024*1024 for kind,name,body in store.objects.values()))
            attempt=engine.state["attempts"][0]
            attempt.update(input_snapshot=snapshot,deadline=1900)
            google=object.__new__(GoogleProvider);google._session=Mock(return_value=object());google._execute=Mock();google._upload=Mock()
            google.boot("owned",plan,attempt,{"access_token":"fixture"})
            uploads={call.args[1].rsplit("/",1)[-1]:call.args[2] for call in google._upload.call_args_list}
            self.assertNotIn("notebook.ipynb",uploads)
            config=json.loads(uploads["worker.json"])
            for name in ("lifecycle_worker.py","lifecycle_io.py"):(remote/name).write_bytes(uploads[name])
            acquire_inputs(config,store,remote)
            verified_inputs(remote,hashes)
            self.assertEqual((remote/"notebook.ipynb").stat().st_size,65*1024*1024)
            wrong=copy.deepcopy(config);wrong["job_id"]="c"*64
            with self.assertRaises(LifecycleError):acquire_inputs(wrong,store,root/"wrong")
            bad=copy.deepcopy(config);bad["files"]["notebook.ipynb"]="0"*64
            with self.assertRaises(LifecycleError):acquire_inputs(bad,store,root/"bad")
            self.assertFalse((root/"bad").exists())

    def test_input_set_and_size_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.assertRaises(LifecycleError):verified_inputs(root,{"other":"0"*64})
            for n in ("notebook.ipynb","lifecycle_worker.py","lifecycle_io.py"):(root/n).write_bytes(b"x")
            hashes={p.name:digest(p.read_bytes()) for p in root.iterdir()}
            with (root/"notebook.ipynb").open("wb") as f:f.truncate(512*1024*1024+1)
            with self.assertRaises(LifecycleError):verified_inputs(root,hashes)

if __name__=="__main__":unittest.main()
