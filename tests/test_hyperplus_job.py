import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

source = Path(__file__).resolve().parents[1] / "examples/hyperplus_image_fit/make_job.py"
spec = importlib.util.spec_from_file_location("hyperplus_job", source)
maker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maker)

class HyperplusJobTests(unittest.TestCase):
    def generate(self, runtime):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            wheel=root/"hyperplus.whl"; wheel.write_bytes(b"fixture")
            (root/"HYPERPLUS_PROFILE.json").write_text(json.dumps({"wheel_sha256":hashlib.sha256(b"fixture").hexdigest()}))
            profile=json.loads((source.parent/"COLAB_RUNTIME_20261015.json").read_text())
            (root/"COLAB_RUNTIME_20261015.json").write_text(json.dumps(profile))
            for name in ("train.npy","heldout.npy","manifest.json","train.py"):
                (root/name).write_text("fixture")
            previous=maker.__file__
            try:
                maker.__file__=str(root/"make_job.py")
                args=SimpleNamespace(wheel=str(wheel),data=str(root),output=str(root/"job.colajob"),
                                     local_cpu=False,runtime=runtime,name="fixture",steps=2000)
                with contextlib.redirect_stdout(io.StringIO()):
                    maker.make_job(args)
            finally:
                maker.__file__=previous
            return json.loads((root/"job.colajob").read_text()),profile

    def test_latest_binds_explicit_environment_and_profile_payload(self):
        job,profile=self.generate("latest")
        self.assertEqual(job["python"]["requires_python"],"==3.13.15")
        self.assertEqual(set(job["python"]["runtime_requirements"]),
                         {"torch==2.11.0+cu130","numpy==2.1.3","pillow==11.3.0"})
        self.assertIn("workload/COLAB_RUNTIME_20261015.json",{p["target"] for p in job["payload"]})
        self.assertEqual(job["python"]["install_requirements"],[])
        self.assertEqual(job["resources"]["accelerator"],"gpu")

    def test_dated_job_retains_legacy_environment(self):
        job,profile=self.generate("2026.07")
        self.assertEqual(job["python"]["requires_python"],"==3.12.13")
        self.assertIn("torch==2.11.0+cu128",job["python"]["runtime_requirements"])
        self.assertNotIn("workload/COLAB_RUNTIME_20261015.json",{p["target"] for p in job["payload"]})
