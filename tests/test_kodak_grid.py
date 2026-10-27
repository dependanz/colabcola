"""Small CPU correctness checks; no reconstruction search or cloud allocation."""
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

try:
    import numpy as np
    from PIL import Image
    import torch
except ImportError:
    raise unittest.SkipTest("Kodak workload checks require the optional torch/numpy/Pillow test environment")

spec = importlib.util.spec_from_file_location("kodak_train", Path(__file__).resolve().parents[1] / "examples/kodak_grid/run.py")
train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)


class GridTests(unittest.TestCase):
    def test_pair_initial_tensors_and_independent_gradients(self):
        torch.set_num_threads(2)
        pair = train.Pair(3, 32, 1, 30, 1234)
        self.assertEqual(*pair.initial_hashes())
        xy = torch.randn(8, 2); target = torch.randn(8, 3)
        losses = (pair(xy) - target.unsqueeze(0)).square().mean((1, 2))
        losses.sum().backward()
        for activation in range(2):
            weights = [p[activation].detach().clone().requires_grad_() for p in pair.weights]
            biases = [p[activation].detach().clone().requires_grad_() for p in pair.biases]
            value = xy
            for i, (w, b) in enumerate(zip(weights, biases)):
                value = value @ w + b
                if i < len(weights) - 1:
                    value = torch.sin(30 * value)
                    if activation == 0:
                        value = torch.tanh(value)
            ((value - target).square().mean()).backward()
            for batch, single in zip(pair.weights, weights):
                torch.testing.assert_close(batch.grad[activation], single.grad, rtol=2e-5, atol=2e-5)

    def test_fresh_process_resume_preserves_completed_pair_and_active_adam(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"; data.mkdir()
            image = np.random.default_rng(1).integers(0, 256, (8, 8, 3), dtype=np.uint8)
            Image.fromarray(image).save(data / "kodim01.png")
            train.atomic(data / "manifest.json", {"images": [{"name": "kodim01.png",
                "sha256": train.digest(data / "kodim01.png"), "shape": [8, 8, 3]}]})
            base = [sys.executable, str(Path(train.__file__)), "--data", str(data), "--device", "cpu",
                    "--steps", "4", "--pixels", "16", "--checkpoint-every", "2", "--depths", "1",
                    "--widths", "32", "64"]
            def execute(name, extra=()):
                result = subprocess.run(base + ["--output", str(root / name), *extra],
                                        capture_output=True, text=True, timeout=40)
                self.assertEqual(result.returncode, 0, result.stderr)
            execute("whole")
            # First pair completed at global step5, second pair checkpoint at step7.
            execute("split", ["--stop-after", "7"])
            manifest = root / "split/state-000000007.pt.json"
            state, meta = train.load_resume(manifest)
            self.assertEqual(len(state["completed"]), 1)
            self.assertEqual(state["step"], 2)
            execute("resumed", ["--resume", str(manifest)])
            whole = json.loads((root / "whole/summary.json").read_text())
            resumed = json.loads((root / "resumed/summary.json").read_text())
            def scientific(rows):
                return [{k: v for k, v in row.items() if k not in
                    ("seconds_this_attempt", "training_steps_this_attempt", "peak_cuda_memory_bytes")} for row in rows]
            self.assertEqual(scientific(whole["pairs"]), scientific(resumed["pairs"]))
            self.assertEqual(resumed["restored_step"], 7)
            events = [json.loads(x) for x in (root / "resumed/events.ndjson").read_text().splitlines()]
            self.assertEqual(len([x for x in events if x["event"] == "pair_started"]), 1)
            for artifact in meta["artifacts"]:
                self.assertEqual(train.digest(root / "split" / artifact), train.digest(root / "resumed" / artifact))
            for path in (root / "whole/fits").glob("*.zip"):
                with zipfile.ZipFile(path) as a, zipfile.ZipFile(root / "resumed/fits" / path.name) as b:
                    self.assertEqual(a.read("losses.json"), b.read("losses.json"))
                    left = torch.load(io.BytesIO(a.read("models.pt")), weights_only=True)
                    right = torch.load(io.BytesIO(b.read("models.pt")), weights_only=True)
                    for key in left:
                        self.assertTrue(torch.equal(left[key], right[key]))
            # A changed dataset or activation cannot silently resume.
            bad = subprocess.run(base + ["--output", str(root / "bad"), "--resume", str(manifest), "--beta", "2"],
                                 capture_output=True, text=True, timeout=40)
            self.assertNotEqual(bad.returncode, 0)
            artifact = root / "split" / next(iter(meta["artifacts"]))
            artifact.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "integrity"):
                train.load_resume(manifest)


if __name__ == "__main__":
    unittest.main()

