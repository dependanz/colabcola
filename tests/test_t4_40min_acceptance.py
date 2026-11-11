"""Offline acceptance recipe and short CPU-mode workload check."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "examples/t4_lifecycle_40min/probe.py"
RECIPE = ROOT / "examples/t4_lifecycle_40min/t4-40min.colaml"


class T4AcceptanceRecipeTests(unittest.TestCase):
    def test_recipe_requests_one_bounded_t4_and_real_40_minute_workload(self):
        recipe = tomllib.loads(RECIPE.read_text(encoding="utf-8"))
        self.assertEqual(recipe["compute"]["gpu"], "T4")
        self.assertIs(recipe["compute"]["high_mem"], False)
        self.assertEqual(recipe["limits"]["max_runtimes"], 1)
        self.assertEqual(recipe["limits"]["max_seconds"], 3300)
        self.assertEqual(recipe["limits"]["timeout_seconds"], 2520)
        self.assertEqual(recipe["environment"]["runtime_requirements"], ["torch"])
        self.assertEqual(recipe["outputs"]["checkpoints"],
                         ["outputs/checkpoints/state-*.json"])
        command = recipe["commands"][0]
        self.assertEqual(command[:2], ["python", "probe.py"])
        self.assertEqual(command[command.index("--duration-seconds") + 1], "2400")
        self.assertNotIn("--cpu-smoke", command)

    def test_local_cpu_smoke_writes_verified_manifest_and_final(self):
        with tempfile.TemporaryDirectory() as temp:
            run = subprocess.run(
                [sys.executable, str(PROBE),
                 "--cpu-smoke", "--duration-seconds", "2",
                 "--pulse-seconds", "0.5", "--progress-seconds", "0.5",
                 "--checkpoint-seconds", "1"],
                cwd=temp, capture_output=True, text=True, timeout=15)
            self.assertEqual(run.returncode, 0, run.stderr)
            output = Path(temp) / "outputs"
            final = json.loads((output / "acceptance.json").read_text())
            self.assertEqual(final["mode"], "cpu-smoke")
            self.assertGreaterEqual(final["elapsed_seconds"], 2)
            self.assertGreaterEqual(final["gpu_pulses"], 2)
            self.assertGreaterEqual(final["checkpoint_count"], 2)
            progress = (output / "progress.ndjson").read_text().splitlines()
            self.assertGreaterEqual(len(progress), 2)
            self.assertGreaterEqual(final["progress_observations"], 2)
            for manifest_path in (output / "checkpoints").glob("*.json"):
                m = json.loads(manifest_path.read_text())
                data = manifest_path.with_name(m["file"]).read_bytes()
                self.assertEqual(len(data), m["bytes"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), m["sha256"])

    def test_cpu_smoke_cannot_be_used_for_long_run(self):
        with tempfile.TemporaryDirectory() as temp:
            run = subprocess.run(
                [sys.executable, str(PROBE), "--cpu-smoke",
                 "--duration-seconds", "2400"],
                cwd=temp, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(run.returncode, 0)
            self.assertFalse((Path(temp) / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
