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

    def test_recipe_packages_offline_with_exact_lifecycle_bounds(self):
        with tempfile.TemporaryDirectory() as temp:
            job = Path(temp) / "t4-live.colajob"
            command = [
                sys.executable, "-m", "colabcola", "package",
                "--colaml", str(RECIPE), "--non-interactive",
                "--output", str(job),
            ]
            result = subprocess.run(
                command, cwd=ROOT, text=True,
                capture_output=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stderr)
            packaged = json.loads(job.read_text())
            config = json.loads(job.with_suffix(".run.json").read_text())
            self.assertEqual(packaged["run"]["timeout_seconds"], 2520)
            self.assertEqual(packaged["managed_recovery"]["checkpoint_pattern"],
                             "outputs/checkpoints/state-*.json")
            self.assertEqual(packaged["managed_recovery"]["log_patterns"],
                             ["outputs/progress.ndjson"])
            self.assertEqual(json.loads(
                (job.with_suffix(".files") / "_colabcola" / "commands.json").read_text()
            )["resume_arg"], "--resume-manifest")
            self.assertEqual(packaged["resources"]["accelerator"], "gpu")
            self.assertEqual(packaged["python"]["runtime_requirements"], ["torch"])
            self.assertEqual(packaged["outputs"][0]["pattern"],
                             "outputs/acceptance.json")
            self.assertEqual(config["gpu"], "T4")
            self.assertEqual(config["max_attempts"], 1)
            self.assertEqual(config["max_seconds"], 3300)
            self.assertFalse(config["fault_after_checkpoint"])
            self.assertTrue(job.with_suffix(".files").is_dir())
            self.assertNotIn("access_token", json.dumps(packaged))

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

    def test_cpu_smoke_can_resume_verified_checkpoint_in_fresh_directory(self):
        with tempfile.TemporaryDirectory() as root:
            first = Path(root) / "first"
            second = Path(root) / "second"
            first.mkdir()
            second.mkdir()
            command = [sys.executable, str(PROBE), "--cpu-smoke",
                       "--pulse-seconds", "0.5", "--progress-seconds", "0.5",
                       "--checkpoint-seconds", "1"]
            a = subprocess.run(command + ["--duration-seconds", "2"],
                               cwd=first, capture_output=True, text=True, timeout=15)
            self.assertEqual(a.returncode, 0, a.stderr)
            manifests = sorted((first / "outputs/checkpoints").glob("*.json"))
            self.assertTrue(manifests)
            latest = manifests[-1]
            b = subprocess.run(command + ["--duration-seconds", "4",
                                           "--resume-manifest", str(latest)],
                               cwd=second, capture_output=True, text=True, timeout=15)
            self.assertEqual(b.returncode, 0, b.stderr)
            result = json.loads((second / "outputs/acceptance.json").read_text())
            self.assertEqual(result["mode"], "cpu-smoke")
            self.assertEqual(result["resumed_from_step"],
                             json.loads(latest.read_text())["step"])
            self.assertGreaterEqual(result["elapsed_seconds"], 4)
            self.assertLess(result["segment_elapsed_seconds"], 4)
            self.assertGreater(result["checkpoint_count"], result["resumed_from_step"])

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
