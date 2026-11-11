"""Bounded host/cgroup/disk telemetry without a live Colab runtime."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from colabcola.gpu_check import _colabcola_boot_memory
from colabcola.lifecycle_worker import collect_host_telemetry, log_members


class HostTelemetryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.proc = root / "proc"
        self.cg = root / "cgroup"
        self.control = root / "control"
        self.job = root / "job"
        for folder in (self.proc / "self", self.proc / "pressure",
                       self.cg, self.control, self.job):
            folder.mkdir(parents=True, exist_ok=True)

    def fixture_memory(self):
        (self.proc / "meminfo").write_text(
            "MemTotal: 1000 kB\nMemAvailable: 750 kB\n"
            "Cached: 400 kB\nDirty: 10 kB\nWriteback: 5 kB\n")
        (self.proc / "pressure/memory").write_text(
            "some avg10=0.10 avg60=0.20 avg300=0.30 total=123\n"
            "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n")
        (self.proc / "pressure/io").write_text(
            "some avg10=1.00 avg60=0.00 avg300=0.00 total=9\n")

    def test_cgroup_v2_and_meminfo_pressure_disk(self):
        self.fixture_memory()
        (self.proc / "self/cgroup").write_text("0::/scope\n")
        (self.cg / "cgroup.controllers").write_text("memory io")
        scope = self.cg / "scope"
        scope.mkdir()
        (scope / "memory.max").write_text("900000\n")
        (scope / "memory.current").write_text("123456\n")
        (scope / "memory.events").write_text("high 2\noom 1\noom_kill 0\n")
        (scope / "memory.pressure").write_text(
            "some avg10=2.5 avg60=1 avg300=0 total=999\n")
        report = collect_host_telemetry(self.control, self.job,
                                        proc_root=self.proc, cgroup_root=self.cg,
                                        clock=lambda: 42)
        self.assertEqual(report["observed_at"], 42)
        self.assertEqual(report["memory_bytes"]["MemTotal"], 1024000)
        self.assertEqual(report["memory_bytes"]["MemAvailable"], 768000)
        self.assertEqual(report["memory_bytes"]["Dirty"], 10240)
        self.assertEqual(report["cgroup"]["version"], 2)
        self.assertEqual(report["cgroup"]["memory_limit_bytes"], 900000)
        self.assertEqual(report["cgroup"]["memory_usage_bytes"], 123456)
        self.assertEqual(report["cgroup"]["memory_events"]["oom"], 1)
        self.assertEqual(report["pressure"]["cgroup_memory"]["some"]["avg10"], 2.5)
        self.assertEqual(report["pressure"]["memory"]["some"]["total"], 123)
        self.assertGreaterEqual(report["disk_free_bytes"]["job"], 0)
        self.assertGreaterEqual(report["disk_free_bytes"]["control"], 0)
        self.assertEqual(_colabcola_boot_memory(self.proc, self.cg),
                         {"mem_total_bytes": 1024000,
                          "cgroup_memory_limit_bytes": 900000})

    def test_cgroup_v1_memory_and_unavailable_pressure(self):
        self.fixture_memory()
        (self.proc / "self/cgroup").write_text("3:memory:/group\n")
        memory = self.cg / "memory/group"
        memory.mkdir(parents=True)
        (memory / "memory.limit_in_bytes").write_text("800000\n")
        (memory / "memory.usage_in_bytes").write_text("400000\n")
        (memory / "memory.failcnt").write_text("3\n")
        report = collect_host_telemetry(self.control, self.job,
                                        proc_root=self.proc, cgroup_root=self.cg)
        self.assertEqual(report["cgroup"]["version"], 1)
        self.assertEqual(report["cgroup"]["memory_limit_bytes"], 800000)
        self.assertEqual(report["cgroup"]["memory_usage_bytes"], 400000)
        self.assertEqual(report["cgroup"]["memory_events"]["failcnt"], 3)
        self.assertEqual(_colabcola_boot_memory(self.proc, self.cg)
                         ["cgroup_memory_limit_bytes"], 800000)

    def test_missing_probes_do_not_fail_or_invent_readings(self):
        report = collect_host_telemetry(self.control, self.job,
                                        proc_root=self.proc, cgroup_root=self.cg)
        self.assertIsNone(report["memory_bytes"]["MemTotal"])
        self.assertIsNone(report["cgroup"]["version"])
        self.assertIsNone(report["cgroup"]["memory_limit_bytes"])
        self.assertEqual(report["pressure"]["memory"], {})
        self.assertEqual(_colabcola_boot_memory(self.proc, self.cg),
                         {"mem_total_bytes": None,
                          "cgroup_memory_limit_bytes": None})

    def test_logs_include_telemetry_and_bounded_execution_tail(self):
        result = self.control / "results"
        result.mkdir()
        (result / "stdout.log").write_bytes(b"work")
        (self.control / "execution.log").write_bytes(b"x" * 100000 + b"last")
        with mock.patch("colabcola.lifecycle_worker.collect_host_telemetry",
                        return_value={"schema": 1, "observed_at": 42}):
            logs = log_members(result, self.job, [], control_root=self.control)
        self.assertEqual(logs["runtime/stdout.log"], b"work")
        self.assertEqual(json.loads(logs["runtime/host-telemetry.json"]),
                         {"schema": 1, "observed_at": 42})
        self.assertEqual(len(logs["runtime/execution-tail.log"]), 65536)
        self.assertTrue(logs["runtime/execution-tail.log"].endswith(b"last"))

    def test_legacy_log_collector_remains_unchanged_without_control_root(self):
        result = self.control / "results"
        result.mkdir()
        (result / "stderr.log").write_bytes(b"old")
        self.assertEqual(log_members(result, self.job, []),
                         {"runtime/stderr.log": b"old"})


if __name__ == "__main__":
    unittest.main()
