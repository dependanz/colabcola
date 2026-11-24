"""Generated notebook integration with provisioning mocked at its boundary."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from colabcola import JobSpec, bind_job_arguments, build_bundle, build_notebook
from colabcola import runtime
from test_notebook import notebook_job


_LEGACY_SOURCE_HASHES = {
    (False, False, False): "1da86ffe2aa4991041266643d2b0a32bf264097d1fe94dc03729e579888c4c11",
    (False, False, True): "722747539cb6965e75f14b6a1b040d7ae07e4e155dfc086d41878a939d169722",
    (False, True, False): "cbba62352cfa3af0c4ea53f6c46266abe7d3e5dc01f88c360a0b5b0ca07e9a35",
    (False, True, True): "f30b6f3737476bb4525f37bf3d24467d1349c9ebe0e27961f6302379e144bea3",
    (True, False, False): "a7223a1e50abe1caaea50d8d3a810cb28cbcfaa2b52dc86e1fbfd9bd82aba5f0",
    (True, False, True): "6b064a66185e8d65327914beafc435203aa30ac1eaf040215c5e5e27835bbcea",
    (True, True, False): "17d95f7f8f86c1446f1ac5b389a66ab2bf40bbc1d6970e1f3aa0d9d6a629c3b2",
    (True, True, True): "bc7482f2cc984d2feb0003567e6d5b20b7eb7a4c198b87de28b988befd0536c1",
}


class NamedEnvironmentNotebookTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.prepared = []

    def artifact(self, *, launcher=None, first_failure=False):
        source = self.root / "launch.py"
        source.write_text(launcher if launcher is not None else Path(runtime.__file__).with_name("step_launcher.py").read_text(), encoding="utf-8")
        helper = self.root / "environment_runtime.py"
        # Exercise the actual ordered launcher while avoiding environment downloads.
        # The provisioning/runtime-command boundary is the only substituted part.
        helper.write_text('''import os, sys
def _colabcola_environment_command(descriptor, argv, job_root, declared=None):
    env = os.environ.copy()
    env.update(descriptor.get("env", {}))
    env.update(declared or {})
    env["FIXTURE_ENVIRONMENT"] = descriptor["name"]
    return [sys.executable, *argv[1:]], env
''', encoding="utf-8")
        first = self.root / "first.py"
        first.write_text('''import json, os
from pathlib import Path
Path("outputs").mkdir(exist_ok=True)
Path("outputs/order.json").write_text(json.dumps([os.environ["FIXTURE_ENVIRONMENT"]]))
''' + ("raise SystemExit(7)\n" if first_failure else ""), encoding="utf-8")
        second = self.root / "second.py"
        second.write_text('''import json, os
from pathlib import Path
p = Path("outputs/order.json")
values = json.loads(p.read_text())
values.append(os.environ["FIXTURE_ENVIRONMENT"])
p.write_text(json.dumps(values))
''', encoding="utf-8")
        commands = self.root / "commands.json"
        mapping = notebook_job(source, target="_colabcola/launch.py")
        mapping["entrypoint"]["argv"] = []
        mapping["environments"] = {"prepare": {"python": "3.10"}, "evaluate": {"python": "3.12"}}
        mapping["steps"] = [
            {"name": "prepare", "environment": "prepare", "command": ["python", "first.py"], "cwd": "."},
            {"name": "evaluate", "environment": "evaluate", "command": ["python", "second.py"], "cwd": "."},
        ]
        for file, target in ((helper, "_colabcola/environment_runtime.py"), (commands, "_colabcola/commands.json"),
                             (first, "first.py"), (second, "second.py")):
            mapping["payload"].append({"role": "source", "source": str(file), "target": target})
        commands.write_text(json.dumps({"steps": mapping["steps"]}), encoding="utf-8")
        spec = JobSpec.from_mapping(mapping, base_dir=self.root)
        commands.write_text(json.dumps({"steps": [step.mapping() for step in spec.steps]}), encoding="utf-8")
        return build_notebook(build_bundle(bind_job_arguments(spec)))

    def prepare(self, environments, manager, job_root, environment_root, result_root, deadline=None):
        self.prepared.append({"deadline": deadline, "at": time.time()})
        facts = {"schema": 1, "state": "ready", "environments": {
            name: {"state": "ready", "definition": definition} for name, definition in environments.items()
        }, "failed_environment": None}
        (result_root / "environments.json").write_text(json.dumps(facts), encoding="utf-8")
        return {name: {"name": name, "env": definition["env"]} for name, definition in environments.items()}, facts

    def execute(self, artifact, *, prepare=None, lifecycle_deadline=None):
        namespace = {"COLABCOLA_CONTENT_ROOT_OVERRIDE": str(self.root / "content"),
                     "_test_prepare_environments": prepare or self.prepare}
        if lifecycle_deadline is not None:
            namespace["COLABCOLA_LIFECYCLE_DEADLINE"] = lifecycle_deadline
        error = None
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            for cell in json.loads(artifact.notebook_bytes)["cells"]:
                if cell["cell_type"] != "code":
                    continue
                source = cell["source"]
                if cell["metadata"]["colabcola"]["group"] == "environment":
                    source = source.replace("descriptors, environment_facts = _colabcola_prepare_environments(",
                                            "descriptors, environment_facts = _test_prepare_environments(", 1)
                try:
                    exec(source, namespace)
                except RuntimeError as exc:
                    error = exc
                    break
        result_root = self.root / "content" / "colabcola" / "results" / artifact.bundle.job_id
        result = json.loads((result_root / "result.json").read_text())
        return result, namespace, result_root, error

    def test_legacy_runtime_sources_are_byte_identical_for_all_eight_variants(self):
        for (drive, isolated, remote), expected in _LEGACY_SOURCE_HASHES.items():
            with self.subTest(drive=drive, isolated=isolated, remote=remote):
                function = runtime.drive_runtime_cell_sources if drive else runtime.runtime_cell_sources
                source = function(isolated=isolated, remote_inputs=remote)
                self.assertEqual(hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(), expected)

    def test_named_generated_sources_compile_for_both_transports(self):
        for function in (runtime.runtime_cell_sources, runtime.drive_runtime_cell_sources):
            for remote in (False, True):
                with self.subTest(function=function.__name__, remote=remote):
                    cells = function(named_environments=True, remote_inputs=remote)
                    for group, source in cells:
                        compile(source, "<" + group + ">", "exec")
                    self.assertIn("_colabcola_prepare_environments(", dict(cells)["environment"])
                    self.assertIn("COLABCOLA_ENVIRONMENTS_CONFIG", dict(cells)["execution"])

    def test_named_steps_execute_in_order_and_result_records_setup(self):
        artifact = self.artifact()
        with mock.patch.dict(os.environ, {"FIXTURE_PRIVATE_TOKEN": "never-copy-to-config"}):
            result, namespace, result_root, error = self.execute(artifact)
        self.assertIsNone(error)
        self.assertEqual(result["terminal_state"], "success", result["failure"])
        self.assertEqual([step["name"] for step in result["steps"]], ["prepare", "evaluate"])
        self.assertTrue(all(step["state"] == "success" for step in result["steps"]))
        self.assertEqual(result["stages"]["environment"]["state"], "success")
        self.assertEqual(result["environment"]["environments"]["state"], "ready")
        self.assertIsNone(result["failed_step"])
        work = namespace["COLABCOLA_JOB_ROOT"]
        self.assertEqual(json.loads((work / "outputs/order.json").read_text()), ["prepare", "evaluate"])
        config = (result_root / "environments-runtime.json").read_text()
        self.assertNotIn("never-copy-to-config", config)
        self.assertNotIn("FIXTURE_PRIVATE_TOKEN", config)
        self.assertGreater(json.loads(config)["deadline"], time.time())

    def test_failed_step_stops_later_steps_and_names_failure(self):
        result, namespace, _, error = self.execute(self.artifact(first_failure=True))
        self.assertIsNotNone(error)
        self.assertEqual(result["terminal_state"], "job_failed")
        self.assertEqual(result["failed_step"], "prepare")
        self.assertEqual(result["failure"]["step"], "prepare")
        self.assertEqual(result["steps"][0]["exit_code"], 7)
        self.assertNotEqual(result["steps"][1]["state"], "success")
        self.assertEqual(json.loads((namespace["COLABCOLA_JOB_ROOT"] / "outputs/order.json").read_text()), ["prepare"])

    @unittest.skipUnless(os.name == "posix", "owned process-group teardown requires POSIX")
    def test_failed_step_descendant_cannot_keep_launcher_output_pipe_open(self):
        artifact = self.artifact()
        (self.root / "first.py").write_text(
            "import subprocess, sys\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "raise SystemExit(7)\n", encoding="utf-8")
        result_root = self.root / "result"
        result_root.mkdir()
        config = {
            "schema": 1, "job_id": artifact.bundle.job_id,
            "job_root": str(self.root), "result_root": str(result_root),
            "env": {}, "deadline": time.time() + 30,
            "steps": json.loads((self.root / "commands.json").read_text())["steps"],
            "environments": {name: {"name": name, "env": {}} for name in ("prepare", "evaluate")},
        }
        config_path = result_root / "environments-runtime.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        env = dict(os.environ, COLABCOLA_ENVIRONMENTS_CONFIG=str(config_path))
        process = subprocess.Popen([sys.executable, str(self.root / "launch.py")],
            cwd=self.root, env=env, start_new_session=True, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        timed_out = False
        try:
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                timed_out = True
        finally:
            # Also reap descendants if a regression lets the launcher exit first.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate(timeout=2)
        self.assertFalse(timed_out, "a failed step's descendant retained the output pipe")
        self.assertNotEqual(process.returncode, 0)
        evidence = json.loads((result_root / "steps.json").read_text())
        self.assertEqual(evidence["state"], "failed")
        self.assertEqual(evidence["failed_step"], "prepare")
        self.assertEqual(evidence["steps"][0]["exit_code"], 7)
        self.assertEqual(evidence["steps"][1]["state"], "not_run")

    def test_zero_exit_without_step_evidence_cannot_claim_success(self):
        result, _, _, error = self.execute(self.artifact(launcher="pass\n"))
        self.assertIsNotNone(error)
        self.assertEqual(result["terminal_state"], "job_failed")
        self.assertIn("evidence is missing", result["failure"]["message"])

    def test_zero_exit_with_incomplete_step_evidence_cannot_claim_success(self):
        template = '''import json, os
from pathlib import Path
config=json.loads(Path(os.environ["COLABCOLA_ENVIRONMENTS_CONFIG"]).read_text())
steps=[dict(step,state="success",exit_code=0) for step in config["steps"]]
{change}
Path(config["result_root"],"steps.json").write_text(json.dumps(dict(schema=1,job_id=config["job_id"],state="success",steps=steps,failed_step=None)))
'''
        for change in ('steps.pop()', 'steps[-1]["state"]="pending"', 'steps[-1]["environment"]="unexpected"'):
            with self.subTest(change=change):
                result, _, _, error = self.execute(self.artifact(launcher=template.format(change=change)))
                self.assertIsNotNone(error)
                self.assertEqual(result["terminal_state"], "job_failed")

    def test_setup_failure_preserves_partial_environment_evidence(self):
        def failing(environments, manager, job_root, environment_root, result_root, deadline=None):
            facts = {"schema": 1, "state": "failed", "environments": {"prepare": {"state": "failed"}}, "failed_environment": "prepare"}
            (result_root / "environments.json").write_text(json.dumps(facts))
            raise RuntimeError("declared environment could not be resolved")
        with mock.patch("subprocess.Popen", side_effect=AssertionError("workload must not start")):
            result, _, _, error = self.execute(self.artifact(), prepare=failing)
        self.assertIsNotNone(error)
        self.assertEqual(result["terminal_state"], "environment_failed")
        self.assertEqual(result["environment"]["environments"]["failed_environment"], "prepare")
        self.assertEqual(result["failure"]["environment"], "prepare")
        self.assertEqual(result["stages"]["environment"]["state"], "failed")
        self.assertEqual(result["steps"], [])

    def test_finalization_reserve_exhausted_before_setup_prevents_any_launch(self):
        with mock.patch("subprocess.Popen", side_effect=AssertionError("workload must not start")):
            result, _, _, error = self.execute(self.artifact(), lifecycle_deadline=time.time() + 10)
        self.assertIsNotNone(error)
        self.assertEqual(self.prepared, [])
        self.assertEqual(result["terminal_state"], "environment_failed")
        self.assertIn("finalization reserve", result["failure"]["message"])

    def test_setup_exhausting_reserved_budget_never_launches_workload(self):
        start = time.time()
        with mock.patch("time.time", return_value=start) as clock:
            def slow_prepare(*args, **kwargs):
                result = self.prepare(*args, **kwargs)
                clock.return_value = kwargs["deadline"] + 1
                return result
            with mock.patch("subprocess.Popen", side_effect=AssertionError("workload must not start")):
                result, _, _, error = self.execute(self.artifact(), prepare=slow_prepare, lifecycle_deadline=start + 90)
        self.assertIsNotNone(error)
        self.assertEqual(result["terminal_state"], "environment_failed")
        self.assertIn("setup exhausted", result["failure"]["message"])

    def test_setup_and_workload_obey_same_reserved_lifecycle_deadline(self):
        cap = time.time() + 90
        result, _, result_root, error = self.execute(self.artifact(), lifecycle_deadline=cap)
        self.assertIsNone(error)
        self.assertEqual(self.prepared[0]["deadline"], cap - 30)
        config = json.loads((result_root / "environments-runtime.json").read_text())
        self.assertLessEqual(config["deadline"], cap - 30)
        self.assertEqual(result["terminal_state"], "success")


if __name__ == "__main__":
    unittest.main()
