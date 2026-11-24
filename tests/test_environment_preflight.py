"""Offline admission and recovery evidence for named environment jobs."""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest import mock

from colabcola.cli import main
from colabcola.lifecycle_io import LifecycleError, atomic
from colabcola.lifecycle_worker import log_members
from colabcola.package_guide import save_answers
from colabcola.preflight import _manifest_identity, frozen_job


class EnvironmentPreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        example = Path(__file__).parents[1] / "examples" / "multi_environment"
        recipe = tomllib.loads((example / "pipeline.colaml").read_text(encoding="utf-8"))
        recipe["project"] = str(example)
        atomic(self.root / "helper.json", ["must-not-run"])
        recipe["execution"].update(drive_parent="fixture", token_command="helper.json")
        self.recipe = self.root / "recipe.colaml"
        save_answers(self.recipe, recipe)
        self.workspace = self.root / "run"

    def prepare(self):
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error), \
                mock.patch("colabcola.lifecycle.get_token", side_effect=AssertionError("offline")), \
                mock.patch("colabcola.lifecycle_provider.provider_for_plan", side_effect=AssertionError("offline")):
            result = main(["run", "--colaml", str(self.recipe), "--run-dir", str(self.workspace),
                           "--non-interactive", "--prepare-only"])
        self.assertEqual(result, 0, error.getvalue())
        self.directory = self.workspace / "recipe.run"
        self.plan = json.loads((self.directory / "plan.json").read_text(encoding="utf-8"))
        return error.getvalue()

    def test_example_preparation_freezes_envs_steps_and_bootstrap_without_network(self):
        progress = self.prepare()
        manifest = frozen_job(self.directory, self.plan)
        detail = _manifest_identity(self.directory, self.plan)
        for key in ("environments", "steps", "environment_manager"):
            self.assertEqual(self.plan[key], manifest[key])
            self.assertEqual(detail[key], manifest[key])
        self.assertEqual([step["name"] for step in self.plan["steps"]], ["prepare", "evaluate"])
        self.assertEqual(self.plan["environments"]["prepare"]["python"], "3.10")
        self.assertIn("Environments:", progress)
        self.assertIn("Steps: prepare [prepare] -> evaluate [evaluate]", progress)
        self.assertEqual(self.plan["artifact_groups"][0]["files"], ["data.json"])
        self.assertEqual(self.plan["max_seconds"], 1200)
        self.assertEqual(json.loads((self.directory / "state.json").read_text())["attempts"], [])
        before = (self.directory / "plan.json").read_bytes()
        self.prepare()
        self.assertEqual((self.directory / "plan.json").read_bytes(), before)

    def test_preflight_rejects_plan_metadata_changed_or_omitted_from_frozen_job(self):
        self.prepare()
        for key in ("environments", "steps", "environment_manager"):
            for operation in ("remove", "change"):
                with self.subTest(key=key, operation=operation):
                    plan = copy.deepcopy(self.plan)
                    if operation == "remove":
                        del plan[key]
                    else:
                        plan[key] = [] if isinstance(plan[key], dict) else {}
                    with self.assertRaisesRegex(LifecycleError, "frozen job"):
                        _manifest_identity(self.directory, plan)

    def test_environment_and_step_evidence_join_live_runtime_logs(self):
        result, job = self.root / "results", self.root / "job"
        result.mkdir()
        job.mkdir()
        environments = {"state": "failed", "failed_environment": "evaluate"}
        steps = {"steps": [{"name": "prepare", "state": "success", "exit_code": 0}]}
        atomic(result / "environments.json", environments)
        atomic(result / "steps.json", steps)
        (result / "environment.log").write_bytes(b"setup evidence\n")
        logs = log_members(result, job, [])
        self.assertEqual(json.loads(logs["runtime/environments.json"]), environments)
        self.assertEqual(json.loads(logs["runtime/steps.json"]), steps)
        self.assertEqual(logs["runtime/environment.log"], b"setup evidence\n")
