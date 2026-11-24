"""Freeze named steps and their dependencies without host execution or network."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest import mock

from colabcola.bundle import build_bundle
from colabcola.cli import main
from colabcola.errors import SpecError
from colabcola.invocation import load_job_invocation
from colabcola.package_guide import save_answers
from colabcola.packager import package, parser


class EnvironmentPackagingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.project = self.root / "project"
        self.project.mkdir()
        (self.project / "stage.py").write_text("pass\n", encoding="utf-8")
        self.environments = {"prepare": {"python": "3.10"}, "evaluate": {"python": "3.12"}}
        self.steps = [{"name": "prepare", "environment": "prepare", "command": ["python", "stage.py", "$HOME", "; literal"]},
                      {"name": "evaluate", "environment": "evaluate", "command": ["python", "stage.py", "--evaluate"]}]

    def package(self, stem="job", *extra):
        job = self.root / (stem + ".colajob")
        args = parser().parse_args(["--project", str(self.project), "--output", str(job), *extra])
        args.environments = self.environments
        args.steps = self.steps
        with mock.patch("subprocess.run", side_effect=AssertionError("host execution")), \
             mock.patch("urllib.request.urlopen", side_effect=AssertionError("host network")):
            result = package(args)
        return job, result

    def test_freezes_steps_sources_and_pinned_manager_and_preserves_legacy_identity_shape(self):
        job, result = self.package()
        first = build_bundle(load_job_invocation(job))
        manifest = json.loads(first.canonical_manifest_bytes)
        self.assertEqual([step["name"] for step in manifest["steps"]], ["prepare", "evaluate"])
        self.assertEqual(manifest["steps"][0]["command"][-2:], ["$HOME", "; literal"])
        self.assertEqual(manifest["environment_manager"]["kind"], "micromamba-linux-x86_64")
        self.assertEqual(len(manifest["environment_manager"]["sha256"]), 64)
        self.assertEqual(manifest["entrypoint"]["argv"], [])
        (self.project / "stage.py").write_text("raise RuntimeError('changed')\n")
        self.assertEqual(build_bundle(load_job_invocation(job)).job_id, first.job_id)
        old = self.root / "legacy.colajob"
        package(parser().parse_args(["--output", str(old), str(self.project / "stage.py")]))
        legacy = json.loads(build_bundle(load_job_invocation(old)).canonical_manifest_bytes)
        self.assertNotIn("environments", legacy)
        self.assertNotIn("steps", legacy)
        self.assertNotIn("environment_manager", legacy)

    def test_frozen_step_configuration_cannot_disagree_with_manifest(self):
        job, _ = self.package()
        commands = job.with_suffix(".files") / "_colabcola/commands.json"
        document = json.loads(commands.read_text())
        document["steps"].reverse()
        commands.write_text(json.dumps(document))
        with self.assertRaisesRegex(SpecError, "does not match"):
            build_bundle(load_job_invocation(job))

    def test_raw_job_validation_rechecks_requirement_contents_and_local_payload(self):
        req = self.project / "requirements.txt"
        req.write_text("./native\n", encoding="utf-8")
        native = self.project / "native"
        native.mkdir()
        (native / "pyproject.toml").write_text('[project]\nname="native"\nversion="1"\n')
        self.environments["prepare"]["pip"] = {"requirements": ["requirements.txt"]}
        job, _ = self.package()
        frozen = job.with_suffix(".files") / "requirements.txt"
        frozen.write_text("-r unbound.txt\n")
        with self.assertRaisesRegex(SpecError, "pip options"):
            build_bundle(load_job_invocation(job))
        frozen.write_text("./missing\n")
        with self.assertRaisesRegex(SpecError, "not packaged"):
            build_bundle(load_job_invocation(job))

    def test_excluded_scripts_and_dependencies_fail_without_publishing(self):
        self.environments["prepare"]["pip"] = {"requirements": ["missing.txt"]}
        with self.assertRaisesRegex(SpecError, "requirements file is not packaged"):
            self.package()
        self.assertFalse((self.root / "job.colajob").exists())
        self.assertFalse((self.root / "job.files").exists())
        self.environments["prepare"].pop("pip")
        self.steps[0]["command"][1] = "missing.py"
        with self.assertRaisesRegex(SpecError, "step script is not"):
            self.package()

    def test_old_frozen_helper_is_bound_by_payload_hash_not_current_install_bytes(self):
        job, _ = self.package()
        helper = job.with_suffix(".files") / "_colabcola/environment_runtime.py"
        before = build_bundle(load_job_invocation(job)).job_id
        helper.write_text(helper.read_text() + "\n# frozen helper from a different compiler build\n")
        self.assertNotEqual(build_bundle(load_job_invocation(job)).job_id, before)

    def test_global_wheel_and_dependency_flags_are_rejected_for_named_steps(self):
        for flag, value in (("--requirements", "requirements.txt"), ("--wheel", "native.whl"),
                            ("--runtime-requirement", "torch>=2"), ("--isolated-python", "3.12")):
            with self.subTest(flag=flag), self.assertRaisesRegex(SpecError, "global dependency"):
                self.package("job", flag, value)

    def test_saved_recipe_replays_and_maps_inputs_relative_to_step_cwd(self):
        (self.project / "analysis").mkdir()
        (self.project / "analysis/worker.py").write_text("pass\n")
        (self.project / "pyproject.toml").write_text('[project]\nname="example"\nversion="1"\nrequires-python="<3.9"\ndependencies=["inferred-package"]\n')
        source = self.root / "input.txt"
        source.write_text("data")
        config = {"project": str(self.project), "name": "recipe", "environments": self.environments,
            "steps": [{"name": "analyze", "environment": "evaluate", "cwd": "analysis", "command": ["python", "worker.py", str(source)]}],
            "inputs": [{"source": str(source), "target": "data/input.txt"}], "compute": {"gpu": False},
            "outputs": {"include": ["outputs/**"], "checkpoints": []}, "limits": {"timeout_seconds": 60}}
        recipe = self.root / "recipe.colaml"
        save_answers(recipe, config)
        saved = self.root / "saved.colaml"
        ids = []
        for index, current in enumerate((recipe, saved)):
            output = self.root / f"run{index}.colajob"
            arguments = ["package", "--colaml", str(current), "--non-interactive", "--output", str(output)]
            if index == 0:
                arguments.extend(["--save-colaml", str(saved)])
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(main(arguments), 0, stderr.getvalue())
            invocation = load_job_invocation(output)
            self.assertEqual(invocation.spec.steps[0].command[-1], "../data/input.txt")
            self.assertEqual(invocation.spec.python.requires_python, ">=3.11")
            self.assertEqual(invocation.spec.python.pip_requirements, ())
            ids.append(build_bundle(invocation).job_id)
        self.assertEqual(ids[0], ids[1])
        self.assertNotIn("commands", tomllib.loads(saved.read_text()))
