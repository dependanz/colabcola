"""Offline declaration tests for per-step workload environments."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from colabcola.environments import (normalize_environments, normalize_steps,
    normalize_environment_variables, normalize_pip_package, requirement_file_packages)
from colabcola.errors import SpecError
from colabcola.invocation import bind_job_arguments
from colabcola.spec import JobSpec


def named_spec(source):
    return {
        "schema_version": 1, "name": "two-environments",
        "payload": [{"role": "source", "source": str(source), "target": "_colabcola/launch.py"}],
        "entrypoint": {"kind": "python_script", "path": "_colabcola/launch.py", "argv": [], "cwd": "."},
        "python": {"requires_python": ">=3.12", "runtime_requirements": [], "install_requirements": [], "embedded_wheels": []},
        "runtime": {"provider": "google_colab", "version": "latest"},
        "resources": {"accelerator": "cpu", "gpu_count": 0, "cuda_required": False, "minimum_gpu_memory_bytes": None},
        "run": {"timeout_seconds": 120, "seed": 1, "environment": {}, "capture_stdout": True, "capture_stderr": True},
        "outputs": [{"pattern": "outputs/**", "required": True, "maximum_bytes": 1024}],
        "limits": {"maximum_file_count": 100},
        "environments": {"prepare": {"python": "3.10"}, "evaluate": {"python": "3.12"}},
        "steps": [{"name": "prepare", "environment": "prepare", "command": ["python", "stage.py"]},
                  {"name": "evaluate", "environment": "evaluate", "command": ["python", "stage.py", "--evaluate"]}],
    }


class EnvironmentSpecTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "stage.py"
        self.source.write_text("pass\n", encoding="utf-8")

    def test_named_spec_is_detached_and_preserves_step_order(self):
        value = named_spec(self.source)
        spec = JobSpec.from_mapping(value)
        self.assertEqual([e.name for e in spec.environments], ["evaluate", "prepare"])
        self.assertEqual([s.name for s in spec.steps], ["prepare", "evaluate"])
        self.assertEqual(spec.environments[0].mapping()["conda"]["channels"], ["conda-forge"])
        value["steps"][0]["command"].append("later mutation")
        first = spec.environments[0].mapping()
        first["pip"]["packages"].append("later-mutation")
        self.assertEqual(spec.steps[0].command, ("python", "stage.py"))
        self.assertEqual(spec.environments[0].mapping()["pip"]["packages"], [])
        with self.assertRaisesRegex(SpecError, "invocation arguments"):
            bind_job_arguments(spec, ["extra"])

    def test_declarations_must_be_paired_and_generated_entrypoint_is_required(self):
        for key in ("environments", "steps"):
            value = named_spec(self.source)
            del value[key]
            with self.subTest(key=key), self.assertRaisesRegex(SpecError, "together"):
                JobSpec.from_mapping(value)
        for field, value in (("path", "stage.py"), ("cwd", "work"), ("argv", ["override"])):
            spec = named_spec(self.source)
            spec["entrypoint"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(SpecError, "generated"):
                JobSpec.from_mapping(spec)

    def test_global_dependencies_cannot_silently_target_control_python(self):
        for field, value in (("runtime_requirements", ["numpy>=2"]),
                             ("pip_requirements", ["numpy>=2"]),
                             ("install_requirements", ["example==1 --hash=sha256:" + "a" * 64])):
            spec = named_spec(self.source)
            spec["python"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(SpecError, "global Python"):
                JobSpec.from_mapping(spec)

    def test_protected_variables_and_secret_names_are_rejected(self):
        for key in ("PATH", "PYTHONHOME", "PIP_CONFIG_FILE", "CONDA_PREFIX", "MAMBA_ROOT_PREFIX",
                    "LD_LIBRARY_PATH", "LD_PRELOAD", "CUDA_HOME", "CPATH", "BASH_ENV", "COLABCOLA_SEED", "API_TOKEN"):
            with self.subTest(key=key), self.assertRaises(SpecError):
                normalize_environment_variables({key: "value"})
        self.assertEqual(normalize_environment_variables({"PYTHONPATH": ".:src", "MAX_JOBS": "2"}),
                         {"MAX_JOBS": "2", "PYTHONPATH": ".:src"})
        for path in ("/tmp", "../src", "src:", r"C:\src"):
            with self.subTest(path=path), self.assertRaises(SpecError):
                normalize_environment_variables({"PYTHONPATH": path})

    def test_environment_versions_channels_and_fields_are_strict(self):
        for definition in ({"python": "3.7"}, {"python": ">=3.10"}, {"python": "3.12", "extra": True},
                           {"python": "3.12", "conda": {"channels": []}},
                           {"python": "3.12", "conda": {"packages": ["--prefix=/tmp"]}},
                           {"python": "3.12", "conda": {"packages": ["python=3.9"]}},
                           {"python": "3.12", "pip": {"no_build_isolation": "false"}}):
            with self.subTest(definition=definition), self.assertRaises(SpecError):
                normalize_environments({"work": definition})
        value = normalize_environments({"work": {"python": "3.10.16", "cuda": "12.8",
            "conda": {"channels": ["nvidia/label/cuda-12.8.1", "conda-forge"], "packages": ["cuda-toolkit=12.8"]}}})
        self.assertEqual(value["work"]["cuda"], "12.8")
        for name in ("Work", "work_env", "a" * 65):
            with self.subTest(name=name), self.assertRaises(SpecError):
                normalize_environments({name: {"python": "3.12"}})

    def test_malformed_declaration_types_raise_spec_errors(self):
        for value in (None, True, [], "work", {}, {1: {"python": "3.12"}, "work": {"python": "3.12"}},
                      {"work": None}, {"work": {"python": 3.12}},
                      {"work": {"python": "3.12", 1: "unknown"}},
                      {"work": {"python": "3.12", "conda": {"channels": "conda-forge"}}},
                      {"work": {"python": "3.12", "pip": {"packages": [True]}}},
                      {"work": {"python": "3.12", "env": {1: "value", "MAX_JOBS": "2"}}}):
            with self.subTest(value=value), self.assertRaises(SpecError):
                normalize_environments(value)
        env = normalize_environments({"work": {"python": "3.12"}})
        for value in (None, {}, "python stage.py", [None],
                      [{"name": "work", "environment": True, "command": ["python", "stage.py"]}],
                      [{"name": "work", "environment": "work", "command": [None]}]):
            with self.subTest(value=value), self.assertRaises(SpecError):
                normalize_steps(value, env)

    def test_steps_preserve_literal_arguments_and_reject_unsupported_shells_and_escape(self):
        env = normalize_environments({"work": {"python": "3.12"}})
        raw = {"name": "step", "environment": "work", "command": ["python", "../stage.py", "$HOME", "; literal"], "cwd": "analysis"}
        self.assertEqual(normalize_steps([raw], env)[0], raw)
        shell = dict(raw, command=["bash", "script.sh", "$HOME", "; literal"])
        self.assertEqual(normalize_steps([shell], env)[0], shell)
        for command in (["bash", "-c", "true"], ["bash", "../script.sh"], ["sh", "-c", "true"], ["python", "-c", "pass"], ["python", "../../escape.py"], ["/bin/tool"]):
            changed = dict(raw, command=command)
            with self.subTest(command=command), self.assertRaises(SpecError):
                normalize_steps([changed], env)
        with self.assertRaisesRegex(SpecError, "unknown environment"):
            normalize_steps([dict(raw, environment="missing")], env)
        with self.assertRaisesRegex(SpecError, "duplicate step"):
            normalize_steps([raw, raw], env)
        with self.assertRaises(SpecError):
            normalize_steps([dict(raw, command="python stage.py")], env)

    def test_requirements_keep_git_fragments_and_reject_mutable_or_secret_urls(self):
        reference = "example @ git+https://example.org/source.git@" + "a" * 40 + "#subdirectory=bindings/python"
        self.assertEqual(requirement_file_packages(reference + " # a comment\n"), [reference])
        hashed = "example==1 --hash=sha256:" + "a" * 64 + " --hash=sha256:" + "b" * 64
        self.assertEqual(requirement_file_packages(hashed), [hashed])
        self.assertEqual(requirement_file_packages("./native-extension\n"), ["./native-extension"])
        for requirement in ("-r nested.txt", "--extra-index-url https://example.org", "-e ./source",
                            "example @ git+https://example.org/source.git@main", "example @ https://example.org/file.whl",
                            "example @ git+https://user:secret@example.org/source.git@" + "a" * 40,
                            "./../outside", "example==1 --hash=md5:abc"):
            with self.subTest(requirement=requirement), self.assertRaises(SpecError):
                requirement_file_packages(requirement)

    def test_json_schema_accepts_named_spec_and_requires_pair(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("optional JSON schema validator unavailable")
        schema = json.loads((Path(__file__).parents[1] / "schemas/job-v1.schema.json").read_text())
        value = named_spec(self.source)
        jsonschema.Draft202012Validator(schema).validate(value)
        del value["steps"]
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(schema).validate(value)
