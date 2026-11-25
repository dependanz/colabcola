"""Packaged shell commands: source identity, literal argv and step execution."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from colabcola import environment_runtime
from colabcola.bundle import build_bundle
from colabcola.cli import main
from colabcola.environments import normalize_environments, normalize_steps, shell_script_path
from colabcola.errors import SpecError
from colabcola.invocation import load_job_invocation
from colabcola.package_guide import inspect_project, save_answers
from colabcola.packager import package, parser, _LAUNCHER
from colabcola.shell_runtime import packaged_bash_argv


LITERALS = ["two words", "$HOME", "$(touch unexpected)", "; echo injected", '"quoted"', "", "line\nbreak", "--flag"]
SCRIPT = b'#!/bin/bash\nset -eu\nprintf "shell stdout\\n"\nprintf "shell stderr\\n" >&2\nprintf "%s\\0" "$@" > args.bin\n'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PackagedShellTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.script = self.project / "step.sh"
        self.script.write_bytes(SCRIPT)
        self.envs = {"work": {"python": "3.12"}}

    def package(self, name, command, *, named=False, cwd=".", extra=()):
        target = self.root / (name + ".colajob")
        options = ["--project", str(self.project), "--output", str(target), "--name", "shell-test", *extra]
        args = parser().parse_args(options if named else [*options, "--", *command])
        if named:
            args.environments = self.envs
            args.steps = [{"name": "shell", "environment": "work", "command": command, "cwd": cwd}]
        with mock.patch("subprocess.run", side_effect=AssertionError("packaging executed workload")), \
             mock.patch("urllib.request.urlopen", side_effect=AssertionError("packaging accessed network")):
            package(args)
        return target

    def test_valid_shell_argv_and_invalid_script_paths(self):
        envs = normalize_environments(self.envs)
        raw = {"name": "shell", "environment": "work", "command": ["bash", "step.sh", *LITERALS], "cwd": "nested"}
        self.assertEqual(normalize_steps([raw], envs)[0], raw)
        self.assertEqual(shell_script_path(raw["command"], raw["cwd"]), "nested/step.sh")
        for argv in (["bash"], ["bash", "-c", "true"], ["bash", "-x", "step.sh"],
                     ["bash", "--", "step.sh"], ["bash", "-step.sh"], ["bash", "step.py"],
                     ["bash", "/tmp/step.sh"], ["bash", "C:/step.sh"], ["bash", "C:\\step.sh"],
                     ["bash", "../step.sh"], ["bash", "dir/../step.sh"], ["bash", "./step.sh"],
                     ["bash", "dir//step.sh"], ["bash", "_colabcola/step.sh"],
                     ["bash", "a\n.sh"], ["sh", "step.sh"], ["BASH", "step.sh"],
                     ["bash.exe", "step.sh"], ["/bin/bash", "step.sh"]):
            with self.subTest(argv=argv), self.assertRaises(SpecError):
                normalize_steps([dict(raw, cwd=".", command=argv)], envs)
            with self.subTest(legacy=argv), self.assertRaises(SpecError):
                self.package("invalid", argv)

    def test_frozen_hash_and_snapshot_for_both_recipe_forms(self):
        for named in (False, True):
            with self.subTest(named=named):
                self.script.write_bytes(SCRIPT)
                job = self.package(f"first-{named}", ["bash", "step.sh", *LITERALS], named=named)
                first = build_bundle(load_job_invocation(job))
                snapshot = job.with_suffix(".files")
                config = json.loads((snapshot / "_colabcola/commands.json").read_text())
                self.assertEqual(config["shell_scripts"], {"step.sh": digest(self.script)})
                self.assertIn("_colabcola/shell_runtime.py", {f.path for f in first.files})
                self.assertEqual((snapshot / "step.sh").read_bytes(), SCRIPT)
                self.script.write_bytes(SCRIPT + b"# changed\n")
                self.assertEqual(build_bundle(load_job_invocation(job)).job_id, first.job_id)
                other = self.package(f"second-{named}", ["bash", "step.sh", *LITERALS], named=named)
                self.assertNotEqual(build_bundle(load_job_invocation(other)).job_id, first.job_id)
                if named:
                    (snapshot / "step.sh").write_bytes(b"exit 1\n")
                    with self.assertRaisesRegex(SpecError, "shell script hashes"):
                        build_bundle(load_job_invocation(job))

    def test_missing_excluded_directory_or_external_input_script_is_refused(self):
        external = self.root / "external.sh"
        external.write_bytes(SCRIPT)
        (self.project / "directory.sh").mkdir()
        for named in (False, True):
            for args, extra in ((["bash", "missing.sh"], ()),
                                (["bash", "step.sh"], ("--exclude", "step.sh")),
                                (["bash", "directory.sh"], ()),
                                (["bash", "missing.sh"], ("--data", str(external) + "=missing.sh"))):
                with self.subTest(named=named, args=args, extra=extra), self.assertRaisesRegex(SpecError, "packaged project"):
                    self.package("invalid", args, named=named, extra=extra)
                self.assertFalse((self.root / "invalid.colajob").exists())
                self.assertFalse((self.root / "invalid.files").exists())

    @unittest.skipUnless(os.name == "posix", "POSIX symlink fixture")
    def test_project_script_and_parent_links_are_refused(self):
        external = self.root / "external"
        external.mkdir()
        (external / "outside.sh").write_bytes(SCRIPT)
        (self.project / "link.sh").symlink_to(external / "outside.sh")
        for named in (False, True):
            with self.assertRaisesRegex(SpecError, "links/junctions"):
                self.package("invalid", ["bash", "link.sh"], named=named)
        (self.project / "link.sh").unlink()
        (self.project / "linked").symlink_to(external, target_is_directory=True)
        for named in (False, True):
            with self.assertRaisesRegex(SpecError, "links/junctions"):
                self.package("invalid", ["bash", "linked/outside.sh"], named=named)

    def test_static_inspection_does_not_treat_bash_as_import(self):
        self.assertEqual(inspect_project(self.project, [["bash", "step.sh"]], [], [])["imports"], [])

    def test_guide_refuses_absolute_entrypoint_before_input_rewriting(self):
        recipe = self.root / "bad.colaml"
        save_answers(recipe, {"project": str(self.project), "commands": [["bash", self.script.as_posix()]],
            "inputs": [], "environment": {"dependencies": []}, "compute": {"gpu": False},
            "outputs": {"include": ["outputs/**"], "checkpoints": []}, "limits": {"timeout_seconds": 60}})
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            code = main(["package", "--colaml", str(recipe), "--non-interactive", "--output", str(self.root / "bad.colajob")])
        self.assertNotEqual(code, 0)
        self.assertIn("project-relative", err.getvalue())

    def test_old_python_launcher_is_unchanged(self):
        (self.project / "task.py").write_bytes(b"pass\n")
        job = self.package("python-only", ["python", "task.py"])
        generated = job.with_suffix(".files") / "_colabcola"
        self.assertEqual((generated / "launch.py").read_bytes(), _LAUNCHER.encode("utf-8"))
        self.assertFalse((generated / "shell_runtime.py").exists())
        self.assertNotIn("shell_scripts", json.loads((generated / "commands.json").read_text()))

    @unittest.skipUnless(os.name == "posix" and Path("/bin/bash").is_file(), "real Linux Bash execution")
    def test_legacy_launcher_literal_arguments_logs_and_startup_isolation(self):
        job = self.package("legacy", ["bash", "step.sh", *LITERALS])
        snapshot = job.with_suffix(".files")
        startup = self.root / "startup.sh"
        startup.write_bytes(b"touch unwanted-startup\n")
        proc = subprocess.run([sys.executable, str(snapshot / "_colabcola/launch.py")], cwd=snapshot,
            env={**os.environ, "BASH_ENV": str(startup)}, capture_output=True, timeout=10)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode())
        self.assertIn(b"shell stdout", proc.stdout)
        self.assertIn(b"shell stderr", proc.stderr)
        self.assertEqual((snapshot / "args.bin").read_bytes().split(b"\0")[:-1], [a.encode() for a in LITERALS])
        self.assertFalse((snapshot / "unexpected").exists())
        self.assertFalse((snapshot / "unwanted-startup").exists())
        (snapshot / "step.sh").write_bytes(b"touch unexpected\n")
        proc = subprocess.run([sys.executable, str(snapshot / "_colabcola/launch.py")], cwd=snapshot,
            capture_output=True, timeout=10)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(b"frozen SHA-256", proc.stderr)
        self.assertFalse((snapshot / "unexpected").exists())

    @unittest.skipUnless(os.name == "posix" and Path("/bin/bash").is_file(), "real Linux Bash execution")
    def test_named_launcher_activation_cwd_and_failure_attribution(self):
        nested = self.project / "nested"
        nested.mkdir()
        script = nested / "script.sh"
        script.write_bytes(SCRIPT + b'printf "%s" "$CONDA_PREFIX" > active-prefix\nexit 7\n')
        job = self.package("named", ["bash", "script.sh", *LITERALS], named=True, cwd="nested")
        snapshot = job.with_suffix(".files")
        helper = snapshot / "_colabcola"
        config = json.loads((helper / "commands.json").read_text())
        # Add a second frozen step to exercise the actual launcher stopping at the failure.
        # This fixture constructs the local launcher config directly; packaging is tested above.
        config["steps"].append({"name": "later", "environment": "work", "command": ["bash", "step.sh"], "cwd": "."})
        config["shell_scripts"]["step.sh"] = digest(snapshot / "step.sh")
        (helper / "commands.json").write_text(json.dumps(config), encoding="utf-8")
        prefix = self.root / "environment"
        (prefix / "bin").mkdir(parents=True)
        manager = self.root / "manager"
        # Substitute only micromamba's activation boundary; execute the real
        # launcher, environment command helper and Bash child.
        manager.write_text(f"#!{sys.executable}\n" + '''import os, sys
args = sys.argv[1:]
i = args.index("--prefix")
os.environ["CONDA_PREFIX"] = args[i+1]
os.execv(args[i+2], args[i+2:])
''', encoding="utf-8")
        manager.chmod(0o700)
        result_root = self.root / "results"
        result_root.mkdir()
        runtime_config = self.root / "runtime.json"
        runtime_config.write_text(json.dumps({"schema": 1, "job_id": "fixture", "steps": config["steps"],
            "job_root": str(snapshot), "result_root": str(result_root), "env": {}, "environments": {"work": {
                "name": "work", "manager": str(manager), "prefix": str(prefix), "root_prefix": str(self.root / "mamba"),
                "definition": {"env": {}}, "toolchain": {"cuda_home": str(prefix), "nvcc": None, "include_dirs": [], "library_dirs": []}}}}), encoding="utf-8")
        proc = subprocess.run([sys.executable, str(helper / "launch.py")], cwd=snapshot,
            env={**os.environ, "COLABCOLA_ENVIRONMENTS_CONFIG": str(runtime_config)}, capture_output=True, timeout=10)
        self.assertEqual(proc.returncode, 7, proc.stderr.decode())
        self.assertIn(b"shell stdout", proc.stdout)
        self.assertIn(b"shell stderr", proc.stderr)
        self.assertEqual((snapshot / "nested/args.bin").read_bytes().split(b"\0")[:-1], [a.encode() for a in LITERALS])
        self.assertEqual((snapshot / "nested/active-prefix").read_text(), str(prefix))
        evidence = json.loads((result_root / "steps.json").read_text())
        self.assertEqual(evidence["failed_step"], "shell")
        self.assertEqual(evidence["steps"][0]["exit_code"], 7)
        self.assertEqual(evidence["steps"][1]["state"], "not_run")
        self.assertFalse((snapshot / "args.bin").exists())
        self.assertFalse((snapshot / "nested/unexpected").exists())

    @unittest.skipUnless(os.name == "posix" and Path("/bin/bash").is_file(), "real Linux Bash execution")
    def test_runtime_rechecks_symlink_and_content_identity(self):
        before = digest(self.script)
        command = ["bash", "step.sh", *LITERALS]
        self.assertEqual(packaged_bash_argv(command, self.project, expected_sha256=before)[6:], LITERALS)
        self.script.write_bytes(b"exit 1\n")
        with self.assertRaisesRegex(RuntimeError, "frozen SHA-256"):
            packaged_bash_argv(command, self.project, expected_sha256=before)
        self.script.unlink()
        outside = self.root / "outside.sh"
        outside.write_bytes(SCRIPT)
        self.script.symlink_to(outside)
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            packaged_bash_argv(command, self.project, expected_sha256=before)
