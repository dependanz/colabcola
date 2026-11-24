"""Named environment boundaries, setup evidence and finite process lifetimes."""
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from colabcola import environment_runtime as runtime
from colabcola.environment_profiles import MICROMAMBA


def definition(**overrides):
    result = {"python": "3.10", "conda": {"channels": ["conda-forge"], "packages": []},
              "pip": {"packages": [], "requirements": [], "extra_index_urls": [], "no_build_isolation": False}, "env": {}}
    result.update(overrides)
    return result


class EnvironmentBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prefix = self.root / "environment"
        (self.prefix / "bin").mkdir(parents=True)
        self.descriptor = {"name": "test", "prefix": str(self.prefix), "root_prefix": str(self.root / "mamba"),
                           "manager": str(self.root / "micromamba"), "definition": definition()}

    def test_activation_uses_explicit_manager_and_prefix(self):
        command, env = runtime._colabcola_environment_command(self.descriptor, ["python", "run.py", "two words"], self.root)
        self.assertEqual(command[-3:], [str(self.prefix / "bin/python"), "run.py", "two words"])
        self.assertIn("--no-rc", command)
        self.assertIn("--no-env", command)
        self.assertIn("run", command)
        self.assertEqual(command[command.index("--prefix") + 1], str(self.prefix))
        self.assertEqual(env["MAMBA_NO_RC"], "true")
        self.assertNotIn("CONDA_PREFIX", env)

    def test_bare_script_and_module_use_selected_interpreter(self):
        for argv in (["task.py", "--arg"], ["-m", "package.module"]):
            with self.subTest(argv=argv):
                command, _ = runtime._colabcola_environment_command(self.descriptor, argv, self.root)
                self.assertEqual(command[-len(argv)-1:], [str(self.prefix / "bin/python"), *argv])

    def test_ambient_control_secrets_and_python_configuration_do_not_reach_child(self):
        with mock.patch.dict(os.environ, {"EXAMPLE_TOKEN": "secret", "COLABCOLA_TOKEN_HELPER": "secret",
                "PYTHONPATH": "/caller/site-packages", "LD_PRELOAD": "bad.so", "PIP_INDEX_URL": "private",
                "BASH_ENV": "bad.sh", "CC": "unavailable-compiler", "CONDA_PREFIX": "/caller/conda"}):
            _, env = runtime._colabcola_environment_command(self.descriptor, ["python", "task.py"], self.root)
        for name in ("EXAMPLE_TOKEN", "COLABCOLA_TOKEN_HELPER", "PYTHONPATH", "LD_PRELOAD", "PIP_INDEX_URL", "BASH_ENV", "CC", "CONDA_PREFIX"):
            self.assertNotIn(name, env)
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")
        self.assertEqual(env["PIP_CONFIG_FILE"], os.devnull)

    def test_declared_env_precedence_and_project_relative_pythonpath(self):
        self.descriptor["definition"]["env"] = {"COLOR": "blue", "PYTHONPATH": ".:lib"}
        _, env = runtime._colabcola_environment_command(self.descriptor, ["python"], self.root,
                                                        {"COLOR": "red", "MAX_JOBS": "2"})
        self.assertEqual(env["COLOR"], "blue")
        self.assertEqual(env["MAX_JOBS"], "2")
        self.assertEqual(env["PYTHONPATH"], os.pathsep.join([str(self.root), str(self.root / "lib")]))

    def test_proxy_routing_is_available_to_builds_only(self):
        proxy = "http://build-user:private-value@proxy.invalid:8080"
        with mock.patch.dict(os.environ, {"HTTPS_PROXY": proxy}):
            build_env = runtime._colabcola_env_clean_environment(network=True)
            _, workload_env = runtime._colabcola_environment_command(self.descriptor, ["python"], self.root)
            pip_build_env = runtime._colabcola_env_network_environment(workload_env)
        self.assertEqual(build_env["HTTPS_PROXY"], proxy)
        self.assertEqual(pip_build_env["HTTPS_PROXY"], proxy)
        self.assertNotIn("HTTPS_PROXY", workload_env)
        self.assertNotIn("private-value", json.dumps(self.descriptor))

    def test_managed_paths_cannot_be_overridden(self):
        for key in ("PATH", "CUDA_HOME", "PIP_TARGET", "MAMBA_ROOT_PREFIX", "COLABCOLA_SEED", "PYTHONHOME", "LD_LIBRARY_PATH"):
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "forbids override"):
                runtime._colabcola_environment_command(self.descriptor, ["python"], self.root, {key: "x"})
        for path in ("/outside", "..", "lib:../outside", ""):
            with self.subTest(path=path), self.assertRaises((RuntimeError, ValueError)):
                runtime._colabcola_environment_command(self.descriptor, ["python"], self.root, {"PYTHONPATH": path})

    def test_console_lookup_does_not_fall_back_to_host(self):
        with self.assertRaisesRegex(RuntimeError, "not installed"):
            runtime._colabcola_environment_command(self.descriptor, ["git", "--version"], self.root)
        console_name = "example-console.exe" if os.name == "nt" else "example-console"
        script = self.prefix / "bin" / console_name
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o700)
        command, _ = runtime._colabcola_environment_command(self.descriptor, [console_name, "arg"], self.root)
        self.assertEqual(command[-2:], [str(script), "arg"])

    def test_shell_commands_remain_rejected(self):
        for command in ("bash", "sh", "powershell", "/usr/bin/git"):
            with self.subTest(command=command), self.assertRaisesRegex(RuntimeError, "unsupported"):
                runtime._colabcola_environment_command(self.descriptor, [command], self.root)

    @unittest.skipIf(os.name == "nt", "link creation needs privileges on Windows")
    def test_interpreter_and_console_links_cannot_escape_prefix(self):
        (self.prefix / "bin/python").symlink_to(sys.executable)
        with self.assertRaisesRegex(RuntimeError, "escaped"):
            runtime._colabcola_environment_command(self.descriptor, ["python"], self.root)
        (self.prefix / "bin/example-console").symlink_to(sys.executable)
        with self.assertRaisesRegex(RuntimeError, "escaped"):
            runtime._colabcola_environment_command(self.descriptor, ["example-console"], self.root)

    def test_toolchain_collects_target_headers_and_wheel_dependencies(self):
        paths = ["include", "targets/x86_64-linux/include", "targets/x86_64-linux/lib",
                 "targets/x86_64-linux/lib/stubs", "lib/python3.10/site-packages/nvidia/cublas/include",
                 "lib/python3.10/site-packages/nvidia/cusparse/include", "lib/python3.10/site-packages/nvidia/cusolver/include",
                 "lib/python3.10/site-packages/nvidia/cublas/lib", "lib/python3.10/site-packages/torch/lib"]
        for name in paths:
            (self.prefix / name).mkdir(parents=True, exist_ok=True)
        nvcc = self.prefix / "bin/nvcc"
        nvcc.write_text("compiler")
        toolchain = runtime._colabcola_env_toolchain(self.prefix)
        self.assertEqual(toolchain["nvcc"], str(nvcc))
        self.assertIn(str(self.prefix / paths[1]), toolchain["include_dirs"])
        self.assertIn(str(self.prefix / paths[5]), toolchain["include_dirs"])
        self.assertIn(str(self.prefix / paths[-1]), toolchain["library_dirs"])
        self.assertNotIn(str(self.prefix / paths[3]), toolchain["library_dirs"])
        self.descriptor["toolchain"] = toolchain
        _, env = runtime._colabcola_environment_command(self.descriptor, ["python"], self.root)
        self.assertEqual(env["CUDA_HOME"], str(self.prefix))
        self.assertEqual(env["CUDACXX"], str(nvcc))
        self.assertIn(str(self.prefix / paths[1]), env["CPATH"])
        self.assertNotIn("stubs", env["LD_LIBRARY_PATH"])

    def test_cuda_version_mismatch_is_actionable(self):
        runtime._colabcola_env_cuda_compatibility({"cuda": "12.8"}, "12.8", {"torch": {"cuda": "12.8"}})
        with self.assertRaisesRegex(RuntimeError, "declared version"):
            runtime._colabcola_env_cuda_compatibility({"cuda": "12.8"}, "13.0")
        with self.assertRaisesRegex(RuntimeError, "torch CUDA"):
            runtime._colabcola_env_cuda_compatibility({}, "13.0", {"torch": {"cuda": "12.8"}})

    def test_identity_probe_rejects_wrong_prefix_activation_and_version(self):
        facts = {"prefix": str(self.prefix), "conda_prefix": str(self.prefix),
                 "executable": str(self.prefix / "bin/python"), "python": "3.10.20", "user_site_enabled": False}
        changes = ({"prefix": str(self.root / "other")}, {"conda_prefix": None},
                   {"executable": str(self.root / "external/python")}, {"python": "3.12.1"},
                   {"user_site_enabled": True})
        for change in changes:
            with self.subTest(change=change), mock.patch.object(runtime, "_colabcola_env_run", return_value=json.dumps(dict(facts, **change))):
                with self.assertRaisesRegex(RuntimeError, "boundary|external interpreter|Python version"):
                    runtime._colabcola_env_probe(self.descriptor, self.root, None, None)


class EnvironmentBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_checksum_verified_before_executable_permission(self):
        data = b"executable test bytes"
        pin = dict(MICROMAMBA, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        destination = self.root / "manager"
        with mock.patch.object(runtime.urllib.request, "urlopen", return_value=io.BytesIO(data)):
            runtime._colabcola_env_download(pin, destination, time.time() + 5)
        self.assertEqual(destination.read_bytes(), data)
        if os.name != "nt":
            self.assertTrue(destination.stat().st_mode & 0o100)

    def test_bad_digest_and_oversize_download_leave_no_binary(self):
        for payload in (b"bad", b"too many bytes"):
            pin = dict(MICROMAMBA, bytes=3, sha256=hashlib.sha256(b"good").hexdigest())
            destination = self.root / "manager"
            with self.subTest(payload=payload), mock.patch.object(runtime.urllib.request, "urlopen", return_value=io.BytesIO(payload)):
                with self.assertRaisesRegex(RuntimeError, "SHA-256|byte count"):
                    runtime._colabcola_env_download(pin, destination, time.time() + 5)
            self.assertFalse(destination.exists())

    def test_expired_deadline_refuses_download(self):
        with mock.patch.object(runtime.urllib.request, "urlopen") as fetch, self.assertRaises(TimeoutError):
            runtime._colabcola_env_download(MICROMAMBA, self.root / "manager", time.time() - 1)
        fetch.assert_not_called()

    def test_setup_failure_preserves_stage_and_environment(self):
        result = self.root / "result"
        with mock.patch.object(runtime.platform, "system", return_value="Linux"), mock.patch.object(runtime.platform, "machine", return_value="x86_64"), \
             mock.patch.object(runtime, "_colabcola_env_download"), mock.patch.object(runtime, "_colabcola_env_run", side_effect=RuntimeError("solver failed")):
            with self.assertRaisesRegex(RuntimeError, "solver failed"):
                runtime._colabcola_prepare_environments({"first": definition(), "second": definition()}, MICROMAMBA,
                    self.root, self.root / "build", result, time.time() + 20)
        evidence = json.loads((result / "environments.json").read_text())
        self.assertEqual(evidence["state"], "failed")
        self.assertEqual(evidence["failed_environment"], "first")
        self.assertEqual(evidence["environments"]["first"]["stages"][0]["state"], "failed")
        self.assertEqual(evidence["environments"]["second"]["state"], "pending")
        self.assertGreaterEqual(evidence["elapsed_seconds"], 0)

    def test_package_then_requirement_files_install_in_declared_order(self):
        envdef = definition()
        envdef["pip"].update(packages=["example-base==1"], requirements=["first.txt", "second.txt"],
                             index_url="https://packages.example/simple", no_build_isolation=True)
        commands = []
        def run(argv, **kwargs):
            commands.append(list(argv))
            return "[]" if "list" in argv else ""
        resolved = {"python": "3.10.20", "packages": []}
        with mock.patch.object(runtime.platform, "system", return_value="Linux"), mock.patch.object(runtime.platform, "machine", return_value="x86_64"), \
             mock.patch.object(runtime, "_colabcola_env_download"), mock.patch.object(runtime, "_colabcola_env_run", side_effect=run), \
             mock.patch.object(runtime, "_colabcola_env_probe", return_value=resolved):
            descriptors, facts = runtime._colabcola_prepare_environments({"example": envdef}, MICROMAMBA,
                self.root, self.root / "build", self.root / "result", time.time() + 20)
        installs = [c for c in commands if "install" in c]
        self.assertEqual(len(installs), 3)
        self.assertEqual(installs[0][-1], "example-base==1")
        self.assertEqual(installs[1][-2:], ["--requirement", "first.txt"])
        self.assertEqual(installs[2][-2:], ["--requirement", "second.txt"])
        self.assertTrue(all("--no-build-isolation" in c for c in installs))
        self.assertTrue(all("--index-url" in c for c in installs))
        self.assertEqual(facts["state"], "ready")
        self.assertEqual(facts["environments"]["example"]["facts"]["conda_packages"], [])
        self.assertEqual(descriptors["example"]["definition"], envdef)

    def test_declared_cuda_without_toolkit_fails_before_pip(self):
        envdef = definition(cuda="12.8")
        envdef["pip"]["packages"] = ["example-native==1"]
        with mock.patch.object(runtime.platform, "system", return_value="Linux"), mock.patch.object(runtime.platform, "machine", return_value="x86_64"), \
             mock.patch.object(runtime, "_colabcola_env_download"), mock.patch.object(runtime, "_colabcola_env_run", return_value="") as run, \
             mock.patch.object(runtime, "_colabcola_env_probe", return_value={"python": "3.10.20"}):
            with self.assertRaisesRegex(RuntimeError, "own nvcc"):
                runtime._colabcola_prepare_environments({"example": envdef}, MICROMAMBA,
                    self.root, self.root / "build", self.root / "result", time.time() + 20)
        self.assertEqual(run.call_count, 1)

    def test_log_bound_does_not_expand_after_limit(self):
        log = self.root / "environment.log"
        with mock.patch.object(runtime, "_COLABCOLA_ENV_LOG_BYTES", 512):
            runtime._colabcola_env_log(log, b"x" * 600)
            first = log.read_bytes()
            runtime._colabcola_env_log(log, b"y" * 500)
        self.assertEqual(log.stat().st_size, 512)
        self.assertIn(b"log limit", first)
        self.assertEqual(first, log.read_bytes())


@unittest.skipUnless(os.name == "posix", "nonblocking pipe and process-group checks require POSIX")
class EnvironmentProcessTests(unittest.TestCase):
    def test_process_failure_keeps_diagnostic_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "exit 3.*example failure"):
                runtime._colabcola_env_run([sys.executable, "-c", "print('example failure'); raise SystemExit(3)"],
                    cwd=directory, env=os.environ.copy(), timeout=5, label="install")

    def test_execution_deadline_terminates_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            begin = time.monotonic()
            with self.assertRaises(TimeoutError):
                runtime._colabcola_env_run([sys.executable, "-c", "import time; time.sleep(30)"],
                    cwd=directory, env=os.environ.copy(), timeout=0.2, label="install")
            self.assertLess(time.monotonic() - begin, 4)

    def test_inherited_pipe_cannot_outlive_build_deadline(self):
        code = "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])"
        with tempfile.TemporaryDirectory() as directory:
            begin = time.monotonic()
            with self.assertRaises(TimeoutError):
                runtime._colabcola_env_run([sys.executable, "-c", code], cwd=directory,
                    env=os.environ.copy(), timeout=0.3, label="install")
            self.assertLess(time.monotonic() - begin, 4)


if __name__ == "__main__":
    unittest.main()
