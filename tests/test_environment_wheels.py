"""Hash-bound local wheels are installed only in their declared environment."""
from __future__ import annotations

import base64
import contextlib
import csv
import hashlib
import io
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import time
import unittest
from unittest import mock
import venv
import zipfile

from colabcola import build_bundle, build_notebook, environment_runtime as runtime
from colabcola.environments import normalize_environments
from colabcola.errors import SpecError
from colabcola.invocation import load_job_invocation
from colabcola.lifecycle import verify_final
from colabcola.packager import package, parser
from colabcola.package_guide import save_answers
from colabcola.cli import main

WHEEL = "wheels/cc_wheel_probe-1.0-py3-none-any.whl"


def write_wheel(path, *, value=17, native=None, tag="py3-none-any"):
    """Small valid wheel with RECORD hashes; no build dependency or index needed."""
    dist = "cc_wheel_probe-1.0.dist-info"
    members = {
        "cc_wheel_probe.py": f"VALUE = {value}\n".encode(),
        dist + "/METADATA": b"Metadata-Version: 2.1\nName: cc-wheel-probe\nVersion: 1.0\n\n",
        dist + "/WHEEL": ("Wheel-Version: 1.0\nGenerator: offline-fixture\nRoot-Is-Purelib: "
                          + ("false" if native else "true") + "\nTag: " + tag + "\n").encode(),
    }
    if native:
        members["cc_wheel_probe.py"] = b"from _cc_wheel_probe import value\nVALUE = value()\n"
        members[native.name] = native.read_bytes()
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name, body in sorted(members.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).decode().rstrip("=")
        writer.writerow([name, "sha256=" + digest, len(body)])
    writer.writerow([dist + "/RECORD", "", ""])
    members[dist + "/RECORD"] = record.getvalue().encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in sorted(members.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            archive.writestr(info, body)


def wheel_record(path, relative=WHEEL):
    return {"path": relative, "role": "wheel", "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class NamedWheelTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.project = self.root / "project"
        self.project.mkdir()
        (self.project / "stage.py").write_bytes(b"pass\n")
        self.wheel = self.project / WHEEL
        write_wheel(self.wheel)
        self.environments = {"train": {"python": "3.12", "pip": {"wheels": [WHEEL]}},
                             "other": {"python": "3.12"}}
        self.steps = [{"name": "train", "environment": "train", "command": ["python", "stage.py"]}]

    def pack(self, stem="job", *extra):
        job = self.root / (stem + ".colajob")
        args = parser().parse_args(["--project", str(self.project), "--output", str(job), *extra])
        args.environments = self.environments
        args.steps = self.steps
        with mock.patch("subprocess.run", side_effect=AssertionError("packaging executed code")), \
             mock.patch("urllib.request.urlopen", side_effect=AssertionError("packaging accessed network")):
            package(args)
        return job

    def test_declarations_and_invalid_paths_or_names(self):
        self.assertEqual(normalize_environments(self.environments)["train"]["pip"]["wheels"], [WHEEL])
        self.assertNotIn("wheels", normalize_environments(self.environments)["other"]["pip"])
        for bad in (True, WHEEL, [None], ["../" + WHEEL], ["/" + WHEEL], ["C:/" + WHEEL],
                    [WHEEL.replace("/", "\\")], ["./" + WHEEL], ["wheels/bad.whl"], ["wheels/foo.py"],
                    ["_colabcola/" + WHEEL.rsplit("/", 1)[-1]], [WHEEL, WHEEL],
                    [WHEEL, "wheels/cc_wheel_probe-2.0-py3-none-any.whl"]):
            with self.subTest(bad=bad), self.assertRaises(SpecError):
                normalize_environments({"train": {"python": "3.12", "pip": {"wheels": bad}}})

    def test_snapshot_and_manifest_identity_and_shared_wheel(self):
        self.environments["other"]["pip"] = {"wheels": [WHEEL]}
        job = self.pack()
        first = build_bundle(load_job_invocation(job))
        manifest = json.loads(first.canonical_manifest_bytes)
        self.assertEqual(manifest["python"]["embedded_wheels"], [])
        self.assertEqual(manifest["environments"]["train"]["pip"]["wheels"], [WHEEL])
        self.assertEqual(manifest["environments"]["other"]["pip"]["wheels"], [WHEEL])
        records = [x for x in manifest["files"] if x["path"] == WHEEL]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0], dict(wheel_record(self.wheel), mode="0644"))
        write_wheel(self.wheel, value=21)
        self.assertEqual(build_bundle(load_job_invocation(job)).job_id, first.job_id)
        self.assertNotEqual(build_bundle(load_job_invocation(self.pack("changed"))).job_id, first.job_id)

    def test_missing_excluded_data_wheel_and_raw_role_are_rejected(self):
        with self.assertRaisesRegex(SpecError, "environment wheel"):
            self.pack("excluded", "--exclude", WHEEL)
        self.wheel.unlink()
        other = self.root / self.wheel.name
        write_wheel(other)
        with self.assertRaisesRegex(SpecError, "environment wheel"):
            self.pack("external", "--data", str(other) + "=" + WHEEL)
        write_wheel(self.wheel)
        job = self.pack()
        raw = json.loads(job.read_text())
        for file in raw["payload"]:
            if file["target"] == WHEEL:
                file["role"] = "source"
        job.write_text(json.dumps(raw))
        with self.assertRaisesRegex(SpecError, "role 'wheel'"):
            build_bundle(load_job_invocation(job))
        self.assertFalse((self.root / "excluded.colajob").exists())
        self.assertFalse((self.root / "external.files").exists())

    @unittest.skipUnless(os.name == "posix", "symlink fixture")
    def test_wheel_link_rejected(self):
        self.wheel.unlink()
        other = self.root / self.wheel.name
        write_wheel(other)
        self.wheel.symlink_to(other)
        with self.assertRaisesRegex(SpecError, "links/junctions"):
            self.pack()

    def test_colaml_save_replay_preserves_wheels(self):
        config = {"project": str(self.project), "name": "wheel-test", "environments": self.environments,
                  "steps": self.steps, "inputs": [], "compute": {"gpu": False},
                  "outputs": {"include": ["outputs/**"], "checkpoints": []},
                  "limits": {"timeout_seconds": 60}}
        recipe = self.root / "recipe.colaml"
        save_answers(recipe, config)
        saved = self.root / "saved.colaml"
        ids = []
        for n, current in enumerate((recipe, saved)):
            job = self.root / f"recipe-{n}.colajob"
            args = ["package", "--colaml", str(current), "--non-interactive", "--output", str(job)]
            if n == 0:
                args += ["--save-colaml", str(saved)]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(main(args), 0, err.getvalue())
            ids.append(build_bundle(load_job_invocation(job)).job_id)
        self.assertEqual(ids[0], ids[1])

    def test_named_wheel_schema_agrees_with_source_parser(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("optional JSON schema validator unavailable")
        job = self.pack()
        schema = json.loads((Path(__file__).parents[1] / "schemas/job-v1.schema.json").read_text())
        mapping = json.loads(job.read_text())
        jsonschema.Draft202012Validator(schema).validate(mapping)
        mapping["environments"]["train"]["pip"]["wheels"] = ["wheels/not-a-wheel.txt"]
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(schema).validate(mapping)

    def test_integrity_failure_is_attributed_to_environment_setup(self):
        definition = normalize_environments(self.environments)["train"]
        result_root = self.root / "result"
        with mock.patch.object(runtime.platform, "system", return_value="Linux"), \
             mock.patch.object(runtime.platform, "machine", return_value="x86_64"), \
             mock.patch.object(runtime, "_colabcola_env_download"), \
             mock.patch.object(runtime, "_colabcola_env_run", return_value="") as run, \
             mock.patch.object(runtime, "_colabcola_env_probe", return_value={"python": "3.12.0"}):
            with self.assertRaisesRegex(RuntimeError, "frozen identity"):
                runtime._colabcola_prepare_environments({"train": definition}, {}, self.project,
                    self.root / "envs", result_root, time.time() + 30, wheel_records={})
        result = json.loads((result_root / "environments.json").read_text())
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["failed_environment"], "train")
        self.assertEqual(result["environments"]["train"]["stages"][-1]["name"], "embedded-wheels")
        self.assertEqual(result["environments"]["train"]["stages"][-1]["state"], "failed")
        self.assertFalse(any("install" in call.args[0] for call in run.call_args_list))

    def test_install_command_offline_targeted_and_verified(self):
        definition = normalize_environments(self.environments)["train"]
        descriptor = {"name": "train", "prefix": str(self.root / "environment"),
                      "manager": str(self.root / "manager"), "root_prefix": str(self.root / "mamba"),
                      "definition": definition}
        record = wheel_record(self.wheel)
        with mock.patch.object(runtime, "_colabcola_env_run") as run:
            verified = runtime._colabcola_env_install_wheels(descriptor, self.project, {WHEEL: record}, None, None)
        self.assertEqual(verified, [{k: v for k, v in record.items() if k != "role"}])
        argv = run.call_args.args[0]
        self.assertEqual(argv[argv.index("--prefix") + 1], descriptor["prefix"])
        self.assertIn(str(Path(descriptor["prefix"]) / "bin/python"), argv)
        self.assertIn("--no-index", argv)
        self.assertIn("--no-deps", argv)
        self.assertIn("--force-reinstall", argv)
        self.assertEqual(argv[-1], str(self.wheel))
        self.assertNotIn("PIP_INDEX_URL", run.call_args.kwargs["env"])
        for bad in ({}, {WHEEL: dict(record, role="source")}, {WHEEL: dict(record, sha256="0" * 64)},
                    {WHEEL: dict(record, size=record["size"] + 1)}):
            with self.subTest(bad=bad), mock.patch.object(runtime, "_colabcola_env_run") as run:
                with self.assertRaisesRegex(RuntimeError, "frozen identity"):
                    runtime._colabcola_env_install_wheels(descriptor, self.project, bad, None, None)
                run.assert_not_called()
        with self.assertRaises(TimeoutError):
            runtime._colabcola_env_wheel_inputs(definition, self.project, {WHEEL: record}, time.time() - 1)

    @unittest.skipUnless(sys.platform == "linux", "real Linux wheel installation rehearsal")
    def test_compiled_notebook_installs_actual_wheel_in_only_one_environment(self):
        self._execute_real_wheel(WHEEL)

    @unittest.skipUnless(sys.platform == "linux" and shutil.which("cc")
                        and Path(sysconfig.get_path("include"), "Python.h").is_file(),
                        "native wheel fixture requires Linux C compiler and Python headers")
    def test_compiled_notebook_installs_actual_native_wheel(self):
        source = self.root / "probe.c"
        source.write_text(r'''#include <Python.h>
static PyObject *value(PyObject *self, PyObject *args) { return PyLong_FromLong(17); }
static PyMethodDef methods[] = {{"value", value, METH_NOARGS, NULL}, {NULL, NULL, 0, NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_cc_wheel_probe", NULL, -1, methods};
PyMODINIT_FUNC PyInit__cc_wheel_probe(void) { return PyModule_Create(&module); }
''', encoding="utf-8")
        tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        wheel_tag = f"{tag}-{tag}-{platform_tag}"
        native = self.root / ("_cc_wheel_probe" + sysconfig.get_config_var("EXT_SUFFIX"))
        subprocess.run(["cc", "-shared", "-fPIC", "-I" + sysconfig.get_path("include"),
                        str(source), "-o", str(native)], check=True, capture_output=True, timeout=30)
        self.wheel.unlink()
        relative = f"wheels/cc_wheel_probe-1.0-{wheel_tag}.whl"
        self.wheel = self.project / relative
        write_wheel(self.wheel, native=native, tag=wheel_tag)
        self.environments["train"]["pip"]["wheels"] = [relative]
        self._execute_real_wheel(relative)

    def _execute_real_wheel(self, relative):
        # Provision real local venvs instead of network-dependent conda solves.
        # All wheel verification/install, result collection, and step execution
        # use the actual generated notebook and runtime code.
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        for definition in self.environments.values():
            definition["python"] = version
        self.steps.append({"name": "other", "environment": "other", "command": ["python", "stage.py", "other"]})
        (self.project / "stage.py").write_text('''import importlib.util, sys, json, os
from pathlib import Path
Path("outputs").mkdir(exist_ok=True)
assert Path(sys.prefix) == Path(os.environ["CONDA_PREFIX"])
if len(sys.argv) == 1:
    import cc_wheel_probe
    assert cc_wheel_probe.VALUE == 17
    Path("outputs/train.json").write_text(json.dumps({"value": cc_wheel_probe.VALUE}))
else:
    assert importlib.util.find_spec("cc_wheel_probe") is None
    Path("outputs/other.json").write_text('{"wheel_absent": true}')
''', encoding="utf-8")
        job = self.pack("real", "--timeout", "180", "--max-seconds", "240", "--artifact", "outputs/**")
        artifact = build_notebook(build_bundle(load_job_invocation(job)))
        def manager_download(spec, destination, deadline):
            destination.write_bytes(b'''#!/bin/bash
set -eu
while [[ "$1" != --prefix ]]; do shift; done
shift
export CONDA_PREFIX="$1"
shift
exec "$@"
''')
            destination.chmod(0o700)
        def manager_run(argv, **kwargs):
            if "create" in argv and "--prefix" in argv:
                prefix = argv[argv.index("--prefix") + 1]
                venv.EnvBuilder(with_pip=True, symlinks=False).create(prefix)
                # Hosted CPython can link libpython dynamically. Preserve that
                # library locally in the fixture when the runtime cleans LD_LIBRARY_PATH.
                libdir = sysconfig.get_config_var("LIBDIR")
                for name in {sysconfig.get_config_var("LDLIBRARY"), sysconfig.get_config_var("INSTSONAME")}:
                    if libdir and name and (Path(libdir) / name).is_file():
                        destination = Path(prefix) / "lib" / name
                        if not destination.exists():
                            shutil.copyfile(Path(libdir) / name, destination)
                return ""
            if "list" in argv and "--json" in argv:
                return "[]"
            return runtime._colabcola_env_run(argv, **kwargs)
        namespace = {"COLABCOLA_CONTENT_ROOT_OVERRIDE": str(self.root / "content"),
                     "_test_download": manager_download, "_test_run": manager_run}
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            for cell in json.loads(artifact.notebook_bytes)["cells"]:
                if cell["cell_type"] != "code":
                    continue
                source = cell["source"]
                if cell["metadata"]["colabcola"]["group"] == "environment":
                    source = source.replace("# Build all declared environments as a separate stage before any workload step.",
                        "_colabcola_env_download = _test_download\n_colabcola_env_run = _test_run", 1)
                exec(source, namespace)
        result_root = self.root / "content" / "colabcola" / "results" / artifact.bundle.job_id
        result = verify_final(result_root, artifact.bundle.job_id)
        self.assertEqual(result["terminal_state"], "success", result.get("failure"))
        records = result["environment"]["environments"]["environments"]
        self.assertEqual(records["train"]["embedded_wheels"], [{k: v for k, v in wheel_record(self.wheel, relative).items() if k != "role"}])
        self.assertNotIn("embedded_wheels", records["other"])
        with zipfile.ZipFile(result_root / "artifacts.zip") as archive:
            self.assertEqual(json.loads(archive.read("outputs/train.json")), {"value": 17})
            self.assertEqual(json.loads(archive.read("outputs/other.json")), {"wheel_absent": True})
