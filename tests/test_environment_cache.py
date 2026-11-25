"""Integration of frozen cache recipes, real prefixes and scoped Drive transfers.

Only Colab allocation, micromamba provisioning and the remote HTTP service are
substituted. The generated notebook, Python subprocesses, archive format,
DriveStore protocol and result verification are production code.
"""
from __future__ import annotations

import contextlib
import gzip
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import sysconfig
import tempfile
import time
import unittest
import urllib.error
import urllib.parse
import venv
import zipfile
from pathlib import Path
from unittest import mock

from colabcola import build_bundle, build_notebook, environment_cache as cache
from colabcola import environment_runtime as env_runtime, lifecycle_io as lio, lifecycle_worker as worker
from colabcola.environments import normalize_environments
from colabcola.errors import SpecError
from colabcola.invocation import load_job_invocation
from colabcola.lifecycle import verify_final
from colabcola.packager import package, parser
from test_environment_wheels import WHEEL, write_wheel


class FakeDriveHTTP:
    """In-memory HTTP endpoint for real DriveStore list/upload/readback requests."""
    def __init__(self):
        self.objects = {}
        self.events = []
        self.tokens = []
        self.fail_upload = None
        self.corrupt_readback = False

    def request(self, req, timeout):
        assert 0 < timeout <= 30
        token = req.get_header("Authorization")
        assert token in ("Bearer fixture-token", "Bearer rotated-fixture-token")
        self.tokens.append(token)
        url = urllib.parse.urlparse(req.full_url)
        assert url.hostname == "www.googleapis.com"
        params = urllib.parse.parse_qs(url.query)
        if req.method == "POST":
            assert params["uploadType"] == ["multipart"]
            boundary = req.get_header("Content-type").split("boundary=", 1)[1].encode()
            metadata_raw, payload = req.data.split(b"\r\n--" + boundary + b"\r\n", 1)
            meta = json.loads(metadata_raw.split(b"\r\n\r\n", 1)[1])
            body = payload.split(b"\r\n\r\n", 1)[1].removesuffix(b"\r\n--" + boundary + b"--\r\n")
            kind = meta["appProperties"]["cc_kind"]
            if self.fail_upload == kind:
                raise urllib.error.HTTPError(req.full_url, 403, "fixture error", {}, None)
            file_id = "fixture_" + str(len(self.objects) + 1).zfill(5)
            assert meta["appProperties"]["cc_sha"] == lio.digest(body)
            self.objects[file_id] = (meta, body)
            self.events.append(("put", kind, file_id))
            return io.BytesIO(lio.encoded({"id": file_id}))
        if params.get("alt") == ["media"]:
            file_id = url.path.rsplit("/", 1)[1]
            meta, body = self.objects[file_id]
            self.events.append(("get", meta["appProperties"]["cc_kind"], file_id))
            if self.corrupt_readback:
                return io.BytesIO(body + b"broken")
            return io.BytesIO(body)
        query = params["q"][0]
        parent = re.search(r"'([^']+)' in parents", query).group(1)
        run_id = re.search(r"key='cc_run' and value='([^']+)'", query).group(1)
        kind = re.search(r"key='cc_kind' and value='([^']+)'", query).group(1)
        self.events.append(("list", kind, run_id))
        records = []
        for file_id, (meta, body) in self.objects.items():
            props = meta["appProperties"]
            if meta["parents"] == [parent] and props["cc_run"] == run_id and props["cc_kind"] == kind:
                records.append({"id": file_id, "size": str(len(body)), "name": meta["name"],
                                "appProperties": props, "createdTime": file_id})
        return io.BytesIO(lio.encoded({"files": records}))

    def factory(self, *args, **kwargs):
        return lio.DriveStore(*args, request=self.request, **kwargs)

    def storage(self, parent="private-parent", token=lambda: "fixture-token"):
        return lio.EnvironmentCacheStorage(parent, token, store_factory=self.factory)


class CacheIdentityTests(unittest.TestCase):
    def test_declaration_and_key_invalidation(self):
        from copy import deepcopy
        definition = normalize_environments({"work": {"python": "3.12", "cache": True,
            "pip": {"wheels": [WHEEL]}}})["work"]
        self.assertIs(definition["cache"], True)
        self.assertNotIn("cache", normalize_environments({"work": {"python": "3.12"}})["work"])
        for value in ("true", 1, None, {}):
            with self.subTest(value=value), self.assertRaises(SpecError):
                normalize_environments({"work": {"python": "3.12", "cache": value}})
        manifest = {"environment_manager": {"sha256": "1" * 64}, "job_id": "2" * 64,
                    "files": [{"path": WHEEL, "role": "wheel", "size": 9, "sha256": "3" * 64}]}
        host = {"release": "release-A", "gpus": [], "libc": ["glibc", "2.35"]}
        def key(d=definition, m=manifest, h=host, root="/content/prefixes", impl="4" * 64):
            return cache._cc_cache_hash(cache._cc_cache_identity("work", d, m, h, root, impl))
        original = key()
        for changed in (dict(definition, python="3.13"), dict(definition, env={"MODE": "other"})):
            self.assertNotEqual(original, key(d=changed))
        for h in (dict(host, release="release-B"), dict(host, libc=["glibc", "2.36"]),
                  dict(host, gpus=[{"compute_capability": "8.0", "driver": "555.1"}])):
            self.assertNotEqual(original, key(h=h))
        changed = deepcopy(manifest); changed["files"][0]["sha256"] = "5" * 64
        self.assertNotEqual(original, key(m=changed))
        changed = deepcopy(manifest); changed["environment_manager"]["sha256"] = "5" * 64
        self.assertNotEqual(original, key(m=changed))
        self.assertNotEqual(original, key(root="/other/prefixes"))
        self.assertNotEqual(original, key(impl="5" * 64))
        self.assertEqual(original, key(m=dict(manifest, job_id="9" * 64)))
        local = deepcopy(definition); local["pip"]["packages"] = ["./source"]
        self.assertNotEqual(key(d=local), key(d=local, m=dict(manifest, job_id="9" * 64)))
        with mock.patch.dict(os.environ, {"COLAB_RELEASE_TAG": ""}):
            with self.assertRaisesRegex(cache.EnvironmentCacheError, "image identity unavailable"):
                cache._cc_cache_runtime_fingerprint({"accelerator": "cpu"}, {"gpus": []}, time.time() + 10)


class CacheStorageTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "source"; self.source.mkdir()
        self.key = "a" * 64
        self.body = b"archive fixture bytes" * 97
        (self.source / "environment.ccenv.gz").write_bytes(self.body)
        (self.source / "manifest.json").write_bytes(lio.encoded({"key": self.key, "archive": {
            "bytes": len(self.body), "sha256": lio.digest(self.body)}}))
        self.drive = FakeDriveHTTP()
        self.deadline = time.time() + 30

    def test_commit_last_roundtrip_and_namespace_isolation(self):
        storage = self.drive.storage()
        with mock.patch.object(lio, "CHUNK_BYTES", 128):
            self.assertFalse(storage.load(self.key, self.root / "missing", self.deadline))
            ref = storage.save(self.key, self.source, self.deadline)
            commit = json.loads(self.drive.objects[ref["id"]][1])
            self.assertEqual(commit["cache_key"], self.key)
            self.assertGreater(len(commit["members"]["environment.ccenv.gz"]["chunks"]), 1)
            commit_put = self.drive.events.index(("put", "environment-cache-v1-commit", ref["id"]))
            for member in commit["members"].values():
                for chunk in member["chunks"]:
                    self.assertLess(self.drive.events.index(("get", "environment-cache-v1-member", chunk["id"])), commit_put)
            # New storage object and rotated lease model a distinct controller.
            warm = self.drive.storage(token=lambda: "rotated-fixture-token")
            target = self.root / "restored"
            self.assertTrue(warm.load(self.key, target, self.deadline))
            self.assertEqual((target / "environment.ccenv.gz").read_bytes(), self.body)
            self.assertEqual((target / "manifest.json").read_bytes(), (self.source / "manifest.json").read_bytes())
            self.assertFalse(warm.load("b" * 64, self.root / "wrong-key", self.deadline))
            self.assertFalse(self.drive.storage("other-parent").load(self.key, self.root / "wrong-parent", self.deadline))
            self.assertIn("Bearer rotated-fixture-token", self.drive.tokens)

    def test_failed_publication_and_readback_leave_no_commit(self):
        storage = self.drive.storage()
        self.drive.fail_upload = "environment-cache-v1-commit"
        with self.assertRaisesRegex(lio.LifecycleError, "Drive HTTP 403"):
            storage.save(self.key, self.source, self.deadline)
        self.assertFalse(storage.load(self.key, self.root / "missing", self.deadline))
        self.drive.fail_upload = None
        self.drive.corrupt_readback = True
        with self.assertRaisesRegex(lio.LifecycleError, "integrity mismatch"):
            storage.save(self.key, self.source, self.deadline)
        self.assertFalse(any(meta["appProperties"]["cc_kind"].endswith("-commit") for meta, _ in self.drive.objects.values()))

    def test_corrupt_chunk_and_wrong_namespace_ref_rejected(self):
        storage = self.drive.storage()
        ref = storage.save(self.key, self.source, self.deadline)
        meta, body = self.drive.objects[ref["id"]]
        commit = json.loads(body)
        chunk = commit["members"]["environment.ccenv.gz"]["chunks"][0]
        chunk_meta, chunk_body = self.drive.objects[chunk["id"]]
        self.drive.objects[chunk["id"]] = (chunk_meta, b"corrupt" + chunk_body)
        target = self.root / "bad"
        with self.assertRaises(lio.LifecycleError):
            storage.load(self.key, target, self.deadline)
        self.assertFalse(target.exists())
        self.assertFalse(list(self.root.glob(".download-*")))
        self.drive.objects[chunk["id"]] = (dict(chunk_meta, parents=["other-parent"]), chunk_body)
        with self.assertRaisesRegex(lio.LifecycleError, "outside its Drive namespace"):
            storage.load(self.key, target, self.deadline)
        self.assertFalse(target.exists())

    def test_expired_deadline_and_insufficient_disk(self):
        storage = self.drive.storage()
        with self.assertRaises(TimeoutError):
            storage.save(self.key, self.source, time.time() - 1)
        self.assertEqual(self.drive.events, [])
        storage.save(self.key, self.source, self.deadline)
        with mock.patch("shutil.disk_usage", return_value=shutil._ntuple_diskusage(100, 99, 1)):
            with self.assertRaisesRegex(lio.LifecycleError, "insufficient.*disk"):
                storage.load(self.key, self.root / "no-disk", self.deadline)
        self.assertFalse((self.root / "no-disk").exists())


@unittest.skipUnless(sys.platform == "linux", "POSIX prefix archive contract")
class CacheArchiveTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.prefix = self.root / "prefix"; self.prefix.mkdir()
        self.archive = self.root / "archive.gz"
        self.deadline = time.time() + 30

    def test_permissions_links_and_roundtrip(self):
        (self.prefix / "bin").mkdir()
        script = self.prefix / "bin" / "tool"
        script.write_bytes(b"#!/bin/sh\nprintf success\n"); script.chmod(0o750)
        (self.prefix / "link").symlink_to("bin/tool")
        record = cache._cc_cache_create_archive(self.prefix, self.archive, self.deadline)
        shutil.rmtree(self.prefix)
        cache._cc_cache_extract_archive(self.archive, self.prefix, record, self.deadline)
        self.assertEqual((script.stat().st_mode & 0o777), 0o750)
        self.assertEqual(os.readlink(self.prefix / "link"), "bin/tool")
        self.assertEqual(subprocess.check_output([str(self.prefix / "link")]), b"success")

    def test_archive_rejects_escaping_links_and_special_files(self):
        link = self.prefix / "bad"
        for target in ("../outside", "/outside"):
            link.symlink_to(target)
            with self.assertRaises(cache.EnvironmentCacheError):
                cache._cc_cache_create_archive(self.prefix, self.archive, self.deadline)
            self.archive.unlink(missing_ok=True); link.unlink()
        os.mkfifo(link)
        with self.assertRaisesRegex(cache.EnvironmentCacheError, "special file"):
            cache._cc_cache_create_archive(self.prefix, self.archive, self.deadline)

    def write_malicious(self, headers):
        self.archive.unlink(missing_ok=True)
        with gzip.open(self.archive, "wb") as out:
            out.write(cache._CC_CACHE_MAGIC)
            for header in headers:
                raw = cache._cc_cache_json(header)
                out.write(struct.pack(">I", len(raw))); out.write(raw)
            out.write(struct.pack(">I", 0))
        return dict(cache._cc_cache_file_digest(self.archive, self.deadline, 1024**2),
                    entries=len(headers), expanded_bytes=0)

    def test_unsafe_archive_metadata_never_exposes_prefix(self):
        target = self.root / "restored"
        for headers in ([{"path": "../escape", "kind": "dir", "mode": 0o700}],
            [{"path": "a", "kind": "dir", "mode": 0o700}] * 2,
            [{"path": "link", "kind": "link", "target": "../outside"}],
            [{"path": "link", "kind": "link", "target": "sub"},
             {"path": "link/file", "kind": "file", "mode": 0o600, "size": 0}],
            [{"path": "huge", "kind": "file", "mode": 0o600, "size": 10**12}],
            [{"path": "loop", "kind": "link", "target": "loop"}]):
            with self.subTest(headers=headers):
                record = self.write_malicious(headers)
                with self.assertRaises(cache.EnvironmentCacheError):
                    cache._cc_cache_extract_archive(self.archive, target, record, self.deadline)
                self.assertFalse(target.exists())
                self.assertFalse(list(self.root.glob(".restore-*")))
                self.assertFalse((self.root / "escape").exists())

    def test_hash_deadline_disk_and_existing_prefix(self):
        (self.prefix / "data").write_bytes(b"data")
        record = cache._cc_cache_create_archive(self.prefix, self.archive, self.deadline)
        with self.assertRaisesRegex(cache.EnvironmentCacheError, "already exists"):
            cache._cc_cache_extract_archive(self.archive, self.prefix, record, self.deadline)
        target = self.root / "restored"
        with mock.patch("shutil.disk_usage", return_value=shutil._ntuple_diskusage(100, 99, 1)):
            with self.assertRaisesRegex(cache.EnvironmentCacheError, "disk space"):
                cache._cc_cache_extract_archive(self.archive, target, record, self.deadline)
        with self.assertRaises(TimeoutError):
            cache._cc_cache_extract_archive(self.archive, target, record, time.time()-1)
        self.archive.write_bytes(self.archive.read_bytes() + b"changed")
        with self.assertRaisesRegex(cache.EnvironmentCacheError, "SHA-256 or size"):
            cache._cc_cache_extract_archive(self.archive, target, record, self.deadline)
        self.assertFalse(target.exists())

    def test_failed_validation_cleans_restored_prefix_before_rebuild(self):
        drive = FakeDriveHTTP()
        definition = normalize_environments({"work": {"python": "3.12", "cache": True}})["work"]
        manifest = {"resources": {"accelerator": "cpu"}, "environment_manager": {}, "files": []}
        make = lambda: cache.EnvironmentCache(drive.storage(), self.root / "stable", manifest, "1"*64,
            {"gpus": []}, time.time()+60, 0, fingerprint=lambda *a: {"release": "fixture"})
        first = make(); selected = first.choose("work", definition)
        prefix = selected["prefix"]; prefix.mkdir(); (prefix / "data").write_bytes(b"valid")
        first.publish(selected, {"python": "3.12.0"}, 5.0)
        self.assertTrue(selected["report"]["published"], selected["report"])
        first.close(); shutil.rmtree(prefix)
        second = make(); selected = second.choose("work", definition)
        try:
            self.assertIsNone(second.restore(selected, lambda deadline: {"python": "3.13.0"}))
            self.assertEqual(selected["report"]["state"], "rebuild")
            self.assertFalse(prefix.exists())
            prefix.mkdir(); (prefix / "data").write_bytes(b"rebuilt")
        finally:
            second.close()
        self.assertFalse(list((self.root / "stable").glob("*.lock")))


# Test-only hooks into the manager boundary of a generated notebook. Actual
# Python and pip children still run through environment_runtime._env_run.
_FIXTURE_BUILDS = []
_FIXTURE_INSTALLS = []


def fixture_manager_download(spec, destination, deadline):
    destination.write_bytes(b'''#!/bin/bash
set -eu
while [[ "$1" != --prefix ]]; do shift; done
shift
export CONDA_PREFIX="$1"
shift
exec "$@"
''')
    destination.chmod(0o700)


def fixture_manager_run(argv, **kwargs):
    if "create" in argv and "--prefix" in argv:
        prefix = Path(argv[argv.index("--prefix") + 1]); _FIXTURE_BUILDS.append(str(prefix))
        venv.EnvBuilder(with_pip=True, symlinks=False).create(prefix)
        libdir = sysconfig.get_config_var("LIBDIR")
        for name in {sysconfig.get_config_var("LDLIBRARY"), sysconfig.get_config_var("INSTSONAME")}:
            if libdir and name and (Path(libdir) / name).is_file():
                destination = prefix / "lib" / name
                if not destination.exists(): shutil.copyfile(Path(libdir) / name, destination)
        return ""
    if "list" in argv and "--json" in argv:
        return "[]"
    if "pip" in argv and "install" in argv:
        _FIXTURE_INSTALLS.append(list(argv))
    return env_runtime._colabcola_env_run(argv, **kwargs)


@unittest.skipUnless(sys.platform == "linux", "real Linux prefix and subprocess rehearsal")
class CacheNotebookIntegrationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name); self.content = self.root / "content"
        self.project = self.root / "project"; self.project.mkdir()
        self.wheel_rel = WHEEL
        self.wheel = self.project / self.wheel_rel; write_wheel(self.wheel)
        (self.project / "stage.py").write_text('''import json, os, sys, cc_wheel_probe
from pathlib import Path
assert Path(sys.prefix) == Path(os.environ["CONDA_PREFIX"])
assert "fixture-token" not in str(dict(os.environ))
assert "COLABCOLA_ENVIRONMENT_CACHE_STORAGE" not in os.environ
assert "PRIVATE_FIXTURE_TOKEN" not in os.environ
Path("outputs").mkdir(exist_ok=True)
Path("outputs/result.json").write_text(json.dumps({"value": cc_wheel_probe.VALUE}))
''', encoding="utf-8")
        self.drive = FakeDriveHTTP()
        _FIXTURE_BUILDS.clear(); _FIXTURE_INSTALLS.clear()
        self.version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.release = "fixture-release-A"

    def artifact(self, stem, *, enabled=True):
        args = parser().parse_args(["--project", str(self.project), "--output", str(self.root / (stem+".colajob")),
            "--name", "cache-fixture", "--artifact", "outputs/**", "--timeout", "120", "--max-seconds", "600"])
        args.environments = {"work": {"python": self.version, "cache": enabled, "pip": {"wheels": [self.wheel_rel]}}}
        args.steps = [{"name": "work", "environment": "work", "command": ["python", "stage.py"]}]
        package(args)
        return build_notebook(build_bundle(load_job_invocation(args.output)))

    def execute(self, artifact, stem, *, managed=True, release=None):
        # Fresh runtime filesystem; the absolute runtime prefix remains the same.
        if self.content.exists(): shutil.rmtree(self.content)
        document = json.loads(artifact.notebook_bytes)
        for cell in document["cells"]:
            if cell["cell_type"] == "code" and cell["metadata"]["colabcola"]["group"] == "environment":
                anchor = "# Build all declared environments as a separate stage before any workload step."
                assert cell["source"].count(anchor) == 1
                cell["source"] = cell["source"].replace(anchor,
                    "from test_environment_cache import fixture_manager_download, fixture_manager_run\n"
                    "_colabcola_env_download = fixture_manager_download\n_colabcola_env_run = fixture_manager_run", 1)
        control = self.root / stem; control.mkdir()
        raw = json.dumps(document).encode()
        (control / "notebook.ipynb").write_bytes(raw)
        lio.atomic(control / "drive-token.json", {"access_token": "fixture-token", "expires_at": time.time()+1800})
        config = {"directory": str(control), "content_root": str(self.content), "deadline": time.time()+600,
                  "files": {"notebook.ipynb": lio.digest(raw)}}
        if managed: config["parent"] = "private-parent"
        lio.atomic(control / "worker.json", config)
        with mock.patch.dict(os.environ, {"COLAB_RELEASE_TAG": self.release if release is None else release,
                                          "PRIVATE_FIXTURE_TOKEN": "fixture-token"}), \
             mock.patch.object(worker, "EnvironmentCacheStorage", side_effect=self.drive.storage), \
             contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            worker.execute_notebook(control / "worker.json")
        root = self.content / "colabcola/results" / artifact.bundle.job_id
        result = verify_final(root, artifact.bundle.job_id)
        self.assertEqual(result["terminal_state"], "success", result.get("failure"))
        record = result["environment"]["environments"]["environments"]["work"]
        self.assertNotIn("fixture-token", json.dumps(result))
        with zipfile.ZipFile(root / "artifacts.zip") as archive:
            output = json.loads(archive.read("outputs/result.json"))
        return record, output, out.getvalue()

    def test_managed_cold_warm_then_wheel_change_and_release_change(self):
        artifact = self.artifact("job")
        first, output, _ = self.execute(artifact, "cold")
        self.assertEqual(first["cache"]["state"], "miss", first["cache"])
        self.assertTrue(first["cache"]["published"], first["cache"])
        self.assertEqual(output, {"value": 17})
        self.assertEqual(len(_FIXTURE_BUILDS), 1); self.assertEqual(len(_FIXTURE_INSTALLS), 1)
        second, warm, stdout = self.execute(artifact, "warm")
        self.assertEqual(second["cache"]["state"], "hit", second["cache"])
        self.assertEqual(warm, output)
        self.assertEqual(second["cache"]["key"], first["cache"]["key"])
        self.assertEqual(first["facts"]["prefix"], second["facts"]["prefix"])
        self.assertEqual(len(_FIXTURE_BUILDS), 1); self.assertEqual(len(_FIXTURE_INSTALLS), 1)
        self.assertNotIn("conda", [s["name"] for s in second["stages"]])
        self.assertNotIn("embedded-wheels", [s["name"] for s in second["stages"]])
        self.assertIn("restored from cache", stdout)
        self.assertIsInstance(second["cache"]["estimated_seconds_saved"], float)
        # Content changes, even with the same wheel filename/version, invalidate.
        write_wheel(self.wheel, value=29)
        changed = self.artifact("changed")
        third, changed_output, _ = self.execute(changed, "changed-wheel")
        self.assertEqual(third["cache"]["state"], "miss")
        self.assertNotEqual(third["cache"]["key"], first["cache"]["key"])
        self.assertEqual(changed_output, {"value": 29})
        fourth, _, _ = self.execute(changed, "changed-release", release="fixture-release-B")
        self.assertEqual(fourth["cache"]["state"], "miss")
        self.assertNotEqual(fourth["cache"]["key"], third["cache"]["key"])
        self.assertEqual(len(_FIXTURE_BUILDS), 3); self.assertEqual(len(_FIXTURE_INSTALLS), 3)
        # Emit reproducible timing evidence without requiring positive savings.
        print("CACHE INTEGRATION EVIDENCE " + json.dumps({"cold": first["cache"], "warm": second["cache"],
             "wheel_change": third["cache"]["state"], "release_change": fourth["cache"]["state"],
             "builds": len(_FIXTURE_BUILDS), "installs": len(_FIXTURE_INSTALLS)}, sort_keys=True))

    def test_opt_out_or_missing_runtime_identity_never_touches_drive_cache(self):
        artifact = self.artifact("disabled", enabled=False)
        record, output, _ = self.execute(artifact, "disabled-run")
        self.assertNotIn("cache", record); self.assertEqual(output, {"value": 17})
        self.assertEqual(self.drive.events, [])
        artifact = self.artifact("enabled")
        record, _, _ = self.execute(artifact, "no-release", release="")
        self.assertEqual(record["cache"]["state"], "bypass")
        self.assertIn("image identity unavailable", record["cache"]["reason"])
        self.assertEqual(self.drive.events, [])
        record, _, _ = self.execute(artifact, "standalone", managed=False)
        self.assertEqual(record["cache"]["reason"], "managed_cache_storage_unavailable")
        self.assertEqual(self.drive.events, [])

    def test_corrupt_cache_and_storage_publication_failure_fall_back_to_real_build(self):
        artifact = self.artifact("job")
        first, output, _ = self.execute(artifact, "cold")
        self.assertTrue(first["cache"]["published"])
        commits = [json.loads(body) for meta, body in self.drive.objects.values()
                   if meta["appProperties"]["cc_kind"] == "environment-cache-v1-commit"]
        self.assertEqual(len(commits), 1)
        chunk = commits[0]["members"]["environment.ccenv.gz"]["chunks"][0]
        meta, body = self.drive.objects[chunk["id"]]
        self.drive.objects[chunk["id"]] = (meta, body + b"corrupt")
        # Simulate both corrupt input and unavailable upload on the same run.
        # The workload should execute from a newly built environment regardless.
        self.drive.fail_upload = "environment-cache-v1-commit"
        fallback, actual, stdout = self.execute(artifact, "corrupt-fallback")
        self.assertEqual(fallback["cache"]["state"], "rebuild", fallback["cache"])
        self.assertFalse(fallback["cache"]["published"])
        self.assertIn("publish_error", fallback["cache"])
        self.assertEqual(fallback["cache"]["key"], first["cache"]["key"])
        self.assertEqual(actual, output)
        self.assertEqual(len(_FIXTURE_BUILDS), 2)
        self.assertEqual(len(_FIXTURE_INSTALLS), 2)
        self.assertNotIn("restored from cache", stdout)
        self.assertFalse(list((self.content / "colabcola/environment-cache-v1/prefixes").glob("*.lock")))

    @unittest.skipUnless(shutil.which("cc") and Path(sysconfig.get_path("include"), "Python.h").is_file(),
                         "native wheel fixture requires C compiler and Python headers")
    def test_native_extension_imports_after_warm_restore(self):
        source = self.root / "native.c"
        source.write_text(r'''#include <Python.h>
static PyObject *value(PyObject *self, PyObject *args) { return PyLong_FromLong(17); }
static PyMethodDef methods[] = {{"value", value, METH_NOARGS, NULL}, {NULL, NULL, 0, NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_cc_wheel_probe", NULL, -1, methods};
PyMODINIT_FUNC PyInit__cc_wheel_probe(void) { return PyModule_Create(&module); }
''', encoding="utf-8")
        tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        wheel_tag = f"{tag}-{tag}-{platform_tag}"
        binary = self.root / ("_cc_wheel_probe" + sysconfig.get_config_var("EXT_SUFFIX"))
        subprocess.run(["cc", "-shared", "-fPIC", "-I" + sysconfig.get_path("include"),
                        str(source), "-o", str(binary)], check=True, capture_output=True, timeout=30)
        self.wheel.unlink()
        self.wheel_rel = f"wheels/cc_wheel_probe-1.0-{wheel_tag}.whl"
        self.wheel = self.project / self.wheel_rel
        write_wheel(self.wheel, native=binary, tag=wheel_tag)
        artifact = self.artifact("native")
        cold, value, _ = self.execute(artifact, "native-cold")
        self.assertTrue(cold["cache"]["published"], cold["cache"])
        warm, restored, _ = self.execute(artifact, "native-warm")
        self.assertEqual(warm["cache"]["state"], "hit", warm["cache"])
        self.assertEqual(restored, value)
        self.assertEqual(restored, {"value": 17})
        self.assertEqual(len(_FIXTURE_BUILDS), 1)
        self.assertEqual(len(_FIXTURE_INSTALLS), 1)
