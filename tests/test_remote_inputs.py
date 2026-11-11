from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from colabcola import (
    DriveControllerAuth,
    JobSpec,
    RemoteInputSpec,
    SpecError,
    bind_job_arguments,
    build_bundle,
    build_drive_controller,
    build_notebook,
    build_staged_bundle,
    drive_receipt_for_observed_upload,
)

from test_notebook import notebook_job

HOST = "models.example.invalid"
URL_A = f"https://{HOST}/owner/repo/resolve/0123456789abcdef0123456789abcdef01234567/a.bin"
URL_B = f"https://{HOST}/owner/repo/resolve/0123456789abcdef0123456789abcdef01234567/sub/b.bin"
DATA_A = bytes(range(256)) * 300
DATA_B = b"colabcola remote input fixture\n" * 97


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def remote_entries() -> list[dict[str, object]]:
    return [
        {"target": "weights/a.bin", "url": URL_A, "bytes": len(DATA_A), "sha256": sha(DATA_A)},
        {"target": "weights/sub/b.bin", "url": URL_B, "bytes": len(DATA_B), "sha256": sha(DATA_B)},
    ]


class _FakeResponse:
    def __init__(self, body: bytes, status: int, fail_after: int | None = None) -> None:
        self._body = body
        self._offset = 0
        self._fail_after = fail_after
        self.status = status
        self.headers = {}

    def read(self, amount: int = -1) -> bytes:
        if self._fail_after is not None and self._offset >= self._fail_after:
            raise ConnectionResetError("simulated connection reset")
        end = len(self._body) if amount is None or amount < 0 else min(len(self._body), self._offset + amount)
        if self._fail_after is not None:
            end = min(end, self._fail_after)
        chunk = self._body[self._offset:end]
        self._offset = end
        return chunk

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def fake_urlopen(
    files: dict[str, bytes],
    *,
    fail_first_after: dict[str, int] | None = None,
    calls: list[tuple[str, int, int]] | None = None,
):
    """A stand-in for urllib.request.urlopen that serves bytes, honours Range, and can drop a first attempt."""

    seen: dict[str, int] = {}

    def opener(request, timeout=None):
        url = request.full_url
        headers = {name.lower(): value for name, value in request.header_items()}
        body = files[url]
        start, status = 0, 200
        if "range" in headers:
            start = int(headers["range"].split("=", 1)[1].rstrip("-"))
            status = 206
        seen[url] = seen.get(url, 0) + 1
        if calls is not None:
            calls.append((url, start, status))
        fail_after = None
        if fail_first_after and url in fail_first_after and seen[url] == 1:
            fail_after = max(0, fail_first_after[url] - start)
        return _FakeResponse(body[start:], status, fail_after)

    return opener


class RemoteInputSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "main.py"
        self.source.write_text("print('ok')\n", encoding="utf-8")

    def spec(self, entries: list[dict[str, object]]) -> JobSpec:
        mapping = notebook_job(self.source)
        mapping["remote_inputs"] = entries
        return JobSpec.from_mapping(mapping, base_dir=self.root)

    def test_remote_inputs_parse_in_order_and_normalize_the_digest(self) -> None:
        entries = remote_entries()
        entries[0]["sha256"] = entries[0]["sha256"].upper()
        spec = self.spec(entries)
        self.assertEqual(
            spec.remote_inputs,
            (
                RemoteInputSpec("weights/a.bin", URL_A, len(DATA_A), sha(DATA_A)),
                RemoteInputSpec("weights/sub/b.bin", URL_B, len(DATA_B), sha(DATA_B)),
            ),
        )
        self.assertEqual(JobSpec.from_mapping(notebook_job(self.source), base_dir=self.root).remote_inputs, ())

    def test_invalid_remote_inputs_are_rejected(self) -> None:
        good = remote_entries()[0]
        cases = {
            "http": {**good, "url": good["url"].replace("https://", "http://")},
            "user information": {**good, "url": f"https://user:secret@{HOST}/a.bin"},
            "whitespace": {**good, "url": f"https://{HOST}/a b.bin"},
            "fragment": {**good, "url": f"https://{HOST}/a.bin#part"},
            "short digest": {**good, "sha256": "abc"},
            "zero bytes": {**good, "bytes": 0},
            "unknown field": {**good, "note": "x"},
            "payload target": {**good, "target": "workload/main.py"},
            "absolute target": {**good, "target": "/weights/a.bin"},
        }
        for label, entry in cases.items():
            with self.subTest(label):
                with self.assertRaises(SpecError):
                    self.spec([entry])
        with self.assertRaisesRegex(SpecError, "duplicate targets"):
            self.spec([good, dict(good)])


class RemoteInputBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "main.py"
        self.source.write_text("print('ok')\n", encoding="utf-8")

    def bundle(self, mapping: dict[str, object]):
        return build_bundle(bind_job_arguments(JobSpec.from_mapping(mapping, base_dir=self.root)))

    def test_manifest_carries_sorted_remote_inputs_only_when_declared(self) -> None:
        plain = self.bundle(notebook_job(self.source))
        self.assertNotIn("remote_inputs", json.loads(plain.canonical_manifest_bytes))

        mapping = notebook_job(self.source)
        mapping["remote_inputs"] = list(reversed(remote_entries()))
        bundle = self.bundle(mapping)
        manifest = json.loads(bundle.canonical_manifest_bytes)
        self.assertEqual(
            manifest["remote_inputs"],
            [
                {"bytes": len(DATA_A), "sha256": sha(DATA_A), "target": "weights/a.bin", "url": URL_A},
                {"bytes": len(DATA_B), "sha256": sha(DATA_B), "target": "weights/sub/b.bin", "url": URL_B},
            ],
        )
        self.assertEqual(manifest["compiler_contract_version"], "0.10")
        self.assertNotEqual(bundle.job_id, plain.job_id)
        self.assertEqual(bundle.archive_bytes, plain.archive_bytes)

        changed = copy.deepcopy(mapping)
        changed["remote_inputs"][0]["sha256"] = sha(b"other")
        self.assertNotEqual(self.bundle(changed).job_id, bundle.job_id)

    def test_remote_targets_may_not_nest_with_files_and_count_toward_the_limit(self) -> None:
        nested = notebook_job(self.source)
        nested["remote_inputs"] = [
            {"target": "workload", "url": URL_A, "bytes": 1, "sha256": sha(b"x")}
        ]
        with self.assertRaisesRegex(SpecError, "beneath"):
            self.bundle(nested)

        beneath = notebook_job(self.source)
        beneath["remote_inputs"] = [
            {"target": "weights", "url": URL_A, "bytes": 1, "sha256": sha(b"x")},
            {"target": "weights/a.bin", "url": URL_B, "bytes": 1, "sha256": sha(b"y")},
        ]
        with self.assertRaisesRegex(SpecError, "beneath"):
            self.bundle(beneath)

        limited = notebook_job(self.source)
        limited["remote_inputs"] = remote_entries()
        limited["limits"] = {"maximum_file_count": 2}
        with self.assertRaisesRegex(SpecError, "maximum_file_count"):
            self.bundle(limited)


class RemoteInputRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "main.py"
        self.source.write_text(
            "from pathlib import Path\n"
            "sizes = [Path('weights/a.bin').stat().st_size, Path('weights/sub/b.bin').stat().st_size]\n"
            "Path('outputs').mkdir(exist_ok=True)\n"
            "Path('outputs/sizes.txt').write_text(' '.join(str(size) for size in sizes), encoding='utf-8')\n",
            encoding="utf-8",
            newline="\n",
        )

    def mapping(self) -> dict[str, object]:
        mapping = notebook_job(self.source)
        mapping["remote_inputs"] = remote_entries()
        mapping["outputs"] = [{"pattern": "outputs/sizes.txt", "required": True, "maximum_bytes": 64}]
        mapping["limits"] = {"maximum_file_count": 10}
        return mapping

    def run_notebook(self, mapping: dict[str, object], content_root: Path, opener, *, remote_clock=None):
        artifact = build_notebook(
            build_bundle(bind_job_arguments(JobSpec.from_mapping(mapping, base_dir=self.root)))
        )
        document = json.loads(artifact.notebook_bytes)
        namespace = {"COLABCOLA_CONTENT_ROOT_OVERRIDE": os.fspath(content_root)}
        error: RuntimeError | None = None
        with mock.patch("urllib.request.urlopen", opener), mock.patch("time.sleep"):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    for cell in document["cells"]:
                        if cell["cell_type"] == "code":
                            clock = (mock.patch("time.monotonic", side_effect=remote_clock)
                                     if remote_clock and "def _colabcola_remote_progress" in cell["source"]
                                     else contextlib.nullcontext())
                            with clock:
                                exec(cell["source"], namespace)
                except RuntimeError as exc:
                    error = exc
        result_root = content_root / "colabcola" / "results" / artifact.bundle.job_id
        result = json.loads((result_root / "result.json").read_text(encoding="utf-8"))
        job_root = content_root / "colabcola" / "jobs" / artifact.bundle.job_id
        return result, job_root, error

    def test_declared_remote_inputs_are_downloaded_verified_and_available_to_the_job(self) -> None:
        calls: list[tuple[str, int, int]] = []
        result, job_root, error = self.run_notebook(
            self.mapping(),
            self.root / "content-success",
            fake_urlopen({URL_A: DATA_A, URL_B: DATA_B}, calls=calls),
        )
        self.assertIsNone(error)
        self.assertEqual(result["terminal_state"], "success")
        self.assertEqual((job_root / "weights" / "a.bin").read_bytes(), DATA_A)
        self.assertEqual((job_root / "weights" / "sub" / "b.bin").read_bytes(), DATA_B)
        self.assertFalse(list(job_root.rglob("*.partial")))
        facts = result["transport"]["remote_inputs"]
        self.assertEqual((facts["count"], facts["bytes"]), (2, len(DATA_A) + len(DATA_B)))
        self.assertEqual([entry["target"] for entry in facts["entries"]], ["weights/a.bin", "weights/sub/b.bin"])
        self.assertEqual([entry["attempts"] for entry in facts["entries"]], [1, 1])
        self.assertEqual([entry["http_status"] for entry in facts["entries"]], [200, 200])
        self.assertEqual(calls, [(URL_A, 0, 200), (URL_B, 0, 200)])
        self.assertEqual(
            (job_root / "outputs" / "sizes.txt").read_text(encoding="utf-8"),
            f"{len(DATA_A)} {len(DATA_B)}",
        )

    def test_download_byte_progress_is_durable_before_download_finishes(self) -> None:
        content=self.root/'content-progress'
        observations=[]
        from colabcola.lifecycle_worker import log_members
        from colabcola.lifecycle_io import publish, recover
        from colabcola.managed import Progress
        from test_lifecycle import MemoryStore
        store=MemoryStore(); host=io.StringIO(); reporter=Progress(self.root,host)
        identity={'run_id':'a'*32,'job_id':'b'*64}
        class Streaming(_FakeResponse):
            def read(response, amount=-1):
                paths=list(content.glob('colabcola/results/*/transfer.log'))
                observations.append(paths[0].read_text() if paths else '')
                if paths:
                    snapshot=publish(store,identity,'logs',log_members(paths[0].parent,self.root,[]),attempt=1)
                    recovered=recover(store,snapshot,self.root/'host'/snapshot['sha256'])
                    reporter({'logs':{**snapshot,'local_directory':str(recovered)}})
                return super(Streaming,response).read(min(amount,16000))
        def opener(request,timeout=None):
            return Streaming({URL_A:DATA_A,URL_B:DATA_B}[request.full_url],200)
        # Force the deterministic two-second throttle forward for each block.
        clock=iter(range(0,10000,3))
        result,job_root,error=self.run_notebook(self.mapping(),content,opener,
            remote_clock=lambda:next(clock))
        self.assertIsNone(error)
        self.assertTrue(any('16000/'+str(len(DATA_A))+' bytes | downloading' in line for line in observations))
        log=next(content.glob('colabcola/results/*/transfer.log')).read_text()
        total=len(DATA_A)+len(DATA_B)
        self.assertIn(f'2/2 files verified | {total}/{total} bytes received',log)
        self.assertIn('MiB/s',log)
        self.assertIn('16000/'+str(len(DATA_A))+' bytes | downloading',host.getvalue())
        self.assertIn('[attempt 1 | transfer.log]',host.getvalue())
        self.assertNotIn('https://',log)
        self.assertEqual(result['terminal_state'],'success')

    def test_retry_progress_does_not_double_count_partial_bytes(self) -> None:
        calls=[];half=len(DATA_A)//2
        result,job_root,error=self.run_notebook(self.mapping(),self.root/'content-retry-progress',
            fake_urlopen({URL_A:DATA_A,URL_B:DATA_B},fail_first_after={URL_A:half},calls=calls))
        self.assertIsNone(error)
        log=next((self.root/'content-retry-progress').glob('colabcola/results/*/transfer.log')).read_text()
        total=len(DATA_A)+len(DATA_B)
        self.assertIn('retrying (attempt 1)',log)
        self.assertIn(f'{total}/{total} bytes received',log)
        self.assertIn('verified (attempt 2)',log)
        self.assertEqual(calls[1],(URL_A,half,206))

    def test_unverified_bytes_never_become_verified_progress(self) -> None:
        mapping=self.mapping(); mapping['remote_inputs'][0]['sha256']=sha(b'wrong')
        result,job_root,error=self.run_notebook(mapping,self.root/'content-progress-bad',fake_urlopen({URL_A:DATA_A,URL_B:DATA_B}))
        self.assertIsNotNone(error)
        log=next((self.root/'content-progress-bad').glob('colabcola/results/*/transfer.log')).read_text()
        self.assertIn('verification failed',log)
        self.assertNotIn('1/2 files verified',log)

    def test_interrupted_download_resumes_with_a_range_request(self) -> None:
        calls: list[tuple[str, int, int]] = []
        half = len(DATA_A) // 2
        result, job_root, error = self.run_notebook(
            self.mapping(),
            self.root / "content-resume",
            fake_urlopen({URL_A: DATA_A, URL_B: DATA_B}, fail_first_after={URL_A: half}, calls=calls),
        )
        self.assertIsNone(error)
        self.assertEqual(result["terminal_state"], "success")
        self.assertEqual((job_root / "weights" / "a.bin").read_bytes(), DATA_A)
        entry = result["transport"]["remote_inputs"]["entries"][0]
        self.assertEqual((entry["attempts"], entry["http_status"]), (2, 206))
        self.assertEqual(calls[:2], [(URL_A, 0, 200), (URL_A, half, 206)])

    def test_ignored_range_restarts_entire_file_and_preserves_sha(self) -> None:
        calls = []
        attempts = 0

        def opener(request, timeout=None):
            nonlocal attempts
            calls.append(request.full_url)
            if request.full_url == URL_A:
                attempts += 1
                return _FakeResponse(DATA_A, 200,
                                     fail_after=len(DATA_A) // 2 if attempts == 1 else None)
            return _FakeResponse(DATA_B, 200)

        result, job_root, error = self.run_notebook(
            self.mapping(), self.root / "content-range-200", opener)
        self.assertIsNone(error)
        self.assertEqual(result["terminal_state"], "success")
        self.assertEqual((job_root / "weights/a.bin").read_bytes(), DATA_A)
        self.assertEqual((job_root / "weights/sub/b.bin").read_bytes(), DATA_B)
        self.assertEqual(calls, [URL_A, URL_A, URL_B])
        self.assertFalse(list(job_root.rglob("*.partial")))

    def test_ignored_range_restart_fails_closed_when_full_file_wont_fit(self) -> None:
        calls = []
        attempts = 0
        total = len(DATA_A) + len(DATA_B)

        def opener(request, timeout=None):
            nonlocal attempts
            calls.append(request.full_url)
            if request.full_url == URL_A:
                attempts += 1
                return _FakeResponse(DATA_A, 200,
                                     fail_after=len(DATA_A) // 2 if attempts == 1 else None)
            return _FakeResponse(DATA_B, 200)

        def usage(_path):
            # Space is ample before the second HTTP response arrives. Once
            # the server refuses Range, a whole-file restart no longer fits.
            free = (2 ** 42 if attempts < 2
                    else 1024 ** 3 + total - len(DATA_A) // 2 + 1)
            return types.SimpleNamespace(total=2 ** 43, used=0, free=free)

        with mock.patch("shutil.disk_usage", side_effect=usage):
            result, job_root, error = self.run_notebook(
                self.mapping(), self.root / "content-range-200-short", opener)
        self.assertIsNotNone(error)
        self.assertEqual(result["terminal_state"], "preflight_failed")
        self.assertEqual(result["failure"]["stage"], "remote_inputs")
        self.assertIn("insufficient free disk", result["failure"]["message"])
        self.assertEqual(calls, [URL_A, URL_A])
        self.assertFalse((job_root / "weights/a.bin").exists())
        self.assertFalse(list(job_root.rglob("*.partial")))

    def test_hash_mismatch_is_a_remote_input_failure_with_no_file_left_behind(self) -> None:
        mapping = self.mapping()
        mapping["remote_inputs"][1]["sha256"] = sha(b"not the file")
        result, job_root, error = self.run_notebook(
            mapping,
            self.root / "content-mismatch",
            fake_urlopen({URL_A: DATA_A, URL_B: DATA_B}),
        )
        self.assertIsNotNone(error)
        self.assertIn("remote_input_failed", str(error))
        self.assertEqual(result["terminal_state"], "remote_input_failed")
        self.assertEqual(result["failure"]["stage"], "remote_inputs")
        self.assertIn("SHA-256 does not match", result["failure"]["message"])
        self.assertIsNone(result["process_exit_code"])
        self.assertTrue((job_root / "weights" / "a.bin").exists())
        self.assertFalse((job_root / "weights" / "sub" / "b.bin").exists())
        self.assertFalse(list(job_root.rglob("*.partial")))

    def test_remote_cache_control_syncs_in_windows_and_advices_on_posix(self) -> None:
        from colabcola import runtime as runtime_source

        # Use a small window only in this test's compiled notebook so that
        # tiny in-memory fixtures exercise multiple fdatasync intervals.
        short_source = runtime_source._REMOTE_INPUTS_SOURCE.replace(
            '64 * 1024 ** 2', '32 * 1024')
        self.assertNotEqual(short_source, runtime_source._REMOTE_INPUTS_SOURCE)

        class SmallBlocks(_FakeResponse):
            def read(response, amount=-1):
                return super(SmallBlocks, response).read(min(amount, 16384))

        def opener(request, timeout=None):
            return SmallBlocks({URL_A: DATA_A, URL_B: DATA_B}[request.full_url], 200)

        with mock.patch.object(runtime_source, '_REMOTE_INPUTS_SOURCE', short_source):
            result, job_root, error = self.run_notebook(
                self.mapping(), self.root / 'content-cache', opener)
        self.assertIsNone(error)
        self.assertEqual(result['terminal_state'], 'success')
        self.assertEqual((job_root / 'weights/a.bin').read_bytes(), DATA_A)
        self.assertEqual((job_root / 'weights/sub/b.bin').read_bytes(), DATA_B)
        policy = result['transport']['remote_inputs']['cache_control']
        self.assertEqual(policy['window_bytes'], 32768)
        self.assertGreaterEqual(policy['sync_calls'], 4)
        self.assertEqual(policy['advice_supported'], hasattr(os, 'posix_fadvise'))
        if policy['advice_supported']:
            self.assertGreaterEqual(policy['advice_calls'], 1)
            self.assertEqual(policy['advice_errors'], 0)
        self.assertNotIn('https://', json.dumps(policy))

    def test_free_disk_is_rechecked_before_second_remote_file(self) -> None:
        import shutil
        calls: list[tuple[str, int, int]] = []
        content = self.root / 'content-space-recheck'
        real_usage = shutil.disk_usage

        def disk_usage(path):
            # The first complete file intentionally consumes the remaining
            # simulated free space. The next file must never be requested.
            first = content / 'colabcola' / 'jobs'
            if first in Path(path).parents and list(first.glob('*/weights/a.bin')):
                return types.SimpleNamespace(total=100, used=99, free=1)
            return types.SimpleNamespace(total=2 ** 42, used=0, free=2 ** 42)

        with mock.patch('shutil.disk_usage', side_effect=disk_usage):
            result, job_root, error = self.run_notebook(
                self.mapping(), content,
                fake_urlopen({URL_A: DATA_A, URL_B: DATA_B}, calls=calls))
        self.assertIsNotNone(error)
        self.assertEqual(result['terminal_state'], 'preflight_failed')
        self.assertEqual(result['failure']['stage'], 'remote_inputs')
        self.assertIn('insufficient free disk', result['failure']['message'])
        self.assertEqual(calls, [(URL_A, 0, 200)])
        self.assertTrue((job_root / 'weights/a.bin').is_file())
        self.assertFalse((job_root / 'weights/sub/b.bin').exists())

    def test_scratch_evaluation_is_read_only_and_preserves_job_paths(self) -> None:
        from colabcola import runtime as runtime_source
        import shutil
        candidate = self.root / "optional-scratch"
        candidate.mkdir()
        source = runtime_source._REMOTE_INPUTS_SOURCE
        patched = source.replace(
            '"/mnt/local-scratch", "/mnt/data", "/scratch"',
            repr(str(candidate)) + ',')
        self.assertNotEqual(source, patched)

        with mock.patch.object(runtime_source, "_REMOTE_INPUTS_SOURCE", patched):
            result, job_root, error = self.run_notebook(
                self.mapping(), self.root / "content-scratch-assessment",
                fake_urlopen({URL_A: DATA_A, URL_B: DATA_B}))
        self.assertIsNone(error)
        self.assertEqual(result["terminal_state"], "success")
        report = result["transport"]["remote_input_storage_evaluation"]
        self.assertEqual(report["mode"], "diagnostic_only")
        self.assertEqual(report["schema"], 1)
        self.assertEqual(report["required_bytes_including_margin"],
                         len(DATA_A) + len(DATA_B) + 1024 ** 3)
        self.assertEqual(len(report["candidates"]), 1)
        self.assertEqual(report["candidates"][0]["path"], str(candidate))
        self.assertFalse(report["candidates"][0]["different_device"])
        self.assertEqual(
            report["candidates"][0]["capacity_sufficient"],
            shutil.disk_usage(candidate).free >= report["required_bytes_including_margin"])
        result_root = job_root.parents[1] / "results" / job_root.name
        self.assertEqual(
            json.loads((result_root / "storage-assessment.json").read_text()),
            report)
        self.assertEqual(list(candidate.iterdir()), [])
        self.assertEqual((job_root / "weights/a.bin").read_bytes(), DATA_A)
        self.assertEqual((job_root / "weights/sub/b.bin").read_bytes(), DATA_B)
        self.assertFalse(list(job_root.rglob("*.partial")))
        self.assertNotIn("https://", json.dumps(report))

    def test_scratch_symlink_is_not_selected_or_followed(self) -> None:
        from colabcola import runtime as runtime_source
        real_dir = self.root / "actual-scratch"
        real_dir.mkdir()
        link = self.root / "scratch-link"
        try:
            link.symlink_to(real_dir, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest("directory symlinks unavailable: " + type(exc).__name__)
        source = runtime_source._REMOTE_INPUTS_SOURCE
        patched = source.replace(
            '"/mnt/local-scratch", "/mnt/data", "/scratch"', repr(str(link)) + ',')
        self.assertNotEqual(source, patched)
        with mock.patch.object(runtime_source, "_REMOTE_INPUTS_SOURCE", patched):
            result, job_root, error = self.run_notebook(
                self.mapping(), self.root / "content-scratch-symlink",
                fake_urlopen({URL_A: DATA_A, URL_B: DATA_B}))
        self.assertIsNone(error)
        self.assertEqual(result["terminal_state"], "success")
        report = result["transport"]["remote_input_storage_evaluation"]
        self.assertEqual(report["candidates"], [])
        self.assertEqual(list(real_dir.iterdir()), [])
        self.assertEqual((job_root / "weights/a.bin").read_bytes(), DATA_A)

    def test_disk_shortfall_fails_before_any_download(self) -> None:
        calls: list[tuple[str, int, int]] = []
        with mock.patch("shutil.disk_usage", return_value=types.SimpleNamespace(total=10, used=5, free=5)):
            result, job_root, error = self.run_notebook(
                self.mapping(),
                self.root / "content-disk",
                fake_urlopen({URL_A: DATA_A, URL_B: DATA_B}, calls=calls),
            )
        self.assertIsNotNone(error)
        self.assertEqual(result["terminal_state"], "preflight_failed")
        self.assertEqual(result["failure"]["stage"], "remote_inputs")
        self.assertIn("insufficient free disk", result["failure"]["message"])
        self.assertEqual(calls, [])


class RemoteInputSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "main.py"
        self.source.write_text("print('ok')\n", encoding="utf-8")
        mapping = notebook_job(self.source)
        mapping["remote_inputs"] = remote_entries()
        self.bundle = build_bundle(bind_job_arguments(JobSpec.from_mapping(mapping, base_dir=self.root)))

    def test_embedded_summary_names_remote_inputs_and_hosts(self) -> None:
        document = json.loads(build_notebook(self.bundle).notebook_bytes)
        summary = document["cells"][0]["source"]
        self.assertIn("Remote inputs: 2 files", summary)
        self.assertIn(f"`{HOST}`", summary)
        self.assertIn("verifies each against its declared byte count and SHA-256", summary)
        self.assertEqual(document["metadata"]["colabcola"]["implemented_cell_groups"], 7)

    def test_controller_summary_names_remote_inputs_and_keeps_seven_groups(self) -> None:
        staged = build_staged_bundle(self.bundle)
        receipt = drive_receipt_for_observed_upload(
            staged,
            remote_file_id="drive_payload_remote",
            remote_parent_id="drive_job_folder_remote",
            remote_relative_path=f"colabcola/staging/{staged.bundle.job_id}/{staged.file_name}",
        )
        auth = DriveControllerAuth(
            google_cloud_project_id="colabcola-fixture",
            oauth_client_id="controller-fixture.apps.googleusercontent.com",
            colab_secret_name="COLABCOLA_GOOGLE_DRIVE_OAUTH",
        )
        document = json.loads(build_drive_controller(staged, receipt, auth).notebook_bytes)
        summary = document["cells"][0]["source"]
        self.assertIn("Remote inputs: 2 files", summary)
        self.assertIn(f"`{HOST}`", summary)
        self.assertEqual(
            [cell["metadata"]["colabcola"]["group"] for cell in document["cells"]],
            ["summary", "manifest", "preflight", "acquisition", "extraction", "environment", "execution"],
        )
        extraction = next(
            cell["source"] for cell in document["cells"] if cell["metadata"]["colabcola"]["group"] == "extraction"
        )
        self.assertIn("remote_inputs", extraction)


if __name__ == "__main__":
    unittest.main()
