"""Opt-in Linux multi-GiB page-cache proof with a synthetic streaming source.

Set COLABCOLA_LARGE_INPUT_PROOF=1. Uses local disk only; never allocates
Colab compute, downloads external data, or materializes its input in RAM.
"""
from __future__ import annotations

import contextlib
import ctypes
import hashlib
import io
import json
import mmap
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from colabcola import JobSpec, bind_job_arguments, build_bundle, build_notebook
from test_notebook import notebook_job


@unittest.skipUnless(
    sys.platform == "linux" and os.environ.get("COLABCOLA_LARGE_INPUT_PROOF") == "1"
    and hasattr(os, "posix_fadvise"),
    "opt-in Linux page-cache proof requires COLABCOLA_LARGE_INPUT_PROOF=1",
)
class LargeRemoteInputCacheTests(unittest.TestCase):
    def test_two_gibibyte_stream_has_bounded_residency_and_correct_hash(self):
        block = hashlib.shake_256(b"colabcola-cache-proof-v1").digest(1024 ** 2)
        total = 2 * 1024 ** 3
        expected_digest = hashlib.sha256()
        for _ in range(total // len(block)):
            expected_digest.update(block)
        expected_sha = expected_digest.hexdigest()

        class StreamingInput:
            status = 200

            def __init__(self):
                self.remaining = total

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self, amount=-1):
                if self.remaining == 0:
                    return b""
                amount = min(8 * 1024 ** 2 if amount < 0 else amount,
                             self.remaining, 8 * 1024 ** 2)
                self.remaining -= amount
                return block * (amount // len(block)) + block[:amount % len(block)]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            if shutil.disk_usage(root).free < total + 3 * 1024 ** 3:
                self.skipTest("insufficient available disk for multi-GiB cache proof")
            script = root / "script.py"
            script.write_text(
                "from pathlib import Path\n"
                "Path('outputs').mkdir()\n"
                "Path('outputs/size.txt').write_text("
                "str(Path('weights/large.bin').stat().st_size))\n")
            mapping = notebook_job(script)
            mapping["entrypoint"]["argv"] = []
            mapping["remote_inputs"] = [{
                "target": "weights/large.bin",
                "url": "https://models.example.invalid/owner/repo/resolve/"
                       "0123456789abcdef0123456789abcdef01234567/large.bin",
                "bytes": total, "sha256": expected_sha,
            }]
            mapping["outputs"] = [{
                "pattern": "outputs/size.txt", "required": True,
                "maximum_bytes": 100,
            }]
            compiled = build_notebook(build_bundle(
                bind_job_arguments(JobSpec.from_mapping(mapping, base_dir=root))))
            content = root / "content"
            namespace = {"COLABCOLA_CONTENT_ROOT_OVERRIDE": str(content)}
            def opener(request, timeout=None):
                self.assertTrue(request.full_url.startswith("https://"))
                return StreamingInput()
            with mock.patch("urllib.request.urlopen", side_effect=opener):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    for cell in json.loads(compiled.notebook_bytes)["cells"]:
                        if cell["cell_type"] == "code":
                            exec(cell["source"], namespace)

            job = content / "colabcola" / "jobs" / compiled.bundle.job_id
            result_root = content / "colabcola" / "results" / compiled.bundle.job_id
            result = json.loads((result_root / "result.json").read_text())
            self.assertEqual(result["terminal_state"], "success")
            self.assertEqual((job / "outputs/size.txt").read_text(), str(total))
            target = job / "weights/large.bin"
            self.assertEqual(target.stat().st_size, total)
            control = result["transport"]["remote_inputs"]["cache_control"]
            self.assertEqual(control["window_bytes"], 64 * 1024 ** 2)
            self.assertGreaterEqual(control["sync_calls"], 32)
            self.assertGreaterEqual(control["advice_calls"], 32)
            self.assertEqual(control["advice_errors"], 0)

            # mincore reads only the resident-page bitmap; mapping the file
            # doesn't fault its pages into memory.
            page = os.sysconf("SC_PAGE_SIZE")
            pages = (total + page - 1) // page
            resident = (ctypes.c_ubyte * pages)()
            libc = ctypes.CDLL(None, use_errno=True)
            mincore = libc.mincore
            mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t,
                                ctypes.POINTER(ctypes.c_ubyte))
            mincore.restype = ctypes.c_int
            with target.open("r+b") as handle:
                with mmap.mmap(handle.fileno(), total, access=mmap.ACCESS_WRITE) as mapping:
                    pointer = ctypes.c_char.from_buffer(mapping)
                    try:
                        rc = mincore(ctypes.addressof(pointer), total, resident)
                        if rc != 0:
                            raise OSError(ctypes.get_errno(), "mincore failed")
                    finally:
                        del pointer
            resident_bytes = sum(bool(v & 1) for v in resident) * page
            print("cache-proof:", json.dumps({
                "download_bytes": total, "resident_bytes": resident_bytes,
                "resident_fraction": resident_bytes / total,
                "sha256": expected_sha, "cache_control": control,
            }, sort_keys=True), flush=True)
            self.assertLessEqual(resident_bytes, total // 4)

            # Check the actual final on-disk file after inspecting residency.
            with target.open("rb") as stream:
                self.assertEqual(hashlib.file_digest(stream, "sha256").hexdigest(),
                                 expected_sha)


if __name__ == "__main__":
    unittest.main()
