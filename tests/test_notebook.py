from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
import wave
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest import mock

from colabcola._version import __version__

from colabcola import (
    JobSpec,
    NotebookError,
    bind_job_arguments,
    build_bundle,
    build_notebook,
    encode_archive_chunks,
    verify_notebook,
)
from colabcola.notebook import PAYLOAD_LITERAL_LINE_CHARS, RAW_PAYLOAD_CHUNK_BYTES


def notebook_job(
    source: Path,
    *,
    name: str = "notebook-fixture",
    target: str = "workload/main.py",
    accelerator: str = "cpu",
) -> dict[str, object]:
    gpu = accelerator == "gpu"
    return {
        "schema_version": 1,
        "name": name,
        "payload": [
            {
                "role": "source",
                "source": os.fspath(source),
                "target": target,
            }
        ],
        "entrypoint": {
            "kind": "python_script",
            "path": target,
            "argv": ["--configured", "value"],
            "cwd": ".",
        },
        "python": {
            "requires_python": ">=3.11",
            "runtime_requirements": [],
            "install_requirements": [],
            "embedded_wheels": [],
        },
        "runtime": {"provider": "google_colab", "version": "latest"},
        "resources": {
            "accelerator": accelerator,
            "gpu_count": 1 if gpu else 0,
            "cuda_required": gpu,
            "minimum_gpu_memory_bytes": None,
        },
        "run": {
            "timeout_seconds": 60,
            "seed": 1234,
            "environment": {},
            "capture_stdout": True,
            "capture_stderr": True,
        },
        "outputs": [],
        "limits": {
            "maximum_file_count": 10,
        },
    }


class NotebookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "main.py"
        self.source.write_text("print('ok')\n", encoding="utf-8")

    def bundle(
        self,
        *,
        mapping: dict[str, object] | None = None,
        job_args: tuple[str, ...] = (),
    ):
        spec = JobSpec.from_mapping(
            mapping or notebook_job(self.source),
            base_dir=self.root,
        )
        return build_bundle(bind_job_arguments(spec, job_args))

    def test_chunk_boundaries_and_standard_base64_reconstruct_bytes(self) -> None:
        for size, expected_count in ((5, 1), (6, 1), (7, 2), (13, 3)):
            with self.subTest(size=size):
                payload = bytes(range(size))
                chunks = encode_archive_chunks(payload, raw_chunk_bytes=6)
                self.assertEqual(len(chunks), expected_count)
                self.assertTrue(all(len(chunk) <= 8 for chunk in chunks))
                self.assertEqual(base64.b64decode("".join(chunks), validate=True), payload)

        with self.assertRaisesRegex(NotebookError, "must not be empty"):
            encode_archive_chunks(b"")
        with self.assertRaisesRegex(NotebookError, "positive multiple of 3"):
            encode_archive_chunks(b"data", raw_chunk_bytes=4)

    def test_repeated_notebook_builds_are_byte_identical_stable_json(self) -> None:
        bundle = self.bundle()
        first = build_notebook(bundle)
        second = build_notebook(bundle)

        self.assertEqual(first.notebook_bytes, second.notebook_bytes)
        self.assertEqual(first.notebook_sha256, second.notebook_sha256)
        self.assertTrue(first.notebook_bytes.endswith(b"\n"))
        document = json.loads(first.notebook_bytes)
        expected = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8") + b"\n"
        self.assertEqual(first.notebook_bytes, expected)
        first.verify()

    def test_metadata_and_cell_groups_are_pinned_and_ordered(self) -> None:
        artifact = build_notebook(self.bundle())
        document = json.loads(artifact.notebook_bytes)

        self.assertEqual(document["nbformat"], 4)
        self.assertEqual(document["nbformat_minor"], 4)
        metadata = document["metadata"]
        self.assertNotIn("accelerator", metadata)
        self.assertEqual(metadata["kernelspec"]["name"], "python3")
        self.assertEqual(metadata["colabcola"]["compiler_contract_version"], "0.10")
        self.assertEqual(metadata["colabcola"]["compiler_version"], __version__)
        self.assertEqual(metadata["colabcola"]["job_id"], artifact.bundle.job_id)
        self.assertEqual(metadata["colabcola"]["payload_cell_count"], 1)
        self.assertEqual(metadata["colabcola"]["payload_display"], "hidden_form")
        self.assertEqual(
            metadata["colabcola"]["payload_literal_line_chars"],
            PAYLOAD_LITERAL_LINE_CHARS,
        )
        self.assertEqual(metadata["colabcola"]["implemented_cell_groups"], 7)
        self.assertEqual(metadata["colabcola"]["archive_warning_bytes"], 1024**3)
        self.assertEqual(metadata["colabcola"]["warnings"], [])

        cells = document["cells"]
        self.assertEqual(cells[0]["metadata"]["colabcola"]["group"], "summary")
        self.assertEqual(cells[1]["metadata"]["colabcola"]["group"], "manifest")
        self.assertEqual(cells[2]["metadata"]["colabcola"]["group"], "payload")
        self.assertEqual(
            [cell["metadata"]["colabcola"]["group"] for cell in cells[-4:]],
            ["preflight", "extraction", "environment", "execution"],
        )
        self.assertTrue(all("id" not in cell for cell in cells))
        for cell in cells[1:]:
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            compile(cell["source"], "<colabcola-cell>", "exec")

        payload = cells[2]
        self.assertEqual(payload["metadata"]["cellView"], "form")
        self.assertEqual(payload["metadata"]["colabcola"]["display"], "hidden_form")
        self.assertEqual(
            payload["metadata"]["colabcola"]["literal_line_chars"],
            PAYLOAD_LITERAL_LINE_CHARS,
        )
        self.assertTrue(payload["source"].startswith("# @title WARNING: Embedded payload"))
        self.assertIn("keep hidden", payload["source"].splitlines()[0])
        self.assertIn("Do not reveal or edit", payload["source"])
        self.assertLessEqual(
            max(map(len, payload["source"].splitlines())),
            PAYLOAD_LITERAL_LINE_CHARS + 8,
        )

    def test_summary_warns_users_not_to_show_payload(self) -> None:
        document = json.loads(build_notebook(self.bundle()).notebook_bytes)
        summary = document["cells"][0]["source"]

        self.assertIn("**Payload visibility warning:**", summary)
        self.assertIn("Keep every embedded payload cell hidden", summary)
        self.assertIn("do not expand the payload cells", summary)

    def test_archive_size_warning_is_embedded_in_summary_and_metadata(self) -> None:
        with (
            mock.patch("colabcola.bundle.ARCHIVE_WARNING_BYTES", 1),
            mock.patch("colabcola.notebook.ARCHIVE_WARNING_BYTES", 1),
            self.assertWarnsRegex(RuntimeWarning, "warning threshold"),
        ):
            artifact = build_notebook(self.bundle())
        document = json.loads(artifact.notebook_bytes)

        self.assertEqual(
            document["metadata"]["colabcola"]["warnings"],
            list(artifact.bundle.warnings),
        )
        self.assertEqual(document["metadata"]["colabcola"]["archive_warning_bytes"], 1)
        self.assertIn("**Size warning:**", document["cells"][0]["source"])
        self.assertIn(artifact.bundle.warnings[0], document["cells"][0]["source"])

    def test_gpu_jobs_receive_colab_gpu_preference_metadata(self) -> None:
        mapping = notebook_job(self.source, accelerator="gpu")
        document = json.loads(build_notebook(self.bundle(mapping=mapping)).notebook_bytes)
        self.assertEqual(document["metadata"]["accelerator"], "GPU")
        self.assertEqual(
            document["metadata"]["colabcola"]["archive_sha256"],
            self.bundle(mapping=mapping).archive_sha256,
        )

    def test_manifest_and_payload_cells_execute_to_the_exact_archive(self) -> None:
        artifact = build_notebook(self.bundle(job_args=("snowman-â˜ƒ",)))
        document = json.loads(artifact.notebook_bytes)
        namespace: dict[str, object] = {}
        for cell in document["cells"][1:-4]:
            exec(cell["source"], namespace)

        self.assertEqual(
            namespace["COLABCOLA_MANIFEST_JSON"].encode("utf-8"),
            artifact.bundle.canonical_manifest_bytes,
        )
        chunks = namespace["COLABCOLA_PAYLOAD_CHUNKS"]
        reconstructed = base64.b64decode("".join(chunks), validate=True)
        self.assertEqual(reconstructed, artifact.bundle.archive_bytes)
        self.assertEqual(
            hashlib.sha256(reconstructed).hexdigest(),
            namespace["COLABCOLA_ARCHIVE_SHA256"],
        )

    def test_large_archive_uses_multiple_bounded_payload_cells(self) -> None:
        self.source.write_bytes(hashlib.shake_256(b"colabcola-fixture").digest(900_000))
        artifact = build_notebook(self.bundle())
        self.assertGreater(len(artifact.bundle.archive_bytes), RAW_PAYLOAD_CHUNK_BYTES)
        self.assertEqual(artifact.payload_cell_count, 2)
        self.assertEqual(len(artifact.payload_chunks[0]), 1024 * 1024)
        document = json.loads(artifact.notebook_bytes)
        payload_cells = document["cells"][2:]
        self.assertEqual(payload_cells[0]["metadata"]["colabcola"]["raw_bytes"], 768 * 1024)
        self.assertLessEqual(
            payload_cells[1]["metadata"]["colabcola"]["raw_bytes"],
            768 * 1024,
        )

    def test_unicode_is_preserved_without_host_paths_or_timestamps(self) -> None:
        mapping = notebook_job(self.source, target="workload/é›ª.py")
        artifact = build_notebook(self.bundle(mapping=mapping, job_args=("snowman-â˜ƒ",)))

        self.assertIn("é›ª".encode(), artifact.notebook_bytes)
        self.assertIn("â˜ƒ".encode(), artifact.notebook_bytes)
        self.assertNotIn(os.fsencode(self.root), artifact.notebook_bytes)
        self.assertNotIn(b"timestamp", artifact.notebook_bytes.lower())
        self.assertNotIn(artifact.notebook_sha256.encode("ascii"), artifact.notebook_bytes)

    def test_display_name_changes_summary_not_portable_job_identity(self) -> None:
        first_bundle = self.bundle(mapping=notebook_job(self.source, name="first-name"))
        second_bundle = self.bundle(mapping=notebook_job(self.source, name="second-name"))
        first = build_notebook(first_bundle)
        second = build_notebook(second_bundle)

        self.assertEqual(first.bundle.job_id, second.bundle.job_id)
        self.assertNotEqual(first.notebook_bytes, second.notebook_bytes)
        self.assertIn(b"first-name", first.notebook_bytes)
        self.assertIn(b"second-name", second.notebook_bytes)

    def test_notebook_and_chunk_corruption_are_detected(self) -> None:
        artifact = build_notebook(self.bundle())
        changed_bytes = artifact.notebook_bytes.replace(b"notebook-fixture", b"notebook-fixturz")
        changed_hash = hashlib.sha256(changed_bytes).hexdigest()
        with self.assertRaisesRegex(NotebookError, "deterministic notebook document"):
            verify_notebook(
                replace(
                    artifact,
                    notebook_bytes=changed_bytes,
                    notebook_sha256=changed_hash,
                )
            )

        changed_chunk = ("A" + artifact.payload_chunks[0][1:],) + artifact.payload_chunks[1:]
        with self.assertRaisesRegex(NotebookError, "payload chunks do not match"):
            verify_notebook(replace(artifact, payload_chunks=changed_chunk))

    def test_building_notebook_bytes_writes_no_sidecar_or_output(self) -> None:
        before = sorted(path.relative_to(self.root) for path in self.root.rglob("*"))
        artifact = build_notebook(self.bundle())
        after = sorted(path.relative_to(self.root) for path in self.root.rglob("*"))

        self.assertEqual(before, after)
        self.assertGreater(len(artifact.notebook_bytes), len(artifact.bundle.archive_bytes))
        self.assertFalse(any(path.suffix in {".zip", ".ipynb"} for path in self.root.rglob("*")))

    def test_generated_audio_fixture_round_trips_and_packages_verified_output(self) -> None:
        self.source.write_text(
            """from pathlib import Path
import math
import struct
import wave

output = Path("outputs/tone.wav")
output.parent.mkdir(parents=True, exist_ok=True)
samples = [int(12000 * math.sin(2 * math.pi * 440 * index / 8000)) for index in range(800)]
with wave.open(str(output), "wb") as stream:
    stream.setnchannels(1)
    stream.setsampwidth(2)
    stream.setframerate(8000)
    stream.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))
print("generated", output)
""",
            encoding="utf-8",
            newline="\n",
        )
        mapping = notebook_job(self.source)
        mapping["outputs"] = [
            {
                "pattern": "outputs/*.wav",
                "required": True,
                "maximum_bytes": 4096,
            }
        ]
        artifact = build_notebook(self.bundle(mapping=mapping))
        second = build_notebook(self.bundle(mapping=mapping))
        self.assertEqual(artifact.notebook_bytes, second.notebook_bytes)

        content_root = self.root / "content"
        namespace = {"COLABCOLA_CONTENT_ROOT_OVERRIDE": os.fspath(content_root)}
        document = json.loads(artifact.notebook_bytes)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            for cell in document["cells"]:
                if cell["cell_type"] == "code":
                    exec(cell["source"], namespace)

        result_root = content_root / "colabcola" / "results" / artifact.bundle.job_id
        job_root = content_root / "colabcola" / "jobs" / artifact.bundle.job_id
        result = json.loads((result_root / "result.json").read_text(encoding="utf-8"))
        output = job_root / "outputs" / "tone.wav"
        self.assertEqual(result["terminal_state"], "success")
        self.assertEqual(result["process_exit_code"], 0)
        self.assertIsNone(result["notebook_sha256"])
        self.assertEqual(result["outputs"][0]["path"], "outputs/tone.wav")
        self.assertEqual(result["outputs"][0]["sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertEqual(output.stat().st_size, 1644)
        self.assertEqual(result["stdout"]["sha256"], hashlib.sha256((result_root / "stdout.log").read_bytes()).hexdigest())
        with wave.open(str(output), "rb") as stream:
            self.assertEqual((stream.getnchannels(), stream.getframerate(), stream.getnframes()), (1, 8000, 800))
        with zipfile.ZipFile(result_root / "artifacts.zip") as archive:
            self.assertEqual(archive.namelist(), ["result.json", "outputs/tone.wav"])
            self.assertEqual(archive.read("outputs/tone.wav"), output.read_bytes())

    def test_preflight_failure_writes_result_before_final_exception(self) -> None:
        mapping = notebook_job(self.source)
        mapping["python"]["requires_python"] = ">=99"
        artifact = build_notebook(self.bundle(mapping=mapping))
        content_root = self.root / "preflight-content"
        namespace = {"COLABCOLA_CONTENT_ROOT_OVERRIDE": os.fspath(content_root)}
        document = json.loads(artifact.notebook_bytes)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "preflight_failed"):
                for cell in document["cells"]:
                    if cell["cell_type"] == "code":
                        exec(cell["source"], namespace)
        result_root = content_root / "colabcola" / "results" / artifact.bundle.job_id
        result = json.loads((result_root / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["terminal_state"], "preflight_failed")
        self.assertEqual(result["failure"]["stage"], "preflight")
        self.assertTrue((result_root / "artifacts.zip").is_file())

    def test_corrupt_runtime_chunk_produces_integrity_failure_evidence(self) -> None:
        artifact = build_notebook(self.bundle())
        content_root = self.root / "integrity-content"
        namespace = {"COLABCOLA_CONTENT_ROOT_OVERRIDE": os.fspath(content_root)}
        document = json.loads(artifact.notebook_bytes)
        code_cells = [cell for cell in document["cells"] if cell["cell_type"] == "code"]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(code_cells[0]["source"], namespace)
            for cell in code_cells[1:-4]:
                exec(cell["source"], namespace)
            chunks = namespace["COLABCOLA_PAYLOAD_CHUNKS"]
            chunks[0] = "!" + chunks[0][1:]
            exec(code_cells[-4]["source"], namespace)
            exec(code_cells[-3]["source"], namespace)
            exec(code_cells[-2]["source"], namespace)
            with self.assertRaisesRegex(RuntimeError, "integrity_failed"):
                exec(code_cells[-1]["source"], namespace)
        result_root = content_root / "colabcola" / "results" / artifact.bundle.job_id
        result = json.loads((result_root / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["terminal_state"], "integrity_failed")
        self.assertEqual(result["failure"]["stage"], "integrity")

    def test_job_and_output_failures_are_distinct_and_durable(self) -> None:
        cases = (
            ("import sys\nsys.stderr.write('failed\\n')\nraise SystemExit(7)\n", [], "job_failed", 7),
            (
                "print('no output')\n",
                [{"pattern": "outputs/*.wav", "required": True, "maximum_bytes": 10}],
                "output_validation_failed",
                0,
            ),
        )
        for index, (source, outputs, expected_state, expected_exit) in enumerate(cases):
            with self.subTest(state=expected_state):
                self.source.write_text(source, encoding="utf-8", newline="\n")
                mapping = notebook_job(self.source)
                mapping["outputs"] = outputs
                artifact = build_notebook(self.bundle(mapping=mapping))
                content_root = self.root / f"failure-content-{index}"
                namespace = {"COLABCOLA_CONTENT_ROOT_OVERRIDE": os.fspath(content_root)}
                document = json.loads(artifact.notebook_bytes)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, expected_state):
                        for cell in document["cells"]:
                            if cell["cell_type"] == "code":
                                exec(cell["source"], namespace)
                result_root = content_root / "colabcola" / "results" / artifact.bundle.job_id
                result = json.loads((result_root / "result.json").read_text(encoding="utf-8"))
                self.assertEqual(result["terminal_state"], expected_state)
                self.assertEqual(result["process_exit_code"], expected_exit)

    def test_invalid_embedded_wheel_produces_environment_failure_evidence(self) -> None:
        wheel = self.root / "fixture-0.0-py3-none-any.whl"
        wheel.write_bytes(b"not a wheel")
        mapping = notebook_job(self.source)
        mapping["payload"].append(
            {"role": "wheel", "source": os.fspath(wheel), "target": "wheels/fixture-0.0-py3-none-any.whl"}
        )
        mapping["python"]["embedded_wheels"] = ["wheels/fixture-0.0-py3-none-any.whl"]
        artifact = build_notebook(self.bundle(mapping=mapping))
        content_root = self.root / "environment-content"
        namespace = {"COLABCOLA_CONTENT_ROOT_OVERRIDE": os.fspath(content_root)}
        document = json.loads(artifact.notebook_bytes)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "environment_failed"):
                for cell in document["cells"]:
                    if cell["cell_type"] == "code":
                        exec(cell["source"], namespace)
        result_root = content_root / "colabcola" / "results" / artifact.bundle.job_id
        result = json.loads((result_root / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["terminal_state"], "environment_failed")
        self.assertEqual(result["failure"]["stage"], "environment")


if __name__ == "__main__":
    unittest.main()
