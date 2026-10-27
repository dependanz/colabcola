from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import struct
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest import mock

from colabcola import (
    ARCHIVE_WARNING_BYTES,
    BundleError,
    JobSpec,
    SpecError,
    archive_size_warnings,
    bind_job_arguments,
    build_bundle,
    verify_bundle,
)


def job_mapping(
    payload: list[dict[str, object]],
    *,
    name: str = "bundle-fixture",
    entrypoint_path: str = "workload/main.py",
    cwd: str = ".",
    maximum_file_count: int = 100,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": name,
        "payload": payload,
        "entrypoint": {
            "kind": "python_script",
            "path": entrypoint_path,
            "argv": ["--configured", "value"],
            "cwd": cwd,
        },
        "python": {
            "requires_python": ">=3.11",
            "runtime_requirements": [],
            "install_requirements": [],
            "embedded_wheels": [],
        },
        "runtime": {"provider": "google_colab", "version": "latest"},
        "resources": {
            "accelerator": "cpu",
            "gpu_count": 0,
            "cuda_required": False,
            "minimum_gpu_memory_bytes": None,
        },
        "run": {
            "timeout_seconds": 60,
            "seed": 1234,
            "environment": {"COLABCOLA_MODE": "test"},
            "capture_stdout": True,
            "capture_stderr": True,
        },
        "outputs": [],
        "limits": {
            "maximum_file_count": maximum_file_count,
        },
    }


class BundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.workload = self.root / "workload"
        self.workload.mkdir()
        (self.workload / "main.py").write_text("print('ok')\n", encoding="utf-8")

    def payload_file(
        self,
        source: Path | None = None,
        target: str = "workload/main.py",
        *,
        role: str = "source",
        exclude: list[str] | None = None,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "role": role,
            "source": os.fspath(source or self.workload / "main.py"),
            "target": target,
        }
        if exclude is not None:
            result["exclude"] = exclude
        return result

    def invocation(
        self,
        mapping: dict[str, object],
        job_args: tuple[str, ...] = (),
    ):
        spec = JobSpec.from_mapping(mapping, base_dir=self.root)
        return bind_job_arguments(spec, job_args)

    def test_directory_expansion_exclusions_and_manifest_records(self) -> None:
        (self.workload / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
        (self.workload / "ignored.tmp").write_text("ignore", encoding="utf-8")
        data = self.root / "data"
        (data / "cache").mkdir(parents=True)
        (data / "nested").mkdir()
        (data / "a.bin").write_bytes(b"a")
        (data / "cache" / "skip.bin").write_bytes(b"skip")
        (data / "nested" / "z.bin").write_bytes(b"z")
        payload = [
            self.payload_file(
                self.workload,
                "workload",
                exclude=["**/*.tmp"],
            ),
            self.payload_file(
                data,
                "data",
                role="data",
                exclude=["cache/**"],
            ),
        ]

        bundle = build_bundle(
            self.invocation(
                job_mapping(payload, entrypoint_path="workload/main.py", cwd="workload")
            )
        )

        self.assertEqual(
            [record.path for record in bundle.files],
            [
                "data/a.bin",
                "data/nested/z.bin",
                "workload/helper.py",
                "workload/main.py",
            ],
        )
        self.assertEqual([record.role for record in bundle.files[:2]], ["data", "data"])
        manifest = json.loads(bundle.canonical_manifest_bytes)
        self.assertEqual(manifest["files"], [record.as_manifest_value() for record in bundle.files])
        self.assertEqual(manifest["entrypoint"]["cwd"], "workload")
        bundle.verify()

    def test_durable_checkpoint_contract_is_bound_into_job_identity(self) -> None:
        mapping = job_mapping([self.payload_file()])
        baseline = build_bundle(self.invocation(mapping))
        mapping["durable_checkpoints"] = {
            "directory": "workload/checkpoints",
            "generator_prefix": "g",
            "state_prefix": "do",
            "filename_digits": 8,
            "poll_seconds": 30,
            "remote_retention": 2,
        }
        durable = build_bundle(self.invocation(mapping))
        manifest = json.loads(durable.canonical_manifest_bytes)

        self.assertNotIn(
            "durable_checkpoints", json.loads(baseline.canonical_manifest_bytes)
        )
        self.assertEqual(manifest["durable_checkpoints"], mapping["durable_checkpoints"])
        self.assertNotEqual(baseline.job_id, durable.job_id)

    def test_repeated_builds_and_payload_declaration_order_are_identical(self) -> None:
        data = self.root / "data.bin"
        data.write_bytes(b"payload")
        source_entry = self.payload_file()
        data_entry = self.payload_file(data, "data/data.bin", role="data")
        first = build_bundle(self.invocation(job_mapping([source_entry, data_entry])))
        second = build_bundle(self.invocation(job_mapping([data_entry, source_entry])))

        self.assertEqual(first.canonical_manifest_bytes, second.canonical_manifest_bytes)
        self.assertEqual(first.archive_bytes, second.archive_bytes)
        self.assertEqual(first.job_id, second.job_id)

    def test_effective_arguments_change_manifest_and_identity_not_archive(self) -> None:
        mapping = job_mapping([self.payload_file()])
        first = build_bundle(self.invocation(mapping, ("--epochs", "10")))
        second = build_bundle(self.invocation(mapping, ("--epochs", "20")))

        self.assertEqual(first.archive_bytes, second.archive_bytes)
        self.assertNotEqual(first.canonical_manifest_bytes, second.canonical_manifest_bytes)
        self.assertNotEqual(first.job_id, second.job_id)
        manifest = json.loads(second.canonical_manifest_bytes)
        self.assertEqual(
            manifest["entrypoint"]["argv"],
            ["--configured", "value", "--epochs", "20"],
        )

    def test_display_name_and_host_source_location_do_not_affect_identity(self) -> None:
        mirror = self.root / "mirror.py"
        mirror.write_bytes((self.workload / "main.py").read_bytes())
        first = build_bundle(
            self.invocation(job_mapping([self.payload_file()], name="first-name"))
        )
        second = build_bundle(
            self.invocation(
                job_mapping(
                    [self.payload_file(mirror, "workload/main.py")],
                    name="second-name",
                )
            )
        )

        self.assertEqual(first.canonical_manifest_bytes, second.canonical_manifest_bytes)
        self.assertEqual(first.archive_bytes, second.archive_bytes)
        self.assertEqual(first.job_id, second.job_id)
        self.assertNotEqual(first.name, second.name)

    def test_changed_file_content_changes_archive_and_identity(self) -> None:
        mapping = job_mapping([self.payload_file()])
        first = build_bundle(self.invocation(mapping))
        (self.workload / "main.py").write_text("print('changed')\n", encoding="utf-8")
        second = build_bundle(self.invocation(mapping))

        self.assertNotEqual(first.archive_sha256, second.archive_sha256)
        self.assertNotEqual(first.job_id, second.job_id)

    def test_duplicate_and_prefix_expansion_collisions_are_rejected(self) -> None:
        data_dir = self.root / "data"
        data_dir.mkdir()
        (data_dir / "item.bin").write_bytes(b"first")
        other = self.root / "other.bin"
        other.write_bytes(b"second")
        source = self.payload_file()

        duplicate = job_mapping(
            [
                source,
                self.payload_file(data_dir, "payload", role="data"),
                self.payload_file(other, "payload/item.bin", role="data"),
            ]
        )
        with self.assertRaisesRegex(SpecError, "duplicate target: payload/item.bin"):
            build_bundle(self.invocation(duplicate))

        prefix = job_mapping(
            [
                source,
                self.payload_file(data_dir / "item.bin", "payload/item", role="data"),
                self.payload_file(other, "payload/item/child", role="data"),
            ]
        )
        with self.assertRaisesRegex(SpecError, "file/directory target conflict"):
            build_bundle(self.invocation(prefix))

    def test_file_count_limit_is_enforced_after_expansion(self) -> None:
        (self.workload / "helper.py").write_text("pass\n", encoding="utf-8")
        mapping = job_mapping(
            [self.payload_file(self.workload, "workload")],
            maximum_file_count=1,
        )
        with self.assertRaisesRegex(SpecError, "maximum_file_count=1"):
            build_bundle(self.invocation(mapping))

    def test_archive_size_warning_is_nonblocking_structured_and_verified(self) -> None:
        self.assertEqual(archive_size_warnings(ARCHIVE_WARNING_BYTES - 1), ())
        warning = archive_size_warnings(ARCHIVE_WARNING_BYTES)
        self.assertEqual(len(warning), 1)
        self.assertIn("1073741824-byte (1024 MiB) warning threshold", warning[0])
        self.assertIn("1431655768 bytes before notebook JSON overhead", warning[0])

        with mock.patch("colabcola.bundle.ARCHIVE_WARNING_BYTES", 1):
            with self.assertWarnsRegex(RuntimeWarning, "warning threshold"):
                bundle = build_bundle(self.invocation(job_mapping([self.payload_file()])))
            self.assertEqual(len(bundle.warnings), 1)
            bundle.verify()
            with self.assertRaisesRegex(BundleError, "warnings do not match"):
                verify_bundle(replace(bundle, warnings=()))

    def test_exclusions_require_directory_sources_and_portable_globs(self) -> None:
        file_mapping = job_mapping(
            [self.payload_file(exclude=["*.tmp"])]
        )
        with self.assertRaisesRegex(SpecError, "only valid for directory sources"):
            build_bundle(self.invocation(file_mapping))

        directory_mapping = job_mapping(
            [self.payload_file(self.workload, "workload", exclude=["../secret"])]
        )
        with self.assertRaisesRegex(SpecError, "normalized relative POSIX glob"):
            build_bundle(self.invocation(directory_mapping))

    def test_entrypoint_must_survive_expansion_as_a_source_file(self) -> None:
        (self.workload / "helper.py").write_text("pass\n", encoding="utf-8")
        mapping = job_mapping(
            [self.payload_file(self.workload, "workload", exclude=["main.py"])]
        )
        with self.assertRaisesRegex(SpecError, "entrypoint.path must resolve"):
            build_bundle(self.invocation(mapping))

    def test_zip_metadata_is_normalized(self) -> None:
        bundle = build_bundle(self.invocation(job_mapping([self.payload_file()])))
        with zipfile.ZipFile(io.BytesIO(bundle.archive_bytes), "r") as archive:
            self.assertEqual(archive.comment, b"")
            info = archive.infolist()[0]
            self.assertEqual(info.filename, "workload/main.py")
            self.assertEqual(info.date_time, (1980, 1, 1, 0, 0, 0))
            self.assertEqual(info.compress_type, zipfile.ZIP_DEFLATED)
            self.assertEqual(info.create_system, 3)
            self.assertEqual(info.extra, b"")
            self.assertEqual(info.comment, b"")
            archived_mode = info.external_attr >> 16
            self.assertTrue(stat.S_ISREG(archived_mode))
            self.assertEqual(stat.S_IMODE(archived_mode), 0o644)

    def test_zip64_offset_metadata_is_canonical_when_required(self) -> None:
        for index in range(4):
            (self.workload / f"payload-{index}.bin").write_bytes(bytes(range(80)))
        mapping = job_mapping(
            [self.payload_file(self.workload, "workload")],
            maximum_file_count=10,
        )

        with mock.patch.object(zipfile, "ZIP64_LIMIT", 128):
            bundle = build_bundle(self.invocation(mapping))
            with zipfile.ZipFile(io.BytesIO(bundle.archive_bytes), "r") as archive:
                infos = archive.infolist()
                self.assertTrue(any(info.extra for info in infos))
                for info in infos:
                    if info.header_offset <= zipfile.ZIP64_LIMIT:
                        self.assertEqual(info.extra, b"")
                        self.assertEqual(info.create_version, 20)
                        self.assertEqual(info.extract_version, 20)
                        continue
                    expected = struct.pack("<HHQ", 0x0001, 8, info.header_offset)
                    self.assertEqual(info.extra, expected)
                    self.assertEqual(info.create_version, zipfile.ZIP64_VERSION)
                    self.assertEqual(info.extract_version, zipfile.ZIP64_VERSION)

    def test_corruption_and_record_tampering_are_detected(self) -> None:
        bundle = build_bundle(self.invocation(job_mapping([self.payload_file()])))
        damaged_archive = bytearray(bundle.archive_bytes)
        damaged_archive[len(damaged_archive) // 2] ^= 0x01
        with self.assertRaisesRegex(BundleError, "archive SHA-256 does not match"):
            verify_bundle(bundle=replace(bundle, archive_bytes=bytes(damaged_archive)))

        damaged_record = replace(bundle.files[0], sha256="0" * 64)
        with self.assertRaisesRegex(BundleError, "file records do not match"):
            verify_bundle(bundle=replace(bundle, files=(damaged_record,)))

    def test_manifest_is_canonical_host_free_and_uses_framed_job_identity(self) -> None:
        bundle = build_bundle(
            self.invocation(job_mapping([self.payload_file()]), ("snowman-☃",))
        )
        manifest = json.loads(bundle.canonical_manifest_bytes)
        self.assertTrue(bundle.canonical_manifest_bytes.endswith(b"\n"))
        self.assertNotIn("name", manifest)
        self.assertNotIn(os.fsencode(self.root), bundle.canonical_manifest_bytes)
        self.assertNotIn(b"timestamp", bundle.canonical_manifest_bytes)
        self.assertEqual(manifest["compiler_contract_version"], "0.10")
        self.assertEqual(manifest["archive"]["compression_level"], 9)

        archive_digest = bytes.fromhex(bundle.archive_sha256)
        framed = (
            len(bundle.canonical_manifest_bytes).to_bytes(8, "big")
            + bundle.canonical_manifest_bytes
            + len(archive_digest).to_bytes(8, "big")
            + archive_digest
        )
        self.assertEqual(hashlib.sha256(framed).hexdigest(), bundle.job_id)


if __name__ == "__main__":
    unittest.main()
