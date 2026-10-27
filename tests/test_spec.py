from __future__ import annotations

import copy
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from colabcola import JobSpec, SpecError, load_job_spec


def valid_spec(source: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "fixture-job",
        "payload": [
            {
                "role": "source",
                "source": os.fspath(source),
                "target": "workload/main.py",
            }
        ],
        "entrypoint": {
            "kind": "python_script",
            "path": "workload/main.py",
            "argv": [],
            "cwd": ".",
        },
        "python": {
            "requires_python": ">=3.11,<3.13",
            "runtime_requirements": ["torch>=2.6"],
            "install_requirements": [],
            "embedded_wheels": [],
        },
        "runtime": {"provider": "google_colab", "version": "2026.07"},
        "resources": {
            "accelerator": "gpu",
            "gpu_count": 1,
            "cuda_required": True,
            "minimum_gpu_memory_bytes": None,
        },
        "run": {
            "timeout_seconds": 3600,
            "seed": 1234,
            "environment": {"COLABCOLA_MODE": "test"},
            "capture_stdout": True,
            "capture_stderr": True,
        },
        "outputs": [
            {
                "pattern": "outputs/**",
                "required": True,
                "maximum_bytes": 1024,
            }
        ],
        "limits": {
            "maximum_file_count": 100,
        },
    }


class JobSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "main.py"
        self.source.write_text("print('ok')\n", encoding="utf-8")

    def test_valid_spec_parses_to_immutable_model(self) -> None:
        spec = JobSpec.from_mapping(valid_spec(self.source), base_dir=self.root)
        self.assertEqual(spec.schema_version, 1)
        self.assertEqual(spec.name, "fixture-job")
        self.assertEqual(spec.payload[0].source, self.source)
        self.assertEqual(spec.resources.accelerator, "gpu")
        self.assertEqual(spec.runtime.provider, "google_colab")
        self.assertEqual(spec.runtime.version, "2026.07")
        self.assertEqual(dict(spec.run.environment), {"COLABCOLA_MODE": "test"})
        self.assertEqual(spec.limits.maximum_file_count, 100)
        self.assertIsNone(spec.durable_checkpoints)

    def test_durable_checkpoint_contract_is_strict(self) -> None:
        value = valid_spec(self.source)
        value["durable_checkpoints"] = {
            "directory": "workload/checkpoints",
            "generator_prefix": "g",
            "state_prefix": "do",
            "filename_digits": 8,
            "poll_seconds": 30,
            "remote_retention": 2,
        }
        spec = JobSpec.from_mapping(value, base_dir=self.root)
        assert spec.durable_checkpoints is not None
        self.assertEqual(spec.durable_checkpoints.directory, "workload/checkpoints")
        self.assertEqual(spec.durable_checkpoints.remote_retention, 2)

        for field, invalid, message in (
            ("directory", "../checkpoints", "normalized relative POSIX path"),
            ("generator_prefix", "g/path", "portable filename prefix"),
            ("filename_digits", 17, "must not exceed 16"),
            ("poll_seconds", 4, "between 5 and 3600"),
            ("remote_retention", 1, "between 2 and 10"),
        ):
            with self.subTest(field=field):
                invalid_value = copy.deepcopy(value)
                invalid_value["durable_checkpoints"][field] = invalid  # type: ignore[index]
                with self.assertRaisesRegex(SpecError, message):
                    JobSpec.from_mapping(invalid_value, base_dir=self.root)

        duplicate_prefix = copy.deepcopy(value)
        duplicate_prefix["durable_checkpoints"]["state_prefix"] = "g"  # type: ignore[index]
        with self.assertRaisesRegex(SpecError, "prefixes must be distinct"):
            JobSpec.from_mapping(duplicate_prefix, base_dir=self.root)

    def test_retired_archive_ceiling_field_is_rejected(self) -> None:
        value = valid_spec(self.source)
        value["limits"]["maximum_archive_bytes"] = 64 * 1024 * 1024  # type: ignore[index]
        with self.assertRaisesRegex(SpecError, "unknown fields: maximum_archive_bytes"):
            JobSpec.from_mapping(value, base_dir=self.root)

    def test_missing_required_field_is_rejected(self) -> None:
        value = valid_spec(self.source)
        del value["entrypoint"]
        with self.assertRaisesRegex(SpecError, "missing required fields: entrypoint"):
            JobSpec.from_mapping(value, base_dir=self.root)

    def test_unknown_top_level_and_nested_fields_are_rejected(self) -> None:
        top = valid_spec(self.source)
        top["surprise"] = True
        with self.assertRaisesRegex(SpecError, "unknown fields: surprise"):
            JobSpec.from_mapping(top, base_dir=self.root)

        nested = valid_spec(self.source)
        nested["resources"]["surprise"] = True  # type: ignore[index]
        with self.assertRaisesRegex(SpecError, "unknown fields: surprise"):
            JobSpec.from_mapping(nested, base_dir=self.root)

    def test_non_normalized_or_escaping_targets_are_rejected(self) -> None:
        invalid_targets = [
            "../main.py",
            "/main.py",
            "C:/main.py",
            "./main.py",
            "workload//main.py",
            "workload/../main.py",
            "workload\\main.py",
            "workload/main.py/",
        ]
        for target in invalid_targets:
            with self.subTest(target=target):
                value = valid_spec(self.source)
                value["payload"][0]["target"] = target  # type: ignore[index]
                with self.assertRaisesRegex(SpecError, "normalized relative POSIX path"):
                    JobSpec.from_mapping(value, base_dir=self.root)

    def test_duplicate_payload_targets_are_rejected(self) -> None:
        value = valid_spec(self.source)
        duplicate = copy.deepcopy(value["payload"][0])  # type: ignore[index]
        value["payload"].append(duplicate)  # type: ignore[union-attr]
        with self.assertRaisesRegex(SpecError, "duplicate targets: workload/main.py"):
            JobSpec.from_mapping(value, base_dir=self.root)

    def test_missing_source_is_rejected(self) -> None:
        value = valid_spec(self.root / "missing.py")
        with self.assertRaisesRegex(SpecError, "does not exist"):
            JobSpec.from_mapping(value, base_dir=self.root)

    def test_symlink_source_is_rejected_without_creating_a_symlink(self) -> None:
        value = valid_spec(self.source)
        metadata = SimpleNamespace(st_mode=stat.S_IFLNK | 0o777, st_file_attributes=0)
        with mock.patch("colabcola.spec.os.lstat", return_value=metadata):
            with self.assertRaisesRegex(SpecError, "symlink or reparse point"):
                JobSpec.from_mapping(value, base_dir=self.root)

    def test_special_file_source_is_rejected(self) -> None:
        value = valid_spec(self.source)
        metadata = SimpleNamespace(st_mode=stat.S_IFIFO | 0o600, st_file_attributes=0)
        with mock.patch("colabcola.spec.os.lstat", return_value=metadata):
            with self.assertRaisesRegex(SpecError, "forbidden special file"):
                JobSpec.from_mapping(value, base_dir=self.root)

    def test_secret_bearing_environment_key_is_rejected(self) -> None:
        value = valid_spec(self.source)
        value["run"]["API_TOKEN"] = "not-in-the-right-object"  # type: ignore[index]
        with self.assertRaisesRegex(SpecError, "unknown fields: API_TOKEN"):
            JobSpec.from_mapping(value, base_dir=self.root)

        value = valid_spec(self.source)
        value["run"]["environment"]["API_TOKEN"] = "secret"  # type: ignore[index]
        with self.assertRaisesRegex(SpecError, "secret values are forbidden"):
            JobSpec.from_mapping(value, base_dir=self.root)

    def test_runtime_seed_environment_key_is_reserved(self) -> None:
        value = valid_spec(self.source)
        value["run"]["environment"] = {"COLABCOLA_SEED": "999"}  # type: ignore[index]
        with self.assertRaisesRegex(SpecError, "reserved for the configured run seed"):
            JobSpec.from_mapping(value, base_dir=self.root)

    def test_runtime_provider_and_version_are_explicit_and_strict(self) -> None:
        value = valid_spec(self.source)
        value["runtime"]["provider"] = "generic_cloud"  # type: ignore[index]
        with self.assertRaisesRegex(SpecError, "provider must equal 'google_colab'"):
            JobSpec.from_mapping(value, base_dir=self.root)

        for version in ("2026.7", "2026.07.1", "current", ""):
            with self.subTest(version=version):
                value = valid_spec(self.source)
                value["runtime"]["version"] = version  # type: ignore[index]
                with self.assertRaisesRegex(SpecError, "nonempty|'latest' or YYYY.MM"):
                    JobSpec.from_mapping(value, base_dir=self.root)

    def test_dependency_requirements_reject_urls_and_option_injection(self) -> None:
        digest = "a" * 64
        valid = valid_spec(self.source)
        valid["python"]["install_requirements"] = [  # type: ignore[index]
            f"example-package==1.2.3 --hash=sha256:{digest}"
        ]
        parsed = JobSpec.from_mapping(valid, base_dir=self.root)
        self.assertEqual(parsed.python.install_requirements, tuple(valid["python"]["install_requirements"]))  # type: ignore[index]

        for requirement in (
            f"--extra-index-url=https://example.invalid package==1 --hash=sha256:{digest}",
            "package==1 --hash=sha256:short",
            f"package>=1 --hash=sha256:{digest}",
        ):
            with self.subTest(requirement=requirement):
                value = valid_spec(self.source)
                value["python"]["install_requirements"] = [requirement]  # type: ignore[index]
                with self.assertRaisesRegex(SpecError, "exact index pin"):
                    JobSpec.from_mapping(value, base_dir=self.root)

        value = valid_spec(self.source)
        value["python"]["runtime_requirements"] = ["package @ https://example.invalid/pkg.whl"]  # type: ignore[index]
        with self.assertRaisesRegex(SpecError, "installed-package constraint"):
            JobSpec.from_mapping(value, base_dir=self.root)

    def test_gpu_and_cpu_resource_invariants_are_rejected(self) -> None:
        value = valid_spec(self.source)
        value["resources"]["gpu_count"] = 0  # type: ignore[index]
        with self.assertRaisesRegex(SpecError, "GPU jobs require"):
            JobSpec.from_mapping(value, base_dir=self.root)

        value = valid_spec(self.source)
        value["resources"].update(  # type: ignore[union-attr]
            accelerator="cpu", gpu_count=0, cuda_required=False
        )
        value["resources"]["minimum_gpu_memory_bytes"] = 1  # type: ignore[index]
        with self.assertRaisesRegex(SpecError, "CPU jobs require"):
            JobSpec.from_mapping(value, base_dir=self.root)

    def test_embedded_wheel_must_reference_wheel_payload(self) -> None:
        value = valid_spec(self.source)
        value["python"]["embedded_wheels"] = ["wheels/demo.whl"]  # type: ignore[index]
        with self.assertRaisesRegex(SpecError, "do not reference wheel payload targets"):
            JobSpec.from_mapping(value, base_dir=self.root)

    def test_duplicate_json_keys_are_rejected(self) -> None:
        path = self.root / "job.json"
        path.write_text(
            '{"schema_version":1,"schema_version":1}', encoding="utf-8"
        )
        with self.assertRaisesRegex(SpecError, "duplicate JSON object key"):
            load_job_spec(path)

    def test_relative_source_is_resolved_against_job_spec(self) -> None:
        path = self.root / "job.json"
        value = valid_spec(self.source)
        value["payload"][0]["source"] = "main.py"  # type: ignore[index]
        path.write_text(json.dumps(value), encoding="utf-8")
        spec = load_job_spec(path)
        self.assertEqual(spec.payload[0].source, self.source)


if __name__ == "__main__":
    unittest.main()
