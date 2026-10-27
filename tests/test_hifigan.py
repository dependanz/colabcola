from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from colabcola import (
    JobSpec,
    SpecError,
    bind_job_arguments,
    build_bundle,
    hifigan_source_payload,
    load_hifigan_profile,
    verify_hifigan_checkout,
    verify_hifigan_profile_assets,
)


UPSTREAM_COMMIT = "4769534d45265d52a904b850da5a622601885777"
BACKEND_INFO_COMMIT = "ec53f93fe44a9cab118dc5451527b18e36e1a643"


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class HiFiGanProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def custom_profile(self) -> tuple[Path, Path]:
        checkout = self.root / "checkout"
        checkout.mkdir()
        license_bytes = b"fixture license\n"
        source_bytes = b"VALUE = 1\n"
        (checkout / "LICENSE").write_bytes(license_bytes)
        (checkout / "source.py").write_bytes(source_bytes)

        assets = self.root / "assets"
        assets.mkdir()
        overlay_bytes = b"print('overlay')\n"
        (assets / "overlay.py").write_bytes(overlay_bytes)
        profile = {
            "schema_version": 1,
            "workload": "hifi-gan-v1",
            "upstream": {
                "repository": "https://example.test/hifi-gan.git",
                "commit": "a" * 40,
                "license_spdx": "MIT",
                "license_notice": "fixture notice",
                "files": [
                    {
                        "path": "LICENSE",
                        "size": len(license_bytes),
                        "sha256": digest(license_bytes),
                        "role": "license",
                        "bundle_target": "workload/hifi_gan/LICENSE",
                    },
                    {
                        "path": "source.py",
                        "size": len(source_bytes),
                        "sha256": digest(source_bytes),
                        "role": "source",
                        "bundle_target": "workload/hifi_gan/source.py",
                    },
                ],
            },
            "overlays": [
                {
                    "path": "overlay.py",
                    "size": len(overlay_bytes),
                    "sha256": digest(overlay_bytes),
                    "role": "source",
                    "bundle_target": "workload/hifi_gan/overlay.py",
                }
            ],
            "runtime": {
                "provider": "google_colab",
                "runtime_version": "fixture",
                "backend_info_commit": "b" * 40,
                "gpu_package_snapshot_url": "https://example.test/pip-freeze.gpu.txt",
                "requires_python": "==3.12.13",
                "runtime_requirements": ["torch==2.11.0+cu128"],
                "install_requirements": [],
            },
            "smoke": {
                "initial_completed_steps": 100,
                "resumed_completed_steps": 110,
                "checkpoint_interval": 50,
            },
        }
        profile_path = assets / "profile.json"
        profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        return checkout, profile_path

    def test_checked_in_profile_pins_source_runtime_lock_and_budget(self) -> None:
        profile = load_hifigan_profile()

        self.assertEqual(profile.repository, "https://github.com/jik876/hifi-gan.git")
        self.assertEqual(profile.commit, UPSTREAM_COMMIT)
        self.assertEqual(profile.license_spdx, "MIT")
        self.assertEqual(profile.license_notice, "Copyright (c) 2020 Jungil Kong")
        self.assertEqual(len(profile.source_files), 11)
        self.assertEqual(profile.runtime.provider, "google_colab")
        self.assertEqual(profile.runtime.runtime_version, "2026.07")
        self.assertEqual(profile.runtime.backend_info_commit, BACKEND_INFO_COMMIT)
        self.assertEqual(profile.runtime.requires_python, "==3.12.13")
        self.assertEqual(
            profile.runtime.runtime_requirements,
            (
                "torch==2.11.0+cu128",
                "numpy==2.0.2",
                "librosa==0.11.0",
                "scipy==1.16.3",
                "tensorboard==2.20.0",
                "soundfile==0.14.0",
                "matplotlib==3.10.0",
            ),
        )
        self.assertEqual(profile.runtime.install_requirements, ())
        self.assertEqual(
            {overlay.path: overlay.bundle_target for overlay in profile.overlays},
            {
                "train_colabcola.py": "workload/hifi_gan/train_colabcola.py",
                "resume_state.py": "workload/hifi_gan/resume_state.py",
            },
        )
        self.assertEqual(
            (
                profile.initial_completed_steps,
                profile.resumed_completed_steps,
                profile.checkpoint_interval,
            ),
            (100, 110, 50),
        )
        self.assertEqual(len(verify_hifigan_profile_assets(profile)), 2)

    def test_checked_in_source_records_cover_runtime_and_split_authorities(self) -> None:
        profile = load_hifigan_profile()
        records = {record.path: record for record in profile.source_files}

        self.assertEqual(
            set(records),
            {
                "LICENSE",
                "README.md",
                "config_v1.json",
                "env.py",
                "meldataset.py",
                "models.py",
                "requirements.txt",
                "train.py",
                "utils.py",
                "LJSpeech-1.1/training.txt",
                "LJSpeech-1.1/validation.txt",
            },
        )
        self.assertIsNone(records["LJSpeech-1.1/training.txt"].bundle_target)
        self.assertIsNone(records["LJSpeech-1.1/validation.txt"].bundle_target)
        self.assertEqual(
            records["LICENSE"].sha256,
            "dc2e017a9a4430fbd21aa83345310fa34b7050d6c1d7a06d30436f7e5c4be2f4",
        )

    def test_overlay_is_syntax_valid_and_encodes_modern_and_exact_step_semantics(self) -> None:
        profile = load_hifigan_profile()
        overlays = {path.name: path for path in verify_hifigan_profile_assets(profile)}
        trainer = overlays["train_colabcola.py"]
        resume_state = overlays["resume_state.py"]
        source = trainer.read_text(encoding="utf-8")

        compile(source, os.fspath(trainer), "exec")
        compile(resume_state.read_text(encoding="utf-8"), os.fspath(resume_state), "exec")
        self.assertIn("return_complex=True", source)
        self.assertIn("n_mels=num_mels", source)
        self.assertIn("completed_steps += 1", source)
        self.assertIn('"completed_steps": completed_steps', source)
        self.assertIn("if completed_steps >= a.maximum_steps:", source)
        self.assertIn("if last_checkpointed_step != completed_steps:", source)
        self.assertIn("if not torch.cuda.is_available():", source)
        self.assertIn('"starting_step": starting_step', source)
        self.assertIn('"next_batch_in_epoch": next_batch_in_epoch', source)
        self.assertIn('"python_random_state": python_random_state', source)
        self.assertIn("normalize_torch_rng_state_devices", source)
        self.assertIn("remaining_epoch_sample_keys", source)
        self.assertIn("--checkpoint_retention", source)
        self.assertIn("def _atomic_save_checkpoint", source)
        self.assertIn("os.fsync(stream.fileno())", source)
        self.assertIn("os.replace(temporary, path)", source)
        self.assertIn('"retained_checkpoint_steps": list(retained_checkpoint_steps)', source)
        self.assertIn("one visible compatible CUDA GPU", source)

    def test_checkout_verifier_accepts_exact_bytes_then_rejects_tampering(self) -> None:
        checkout, profile_path = self.custom_profile()
        profile = load_hifigan_profile(profile_path)

        verified = verify_hifigan_checkout(checkout, profile)
        self.assertEqual(len(verified), 2)
        (checkout / "source.py").write_bytes(b"VALUE = 2\n")
        with self.assertRaisesRegex(SpecError, "SHA-256 mismatch"):
            verify_hifigan_checkout(checkout, profile)

    def test_payload_places_license_profile_and_overlay_in_bundle_manifest(self) -> None:
        checkout, profile_path = self.custom_profile()
        entries = hifigan_source_payload(checkout, profile_path)
        payload = [
            {"role": item.role, "source": os.fspath(item.source), "target": item.target}
            for item in entries
        ]
        spec = JobSpec.from_mapping(
            {
                "schema_version": 1,
                "name": "hifigan-profile-fixture",
                "payload": payload,
                "entrypoint": {
                    "kind": "python_script",
                    "path": "workload/hifi_gan/overlay.py",
                    "argv": ["--maximum_steps", "100"],
                    "cwd": "workload/hifi_gan",
                },
                "python": {
                    "requires_python": "==3.12.13",
                    "runtime_requirements": ["torch==2.11.0+cu128"],
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
                    "environment": {},
                    "capture_stdout": True,
                    "capture_stderr": True,
                },
                "outputs": [],
                "limits": {
                    "maximum_file_count": 20,
                },
            },
            base_dir=self.root,
        )
        bundle = build_bundle(bind_job_arguments(spec))
        manifest = json.loads(bundle.canonical_manifest_bytes)
        records = {record["path"]: record for record in manifest["files"]}

        self.assertEqual(records["workload/hifi_gan/LICENSE"]["role"], "license")
        self.assertEqual(
            records["workload/hifi_gan/HIFIGAN_PROFILE.json"]["role"], "license"
        )
        self.assertEqual(records["workload/hifi_gan/overlay.py"]["role"], "source")
        self.assertEqual(
            records["workload/hifi_gan/HIFIGAN_PROFILE.json"]["sha256"],
            digest(profile_path.read_bytes()),
        )
        bundle.verify()

    def test_profile_rejects_duplicate_json_keys_and_bad_overlay_hash(self) -> None:
        duplicate = self.root / "duplicate.json"
        duplicate.write_text('{"schema_version":1,"schema_version":1}\n', encoding="utf-8")
        with self.assertRaisesRegex(SpecError, "duplicate HiFi-GAN profile key"):
            load_hifigan_profile(duplicate)

        _, profile_path = self.custom_profile()
        value = json.loads(profile_path.read_text(encoding="utf-8"))
        value["overlays"][0]["sha256"] = "BAD"
        profile_path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(SpecError, "lowercase SHA-256"):
            load_hifigan_profile(profile_path)


if __name__ == "__main__":
    unittest.main()
