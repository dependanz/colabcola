from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from colabcola._hifigan.extract_audio_diagnostic_input import extract_checkpoint
from colabcola.spec import JobSpec


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
JOB_PATH = REPOSITORY_ROOT / "jobs" / "hifigan-ljspeech-audio-diagnostic.colajob"
SCRIPT_PATH = REPOSITORY_ROOT / "src" / "colabcola" / "_hifigan" / "audio_diagnostic.py"


class AudioDiagnosticPreparationTests(unittest.TestCase):
    def test_checkpoint_extractor_is_verified_and_idempotent(self) -> None:
        payload = b"verified-checkpoint-fixture"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "artifacts.zip"
            output_path = root / "input" / "g_00005000"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("outputs/cp_hifigan/g_00005000", payload)

            first = extract_checkpoint(
                archive_path,
                output_path,
                expected_size=len(payload),
                expected_sha256=digest,
            )
            second = extract_checkpoint(
                archive_path,
                output_path,
                expected_size=len(payload),
                expected_sha256=digest,
            )

            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            self.assertEqual(output_path.read_bytes(), payload)

    def test_jobspec_contract_is_bounded(self) -> None:
        mapping = json.loads(JOB_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for index, entry in enumerate(mapping["payload"]):
                fixture = root / f"payload-{index}"
                fixture.write_bytes(f"fixture-{index}".encode())
                entry["source"] = str(fixture)
            spec = JobSpec.from_mapping(mapping, base_dir=JOB_PATH.parent)

        self.assertEqual(spec.resources.accelerator, "gpu")
        self.assertTrue(spec.resources.cuda_required)
        self.assertEqual(len(spec.payload), 12)
        self.assertEqual(len(spec.outputs), 3)
        self.assertEqual(
            [output.pattern for output in spec.outputs],
            [
                "outputs/audio/reference/*.wav",
                "outputs/audio/generated/*.wav",
                "outputs/audio_diagnostic.json",
            ],
        )

    def test_diagnostic_script_has_safety_and_interpretation_boundaries(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("not a convergence, perceptual-quality", source)
        self.assertIn("np.clip", source)
        self.assertIn("generator checkpoint SHA-256 does not match", source)
        self.assertIn("validation WAV identity does not match", source)
        self.assertIn("audio diagnostic requires one CUDA GPU", source)


if __name__ == "__main__":
    unittest.main()
