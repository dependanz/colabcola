from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from colabcola._hifigan.measure_full_transport import measure_dataset


class FullTransportProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "LJSpeech-1.1"
        self.wavs = self.root / "wavs"
        self.wavs.mkdir(parents=True)
        (self.root / "training.txt").write_bytes(b"LJ001-0001\nLJ001-0002\n")
        (self.root / "validation.txt").write_bytes(b"LJ001-0003\n")
        (self.wavs / "LJ001-0001.wav").write_bytes(b"one")
        (self.wavs / "LJ001-0002.wav").write_bytes(b"two-two")
        (self.wavs / "LJ001-0003.wav").write_bytes(b"three")

    def test_measurement_reconciles_split_topology_and_bytes(self) -> None:
        value = measure_dataset(
            self.root,
            expected_training_count=2,
            expected_validation_count=1,
            expected_wav_bytes=15,
            expected_payload_bytes=48,
        )
        self.assertEqual(value["state"], "transport_probe_complete")
        self.assertEqual(value["wav_file_count"], 3)
        self.assertEqual(value["dataset_payload_bytes"], 48)

    def test_measurement_rejects_an_extra_wav(self) -> None:
        (self.wavs / "LJ001-0004.wav").write_bytes(b"extra")
        with self.assertRaisesRegex(RuntimeError, "missing=0, extra=1"):
            measure_dataset(
                self.root,
                expected_training_count=2,
                expected_validation_count=1,
                expected_wav_bytes=15,
                expected_payload_bytes=48,
            )


if __name__ == "__main__":
    unittest.main()
