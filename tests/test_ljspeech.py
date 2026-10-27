from __future__ import annotations

import hashlib
import io
import json
import tarfile
import tempfile
import unittest
import wave
from dataclasses import replace
from pathlib import Path

from colabcola import SpecError, load_hifigan_profile
from colabcola.ljspeech import (
    LJSPEECH_MANIFEST_NAME,
    build_ljspeech_subset,
    ljspeech_subset_payload,
    select_ljspeech_ids,
    verify_ljspeech_subset,
)


def wav_bytes(frames: int, *, channels: int = 1, sample_rate: int = 22_050) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\0\0" * frames * channels)
    return output.getvalue()


class LJSpeechSubsetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.training_ids = tuple(f"LJ001-{index:04d}" for index in range(1, 71))
        self.validation_ids = tuple(f"LJ002-{index:04d}" for index in range(1, 11))
        self.training_split = self.root / "training.txt"
        self.validation_split = self.root / "validation.txt"
        training_lines = "".join(
            f"{clip_id}|training fixture\n" for clip_id in reversed(self.training_ids)
        )
        self.training_split.write_bytes(
            training_lines.encode()
        )
        self.validation_split.write_bytes(
            "".join(
                f"{clip_id}|validation fixture\n" for clip_id in reversed(self.validation_ids)
            ).encode()
        )
        self.archive = self.root / "LJSpeech-1.1.tar.bz2"
        self._write_archive()
        profile = load_hifigan_profile()
        split_paths = {
            "LJSpeech-1.1/training.txt": self.training_split,
            "LJSpeech-1.1/validation.txt": self.validation_split,
        }
        source_files = []
        for record in profile.source_files:
            path = split_paths.get(record.path)
            if path is None:
                source_files.append(record)
            else:
                source_files.append(
                    replace(
                        record,
                        size=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                )
        self.profile = replace(profile, source_files=tuple(source_files))

    def _write_archive(self, *, bad_clip: str | None = None) -> None:
        with tarfile.open(self.archive, "w:bz2") as archive:
            for offset, clip_id in enumerate(self.training_ids + self.validation_ids):
                content = wav_bytes(
                    2_205 + offset,
                    channels=2 if clip_id == bad_clip else 1,
                )
                info = tarfile.TarInfo(f"LJSpeech-1.1/wavs/{clip_id}.wav")
                info.size = len(content)
                info.mode = 0o644
                info.mtime = 0
                archive.addfile(info, io.BytesIO(content))

    def test_selection_is_lexicographic_unique_and_disjoint(self) -> None:
        selection = select_ljspeech_ids(self.training_split, self.validation_split)

        self.assertEqual(selection.training_ids, self.training_ids[:64])
        self.assertEqual(selection.validation_ids, self.validation_ids[:8])
        self.assertFalse(set(selection.training_ids) & set(selection.validation_ids))

    def test_build_and_verify_records_every_clip_and_transport_fact(self) -> None:
        destination = self.root / "subset"
        subset = build_ljspeech_subset(
            self.archive,
            self.training_split,
            self.validation_split,
            destination,
            profile=self.profile,
        )
        verified = verify_ljspeech_subset(destination, self.profile)
        manifest = verified.manifest

        self.assertEqual(subset.root, destination.resolve())
        self.assertEqual(len(manifest["clips"]), 72)
        self.assertEqual(manifest["aggregates"]["training_clip_count"], 64)
        self.assertEqual(manifest["aggregates"]["validation_clip_count"], 8)
        self.assertEqual(manifest["aggregates"]["payload_file_count"], 74)
        self.assertGreater(manifest["aggregates"]["transport_archive_bytes"], 0)
        self.assertEqual(manifest["aggregates"]["transport_base64_bytes"] % 4, 0)
        self.assertTrue((destination / LJSPEECH_MANIFEST_NAME).is_file())
        self.assertEqual(
            len(list((destination / "LJSpeech-1.1" / "wavs").glob("*.wav"))),
            72,
        )

    def test_repeated_builds_have_identical_manifests(self) -> None:
        first = build_ljspeech_subset(
            self.archive,
            self.training_split,
            self.validation_split,
            self.root / "first",
            profile=self.profile,
        )
        second = build_ljspeech_subset(
            self.archive,
            self.training_split,
            self.validation_split,
            self.root / "second",
            profile=self.profile,
        )

        self.assertEqual(first.manifest_path.read_bytes(), second.manifest_path.read_bytes())
        self.assertEqual(
            first.manifest["aggregates"]["transport_archive_sha256"],
            second.manifest["aggregates"]["transport_archive_sha256"],
        )

    def test_builder_and_verifier_support_a_larger_declared_subset(self) -> None:
        subset = build_ljspeech_subset(
            self.archive,
            self.training_split,
            self.validation_split,
            self.root / "larger",
            profile=self.profile,
            training_count=65,
            validation_count=9,
        )

        verified = verify_ljspeech_subset(subset.root, self.profile)
        self.assertEqual(len(verified.selection.training_ids), 65)
        self.assertEqual(len(verified.selection.validation_ids), 9)
        self.assertEqual(verified.manifest["aggregates"]["clip_count"], 74)
        self.assertEqual(verified.manifest["aggregates"]["payload_file_count"], 76)

    def test_verifier_rejects_tampered_wav(self) -> None:
        destination = self.root / "subset"
        subset = build_ljspeech_subset(
            self.archive,
            self.training_split,
            self.validation_split,
            destination,
            profile=self.profile,
        )
        selected = subset.selection.training_ids[0]
        path = destination / "LJSpeech-1.1" / "wavs" / f"{selected}.wav"
        content = bytearray(path.read_bytes())
        content[-1] ^= 1
        path.write_bytes(content)

        with self.assertRaisesRegex(SpecError, "changed after extraction"):
            verify_ljspeech_subset(destination, self.profile)

    def test_verifier_rejects_undeclared_payload_file(self) -> None:
        destination = self.root / "subset"
        build_ljspeech_subset(
            self.archive,
            self.training_split,
            self.validation_split,
            destination,
            profile=self.profile,
        )
        extra = destination / "LJSpeech-1.1" / "wavs" / "undeclared.bin"
        extra.write_bytes(b"not declared by the subset manifest")

        with self.assertRaisesRegex(SpecError, "undeclared file"):
            verify_ljspeech_subset(destination, self.profile)

    def test_invalid_audio_fails_without_publishing_partial_subset(self) -> None:
        bad_clip = self.training_ids[0]
        self._write_archive(bad_clip=bad_clip)
        destination = self.root / "bad-subset"

        with self.assertRaisesRegex(SpecError, "must be mono 16-bit PCM at 22050 Hz"):
            build_ljspeech_subset(
                self.archive,
                self.training_split,
                self.validation_split,
                destination,
                profile=self.profile,
            )
        self.assertFalse(destination.exists())
        self.assertFalse(any(self.root.glob(".bad-subset.staging-*")))

    def test_overlapping_authority_and_existing_destination_are_rejected(self) -> None:
        overlapping = self.root / "overlap.txt"
        overlapping.write_text(f"{self.training_ids[0]}|duplicate\n", encoding="utf-8")
        with self.assertRaisesRegex(SpecError, "split authorities overlap"):
            select_ljspeech_ids(
                self.training_split,
                overlapping,
                training_count=1,
                validation_count=1,
            )

        destination = self.root / "exists"
        destination.mkdir()
        with self.assertRaisesRegex(SpecError, "destination already exists"):
            build_ljspeech_subset(
                self.archive,
                self.training_split,
                self.validation_split,
                destination,
                profile=self.profile,
            )

    def test_payload_roles_keep_audio_lists_and_manifest_distinct(self) -> None:
        subset = build_ljspeech_subset(
            self.archive,
            self.training_split,
            self.validation_split,
            self.root / "subset",
            profile=self.profile,
        )
        entries = ljspeech_subset_payload(subset.root, self.profile)
        by_target = {entry.target: entry.role for entry in entries}

        self.assertEqual(
            by_target["workload/hifi_gan/LJSPEECH_SUBSET_MANIFEST.json"],
            "license",
        )
        self.assertEqual(
            by_target["workload/hifi_gan/LJSpeech-1.1/training.txt"],
            "config",
        )
        self.assertEqual(
            by_target["workload/hifi_gan/LJSpeech-1.1/validation.txt"],
            "config",
        )
        self.assertEqual(
            by_target["workload/hifi_gan/LJSpeech-1.1/wavs"],
            "data",
        )
        json.loads(subset.manifest_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
