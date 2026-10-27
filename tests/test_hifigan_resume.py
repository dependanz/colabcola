from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from colabcola._hifigan.resume_state import (
    ResumeStateError,
    deterministic_crop_seed,
    normalize_torch_rng_state_devices,
    ordered_file_identity,
    prune_checkpoint_pairs,
    remaining_epoch_sample_keys,
    validate_continuation_state,
)


class HiFiGanResumeStateTests(unittest.TestCase):
    def valid_state(self) -> dict[str, object]:
        return {
            "colabcola_state_version": 1,
            "training_files_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "batch_size": 4,
            "batches_per_epoch": 3,
            "seed": 0,
            "completed_steps": 5,
            "epoch": 1,
            "next_batch_in_epoch": 2,
            "python_random_state": (3, (), None),
            "torch_rng_state": b"cpu",
            "torch_cuda_rng_state_all": [b"gpu"],
            "scheduler_g": {},
            "scheduler_d": {},
        }

    def validate(self, state: dict[str, object]):
        return validate_continuation_state(
            state,
            expected_training_files_sha256="a" * 64,
            expected_config_sha256="b" * 64,
            expected_batch_size=4,
            expected_batches_per_epoch=3,
            expected_seed=0,
        )

    def test_ordered_file_identity_is_root_independent_and_order_sensitive(self) -> None:
        first = ordered_file_identity(["C:/one/a.wav", "C:/one/b.wav"])
        same_names = ordered_file_identity(["/different/a.wav", "/different/b.wav"])
        reversed_names = ordered_file_identity(["/different/b.wav", "/different/a.wav"])

        self.assertEqual(first, same_names)
        self.assertNotEqual(first, reversed_names)
        with self.assertRaisesRegex(ResumeStateError, "duplicate basenames"):
            ordered_file_identity(["one/a.wav", "two/a.wav"])

    def test_crop_seed_is_stable_and_bound_to_epoch_position_and_sample(self) -> None:
        baseline = deterministic_crop_seed(
            seed=1234, epoch=2, sample_position=7, sample_name="LJ001.wav"
        )
        self.assertEqual(
            baseline,
            deterministic_crop_seed(
                seed=1234, epoch=2, sample_position=7, sample_name="LJ001.wav"
            ),
        )
        self.assertNotEqual(
            baseline,
            deterministic_crop_seed(
                seed=1234, epoch=3, sample_position=7, sample_name="LJ001.wav"
            ),
        )
        self.assertNotEqual(
            baseline,
            deterministic_crop_seed(
                seed=1234, epoch=2, sample_position=8, sample_name="LJ002.wav"
            ),
        )

    def test_split_epoch_consumes_exactly_the_same_remaining_sample_keys(self) -> None:
        files = [f"LJ{i:03d}.wav" for i in range(10)]
        full = list(
            remaining_epoch_sample_keys(
                files, batch_size=2, epoch=4, start_batch=0, seed=1234
            )
        )
        resumed = list(
            remaining_epoch_sample_keys(
                files, batch_size=2, epoch=4, start_batch=3, seed=1234
            )
        )

        self.assertEqual(full[6:], resumed)
        self.assertEqual([position for position, _ in resumed], [6, 7, 8, 9])
        uninterrupted_steps = list(range(1, 6))
        split_steps = list(range(1, 4)) + list(range(4, 6))
        self.assertEqual(split_steps, uninterrupted_steps)
        self.assertEqual(len(split_steps), len(set(split_steps)))

    def test_versioned_cursor_accepts_exact_state(self) -> None:
        cursor = self.validate(self.valid_state())

        self.assertEqual(cursor.completed_steps, 5)
        self.assertEqual(cursor.epoch, 1)
        self.assertEqual(cursor.next_batch_in_epoch, 2)

    def test_versioned_cursor_rejects_every_identity_and_position_mismatch(self) -> None:
        mutations = {
            "version": ("colabcola_state_version", 0),
            "boolean version": ("colabcola_state_version", True),
            "training identity": ("training_files_sha256", "c" * 64),
            "configuration": ("config_sha256", "d" * 64),
            "batch size": ("batch_size", 8),
            "batches per epoch": ("batches_per_epoch", 4),
            "seed": ("seed", 1),
            "boolean seed": ("seed", False),
            "cursor equation": ("completed_steps", 6),
        }
        for label, (key, value) in mutations.items():
            with self.subTest(label=label):
                state = copy.deepcopy(self.valid_state())
                state[key] = value
                with self.assertRaises(ResumeStateError):
                    self.validate(state)

        missing_rng = self.valid_state()
        del missing_rng["torch_cuda_rng_state_all"]
        with self.assertRaisesRegex(ResumeStateError, "torch_cuda_rng_state_all"):
            self.validate(missing_rng)

    def test_rng_states_are_moved_back_to_cpu_after_device_mapped_load(self) -> None:
        class FakeRngTensor:
            def __init__(self, name: str, device: str) -> None:
                self.name = name
                self.device = device

            def cpu(self):
                return FakeRngTensor(self.name, "cpu")

        torch_state, cuda_states = normalize_torch_rng_state_devices(
            FakeRngTensor("torch", "cuda:0"),
            [FakeRngTensor("cuda-0", "cuda:0")],
        )

        self.assertEqual(torch_state.device, "cpu")
        self.assertEqual([state.device for state in cuda_states], ["cpu"])
        with self.assertRaisesRegex(ResumeStateError, "torch_rng_state"):
            normalize_torch_rng_state_devices(b"not-a-tensor", [])
        with self.assertRaisesRegex(ResumeStateError, "must be a sequence"):
            normalize_torch_rng_state_devices(FakeRngTensor("torch", "cpu"), b"bad")

    def test_retention_keeps_newest_complete_pairs_and_unrelated_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for step in (10, 20, 30):
                (root / f"g_{step:08d}").write_bytes(b"g")
                (root / f"do_{step:08d}").write_bytes(b"do")
            (root / "config.json").write_text("{}\n", encoding="utf-8")

            kept = prune_checkpoint_pairs(root, 2)

            self.assertEqual(kept, (20, 30))
            self.assertFalse((root / "g_00000010").exists())
            self.assertFalse((root / "do_00000010").exists())
            self.assertTrue((root / "config.json").exists())

    def test_retention_rejects_an_unpaired_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "g_00000010").write_bytes(b"g")
            with self.assertRaisesRegex(ResumeStateError, "unpaired"):
                prune_checkpoint_pairs(root, 2)


if __name__ == "__main__":
    unittest.main()
