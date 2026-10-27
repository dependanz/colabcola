"""Bounded HiFi-GAN V1 trainer for resumable colabcola CUDA jobs.

Derived from ``train.py`` and ``meldataset.py`` in jik876/hifi-gan commit
4769534d45265d52a904b850da5a622601885777. The bundled upstream LICENSE
applies to the derived portions of this file.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import pathlib
import random
import time
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import meldataset as upstream_meldataset
from env import AttrDict, build_env
from meldataset import MelDataset, get_dataset_filelist
from models import (
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from utils import load_checkpoint, plot_spectrogram, save_checkpoint, scan_checkpoint

from resume_state import (
    STATE_VERSION,
    ContinuationCursor,
    deterministic_crop_seed,
    normalize_torch_rng_state_devices,
    ordered_file_identity,
    prune_checkpoint_pairs,
    remaining_epoch_sample_keys,
    validate_continuation_state,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
torch.backends.cudnn.benchmark = True

UPSTREAM_COMMIT = "4769534d45265d52a904b850da5a622601885777"


def mel_spectrogram(
    y,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
    center=False,
):
    """Pinned upstream transform adapted to modern librosa and PyTorch APIs."""

    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    cache_key = f"{fmax}_{y.device}"
    if cache_key not in upstream_meldataset.mel_basis:
        mel = upstream_meldataset.librosa_mel_fn(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )
        upstream_meldataset.mel_basis[cache_key] = torch.from_numpy(mel).float().to(y.device)
        upstream_meldataset.hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    ).squeeze(1)
    spectrum = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=upstream_meldataset.hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    ).abs()
    spectrum = torch.matmul(upstream_meldataset.mel_basis[cache_key], spectrum)
    return upstream_meldataset.spectral_normalize_torch(spectrum)


# MelDataset resolves this module global at call time, so the original data path
# uses the explicit modern transform without modifying the pinned upstream file.
upstream_meldataset.mel_spectrogram = mel_spectrogram


class DeterministicMelDataset(MelDataset):
    """Make random cropping independent of worker scheduling and prefetch."""

    def __getitem__(self, key):
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or isinstance(key[0], bool)
            or not isinstance(key[0], int)
            or isinstance(key[1], bool)
            or not isinstance(key[1], int)
        ):
            raise RuntimeError("training sampler emitted an invalid deterministic sample key")
        index, crop_seed = key
        previous = random.getstate()
        random.seed(crop_seed)
        try:
            return super().__getitem__(index)
        finally:
            random.setstate(previous)


class EpochSampleSampler:
    """Yield the unconsumed deterministic sample keys for one epoch."""

    def __init__(
        self,
        audio_files,
        *,
        batch_size: int,
        epoch: int,
        start_batch: int,
        seed: int,
    ) -> None:
        self.audio_files = tuple(audio_files)
        self.batch_size = batch_size
        self.epoch = epoch
        self.start = start_batch * batch_size
        self.stop = (len(self.audio_files) // batch_size) * batch_size
        self.seed = seed
        if self.start < 0 or self.start > self.stop:
            raise RuntimeError("resume batch cursor is outside the training epoch")

    def __iter__(self):
        yield from remaining_epoch_sample_keys(
            self.audio_files,
            batch_size=self.batch_size,
            epoch=self.epoch,
            start_batch=self.start // self.batch_size,
            seed=self.seed,
        )

    def __len__(self) -> int:
        return self.stop - self.start


def _output_path(value: str, label: str) -> pathlib.Path:
    portable = pathlib.PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or portable.as_posix() != value
        or not portable.parts
        or portable.parts[0] != "outputs"
        or any(part in {"", ".", ".."} or ":" in part for part in portable.parts)
    ):
        raise ValueError(f"{label} must be a normalized relative path below outputs/")
    root = pathlib.Path.cwd().resolve()
    result = root.joinpath(*portable.parts).resolve()
    result.relative_to(root)
    return result


def _write_metric(stream, value: dict[str, object]) -> None:
    stream.write(json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _atomic_save_checkpoint(path: pathlib.Path, value) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        save_checkpoint(temporary, value)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_pair(
    checkpoint_root: pathlib.Path,
    generator,
    mpd,
    msd,
    optim_g,
    optim_d,
    scheduler_g,
    scheduler_d,
    completed_steps: int,
    epoch: int,
    next_batch_in_epoch: int,
    batches_per_epoch: int,
    training_files_sha256: str,
    config_sha256: str,
    batch_size: int,
    seed: int,
    checkpoint_retention: int,
) -> tuple[int, ...]:
    python_random_state = random.getstate()
    torch_rng_state = torch.get_rng_state()
    torch_cuda_rng_state_all = torch.cuda.get_rng_state_all()
    generator_path = checkpoint_root / f"g_{completed_steps:08d}"
    state_path = checkpoint_root / f"do_{completed_steps:08d}"
    _atomic_save_checkpoint(generator_path, {"generator": generator.state_dict()})
    _atomic_save_checkpoint(
        state_path,
        {
            "mpd": mpd.state_dict(),
            "msd": msd.state_dict(),
            "optim_g": optim_g.state_dict(),
            "optim_d": optim_d.state_dict(),
            # Keep the historical field readable while making completed-step
            # semantics explicit for this overlay and future resumes.
            "steps": completed_steps - 1,
            "completed_steps": completed_steps,
            "epoch": epoch,
            "next_batch_in_epoch": next_batch_in_epoch,
            "batches_per_epoch": batches_per_epoch,
            "training_files_sha256": training_files_sha256,
            "config_sha256": config_sha256,
            "batch_size": batch_size,
            "seed": seed,
            "python_random_state": python_random_state,
            "torch_rng_state": torch_rng_state,
            "torch_cuda_rng_state_all": torch_cuda_rng_state_all,
            "scheduler_g": scheduler_g.state_dict(),
            "scheduler_d": scheduler_d.state_dict(),
            "colabcola_state_version": STATE_VERSION,
            "upstream_commit": UPSTREAM_COMMIT,
        },
    )
    return prune_checkpoint_pairs(checkpoint_root, checkpoint_retention)


def _load_pair(checkpoint_root: pathlib.Path, device, generator, mpd, msd):
    generator_path = scan_checkpoint(checkpoint_root, "g_")
    state_path = scan_checkpoint(checkpoint_root, "do_")
    if (generator_path is None) != (state_path is None):
        raise RuntimeError("checkpoint directory contains an unpaired latest checkpoint")
    if generator_path is None:
        return None, 0

    state_dict_g = load_checkpoint(generator_path, device)
    state_dict_do = load_checkpoint(state_path, device)
    generator.load_state_dict(state_dict_g["generator"])
    mpd.load_state_dict(state_dict_do["mpd"])
    msd.load_state_dict(state_dict_do["msd"])
    completed_steps = int(state_dict_do.get("completed_steps", state_dict_do["steps"] + 1))
    expected_g = f"g_{completed_steps:08d}"
    expected_do = f"do_{completed_steps:08d}"
    if (
        pathlib.Path(generator_path).name != expected_g
        or pathlib.Path(state_path).name != expected_do
    ):
        raise RuntimeError(
            "checkpoint filenames do not agree with the recorded completed step: "
            f"expected {expected_g} and {expected_do}"
        )
    if completed_steps < 0:
        raise RuntimeError("checkpoint completed_steps must be nonnegative")
    return state_dict_do, completed_steps


def train(a, h, checkpoint_root: pathlib.Path, metrics_path: pathlib.Path) -> None:
    torch.cuda.manual_seed(h.seed)
    device = torch.device("cuda:0")

    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    print("checkpoints directory : ", checkpoint_root)

    state_dict_do, completed_steps = _load_pair(checkpoint_root, device, generator, mpd, msd)
    starting_step = completed_steps
    if completed_steps >= a.maximum_steps:
        raise RuntimeError(
            f"maximum_steps={a.maximum_steps} must exceed resumed starting step {completed_steps}"
        )

    optim_g = torch.optim.AdamW(
        generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(msd.parameters(), mpd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2],
    )
    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do["optim_g"])
        optim_d.load_state_dict(state_dict_do["optim_d"])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay)

    training_filelist, validation_filelist = get_dataset_filelist(a)
    trainset = DeterministicMelDataset(
        training_filelist,
        h.segment_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        n_cache_reuse=0,
        shuffle=True,
        fmax_loss=h.fmax_for_loss,
        device=device,
        fine_tuning=a.fine_tuning,
        base_mels_path=a.input_mels_dir,
    )
    batches_per_epoch = len(trainset) // h.batch_size
    if batches_per_epoch == 0:
        raise RuntimeError("training subset does not produce a complete batch")
    training_files_sha256 = ordered_file_identity(trainset.audio_files)
    config_sha256 = hashlib.sha256(pathlib.Path(a.config).read_bytes()).hexdigest()

    validset = MelDataset(
        validation_filelist,
        h.segment_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        False,
        False,
        n_cache_reuse=0,
        fmax_loss=h.fmax_for_loss,
        device=device,
        fine_tuning=a.fine_tuning,
        base_mels_path=a.input_mels_dir,
    )
    validation_loader = DataLoader(
        validset,
        num_workers=1,
        shuffle=False,
        sampler=None,
        batch_size=1,
        pin_memory=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(h.seed),
    )

    if state_dict_do is None:
        cursor = ContinuationCursor(completed_steps=0, epoch=0, next_batch_in_epoch=0)
        random.seed(h.seed)
    else:
        if state_dict_do.get("upstream_commit") != UPSTREAM_COMMIT:
            raise RuntimeError("checkpoint upstream commit does not match this trainer")
        cursor = validate_continuation_state(
            state_dict_do,
            expected_training_files_sha256=training_files_sha256,
            expected_config_sha256=config_sha256,
            expected_batch_size=h.batch_size,
            expected_batches_per_epoch=batches_per_epoch,
            expected_seed=h.seed,
        )
        if cursor.completed_steps != completed_steps:
            raise RuntimeError("checkpoint cursor disagrees with the loaded checkpoint pair")
        scheduler_g.load_state_dict(state_dict_do["scheduler_g"])
        scheduler_d.load_state_dict(state_dict_do["scheduler_d"])
        random.setstate(state_dict_do["python_random_state"])
        torch_rng_state, torch_cuda_rng_state_all = normalize_torch_rng_state_devices(
            state_dict_do["torch_rng_state"],
            state_dict_do["torch_cuda_rng_state_all"],
        )
        torch.set_rng_state(torch_rng_state)
        torch.cuda.set_rng_state_all(torch_cuda_rng_state_all)

    writer = SummaryWriter(checkpoint_root / "logs")

    generator.train()
    mpd.train()
    msd.train()
    last_checkpointed_step = starting_step
    retained_checkpoint_steps: tuple[int, ...] = ()
    with metrics_path.open("a", encoding="utf-8", newline="\n") as metrics:
        _write_metric(
            metrics,
            {
                "event": "run_start",
                "maximum_steps": a.maximum_steps,
                "starting_batch_in_epoch": cursor.next_batch_in_epoch,
                "starting_epoch": cursor.epoch,
                "starting_step": starting_step,
                "state_version": STATE_VERSION,
                "training_files_sha256": training_files_sha256,
                "upstream_commit": UPSTREAM_COMMIT,
            },
        )
        for epoch in range(cursor.epoch, a.training_epochs):
            epoch_started = time.monotonic()
            print(f"Epoch: {epoch + 1}")

            start_batch = cursor.next_batch_in_epoch if epoch == cursor.epoch else 0
            sampler = EpochSampleSampler(
                trainset.audio_files,
                batch_size=h.batch_size,
                epoch=epoch,
                start_batch=start_batch,
                seed=h.seed,
            )
            loader_generator = torch.Generator()
            loader_generator.manual_seed(
                deterministic_crop_seed(
                    seed=h.seed,
                    epoch=epoch,
                    sample_position=start_batch * h.batch_size,
                    sample_name="__colabcola_worker_seed__",
                )
            )
            train_loader = DataLoader(
                trainset,
                num_workers=h.num_workers,
                shuffle=False,
                sampler=sampler,
                batch_size=h.batch_size,
                pin_memory=True,
                drop_last=True,
                generator=loader_generator,
            )

            for batch_index, batch in enumerate(train_loader, start=start_batch):
                batch_started = time.monotonic()
                x, y, _, y_mel = batch
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                y_mel = y_mel.to(device, non_blocking=True)
                y = y.unsqueeze(1)

                y_g_hat = generator(x)
                y_g_hat_mel = mel_spectrogram(
                    y_g_hat.squeeze(1),
                    h.n_fft,
                    h.num_mels,
                    h.sampling_rate,
                    h.hop_size,
                    h.win_size,
                    h.fmin,
                    h.fmax_for_loss,
                )

                optim_d.zero_grad()
                y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
                loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
                y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
                loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
                loss_disc_all = loss_disc_s + loss_disc_f
                loss_disc_all.backward()
                optim_d.step()

                optim_g.zero_grad()
                loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, _ = generator_loss(y_df_hat_g)
                loss_gen_s, _ = generator_loss(y_ds_hat_g)
                loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
                loss_gen_all.backward()
                optim_g.step()

                completed_steps += 1
                generator_value = float(loss_gen_all.detach().item())
                discriminator_value = float(loss_disc_all.detach().item())
                mel_error = float(F.l1_loss(y_mel, y_g_hat_mel).detach().item())
                seconds_per_batch = float(time.monotonic() - batch_started)
                finite = all(
                    math.isfinite(value)
                    for value in (
                        generator_value,
                        discriminator_value,
                        mel_error,
                        seconds_per_batch,
                    )
                )
                _write_metric(
                    metrics,
                    {
                        "discriminator_loss": discriminator_value,
                        "event": "step",
                        "finite": finite,
                        "generator_loss": generator_value,
                        "mel_spectrogram_error": mel_error,
                        "seconds_per_batch": seconds_per_batch,
                        "step": completed_steps,
                    },
                )
                if not finite:
                    raise RuntimeError(
                        f"non-finite training metric at completed step {completed_steps}"
                    )

                if completed_steps % a.stdout_interval == 0:
                    print(
                        "Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : "
                        "{:4.3f}, s/b : {:4.3f}".format(
                            completed_steps,
                            generator_value,
                            mel_error,
                            seconds_per_batch,
                        )
                    )

                if completed_steps % a.summary_interval == 0:
                    writer.add_scalar("training/gen_loss_total", generator_value, completed_steps)
                    writer.add_scalar(
                        "training/disc_loss_total", discriminator_value, completed_steps
                    )
                    writer.add_scalar("training/mel_spec_error", mel_error, completed_steps)

                if completed_steps % a.validation_interval == 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    validation_error = 0.0
                    with torch.no_grad():
                        for index, validation_batch in enumerate(validation_loader):
                            x_val, y_val, _, y_mel_val = validation_batch
                            y_g_hat_val = generator(x_val.to(device))
                            y_mel_val = y_mel_val.to(device, non_blocking=True)
                            y_g_hat_mel_val = mel_spectrogram(
                                y_g_hat_val.squeeze(1),
                                h.n_fft,
                                h.num_mels,
                                h.sampling_rate,
                                h.hop_size,
                                h.win_size,
                                h.fmin,
                                h.fmax_for_loss,
                            )
                            validation_error += F.l1_loss(
                                y_mel_val, y_g_hat_mel_val
                            ).item()
                            if index <= 4:
                                writer.add_audio(
                                    f"gt/y_{index}", y_val[0], completed_steps, h.sampling_rate
                                )
                                writer.add_figure(
                                    f"gt/y_spec_{index}",
                                    plot_spectrogram(x_val[0]),
                                    completed_steps,
                                )
                                writer.add_audio(
                                    f"generated/y_hat_{index}",
                                    y_g_hat_val[0],
                                    completed_steps,
                                    h.sampling_rate,
                                )
                        validation_error /= len(validation_loader)
                        writer.add_scalar(
                            "validation/mel_spec_error", validation_error, completed_steps
                        )
                    generator.train()

                if completed_steps % a.checkpoint_interval == 0:
                    retained_checkpoint_steps = _checkpoint_pair(
                        checkpoint_root,
                        generator,
                        mpd,
                        msd,
                        optim_g,
                        optim_d,
                        scheduler_g,
                        scheduler_d,
                        completed_steps,
                        epoch,
                        batch_index + 1,
                        batches_per_epoch,
                        training_files_sha256,
                        config_sha256,
                        h.batch_size,
                        h.seed,
                        a.checkpoint_retention,
                    )
                    last_checkpointed_step = completed_steps

                if completed_steps >= a.maximum_steps:
                    if last_checkpointed_step != completed_steps:
                        retained_checkpoint_steps = _checkpoint_pair(
                            checkpoint_root,
                            generator,
                            mpd,
                            msd,
                            optim_g,
                            optim_d,
                            scheduler_g,
                            scheduler_d,
                            completed_steps,
                            epoch,
                            batch_index + 1,
                            batches_per_epoch,
                            training_files_sha256,
                            config_sha256,
                            h.batch_size,
                            h.seed,
                            a.checkpoint_retention,
                        )
                    _write_metric(
                        metrics,
                        {
                            "ending_step": completed_steps,
                            "ending_batch_in_epoch": batch_index + 1,
                            "ending_epoch": epoch,
                            "event": "run_complete",
                            "retained_checkpoint_steps": list(retained_checkpoint_steps),
                            "starting_step": starting_step,
                        },
                    )
                    writer.close()
                    return

            scheduler_g.step()
            scheduler_d.step()
            elapsed = int(time.monotonic() - epoch_started)
            print(f"Time taken for epoch {epoch + 1} is {elapsed} sec\n")

    writer.close()
    raise RuntimeError(
        f"training_epochs exhausted at step {completed_steps} before "
        f"maximum_steps={a.maximum_steps}"
    )


def main() -> None:
    print("Initializing bounded colabcola HiFi-GAN training process..")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_wavs_dir", default="LJSpeech-1.1/wavs")
    parser.add_argument("--input_mels_dir", default="ft_dataset")
    parser.add_argument("--input_training_file", default="LJSpeech-1.1/training.txt")
    parser.add_argument("--input_validation_file", default="LJSpeech-1.1/validation.txt")
    parser.add_argument("--checkpoint_path", default="outputs/cp_hifigan")
    parser.add_argument("--metrics_path", default="outputs/metrics.ndjson")
    parser.add_argument("--config", required=True)
    parser.add_argument("--training_epochs", default=3100, type=int)
    parser.add_argument("--maximum_steps", required=True, type=int)
    parser.add_argument("--stdout_interval", default=5, type=int)
    parser.add_argument("--checkpoint_interval", default=50, type=int)
    parser.add_argument("--checkpoint_retention", default=2, type=int)
    parser.add_argument("--summary_interval", default=10, type=int)
    parser.add_argument("--validation_interval", default=100, type=int)
    parser.add_argument("--fine_tuning", action="store_true")
    a = parser.parse_args()

    if a.maximum_steps <= 0:
        raise ValueError("maximum_steps must be greater than zero")
    for name in (
        "training_epochs",
        "stdout_interval",
        "checkpoint_interval",
        "checkpoint_retention",
        "summary_interval",
        "validation_interval",
    ):
        if getattr(a, name) <= 0:
            raise ValueError(f"{name} must be greater than zero")

    checkpoint_root = _output_path(a.checkpoint_path, "checkpoint_path")
    metrics_path = _output_path(a.metrics_path, "metrics_path")
    if not torch.cuda.is_available():
        raise RuntimeError("the HiFi-GAN profile requires one compatible CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "the HiFi-GAN profile requires exactly one visible compatible CUDA GPU; "
            f"observed {torch.cuda.device_count()}"
        )

    with open(a.config, encoding="utf-8") as stream:
        h = AttrDict(json.load(stream))
    build_env(a.config, "config.json", checkpoint_root)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed(h.seed)
    h.num_gpus = 1
    print("Batch size per GPU :", h.batch_size)
    train(a, h, checkpoint_root, metrics_path)


if __name__ == "__main__":
    main()
