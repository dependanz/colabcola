"""Bounded HiFi-GAN generated-audio diagnostic for a trusted checkpoint.

Derived from ``inference.py`` and ``meldataset.py`` in jik876/hifi-gan commit
4769534d45265d52a904b850da5a622601885777. The bundled upstream LICENSE
applies to the derived portions of this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil

import numpy as np
import torch
from scipy.io.wavfile import read, write

import meldataset as upstream_meldataset
from env import AttrDict
from models import Generator


MAX_WAV_VALUE = 32768.0
UPSTREAM_COMMIT = "4769534d45265d52a904b850da5a622601885777"
EXPECTED_INPUTS = {
    "LJ001-0028.wav": (
        261_478,
        "58831256095edbcc72346a2db9be39efd39662f442f8115f19da4d55876cfa0e",
    ),
    "LJ001-0068.wav": (
        402_278,
        "9ec4f7944fdf753a62792aef59da5f2145c0a69d8877fc07c4c77c7189dcdc0e",
    ),
    "LJ002-0149.wav": (
        349_030,
        "d71743df2a5842ae0130920afe910e53516374dae97e8f24da403efdc4cc9c60",
    ),
}


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
    """Pinned upstream transform adapted to current librosa and PyTorch APIs."""

    cache_key = f"{fmax}_{y.device}"
    if cache_key not in upstream_meldataset.mel_basis:
        mel = upstream_meldataset.librosa_mel_fn(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )
        upstream_meldataset.mel_basis[cache_key] = torch.from_numpy(mel).float().to(
            y.device
        )
        upstream_meldataset.hann_window[str(y.device)] = torch.hann_window(win_size).to(
            y.device
        )

    padded = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    ).squeeze(1)
    spectrum = torch.stft(
        padded,
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


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _audio_stats(samples: np.ndarray) -> dict[str, object]:
    normalized = samples.astype(np.float64) / MAX_WAV_VALUE
    return {
        "frames": int(samples.size),
        "peak_absolute": float(np.max(np.abs(normalized))) if samples.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(normalized)))) if samples.size else 0.0,
    }


def _atomic_json(path: pathlib.Path, value: object) -> None:
    content = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = pathlib.Path(args.checkpoint)
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise RuntimeError("generator checkpoint SHA-256 does not match")
    if not torch.cuda.is_available():
        raise RuntimeError("audio diagnostic requires one CUDA GPU")

    config_value = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
    config = AttrDict(config_value)
    device = torch.device("cuda")
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False

    generator = Generator(config).to(device)
    checkpoint_value = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(checkpoint_value, dict) or "generator" not in checkpoint_value:
        raise RuntimeError("generator checkpoint has an unexpected structure")
    generator.load_state_dict(checkpoint_value["generator"], strict=True)
    generator.eval()
    generator.remove_weight_norm()

    output_root = pathlib.Path(args.output_root)
    reference_root = output_root / "reference"
    generated_root = output_root / "generated"
    reference_root.mkdir(parents=True, exist_ok=False)
    generated_root.mkdir(parents=True, exist_ok=False)

    records: list[dict[str, object]] = []
    if [pathlib.Path(value).name for value in args.input_wav] != list(EXPECTED_INPUTS):
        raise RuntimeError("audio diagnostic inputs do not match the pinned ordered clip set")
    with torch.no_grad():
        for input_value in args.input_wav:
            source = pathlib.Path(input_value)
            expected_size, expected_sha256 = EXPECTED_INPUTS[source.name]
            if source.stat().st_size != expected_size or _sha256(source) != expected_sha256:
                raise RuntimeError(f"validation WAV identity does not match for {source.name}")
            sampling_rate, samples = read(source)
            if sampling_rate != config.sampling_rate:
                raise RuntimeError(f"unexpected sampling rate for {source.name}")
            if samples.dtype != np.int16 or samples.ndim != 1 or samples.size == 0:
                raise RuntimeError(f"unexpected WAV format for {source.name}")

            reference_path = reference_root / source.name
            generated_path = generated_root / f"{source.stem}_generated.wav"
            shutil.copyfile(source, reference_path)

            reference_tensor = (
                torch.from_numpy(samples.astype(np.float32)).to(device) / MAX_WAV_VALUE
            ).unsqueeze(0)
            input_mel = mel_spectrogram(
                reference_tensor,
                config.n_fft,
                config.num_mels,
                config.sampling_rate,
                config.hop_size,
                config.win_size,
                config.fmin,
                config.fmax,
            )
            generated_tensor = generator(input_mel).squeeze(0).squeeze(0)
            if not bool(torch.isfinite(generated_tensor).all().item()):
                raise RuntimeError(f"generated non-finite audio for {source.name}")

            generated_float = generated_tensor.detach().cpu().numpy()
            clipped_samples = int(np.count_nonzero(np.abs(generated_float) > 1.0))
            generated_samples = np.rint(
                np.clip(generated_float, -1.0, 1.0) * 32767.0
            ).astype(np.int16)
            write(generated_path, config.sampling_rate, generated_samples)

            generated_for_mel = torch.from_numpy(generated_float).to(device).unsqueeze(0)
            reference_loss_mel = mel_spectrogram(
                reference_tensor,
                config.n_fft,
                config.num_mels,
                config.sampling_rate,
                config.hop_size,
                config.win_size,
                config.fmin,
                config.fmax_for_loss,
            )
            generated_loss_mel = mel_spectrogram(
                generated_for_mel,
                config.n_fft,
                config.num_mels,
                config.sampling_rate,
                config.hop_size,
                config.win_size,
                config.fmin,
                config.fmax_for_loss,
            )
            aligned_frames = min(reference_loss_mel.shape[-1], generated_loss_mel.shape[-1])
            mel_l1 = torch.nn.functional.l1_loss(
                reference_loss_mel[..., :aligned_frames],
                generated_loss_mel[..., :aligned_frames],
            ).item()

            records.append(
                {
                    "clip_id": source.stem,
                    "generated": {
                        **_audio_stats(generated_samples),
                        "clipped_samples_before_clamp": clipped_samples,
                        "path": generated_path.as_posix(),
                        "sha256": _sha256(generated_path),
                        "size": generated_path.stat().st_size,
                    },
                    "mel_l1_aligned": float(mel_l1),
                    "mel_l1_frames": int(aligned_frames),
                    "reference": {
                        **_audio_stats(samples),
                        "path": reference_path.as_posix(),
                        "sha256": _sha256(reference_path),
                        "size": reference_path.stat().st_size,
                    },
                    "sampling_rate": int(sampling_rate),
                }
            )

    report = {
        "checkpoint": {
            "path": checkpoint.as_posix(),
            "sha256": checkpoint_sha256,
            "size": checkpoint.stat().st_size,
        },
        "clips": records,
        "diagnostic_boundary": (
            "Operational generated-audio evidence only; not a convergence, perceptual-quality, "
            "MOS, or paper-reproduction result."
        ),
        "gpu": {
            "count": torch.cuda.device_count(),
            "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "name": torch.cuda.get_device_name(0),
        },
        "sample_count": len(records),
        "state": "audio_diagnostic_complete",
        "upstream_commit": UPSTREAM_COMMIT,
    }
    _atomic_json(pathlib.Path(args.report), report)
    print(json.dumps(report, allow_nan=False, separators=(",", ":"), sort_keys=True))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--input-wav", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report", required=True)
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
