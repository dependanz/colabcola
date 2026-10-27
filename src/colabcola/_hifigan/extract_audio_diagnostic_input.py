"""Extract the one verified HiFi-GAN checkpoint needed by the audio diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile


CHECKPOINT_MEMBER = "outputs/cp_hifigan/g_00005000"
CHECKPOINT_SIZE = 55_825_339
CHECKPOINT_SHA256 = "836627cc247ec688589593d2324aae2cfadc8958eb17304c9426c44de56b46e3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_checkpoint(
    artifacts_zip: Path,
    output_path: Path,
    *,
    member: str = CHECKPOINT_MEMBER,
    expected_size: int = CHECKPOINT_SIZE,
    expected_sha256: str = CHECKPOINT_SHA256,
) -> dict[str, object]:
    artifacts_zip = artifacts_zip.resolve(strict=True)
    if not artifacts_zip.is_file() or artifacts_zip.is_symlink():
        raise ValueError(f"artifacts path is not a regular file: {artifacts_zip}")

    output_path = output_path.resolve(strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if not output_path.is_file() or output_path.is_symlink():
            raise ValueError(f"output path is not a regular file: {output_path}")
        actual_size = output_path.stat().st_size
        actual_sha256 = _sha256(output_path)
        if actual_size != expected_size or actual_sha256 != expected_sha256:
            raise ValueError(
                "existing output does not match the verified checkpoint: "
                f"size={actual_size}, sha256={actual_sha256}"
            )
        return {
            "artifacts_zip": artifacts_zip.as_posix(),
            "checkpoint_path": output_path.as_posix(),
            "checkpoint_size": actual_size,
            "checkpoint_sha256": actual_sha256,
            "member": member,
            "reused": True,
        }

    with zipfile.ZipFile(artifacts_zip) as archive:
        matching = [info for info in archive.infolist() if info.filename == member]
        if len(matching) != 1:
            raise ValueError(f"expected exactly one {member!r} member; found {len(matching)}")
        info = matching[0]
        if info.is_dir() or info.file_size != expected_size:
            raise ValueError(
                f"checkpoint member size mismatch: expected {expected_size}, got {info.file_size}"
            )

        handle, temporary_name = tempfile.mkstemp(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
        )
        os.close(handle)
        temporary_path = Path(temporary_name)
        try:
            digest = hashlib.sha256()
            with archive.open(info, "r") as source, temporary_path.open("wb") as destination:
                while block := source.read(1024 * 1024):
                    destination.write(block)
                    digest.update(block)
                destination.flush()
                os.fsync(destination.fileno())
            actual_size = temporary_path.stat().st_size
            actual_sha256 = digest.hexdigest()
            if actual_size != expected_size or actual_sha256 != expected_sha256:
                raise ValueError(
                    "extracted checkpoint failed verification: "
                    f"size={actual_size}, sha256={actual_sha256}"
                )
            os.replace(temporary_path, output_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    return {
        "artifacts_zip": artifacts_zip.as_posix(),
        "checkpoint_path": output_path.as_posix(),
        "checkpoint_size": expected_size,
        "checkpoint_sha256": expected_sha256,
        "member": member,
        "reused": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts_zip", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    result = extract_checkpoint(args.artifacts_zip, args.output_path)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
