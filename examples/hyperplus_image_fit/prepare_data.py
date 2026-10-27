"""Prepare a fixed CIFAR-10 binary subset without deserializing pickle files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import urllib.request

import numpy as np

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
MD5 = "c32a1d4ab5d03f1284b67883e8d87530"


def digest(path, algorithm="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def prepare(root, train_count=128, heldout_count=32):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if not (1 <= train_count <= 10000 and 1 <= heldout_count <= 10000):
        raise ValueError("subset counts must be in [1,10000]")
    archive = root / "cifar-10-binary.tar.gz"
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(URL, headers={"Range": f"bytes={offset}-"} if offset else {})
        with urllib.request.urlopen(request, timeout=60) as response:
            resumed = offset > 0 and response.status == 206
            if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("server returned an invalid resume range")
            with partial.open("ab" if resumed else "wb") as out:
                total = offset if resumed else 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > 200_000_000:
                        raise ValueError("archive exceeds expected size bound")
                    out.write(chunk)
        if digest(partial, "md5") != MD5:
            raise ValueError("official CIFAR binary MD5 mismatch")
        os.replace(partial, archive)
    if digest(archive, "md5") != MD5:
        raise ValueError("official CIFAR binary MD5 mismatch")
    manifest = {"schema": 1, "source_url": URL, "archive_sha256": digest(archive),
                "official_md5": MD5, "citation": "Alex Krizhevsky, Learning Multiple Layers of Features from Tiny Images, 2009",
                "selection": "first N records of data_batch_1.bin and test_batch.bin; labels unused",
                "files": {}}
    with tarfile.open(archive, "r:gz") as tar:
        for split, member, count in (("train", "data_batch_1.bin", train_count),
                                     ("heldout", "test_batch.bin", heldout_count)):
            entry = tar.getmember("cifar-10-batches-bin/" + member)
            if not entry.isfile() or entry.size != 30_730_000:
                raise ValueError("unexpected CIFAR binary member")
            with tar.extractfile(entry) as stream:
                raw = stream.read(count * 3073)
            if len(raw) != count * 3073:
                raise ValueError("short CIFAR binary member")
            images = np.frombuffer(raw, dtype=np.uint8).reshape(count, 3073)[:, 1:]
            path = root / (split + ".npy")
            np.save(path, images.reshape(count, 3, 32, 32), allow_pickle=False)
            manifest["files"][path.name] = {"sha256": digest(path), "images": count,
                                          "source_member": member, "start_record": 0}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--train-count", type=int, default=128)
    p.add_argument("--heldout-count", type=int, default=32)
    args = p.parse_args()
    prepare(args.output, args.train_count, args.heldout_count)
