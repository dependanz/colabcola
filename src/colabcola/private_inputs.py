"""Host-side binding for large assets; no private bytes enter the notebook."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat

from .errors import SpecError
from .lifecycle_io import PRIVATE_ASSET_BYTES, CHUNK_BYTES

PRIVATE_INPUT_COUNT = 64
PRIVATE_INPUT_TOTAL_BYTES = 8 * 1024**3
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_WINDOWS = os.name == "nt"


def private_target(value, where):
    from .spec import _portable_path
    target = _portable_path(value, where)
    parts = target.split("/")
    reserved = {"con", "prn", "aux", "nul"} | {f"{p}{i}" for p in ("com", "lpt") for i in range(1, 10)}
    if (parts[0].casefold() in {"_colabcola", "outputs"}
            or any(part.rstrip(" .") != part or part.split(".")[0].casefold() in reserved
                   or any(ord(c) < 32 or c in '<>"|?*' for c in part) for part in parts)):
        raise SpecError(f"{where} is reserved or is not a portable asset target")
    return target


def local_binding(value, base):
    """Normalize a local path without following any link or reading the asset."""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise SpecError("private input requires a local source path")
    path = Path(value).expanduser()
    path = Path(os.path.abspath(path if path.is_absolute() else Path(base) / path))
    from .packager import _private_host_path
    if _private_host_path(path):
        raise SpecError("Colabcola account configuration cannot be a private input")
    return path


def inspect_source(path, *, missing_ok=False):
    """Check every existing path component, including Windows reparse points."""
    path = Path(path)
    missing = False
    for part in reversed((path, *path.parents)):
        try:
            metadata = part.lstat()
        except FileNotFoundError:
            if missing_ok:
                missing = True
                continue
            raise SpecError("private input source is missing") from None
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & _REPARSE:
            raise SpecError("private input source contains a link or reparse point")
        if part != path and not stat.S_ISDIR(metadata.st_mode):
            raise SpecError("private input source ancestor is not a directory")
        if part == path:
            if not stat.S_ISREG(metadata.st_mode):
                raise SpecError("private input source must be a regular file")
            return metadata
    if missing and missing_ok:
        return None
    raise SpecError("private input source is missing")


def _source_identity(metadata, *, cross_api=False):
    # CPython's Windows path-stat compatibility shim returns creation time
    # in st_ctime, while fstat may return the NTFS metadata change time
    # (python/cpython#157671). Compare creation time across those APIs;
    # keep the change-time field in all same-API comparisons.
    timestamp = (metadata.st_birthtime_ns if _WINDOWS and cross_api
                 else metadata.st_ctime_ns)
    return (metadata.st_dev, metadata.st_ino, metadata.st_size,
            metadata.st_mtime_ns, timestamp)


def freeze_source(path):
    """Stream a bounded source, rejecting identity/metadata changes while hashing."""
    before = inspect_source(path)
    if not 0 < before.st_size <= PRIVATE_ASSET_BYTES:
        raise SpecError("private input size must be positive and at most 4 GiB")
    sha, total = hashlib.sha256(), 0
    with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0)
                           | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)), "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (not stat.S_ISREG(opened.st_mode)
                or _source_identity(opened, cross_api=True) != _source_identity(before, cross_api=True)):
            raise SpecError("private input source changed before hashing")
        while total < before.st_size:
            body = stream.read(min(CHUNK_BYTES, before.st_size - total))
            if not body:
                raise SpecError("private input source truncated during hashing")
            sha.update(body)
            total += len(body)
        if (stream.read(1) or _source_identity(os.fstat(stream.fileno())) != _source_identity(opened)
                or _source_identity(inspect_source(path)) != _source_identity(before)):
            raise SpecError("private input source changed while hashing")
    return {"bytes": total, "sha256": sha.hexdigest()}


def source_filter(paths):
    """Exclude explicitly declared source files and hardlinked aliases from payloads."""
    paths = {Path(path) for path in paths}
    if not paths:
        return lambda path: False
    inodes = set()
    for path in paths:
        metadata = inspect_source(path, missing_ok=True)
        if metadata is not None and metadata.st_ino:
            inodes.add((metadata.st_dev, metadata.st_ino))
    def contains(path):
        path = Path(path)
        if path in paths:
            return True
        try:
            metadata = path.stat()
        except FileNotFoundError:
            return False
        return bool(metadata.st_ino and (metadata.st_dev, metadata.st_ino) in inodes)
    return contains
