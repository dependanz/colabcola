"""Bounded detection of Google Desktop OAuth downloads for one selected project.

This validates file format and local project binding only. It does not establish
Cloud ownership, sign in, import a client, or read saved authorizations. Callers
must establish stability across scans before offering automatic import.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import re
import stat
import subprocess
import sys

MAX_DIRECTORIES = 8
MAX_ENTRIES = 2048
MAX_MATCHING_FILES = 64
MAX_FILE_BYTES = 16384
_DOWNLOAD_NAME = re.compile(r"client_secret_[A-Za-z0-9._-]+(?: ?\([0-9]+\))?\.json", re.IGNORECASE)
_DOWNLOADS_GUID = "374DE290-123F-4565-9164-39C4925E467B"
_POWERSHELL_DOWNLOADS = r"""$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; [Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); Add-Type -Namespace Colabcola -Name KnownFolders -MemberDefinition '[System.Runtime.InteropServices.DllImport("shell32.dll", CharSet=System.Runtime.InteropServices.CharSet.Unicode)] public static extern int SHGetKnownFolderPath(ref System.Guid id, uint flags, System.IntPtr token, out System.IntPtr path); [System.Runtime.InteropServices.DllImport("ole32.dll")] public static extern void CoTaskMemFree(System.IntPtr pointer);'; $id=[Guid]'374DE290-123F-4565-9164-39C4925E467B'; $pointer=[IntPtr]::Zero; try { if ([Colabcola.KnownFolders]::SHGetKnownFolderPath([ref]$id,0x4000,[IntPtr]::Zero,[ref]$pointer) -eq 0) { $path=[Runtime.InteropServices.Marshal]::PtrToStringUni($pointer); if ($path -and $path.Length -le 4096) { ConvertTo-Json -InputObject $path -Compress } } } finally { if ($pointer -ne [IntPtr]::Zero) { [Colabcola.KnownFolders]::CoTaskMemFree($pointer) } }"""


@dataclass(frozen=True)
class Candidate:
    path: Path
    client: dict = field(repr=False, compare=False)
    sha256: str


@dataclass(frozen=True)
class ScanResult:
    candidates: tuple[Candidate, ...]
    incomplete: bool


class _IgnoredFile(Exception):
    pass


class _UncertainFile(Exception):
    pass


def _no_links(path):
    """Check the selected path and ancestors without resolving a link target."""
    for part in (path, *path.parents):
        metadata = part.lstat()
        if stat.S_ISLNK(metadata.st_mode) or (hasattr(part, "is_junction") and part.is_junction()):
            raise _IgnoredFile()


def _identity(metadata):
    # Windows lstat/fstat can expose different ctime semantics for the same
    # unchanged file. Compare stable file identity, size and write timestamp.
    return (metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_size, metadata.st_mtime_ns)


def _read_snapshot(path):
    """Read at most 16 KiB plus an overflow byte; reject changed or special files."""
    descriptor = None
    try:
        _no_links(path)
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise _IgnoredFile()
        if before.st_size > MAX_FILE_BYTES:
            raise _UncertainFile()
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _identity(before) != _identity(opened):
            raise _UncertainFile()
        body = bytearray()
        while len(body) <= MAX_FILE_BYTES:
            chunk = os.read(descriptor, MAX_FILE_BYTES + 1 - len(body))
            if not chunk:
                break
            body.extend(chunk)
        after = os.fstat(descriptor)
        _no_links(path)
        if _identity(opened) != _identity(after) or _identity(after) != _identity(path.lstat()):
            raise _UncertainFile()
        if len(body) > MAX_FILE_BYTES:
            raise _UncertainFile()
        return bytes(body)
    except _IgnoredFile:
        raise
    except (OSError, ValueError):
        raise _UncertainFile() from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _known_folder_downloads():
    """Read the current Windows user's Known Folder; never create or verify it."""
    import ctypes
    from ctypes import wintypes
    import uuid

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]

    shell = ctypes.WinDLL("shell32", use_last_error=True)
    ole = ctypes.WinDLL("ole32", use_last_error=True)
    get_folder = shell.SHGetKnownFolderPath
    get_folder.argtypes = [ctypes.POINTER(GUID), wintypes.DWORD, wintypes.HANDLE, ctypes.POINTER(ctypes.c_wchar_p)]
    get_folder.restype = ctypes.c_long
    ole.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    ole.CoTaskMemFree.restype = None
    identity = GUID.from_buffer_copy(uuid.UUID(_DOWNLOADS_GUID).bytes_le)
    pointer = ctypes.c_wchar_p()
    try:
        # KF_FLAG_DONT_VERIFY avoids touching a potentially redirected location.
        if get_folder(ctypes.byref(identity), 0x4000, None, ctypes.byref(pointer)) != 0:
            return None
        value = pointer.value
        return Path(value) if value and len(value) <= 4096 and not value.startswith("\\\\") else None
    finally:
        if pointer:
            ole.CoTaskMemFree(ctypes.cast(pointer, ctypes.c_void_p))


def _windows_downloads_from_wsl():
    """One static, bounded query; no profile enumeration or guessed Windows user."""
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _POWERSHELL_DOWNLOADS],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=5, check=False, encoding="utf-8-sig", errors="strict",
        )
        if result.returncode or not isinstance(result.stdout, str) or len(result.stdout) > 16384:
            return None
        value = json.loads(result.stdout)
        if not isinstance(value, str) or len(value) > 4096 or any(ord(char) < 32 for char in value):
            return None
        path = PureWindowsPath(value)
        if not path.is_absolute() or not re.fullmatch(r"[A-Za-z]:", path.drive):
            return None
        return Path("/mnt") / path.drive[0].lower() / Path(*path.parts[1:])
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError, TypeError):
        return None


def _linux_downloads(home):
    selected = home / "Downloads"
    config = Path(os.environ.get("XDG_CONFIG_HOME", str(home / ".config")))
    if not config.is_absolute():
        config = home / ".config"
    try:
        body = _read_snapshot(config / "user-dirs.dirs").decode("utf-8-sig")
        for line in body.splitlines():
            match = re.fullmatch(r'\s*XDG_DOWNLOAD_DIR="((?:[^"\\]|\\.)*)"\s*(?:#.*)?', line)
            if not match:
                continue
            value = re.sub(r'\\([\\"$`])', r'\1', match[1])
            if value == "$HOME" or value == "${HOME}":
                return None  # XDG marks a disabled user directory with HOME.
            if value.startswith("$HOME/"):
                value = str(home) + value[5:]
            elif value.startswith("${HOME}/"):
                value = str(home) + value[7:]
            if "$" in value or "`" in value or any(ord(char) < 32 for char in value):
                return selected
            path = Path(value)
            if path.is_absolute() and path != home and path != Path(path.anchor):
                return path
            return None if path == home else selected
    except (_IgnoredFile, _UncertainFile, UnicodeError, OSError, ValueError):
        pass
    return selected


def download_directories():
    """Discover only this user's configured Downloads locations, without scanning.

    Windows uses Known Folders, Linux honors bounded XDG configuration, and WSL
    additionally asks Windows for its current user's actual Downloads folder.
    Unsupported discovery is omitted; callers retain manual path entry.
    """
    paths = []
    try:
        if sys.platform == "win32":
            path = _known_folder_downloads()
            if path is not None:
                paths.append(path)
        elif sys.platform.startswith("linux"):
            path = _linux_downloads(Path.home())
            if path is not None:
                paths.append(path)
            if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP") or "microsoft" in platform.release().lower():
                path = _windows_downloads_from_wsl()
                if path is not None:
                    paths.append(path)
    except (OSError, ValueError, AttributeError, RuntimeError):
        pass
    return list(dict.fromkeys(paths))


def scan_downloads(directories, *, project_id, expected_sha256=None):
    """Find validated Desktop clients for one project, never import or log them.

    Scanning is nonrecursive and bounded across all supplied directories. An
    incomplete result cannot establish uniqueness. Wrong-project or non-Desktop
    downloads are ignored; malformed, oversized, inaccessible or changing
    matching files make the result incomplete. Equal clients are one candidate; credentials
    with the same client ID but a different secret remain distinct candidates.
    """
    from .drive_oauth import DriveOAuthError, validate_client
    from .lifecycle_io import digest, encoded
    from .project_setup import _project_id

    project_id = _project_id(project_id)
    if expected_sha256 is not None and (not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)):
        raise ValueError("Expected client SHA-256 must be a lowercase SHA-256 digest")
    candidates = {}
    visited = set()
    incomplete = False
    entries = matches = 0
    for index, directory in enumerate(directories):
        if index >= MAX_DIRECTORIES:
            incomplete = True
            break
        try:
            directory = Path(os.path.abspath(directory))
            if directory in visited:
                continue
            visited.add(directory)
            _no_links(directory)
            if not stat.S_ISDIR(directory.lstat().st_mode):
                continue
            with os.scandir(directory) as listing:
                for entry in listing:
                    entries += 1
                    if entries > MAX_ENTRIES:
                        return ScanResult(tuple(sorted(candidates.values(), key=lambda item: str(item.path))), True)
                    if not _DOWNLOAD_NAME.fullmatch(entry.name):
                        continue
                    matches += 1
                    if matches > MAX_MATCHING_FILES:
                        return ScanResult(tuple(sorted(candidates.values(), key=lambda item: str(item.path))), True)
                    path = directory / entry.name
                    try:
                        raw = _read_snapshot(path)
                        try:
                            value = json.loads(raw.decode("utf-8-sig"))
                        except (ValueError, UnicodeError):
                            raise _UncertainFile() from None
                        client = validate_client(value)
                        if client["installed"]["project_id"] != project_id:
                            continue
                        sha256 = digest(encoded(client))
                        if expected_sha256 is not None and sha256 != expected_sha256:
                            continue
                        candidate = Candidate(path=path, client=client, sha256=sha256)
                        previous = candidates.get(sha256)
                        if previous is None or str(path) < str(previous.path):
                            candidates[sha256] = candidate
                    except (_IgnoredFile, DriveOAuthError, ValueError, UnicodeError, TypeError):
                        continue
                    except _UncertainFile:
                        incomplete = True
        except FileNotFoundError:
            # A Downloads directory that has never existed contributes no files.
            continue
        except _IgnoredFile:
            continue
        except (OSError, ValueError, TypeError):
            incomplete = True
    return ScanResult(tuple(sorted(candidates.values(), key=lambda item: str(item.path))), incomplete)
