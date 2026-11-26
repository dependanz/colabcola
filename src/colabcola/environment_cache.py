"""Opt-in, same-prefix environment cache. Embedded in cache-enabled notebooks.

The versioned archive is a gzip stream of bounded JSON headers and file bytes.
No tar/ZIP extractor, symlink traversal, hardlink restoration or pickle is used.
The storage interface only sees an identity key and two verified files; it is
provided by the managed control process, never by the workload environment.
"""
import gzip
import hashlib
import json
import math
import os
import pathlib
import platform
import posixpath
import re
import shutil
import stat
import struct
import subprocess
import tempfile
import time


_CC_CACHE_FORMAT = 1
_CC_CACHE_MAGIC = b"COLABCOLA-ENV-CACHE\x00\x01"
_CC_CACHE_ARCHIVE_BYTES = 4 * 1024**3
_CC_CACHE_EXPANDED_BYTES = 12 * 1024**3
_CC_CACHE_FILES = 200_000
_CC_CACHE_HEADER_BYTES = 16 * 1024
_CC_CACHE_METADATA_BYTES = 8 * 1024**2
_CC_CACHE_FREE_RESERVE = 128 * 1024**2
_CC_CACHE_BLOCK = 1024**2
_CC_CACHE_TRANSFER_SECONDS = 300
# Linux /proc/cpuinfo names for the baseline and v2 x86-64 psABI features.
# The x86-64 Linux host check also establishes the ordinary OS support for
# baseline SSE and syscall. Do not infer a target from a wheel's platform tag.
_CC_CACHE_X86_V2_FLAGS = frozenset(("cmov", "cx8", "fpu", "fxsr", "mmx", "syscall",
                                   "sse", "sse2", "cx16", "lahf_lm", "popcnt",
                                   "pni", "sse4_1", "sse4_2", "ssse3"))


class EnvironmentCacheError(RuntimeError):
    """A cache optimization failed; an ordinary cold build can still proceed."""


def _cc_cache_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _cc_cache_hash(value):
    return hashlib.sha256(_cc_cache_json(value)).hexdigest()


def _cc_cache_digest_valid(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _cc_cache_remaining(deadline):
    remaining = float(deadline) - time.time()
    if remaining <= 0:
        raise TimeoutError("environment cache operation exceeded its deadline")
    return remaining


def _cc_cache_file_digest(path, deadline, maximum):
    path = pathlib.Path(path)
    if path.is_symlink() or not path.is_file():
        raise EnvironmentCacheError("cache input is not a regular file")
    size = path.stat().st_size
    if not 0 <= size <= maximum:
        raise EnvironmentCacheError("cache input exceeds its size bound")
    total = 0
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            _cc_cache_remaining(deadline)
            data = stream.read(_CC_CACHE_BLOCK)
            if not data:
                break
            total += len(data)
            if total > size:
                raise EnvironmentCacheError("cache input changed during hashing")
            sha.update(data)
    if total != size or path.stat().st_size != size:
        raise EnvironmentCacheError("cache input changed during hashing")
    return {"bytes": size, "sha256": sha.hexdigest()}


def _cc_cache_gpu_identity(observed, deadline):
    """Query host GPU facts without inheriting arbitrary workload or auth variables."""
    executable = pathlib.Path("/usr/bin/nvidia-smi")
    if not executable.is_file():
        raise EnvironmentCacheError("GPU compatibility query executable unavailable")
    # Colab may expose driver libraries outside the default dynamic loader
    # search path. Use the same fixed host directories as named CUDA steps,
    # never a user-supplied LD_LIBRARY_PATH, toolkit prefix, or linker stubs.
    env = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
    libraries = [str(path) for path in map(pathlib.Path,
        ("/usr/lib64-nvidia", "/usr/local/nvidia/lib", "/usr/local/nvidia/lib64")) if path.is_dir()]
    if libraries:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(libraries)
    try:
        output = subprocess.run([str(executable), "--query-gpu=name,compute_cap,driver_version",
                                 "--format=csv,noheader,nounits"], check=True,
                                stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                env=env, timeout=min(10, _cc_cache_remaining(deadline))).stdout
    except subprocess.CalledProcessError as exc:
        # Keep useful fixed diagnostic categories, never raw subprocess output:
        # an external tool could include private paths, URLs or credentials.
        stderr = (exc.stderr or "").lower()[:16384]
        if "error while loading shared libraries" in stderr or "couldn't find libnvidia-ml" in stderr:
            category = "driver_library_unavailable"
        elif "failed to initialize nvml" in stderr:
            category = "nvml_initialization_failed"
        elif "not a valid field" in stderr or "invalid query" in stderr:
            category = "query_field_unsupported"
        else:
            category = "nonzero_exit"
        raise EnvironmentCacheError(f"GPU compatibility query failed ({category}; exit {exc.returncode})") from None
    except subprocess.TimeoutExpired:
        raise EnvironmentCacheError("GPU compatibility query timed out") from None
    except OSError:
        raise EnvironmentCacheError("GPU compatibility query could not start") from None
    if len(output) > 16384:
        raise EnvironmentCacheError("GPU compatibility identity exceeds its bound")
    gpus = []
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if (len(fields) != 3 or not re.fullmatch(r"[A-Za-z0-9_. +()-]{1,128}", fields[0])
                or not re.fullmatch(r"\d+\.\d+", fields[1])
                or not re.fullmatch(r"\d+(?:\.\d+)+", fields[2])):
            raise EnvironmentCacheError("GPU compatibility query returned invalid identity")
        gpus.append({"name": fields[0], "compute_capability": fields[1], "driver": fields[2]})
    # Bind the actual queried GPU set to the earlier resource check. Allow
    # vendor prefixes in names, not a different model or number of devices.
    def normalized_name(name):
        return re.sub(r"^(?:(?:NVIDIA|TESLA) +)+", "", name.upper().strip())
    if (not gpus or len(gpus) != len(observed)
            or any(not isinstance(item, dict) or not isinstance(item.get("name"), str) for item in observed)
            or sorted(normalized_name(item["name"]) for item in observed)
                != sorted(normalized_name(item["name"]) for item in gpus)):
        raise EnvironmentCacheError("GPU compatibility identity disagrees with preflight")
    return gpus


def _cc_cache_runtime_fingerprint(resources, hardware, deadline):
    """Fail closed unless the runtime provides an actual image release identity.

    `latest` is a request, not a fingerprint. Host package inventory, ABI and
    conservative CPU/GPU compatibility facts further distinguish image changes.
    Do not include host names, allocation IDs, GPU serials, or credentials.
    """
    release = os.environ.get("COLAB_RELEASE_TAG", "")
    if (platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64")
            or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", release)):
        raise EnvironmentCacheError("runtime image identity unavailable")
    libc = platform.libc_ver()
    if not all(libc):
        raise EnvironmentCacheError("host libc identity unavailable")
    host_files = {}
    for name in ("/etc/os-release", "/var/lib/dpkg/status"):
        path = pathlib.Path(name).resolve(strict=True)
        host_files[name] = _cc_cache_file_digest(path, deadline, 32 * 1024**2)["sha256"]
    # Compile-time -march=native is not necessarily encoded in the wheel tag.
    # Use the intersection of every visible CPU's flags; be conservative.
    with open("/proc/cpuinfo", encoding="ascii", errors="replace") as stream:
        cpuinfo = stream.read(4 * 1024**2 + 1)
    if len(cpuinfo) > 4 * 1024**2:
        raise EnvironmentCacheError("host CPU identity exceeds its bound")
    flags = [set(line.split(":", 1)[1].split()) for line in cpuinfo.splitlines()
             if re.match(r"^flags\s*:", line)]
    if not flags or not all(flags):
        raise EnvironmentCacheError("host CPU flags unavailable")
    cpu_flags = sorted(set.intersection(*flags))
    if not cpu_flags:
        raise EnvironmentCacheError("host CPU feature identity unavailable")
    observed = hardware.get("gpus", [])
    if not isinstance(observed, list):
        raise EnvironmentCacheError("invalid runtime hardware identity")
    gpus = (_cc_cache_gpu_identity(observed, deadline)
            if resources.get("accelerator") == "gpu" or observed else [])
    return {"release": release, "system": "Linux", "architecture": "x86_64",
            "kernel": platform.release(), "libc": list(libc), "host_files": host_files,
            "cpu_flags": cpu_flags, "gpus": sorted(gpus, key=_cc_cache_json)}


def _cc_cache_cpu_identity(definition, runtime):
    """Exact host matching by default; a portable target is a user assertion.

    Only the CPU portion is projected. The original image/ABI/GPU constraints
    still apply and the actual flags must support the declared ISA on every
    visible CPU. This is not proof of the installed binaries' ISA requirements.
    """
    policy = definition.get("cache_cpu", "exact")
    if policy == "exact":
        return runtime
    if policy != "x86-64-v2" or definition.get("cache") is not True:
        raise EnvironmentCacheError("unsupported cache CPU policy")
    flags = runtime.get("cpu_flags")
    if (runtime.get("system") != "Linux" or runtime.get("architecture") != "x86_64"
            or not isinstance(flags, list) or not all(isinstance(f, str) for f in flags)
            or not _CC_CACHE_X86_V2_FLAGS.issubset(flags)):
        raise EnvironmentCacheError("host does not establish x86-64-v2 CPU compatibility")
    projected = dict(runtime)
    del projected["cpu_flags"]
    projected["cpu_target"] = policy
    return projected


def _cc_cache_identity(name, definition, manifest, runtime, prefix_root, implementation):
    """Bind the declaration and every known build input, not a random run ID."""
    if not _cc_cache_digest_valid(implementation):
        raise EnvironmentCacheError("cache implementation identity unavailable")
    pip = definition["pip"]
    files = {entry["path"]: entry for entry in manifest["files"]}
    inputs = []
    for path in sorted(set(pip.get("wheels", []) + pip.get("requirements", []))):
        entry = files.get(path)
        if (not isinstance(entry, dict) or type(entry.get("size")) is not int
                or entry["size"] < 0 or not _cc_cache_digest_valid(entry.get("sha256"))
                or (path in pip.get("wheels", []) and entry.get("role") != "wheel")):
            raise EnvironmentCacheError("dependency input identity unavailable")
        inputs.append({"path": path, "size": entry["size"], "sha256": entry["sha256"]})
    identity = {"format": _CC_CACHE_FORMAT, "implementation": implementation,
                "name": name, "definition": definition, "manager": manifest["environment_manager"],
                "runtime": _cc_cache_cpu_identity(definition, runtime),
                "prefix_root": str(prefix_root), "inputs": inputs}
    # Requirements may reference packaged local sources; local builds may read
    # arbitrary project data. Bind the whole job conservatively in these cases.
    if (pip.get("requirements") or any(p.startswith("./") for p in pip.get("packages", []))
            or "PYTHONPATH" in definition.get("env", {})):
        if not _cc_cache_digest_valid(manifest.get("job_id")):
            raise EnvironmentCacheError("local build job identity unavailable")
        identity["local_build_job_id"] = manifest["job_id"]
    return identity


def _cc_cache_relative(name):
    if (not isinstance(name, str) or not name or len(name) > 4096
            or name.startswith("/") or any(c in name for c in "\\:\x00\r\n")
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise EnvironmentCacheError("unsafe cache archive path")
    return name


def _cc_cache_link(name, target):
    if (not isinstance(target, str) or not target or len(target) > 4096
            or target.startswith("/") or any(c in target for c in "\\:\x00\r\n")):
        raise EnvironmentCacheError("unsafe cache archive link")
    combined = posixpath.normpath(posixpath.join(posixpath.dirname(name), target))
    if combined == ".." or combined.startswith("../") or combined.startswith("/"):
        raise EnvironmentCacheError("cache archive link escapes the prefix")
    return target


def _cc_cache_check_links(root, links):
    for name in links:
        try:
            path = root / name
            # Python 3.13's non-strict resolve may return an unresolved cyclic
            # link. Strict resolution detects cycles; a dangling internal link
            # is still safe if its non-strict resolved destination is contained.
            try:
                resolved = path.resolve(strict=True)
            except FileNotFoundError:
                resolved = path.resolve(strict=False)
            resolved.relative_to(root)
        except (ValueError, RuntimeError, OSError) as exc:
            raise EnvironmentCacheError("cache archive contains an escaping or cyclic link") from exc


def _cc_cache_check_root(path):
    """Never operate through a link or a shared-writable cache prefix root."""
    path = pathlib.Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise EnvironmentCacheError("cache root contains a symlink")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.stat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
            or info.st_mode & 0o022):
        raise EnvironmentCacheError("cache root is not owned and private to this runtime")
    return path


class _CCCacheBoundedIO:
    """Deadline and byte bound on every compressed read/write."""
    def __init__(self, stream, deadline, maximum):
        self.stream, self.deadline, self.maximum = stream, deadline, maximum
        self.count = 0

    def write(self, data):
        _cc_cache_remaining(self.deadline)
        if self.count + len(data) > self.maximum:
            raise EnvironmentCacheError("compressed cache archive exceeds its bound")
        n = self.stream.write(data)
        self.count += n
        return n

    def read(self, size=-1):
        _cc_cache_remaining(self.deadline)
        if size < 0 or size > _CC_CACHE_BLOCK:
            raise EnvironmentCacheError("unbounded cache archive read")
        data = self.stream.read(min(size, self.maximum - self.count + 1))
        self.count += len(data)
        if self.count > self.maximum:
            raise EnvironmentCacheError("compressed cache archive exceeds its bound")
        return data

    def flush(self):
        self.stream.flush()


def _cc_cache_header(stream, header):
    raw = _cc_cache_json(header)
    if not 0 < len(raw) <= _CC_CACHE_HEADER_BYTES:
        raise EnvironmentCacheError("cache archive header exceeds its bound")
    stream.write(struct.pack(">I", len(raw)))
    stream.write(raw)


def _cc_cache_create_archive(prefix, archive, deadline):
    """Copy only a checked prefix. Hardlinked files become independent copies."""
    prefix = pathlib.Path(prefix)
    if prefix.is_symlink() or not prefix.is_dir():
        raise EnvironmentCacheError("cache prefix is missing or unsafe")
    prefix = prefix.resolve()
    count = total = 0
    links = []
    # Bound both disk and compressed bytes, checking free space throughout.
    with pathlib.Path(archive).open("xb") as raw:
        writer = _CCCacheBoundedIO(raw, deadline, _CC_CACHE_ARCHIVE_BYTES)
        with gzip.GzipFile(filename="", fileobj=writer, mode="wb", compresslevel=1, mtime=0) as stream:
            stream.write(_CC_CACHE_MAGIC)
            def visit(directory):
                nonlocal count, total
                _cc_cache_remaining(deadline)
                # scandir is consumed incrementally, not as an unbounded list.
                with os.scandir(directory) as children:
                    for child in children:
                        _cc_cache_remaining(deadline)
                        path = pathlib.Path(child.path)
                        name = _cc_cache_relative(path.relative_to(prefix).as_posix())
                        info = child.stat(follow_symlinks=False)
                        count += 1
                        if count > _CC_CACHE_FILES:
                            raise EnvironmentCacheError("cache archive file count exceeds its bound")
                        mode = stat.S_IMODE(info.st_mode) & 0o777
                        if stat.S_ISDIR(info.st_mode):
                            _cc_cache_header(stream, {"path": name, "kind": "dir", "mode": mode})
                            visit(path)
                        elif stat.S_ISLNK(info.st_mode):
                            link = _cc_cache_link(name, os.readlink(path))
                            links.append(name)
                            _cc_cache_header(stream, {"path": name, "kind": "link", "target": link})
                        elif stat.S_ISREG(info.st_mode):
                            total += info.st_size
                            if total > _CC_CACHE_EXPANDED_BYTES:
                                raise EnvironmentCacheError("cache archive expanded size exceeds its bound")
                            _cc_cache_header(stream, {"path": name, "kind": "file", "mode": mode,
                                                      "size": info.st_size})
                            # Refuse a changed link before opening; no following it.
                            with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as source:
                                current = os.fstat(source.fileno())
                                if (current.st_dev, current.st_ino, current.st_size) != (info.st_dev, info.st_ino, info.st_size):
                                    raise EnvironmentCacheError("cache prefix changed during archiving")
                                left = info.st_size
                                while left:
                                    _cc_cache_remaining(deadline)
                                    if shutil.disk_usage(pathlib.Path(archive).parent).free < _CC_CACHE_FREE_RESERVE:
                                        raise EnvironmentCacheError("insufficient cache archive disk space")
                                    block = source.read(min(left, _CC_CACHE_BLOCK))
                                    if not block:
                                        raise EnvironmentCacheError("cache prefix changed during archiving")
                                    stream.write(block)
                                    left -= len(block)
                                if source.read(1):
                                    raise EnvironmentCacheError("cache prefix changed during archiving")
                        else:
                            raise EnvironmentCacheError("cache prefix contains a special file")
            visit(prefix)
            _cc_cache_check_links(prefix, links)
            stream.write(struct.pack(">I", 0))
        raw.flush()
        os.fsync(raw.fileno())
    return {**_cc_cache_file_digest(archive, deadline, _CC_CACHE_ARCHIVE_BYTES),
            "entries": count, "expanded_bytes": total}


def _cc_cache_archive_record(value):
    if (not isinstance(value, dict) or set(value) != {"bytes", "sha256", "entries", "expanded_bytes"}
            or not _cc_cache_digest_valid(value.get("sha256"))
            or type(value.get("bytes")) is not int or not 0 < value["bytes"] <= _CC_CACHE_ARCHIVE_BYTES
            or type(value.get("entries")) is not int or not 0 < value["entries"] <= _CC_CACHE_FILES
            or type(value.get("expanded_bytes")) is not int
            or not 0 <= value["expanded_bytes"] <= _CC_CACHE_EXPANDED_BYTES):
        raise EnvironmentCacheError("invalid cache archive metadata")
    return value


def _cc_cache_extract_archive(archive, prefix, record, deadline):
    """Verify compressed identity, then extract to an owned sibling and rename."""
    record = _cc_cache_archive_record(record)
    prefix = pathlib.Path(prefix)
    if os.path.lexists(prefix):
        raise EnvironmentCacheError("cache restore prefix already exists")
    actual = _cc_cache_file_digest(archive, deadline, _CC_CACHE_ARCHIVE_BYTES)
    if actual != {k: record[k] for k in ("bytes", "sha256")}:
        raise EnvironmentCacheError("cache archive SHA-256 or size mismatch")
    if shutil.disk_usage(prefix.parent).free < record["expanded_bytes"] + _CC_CACHE_FREE_RESERVE:
        raise EnvironmentCacheError("insufficient cache extraction disk space")
    temporary = pathlib.Path(tempfile.mkdtemp(prefix=".restore-", dir=prefix.parent))
    try:
        entries = {}
        directories, links = [], []
        total = 0
        with pathlib.Path(archive).open("rb") as raw:
            reader = _CCCacheBoundedIO(raw, deadline, record["bytes"])
            with gzip.GzipFile(fileobj=reader, mode="rb") as stream:
                def exact(size):
                    _cc_cache_remaining(deadline)
                    data = stream.read(size)
                    if len(data) != size:
                        raise EnvironmentCacheError("truncated cache archive")
                    return data
                if exact(len(_CC_CACHE_MAGIC)) != _CC_CACHE_MAGIC:
                    raise EnvironmentCacheError("unsupported cache archive format")
                while True:
                    length = struct.unpack(">I", exact(4))[0]
                    if length == 0:
                        break
                    if length > _CC_CACHE_HEADER_BYTES or len(entries) >= record["entries"]:
                        raise EnvironmentCacheError("cache archive metadata exceeds its bound")
                    header = json.loads(exact(length))
                    if not isinstance(header, dict):
                        raise EnvironmentCacheError("invalid cache archive header")
                    name = _cc_cache_relative(header.get("path"))
                    kind = header.get("kind")
                    if name in entries:
                        raise EnvironmentCacheError("duplicate cache archive entry")
                    parent = posixpath.dirname(name)
                    if parent and entries.get(parent) != "dir":
                        raise EnvironmentCacheError("cache archive parent is not a declared directory")
                    target = temporary / name
                    if kind == "link":
                        if set(header) != {"path", "kind", "target"}:
                            raise EnvironmentCacheError("invalid cache archive link header")
                        os.symlink(_cc_cache_link(name, header["target"]), target)
                        links.append(name)
                    else:
                        mode = header.get("mode")
                        if type(mode) is not int or not 0 <= mode <= 0o777:
                            raise EnvironmentCacheError("invalid cache archive permissions")
                        if kind == "dir" and set(header) == {"path", "kind", "mode"}:
                            target.mkdir(mode=0o700)
                            directories.append((target, mode))
                        elif kind == "file" and set(header) == {"path", "kind", "mode", "size"}:
                            size = header["size"]
                            if type(size) is not int or not 0 <= size <= record["expanded_bytes"] - total:
                                raise EnvironmentCacheError("cache archive expanded size exceeds its bound")
                            total += size
                            with target.open("xb") as output:
                                left = size
                                while left:
                                    data = exact(min(left, _CC_CACHE_BLOCK))
                                    output.write(data)
                                    left -= len(data)
                            target.chmod(mode)
                        else:
                            raise EnvironmentCacheError("unsupported cache archive entry")
                    entries[name] = kind
                if stream.read(1) or len(entries) != record["entries"] or total != record["expanded_bytes"]:
                    raise EnvironmentCacheError("cache archive entry count or expanded size mismatch")
        _cc_cache_check_links(temporary, links)
        for target, mode in reversed(directories):
            target.chmod(mode)
        _cc_cache_remaining(deadline)
        if os.path.lexists(prefix):
            raise EnvironmentCacheError("cache restore prefix already exists")
        os.rename(temporary, prefix)
    finally:
        if temporary.exists():
            # Directory modes from an archive can be read-only. Restore owner
            # permissions before removing an incomplete owned staging tree.
            for directory, dirs, _ in os.walk(temporary, topdown=True, followlinks=False):
                pathlib.Path(directory).chmod(0o700)
                for name in dirs:
                    child = pathlib.Path(directory) / name
                    if not child.is_symlink():
                        child.chmod(0o700)
            shutil.rmtree(temporary)


def _cc_cache_read_metadata(directory):
    path = pathlib.Path(directory) / "manifest.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _CC_CACHE_METADATA_BYTES:
        raise EnvironmentCacheError("cache manifest is missing or exceeds its bound")
    with path.open("rb") as stream:
        raw = stream.read(_CC_CACHE_METADATA_BYTES + 1)
    if len(raw) > _CC_CACHE_METADATA_BYTES:
        raise EnvironmentCacheError("cache manifest exceeds its bound")
    return json.loads(raw)


class EnvironmentCache:
    """Prepare a verified hit or a cold-build destination before workload execution."""
    def __init__(self, storage, prefix_root, manifest, implementation, hardware, deadline,
                 workload_reserve, *, fingerprint=None):
        self.storage = storage
        self.root = pathlib.Path(prefix_root).absolute()
        self.manifest, self.implementation, self.hardware = manifest, implementation, hardware
        self.deadline, self.workload_reserve = deadline, workload_reserve
        self.fingerprint = fingerprint or _cc_cache_runtime_fingerprint
        self.runtime = None
        self.locks = []

    def _operation_deadline(self, restore):
        now = time.time()
        # Reserve the workload and finalization budget; leave half of the setup
        # time for a cold fallback if a restore is slow or invalid.
        remaining = self.deadline - now - self.workload_reserve
        available = remaining / 2 if restore else remaining
        if available < 5:
            raise TimeoutError("insufficient budget for optional cache operation")
        return now + min(_CC_CACHE_TRANSFER_SECONDS, available)

    def choose(self, name, definition):
        report = {"state": "bypass", "published": False,
                  "cpu_policy": definition.get("cache_cpu", "exact")}
        if self.storage is None:
            report["reason"] = "managed_cache_storage_unavailable"
            return {"report": report}
        try:
            deadline = self._operation_deadline(True)
            if self.runtime is None:
                self.runtime = self.fingerprint(self.manifest["resources"], self.hardware, deadline)
            # Retain the actual host evidence, even when the key deliberately
            # projects away non-required CPU flags under the portable contract.
            report["runtime_fingerprint"] = self.runtime
            self.root = _cc_cache_check_root(self.root)
            identity = _cc_cache_identity(name, definition, self.manifest, self.runtime,
                                          self.root, self.implementation)
            key = _cc_cache_hash(identity)
            prefix = self.root / key
            if os.path.lexists(prefix):
                report["reason"] = "prefix_already_exists"
                return {"report": report}
            lock = self.root / (key + ".lock")
            try:
                fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                report["reason"] = "prefix_locked"
                return {"report": report}
            os.close(fd)
            self.locks.append(lock)
            if os.path.lexists(prefix):
                report["reason"] = "prefix_already_exists"
                return {"report": report}
            report.update(state="miss", key=key)
            selected = {"report": report, "identity": identity, "prefix": prefix, "key": key}
            if identity["runtime"].get("cpu_target") == "x86-64-v2":
                # The pinned micromamba release accepts this virtual-package
                # override directly; it scopes solver CPU selection, not codegen.
                selected["conda_archspec"] = "x86_64_v2"
                report["conda_archspec"] = selected["conda_archspec"]
            return selected
        except Exception as exc:
            # Do not copy arbitrary storage or environment exception messages to
            # public evidence; they can contain private URLs or credentials.
            report["reason"] = (str(exc) if type(exc) is EnvironmentCacheError else type(exc).__name__)
            return {"report": report}

    def restore(self, selected, validate):
        report = selected["report"]
        started = time.monotonic()
        prefix = selected["prefix"]
        restored = False
        try:
            deadline = self._operation_deadline(True)
            with tempfile.TemporaryDirectory(prefix=".cache-download-", dir=self.root) as temporary:
                local = pathlib.Path(temporary) / "snapshot"
                transfer_started = time.monotonic()
                found = self.storage.load(selected["key"], local, deadline)
                report["lookup_download_seconds"] = time.monotonic() - transfer_started
                if not found:
                    return None
                meta = _cc_cache_read_metadata(local)
                if (not isinstance(meta, dict) or meta.get("format") != _CC_CACHE_FORMAT
                        or meta.get("identity") != selected["identity"] or meta.get("key") != selected["key"]
                        or meta.get("prefix") != str(prefix)
                        or not isinstance(meta.get("facts"), dict)
                        or type(meta.get("cold_build_seconds")) not in (int, float)
                        or not math.isfinite(meta["cold_build_seconds"]) or meta["cold_build_seconds"] < 0):
                    raise EnvironmentCacheError("cache manifest identity or evidence mismatch")
                restore_started = time.monotonic()
                _cc_cache_extract_archive(local / "environment.ccenv.gz", prefix, meta["archive"], deadline)
                restored = True
                report["extraction_seconds"] = time.monotonic() - restore_started
                validation_started = time.monotonic()
                facts = validate(deadline)
                if _cc_cache_json(facts) != _cc_cache_json(meta["facts"]):
                    raise EnvironmentCacheError("restored package or toolchain evidence differs")
                report["validation_seconds"] = time.monotonic() - validation_started
                report.update(state="hit", cold_build_seconds=meta["cold_build_seconds"],
                              archive_bytes=meta["archive"]["bytes"])
                return facts
        except Exception as exc:
            report.update(state="rebuild", reason=(str(exc) if type(exc) is EnvironmentCacheError else type(exc).__name__))
            if restored:
                # Only remove the exact prefix successfully created by this restore.
                # Do not follow links or delete a pre-existing/foreign prefix.
                if prefix.is_symlink() or not prefix.is_dir():
                    raise EnvironmentCacheError("restored prefix changed unexpectedly")
                for directory, dirs, _ in os.walk(prefix, topdown=True, followlinks=False):
                    pathlib.Path(directory).chmod(0o700)
                    for name in dirs:
                        child = pathlib.Path(directory) / name
                        if not child.is_symlink():
                            child.chmod(0o700)
                shutil.rmtree(prefix)
            return None
        finally:
            report["restore_attempt_seconds"] = time.monotonic() - started

    def publish(self, selected, facts, build_seconds):
        report = selected["report"]
        started = time.monotonic()
        report["cold_build_seconds"] = build_seconds
        try:
            deadline = self._operation_deadline(False)
            with tempfile.TemporaryDirectory(prefix=".cache-upload-", dir=self.root) as temporary:
                local = pathlib.Path(temporary)
                record = _cc_cache_create_archive(selected["prefix"], local / "environment.ccenv.gz", deadline)
                meta = {"format": _CC_CACHE_FORMAT, "identity": selected["identity"], "key": selected["key"],
                        "prefix": str(selected["prefix"]), "archive": record, "facts": facts,
                        "cold_build_seconds": build_seconds}
                body = _cc_cache_json(meta)
                if len(body) > _CC_CACHE_METADATA_BYTES:
                    raise EnvironmentCacheError("cache manifest exceeds its bound")
                (local / "manifest.json").write_bytes(body)
                self.storage.save(selected["key"], local, deadline)
                report.update(published=True, archive_bytes=record["bytes"])
        except Exception as exc:
            report["publish_error"] = str(exc) if type(exc) is EnvironmentCacheError else type(exc).__name__
        finally:
            report["publish_seconds"] = time.monotonic() - started

    def close(self):
        for lock in self.locks:
            lock.unlink(missing_ok=True)
        self.locks.clear()
