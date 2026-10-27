"""Bounded immutable Drive objects and hash-bound lifecycle snapshots."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


class LifecycleError(RuntimeError):
    pass


class AllocationRejected(LifecycleError):
    """Definitive provider refusal: stop rather than allocate repeatedly."""
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
            stream.write(encoded(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def relative(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\0" in name:
        raise LifecycleError("unsafe snapshot path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(p in ("", ".", "..") for p in name.split("/")):
        raise LifecycleError("unsafe snapshot path")
    return name


def read_regular(root, name, maximum=64 * 1024 * 1024):
    root = Path(root).resolve()
    path = root / relative(name)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root and root in p.parents):
        raise LifecycleError("snapshot symlink rejected")
    path.resolve().relative_to(root)
    if not path.is_file() or path.stat().st_size > maximum:
        raise LifecycleError("snapshot file missing or too large")
    return path.read_bytes()


class DriveStore:
    """Only objects beneath an explicitly selected parent and this random run ID."""
    def __init__(self, parent, run_id, token, *, request=None):
        if not parent or not all(c.isalnum() or c in "_-" for c in parent):
            raise LifecycleError("invalid Drive parent")
        if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
            raise LifecycleError("invalid run ID")
        self.parent, self.run_id, self.token = parent, run_id, token
        self.request = request or urllib.request.urlopen

    def api(self, method, route, data=None, headers=None):
        for retry in range(3):
            request = urllib.request.Request(
                "https://www.googleapis.com/" + route, data=data, method=method,
                headers={"Authorization": "Bearer " + self.token(), **(headers or {})})
            try:
                with self.request(request, timeout=30) as response:
                    body = response.read(70 * 1024 * 1024 + 1)
                    if len(body) > 70 * 1024 * 1024:
                        raise LifecycleError("Drive response exceeds proof limit")
                    return body
            except urllib.error.HTTPError as exc:
                # Never include the request URL, body or credentials in errors.
                if exc.code not in (429, 500, 502, 503, 504) or retry == 2:
                    raise LifecycleError(f"Drive HTTP {exc.code}") from None
            except (TimeoutError, urllib.error.URLError):
                if retry == 2:
                    raise LifecycleError("Drive connection unavailable") from None
            time.sleep(0.25 * 2**retry)
        raise LifecycleError("Drive request failed")

    def list(self, kind):
        query = (f"'{self.parent}' in parents and trashed = false and "
                 f"appProperties has {{ key='cc_run' and value='{self.run_id}' }} and "
                 f"appProperties has {{ key='cc_kind' and value='{kind}' }}")
        page, records = None, []
        while True:
            params = {"q": query, "fields": "nextPageToken,incompleteSearch,files(id,name,size,appProperties,createdTime)",
                      "pageSize": 1000}
            if page:
                params["pageToken"] = page
            value = json.loads(self.api("GET", "drive/v3/files?" + urllib.parse.urlencode(params)))
            if value.get("incompleteSearch"):
                raise LifecycleError("Drive listing incomplete")
            records.extend(value.get("files", []))
            if len(records) > 10000:
                raise LifecycleError("Drive proof object limit exceeded")
            page = value.get("nextPageToken")
            if not page:
                return records

    def put(self, kind, name, body):
        sha = digest(body)
        # Refresh the remote catalog once per store/kind, then index content locally.
        # An uncertain upload can still create a duplicate; every chosen ID is verified.
        if not hasattr(self, "_content_index"):
            self._content_index = {}
        if kind not in self._content_index:
            index = {}
            for record in self.list(kind):
                index.setdefault(record.get("appProperties", {}).get("cc_sha"), []).append(record)
            self._content_index[kind] = index
        matches = self._content_index[kind].get(sha, [])
        if matches:
            # Retried uploads may create duplicates after an uncertain response; bind any
            # exact verified object, never trust a filename or silently delete duplicates.
            record = sorted(matches, key=lambda f: f["id"])[0]
            if self.get({"id": record["id"], "sha256": sha, "bytes": len(body)}) != body:
                raise LifecycleError("existing Drive object differs")
            return {"id": record["id"], "sha256": sha, "bytes": len(body)}
        boundary = "colabcola" + uuid.uuid4().hex
        metadata = {"name": name, "parents": [self.parent],
                    "appProperties": {"cc_run": self.run_id, "cc_kind": kind, "cc_sha": sha}}
        payload = (f"--{boundary}\r\nContent-Type: application/json\r\n\r\n".encode()
                   + encoded(metadata)
                   + f"\r\n--{boundary}\r\nContent-Type: application/octet-stream\r\n\r\n".encode()
                   + body + f"\r\n--{boundary}--\r\n".encode())
        record = json.loads(self.api("POST", "upload/drive/v3/files?uploadType=multipart&fields=id",
                                     payload, {"Content-Type": "multipart/related; boundary=" + boundary}))
        ref = {"id": record["id"], "sha256": sha, "bytes": len(body)}
        if self.get(ref) != body:
            raise LifecycleError("Drive readback mismatch")
        self._content_index[kind].setdefault(sha, []).append({"id": ref["id"]})
        return ref

    def get(self, ref):
        if not isinstance(ref.get("bytes"), int) or not 0 <= ref["bytes"] <= 64 * 1024 * 1024:
            raise LifecycleError("invalid object size")
        file_id = ref.get("id", "")
        if not file_id or not all(c.isalnum() or c in "_-" for c in file_id):
            raise LifecycleError("invalid Drive object ID")
        body = self.api("GET", f"drive/v3/files/{file_id}?alt=media")
        if len(body) != ref["bytes"] or digest(body) != ref["sha256"]:
            raise LifecycleError("Drive object integrity mismatch")
        return body


def publish(store, identity, kind, members, *, step=0, attempt=0):
    if len(members) > 64 or sum(len(v) for v in members.values()) > 64 * 1024 * 1024:
        raise LifecycleError("snapshot exceeds bounded proof limits")
    # Only the immediately preceding log publication is reused. Every reference
    # entered this cache after put() completed its authenticated hash readback.
    identity_key = (identity["run_id"], identity["job_id"], attempt)
    prior = getattr(store, "_log_member_refs", {}).get(identity_key, {}) if kind == "logs" else {}
    references = {}
    for name, body in sorted(members.items()):
        relative(name)
        old = prior.get(name)
        if old and old[1:] == (digest(body), len(body)):
            references[name] = dict(zip(("id", "sha256", "bytes"), old))
        else:
            references[name] = store.put("member", name.rsplit("/", 1)[-1], body)
    commit = {"schema": 1, "run_id": identity["run_id"], "job_id": identity["job_id"],
              "kind": kind, "step": step, "attempt": attempt, "members": references}
    # The commit marker is published only after every member has passed remote readback.
    ref = store.put(kind, f"colabcola-{kind}-{step}-{attempt}.json", encoded(commit))
    if kind == "logs":
        store._log_member_refs = {identity_key: {name: (r["id"], r["sha256"], r["bytes"])
                                                for name, r in references.items()}}
    return {**ref, "commit": commit}


def snapshots(store, identity, kind):
    found = []
    # Cache only bytes already fetched through store.get's size/SHA-256 check.
    # Include the full identity and reference: changed metadata must be reverified.
    # Decode a fresh object for each caller so returned commits cannot poison it.
    cache = getattr(store, "_snapshot_cache", {})
    store._snapshot_cache = cache
    catalog = store.list(kind)
    current = set()
    for entry in catalog:
        props = entry.get("appProperties", {})
        ref = {"id": entry["id"], "bytes": int(entry["size"]), "sha256": props["cc_sha"]}
        key = (identity["run_id"], identity["job_id"], kind,
               ref["id"], ref["bytes"], ref["sha256"])
        current.add(key)
        body = cache.get(key)
        if body is None:
            body = store.get(ref)
            if len(body) != ref["bytes"] or digest(body) != ref["sha256"]:
                raise LifecycleError("snapshot commit integrity mismatch")
        commit = json.loads(body)
        if (commit.get("schema") not in (1, 2) or commit.get("run_id") != identity["run_id"]
                or commit.get("job_id") != identity["job_id"] or commit.get("kind") != kind):
            raise LifecycleError("snapshot identity mismatch")
        if (type(commit.get("step")) is not int or commit["step"] < 0
                or type(commit.get("attempt")) is not int or commit["attempt"] < 0):
            raise LifecycleError("invalid snapshot step")
        cache[key] = body
        found.append({**ref, "commit": commit, "created_at": entry.get("createdTime", "")})
    # Keep the current catalog only; a vanished/reappearing object is fetched again.
    for key in list(cache):
        if key[:3] == (identity["run_id"], identity["job_id"], kind) and key not in current:
            del cache[key]
    ordered = sorted(found, key=lambda x: (x["commit"]["step"], x["commit"]["attempt"], x["created_at"], x["id"]))
    for snapshot in ordered:
        snapshot.pop("created_at")  # Ordering metadata is not part of the durable reference contract.
    return ordered


def recover(store, snapshot, destination, *, previous=None):
    if snapshot["commit"].get("schema") == 2:
        return recover_files(store, snapshot, destination)
    target = Path(destination)
    members = snapshot["commit"]["members"]
    if not isinstance(members, dict) or not 1 <= len(members) <= 64:
        raise LifecycleError("invalid snapshot members")
    # An unchanged log member may come from the prior local snapshot, but its
    # bytes are hashed again. Never trust a cached pathname or mutable metadata.
    before = (previous or {}).get("commit", {})
    current = snapshot["commit"]
    reuse = (current.get("kind") == before.get("kind") == "logs"
             and current.get("schema") == before.get("schema") == 1
             and all(current.get(k) == before.get(k) for k in ("run_id", "job_id", "attempt"))
             and (previous or {}).get("local_directory"))
    bodies = {}
    for name, ref in members.items():
        relative(name)
        if (reuse and before.get("members", {}).get(name) == ref
                and (Path(previous["local_directory"]) / name).exists()):
            body = read_regular(previous["local_directory"], name)
            if len(body) != ref["bytes"] or digest(body) != ref["sha256"]:
                raise LifecycleError("cached log member integrity mismatch")
        else:
            body = store.get(ref)
        bodies[name] = body
    # Verify every byte before making a directory visible to the caller.
    if sum(len(v) for v in bodies.values()) > 64 * 1024 * 1024:
        raise LifecycleError("snapshot exceeds proof limit")
    if target.exists():
        for name, body in bodies.items():
            if read_regular(target, name) != body:
                raise LifecycleError("existing recovered snapshot differs")
        return target
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".partial")
    temporary.mkdir(parents=True)
    for name, body in bodies.items():
        path = temporary / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    os.replace(temporary, target)
    return target


# Schema 2 snapshots transfer files in bounded chunks. Completed immutable files reuse
# their exact remote references; no whole grid archive is loaded into host memory.
CHUNK_BYTES = 8 * 1024 * 1024
ARTIFACT_BYTES = 5 * 1024 * 1024 * 1024
RESULT_METADATA_BYTES = 64 * 1024 * 1024
# Includes ZIP overhead, result/attempt metadata and final logs; output bound is separate.
SNAPSHOT_BYTES = 6 * 1024 * 1024 * 1024
MAX_MEMBERS = 4096


def safe_file(root, name):
    root = Path(root).resolve()
    path = root / relative(name)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root and root in p.parents):
        raise LifecycleError("snapshot symlink rejected")
    path.resolve().relative_to(root)
    if not path.is_file():
        raise LifecycleError("snapshot member missing")
    return path


def publish_files(store, identity, kind, members, *, step=0, attempt=0):
    if not 1 <= len(members) <= MAX_MEMBERS:
        raise LifecycleError("snapshot file count exceeded")
    sizes = {name: Path(path).stat().st_size for name, path in members.items()}
    total = sum(sizes.values())
    if total > SNAPSHOT_BYTES:
        raise LifecycleError("snapshot exceeds 6 GiB transfer bound")
    cache = getattr(store, "_file_refs", {})
    store._file_refs = cache
    references = {}
    for name, file in sorted(members.items()):
        relative(name)
        path = Path(file)
        if path.is_symlink() or not path.is_file():
            raise LifecycleError("snapshot requires regular files")
        hasher = hashlib.sha256()
        size, hashed = sizes[name], 0
        with path.open("rb") as stream:
            while data := stream.read(CHUNK_BYTES):
                hashed += len(data)
                if hashed > size:
                    raise LifecycleError("snapshot grew during hashing")
                hasher.update(data)
        if hashed != size or path.stat().st_size != size:
            raise LifecycleError("snapshot changed during hashing")
        sha = hasher.hexdigest()
        key = (size, sha)
        if key not in cache:
            chunks = []
            check = hashlib.sha256()
            transferred = 0
            with path.open("rb") as stream:
                while data := stream.read(CHUNK_BYTES):
                    transferred += len(data)
                    if transferred > size:
                        raise LifecycleError("snapshot grew during publication")
                    check.update(data)
                    chunks.append(store.put("member", "chunk-" + digest(data), data))
            if check.hexdigest() != sha or sum(c["bytes"] for c in chunks) != size:
                raise LifecycleError("snapshot changed during publication")
            cache[key] = {"bytes": size, "sha256": sha, "chunks": chunks}
        references[name] = cache[key]
    commit = {"schema": 2, "run_id": identity["run_id"], "job_id": identity["job_id"],
              "kind": kind, "step": step, "attempt": attempt, "members": references}
    body = encoded(commit)
    if len(body) > 4 * 1024 * 1024:
        raise LifecycleError("snapshot manifest exceeds 4 MiB")
    ref = store.put(kind, f"colabcola-{kind}-{step}-{attempt}.json", body)
    return {**ref, "commit": commit}


def recover_files(store, snapshot, destination):
    import shutil
    members = snapshot["commit"]["members"]
    if not isinstance(members, dict) or not 1 <= len(members) <= MAX_MEMBERS:
        raise LifecycleError("invalid snapshot member count")
    total = 0
    for name, record in members.items():
        relative(name)
        size = record.get("bytes")
        chunks = record.get("chunks")
        if type(size) is not int or not 0 <= size <= SNAPSHOT_BYTES or not isinstance(chunks, list):
            raise LifecycleError("invalid chunked member")
        if len(chunks) > SNAPSHOT_BYTES // CHUNK_BYTES + 1:
            raise LifecycleError("too many chunks")
        if any(not isinstance(c, dict) or type(c.get("bytes")) is not int
               or not 0 < c["bytes"] <= CHUNK_BYTES
               or not isinstance(c.get("sha256"), str) or len(c["sha256"]) != 64
               or any(char not in "0123456789abcdef" for char in c["sha256"]) for c in chunks):
            raise LifecycleError("invalid chunk size")
        if sum(c["bytes"] for c in chunks) != size:
            raise LifecycleError("chunk sizes differ from member")
        total += size
    if total > SNAPSHOT_BYTES:
        raise LifecycleError("snapshot exceeds 6 GiB transfer bound")
    target = Path(destination)
    if target.is_symlink():
        raise LifecycleError("recovery destination symlink")
    if target.exists():
        for name, record in members.items():
            path = safe_file(target, name)
            with path.open("rb") as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
            if path.stat().st_size != record["bytes"] or sha != record["sha256"]:
                raise LifecycleError("existing recovered snapshot differs")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target.parent).free < 3 * total + RESULT_METADATA_BYTES:
        raise LifecycleError("insufficient recovery disk: reserve three times the snapshot size plus 64 MiB (including copy fallback)")
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".partial")
    temporary.mkdir(parents=True)
    try:
        for name, record in members.items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            sha = record["sha256"]
            if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise LifecycleError("invalid member digest")
            cache = target.parent / ".objects"
            if cache.is_symlink():
                raise LifecycleError("recovery cache symlink")
            cache.mkdir(exist_ok=True)
            cached = cache / sha
            if cached.exists():
                if cached.is_symlink() or not cached.is_file():
                    raise LifecycleError("invalid cached object")
                with cached.open("rb") as stream:
                    check = hashlib.file_digest(stream, "sha256").hexdigest()
                if cached.stat().st_size != record["bytes"] or check != sha:
                    raise LifecycleError("cached object integrity mismatch")
                try:
                    os.link(cached, path)
                except OSError:
                    shutil.copyfile(cached, path)
                continue
            hasher = hashlib.sha256()
            with path.open("xb") as stream:
                for ref in record["chunks"]:
                    chunk_root = cache / ".chunks"
                    if chunk_root.is_symlink():
                        raise LifecycleError("chunk cache symlink")
                    chunk_root.mkdir(exist_ok=True)
                    chunk_path = chunk_root / ref["sha256"]
                    if chunk_path.exists():
                        body = read_regular(chunk_root, ref["sha256"], CHUNK_BYTES)
                    else:
                        body = store.get(ref)
                    if len(body) != ref["bytes"] or digest(body) != ref["sha256"]:
                        raise LifecycleError("chunk integrity mismatch")
                    if not chunk_path.exists():
                        pending = chunk_path.with_name(chunk_path.name + "." + uuid.uuid4().hex + ".tmp")
                        try:
                            pending.write_bytes(body)
                            os.replace(pending, chunk_path)
                        finally:
                            pending.unlink(missing_ok=True)
                    stream.write(body)
                    hasher.update(body)
            if hasher.hexdigest() != sha:
                raise LifecycleError("member integrity mismatch")
            try:
                os.link(path, cached)
            except FileExistsError:
                pass
            except OSError:
                shutil.copyfile(path, cached)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target



# Notebook compilation/execution still materializes JSON, so cap input memory separately.
INPUT_BYTES = 512 * 1024 * 1024
INPUT_NAMES = ("notebook.ipynb", "lifecycle_worker.py", "lifecycle_io.py")

def verified_inputs(root, hashes):
    if set(hashes) != set(INPUT_NAMES):
        raise LifecycleError("unexpected prepared input members")
    paths = {name: safe_file(root, name) for name in INPUT_NAMES}
    if sum(p.stat().st_size for p in paths.values()) > INPUT_BYTES:
        raise LifecycleError("managed inputs exceed 512 MiB bound")
    for name, path in paths.items():
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != hashes[name]:
                raise LifecycleError("prepared source changed")
    return paths
