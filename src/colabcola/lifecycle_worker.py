"""Runtime worker: managed execution, checkpoint/log commits and final Drive publication."""
from __future__ import annotations
import contextlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import threading

try:
    from .lifecycle_io import DriveStore, LifecycleError, atomic, digest, encoded, publish, read_regular, recover, publish_files, safe_file, verified_inputs, INPUT_BYTES
except ImportError:
    from lifecycle_io import DriveStore, LifecycleError, atomic, digest, encoded, publish, read_regular, recover, publish_files, safe_file, verified_inputs, INPUT_BYTES


class AuthorizationEvidence:
    """Public counters from successful Drive operations; token values stay in memory."""
    def __init__(self, clock=time.time):
        self.clock = clock
        self.initial_token = None
        self.last_token = None
        self.value = {"schema": 1, "token_changes": 0, "verified_publications": 0}

    def used(self, token):
        if self.initial_token is None:
            self.initial_token = token["access_token"]
            self.value["initial_expires_at"] = token["expires_at"]
        elif self.last_token != token["access_token"]:
            self.value["token_changes"] += 1
        self.last_token = token["access_token"]
        self.value["latest_expires_at"] = token["expires_at"]

    def published(self, root, kind):
        # publish/put return only after the commit's authenticated readback succeeded.
        now = self.clock()
        self.value["verified_publications"] += 1
        self.value["last_verified_publication"] = {"kind": kind, "at": now}
        if self.last_token != self.initial_token and self.initial_token is not None:
            self.value.setdefault("first_verified_rotated_publication_at", now)
            if now >= self.value["initial_expires_at"] + 120:
                self.value.setdefault("verified_after_initial_expiry_at", now)
        atomic(root / "authorization.json", self.value)


_TOKEN_LOCK = threading.Lock()


def current_token(root, now=None):
    """Validate a pending upload and atomically adopt it outside the busy kernel."""
    now = time.time() if now is None else now
    with _TOKEN_LOCK:
        value = json.loads((root / "drive-token.json").read_text())
        try:
            candidate = json.loads((root / "drive-token.next.json").read_text())
            if (set(candidate) == {"access_token", "expires_at"}
                    and isinstance(candidate["access_token"], str) and candidate["access_token"]
                    and type(candidate["expires_at"]) in (int, float)
                    and math.isfinite(candidate["expires_at"])
                    and candidate["expires_at"] > max(now + 120, value["expires_at"])):
                atomic(root / "drive-token.json", candidate)
                value = candidate
            # Leave pending in place:never unlink a newer concurrent server upload.
        except (OSError, ValueError, TypeError, KeyError):
            pass  # Partial/invalid uploads cannot replace the current lease.
        return value


def token_reader(root, evidence=None):
    def read():
        value = current_token(root)
        if value["expires_at"] <= time.time() + 30:
            raise LifecycleError("runtime Drive authorization expired")
        if evidence is not None:
            evidence.used(value)
        return value["access_token"]
    return read



def stop_reason(config, root, now):
    if now >= config['deadline']:
        return 'wall_budget_exhausted'
    if config.get('authorization_policy') == 'renewable-v1':
        try:
            token = current_token(root, now)
            expiry = token['expires_at']
            if (type(expiry) not in (int, float) or not math.isfinite(expiry)
                    or not expiry > now + 120):
                return 'drive_authorization_expiring'
            if not isinstance(token.get('access_token'), str) or not token['access_token']:
                return 'drive_authorization_unavailable'
        except (OSError, ValueError, KeyError, TypeError):
            return 'drive_authorization_unavailable'
    return None


class ExecutionWatchdog:
    """Independent of potentially blocked Drive publication in the worker loop."""
    def __init__(self, config, root, process):
        self.config, self.root, self.process = config, root, process
        self.stopped = threading.Event()
        self.reason = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stopped.wait(1):
            reason = stop_reason(self.config, self.root, time.time())
            if reason and self.process.poll() is None:
                self.reason = reason
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                atomic(self.root / 'watchdog.json', {'reason': reason, 'at': time.time()})
                return

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopped.set()
        self.thread.join(timeout=2)

def checkpoint_members(job_root, pattern):
    candidates = []
    for manifest in Path(job_root).glob(pattern):
        raw = read_regular(job_root, manifest.relative_to(job_root).as_posix(), 1024 * 1024)
        value = json.loads(raw)
        if (value.get("schema") not in (1, 2) or type(value.get("step")) is not int
                or value["step"] < 0 or not value.get("file")
                or Path(value["file"]).name != value["file"]
                or "/" in value["file"] or "\\\\" in value["file"]):
            raise LifecycleError("invalid workload checkpoint manifest")
        candidates.append((value["step"], manifest, value, raw))
    if not candidates:
        return []
    step, manifest, value, raw = max(candidates, key=lambda row: row[0])
    state_relative = (manifest.parent / value["file"]).relative_to(job_root).as_posix()
    state = read_regular(job_root, state_relative)
    if len(state) != value["bytes"] or digest(state) != value["sha256"]:
        raise LifecycleError("workload checkpoint integrity mismatch")
    if value["schema"] == 1:
        return [(step, {manifest.name: raw, value["file"]: state})]
    members = {manifest.name: manifest, value["file"]: safe_file(job_root, state_relative)}
    artifacts = value.get("artifacts", {})
    if not isinstance(artifacts, dict) or len(artifacts) > 4000:
        raise LifecycleError("invalid checkpoint artifact catalog")
    for name, record in artifacts.items():
        path = safe_file(manifest.parent, name)
        import hashlib
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != record["bytes"] or sha != record["sha256"]:
            raise LifecycleError("checkpoint artifact mismatch")
        if name in members:
            raise LifecycleError("checkpoint member collision")
        members[name] = path
    return [(step, members)]


def execute_notebook(config_path):
    config = json.loads(Path(config_path).read_text())
    root = Path(config["directory"])
    document = read_regular(root, "notebook.ipynb", INPUT_BYTES)
    if digest(document) != config["files"]["notebook.ipynb"]:
        raise LifecycleError("notebook identity mismatch")
    namespace = {"COLABCOLA_CONTENT_ROOT_OVERRIDE": config.get("content_root", "/content"),
                 "COLABCOLA_LIFECYCLE_DEADLINE": config["deadline"],
                 "COLABCOLA_REQUESTED_GPU_TYPE": config.get("gpu"),
                 "COLABCOLA_HARDWARE_POLICY": config.get("hardware_policy"),
                 "COLABCOLA_RUN_ID": config.get("run_id"),
                 "COLABCOLA_ATTEMPT": config.get("attempt"),
                 "COLABCOLA_ALLOCATION_ID": config.get("allocation_id")}
    for cell in json.loads(document)["cells"]:
        if cell["cell_type"] == "code":
            exec(cell["source"], namespace)


def acquire_inputs(config, store, root):
    """Recover only the plan-bound input snapshot, before executing any notebook code."""
    snapshot = config.get("input_snapshot")
    if snapshot is None:
        return
    commit = json.loads(store.get(snapshot))
    if (commit != snapshot["commit"] or commit.get("schema") != 2
            or commit.get("kind") != "input" or commit.get("run_id") != config["run_id"]
            or commit.get("job_id") != config["job_id"]
            or set(commit.get("members", {})) != set(config["files"])):
        raise LifecycleError("input snapshot identity mismatch")
    if sum(m["bytes"] for m in commit["members"].values()) > INPUT_BYTES:
        raise LifecycleError("input snapshot exceeds 512 MiB")
    for name, record in commit["members"].items():
        if record["sha256"] != config["files"][name]:
            raise LifecycleError("input snapshot source mismatch")
    recovered = recover(store, snapshot, root / "input")
    verified_inputs(recovered, config["files"])
    import shutil
    shutil.copyfile(recovered / "notebook.ipynb", root / "notebook.ipynb")


def log_members(result_root, job_root, patterns):
    logs = {}
    for name in ("stdout.log", "stderr.log", "environment.log", "transfer.log"):
        if (result_root / name).is_file():
            logs["runtime/" + name] = read_regular(result_root, name, 8 * 1024 * 1024)
    for pattern in patterns:
        for path in job_root.glob(pattern):
            name = path.relative_to(job_root).as_posix()
            logs[name] = read_regular(job_root, name, 8 * 1024 * 1024)
    return logs


class CompletedArtifacts:
    """Freeze files only after a declared JSON completion marker matches its condition."""
    def __init__(self, config, job_root, control_root):
        self.config, self.job_root = config, Path(job_root)
        self.root = Path(control_root) / "completed-files"
        self.markers, self.members = {}, {}
        self.dirty = False

    def collect(self):
        import fnmatch
        import shutil
        import hashlib
        for group in self.config.get("artifact_groups", []):
            for marker in sorted(self.job_root.glob(group["marker"])):
                marker_name = marker.relative_to(self.job_root).as_posix()
                raw = read_regular(self.job_root, marker_name, 1024 * 1024)
                marker_hash = digest(raw)
                if marker_name in self.markers:
                    if self.markers[marker_name] != marker_hash:
                        raise LifecycleError("completed artifact marker changed;use a new output path")
                    continue
                try:
                    value = json.loads(raw)
                except ValueError:
                    continue  # A non-atomic writer may still be completing its marker.
                if not isinstance(value, dict) or not all(value.get(k) == v for k, v in group["ready_when"].items()):
                    continue
                sources = {marker_name: safe_file(self.job_root, marker_name)}
                for pattern in group["files"]:
                    matches = [v for v in marker.parent.glob(pattern) if not v.is_dir()]
                    if not matches:
                        raise LifecycleError("completed artifact member missing")
                    for file in matches:
                        name = file.relative_to(self.job_root).as_posix()
                        sources[name] = safe_file(self.job_root, name)
                sizes = {n: v.stat().st_size for n, v in {**self.members, **sources}.items()}
                if len(sizes) > 4096 or sum(sizes.values()) > 5 * 1024**3:
                    raise LifecycleError("completed artifacts exceed 4096 files or5GiB")
                rules = self.config.get("artifact_output_rules", [])
                if any(not any(fnmatch.fnmatchcase(n, r["pattern"]) for r in rules) for n in sources):
                    raise LifecycleError("completed artifact is outside declared outputs")
                for rule in rules:
                    if sum(size for n, size in sizes.items() if fnmatch.fnmatchcase(n, rule["pattern"])) > rule["maximum_bytes"]:
                        raise LifecycleError("completed artifacts exceed declared output bound")
                staged = {}
                for name, source in sources.items():
                    destination = self.root / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(destination.name + ".partial")
                    before = source.stat()
                    with source.open("rb") as src, temporary.open("wb") as dst:
                        shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
                    after = source.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        temporary.unlink()
                        raise LifecycleError("completed artifact changed while freezing")
                    if name in self.members:
                        with temporary.open("rb") as f: new_hash = hashlib.file_digest(f, "sha256").hexdigest()
                        with self.members[name].open("rb") as f: old_hash = hashlib.file_digest(f, "sha256").hexdigest()
                        temporary.unlink()
                        if new_hash != old_hash:
                            raise LifecycleError("completed artifact path collision")
                    else:
                        os.replace(temporary, destination)
                    staged[name] = destination
                if read_regular(self.job_root, marker_name, 1024 * 1024) != raw or staged[marker_name].read_bytes() != raw:
                    raise LifecycleError("completion marker changed while freezing")
                self.members.update(staged)
                self.markers[marker_name] = marker_hash
                self.dirty = True

    def publish(self, store):
        self.collect()
        if not self.dirty:
            return None
        snapshot = publish_files(store, self.config, "artifacts", self.members,
                                 step=len(self.markers), attempt=self.config["attempt"])
        self.dirty = False
        return snapshot


def _stop_execution_child(process):
    """Kill only the worker-owned execution session, including descendants."""
    if process is None:
        return
    try:
        # The child was created with start_new_session=True; its PID is its PGID.
        # Kill even if the leader exited, to prevent unsupervised descendants.
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run(config_path):
    import fcntl  # Runtime is Linux; the supervisor core has its own portable locking.
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    root = Path(config["directory"])
    lock = (root / "worker.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    # Replayed bootstrap must never rerun a completed or crashed attempt.
    if (root / "started.json").exists():
        return
    atomic(root / "started.json", {"pid": os.getpid(), "at": time.time()})
    evidence = AuthorizationEvidence() if config.get("authorization_policy") == "renewable-v1" else None
    store = DriveStore(config["parent"], config["run_id"], token_reader(root, evidence))
    status = {"state": "preparing", "attempt": config["attempt"], "checkpoint_step": 0}
    atomic(root / "status.json", status)
    process = None
    watchdog = None
    previous_handlers = {}
    received_signal = None

    def record_termination_signal(signum, _frame):
        nonlocal received_signal
        # Python's signal handler has no portable sender-PID information.
        received_signal = {"name": signal.Signals(signum).name, "at": time.time()}
        if signum == signal.SIGINT:
            raise KeyboardInterrupt()
        raise SystemExit(128 + signum)

    try:
        # Signals are delivered to Python's main thread. A worker invoked from
        # another thread cannot install handlers and receives no Python signals.
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, record_termination_signal)
        acquire_inputs(config, store, root)
        verified_inputs(root, config["files"])
        reason = stop_reason(config, root, time.time())
        if reason:
            raise LifecycleError(reason)
        environment = os.environ.copy()
        if config.get("resume"):
            snapshot = config["resume"]
            committed = json.loads(store.get(snapshot))
            if committed != snapshot["commit"]:
                raise LifecycleError("resume commit mismatch")
            restored = recover(store, snapshot, root / "resume")
            manifests = list(restored.glob("*.json"))
            if len(manifests) != 1:
                raise LifecycleError("checkpoint must have one JSON manifest")
            environment["COLABCOLA_RESUME_MANIFEST"] = str(manifests[0])
            status["restored_step"] = snapshot["commit"]["step"]
        reason = stop_reason(config, root, time.time())
        if reason:
            raise LifecycleError(reason)
        runtime_root = Path(config.get("content_root", "/content"))
        job_root = runtime_root / "colabcola/jobs" / config["job_id"]
        result_root = runtime_root / "colabcola/results" / config["job_id"]
        with (root / "execution.log").open("ab") as log:
            process = subprocess.Popen([sys.executable, str(root / "lifecycle_worker.py"),
                                        str(config_path), "--execute"],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       env=environment, start_new_session=True)
        watchdog = ExecutionWatchdog(config, root, process)
        watchdog.start()
        status["state"] = "running"
        status["execution_pid"] = process.pid
        completed = CompletedArtifacts(config, job_root, root)
        last_checkpoint, last_logs, last_log_hash, failures = -1, 0, None, 0
        while True:
            hardware_path = result_root / "hardware.json"
            if hardware_path.is_file():
                hardware = json.loads(read_regular(result_root, "hardware.json", 65536))
                if hardware.get("job_id") == config["job_id"]:
                    status["hardware"] = hardware
                    atomic(root / "status.json", status)
            if watchdog.reason:
                status["reason"] = watchdog.reason
            if time.time() >= config["deadline"] and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                status["reason"] = "wall_budget_exhausted"
            finished = process.poll() is not None
            try:
                artifact_snapshot = completed.publish(store) if job_root.exists() else None
                if artifact_snapshot:
                    if evidence:
                        evidence.published(root, "artifacts")
                    status["artifacts_commit"] = {k: artifact_snapshot[k] for k in ("id", "sha256", "bytes")}
                    status["completed_artifact_groups"] = len(completed.markers)
                    status["completed_artifact_files"] = len(completed.members)
                    # Persist the exact receipt before slower log/checkpoint transfers.
                    atomic(root / "status.json", status)
                candidates = checkpoint_members(job_root, config["checkpoint_pattern"]) if job_root.exists() else []
                if candidates and candidates[-1][0] > last_checkpoint:
                    step, members = candidates[-1]
                    publisher = publish_files if all(isinstance(v, Path) for v in members.values()) else publish
                    snapshot = publisher(store, config, "checkpoint", members,
                                       step=step, attempt=config["attempt"])
                    if evidence:
                        evidence.published(root, "checkpoint")
                    status["checkpoint_step"] = last_checkpoint = step
                    status["checkpoint_commit"] = {k: snapshot[k] for k in ("id", "sha256", "bytes")}
                if time.time() - last_logs >= 5 or finished:
                    logs = log_members(result_root, job_root, config["log_patterns"])
                    log_hash = digest(encoded({n: digest(b) for n, b in logs.items()}))
                    if logs and log_hash != last_log_hash:
                        snapshot = publish(store, config, "logs", logs, step=max(0, last_checkpoint),
                                           attempt=config["attempt"])
                        status["logs_commit"] = {k: snapshot[k] for k in ("id", "sha256", "bytes")}
                        if evidence:
                            evidence.published(root, "logs")
                        last_log_hash = log_hash
                    last_logs = time.time()
                failures = 0
                status.pop("persistence_error", None)
                if finished:
                    members = {name: safe_file(result_root, name)
                               for name in ("result.json", "artifacts.zip")}
                    # Bind terminal logs to the final commit too: fast failures may finish
                    # before the host polls a periodic log snapshot.
                    for name, body in logs.items():
                        path = root / "final-logs" / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(body)
                        members[name] = path
                    result = json.loads(members["result.json"].read_text())
                    atomic(root / "attempt.json", {
                        "attempt": config["attempt"], "restored_step": status.get("restored_step", 0),
                        "resume_sha256": (config.get("resume") or {}).get("sha256"),
                        "checkpoint_step": max(0, last_checkpoint),
                    })
                    members["attempt.json"] = root / "attempt.json"
                    if hardware_path.is_file():
                        members["hardware.json"] = hardware_path
                    if evidence:
                        atomic(root / "authorization.json", evidence.value)
                        members["authorization.json"] = root / "authorization.json"
                    final = publish_files(store, config, "final", members,
                                    step=max(0, last_checkpoint), attempt=config["attempt"])
                    if evidence:
                        evidence.published(root, "final")
                        status["authorization"] = dict(evidence.value)
                    status.update(state="completed" if result["terminal_state"] == "success" else "failed",
                                  final_commit={k: final[k] for k in ("id", "sha256", "bytes")})
                    atomic(root / "status.json", status)
                    return
            except Exception as exc:
                failures += 1
                # Deliberately avoid serializing exception messages containing credentials/URLs.
                status["persistence_error"] = type(exc).__name__
                if failures >= 3:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    status["state"] = "persistence_failed"
                    atomic(root / "status.json", status)
                    return
            if evidence:
                status["authorization"] = dict(evidence.value)
            status["heartbeat_at"] = time.time()
            atomic(root / "status.json", status)
            time.sleep(2 if finished else config.get("poll_seconds", 5))
    except BaseException as exc:
        # Includes KeyboardInterrupt and SystemExit, not just Exception.
        # Stop the owned session before publishing a failed status or re-raising.
        _stop_execution_child(process)
        failure_status = {**status, "state": "failed",
                          "reason": type(exc).__name__}
        if received_signal is not None:
            failure_status["termination_signal"] = received_signal
        try:
            atomic(root / "status.json", failure_status)
        except Exception:
            # Preserve the original BaseException for the tracked kernel receipt.
            if isinstance(exc, Exception):
                raise
        if not isinstance(exc, Exception):
            raise
    finally:
        try:
            # Also cover normal return paths and failures during status writing.
            _stop_execution_child(process)
        finally:
            try:
                if watchdog:
                    watchdog.stop()
            finally:
                try:
                    for signum, handler in previous_handlers.items():
                        signal.signal(signum, handler)
                finally:
                    lock.close()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[2] == "--execute":
        execute_notebook(sys.argv[1])
    else:
        run(sys.argv[1])
