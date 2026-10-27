"""Durable, bounded supervisor for the first unattended Colab lifecycle."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
from datetime import timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import zipfile

from .lifecycle_auth import authorization_health
from .lifecycle_io import (DriveStore, LifecycleError, AllocationRejected, atomic, digest, encoded,
                           publish, read_regular, recover, relative, snapshots, publish_files, verified_inputs, INPUT_BYTES, ARTIFACT_BYTES, RESULT_METADATA_BYTES)

MAX_SECONDS = 86400

TERMINAL = {"completed", "failed", "cancelled", "budget_exhausted"}


@contextlib.contextmanager
def run_lock(directory):
    path = Path(directory) / "supervisor.lock"
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise LifecycleError("another supervisor owns this run") from None
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise LifecycleError("another supervisor owns this run") from None
        yield


def get_token(command):
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
        if result.returncode or len(result.stdout) > 32000:
            raise ValueError()
        token = json.loads(result.stdout)
        if not isinstance(token.get("access_token"), str) or token["expires_at"] <= time.time() + 120:
            raise ValueError()
        return token
    except Exception:
        raise LifecycleError("Drive authorization unavailable; initialize the host credential store") from None


def verify_final(root, job_id):
    result = json.loads(read_regular(root, "result.json"))
    if result.get("job_id") != job_id:
        raise LifecycleError("result job identity mismatch")
    with zipfile.ZipFile(Path(root) / "artifacts.zip") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) > 4096:
            raise LifecycleError("invalid result archive members")
        output_size = sum(item.file_size for item in archive.infolist() if item.filename != "result.json")
        if output_size > ARTIFACT_BYTES or archive.getinfo("result.json").file_size > RESULT_METADATA_BYTES:
            raise LifecycleError("result exceeds 5 GiB outputs or 64 MiB metadata bound")
        for item in archive.infolist():
            relative(item.filename)
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise LifecycleError("result archive symlink rejected")
        if json.loads(archive.read("result.json")) != result:
            raise LifecycleError("result metadata differs inside archive")
        expected = {"result.json"}
        for item in result["outputs"]:
            name = relative(item["path"])
            with archive.open(name) as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
            if archive.getinfo(name).file_size != item["size"] or sha != item["sha256"]:
                raise LifecycleError("result output integrity failure")
            expected.add(name)
        if set(names) != expected:
            raise LifecycleError("undeclared result archive member")
    return result


def reported_snapshot(store, plan, attempt, reference, kind):
    """Resolve a worker-reported commit without relying on Drive listing visibility."""
    if (not isinstance(reference, dict) or not isinstance(reference.get("id"), str)
            or not reference["id"] or not all(c.isalnum() or c in "_-" for c in reference["id"])
            or type(reference.get("bytes")) is not int or not 0 <= reference["bytes"] <= 64 * 1024 * 1024
            or not isinstance(reference.get("sha256"), str) or len(reference["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in reference["sha256"])):
        raise LifecycleError("worker " + kind + " commit reference invalid")
    commit = json.loads(store.get(reference))
    if (commit.get("schema") not in (1, 2) or commit.get("kind") != kind
            or commit.get("run_id") != plan["run_id"] or commit.get("job_id") != plan["job_id"]
            or type(commit.get("attempt")) is not int or commit["attempt"] != attempt["number"]
            or type(commit.get("step")) is not int or commit["step"] < 0):
        raise LifecycleError("worker " + kind + " commit identity mismatch")
    return {**reference, "commit": commit}


def reported_final(store, plan, attempt, reference):
    return reported_snapshot(store, plan, attempt, reference, "final")


def recover_progress(engine, attempt, *, force_catalog=False, terminal=False):
    """Recover current-attempt progress by exact ref or a fresh, scoped catalogue."""
    import fnmatch
    worker = engine.state.get("worker", {})
    for kind, field in (("logs", "logs"), ("artifacts", "artifacts")):
        reference = worker.get(kind + "_commit") if worker.get("attempt") == attempt["number"] else None
        if kind == "artifacts" and not (engine.plan.get("artifact_groups") or reference or engine.state.get(field)):
            continue
        previous = engine.state.get(field, {})
        if force_catalog or not reference:
            candidates = [v for v in snapshots(engine.store, engine.plan, kind)
                          if v["commit"]["attempt"] == attempt["number"]]
            if reference and reference.get("id") not in {v["id"] for v in candidates} and reference.get("id") != previous.get("id"):
                candidates.append(reported_snapshot(engine.store, engine.plan, attempt, reference, kind))
        elif previous.get("id") and all(previous.get(k) == reference.get(k) for k in ("id", "sha256", "bytes")):
            candidates = []
        else:
            candidates = [reported_snapshot(engine.store, engine.plan, attempt, reference, kind)]
        if not candidates:
            continue
        latest = candidates[-1]
        if previous.get("sha256") == latest["sha256"]:
            continue
        if kind == "artifacts":
            members = latest["commit"].get("members", {})
            if latest["commit"].get("schema") != 2 or sum(m["bytes"] for m in members.values()) > ARTIFACT_BYTES:
                raise LifecycleError("completed artifact snapshot exceeds5GiB or is not chunked")
            rules = engine.plan.get("artifact_output_rules", [])
            if any(not n.startswith("outputs/") or not any(fnmatch.fnmatchcase(n, r["pattern"]) for r in rules) for n in members):
                raise LifecycleError("completed artifact snapshot is outside declared outputs")
            for rule in rules:
                if sum(v["bytes"] for n, v in members.items() if fnmatch.fnmatchcase(n, rule["pattern"])) > rule["maximum_bytes"]:
                    raise LifecycleError("completed artifact snapshot exceeds declared output bound")
        root = recover(engine.store, latest, engine.directory / field / latest["sha256"], previous=previous)
        event = ("late_" if terminal else "") + ("logs_verified" if kind == "logs" else "completed_artifacts_verified")
        if not terminal and engine.state["phase"] not in TERMINAL and "missing_since" in attempt:
            attempt["last_progress_recovered_at"] = engine.clock()
        engine.save(event, **{field: {**latest, "local_directory": str(root)}})


def recover_terminal_published(engine):
    """Read-only Drive recovery; retain terminal outcome and consumed attempt bounds."""
    if engine.state["phase"] not in TERMINAL or engine.state.get("final") or not engine.state.get("attempts"):
        return engine.state
    attempt = engine.state["attempts"][-1]
    recover_progress(engine, attempt, force_catalog=True, terminal=True)
    reference = engine.state.get("worker", {}).get("final_commit")
    finals = [f for f in snapshots(engine.store, engine.plan, "final") if f["commit"]["attempt"] == attempt["number"]]
    if reference and reference.get("id") not in {f["id"] for f in finals}:
        finals.append(reported_final(engine.store, engine.plan, attempt, reference))
    if finals:
        final = finals[-1]
        root = recover(engine.store, final, engine.directory / "artifacts" / final["sha256"])
        verify_final(root, engine.plan["job_id"])
        engine.save("late_final_artifacts_verified", final={**final, "local_directory": str(root)})
    return engine.state


def recover_terminal_final(directory, *, store=None, observer=None):
    """Recover late logs/completed outputs/final without constructing a provider."""
    directory = Path(directory)
    with run_lock(directory):
        state = json.loads((directory / "state.json").read_text())
        if state["phase"] not in TERMINAL or state.get("final") or not state.get("attempts"):
            return state
        plan = json.loads((directory / "plan.json").read_text())
        token = lambda: get_token(plan["token_command"])
        store = store if store is not None else DriveStore(plan["parent"], plan["run_id"], lambda: token()["access_token"])
        engine = Supervisor(directory, None, store, token, observer=observer)
        return recover_terminal_published(engine)


class Supervisor:
    def __init__(self, directory, provider, store, token, *, clock=time.time, observer=None):
        self.directory = Path(directory)
        self.plan = json.loads((self.directory / "plan.json").read_text())
        self.state = json.loads((self.directory / "state.json").read_text())
        self.provider, self.store, self.token, self.clock = provider, store, token, clock
        self.observer = observer

    def save(self, event=None, **fields):
        self.state.update(fields)
        if event:
            self.state["events"].append({"sequence": len(self.state["events"]) + 1,
                                         "time": self.clock(), "event": event,
                                         "phase": self.state["phase"]})
        atomic(self.directory / "state.json", self.state)
        # Journal is the authority; regenerating this projection repairs a crash between writes.
        temporary = self.directory / "events.ndjson.tmp"
        temporary.write_bytes(b"".join(encoded(row) for row in self.state["events"]))
        os.replace(temporary, self.directory / "events.ndjson")
        if self.observer:
            # Verified logs reach the CLI before a later provider request can stall.
            self.observer(self.state)

    def finish(self, outcome, reason=None):
        self.save("cleanup_requested", phase="cleanup", outcome=outcome, reason=reason)
        self.cleanup()

    def cleanup(self):
        for attempt in self.state["attempts"]:
            endpoint = attempt.get("endpoint")
            if not endpoint and not attempt.get("released"):
                # Allocation may have succeeded just before the supervisor died.
                endpoint = self.provider.reconcile(attempt["allocation_id"], self.plan["gpu"])
                if endpoint:
                    attempt["endpoint"] = endpoint
                    self.save("uncertain_allocation_reconciled")
                else:
                    attempt["released"] = True
            if endpoint and not attempt.get("released"):
                self.provider.release(endpoint)
                attempt["released"] = True
                self.save("owned_runtime_released")
        self.save("run_finished", phase=self.state["outcome"])

    def runtime_absent(self, attempt, *, booting=False):
        now = self.clock()
        attempt.setdefault("missing_since", now)
        attempt["missing_observations"] = attempt.get("missing_observations", 0) + 1
        self.save("bootstrap_runtime_absent" if booting else "runtime_absence_observed")
        endpoint = self.provider.reconcile(attempt["allocation_id"], self.plan["gpu"])
        if endpoint is not None:
            if endpoint != attempt["endpoint"]:
                raise LifecycleError("allocation lookup returned an unexpected endpoint;no replacement allowed")
            # A present direct assignment can coexist with a stale list and worker
            # ref. Keep recovering Drive progress even while status access is lost.
            recover_progress(self, attempt, force_catalog=True)
            attempt.pop("missing_since", None)
            attempt["missing_observations"] = 0
            self.save("assignment_listing_inconsistent")
            return
        before = tuple(self.state.get(k, {}).get("sha256") for k in ("logs", "artifacts"))
        recover_progress(self, attempt, force_catalog=True)
        after = tuple(self.state.get(k, {}).get("sha256") for k in ("logs", "artifacts"))
        if after != before:
            attempt["last_progress_recovered_at"] = now
        grace = self.plan.get("loss_grace_seconds", 90)
        if self.state.get("fault_pending") or self.state.get("fault_injected"):
            grace = 0  # Deliberate proof stop is already an owned release request.
        since = max(attempt["missing_since"], attempt.get("last_progress_recovered_at", 0))
        if attempt["missing_observations"] < 2 or now - since < grace:
            self.save("runtime_loss_grace", loss_diagnostics={"attempt": attempt["number"],
                "missing_since": attempt["missing_since"], "direct_assignment_absent": True,
                "grace_seconds": grace, "last_worker_heartbeat": self.state.get("worker", {}).get("heartbeat_at")})
            return
        # Never mark released solely from two list misses. Exact-owned release must
        # succeed (or return an authenticated404),then direct lookup must agree.
        release = self.provider.release(attempt["endpoint"])
        if self.provider.reconcile(attempt["allocation_id"], self.plan["gpu"]) is not None:
            raise LifecycleError("owned runtime absence is not confirmed after release")
        attempt["released"] = True
        if isinstance(release, dict):
            attempt["release_evidence"] = release
        if self.state.get("fault_pending"):
            self.save("proof_fault_confirmed", fault_injected=True, fault_pending=False)
        self.save("runtime_loss_confirmed", phase="recovering", last_failure={
            "cause": "runtime_lost", "attempt": attempt["number"], "confirmed_at": self.clock(),
            "missing_since": attempt["missing_since"], "termination_cause": "unknown"})

    def new_attempt(self):
        if len(self.state["attempts"]) >= self.plan["max_attempts"]:
            self.finish("failed", "runtime_lost_attempt_budget_exhausted" if self.state.get("last_failure") else "attempt_budget_exhausted")
            return
        checkpoint = self.state.get("checkpoint")
        if len(self.state["attempts"]) == 1 and self.state.get("fault_injected"):
            # The proof must resume the checkpoint verified before deliberate loss.
            # Later publications from the dying worker must not change that contract.
            checkpoint = self.state.get("proof_checkpoint", checkpoint)
        attempt = {"number": len(self.state["attempts"]) + 1,
                   "allocation_id": str(uuid.uuid4()), "created_at": self.clock(),
                   "deadline": self.state["deadline"], "resume": checkpoint}
        self.state["attempts"].append(attempt)
        self.save("allocation_intent", phase="allocating")

    def tick(self):
        phase = self.state["phase"]
        if phase in TERMINAL:
            return
        if phase == "cleanup":
            self.cleanup()
            return
        if (self.directory / "cancel.request").exists():
            self.finish("cancelled")
            return
        if self.state.get("deadline") is not None and self.clock() >= self.state["deadline"]:
            self.finish("budget_exhausted", "wall_budget_exhausted")
            return
        if (phase == "monitoring" and self.plan.get("authorization_policy") == "renewable-v1"
                and self.state.get("drive_authorization", {}).get("expires_at", float("inf")) <= self.clock() + 120):
            self.finish("failed", "drive_authorization_expiring")
            return
        if phase == "prepared":
            self.save("staging_started", phase="staging",
                      deadline=self.clock() + self.plan["max_seconds"])
        if self.state["phase"] == "staging":
            paths = verified_inputs(self.directory, self.plan["files"])
            if self.plan.get("input_transport") == "drive_chunks":
                self.state["input_commit"] = publish_files(self.store, self.plan, "input", paths)
            else:
                inputs = {name: path.read_bytes() for name, path in paths.items()}
                self.state["input_commit"] = publish(self.store, self.plan, "input", inputs)
            self.save("inputs_verified_on_drive")
            self.new_attempt()
            return
        attempt = self.state["attempts"][-1]
        if phase == "allocating":
            endpoint = self.provider.allocate(attempt["allocation_id"], self.plan["gpu"])
            if endpoint in [a.get("endpoint") for a in self.state["attempts"][:-1]]:
                self.finish("failed", "provider_reused_previous_runtime")
                return
            attempt["endpoint"] = endpoint
            attempt["allocated_at"] = self.clock()
            self.save("runtime_allocated", phase="booting")
            return
        if phase == "booting":
            if self.provider.lookup(attempt["endpoint"]) is None:
                self.runtime_absent(attempt, booting=True)
                return
            attempt.pop("missing_since", None)
            attempt["missing_observations"] = 0
            boot_attempt = dict(attempt)
            if self.plan.get("input_transport") == "drive_chunks":
                boot_attempt["input_snapshot"] = self.state["input_commit"]
            access = self.token()
            self.provider.boot(attempt["endpoint"], self.plan, boot_attempt, access)
            if self.plan.get("authorization_policy") == "renewable-v1":
                self.state["drive_authorization"] = {"state": "bootstrapped", "attempt": attempt["number"],
                                                      "expires_at": access["expires_at"]}
            attempt["bootstrapped_at"] = self.clock()
            self.save("runtime_bootstrapped", phase="monitoring")
            return
        if phase == "recovering":
            self.new_attempt()
            return
        if phase != "monitoring":
            raise LifecycleError("unknown supervisor phase")

        worker = self.state.get("worker", {})
        reference = worker.get("final_commit") if worker.get("attempt") == attempt["number"] else None
        finals = ([reported_final(self.store, self.plan, attempt, reference)] if reference
                  else snapshots(self.store, self.plan, "final"))
        finals = [f for f in finals if f["commit"]["attempt"] == attempt["number"]]
        if attempt["number"] == 1 and (self.state.get("fault_pending")
                                         or self.state.get("fault_injected")):
            # A late terminal commit cannot finish a proof whose restart is underway.
            finals = []
        if finals:
            final = finals[-1]
            root = recover(self.store, final, self.directory / "artifacts" / final["sha256"])
            result = verify_final(root, self.plan["job_id"])
            self.state["final"] = {**final, "local_directory": str(root)}
            if (root / "hardware.json").is_file():
                hardware = json.loads(read_regular(root,"hardware.json",65536))
                if hardware.get("job_id") != self.plan["job_id"]:
                    raise LifecycleError("hardware report job identity mismatch")
                self.state["hardware"] = hardware
            self.save("final_artifacts_verified")
            if self.plan["fault_after_checkpoint"] and result["terminal_state"] == "success":
                evidence = json.loads(read_regular(root, "attempt.json")) if (root / "attempt.json").exists() else {}
                resume = attempt.get("resume") or {}
                restored = evidence.get("restored_step", 0)
                if not (self.state.get("fault_injected") and attempt["number"] > 1
                        and final["commit"]["attempt"] == attempt["number"]
                        and evidence.get("attempt") == attempt["number"]
                        and evidence.get("resume_sha256") == resume.get("sha256")
                        and restored > 0 and restored == resume.get("commit", {}).get("step")
                        and final["commit"]["step"] > restored):
                    self.finish("failed", "recovery_proof_not_demonstrated")
                    return
                self.save("recovery_proof_verified", restored_step=restored)
            self.finish("completed" if result["terminal_state"] == "success" else "failed",
                        result["terminal_state"])
            return
        checkpoints = snapshots(self.store, self.plan, "checkpoint")
        if checkpoints:
            checkpoint = checkpoints[-1]
            if self.state.get("checkpoint", {}).get("sha256") != checkpoint["sha256"]:
                recover(self.store, checkpoint, self.directory / "checkpoints" / checkpoint["sha256"])
                self.save("checkpoint_verified", checkpoint=checkpoint)
        recover_progress(self, attempt)
        endpoint = attempt["endpoint"]
        if self.provider.lookup(endpoint) is None:
            self.runtime_absent(attempt)
            return
        attempt.pop("missing_since", None)
        attempt["missing_observations"] = 0
        if (self.plan["fault_after_checkpoint"] and not self.state.get("fault_injected")
                and self.state.get("checkpoint") and attempt["number"] == 1):
            # Durable intent makes a timed-out stop retry safe; only this exact owned endpoint.
            self.save("proof_fault_requested", fault_pending=True,
                      proof_checkpoint=self.state.get("proof_checkpoint") or self.state["checkpoint"])
            self.provider.release(endpoint)
            self.save("proof_fault_injected", fault_injected=True, fault_pending=False)
            return
        observed = self.provider.observe(endpoint, self.plan, attempt,
            None if self.plan.get("authorization_policy") == "renewable-v1" else self.token())
        self.state["worker"] = observed
        if observed.get("hardware"):
            self.state["hardware"] = observed["hardware"]
        if self.plan.get("authorization_policy") == "renewable-v1":
            self.state["drive_authorization"] = authorization_health(self.state, now=self.clock())
        if observed.get("final_commit"):
            # A list request above can race publication. Persist the exact ref first;
            # the next tick recovers it even if Drive search has not indexed it yet.
            self.save("terminal_commit_reported")
        elif observed.get("state") in {"failed", "persistence_failed"}:
            self.finish("failed", "runtime_" + observed["state"])
        elif (observed.get("state") == "starting"
              and self.clock() - attempt.get("bootstrapped_at", attempt["allocated_at"]) > 180):
            self.finish("failed", "worker_start_timeout")
        else:
            self.save()

    def advance(self):
        try:
            self.tick()
            if self.state.pop("connectivity", None):
                self.state.pop("attention", None)
                self.save("connection_restored")
        except AllocationRejected as exc:
            try:
                self.finish("failed", str(exc))
            except Exception as cleanup_error:
                self.save("attention_required", connectivity="unknown",
                          attention="cleanup pending: " + type(cleanup_error).__name__)
        except LifecycleError as exc:
            # Integrity/configuration/auth errors are actionable, but never silently release
            # an owned runtime based only on inability to communicate with it.
            self.save("attention_required", connectivity="unknown", attention=str(exc))
        except Exception as exc:
            if self.state.get("connectivity") != "unknown":
                self.save("connection_unknown", connectivity="unknown", attention=type(exc).__name__)


def prepare(args):
    from .compiler import compile_invocation
    from .invocation import load_job_invocation
    invocation = load_job_invocation(args.job, getattr(args, "job_args", ()))
    spec = invocation.spec
    backend = getattr(args, "provider_backend", "legacy")
    high_mem = getattr(args, "high_mem", False)
    if backend not in ("legacy", "rest") or type(high_mem) is not bool:
        raise LifecycleError("invalid provider backend or high-memory setting")
    if spec.runtime.version != "latest" and backend != "rest":
        raise LifecycleError("Colab CLI adapter supports runtime.version=latest only; dated selection is unavailable on this path (beta REST requires separate allowlisted access)")
    if spec.managed_recovery is None or spec.durable_checkpoints is not None:
        raise LifecycleError("lifecycle requires managed_recovery and cannot mix the older checkpoint publisher")
    if (spec.resources.accelerator == "cpu") != (args.gpu == "NONE"):
        raise LifecycleError("GPU selection conflicts with JobSpec resources")
    if (type(args.max_attempts) is not int or type(args.max_seconds) is not int
            or not 1 <= args.max_attempts <= 3 or not 60 <= args.max_seconds <= MAX_SECONDS):
        raise LifecycleError("execution limits: 1-3 attempts and 60-86400 total seconds")
    command = json.loads(Path(args.token_command).read_text())
    if not isinstance(command, list) or not command or not all(isinstance(v, str) for v in command):
        raise LifecycleError("token command must be a JSON argv array without credentials")
    directory = Path(args.run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    compiled = compile_invocation(invocation, directory / "notebook.ipynb")
    for name in ("lifecycle_worker.py", "lifecycle_io.py"):
        (directory / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    files = {name: digest(read_regular(directory, name, INPUT_BYTES)) for name in
             ("notebook.ipynb", "lifecycle_worker.py", "lifecycle_io.py")}
    input_size = sum((directory / name).stat().st_size for name in files)
    verified_inputs(directory, files)
    plan = {"schema": 1, "run_id": uuid.uuid4().hex, "job_id": compiled.job_id,
            "directory": str(directory), "parent": args.drive_parent, "gpu": args.gpu,
            "max_attempts": args.max_attempts, "max_seconds": args.max_seconds,
            "provider_backend": backend, "runtime_version": spec.runtime.version, "high_mem": high_mem,
            "authorization_policy": "renewable-v1",
            "fault_after_checkpoint": args.fault_after_checkpoint, "token_command": command,
            "checkpoint_pattern": spec.managed_recovery.checkpoint_pattern,
            "log_patterns": list(spec.managed_recovery.log_patterns), "files": files,
            "loss_grace_seconds": 90, "hardware_policy": "gpu-model-v1"}
    if backend == "legacy":
        plan["execution_mode"] = "kernel-v1"
    if spec.managed_recovery.artifact_groups:
        plan["artifact_groups"] = [g.mapping() for g in spec.managed_recovery.artifact_groups]
        plan["artifact_output_rules"] = [{"pattern": o.pattern, "maximum_bytes": o.maximum_bytes} for o in spec.outputs]
    if input_size > 32 * 1024 * 1024:
        plan["input_transport"] = "drive_chunks"
    DriveStore(plan["parent"], plan["run_id"], lambda: "")  # Local validation only.
    atomic(directory / "plan.json", plan)
    atomic(directory / "state.json", {"phase": "prepared", "attempts": [], "events": []})
    return {"state": "prepared", "run_dir": str(directory), "approval_sha256": digest(encoded(plan)),
            "job_id": compiled.job_id, "limits": {"attempts": args.max_attempts, "seconds": args.max_seconds}}


def supervise(directory, approve, observer=None, *, preflight_only=False, validator=None, allow_active_assignments=()):
    from .lifecycle_provider import provider_for_plan
    from .lifecycle_auth import TokenLease, RenewalPump
    from .preflight import run_preflight
    directory = Path(directory).resolve()
    with run_lock(directory):
        try:
            state = json.loads((directory / "state.json").read_text())
        except (OSError, ValueError):
            # Preserve an unreadable journal rather than resetting it.
            return run_preflight(directory, approve, close_provider=True)[0]
        provider = None
        if state["phase"] == "prepared" or preflight_only:
            report, provider, token = run_preflight(directory, approve, validator=validator,
                allow_active_assignments=allow_active_assignments, observer=observer, close_provider=preflight_only)
            if preflight_only:
                return report
        plan = json.loads((directory / "plan.json").read_text())
        if digest(encoded(plan)) != approve or str(directory) != plan["directory"]:
            raise LifecycleError("approval or run directory differs from prepared plan")
        renewable = plan.get("authorization_policy") == "renewable-v1"
        if state["phase"] in TERMINAL:
            return state
        if provider is None:
            if validator:
                validator(plan)
            token = TokenLease(lambda: get_token(plan["token_command"])) if renewable else lambda: get_token(plan["token_command"])
            provider = provider_for_plan(plan)
        engine = None
        try:
            store = DriveStore(plan["parent"], plan["run_id"], lambda: token()["access_token"])
            engine = Supervisor(directory, provider, store, token, observer=observer)
            if observer:
                observer(engine.state)
            context = RenewalPump(engine, provider, token) if renewable else contextlib.nullcontext()
            with context as renewal:
                while engine.state["phase"] not in TERMINAL:
                    if renewal:
                        engine.state["drive_authorization"] = authorization_health(engine.state, renewal.health)
                        engine.save()
                    engine.advance()
                    if engine.state["phase"] not in TERMINAL:
                        time.sleep(5)
            if not engine.state.get("final") and time.time() < engine.state.get("deadline", 0):
                try:
                    recover_terminal_published(engine)
                except Exception as exc:
                    engine.save("late_recovery_pending", recovery_attention=type(exc).__name__)
            return engine.state
        finally:
            try:
                provider.close()
            except Exception:
                if engine is not None:
                    engine.save("kernel_connection_cleanup_pending", connection_cleanup="client_close_failed")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="colabcola lifecycle")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("login")
    commands.add_parser("doctor")
    commands.add_parser("drive-token", help=argparse.SUPPRESS)
    p = commands.add_parser("prepare")
    p.add_argument("job")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--drive-parent", required=True)
    p.add_argument("--token-command", required=True)
    p.add_argument("--gpu", choices=("NONE", "T4", "L4", "A100", "H100", "G4"), default="T4")
    p.add_argument("--provider-backend", choices=("legacy", "rest"), default="legacy")
    p.add_argument("--high-mem", action="store_true")
    p.add_argument("--max-attempts", type=int, default=2)
    p.add_argument("--max-seconds", type=int, default=900)
    p.add_argument("--fault-after-checkpoint", action="store_true")
    for name in ("start", "supervise"):
        p = commands.add_parser(name)
        p.add_argument("run_dir")
        p.add_argument("--approve", required=True)
        p.add_argument("--allow-active-assignment", action="append", default=[])
        if name == "supervise":
            p.add_argument("--preflight-only", action="store_true")
    for name in ("status", "logs", "cancel"):
        commands.add_parser(name).add_argument("run_dir")
    args = parser.parse_args(argv)
    try:
        if args.command == "login":
            from .lifecycle_provider import login
            login()
            return 0
        if args.command == "drive-token":
            from .google_drive import _desktop_credentials, load_google_drive_settings
            credential = _desktop_credentials(load_google_drive_settings(), force_authorize=False, interactive=False, minimum_lifetime_seconds=1860)
            expiry = credential.expiry.replace(tzinfo=timezone.utc).timestamp()
            print(json.dumps({"access_token": credential.token, "expires_at": expiry}))
            return 0
        if args.command == "doctor":
            from .lifecycle_provider import dependency_report
            result = {**dependency_report(), "platform": sys.platform,
                      "host_supported": sys.platform.startswith("linux") or sys.platform == "darwin",
                      "runtime_selection": "legacy (primary Colab CLI): latest; rest: deferred beta, requires allowlist",
                      "primary_provider_backend": "legacy", "live_provider_verified": False,
                      "credentials_checked": False, "network_accessed": False}
        elif args.command == "prepare":
            result = prepare(args)
        elif args.command == "start":
            root = Path(args.run_dir).resolve()
            plan = json.loads((root / "plan.json").read_text())
            if digest(encoded(plan)) != args.approve:
                raise LifecycleError("approval does not match prepared plan")
            kwargs = {"start_new_session": True} if os.name != "nt" else {
                "creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
            with (root / "supervisor.log").open("ab") as log:
                process = subprocess.Popen([sys.executable, "-m", "colabcola", "lifecycle", "supervise",
                                            str(root), "--approve", args.approve,
                                            *[v for endpoint in args.allow_active_assignment for v in ("--allow-active-assignment", endpoint)]],
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=log, **kwargs)
            result = {"state": "supervisor_spawned", "pid": process.pid, "run_dir": str(root)}
        elif args.command == "supervise":
            result = supervise(args.run_dir, args.approve, preflight_only=args.preflight_only,
                               allow_active_assignments=args.allow_active_assignment)
        elif args.command == "cancel":
            root = Path(args.run_dir)
            atomic(root / "cancel.request", {"at": time.time()})
            result = {"state": "cancellation_requested", "note": "supervisor must be running to release owned compute"}
        else:
            state = json.loads((Path(args.run_dir) / "state.json").read_text())
            if args.command == "logs":
                for event in state["events"]:
                    print(json.dumps(event, sort_keys=True))
                return 0
            result = state
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        # Credential-bearing provider errors must never leak request representations.
        message = str(exc) if isinstance(exc, LifecycleError) else type(exc).__name__
        print(json.dumps({"state": "blocked", "reason": message}), file=sys.stderr)
        return 2
