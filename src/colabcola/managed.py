"""Single-command managed submission with durable journals and readable progress."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile
from .lifecycle import TERMINAL, prepare, supervise, verify_final, recover_terminal_final
from .lifecycle_io import (LifecycleError, atomic, digest, encoded, read_regular, verified_inputs,
                           safe_file, ARTIFACT_BYTES, MAX_MEMBERS)


# Console-only presentation. Journal event names and machine-readable JSON are unchanged.
_EVENT_LABELS = {
    "staging_started": "Upload: preparing input bundle",
    "inputs_verified_on_drive": "Upload: inputs verified on Drive",
    "allocation_intent": "Runtime: requesting allocation",
    "runtime_allocated": "Runtime: allocated",
    "runtime_bootstrapped": "Runtime: worker started",
    "logs_verified": "Logs: verified snapshot available",
    "final_artifacts_verified": "Artifacts: verified from Drive",
    "cleanup_requested": "Cleanup: releasing owned runtime",
    "owned_runtime_released": "Cleanup: owned runtime released",
    "run_finished": "Run: finished",
}


class Progress:
    def __init__(self, directory, stream=None):
        self.directory = Path(directory)
        self.stream = stream or sys.stderr
        self.sequence = 0
        self.last_progress = None
        self.log_positions = {}
        self.existing_run = False
        self.preflight_checks = []
        # Rich is presentation-only, and never writes to machine-readable stdout.
        # Use plain text for redirected output, CI, and tests.
        self._rich = None
        if callable(getattr(self.stream, "isatty", None)) and self.stream.isatty():
            try:
                from rich.console import Console
                self._rich = Console(file=self.stream, highlight=False, markup=False)
            except ImportError:
                pass

    def preflight_checked(self, name):
        """Collect successful checks without printing fourteen repetitive lines."""
        self.preflight_checks.append(name)

    def preflight_completed(self, report):
        count = sum(check.get("state") == "passed" for check in report["checks"])
        self.say(f"Preflight: {count}/{len(report['checks'])} checks passed "
                 "(installation, account, assignments, Drive)")

    def view_existing(self, state):
        """A terminal journal is history; only newly recovered snapshots are updates."""
        self.existing_run = True
        self.saved_snapshots = {key: self._snapshot_identity(state.get(key, {}))
                                for key in ("logs", "artifacts", "final")}

    @staticmethod
    def _snapshot_identity(snapshot):
        return snapshot.get("sha256"), snapshot.get("local_directory")

    def show_recovery(self, state):
        labels = {"logs": "additional saved logs", "artifacts": "additional completed outputs",
                  "final": "final artifact bundle"}
        for key, label in labels.items():
            snapshot = state.get(key, {})
            identity = self._snapshot_identity(snapshot)
            if snapshot.get("local_directory") and identity != self.saved_snapshots.get(key):
                self.say("Recovered " + label + " from Drive for this existing run: " + snapshot["local_directory"])
                self.saved_snapshots[key] = identity

    def say(self, message):
        # Workload strings cannot inject terminal escape sequences.
        message = "".join(c if c.isprintable() else " " for c in str(message))
        if self._rich is None:
            print("[colabcola] " + message, file=self.stream, flush=True)
            return
        from rich.text import Text
        line = Text("[colabcola] ", style="dim cyan")
        if ": " in message and not message.startswith("["):
            stage, detail = message.split(": ", 1)
            stage_style = "bold red" if stage in {"Error", "Failure"} else (
                "bold green" if stage == "Result" and detail.startswith("SUCCESS") else "bold")
            line.append(stage + ": ", style=stage_style)
            line.append(detail)
        else:
            line.append(message)
        self._rich.print(line)

    def show_logs(self, snapshot, final=False):
        root = snapshot.get("local_directory")
        if not root:
            return
        commit = snapshot.get("commit", {})
        attempt = commit.get("attempt", 0)
        # Only verified snapshot members; never glob credential/config files.
        names = [n for n in commit.get("members", {}) if n.endswith(".log")]
        names.sort(key=lambda n: (n not in ("runtime/stdout.log", "runtime/stderr.log"), n))
        for name in names[:32]:
            if name in ("stdout.log", "stderr.log") and "runtime/" + name in names:
                continue  # Final snapshots also contain the raw result copies.
            key = (attempt, name.removeprefix("runtime/"))
            try:
                body = read_regular(root, name, 8 * 1024 * 1024)
            except (OSError, ValueError, LifecycleError):
                continue
            old_length, old_hash, cursor = self.log_positions.get(key, (0, digest(b""), 0))
            if len(body) < old_length or digest(body[:old_length]) != old_hash:
                cursor = 0  # File replaced/truncated, including same-size rewrites.
            end = len(body) if final else max(body.rfind(b"\n"), body.rfind(b"\r")) + 1
            if end > cursor:
                text = body[cursor:end].decode("utf-8", errors="replace")
                lines = [line for line in re.split(r"[\r\n]+", text) if line]
                # Keep root errors even when a launcher adds a long wrapper traceback.
                # Everything displayed still goes through the same redaction/bounds.
                selected = set(range(max(0, len(lines) - 20), len(lines)))
                diagnostic = re.compile(r"(?i)(?:[\w.]*Error|Exception):|CUDA out of memory|"
                                        r"out of memory|Traceback \(most recent call last\)")
                roots = [i for i, line in enumerate(lines[:-20]) if diagnostic.search(line)]
                selected.update(roots[:8])
                if len(lines) > len(selected):
                    self.say(f"[{name}] {len(lines) - len(selected)} earlier updates omitted; full log: {root}/{name}")
                for i in sorted(selected):
                    line = lines[i]
                    # Redact conventional credential fields before truncating.
                    line = re.sub(r"(?i)(bearer\s+)\S+", r"\1[redacted]", line)
                    line = re.sub(r"(?i)((?:access_token|refresh_token|client_secret|colab-runtime-proxy-token|authorization)[\"']?\s*[:=]\s*[\"']?)[^\s\"'&,}]+", r"\1[redacted]", line)
                    if line.startswith("colabcola: command "):
                        line = line.removeprefix("colabcola: ")  # Launcher label is redundant here.
                    channel = name.removeprefix("runtime/")
                    self.say(f"[attempt {attempt} | {channel}] " + line[:1500])
            self.log_positions[key] = (len(body), digest(body), max(cursor, end))

    def __call__(self, state):
        if self.existing_run:
            self.show_recovery(state)
            return
        for event in state.get("events", []):
            if event["sequence"] > self.sequence:
                name = event["event"]
                # The journal retains *every* historical preflight, including an
                # earlier refusal before the user explicitly allowed a runtime.
                # A recovered failure must not look like a new launch failure.
                if not name.startswith("preflight_"):
                    self.say(_EVENT_LABELS.get(name, name.replace("_", " ")))
                self.sequence = event["sequence"]
        if state.get("connectivity") == "unknown" and state.get("attention"):
            value = (state.get("connectivity"), state["attention"])
            if value != getattr(self, "attention", None):
                self.say("Needs attention: " + str(value[1]))
                self.attention = value
        auth = state.get("drive_authorization", {})
        delivery = auth.get("delivery", {})
        health = (auth.get("state"), delivery.get("state"), delivery.get("stage"), delivery.get("error_type"))
        if health != getattr(self, "authorization_health", None) and auth.get("state") in ("waiting", "valid", "expiring", "unavailable"):
            if auth["state"] == "waiting":
                message = "Drive access pending runtime bootstrap"
            elif auth["state"] == "valid":
                message = "Drive access valid"
                if delivery.get("state") == "retrying":
                    action = "credential refresh" if delivery.get("stage") == "credentials" else "token delivery"
                    message += f"; {action} retrying ({delivery.get('error_type', 'connection error')})"
            else:
                message = "Drive access " + auth["state"] + "; expiry protection remains active"
            self.say(message)
            self.authorization_health = health
        hardware = state.get("hardware") or state.get("worker", {}).get("hardware")
        if hardware and hardware != getattr(self, "hardware", None):
            names = ", ".join(f"{g['name']} ({g['memory_bytes']} bytes)" for g in hardware.get("gpus", [])) or "no GPU"
            self.say(f"Runtime hardware: requested {hardware.get('requested_gpu')}; received {names}; check {hardware.get('state')}")
            self.hardware = hardware
        artifacts = state.get("artifacts", {})
        if artifacts.get("sha256") and artifacts["sha256"] != getattr(self, "artifact_snapshot", None):
            self.say("Completed outputs protected on Drive and verified locally: " +
                     str(len(artifacts["commit"]["members"])) + " files; " + artifacts["local_directory"])
            self.artifact_snapshot = artifacts["sha256"]
        snapshot = state.get("final") or state.get("logs") or {}
        self.show_logs(snapshot, final=bool(state.get("final")) or state.get("phase") in TERMINAL)
        latest = (state.get("final", {}).get("local_directory") or
                  state.get("logs", {}).get("local_directory"))
        if latest:
            for path in sorted(Path(latest).rglob("progress.json")):
                try:
                    value = json.loads(read_regular(latest, path.relative_to(latest).as_posix(), 1024 * 1024))
                    if not isinstance(value, dict) or not {"completed_fits", "total_fits"} <= value.keys():
                        continue
                except (OSError, ValueError, LifecycleError):
                    continue  # A workload progress display cannot interrupt supervision.
                if value != self.last_progress:
                    if value.get("state") == "completed":
                        self.say(f"Fits {value.get('completed_fits')}/{value.get('total_fits')} complete")
                        self.last_progress = value
                        continue
                    self.say(f"Fits {value.get('completed_fits', '?')}/{value.get('total_fits', '?')} | "
                             f"{value.get('image', '')} depth {value.get('depth', '?')} "
                             f"width {value.get('width', '?')} HOSC/SIREN | "
                             f"step {value.get('step', '?')}/{value.get('steps', '?')}")
                    self.last_progress = value


def _verify_completed_outputs(snapshot, job_id):
    """Recheck saved intermediate outputs before displaying them as verified."""
    root = Path(snapshot["local_directory"])
    commit = snapshot.get("commit", {})
    members = commit.get("members")
    if (root.is_symlink() or commit.get("job_id") != job_id or not isinstance(members, dict)
            or not 1 <= len(members) <= MAX_MEMBERS):
        raise LifecycleError("saved completed-output snapshot is invalid")
    total = 0
    for name, record in members.items():
        size = record.get("bytes")
        if type(size) is not int or size < 0:
            raise LifecycleError("saved completed-output size is invalid")
        total += size
        if total > ARTIFACT_BYTES:
            raise LifecycleError("saved completed outputs exceed the artifact limit")
        path = safe_file(root, name)
        if path.stat().st_size != size:
            raise LifecycleError("saved completed-output size differs")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != record.get("sha256"):
                raise LifecycleError("saved completed-output integrity failure")


def _report_result(state, plan, directory, progress, *, existing=False):
    final_directory = state.get("final", {}).get("local_directory")
    intermediate_directory = state.get("artifacts", {}).get("local_directory")
    final_status = "missing"
    terminal_state = None
    if final_directory:
        try:
            result = verify_final(final_directory, plan["job_id"])
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            raise LifecycleError("final artifact bundle could not be verified; inspect the saved result directory") from None
        terminal_state = result.get("terminal_state")
        final_status = "verified"
        progress.say("Artifacts: final bundle verified at " + final_directory)
    else:
        progress.say("Final artifact bundle is not available locally; artifact recovery is incomplete")
    if intermediate_directory:
        _verify_completed_outputs(state["artifacts"], plan["job_id"])
        completed_status = "verified"
        progress.say("Intermediate completed-output commits verified: " + intermediate_directory)
    else:
        completed_status = "not_available" if plan.get("artifact_groups") else "not_configured"
        if completed_status == "not_available":
            progress.say("Artifacts: configured intermediate outputs were not recovered")
    if state.get("checkpoint"):
        progress.say("Verified recovery checkpoint retained on Drive and under " + str(directory / "checkpoints"))
    success = state["phase"] == "completed" and final_status == "verified" and terminal_state == "success"
    if state["phase"] == "completed" and not success:
        progress.say("The journal records completion, but verified successful final artifacts are required for a successful command")
    progress.say("Result: " + ("SUCCESS - verified final artifacts" if success else
               "INCOMPLETE - inspect the journal and saved artifacts"))
    print(json.dumps({"state": state["phase"], "job_id": plan["job_id"], "run_dir": str(directory),
                      "result_directory": final_directory, "completed_artifact_directory": intermediate_directory,
                      "final_artifacts_status": final_status, "completed_outputs_status": completed_status,
                      "final_terminal_state": terminal_state, "viewing_existing_run": existing,
                      "run_success": success, "reason": state.get("reason"), "hardware": state.get("hardware")}, indent=2))
    return 0 if success else 3


def run_managed(command, invocation):
    from .bundle import build_bundle
    job = command.job_file.resolve()
    config_path = (command.execution_config or job.with_suffix(".run.json")).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if isinstance(config, dict) and "profile" in config:
        profile_keys = {"profile", "gpu", "max_attempts", "max_seconds",
                        "fault_after_checkpoint", "provider_backend", "high_mem"}
        if config["profile"] != "default" or config.keys() - profile_keys:
            raise LifecycleError("invalid profile execution settings; do not mix account defaults with explicit credentials")
        from .onboarding import execution_defaults
        defaults = execution_defaults()
        config = {key: value for key, value in config.items() if key != "profile"}
        config.update(drive_parent=defaults["drive_parent"], token_command=str(defaults["token_command"]))
    required = {"drive_parent", "token_command", "gpu", "max_attempts", "max_seconds"}
    allowed = required | {"fault_after_checkpoint", "provider_backend", "high_mem"}
    if not isinstance(config, dict) or not required <= config.keys() or config.keys() - allowed:
        raise LifecycleError("execution config needs drive_parent, token_command, gpu, max_attempts, max_seconds")
    if config["gpu"] not in ("NONE", "T4", "L4", "A100", "H100", "G4"):
        raise LifecycleError("unsupported GPU in execution config")
    if type(config["max_attempts"]) is not int or type(config["max_seconds"]) is not int:
        raise LifecycleError("execution bounds must be integers")
    if (config.get("provider_backend", "legacy") not in ("legacy", "rest")
            or type(config.get("high_mem", False)) is not bool):
        raise LifecycleError("invalid provider backend or high-memory setting")
    if type(config.get("fault_after_checkpoint", False)) is not bool:
        raise LifecycleError("fault_after_checkpoint must be boolean")
    token_command = Path(config["token_command"])
    if not token_command.is_absolute():
        token_command = config_path.parent / token_command
    helper_argv = json.loads(token_command.read_text(encoding="utf-8"))
    # The built-in helper pins its original profile. Do not combine that Drive
    # authorization with Colab authorization from another explicitly selected home.
    if (isinstance(helper_argv, list) and len(helper_argv) == 6
            and helper_argv[1:5] == ["-m", "colabcola", "_drive-token", "--config-home"]):
        from .profiles import config_home
        if not isinstance(helper_argv[5], str) or Path(helper_argv[5]).expanduser().absolute() != config_home():
            raise LifecycleError("This job's Drive helper belongs to a different configuration; select its original --profile or COLABCOLA_CONFIG_HOME")
    directory = (command.run_dir or job.with_suffix(".run")).resolve()
    progress = Progress(directory)
    if not (directory / "plan.json").exists():
        progress.say("Prepare: packaging and checking local inputs")
        result = prepare(SimpleNamespace(job=job, job_args=command.job_args, run_dir=directory,
            token_command=token_command, drive_parent=config["drive_parent"], gpu=config["gpu"],
            max_attempts=config["max_attempts"], max_seconds=config["max_seconds"],
            fault_after_checkpoint=config.get("fault_after_checkpoint", False),
            provider_backend=config.get("provider_backend", "legacy"), high_mem=config.get("high_mem", False)))
        if command.resume_from:
            prior = Path(command.resume_from).resolve()
            old_plan = json.loads((prior / "plan.json").read_text())
            old = json.loads((prior / "state.json").read_text())
            if old_plan["job_id"] != result["job_id"] or old["phase"] not in TERMINAL:
                raise LifecycleError("resume-from requires a terminal run of this exact job")
            checkpoint = old.get("checkpoint")
            if not checkpoint:
                raise LifecycleError("prior run has no verified checkpoint")
            plan = json.loads((directory / "plan.json").read_text())
            plan["initial_checkpoint"] = checkpoint
            atomic(directory / "plan.json", plan)
            state = json.loads((directory / "state.json").read_text())
            state["checkpoint"] = checkpoint
            atomic(directory / "state.json", state)
    try:
        plan = json.loads((directory / "plan.json").read_text())
    except (OSError, ValueError):
        # Let the gate persist the plan-read failure without staging.
        return supervise(directory, command.approve or "", preflight_only=True)
    def validate_prepared(plan):
        if helper_argv != plan["token_command"]:
            raise LifecycleError("token helper changed; use a new --run-dir")
        if command.resume_from:
            previous = Path(command.resume_from).resolve()
            old = json.loads((previous / "state.json").read_text())
            selected = old.get("checkpoint")
            if (old["phase"] not in TERMINAL or not selected or
                    selected.get("commit", {}).get("job_id") != plan["job_id"] or
                    plan.get("initial_checkpoint") != selected):
                raise LifecycleError("resume-from does not match this journal; prepare a new --run-dir")
        if build_bundle(invocation).job_id != plan["job_id"]:
            raise LifecycleError("job changed; use a new --run-dir")
        if str(directory) != plan["directory"]:
            raise LifecycleError("run directory moved across hosts")
        for key in ("gpu", "max_attempts", "max_seconds"):
            if config[key] != plan[key]:
                raise LifecycleError("execution bounds changed; use a new --run-dir")
        if config["drive_parent"] != plan["parent"] or config.get("fault_after_checkpoint", False) != plan["fault_after_checkpoint"]:
            raise LifecycleError("execution destination/policy changed; use a new --run-dir")
        for key, default in (("provider_backend", "legacy"), ("high_mem", False)):
            if config.get(key, default) != plan.get(key, default):
                raise LifecycleError("provider policy changed; use a new --run-dir")
        if command.approve and command.approve != digest(encoded(plan)):
            raise LifecycleError("approval does not match prepared plan")
        verified_inputs(directory, plan["files"])
    approval = command.approve or digest(encoded(plan))
    state = None if command.prepare_only else json.loads((directory / "state.json").read_text())
    existing = bool(state and state["phase"] in TERMINAL and not command.preflight_only)
    if existing:
        validate_prepared(plan)
        progress.view_existing(state)
        progress.say("Run: viewing existing " + state["phase"].replace("_", " ") +
                     " journal (no new runtime)")
    if existing:
        accelerator = "CPU" if plan["gpu"] == "NONE" else plan["gpu"]
        count = plan["max_attempts"]
        progress.say(f"Run: {accelerator} | up to {count} "
                     f"{'runtime' if count == 1 else 'runtimes'} | {plan['max_seconds']}s total")
    else:
        accelerator = "CPU" if plan["gpu"] == "NONE" else plan["gpu"]
        count = plan["max_attempts"]
        progress.say(f"Run: {accelerator} | up to {count} "
                     f"{'runtime' if count == 1 else 'runtimes'} | {plan['max_seconds']}s total")
    progress.say(f"Provider: Google Colab ({plan.get('provider_backend', 'legacy')}) | "
                 f"runtime {plan.get('runtime_version', 'latest')} | "
                 f"high-memory {'on' if plan.get('high_mem', False) else 'off'}")
    progress.say("Workspace: " + str(directory))
    if command.prepare_only:
        validate_prepared(plan)
        print(json.dumps({"state": "prepared", "job_id": plan["job_id"], "run_dir": str(directory),
                          "approval_sha256": approval, "gpu": plan["gpu"],
                          "max_attempts": plan["max_attempts"], "max_seconds": plan["max_seconds"],
                          "provider_backend": plan.get("provider_backend", "legacy"),
                          "runtime_version": plan.get("runtime_version", "latest"),
                          "high_mem": plan.get("high_mem", False)}, indent=2))
        return 0
    if existing:
        if not state.get("final") and state.get("attempts"):
            progress.say("Checking Drive for late artifacts and saved logs from this existing run")
        state = recover_terminal_final(directory, observer=progress)
        progress(state)
        return _report_result(state, plan, directory, progress, existing=True)
    # Preflight is read-only. The journal still records every individual check.
    progress.say("Preflight: checking job, account, and Drive access")
    try:
        state = supervise(directory, approval, observer=progress, preflight_only=command.preflight_only,
                          validator=validate_prepared, allow_active_assignments=command.allow_active_assignments)
    except KeyboardInterrupt:
        if json.loads((directory / "state.json").read_text())["phase"] == "prepared":
            raise LifecycleError("preflight interrupted; journal remains prepared") from None
        atomic(directory / "cancel.request", {"reason": "keyboard_interrupt"})
        progress.say("Cancellation requested; releasing owned compute")
        state = supervise(directory, approval, observer=progress)
    except LifecycleError:
        raise
    except Exception as exc:
        raise LifecycleError("supervisor unavailable: " + type(exc).__name__ +
                             "; inspect the journal and reconnect with the same command") from None
    if command.preflight_only:
        print(json.dumps(state, indent=2))
        return 0
    return _report_result(state, plan, directory, progress)

