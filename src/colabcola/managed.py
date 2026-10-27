"""Single-command managed submission with durable journals and readable progress."""
from __future__ import annotations
import json
import re
from pathlib import Path
import sys
from types import SimpleNamespace
from .lifecycle import TERMINAL, prepare, supervise, verify_final, recover_terminal_final
from .lifecycle_io import LifecycleError, atomic, digest, encoded, read_regular, verified_inputs


class Progress:
    def __init__(self, directory, stream=None):
        self.directory = Path(directory)
        self.stream = stream or sys.stderr
        self.sequence = 0
        self.last_progress = None
        self.log_positions = {}

    def say(self, message):
        # Workload strings cannot inject terminal escape sequences.
        message = "".join(c if c.isprintable() else " " for c in str(message))
        print("colabcola: " + message, file=self.stream, flush=True)

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
                    self.say(f"[attempt {attempt} {name}] " + line[:1500])
            self.log_positions[key] = (len(body), digest(body), max(cursor, end))

    def __call__(self, state):
        for event in state.get("events", []):
            if event["sequence"] > self.sequence:
                self.say(event["event"].replace("_", " "))
                self.sequence = event["sequence"]
        if state.get("connectivity") == "unknown" and state.get("attention"):
            value = (state.get("connectivity"), state["attention"])
            if value != getattr(self, "attention", None):
                self.say("Needs attention: " + str(value[1]))
                self.attention = value
        auth = state.get("drive_authorization", {})
        delivery = auth.get("delivery", {})
        health = (auth.get("state"), delivery.get("state"), delivery.get("stage"), delivery.get("error_type"))
        if health != getattr(self, "authorization_health", None) and auth.get("state") in ("valid", "expiring", "unavailable"):
            if auth["state"] == "valid":
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


def run_managed(command, invocation):
    from .bundle import build_bundle
    job = command.job_file.resolve()
    config_path = (command.execution_config or job.with_suffix(".run.json")).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
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
    directory = (command.run_dir or job.with_suffix(".run")).resolve()
    progress = Progress(directory)
    if not (directory / "plan.json").exists():
        progress.say("Packaging and checking inputs (local preparation)")
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
    progress.say(f"{plan['gpu']}; at most {plan['max_attempts']} runtimes; {plan['max_seconds']} seconds total")
    progress.say(f"Provider {plan.get('provider_backend', 'legacy')}; runtime {plan.get('runtime_version', 'latest')}; "
                 f"high host RAM: {plan.get('high_mem', False)}")
    progress.say("Logs and recovered files: " + str(directory))
    progress.say("Live logs: verified snapshots checked every 5 seconds, plus transfer/poll time")
    if command.prepare_only:
        validate_prepared(plan)
        print(json.dumps({"state": "prepared", "job_id": plan["job_id"], "run_dir": str(directory),
                          "approval_sha256": approval, "gpu": plan["gpu"],
                          "max_attempts": plan["max_attempts"], "max_seconds": plan["max_seconds"],
                          "provider_backend": plan.get("provider_backend", "legacy"),
                          "runtime_version": plan.get("runtime_version", "latest"),
                          "high_mem": plan.get("high_mem", False)}, indent=2))
        return 0
    state = json.loads((directory / "state.json").read_text())
    if state["phase"] in TERMINAL and not command.preflight_only:
        validate_prepared(plan)
        state = recover_terminal_final(directory, observer=progress)
        progress(state)
        if state.get("final"):
            verify_final(state["final"]["local_directory"], plan["job_id"])
            progress.say("Verified results: " + state["final"]["local_directory"])
        print(json.dumps({"state": state["phase"], "run_dir": str(directory),
                          "result_directory": state.get("final", {}).get("local_directory"),
                          "completed_artifact_directory": state.get("artifacts", {}).get("local_directory")}, indent=2))
        return 0 if state["phase"] == "completed" else 3
    # The invocation is the execution request. --prepare-only provides a no-network preview.
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
    if state.get("final"):
        root = state["final"]["local_directory"]
        verify_final(root, plan["job_id"])
        progress.say("Verified results: " + root)
    if state.get("checkpoint"):
        progress.say("Verified recovery checkpoint retained on Drive and under " + str(directory / "checkpoints"))
    print(json.dumps({"state": state["phase"], "job_id": plan["job_id"], "run_dir": str(directory),
                      "result_directory": state.get("final", {}).get("local_directory"),
                          "completed_artifact_directory": state.get("artifacts", {}).get("local_directory"),
                      "reason": state.get("reason"), "hardware": state.get("hardware")}, indent=2))
    return 0 if state["phase"] == "completed" else 3

