"""Frozen launcher for ordered recipe steps in named environments."""
import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from environment_runtime import _colabcola_environment_command


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load():
    if len(sys.argv) != 1:
        raise RuntimeError("named recipe steps do not accept additional invocation arguments")
    frozen = json.loads(Path(__file__).with_name("commands.json").read_text(encoding="utf-8"))
    selected = os.environ.get("COLABCOLA_ENVIRONMENTS_CONFIG")
    if not selected or not Path(selected).is_absolute():
        raise RuntimeError("named recipe requires the verified runtime environment configuration")
    config = json.loads(Path(selected).read_text(encoding="utf-8"))
    if config.get("schema") != 1 or not isinstance(config.get("job_id"), str):
        raise RuntimeError("invalid runtime environment configuration")
    if not isinstance(frozen.get("steps"), list) or not frozen["steps"] or config.get("steps") != frozen["steps"]:
        raise RuntimeError("frozen recipe steps differ from the runtime manifest")
    if len(frozen["steps"]) > 128:
        raise RuntimeError("named recipe exceeds its step limit")
    for key in ("job_root", "result_root"):
        if not isinstance(config.get(key), str) or not Path(config[key]).is_absolute():
            raise RuntimeError("runtime configuration requires absolute execution paths")
    deadline = config.get("deadline")
    if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline)):
        raise RuntimeError("invalid runtime execution deadline")
    if not isinstance(config.get("environments"), dict) or not isinstance(config.get("env", {}), dict):
        raise RuntimeError("invalid resolved environment configuration")
    config["shell_scripts"] = frozen.get("shell_scripts", {})
    return config


def _stop(process):
    # Normal managed execution starts this launcher in its own process group.
    # Leave children in that group so the runtime's existing killpg guard reaches
    # every step. Evidence is written before a deadline kills that whole group.
    if os.name != "nt" and os.getpgrp() == os.getpid():
        os.killpg(os.getpgrp(), signal.SIGKILL)
    for method in (process.terminate, process.kill):
        try:
            method()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1)
            return
        except subprocess.TimeoutExpired:
            pass


def main():
    config = _load()
    root = Path(config["job_root"]).resolve()
    result_root = Path(config["result_root"])
    result_root.mkdir(parents=True, exist_ok=True)
    path = result_root / "steps.json"
    steps = config["steps"]
    evidence = {"schema": 1, "job_id": config["job_id"], "state": "running",
                "started_at": _now(), "failed_step": None,
                "steps": [dict(step, state="pending", exit_code=None) for step in steps]}
    started = time.monotonic()
    _write(path, evidence)
    for index, step in enumerate(steps):
        record = evidence["steps"][index]
        process = None
        step_start = time.monotonic()
        record.update(state="running", started_at=_now())
        _write(path, evidence)
        try:
            if config.get("deadline") is not None and time.time() >= config["deadline"]:
                raise TimeoutError("execution deadline reached before step start")
            if step["environment"] not in config["environments"]:
                raise RuntimeError("step refers to an unresolved environment")
            cwd = (root / step["cwd"]).resolve()
            cwd.relative_to(root)
            if not cwd.is_dir():
                raise RuntimeError("step working directory does not exist in the packaged project")
            # Retain the lexical cwd for Bash's link checks, even though the
            # process is launched in the independently resolved working directory.
            options = {}
            if step["command"][0] == "bash":
                target = (Path(step["cwd"]) / step["command"][1]).as_posix()
                options = {"cwd": root / step["cwd"],
                           "script_sha256": config["shell_scripts"].get(target)}
            command, env = _colabcola_environment_command(config["environments"][step["environment"]],
                step["command"], root, declared=config.get("env", {}), **options)
            # Only these explicitly trusted run values cross from the control
            # process; no credentials or helper configuration are copied.
            for name in ("COLABCOLA_SEED", "COLABCOLA_RESUME_MANIFEST"):
                if name in os.environ:
                    env[name] = os.environ[name]
            print("colabcola: step " + step["name"] + " starting (environment " + step["environment"] + ")", flush=True)
            process = subprocess.Popen(command, cwd=str(cwd), env=env,
                stdin=subprocess.DEVNULL, shell=False)
            remaining = None if config.get("deadline") is None else max(0, config["deadline"] - time.time())
            try:
                code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError("execution deadline reached during step " + step["name"]) from exc
            record["exit_code"] = code
            print("colabcola: step " + step["name"] + " exited " + str(code), flush=True)
            if code:
                raise RuntimeError("step exited with status " + str(code))
            record["state"] = "success"
        except BaseException as exc:
            record.update(state="failed", error={"type": type(exc).__name__, "message": str(exc)[:2000]},
                elapsed_seconds=round(time.monotonic() - step_start, 6), finished_at=_now())
            evidence.update(state="failed", failed_step=step["name"], finished_at=_now(),
                            elapsed_seconds=round(time.monotonic() - started, 6))
            for later in evidence["steps"][index + 1:]:
                later["state"] = "not_run"
            _write(path, evidence)
            if process is not None:
                _stop(process)
            print("colabcola: step " + step["name"] + " failed: " + str(exc)[:2000], file=sys.stderr, flush=True)
            code = record.get("exit_code") or 1
            return code if code > 0 else 128 - code
        record.update(elapsed_seconds=round(time.monotonic() - step_start, 6), finished_at=_now())
        _write(path, evidence)
    evidence.update(state="success", finished_at=_now(), elapsed_seconds=round(time.monotonic() - started, 6))
    _write(path, evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
