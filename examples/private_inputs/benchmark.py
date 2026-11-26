"""Create and verify two independent managed runs sharing a synthetic private input."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

SIZE = 1024**3
CHUNK = 8 * 1024**2
TARGET = "models/weights.bin"


def make_fixture(root: Path, *, size: int = SIZE):
    root.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(root).free < size + 128 * 1024**2:
        raise RuntimeError("Insufficient local space for the synthetic fixture")
    project = root / "example"
    project.mkdir()
    shutil.copyfile(Path(__file__).with_name("check.py"), project / "check.py")
    asset = project / "private.bin"
    digest = hashlib.sha256()
    nonce = uuid.uuid4().bytes
    with asset.open("xb") as stream:
        for index, offset in enumerate(range(0, size, CHUNK)):
            n = min(size - offset, CHUNK)
            header = nonce + index.to_bytes(8, "little")
            block = (header + b"a" * max(0, n - len(header)))[:n]
            stream.write(block)
            digest.update(block)
    identity = {"target": TARGET, "bytes": size, "sha256": digest.hexdigest()}
    (root / "identity.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    for label, seed in (("A", 202640), ("B", 202641)):
        (project / f"{label}.colaml").write_text(f'''name = "private-input-acceptance-{label.lower()}"
project = "."
include = ["check.py"]
inputs = []
commands = [["python", "check.py"]]

[[private_inputs]]
source = "private.bin"
target = "{TARGET}"

[environment]
dependencies = []

[compute]
gpu = "NONE"
high_mem = false

[outputs]
include = ["outputs/**"]
maximum_bytes = 1048576
checkpoints = []
logs = []

[[outputs.completed]]
marker = "outputs/complete.json"
files = ["check.json"]
ready_when = {{ state = "complete" }}

[limits]
timeout_seconds = 120
max_seconds = 1800
max_runtimes = 1
seed = {seed}

[execution]
runtime_version = "latest"
provider_backend = "legacy"
''', encoding="utf-8")
    return asset, identity


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_run(workspace, identity, *, expected_state, elapsed):
    from colabcola.lifecycle import verify_final
    from colabcola.lifecycle_io import digest, encoded, verify_private_input_result, verified_inputs
    from colabcola.preflight import frozen_job
    from colabcola.managed import _verify_completed_outputs

    workflow = read_json(workspace / "workflow.json")
    exit_record = read_json(workspace / "cli-exit.json")
    journal = Path(exit_record["journal"])
    plan = read_json(journal / "plan.json")
    state = read_json(journal / "state.json")
    verified_inputs(journal, plan["files"])
    manifest = frozen_job(journal, plan)
    report = verify_final(state["final"]["local_directory"], workflow["job_id"])
    verify_private_input_result([identity], report)
    if state.get("artifacts", {}).get("local_directory"):
        _verify_completed_outputs(state["artifacts"], workflow["job_id"])
    facts = report["transport"]["private_inputs"]
    staging = state["private_input_staging"][identity["target"]]
    with zipfile.ZipFile(Path(state["final"]["local_directory"]) / "artifacts.zip") as archive:
        output = json.loads(archive.read("outputs/check.json"))
    attempts = state["attempts"]
    admission = [read_json(path) for path in (journal / "preflight").glob("*.json")]
    admitted = [p for p in admission if p.get("state") == "passed" and p.get("private_inputs")]
    if not admitted:
        raise RuntimeError("Missing successful private-input preflight evidence")
    admission_report = max(admitted, key=lambda p: p["ended_at"])
    checked = admission_report["private_inputs"]
    expected = {k: identity[k] for k in ("bytes", "sha256")}
    checks = {
        "frozen_identity": plan["private_inputs"] == manifest["private_inputs"] == [identity],
        "asset_not_embedded": not any(f["path"] in {"private.bin", identity["target"]} for f in manifest["files"]),
        "preflight_bound_read_only": admission_report["job_id"] == workflow["job_id"]
            and admission_report["plan_sha256"] == digest(encoded(plan))
            and admission_report["allocation_performed"] is False and admission_report["upload_performed"] is False,
        "preflight_source": checked["assets"] == [{**identity, "state": "local_source" if expected_state == "published" else "committed"}],
        "managed_success": exit_record["exit_code"] == 0 and state["phase"] == "completed" and report["terminal_state"] == "success",
        "workload_digest": output == expected,
        "runtime_restoration": facts["files"] == [{**identity, "state": "restored"}],
        "runtime_released": len(attempts) == 1 and attempts[0].get("released") is True,
        "preflight_bounds": checked["total_bytes"] == identity["bytes"] and checked["per_file_limit_bytes"] == 4 * 1024**3 and checked["total_limit_bytes"] == 8 * 1024**3,
        "disk_headroom": checked["local_recovery_headroom_bytes"] >= 0 and checked["runtime_free_bytes"] is None and facts["runtime_headroom_bytes"] >= 0,
        "staging_state": staging["state"] == expected_state,
        "uploaded_bytes": staging["confirmed_uploaded_bytes"] == (identity["bytes"] if expected_state == "published" else 0),
        "small_prepared_bundle": sum((journal / name).stat().st_size for name in plan["files"]) < 512 * 1024**2,
    }
    summary = {"checks": checks, "job_id": workflow["job_id"], "allocation_id": attempts[0]["allocation_id"],
               "asset": identity, "staging": staging, "preflight": checked, "restoration": facts,
               "controller_elapsed_seconds": elapsed,
               "completed_outputs_verified": bool(state.get("artifacts", {}).get("local_directory"))}
    if not all(checks.values()):
        print(json.dumps(summary, indent=2), flush=True)
        raise RuntimeError("Private-input run acceptance failed")
    return summary, state["private_input_receipts"]


def run_live(root, profile):
    asset, identity = make_fixture(root)
    print(f"BENCH_DIR={root}", flush=True)
    env = {**os.environ, "TERM": "dumb"}
    def command(label, *extra):
        subprocess.run([sys.executable, "-m", "colabcola", "run", "--profile", profile,
                        "--colaml", str(root / "example" / f"{label}.colaml"),
                        "--run-dir", str(root / f"run-{label}"), "--non-interactive", *extra],
                       env=env, check=True)
    # Freeze both before moving the source. No automatic retries or assignment overrides.
    for label in ("A", "B"):
        command(label, "--prepare-only")
    first, second = [read_json(root / f"run-{label}" / "workflow.json")["job_id"] for label in ("A", "B")]
    if first == second:
        raise RuntimeError("Recipes must freeze different job identities")
    receipts = None
    for label, expected_state in (("A", "published"), ("B", "reused")):
        start = time.monotonic()
        command(label, "--preflight-only")
        command(label)
        summary, current = verify_run(root / f"run-{label}", identity, expected_state=expected_state,
                                      elapsed=time.monotonic() - start)
        (root / f"{label}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"PRIVATE INPUT ACCEPTANCE {label}", flush=True)
        print(json.dumps({k:v for k,v in summary.items() if k != "allocation_id"}, indent=2), flush=True)
        if label == "A":
            receipts = current
            # Move only our synthetic fixture, preserving it for later inspection.
            asset.rename(root / "original-synthetic.bin")
            first_allocation = summary["allocation_id"]
        elif receipts != current or first_allocation == summary["allocation_id"]:
            raise RuntimeError("Expected identical receipt on a distinct allocation")
    print("PRIVATE INPUT ACCEPTANCE PASSED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="new benchmark directory, must not exist")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--prepare-fixture-only", action="store_true", help="generate files, no controller or network access")
    args = parser.parse_args()
    root = args.directory.resolve()
    if args.prepare_fixture_only:
        _, identity = make_fixture(root)
        print(json.dumps(identity, indent=2))
    else:
        run_live(root, args.profile)


if __name__ == "__main__":
    main()
