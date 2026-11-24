"""Verify a completed managed run of this example without allocating compute."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

from colabcola.lifecycle import verify_final
from colabcola.managed import _verify_completed_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path, help="the workflow directory passed to colabcola run")
    args = parser.parse_args()
    workspace = args.run_directory.resolve()
    workflow = json.loads((workspace / "workflow.json").read_text(encoding="utf-8"))
    exit_record = json.loads((workspace / "cli-exit.json").read_text(encoding="utf-8"))
    state = json.loads((Path(exit_record["journal"]) / "state.json").read_text(encoding="utf-8"))
    final = state.get("final", {}).get("local_directory")
    if not final:
        raise SystemExit("No recovered final artifact bundle is available; inspect this run's logs.")
    result_root = Path(final)
    result = verify_final(result_root, workflow["job_id"])
    with zipfile.ZipFile(result_root / "artifacts.zip") as archive:
        report = json.loads(archive.read("outputs/cuda/acceptance.json"))
    completed = state.get("artifacts", {})
    if completed.get("local_directory"):
        _verify_completed_outputs(completed, workflow["job_id"])
    attempts = state.get("attempts", [])
    steps = result.get("steps", [])
    checks = {
        "run_completed": state["phase"] == "completed" and result["terminal_state"] == "success",
        "environment_setup_succeeded": result.get("stages", {}).get("environment", {}).get("state") == "success",
        "build_step_succeeded": (
            len(steps) == 1 and steps[0].get("name") == "build-and-check"
            and steps[0].get("state") == "success" and steps[0].get("exit_code") == 0
        ),
        "extension_build_and_execution_passed": report.get("extension_test") == "passed",
        "all_extension_checks_passed": bool(report.get("checks")) and all(value is True for value in report["checks"].values()),
        "expected_element_count": report.get("elements_checked") == 4096,
        "exact_numerical_result": report.get("maximum_absolute_error") == 0,
        "newer_host_toolkit_exercised": report.get("host_mismatch_exercised") is True,
        "completed_output_group_verified": bool(completed.get("local_directory")),
        "runtime_release_recorded": bool(attempts) and all(attempt.get("released") is True for attempt in attempts),
    }
    print("CUDA ACCEPTANCE RESULT")
    print(json.dumps({
        "checks": checks,
        "python": report.get("python"),
        "torch": report.get("torch"),
        "torch_cuda": report.get("torch_cuda"),
        "selected_nvcc_version": report.get("nvcc", {}).get("version"),
        "host_nvcc_version": (report.get("host_nvcc") or {}).get("version"),
        "host_mismatch_detail": report.get("host_mismatch_detail"),
        "cxx_version": report.get("cxx", {}).get("version"),
        "gpu": report.get("gpu"),
        "elements_checked": report.get("elements_checked"),
        "maximum_absolute_error": report.get("maximum_absolute_error"),
        "build_seconds": report.get("build_seconds"),
        "result_directory": str(result_root),
    }, indent=2))
    if not all(checks.values()):
        raise SystemExit("CUDA ACCEPTANCE INCOMPLETE: inspect the false checks above.")
    print("CUDA ACCEPTANCE PASSED")


if __name__ == "__main__":
    main()
