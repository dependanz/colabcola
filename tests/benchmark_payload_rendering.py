"""Compare generated-notebook payload rendering without third-party dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path


def _median_milliseconds(operation, *, repetitions: int) -> float:
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def _measure(path: Path, *, repetitions: int) -> dict[str, object]:
    notebook_bytes = path.read_bytes()
    parse_ms = _median_milliseconds(
        lambda: json.loads(notebook_bytes), repetitions=repetitions
    )
    document = json.loads(notebook_bytes)
    code_cells = [cell for cell in document["cells"] if cell["cell_type"] == "code"]
    payload_cells = [
        cell for cell in code_cells if cell["metadata"]["colabcola"]["group"] == "payload"
    ]
    payload_sources = [cell["source"] for cell in payload_cells]

    def compile_payload() -> None:
        for index, source in enumerate(payload_sources):
            compile(source, f"<payload-{index}>", "exec")

    compiled_payload = [
        compile(source, f"<payload-{index}>", "exec")
        for index, source in enumerate(payload_sources)
    ]

    def execute_payload() -> None:
        namespace: dict[str, object] = {"COLABCOLA_PAYLOAD_CHUNKS": []}
        for code in compiled_payload:
            exec(code, namespace)

    compile_ms = _median_milliseconds(compile_payload, repetitions=repetitions)
    execute_ms = _median_milliseconds(execute_payload, repetitions=repetitions)

    namespace: dict[str, object] = {"COLABCOLA_PAYLOAD_CHUNKS": []}
    for code in compiled_payload:
        exec(code, namespace)
    payload_text = "".join(namespace["COLABCOLA_PAYLOAD_CHUNKS"])
    all_lines = [line for source in payload_sources for line in source.splitlines()]

    return {
        "notebook": path.as_posix(),
        "notebook_bytes": len(notebook_bytes),
        "notebook_sha256": hashlib.sha256(notebook_bytes).hexdigest(),
        "cell_count": len(document["cells"]),
        "payload_cell_count": len(payload_cells),
        "hidden_form_payload_cells": sum(
            cell["metadata"].get("cellView") == "form" for cell in payload_cells
        ),
        "payload_source_chars": sum(map(len, payload_sources)),
        "payload_source_lines": len(all_lines),
        "maximum_payload_source_line_chars": max(map(len, all_lines)),
        "payload_base64_chars": len(payload_text),
        "payload_base64_sha256": hashlib.sha256(payload_text.encode("ascii")).hexdigest(),
        "json_parse_ms_median": parse_ms,
        "payload_compile_ms_median": compile_ms,
        "payload_execute_ms_median": execute_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--repetitions", type=int, default=7)
    arguments = parser.parse_args()
    if arguments.repetitions < 1:
        parser.error("--repetitions must be positive")

    baseline = _measure(arguments.baseline.resolve(), repetitions=arguments.repetitions)
    candidate = _measure(arguments.candidate.resolve(), repetitions=arguments.repetitions)
    if baseline["payload_base64_sha256"] != candidate["payload_base64_sha256"]:
        raise SystemExit("payload base64 differs between baseline and candidate")

    baseline_line = int(baseline["maximum_payload_source_line_chars"])
    candidate_line = int(candidate["maximum_payload_source_line_chars"])
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "repetitions": arguments.repetitions,
        "baseline": baseline,
        "candidate": candidate,
        "comparison": {
            "same_payload_base64": True,
            "notebook_byte_delta": int(candidate["notebook_bytes"])
            - int(baseline["notebook_bytes"]),
            "notebook_percent_delta": (
                int(candidate["notebook_bytes"]) / int(baseline["notebook_bytes"]) - 1
            )
            * 100,
            "maximum_line_reduction_factor": baseline_line / candidate_line,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
