from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def _median_parse_ms(content: bytes, repetitions: int) -> float:
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        json.loads(content)
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def _facts(path: Path, repetitions: int) -> dict[str, object]:
    content = path.read_bytes()
    document = json.loads(content)
    source_lines = [
        line
        for cell in document["cells"]
        for line in str(cell.get("source", "")).splitlines()
    ]
    return {
        "bytes": len(content),
        "cells": len(document["cells"]),
        "maximum_source_line_characters": max(map(len, source_lines), default=0),
        "median_json_parse_milliseconds": _median_parse_ms(content, repetitions),
        "path": path.resolve().as_posix(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("embedded", type=Path)
    parser.add_argument("controller", type=Path)
    parser.add_argument("--repetitions", type=int, default=9)
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")

    embedded = _facts(args.embedded, args.repetitions)
    controller = _facts(args.controller, args.repetitions)
    result = {
        "controller": controller,
        "embedded": embedded,
        "notebook_byte_reduction_ratio": embedded["bytes"] / controller["bytes"],
        "parse_time_reduction_ratio": (
            embedded["median_json_parse_milliseconds"]
            / controller["median_json_parse_milliseconds"]
        ),
        "repetitions": args.repetitions,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
