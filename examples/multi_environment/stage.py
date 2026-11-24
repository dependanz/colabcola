"""A small two-stage workload using only the Python standard library."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "evaluate"))
    stage = parser.parse_args().stage
    if stage == "prepare":
        write_json(Path("outputs/prepare/data.json"), {
            "values": [2, 3, 5, 7], "python": platform.python_version(),
        })
        write_json(Path("outputs/prepare/complete.json"), {"state": "complete"})
    else:
        data = json.loads(Path("outputs/prepare/data.json").read_text(encoding="utf-8"))
        write_json(Path("outputs/evaluation.json"), {
            "sum": sum(data["values"]),
            "prepare_python": data["python"],
            "evaluate_python": platform.python_version(),
        })
    print(stage + " finished using Python " + platform.python_version(), flush=True)


if __name__ == "__main__":
    main()
