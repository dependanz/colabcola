"""Verify a wheel is installed only in the selected environment."""
import importlib.util
import json
import os
from pathlib import Path
import sys

output = Path("outputs")
output.mkdir(exist_ok=True)
assert Path(sys.prefix) == Path(os.environ["CONDA_PREFIX"])
if sys.argv[1] == "train":
    import cc_wheel_probe
    assert cc_wheel_probe.VALUE == 17
    record = {"value": cc_wheel_probe.VALUE, "python": sys.version.split()[0]}
    (output / "train.json").write_text(json.dumps(record) + "\n")
else:
    assert importlib.util.find_spec("cc_wheel_probe") is None
    train = json.loads((output / "train.json").read_text())
    assert train["value"] == 17
    record = {"wheel_absent": True, "train": train, "evaluate_python": sys.version.split()[0]}
    (output / "acceptance.json").write_text(json.dumps(record) + "\n")
    (output / "complete.json").write_text('{"state":"complete"}\n')
print(json.dumps(record), flush=True)
