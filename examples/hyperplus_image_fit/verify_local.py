"""CPU-only compiled-notebook diagnostic on synthetic RGB images, not a Colab acceptance run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).parent.resolve()


def verify(output, wheel):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = output / "synthetic-data"
    data.mkdir()
    manifest = {"schema": 1, "source": "synthetic RGB coordinate patterns; not CIFAR-10", "files": {}}
    y, x = np.mgrid[:8, :8] / 7
    for split, offsets in (("train", [0, .1, .2, .3]), ("heldout", [.15, .25])):
        images = np.stack([np.stack((x, y, np.clip((x + y) / 2 + offset, 0, 1))) for offset in offsets])
        path = data / (split + ".npy")
        np.save(path, np.rint(images * 255).astype(np.uint8), allow_pickle=False)
        manifest["files"][path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                       "images": len(offsets)}
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    job = output / "hyperplus-local-synthetic.colajob"
    subprocess.run([sys.executable, str(ROOT / "make_job.py"), "--data", str(data),
                    "--wheel", str(Path(wheel).resolve()), "--output", str(job),
                    "--name", "hyperplus-local-synthetic", "--steps", "4", "--local-cpu"], check=True)
    sys.path.insert(0, str(ROOT.parents[1] / "src"))
    from colabcola.invocation import load_job_invocation
    from colabcola.compiler import compile_invocation
    invocation = load_job_invocation(job)
    compiled = compile_invocation(invocation, output / "hyperplus-local-synthetic.ipynb")
    second = compile_invocation(invocation, output / "repeat.ipynb")
    if compiled.notebook_bytes != second.notebook_bytes:
        raise AssertionError("compiler reproducibility failure")
    namespace = {"COLABCOLA_CONTENT_ROOT_OVERRIDE": str(output / "runtime")}
    for cell in json.loads(compiled.notebook_bytes)["cells"]:
        if cell["cell_type"] == "code":
            exec(cell["source"], namespace)
    result_root = output / "runtime/colabcola/results" / compiled.job_id
    result = json.loads((result_root / "result.json").read_text())
    if result["terminal_state"] != "success":
        raise AssertionError(result)
    job_root = output / "runtime/colabcola/jobs" / compiled.job_id
    for record in result["outputs"]:
        if hashlib.sha256((job_root / record["path"]).read_bytes()).hexdigest() != record["sha256"]:
            raise AssertionError("output integrity mismatch")
    evidence = {"scope": "local CPU; synthetic images; no Drive/runtime allocation/quality claim",
                "job_id": compiled.job_id, "notebook_sha256": compiled.notebook_sha256,
                "terminal_state": result["terminal_state"], "verified_outputs": len(result["outputs"]),
                "artifact_sha256": hashlib.sha256((result_root / "artifacts.zip").read_bytes()).hexdigest(),
                "result": str(result_root / "result.json")}
    (output / "verification.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wheel", required=True)
    args = parser.parse_args()
    verify(args.output, args.wheel)
