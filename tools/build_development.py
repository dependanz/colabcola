"""Build a self-contained Colabcola wheel/sdist; never installs or authenticates.

The pinned providers are private package modules. Their ordinary dependencies are
resolved by pip from project metadata; no provider wheels or Git checkout are
needed by users. Use vendor_provider.py explicitly to refresh vendored sources.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def sha(body):
    return hashlib.sha256(body).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("*.whl")) or any(output.glob("*.tar.gz")):
        parser.error("output already contains distributions; use a fresh directory")
    root = Path(__file__).resolve().parents[1]
    evidence = {"provider_distribution": "private bundled modules", "sources": {}, "distributions": {}}
    with tempfile.TemporaryDirectory(prefix="colabcola-build-") as temporary:
        stage = Path(temporary) / "colabcola"
        stage.mkdir()
        paths = [p for p in (root / "src/colabcola").rglob("*")
                 if p.is_file() and (p.suffix in (".py", ".json") or p.name == "LICENSE")
                 and "__pycache__" not in p.parts]
        paths += [root / name for name in (
            "pyproject.toml", "MANIFEST.in", "README.md",
            "LICENSE", "NOTICE",
            "tools/build_development.py", "tools/vendor_provider.py"
        )]
        paths += sorted(
            p for p in (root / "docs").rglob("*")
            if p.is_file() and p.suffix.lower() in {".md", ".svg", ".png"}
        )
        for path in paths:
            relative = path.relative_to(root)
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            body = path.read_bytes()
            destination.write_bytes(body)
            evidence["sources"][relative.as_posix()] = sha(body)
        subprocess.run([sys.executable, "-m", "build", "--outdir", str(output), str(stage)], check=True)
    for wheel in output.glob("*.whl"):
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            if not all(n.startswith(("colabcola/", "colabcola-")) for n in names):
                raise ValueError("unexpected wheel namespace")
            if any(n.endswith((".colajob", ".zip", ".pt", ".pth", ".pkl")) for n in names):
                raise ValueError("unexpected workload data in wheel")
            for vendor in ("colab_cli", "jupyter_kernel_client"):
                if "colabcola/_vendor/" + vendor + "/LICENSE" not in names:
                    raise ValueError("bundled provider license missing")
            if any(n.startswith("colabcola/_setup_guide/") or n == "colabcola/setup_guide.py" for n in names):
                raise ValueError("obsolete browser setup guide in wheel")
            provenance = json.loads(archive.read("colabcola/_vendor/provenance.json"))
            for vendor, record in provenance["providers"].items():
                for name, entry in record["files"].items():
                    if sha(archive.read("colabcola/_vendor/" + vendor + "/" + name)) != entry["sha256"]:
                        raise ValueError("bundled provider hash mismatch")
            metadata = archive.read(next(n for n in names if n.endswith(".dist-info/METADATA"))).decode()
            if "Requires-Dist: google-colab-cli" in metadata or "Requires-Dist: jupyter-kernel-client" in metadata:
                raise ValueError("wheel requires an unpublished provider distribution")
            evidence["wheel_members"] = names
    evidence["distributions"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
                                 for p in sorted(output.iterdir()) if p.suffix == ".whl" or p.name.endswith(".tar.gz")}
    (output / "build-provenance.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": "built", "output": str(output), "distributions": evidence["distributions"]}))


if __name__ == "__main__":
    main()
