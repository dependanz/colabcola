"""Build local development wheels; never installs, authenticates or publishes.

Requires build, setuptools, wheel, hatchling, hatch-vcs in a disposable build env.
Pinned provider repositories are exported with git archive (no working-tree edits).
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile

CLI_COMMIT = "79cd126383e72b1e911f20b7030e4247f7861704"
KERNEL_COMMIT = "f18e982c3265df5e923aa9def101ab3fd737e139"
CLI_VERSION = "0.1.dev70+g79cd12638.colabcola1"
KERNEL_VERSION = "0.8.0+colabcola.f18e982c"


def sha(body):
    return hashlib.sha256(body).hexdigest()


def export(repo, commit, destination):
    body = subprocess.check_output(["git", "-c", "core.autocrlf=false", "-C", str(repo), "archive", "--format=tar", commit])
    with tarfile.open(fileobj=io.BytesIO(body)) as archive:
        archive.extractall(destination, filter="data")
    return sha(body)


def build(source, output, *formats):
    subprocess.run([sys.executable, "-m", "build", "--no-isolation", *formats,
                    "--outdir", str(output), str(source)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli-source", type=Path, required=True)
    parser.add_argument("--kernel-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("*.whl")) or any(output.glob("*.tar.gz")):
        parser.error("output already contains distributions; use a fresh directory")
    root = Path(__file__).resolve().parents[1]
    evidence = {"provider_changes": "Local version labels and CLI dependency metadata only; runtime implementation unchanged.", "sources": {}}
    with tempfile.TemporaryDirectory(prefix="colabcola-build-") as temp:
        temp = Path(temp)
        for name, repo, commit in (("google-colab-cli", args.cli_source, CLI_COMMIT),
                                   ("jupyter-kernel-client", args.kernel_source, KERNEL_COMMIT)):
            target = temp / name
            target.mkdir()
            evidence["sources"][name] = {"commit": commit, "git_archive_sha256": export(repo, commit, target)}
        cli = temp / "google-colab-cli"
        p = cli / "pyproject.toml"
        text = p.read_text(encoding="utf-8")
        assert text.count('dynamic = ["version"]') == 1
        assert text.count('jupyter-kernel-client==0.9.0') == 1
        text = text.replace('dynamic = ["version"]', f'version = "{CLI_VERSION}"').replace(
            'jupyter-kernel-client==0.9.0', f'jupyter-kernel-client=={KERNEL_VERSION}')
        p.write_text(text, encoding="utf-8")
        kernel = temp / "jupyter-kernel-client"
        p = kernel / "jupyter_kernel_client/_version.py"
        text = p.read_text(encoding="utf-8")
        assert text.count('__version__ = "0.8.0"') == 1
        p.write_text(text.replace('__version__ = "0.8.0"', f'__version__ = "{KERNEL_VERSION}"'), encoding="utf-8")
        for name, source, package in (("google-colab-cli", cli, "src/colab_cli"),
                                      ("jupyter-kernel-client", kernel, "jupyter_kernel_client")):
            evidence["sources"][name]["package_sha256"] = {
                str(p.relative_to(source / package)).replace("\\", "/"): sha(p.read_bytes())
                for p in sorted((source / package).rglob("*.py"))}
            build(source, output, "--wheel")
        # Build from a deliberate staging allowlist; never capture jobs or user data.
        stage = temp / "colabcola"
        stage.mkdir()
        paths = [p for p in (root / "src/colabcola").rglob("*")
                 if p.is_file() and p.suffix in (".py", ".json") and "__pycache__" not in p.parts]
        paths += [root / name for name in ("pyproject.toml", "MANIFEST.in", "README.md", "DEVELOPMENT.md",
                                           "PACKAGING.md", "LIFECYCLE.md", "tools/build_development.py")]
        evidence["sources"]["colabcola"] = {"files": {}}
        for path in paths:
            relative = path.relative_to(root)
            dest = stage / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            body = path.read_bytes()
            dest.write_bytes(body)
            evidence["sources"]["colabcola"]["files"][relative.as_posix()] = sha(body)
        build(stage, output)  # Default: sdist, then wheel built from that sdist.
    evidence["distributions"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
                                 for p in sorted(output.iterdir()) if p.suffix == ".whl" or p.name.endswith(".tar.gz")}
    for wheel in output.glob("*.whl"):
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            if wheel.name.startswith("colabcola-"):
                assert all(n.startswith(("colabcola/", "colabcola-")) for n in names), names
                assert not any(n.endswith((".colajob", ".zip", ".pt", ".pth", ".pkl")) for n in names)
                evidence["wheel_members"] = names
            else:
                assert any("license" in n.lower() for n in names), "Provider license missing"
                name = "google-colab-cli" if wheel.name.startswith("google_colab_cli") else "jupyter-kernel-client"
                prefix = "colab_cli/" if name == "google-colab-cli" else "jupyter_kernel_client/"
                for relative, expected in evidence["sources"][name]["package_sha256"].items():
                    # Upstream excludes its internal tests from the wheel.
                    if prefix + relative in names:
                        assert sha(archive.read(prefix + relative)) == expected, relative
    (output / "build-provenance.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": "built", "output": str(output), "distributions": evidence["distributions"]}, indent=2))


if __name__ == "__main__":
    main()
