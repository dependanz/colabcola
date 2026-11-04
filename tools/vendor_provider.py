"""Refresh the private, pinned provider subset from local Git objects.

This never fetches, installs, authenticates or publishes. Only import/module names
are relocated; original hashes, relocated hashes and licenses are retained.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

CLI_COMMIT = "79cd126383e72b1e911f20b7030e4247f7861704"
KERNEL_COMMIT = "f18e982c3265df5e923aa9def101ab3fd737e139"
CLI_FILES = ("client.py", "consumption.py", "utils.py", "state.py", "runtime.py", "oauth_config.json")
MODIFICATION_NOTICE = b"# Modified for Colabcola: private module namespace relocation; see ../provenance.json.\n"


def sha(body):
    return hashlib.sha256(body).hexdigest()


def relocated(body):
    text = body.decode("utf-8")
    for name in ("colab_cli", "jupyter_kernel_client"):
        text = text.replace(name, "colabcola._vendor." + name)
    result = text.encode("utf-8")
    return MODIFICATION_NOTICE + result if result != body else result


def export(repo, commit):
    data = subprocess.check_output(["git", "-c", "core.autocrlf=false", "-C", str(repo), "archive", "--format=tar", commit])
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        return {member.name: archive.extractfile(member).read()
                for member in archive.getmembers() if member.isfile()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli-source", type=Path, required=True)
    parser.add_argument("--kernel-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "src/colabcola/_vendor")
    args = parser.parse_args(argv)
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": 1, "changes": "Private namespace relocation only; no provider behavior changes.", "providers": {}}
    for name, repo, commit, prefix, version in (
        ("colab_cli", args.cli_source, CLI_COMMIT, "src/colab_cli/", "0.1.dev70"),
        ("jupyter_kernel_client", args.kernel_source, KERNEL_COMMIT, "jupyter_kernel_client/", "0.8.0"),
    ):
        files = export(repo, commit)
        selected = sorted(k for k in files if k.startswith(prefix) and "/" not in k[len(prefix):]
                          and (k[len(prefix):] in CLI_FILES if name == "colab_cli" else k.endswith(".py")))
        target = root / name
        target.mkdir(exist_ok=True)
        entries = {}
        for source in selected:
            relative = source[len(prefix):]
            original = files[source]
            body = relocated(original) if relative.endswith(".py") else original
            (target / relative).write_bytes(body)
            entries[relative] = {"source": source, "upstream_sha256": sha(original), "sha256": sha(body)}
        if name == "colab_cli":
            body = b'"""Private Google Colab client subset; see ../provenance.json."""\n'
            (target / "__init__.py").write_bytes(body)
            entries["__init__.py"] = {"generated": True, "sha256": sha(body)}
        license_body = files["LICENSE"]
        (target / "LICENSE").write_bytes(license_body)
        entries["LICENSE"] = {"source": "LICENSE", "upstream_sha256": sha(license_body), "sha256": sha(license_body)}
        repository = "https://github.com/googlecolab/" + ("google-colab-cli" if name == "colab_cli" else "jupyter-kernel-client")
        manifest["providers"][name] = {"commit": commit, "repository": repository, "version": version, "files": entries}
    (root / "__init__.py").write_text('"""Private dependencies bundled with Colabcola; not a public import API."""\n', encoding="utf-8")
    (root / "provenance.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"state": "vendored", "providers": {name: len(data["files"]) for name, data in manifest["providers"].items()}}))


if __name__ == "__main__":
    main()
