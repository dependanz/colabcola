"""Portable invocation of a declared Bash script in the extracted project."""
import hashlib
import re
from pathlib import Path


def packaged_bash_argv(argv, job_root, cwd=None, *, expected_sha256=None):
    """Return fixed-interpreter argv, never a shell command string.

    The bundle binds the original script bytes; recheck the path at dispatch so
    a missing file or a link introduced during execution cannot redirect it.
    This is not a sandbox for code inside the script.
    """
    if (not isinstance(argv, (list, tuple)) or len(argv) < 2
            or argv[0] != "bash"
            or any(not isinstance(v, str) or "\0" in v for v in argv)):
        raise RuntimeError("unsupported Bash command: expected bash SCRIPT.sh [args...]")
    script = argv[1]
    if (not script.endswith(".sh") or script.startswith(("-", "/"))
            or any(c in script for c in "\\:\r\n")
            or any(part in ("", ".", "..") for part in script.split("/"))):
        raise RuntimeError("Bash script must be a normalized project-relative .sh file")
    root = Path(job_root).resolve()
    directory = root if cwd is None else Path(cwd)
    if not directory.is_absolute():
        directory = root / directory
    try:
        relative = directory.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("Bash working directory is outside the packaged project") from exc
    parts = (*relative.parts, *script.split("/"))
    if ".." in parts or parts[0] == "_colabcola":
        raise RuntimeError("Bash script must be inside the packaged project")
    target = root
    for part in parts:
        target /= part
        if target.is_symlink():
            raise RuntimeError("Bash script path contains a symlink")
    if not target.is_file() or not target.resolve().is_relative_to(root):
        raise RuntimeError("Bash script is not a regular file in the packaged project")
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
        raise RuntimeError("Bash script has no frozen SHA-256 identity")
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError("Bash script content differs from its frozen SHA-256 identity")
    interpreter = Path("/bin/bash")
    if not interpreter.is_file():
        raise RuntimeError("Bash is unavailable at the runtime's /bin/bash")
    # -p ignores BASH_ENV, ENV, inherited shell functions and startup option
    # variables even if an activation hook populated them. The script itself
    # is explicit, and its arguments are passed separately, never interpolated.
    return [str(interpreter), "--noprofile", "--norc", "-p", "--", str(target), *argv[2:]]
