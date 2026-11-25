"""Portable declarations for named workload environments and ordered steps.

These checks inspect declarations only. They never resolve dependencies or execute
project code on the packaging host.
"""
from __future__ import annotations

import posixpath
import re
from urllib.parse import urlsplit

from .errors import SpecError

_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_PYTHON = re.compile(r"python(?:3(?:\.\d+)?)?(?:\.exe)?\Z")
_SECRET = re.compile(r"(?:^|_)(?:api_?key|auth|credential|password|private_?key|secret|token)(?:_|$)", re.I)
_PROTECTED = {"PATH", "VIRTUAL_ENV", "__PYVENV_LAUNCHER__", "CUDA_HOME", "CUDA_PATH",
              "CUDACXX", "CUDAHOSTCXX", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH",
              "LIBRARY_PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "BASH_ENV", "ENV"}
_SHELLS = {"bash", "sh", "dash", "zsh", "fish", "cmd", "powershell", "pwsh"}
REQUIREMENT_FILE_BYTES = 1024 * 1024


def _object(value, where, allowed, required=()):
    if not isinstance(value, dict):
        raise SpecError(where + " must be a table/object")
    if any(not isinstance(key, str) for key in value):
        raise SpecError(where + " must have string keys")
    if set(value) - set(allowed):
        raise SpecError(where + ": unknown key " + str(sorted(set(value) - set(allowed))[0]))
    if set(required) - set(value):
        raise SpecError(where + ": missing " + sorted(set(required) - set(value))[0])
    return value


def _text(value, where, *, empty=False):
    if not isinstance(value, str) or (not value and not empty) or any(c in value for c in "\0\r\n"):
        raise SpecError(where + " must be a string without control line breaks")
    if len(value) > 8192:
        raise SpecError(where + " exceeds the declaration length limit")
    return value


def _strings(value, where):
    if not isinstance(value, list) or len(value) > 1024:
        raise SpecError(where + " must be an array of at most 1024 strings")
    return [_text(v, where + "[]") for v in value]


def _slug(value, where):
    if not isinstance(value, str) or len(value) > 64 or not _SLUG.fullmatch(value):
        raise SpecError(where + " must be a lowercase hyphen-separated slug of at most 64 characters")
    return value


def relative_path(value, where, *, dot=False):
    value = _text(value, where)
    if value == "." and dot:
        return value
    if (value.startswith("/") or "\\" in value or ":" in value
            or any(p in ("", ".", "..") for p in value.split("/"))):
        raise SpecError(where + " must be a normalized project-relative POSIX path")
    return value


def _https(value, where, *, fragment=False):
    value = _text(value, where)
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise SpecError(where + " must be a public HTTPS URL without credentials") from None
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.query or (parsed.fragment and not fragment)
            or any(c.isspace() for c in value) or "\\" in value):
        raise SpecError(where + " must be an HTTPS URL without credentials, query parameters or unsupported fragments")
    return value


def normalize_environment_variables(value, where="environment.env"):
    if not isinstance(value, dict) or len(value) > 128:
        raise SpecError(where + " must be a table of at most 128 string variables")
    if any(not isinstance(key, str) for key in value):
        raise SpecError(where + " has an invalid variable name")
    result = {}
    for key in sorted(value):
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise SpecError(where + " has an invalid variable name")
        text = _text(value[key], where + "." + key, empty=True)
        if _SECRET.search(key):
            raise SpecError(where + "." + key + " looks secret-bearing")
        if (key in _PROTECTED or key.startswith(("CONDA", "MAMBA", "PIP_", "COLABCOLA_", "LD_"))
                or (key.startswith("PYTHON") and key not in ("PYTHONUNBUFFERED", "PYTHONPATH"))):
            raise SpecError(where + "." + key + " overrides managed environment isolation")
        if key == "PYTHONPATH":
            for path in text.split(":"):
                relative_path(path, where + ".PYTHONPATH", dot=True)
        result[key] = text
    return result


def normalize_pip_package(value, where="pip.packages[]"):
    """Accept index requirements, packaged local paths and immutable Git refs."""
    value = _text(value, where).strip()
    if value.startswith("./"):
        relative_path(value[2:], where)
        return value
    if value.startswith("-"):
        raise SpecError(where + ": pip options are not package declarations")
    from packaging.requirements import Requirement, InvalidRequirement
    try:
        requirement = Requirement(value)
    except InvalidRequirement:
        raise SpecError(where + ": invalid package requirement") from None
    if requirement.url:
        url = requirement.url
        if not url.startswith("git+https://"):
            raise SpecError(where + ": direct references require git+https and an immutable 40-digit commit")
        _https(url[4:], where, fragment=True)
        location, _, fragment = url.partition("#")
        if not re.search(r"@[0-9a-fA-F]{40}$", location):
            raise SpecError(where + ": Git references require an immutable 40-digit commit")
        if fragment:
            if not fragment.startswith("subdirectory="):
                raise SpecError(where + ": only a Git subdirectory fragment is supported")
            subdirectory = fragment[len("subdirectory="):]
            if not re.fullmatch(r"[A-Za-z0-9_./-]+", subdirectory):
                raise SpecError(where + ": Git subdirectory must be an unescaped relative path")
            relative_path(subdirectory, where + " subdirectory")
    return str(requirement)


def requirement_file_packages(text, where="requirements"):
    """Validate a bounded, self-contained requirements file; preserve install order."""
    if not isinstance(text, str) or len(text.encode("utf-8")) > REQUIREMENT_FILE_BYTES:
        raise SpecError(where + " exceeds the 1 MiB requirements limit")
    result = []
    for number, raw in enumerate(text.replace("\\\r\n", " ").replace("\\\n", " ").splitlines(), 1):
        # A Git #subdirectory fragment is part of its URL, unlike a whitespace comment.
        line = re.split(r"\s+#", raw, maxsplit=1)[0].strip()
        if not line or line.startswith("#"):
            continue
        label = f"{where}:{number}"
        parts = re.split(r"\s+--hash=", line)
        package = normalize_pip_package(parts[0], label)
        for digest in parts[1:]:
            if not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest):
                raise SpecError(label + ": only SHA-256 requirement hashes are supported")
        if len(parts) > 1 and (package.startswith("./") or " @ " in package):
            raise SpecError(label + ": hashes apply to index requirements only")
        result.append(package + "".join(" --hash=" + digest for digest in parts[1:]))
    return result


def normalize_environments(value):
    if not isinstance(value, dict) or not 1 <= len(value) <= 16:
        raise SpecError("environments must contain 1 to 16 named environments")
    if any(not isinstance(name, str) for name in value):
        raise SpecError("environment names must be strings")
    result = {}
    for name in sorted(value):
        _slug(name, "environment name")
        where = "environments." + name
        item = _object(value[name], where, {"python", "conda", "pip", "cuda", "env", "cache"}, {"python"})
        version = _text(item["python"], where + ".python")
        if not re.fullmatch(r"3\.(?:[89]|[1-9][0-9])(?:\.[0-9]+)?", version):
            raise SpecError(where + ".python must be a Python 3.8+ minor or patch version")
        conda = _object(item.get("conda", {}), where + ".conda", {"channels", "packages"})
        channels = _strings(conda.get("channels", ["conda-forge"]), where + ".conda.channels")
        if not channels:
            raise SpecError(where + ".conda.channels must not be empty")
        for channel in channels:
            if channel.startswith("https:"):
                _https(channel, where + ".conda.channels")
            elif not re.fullmatch(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_.-]+)*", channel):
                raise SpecError(where + ": invalid conda channel")
        packages = _strings(conda.get("packages", []), where + ".conda.packages")
        for package in packages:
            if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*(?:[ =<>!~][A-Za-z0-9_.+*=<>!~,| -]*)?", package):
                raise SpecError(where + ": conda packages must be package/version declarations")
            if re.split(r"[ =<>!~]", package, maxsplit=1)[0].lower() == "python":
                raise SpecError(where + ": declare the Python version in python, not conda.packages")
        pip = _object(item.get("pip", {}), where + ".pip",
                      {"packages", "requirements", "wheels", "index_url", "extra_index_urls", "no_build_isolation"})
        pip_packages = [normalize_pip_package(p, where + ".pip.packages")
                        for p in _strings(pip.get("packages", []), where + ".pip.packages")]
        requirements = [relative_path(p, where + ".pip.requirements")
                        for p in _strings(pip.get("requirements", []), where + ".pip.requirements")]
        indexes = [_https(p, where + ".pip.extra_index_urls")
                   for p in _strings(pip.get("extra_index_urls", []), where + ".pip.extra_index_urls")]
        no_build = pip.get("no_build_isolation", False)
        if not isinstance(no_build, bool):
            raise SpecError(where + ".pip.no_build_isolation must be boolean")
        normalized_pip = {"packages": pip_packages, "requirements": requirements,
                          "extra_index_urls": indexes, "no_build_isolation": no_build}
        wheels = [relative_path(p, where + ".pip.wheels")
                  for p in _strings(pip.get("wheels", []), where + ".pip.wheels")]
        if len(set(p.casefold() for p in wheels)) != len(wheels):
            raise SpecError(where + ".pip.wheels contains duplicate wheel paths")
        from packaging.utils import parse_wheel_filename, InvalidWheelFilename
        wheel_names = set()
        for target in wheels:
            if target.split("/")[0].casefold() == "_colabcola":
                raise SpecError(where + ".pip.wheels cannot use reserved packaging paths")
            try:
                distribution, _, _, _ = parse_wheel_filename(target.rsplit("/", 1)[-1])
            except InvalidWheelFilename:
                raise SpecError(where + ".pip.wheels must name valid .whl files") from None
            if distribution in wheel_names:
                raise SpecError(where + ".pip.wheels contains multiple wheels for " + distribution)
            wheel_names.add(distribution)
        if wheels:
            normalized_pip["wheels"] = wheels
        if "index_url" in pip:
            normalized_pip["index_url"] = _https(pip["index_url"], where + ".pip.index_url")
        normalized = {"python": version, "conda": {"channels": channels, "packages": packages},
                      "pip": normalized_pip, "env": normalize_environment_variables(item.get("env", {}), where + ".env")}
        if "cache" in item:
            if type(item["cache"]) is not bool:
                raise SpecError(where + ".cache must be boolean")
            normalized["cache"] = item["cache"]
        if "cuda" in item:
            cuda = _text(item["cuda"], where + ".cuda")
            if not re.fullmatch(r"[1-9][0-9]*\.[0-9]+", cuda):
                raise SpecError(where + ".cuda must be a major.minor toolkit version")
            normalized["cuda"] = cuda
        result[name] = normalized
    return result


def shell_script_path(command, cwd="."):
    """Return the packaged path for the only supported shell command form."""
    if not command or command[0] != "bash":
        return None
    if len(command) < 2 or not isinstance(command[1], str):
        raise SpecError("bash requires a packaged .sh script, not shell options or a command string")
    script = command[1]
    if not script.endswith(".sh") or script.startswith("-"):
        raise SpecError("bash requires a packaged .sh script, not shell options or a command string")
    script = relative_path(script, "Bash script")
    cwd = relative_path(cwd, "Bash working directory", dot=True)
    target = script if cwd == "." else cwd + "/" + script
    if target.split("/")[0] == "_colabcola":
        raise SpecError("Bash script must be inside the packaged project")
    return target


def step_script_path(step):
    """Return the hash-bound script path, or None for modules/console commands."""
    command = step["command"]
    shell = shell_script_path(command, step["cwd"])
    if shell is not None:
        return shell
    args = command[1:] if _PYTHON.fullmatch(command[0]) else command
    if not args or args[0] == "-m" or not args[0].endswith(".py"):
        return None
    value = args[0]
    if value.startswith("/") or "\\" in value or ":" in value:
        raise SpecError("step scripts must be inside the packaged project")
    target = posixpath.normpath(posixpath.join(step["cwd"], value))
    return relative_path(target, "step script")


def normalize_steps(value, environments):
    if not isinstance(value, list) or not 1 <= len(value) <= 128:
        raise SpecError("steps must be an ordered array of 1 to 128 steps")
    result = []
    names = set()
    for index, raw in enumerate(value):
        where = f"steps[{index}]"
        item = _object(raw, where, {"name", "environment", "command", "cwd"}, {"name", "environment", "command"})
        name = _slug(item["name"], where + ".name")
        if name in names:
            raise SpecError("duplicate step name: " + name)
        names.add(name)
        environment = _slug(item["environment"], where + ".environment")
        if environment not in environments:
            raise SpecError(where + " refers to unknown environment " + environment)
        command = item["command"]
        if not isinstance(command, list) or not command or len(command) > 4096:
            raise SpecError(where + ".command must be a nonempty argv array")
        if any(not isinstance(v, str) or "\0" in v or len(v) > 8192 for v in command):
            raise SpecError(where + ".command must contain strings without NUL bytes, each at most 8192 characters")
        python = bool(_PYTHON.fullmatch(command[0]))
        args = command[1:] if python else command
        if not args or not args[0]:
            raise SpecError(where + ".command needs a script, module or console command")
        if command[0] == "bash":
            # The script and cwd are validated together by step_script_path.
            pass
        elif args[0] == "-m":
            if len(args) < 2 or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", args[1]):
                raise SpecError(where + ": python -m needs a dotted module name")
        elif python and not args[0].endswith(".py"):
            raise SpecError(where + ": Python commands must run a .py script or -m MODULE")
        elif not args[0].endswith(".py"):
            program = args[0].lower().removesuffix(".exe")
            if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", args[0]) or program in _SHELLS:
                raise SpecError(where + ": use a portable installed console command or bash with a packaged .sh script; other shell programs are not supported")
        step = {"name": name, "environment": environment, "command": list(command),
                "cwd": relative_path(item.get("cwd", "."), where + ".cwd", dot=True)}
        step_script_path(step)
        result.append(step)
    return result
