"""Self-contained named-environment helpers embedded in compiled notebooks.

Only the standard library is available in the control interpreter. Package
installation and workload commands always use the selected environment.
"""
import datetime as _cc_datetime
import hashlib
import json
import os
import pathlib
import platform
import re
import selectors
import shutil
import signal
import subprocess
import time
import urllib.request


_COLABCOLA_ENV_LOG_BYTES = 8 * 1024 * 1024
_COLABCOLA_ENV_CAPTURE_BYTES = 8 * 1024 * 1024
_COLABCOLA_ENV_SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_COLABCOLA_ENV_PROTECTED = {
    "PATH", "VIRTUAL_ENV", "__PYVENV_LAUNCHER__", "CUDA_HOME", "CUDA_PATH",
    "CUDACXX", "CUDAHOSTCXX", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH",
    "LIBRARY_PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "BASH_ENV", "ENV",
}


def _colabcola_environment_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    data = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _colabcola_env_now():
    return _cc_datetime.datetime.now(_cc_datetime.timezone.utc).isoformat()


def _colabcola_env_log(path, data):
    if path is None:
        return
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    size = path.stat().st_size if path.exists() else 0
    remaining = max(0, _COLABCOLA_ENV_LOG_BYTES - size)
    if remaining:
        with path.open("ab") as stream:
            if len(data) > remaining:
                marker = b"\n[colabcola] Environment log limit reached; further output omitted.\n"
                stream.write(data[:max(0, remaining - len(marker))])
                stream.write(marker[:remaining])
            else:
                stream.write(data)


def _colabcola_env_timeout(deadline, default):
    remaining = default if deadline is None else min(default, deadline - time.time())
    if remaining <= 0:
        raise TimeoutError("environment stage deadline reached")
    return remaining


def _colabcola_env_stop(process, group=True):
    """Bound teardown even if a descendant kept the output pipe open."""
    for signum, wait in ((signal.SIGTERM, 0.3), (signal.SIGKILL, 2.0)):
        try:
            if group and os.name != "nt":
                os.killpg(process.pid, signum)
            elif process.poll() is None:
                process.send_signal(signum)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            continue
        if signum == signal.SIGKILL:
            break


def _colabcola_env_run(argv, *, cwd, env, deadline=None, timeout=1200,
                       log_path=None, label="environment command", capture=False):
    """Drain a pipe without threads, with one deadline covering the process and EOF."""
    duration = _colabcola_env_timeout(deadline, timeout)
    end = time.monotonic() + duration
    _colabcola_env_log(log_path, ("\n[colabcola] " + label + "\n").encode("utf-8"))
    process = subprocess.Popen([str(x) for x in argv], cwd=str(cwd), env=env,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=0, start_new_session=(os.name != "nt"))
    output = bytearray()
    tail = bytearray()
    selector = selectors.DefaultSelector()
    successful = False
    exited_at = None
    try:
        os.set_blocking(process.stdout.fileno(), False)
        selector.register(process.stdout, selectors.EVENT_READ)
        while selector.get_map() or process.poll() is None:
            now = time.monotonic()
            if now >= end or (deadline is not None and time.time() >= deadline):
                raise TimeoutError(label + " exceeded its deadline")
            if process.poll() is not None:
                exited_at = exited_at or now
                if now - exited_at > 2:
                    raise RuntimeError(label + " left an output pipe open after exiting")
            for key, _ in selector.select(min(0.1, max(0, end - now))):
                try:
                    block = os.read(key.fd, 65536)
                except BlockingIOError:
                    continue
                except OSError as exc:
                    raise RuntimeError(label + " output drain failed") from exc
                if not block:
                    selector.unregister(key.fileobj)
                    continue
                _colabcola_env_log(log_path, block)
                tail.extend(block)
                del tail[:-4000]
                if capture:
                    output.extend(block)
                    if len(output) > _COLABCOLA_ENV_CAPTURE_BYTES:
                        raise RuntimeError(label + " probe output exceeded its bound")
        code = process.wait(timeout=max(0.01, end - time.monotonic()))
        if code:
            raise RuntimeError(label + " failed (exit " + str(code) + "): "
                               + tail.decode("utf-8", errors="replace")[-2000:])
        successful = True
        return bytes(output if capture else tail).decode("utf-8", errors="replace")
    finally:
        if not successful:
            _colabcola_env_stop(process)
        selector.close()
        process.stdout.close()


def _colabcola_env_declared(declared, job_root):
    result = {}
    root = pathlib.Path(job_root).resolve()
    for name, value in (declared or {}).items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise RuntimeError("invalid declared environment variable")
        if not isinstance(value, str) or any(c in value for c in "\0\r\n"):
            raise RuntimeError("invalid declared environment variable value")
        if (name in _COLABCOLA_ENV_PROTECTED
                or name.startswith(("CONDA", "MAMBA", "PIP_", "COLABCOLA_", "LD_"))
                or (name.startswith("PYTHON") and name not in ("PYTHONPATH", "PYTHONUNBUFFERED"))):
            raise RuntimeError("managed environment forbids override: " + name)
        if name == "PYTHONPATH":
            paths = []
            for item in value.split(":"):
                candidate = pathlib.PurePosixPath(item)
                if not item or candidate.is_absolute() or ".." in candidate.parts or "\\" in item:
                    raise RuntimeError("PYTHONPATH must contain project-relative directories")
                path = (root / item).resolve()
                path.relative_to(root)
                paths.append(str(path))
            value = os.pathsep.join(paths)
        result[name] = value
    return result


def _colabcola_env_network_environment(env):
    # Managed runtimes can require a proxy to reach package indexes. Keep this
    # transport configuration in build subprocesses only; it is never serialized
    # into descriptors or resolved facts and is not inherited by workload steps.
    result = dict(env)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                 "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        if name in os.environ:
            result[name] = os.environ[name]
    return result


def _colabcola_env_clean_environment(prefix=None, root_prefix=None, network=False):
    # Build and workload processes inherit no authentication helpers, Python/pip
    # config, shell startup hooks, or caller compiler/loader overrides.
    keep = {"HOME", "USER", "LOGNAME", "TMPDIR", "TMP", "TEMP", "LANG", "TERM",
            "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES"}
    env = {key: value for key, value in os.environ.items() if key in keep or key.startswith("LC_")}
    env.update(PATH=_COLABCOLA_ENV_SYSTEM_PATH, PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
               PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1",
               MAMBA_NO_RC="true", MAMBA_NO_BANNER="true")
    if prefix:
        env["PATH"] = str(pathlib.Path(prefix) / "bin") + os.pathsep + env["PATH"]
    if root_prefix:
        env["MAMBA_ROOT_PREFIX"] = str(root_prefix)
        env["XDG_CACHE_HOME"] = str(pathlib.Path(root_prefix) / "cache")
    return _colabcola_env_network_environment(env) if network else env


def _colabcola_env_download(manager_spec, destination, deadline):
    destination = pathlib.Path(destination)
    if destination.exists():
        raise RuntimeError("environment manager destination already exists")
    if (manager_spec.get("kind") != "micromamba-linux-x86_64"
            or not str(manager_spec.get("url", "")).startswith("https://github.com/mamba-org/micromamba-releases/releases/download/")
            or not re.fullmatch(r"[a-f0-9]{64}", str(manager_spec.get("sha256", "")))
            or not isinstance(manager_spec.get("bytes"), int)
            or not 0 < manager_spec["bytes"] <= 64 * 1024 * 1024):
        raise RuntimeError("invalid frozen environment manager")
    total_deadline = min(time.time() + 300, deadline) if deadline is not None else time.time() + 300
    digest = hashlib.sha256()
    size = 0
    try:
        # read1 bounds each read to one socket operation, so a slow peer cannot
        # reset the overall bootstrap deadline by continually sending small chunks.
        request = urllib.request.Request(manager_spec["url"], headers={"User-Agent": "colabcola"})
        with urllib.request.urlopen(request, timeout=_colabcola_env_timeout(total_deadline, 30)) as response, destination.open("xb") as output:
            while True:
                remaining = _colabcola_env_timeout(total_deadline, 30)
                raw = getattr(response, "fp", None)
                raw = getattr(raw, "raw", None)
                sock = getattr(raw, "_sock", None)
                if sock is not None:
                    sock.settimeout(remaining)
                block = response.read1(1024 * 1024) if hasattr(response, "read1") else response.read(65536)
                if not block:
                    break
                size += len(block)
                if size > manager_spec["bytes"]:
                    raise RuntimeError("environment manager exceeds its declared byte count")
                digest.update(block)
                output.write(block)
        if size != manager_spec["bytes"] or digest.hexdigest() != manager_spec["sha256"]:
            raise RuntimeError("environment manager size/SHA-256 mismatch")
        destination.chmod(0o700)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _colabcola_env_activation(descriptor, command):
    return [descriptor["manager"], "--no-rc", "--no-env", "--root-prefix", descriptor["root_prefix"],
            "run", "--prefix", descriptor["prefix"], *command]


_COLABCOLA_ENV_PROBE = r'''
import importlib.metadata as metadata,json,os,platform,site,sys
facts={'python':platform.python_version(),'executable':sys.executable,'prefix':sys.prefix,
       'base_prefix':sys.base_prefix,'conda_prefix':os.environ.get('CONDA_PREFIX'),
       'user_site_enabled':site.ENABLE_USER_SITE,
       'packages':sorted([{'name':d.metadata.get('Name',''), 'version':d.version}
                          for d in metadata.distributions()],key=lambda p:p['name'].lower())}
if sys.argv[1]=='cuda':
    try: metadata.version('torch')
    except metadata.PackageNotFoundError: pass
    else:
        import torch
        facts['torch']={'version':str(torch.__version__),'cuda':torch.version.cuda}
print(json.dumps(facts,sort_keys=True))
'''


def _colabcola_env_probe(descriptor, job_root, deadline, log_path, cuda=False):
    command, env = _colabcola_environment_command(descriptor, ["python", "-I", "-c", _COLABCOLA_ENV_PROBE, "cuda" if cuda else "basic"], job_root)
    output = _colabcola_env_run(command, cwd=job_root, env=env, deadline=deadline,
        timeout=120, log_path=log_path, label="verify interpreter", capture=True)
    try:
        facts = json.loads(output.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError("environment interpreter did not produce valid identity evidence") from exc
    prefix = pathlib.Path(descriptor["prefix"]).resolve()
    if (pathlib.Path(facts["prefix"]).resolve() != prefix
            or pathlib.Path(facts.get("conda_prefix") or "/").resolve() != prefix
            or facts["user_site_enabled"] is not False):
        raise RuntimeError("environment activation or interpreter boundary failed")
    try:
        pathlib.Path(facts["executable"]).resolve().relative_to(prefix)
    except ValueError as exc:
        raise RuntimeError("environment selected an external interpreter") from exc
    requested = descriptor["definition"]["python"]
    actual = facts["python"]
    if (actual != requested if requested.count(".") == 2 else actual.rsplit(".", 1)[0] != requested):
        raise RuntimeError("resolved Python version does not match its environment declaration")
    return facts


def _colabcola_env_toolchain(prefix):
    prefix = pathlib.Path(prefix).resolve()
    nvcc = None
    for candidate in (prefix / "bin/nvcc", prefix / "targets/x86_64-linux/bin/nvcc"):
        if candidate.is_file():
            try:
                candidate.resolve().relative_to(prefix)
            except ValueError as exc:
                raise RuntimeError("CUDA compiler escaped the selected environment") from exc
            nvcc = str(candidate)
            break
    includes = [prefix / "include", prefix / "targets/x86_64-linux/include"]
    libraries = [prefix / "lib", prefix / "lib64", prefix / "targets/x86_64-linux/lib"]
    for site in sorted(prefix.glob("lib/python*/site-packages")):
        includes.extend(sorted(site.glob("nvidia/*/include")))
        libraries.extend(sorted(site.glob("nvidia/*/lib")))
        libraries.append(site / "torch/lib")
    def present(paths):
        result = []
        for path in paths:
            if path.is_dir() and path.name != "stubs":
                try:
                    path.resolve().relative_to(prefix)
                except ValueError as exc:
                    raise RuntimeError("environment toolchain directory escaped its prefix") from exc
                if str(path) not in result:
                    result.append(str(path))
        return result
    return {"cuda_home": str(prefix), "nvcc": nvcc,
            "include_dirs": present(includes), "library_dirs": present(libraries)}


def _colabcola_env_cuda_compatibility(definition, compiler_version, resolved=None):
    if definition.get("cuda") and compiler_version != definition["cuda"]:
        raise RuntimeError("environment CUDA compiler does not match the declared version")
    torch_cuda = (resolved or {}).get("torch", {}).get("cuda")
    if compiler_version and torch_cuda and compiler_version != ".".join(torch_cuda.split(".")[:2]):
        raise RuntimeError("environment CUDA compiler does not match the installed torch CUDA version")


def _colabcola_environment_command(descriptor, argv, job_root, declared=None, *, cwd=None, script_sha256=None):
    """Return an activated argv and an explicit, credential-free process environment."""
    if not isinstance(argv, (list, tuple)) or not argv or any(not isinstance(x, str) or "\0" in x for x in argv):
        raise RuntimeError("environment command must be a literal argv array")
    prefix = pathlib.Path(descriptor["prefix"]).resolve()
    python = prefix / "bin/python"
    first = argv[0]
    if first == "bash":
        if len(argv) < 2:
            raise RuntimeError("unsupported Bash command: expected bash SCRIPT.sh [args...]")
        # This function is also frozen into the standalone step launcher.
        if __package__:
            from .shell_runtime import packaged_bash_argv
        else:
            from shell_runtime import packaged_bash_argv
        invocation = packaged_bash_argv(argv, job_root, cwd, expected_sha256=script_sha256)
    elif re.fullmatch(r"python(?:3(?:\.\d+)?)?(?:\.exe)?", first):
        invocation = [str(python), *argv[1:]]
    elif first.endswith(".py") or first == "-m":
        invocation = [str(python), *argv]
    else:
        if "/" in first or "\\" in first or first.lower().removesuffix(".exe") in {"bash", "sh", "dash", "zsh", "fish", "cmd", "powershell", "pwsh"}:
            raise RuntimeError("unsupported named-environment command")
        executable = shutil.which(first, path=str(prefix / "bin"))
        if executable is None:
            raise RuntimeError("command is not installed in the selected environment: " + first)
        try:
            pathlib.Path(executable).resolve().relative_to(prefix)
        except ValueError as exc:
            raise RuntimeError("console command escaped the selected environment") from exc
        invocation = [executable, *argv[1:]]
    if invocation[0] == str(python):
        try:
            python.resolve().relative_to(prefix)
        except ValueError as exc:
            raise RuntimeError("Python executable escaped the selected environment") from exc
    env = _colabcola_env_clean_environment(prefix, descriptor["root_prefix"])
    env.update(_colabcola_env_declared(declared, job_root))
    env.update(_colabcola_env_declared(descriptor["definition"].get("env"), job_root))
    toolchain = descriptor.get("toolchain") or _colabcola_env_toolchain(prefix)
    # Prevent PyTorch's toolkit discovery from silently choosing a host toolkit.
    env.update(CUDA_HOME=toolchain["cuda_home"], CUDA_PATH=toolchain["cuda_home"])
    if toolchain.get("nvcc"):
        env["CUDACXX"] = toolchain["nvcc"]
        env["PATH"] = str(pathlib.Path(toolchain["nvcc"]).parent) + os.pathsep + env["PATH"]
    if toolchain["include_dirs"]:
        include_path = os.pathsep.join(toolchain["include_dirs"])
        env.update(CPATH=include_path, C_INCLUDE_PATH=include_path, CPLUS_INCLUDE_PATH=include_path)
    libraries = list(toolchain["library_dirs"])
    if libraries:
        env["LIBRARY_PATH"] = os.pathsep.join(libraries)
    # Driver libraries belong to the runtime; toolkit and wheel libraries belong
    # to the environment. Never add CUDA linker stubs to the runtime loader path.
    libraries.extend(str(path) for path in map(pathlib.Path,
        ("/usr/lib64-nvidia", "/usr/local/nvidia/lib", "/usr/local/nvidia/lib64")) if path.is_dir())
    if libraries:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(libraries)
    return _colabcola_env_activation(descriptor, invocation), env


def _colabcola_env_wheel_inputs(definition, job_root, wheel_records, deadline):
    """Verify each declared wheel against the authenticated bundle manifest."""
    root = pathlib.Path(job_root).resolve()
    verified = []
    for relative in definition["pip"].get("wheels", []):
        if (not isinstance(relative, str) or not relative.endswith(".whl")
                or any(c in relative for c in "\\:\0\r\n")
                or any(p in ("", ".", "..") for p in relative.split("/"))):
            raise RuntimeError("environment wheel must be a normalized project-relative .whl file")
        record = (wheel_records or {}).get(relative)
        if (not isinstance(record, dict) or record.get("role") != "wheel"
                or record.get("path") != relative or type(record.get("size")) is not int
                or record["size"] < 0 or not isinstance(record.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])):
            raise RuntimeError("environment wheel has no valid frozen identity: " + relative)
        path = root
        for part in relative.split("/"):
            path /= part
            if path.is_symlink():
                raise RuntimeError("environment wheel path contains a symlink: " + relative)
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise RuntimeError("environment wheel is missing or unsafe: " + relative)
        if path.stat().st_size != record["size"]:
            raise RuntimeError("environment wheel size differs from frozen identity: " + relative)
        digest = hashlib.sha256()
        count = 0
        with path.open("rb") as stream:
            while True:
                if deadline is not None and time.time() >= deadline:
                    raise TimeoutError("environment wheel verification exceeded its deadline")
                block = stream.read(1024 * 1024)
                if not block:
                    break
                count += len(block)
                if count > record["size"]:
                    raise RuntimeError("environment wheel grew during verification: " + relative)
                digest.update(block)
        if count != record["size"] or digest.hexdigest() != record["sha256"]:
            raise RuntimeError("environment wheel SHA-256 differs from frozen identity: " + relative)
        verified.append({"path": relative, "size": count, "sha256": digest.hexdigest()})
    return verified


def _colabcola_env_install_wheels(descriptor, job_root, wheel_records, deadline, log_path):
    verified = _colabcola_env_wheel_inputs(descriptor["definition"], job_root, wheel_records, deadline)
    if verified:
        # Do not resolve transitive dependencies or silently keep an index wheel
        # with the same version. The declared local wheel bytes are authoritative.
        argv = ["python", "-m", "pip", "install", "--no-input", "--no-index",
                "--no-deps", "--force-reinstall", *[str(pathlib.Path(job_root).resolve() / w["path"])
                                                     for w in verified]]
        command, env = _colabcola_environment_command(descriptor, argv, job_root)
        _colabcola_env_run(command, cwd=job_root, env=env, deadline=deadline,
                           log_path=log_path, label=descriptor["name"] + ": embedded wheels")
    return verified


def _colabcola_env_compiler_probe(descriptor, job_root, deadline, log_path, stage):
    name, definition = descriptor["name"], descriptor["definition"]
    if definition.get("cuda") and not descriptor["toolchain"].get("nvcc"):
        raise RuntimeError("environment " + name + " declares CUDA but does not provide its own nvcc")
    nvcc = descriptor["toolchain"].get("nvcc")
    compiler_version = None
    if nvcc:
        _, cuda_env = _colabcola_environment_command(descriptor, ["python"], job_root)
        nvcc_command = _colabcola_env_activation(descriptor, [nvcc, "--version"])
        compiler_output = stage("cuda-toolkit", lambda: _colabcola_env_run(nvcc_command,
            cwd=job_root, env=cuda_env, deadline=deadline, timeout=60, log_path=log_path,
            label=name + ": CUDA compiler", capture=True))
        match = re.search(r"release\s+([0-9]+\.[0-9]+)", compiler_output)
        if not match:
            raise RuntimeError("cannot determine the environment CUDA compiler version")
        compiler_version = match.group(1)
    _colabcola_env_cuda_compatibility(definition, compiler_version)
    return compiler_version


def _colabcola_env_resolved_facts(descriptor, job_root, deadline, log_path, stage, compiler_version):
    name, definition = descriptor["name"], descriptor["definition"]
    manager, root_prefix, prefix = descriptor["manager"], descriptor["root_prefix"], descriptor["prefix"]
    nvcc = descriptor["toolchain"].get("nvcc")
    command, env = _colabcola_environment_command(descriptor, ["python", "-m", "pip", "check"], job_root)
    stage("pip-check", lambda: _colabcola_env_run(command, cwd=job_root, env=env,
        deadline=deadline, timeout=120, log_path=log_path, label=name + ": pip check"))
    resolved = stage("resolved", lambda: _colabcola_env_probe(descriptor, job_root, deadline, log_path, cuda=True))
    conda_command = [str(manager), "--no-rc", "--no-env", "--root-prefix", str(root_prefix),
                     "list", "--prefix", str(prefix), "--json"]
    conda_output = stage("conda-packages", lambda: _colabcola_env_run(conda_command, cwd=job_root,
        env=_colabcola_env_clean_environment(root_prefix=root_prefix), deadline=deadline,
        timeout=120, log_path=log_path, label=name + ": resolved conda packages", capture=True))
    resolved["conda_packages"] = json.loads(conda_output)
    _colabcola_env_cuda_compatibility(definition, compiler_version, resolved)
    if compiler_version:
        resolved["cuda_compiler"] = {"path": nvcc, "version": compiler_version}
    return resolved


def _colabcola_prepare_environments(environments, manager_spec, job_root,
                                    environment_root, result_root, deadline=None, *, wheel_records=None, cache_context=None):
    """Build every declared environment and persist resolved evidence as it progresses."""
    job_root = pathlib.Path(job_root).resolve()
    environment_root = pathlib.Path(environment_root).resolve()
    result_root = pathlib.Path(result_root)
    evidence_path = result_root / "environments.json"
    log_path = result_root / "environment.log"
    descriptors = {}
    started = time.monotonic()
    facts = {"schema": 1, "state": "building", "started_at": _colabcola_env_now(),
             "manager": dict(manager_spec), "environments": {}, "failed_environment": None}
    active = None
    _colabcola_environment_json(evidence_path, facts)
    try:
        if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
            raise RuntimeError("named environments require a Linux x86_64 runtime")
        if not isinstance(environments, dict) or not 1 <= len(environments) <= 16:
            raise RuntimeError("invalid named environments in frozen manifest")
        for name, definition in environments.items():
            if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or len(name) > 64:
                raise RuntimeError("invalid environment name")
            facts["environments"][name] = {"definition": definition, "state": "pending", "stages": []}
        environment_root.mkdir(parents=True, exist_ok=True)
        manager = environment_root / "micromamba"
        root_prefix = environment_root / "root"
        _colabcola_env_download(manager_spec, manager, deadline)
        for name in sorted(environments):
            active = name
            definition = environments[name]
            record = facts["environments"][name]
            env_start = time.monotonic()
            record.update(state="building", started_at=_colabcola_env_now())
            _colabcola_environment_json(evidence_path, facts)
            selected = {}
            if definition.get("cache") is True:
                selected = (cache_context.choose(name, definition) if cache_context is not None else
                            {"report": {"state": "bypass", "reason": "managed_cache_storage_unavailable", "published": False}})
                record["cache"] = selected["report"]
                _colabcola_environment_json(evidence_path, facts)
            prefix = selected.get("prefix", environment_root / "envs" / name)
            if os.path.lexists(prefix):
                raise RuntimeError("environment prefix already exists; reuse requires a verified cache")
            descriptor = {"name": name, "manager": str(manager), "root_prefix": str(root_prefix),
                          "prefix": str(prefix), "definition": definition}
            def stage(label, operation):
                stage_start = time.monotonic()
                evidence = {"name": label, "state": "running", "started_at": _colabcola_env_now()}
                record["stages"].append(evidence)
                _colabcola_environment_json(evidence_path, facts)
                try:
                    value = operation()
                    evidence["state"] = "success"
                    return value
                except BaseException as exc:
                    evidence.update(state="failed", error={"type": type(exc).__name__, "message": str(exc)[:2000]})
                    raise
                finally:
                    evidence.update(elapsed_seconds=round(time.monotonic() - stage_start, 6), finished_at=_colabcola_env_now())
                    _colabcola_environment_json(evidence_path, facts)
            if selected.get("prefix") is not None:
                # Wheels are still verified on a hit, not just on installation.
                verified_wheels = _colabcola_env_wheel_inputs(definition, job_root, wheel_records, deadline)
                def validate_restored(restore_deadline):
                    descriptor["toolchain"] = _colabcola_env_toolchain(prefix)
                    direct = lambda label, operation: operation()
                    compiler = _colabcola_env_compiler_probe(descriptor, job_root, restore_deadline, log_path, direct)
                    return _colabcola_env_resolved_facts(descriptor, job_root, restore_deadline, log_path, direct, compiler)
                restored = stage("cache-restore", lambda: cache_context.restore(selected, validate_restored))
                if restored is not None:
                    elapsed = time.monotonic() - env_start
                    selected["report"].update(setup_seconds=elapsed,
                        estimated_seconds_saved=selected["report"]["cold_build_seconds"] - elapsed)
                    if verified_wheels:
                        record["embedded_wheels"] = verified_wheels
                    record.update(state="ready", facts=restored, toolchain=descriptor["toolchain"],
                        elapsed_seconds=round(elapsed, 6), finished_at=_colabcola_env_now())
                    descriptors[name] = descriptor
                    _colabcola_environment_json(evidence_path, facts)
                    print("[colabcola] Environment " + name + " restored from cache (Python " + restored["python"] + ")", flush=True)
                    continue
                # A bad cache is never run; the same stable prefix is rebuilt
                # only after the cache helper has removed its owned restoration.
                if os.path.lexists(prefix):
                    raise RuntimeError("cache restore left an unsafe environment prefix")
                descriptor.pop("toolchain", None)
            build_started = time.monotonic()
            print("[colabcola] Building environment " + name, flush=True)
            conda = definition["conda"]
            create = [str(manager), "--no-rc", "--no-env", "--root-prefix", str(root_prefix),
                      "create", "--yes", "--prefix", str(prefix), "--override-channels", "--strict-channel-priority"]
            for channel in conda["channels"]:
                create.extend(["--channel", channel])
            create.extend(["python=" + definition["python"], "pip", *conda["packages"]])
            stage("conda", lambda: _colabcola_env_run(create, cwd=job_root,
                env=_colabcola_env_clean_environment(root_prefix=root_prefix, network=True), deadline=deadline,
                log_path=log_path, label=name + ": resolve conda environment"))
            descriptor["toolchain"] = _colabcola_env_toolchain(prefix)
            stage("python", lambda: _colabcola_env_probe(descriptor, job_root, deadline, log_path))
            compiler_version = _colabcola_env_compiler_probe(descriptor, job_root, deadline, log_path, stage)
            pip = definition["pip"]
            install = ["python", "-m", "pip", "install", "--no-input"]
            if pip.get("index_url"):
                install.extend(["--index-url", pip["index_url"]])
            for index in pip.get("extra_index_urls", []):
                install.extend(["--extra-index-url", index])
            if pip.get("no_build_isolation"):
                install.append("--no-build-isolation")
            installs = []
            if pip["packages"]:
                installs.append(("pip-packages", list(pip["packages"])))
            installs.extend(("pip-requirements-" + str(i + 1), ["--requirement", item])
                            for i, item in enumerate(pip["requirements"]))
            for label, arguments in installs:
                command, env = _colabcola_environment_command(descriptor, install + arguments, job_root)
                env = _colabcola_env_network_environment(env)
                stage(label, lambda command=command, env=env, label=label: _colabcola_env_run(command,
                    cwd=job_root, env=env, deadline=deadline, log_path=log_path, label=name + ": " + label))
                descriptor["toolchain"] = _colabcola_env_toolchain(prefix)
                pip_facts = stage(label + "-verify", lambda: _colabcola_env_probe(descriptor, job_root, deadline, log_path, cuda=True))
                _colabcola_env_cuda_compatibility(definition, compiler_version, pip_facts)
            if pip.get("wheels"):
                record["embedded_wheels"] = stage("embedded-wheels", lambda: _colabcola_env_install_wheels(
                    descriptor, job_root, wheel_records, deadline, log_path))
                descriptor["toolchain"] = _colabcola_env_toolchain(prefix)
            resolved = _colabcola_env_resolved_facts(descriptor, job_root, deadline, log_path, stage, compiler_version)
            if selected.get("prefix") is not None:
                build_seconds = time.monotonic() - build_started
                cache_context.publish(selected, resolved, build_seconds)
                selected["report"]["setup_seconds"] = time.monotonic() - env_start
            record.update(state="ready", facts=resolved, toolchain=descriptor["toolchain"],
                elapsed_seconds=round(time.monotonic() - env_start, 6), finished_at=_colabcola_env_now())
            descriptors[name] = descriptor
            _colabcola_environment_json(evidence_path, facts)
            print("[colabcola] Environment " + name + " ready (Python " + resolved["python"] + ")", flush=True)
        facts["state"] = "ready"
        return descriptors, facts
    except BaseException as exc:
        facts.update(state="failed", failed_environment=active,
                     error={"type": type(exc).__name__, "message": str(exc)[:2000]})
        if active is not None:
            facts["environments"][active].update(state="failed", error=facts["error"],
                elapsed_seconds=round(time.monotonic() - env_start, 6))
        raise
    finally:
        if cache_context is not None:
            cache_context.close()
        facts.update(elapsed_seconds=round(time.monotonic() - started, 6), finished_at=_colabcola_env_now())
        _colabcola_environment_json(evidence_path, facts)
