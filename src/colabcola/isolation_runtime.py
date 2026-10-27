"""Self-contained notebook helpers for a checksum-bound isolated interpreter.

This file is embedded verbatim in compiled notebooks. Use only the standard library.
"""
import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request


def _colabcola_isolated_env(executable=None, declared=None):
    # An isolated venv is defeated by PYTHONPATH or pip's target/prefix/user config.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('PYTHON', 'PIP_')) and k not in ('VIRTUAL_ENV', '__PYVENV_LAUNCHER__')}
    for key, value in (declared or {}).items():
        if key.startswith(('PYTHON', 'PIP_')) or key in ('VIRTUAL_ENV', '__PYVENV_LAUNCHER__', 'PATH'):
            if key != 'PYTHONUNBUFFERED':
                raise RuntimeError('isolated Python forbids environment override: ' + key)
        env[key] = value
    env.update(PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', PIP_CONFIG_FILE=os.devnull)
    if executable:
        binary = pathlib.Path(executable).parent
        env['VIRTUAL_ENV'] = str(binary.parent)
        env['PATH'] = str(binary) + os.pathsep + env.get('PATH', os.defpath)
    return env


def _colabcola_build_environment(executable):
    """Select a complete host compiler pair, independent of standalone sysconfig."""
    import shlex
    env = _colabcola_isolated_env(executable)
    # Host overrides may refer to the compiler used to build standalone Python.
    # They are not a promise that this runtime has that compiler installed.
    for key in ('CC', 'CXX', 'LDSHARED'):
        env.pop(key, None)
    for family, c_name, cxx_name in (('gcc', 'gcc', 'g++'), ('clang', 'clang', 'clang++')):
        cc = shutil.which(c_name, path=env['PATH'])
        cxx = shutil.which(cxx_name, path=env['PATH'])
        if cc and cxx:
            env.update(CC=shlex.quote(cc), CXX=shlex.quote(cxx),
                       LDSHARED=shlex.quote(cc) + ' -shared')
            return env, {'state': 'available', 'family': family,
                         'cc': cc, 'cxx': cxx, 'ldshared': env['LDSHARED']}
    # Binary-only jobs still work; a source build will fail with pip's diagnostics.
    return env, {'state': 'unavailable', 'reason': 'no complete gcc/g++ or clang/clang++ pair'}


def _colabcola_isolated_timeout(default=600):
    deadline = globals().get('COLABCOLA_LIFECYCLE_DEADLINE')
    if deadline is None:
        return default
    remaining = deadline - time.time() - 30
    if remaining <= 0:
        raise RuntimeError('shared lifecycle deadline reached during Python setup')
    return min(default, remaining)


def _colabcola_extract_python(archive, target):
    """Reject traversal, duplicates, special files and escaping links before extraction."""
    target = pathlib.Path(target).resolve()
    with tarfile.open(archive, 'r:gz') as source:
        members = source.getmembers()
        if len(members) > 30000 or sum(max(0, x.size) for x in members) > 1024**3:
            raise RuntimeError('Python archive exceeds extraction limits')
        names = set()
        for member in members:
            name = pathlib.PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or '\\' in member.name or not name.parts or name.parts[0] != 'python':
                raise RuntimeError('unsafe Python archive path')
            if str(name) in names:
                raise RuntimeError('duplicate Python archive path')
            names.add(str(name))
            if not (member.isfile() or member.isdir() or member.issym()):
                raise RuntimeError('unsafe Python archive member type')
            if member.issym():
                link = pathlib.PurePosixPath(member.linkname)
                if link.is_absolute() or '\\' in member.linkname:
                    raise RuntimeError('unsafe Python archive link')
                try:
                    (target / name.parent / member.linkname).resolve().relative_to(target / 'python')
                except ValueError:
                    raise RuntimeError('escaping Python archive link') from None
        source.extractall(target, members=members, filter='data')


_COLABCOLA_PROBE = r'''
import importlib.metadata as metadata,json,platform,sys,site
from pip._vendor.packaging.requirements import Requirement
request=json.loads(sys.argv[1]);records=[];problems=[]
for text in request:
    req=Requirement(text)
    if req.marker is not None and not req.marker.evaluate():
        records.append({'requirement':text,'status':'marker_skipped','version':None});continue
    try: version=metadata.version(req.name)
    except metadata.PackageNotFoundError: version=None
    ok=version is not None and req.specifier.contains(version,prereleases=True)
    records.append({'requirement':text,'status':'satisfied' if ok else 'unsatisfied','version':version})
    if not ok: problems.append('isolated requirement '+text+' is not satisfied (installed='+repr(version)+')')
print(json.dumps({'python':platform.python_version(),'executable':sys.executable,'prefix':sys.prefix,
 'base_prefix':sys.base_prefix,'user_site_enabled':site.ENABLE_USER_SITE,'sys_path':sys.path,
 'packages':records,'problems':problems}))
'''


def _colabcola_probe_python(executable, requirements=()):
    process = subprocess.run([str(executable), '-I', '-c', _COLABCOLA_PROBE, json.dumps(list(requirements))],
        env=_colabcola_isolated_env(executable), capture_output=True, text=True,
        timeout=_colabcola_isolated_timeout(60))
    if process.returncode:
        raise RuntimeError('isolated Python probe failed: ' + process.stderr[-2000:])
    return json.loads(process.stdout)


def _colabcola_prepare_python(config, parent, archive_source=None):
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        raise RuntimeError('isolated CPython profile requires Linux x86_64 (Colab or WSL)')
    parent = pathlib.Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    workspace = pathlib.Path(tempfile.mkdtemp(prefix='python312-', dir=parent))
    archive = workspace / 'python.tar.gz'
    try:
        print('[colabcola] Acquiring checksum-pinned Python ' + config['version'], flush=True)
        # archive_source is only a local test seam; compiled notebooks do not set it.
        source = open(archive_source, 'rb') if archive_source else urllib.request.urlopen(config['url'], timeout=_colabcola_isolated_timeout(60))
        size = 0
        digest = hashlib.sha256()
        with source, archive.open('xb') as output:
            while block := source.read(1024 * 1024):
                _colabcola_isolated_timeout()
                size += len(block)
                if size > config['bytes']:
                    raise RuntimeError('Python archive exceeds declared byte count')
                digest.update(block)
                output.write(block)
        if size != config['bytes'] or digest.hexdigest() != config['sha256']:
            raise RuntimeError('Python archive size/SHA-256 mismatch')
        _colabcola_extract_python(archive, workspace)
        archive.unlink()
        base = workspace / 'python/bin/python3.12'
        if not base.is_file() or base.is_symlink():
            raise RuntimeError('standalone Python executable is missing or unsafe')
        venv = workspace / 'venv'
        subprocess.run([str(base), '-I', '-m', 'venv', '--copies', str(venv)],
            env=_colabcola_isolated_env(), check=True, timeout=_colabcola_isolated_timeout())
        executable = venv / 'bin/python'
        facts = _colabcola_probe_python(executable)
        if facts['python'] != config['version'] or facts['prefix'] == facts['base_prefix'] or facts['user_site_enabled']:
            raise RuntimeError('isolated interpreter identity or venv boundary failed')
        if 'include-system-site-packages = false' not in (venv / 'pyvenv.cfg').read_text():
            raise RuntimeError('isolated venv inherited system site packages')
        facts['archive'] = config
        _, facts['build_toolchain'] = _colabcola_build_environment(executable)
        print('[colabcola] Isolated Python ' + facts['python'] + ' ready', flush=True)
        return str(executable), facts
    except BaseException:
        shutil.rmtree(workspace)
        raise


def _colabcola_isolated_install(executable, arguments, cwd, log_path):
    """Stream setup output, with a bounded child and no separate process group."""
    import threading
    command = [str(executable), '-m', 'pip', '--no-cache-dir', *arguments]
    build_env, compiler = _colabcola_build_environment(executable)
    with pathlib.Path(log_path).open('ab') as log:
        message = '[colabcola] Build toolchain: ' + json.dumps(compiler, sort_keys=True) + '; pip wheel cache disabled\n'
        log.write(message.encode()); log.flush()
        print(message, end='', flush=True)
        process = subprocess.Popen(command, cwd=cwd, env=build_env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        tail = bytearray()
        def drain():
            while block := process.stdout.read1(8192):
                log.write(block); log.flush()
                tail.extend(block)
                del tail[:-4000]
                print(block.decode('utf-8', errors='replace'), end='', flush=True)
        thread = threading.Thread(target=drain, daemon=True); thread.start()
        try:
            process.wait(timeout=_colabcola_isolated_timeout(3600))
        finally:
            if process.poll() is None:
                # Whole notebook process group is additionally guarded by the worker.
                process.kill(); process.wait()
            thread.join(timeout=10)
            process.stdout.close()
        if process.returncode:
            raise RuntimeError('isolated pip failed (exit ' + str(process.returncode) + '): ' + tail.decode('utf-8',errors='replace')[-1800:])
