"""Strict, dependency-free parsing for the version 1 source JobSpec."""

from __future__ import annotations

import json
import os
import re
import stat
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .errors import SpecError

_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRET_ENV_RE = re.compile(
    r"(?:^|_)(?:api_?key|auth|credential|password|private_?key|secret|token)(?:_|$)",
    re.IGNORECASE,
)
_INSTALL_REQUIREMENT_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:\[[A-Za-z0-9._,-]+\])?"
    r"==[A-Za-z0-9][A-Za-z0-9.!+_-]*"
    r"(?: --hash=sha256:[0-9a-fA-F]{64})+$"
)
_DRIVE_PREFIX_RE = re.compile(r"^[A-Za-z]:")
_PAYLOAD_ROLES = frozenset({"source", "data", "config", "license", "wheel"})
_TERMINAL_ACCELERATORS = frozenset({"cpu", "gpu"})
_RUNTIME_VERSION_RE = re.compile(r"^(?:latest|[0-9]{4}\.[0-9]{2})$")
_CHECKPOINT_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REMOTE_URL_RE = re.compile(
    r"^https://[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?(?::[0-9]{1,5})?/[^\s#@]*$"
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_REPARSE_FLAG = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


@dataclass(frozen=True, slots=True)
class RemoteInputSpec:
    """One file the runtime downloads into the job root, bound by target, URL, bytes, and SHA-256."""

    target: str
    url: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PayloadEntry:
    role: str
    source: Path
    target: str
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EntrypointSpec:
    kind: str
    path: str
    argv: tuple[str, ...]
    cwd: str


@dataclass(frozen=True, slots=True)
class PythonIsolationSpec:
    kind: str
    version: str
    url: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PythonSpec:
    requires_python: str
    runtime_requirements: tuple[str, ...]
    install_requirements: tuple[str, ...]
    embedded_wheels: tuple[str, ...]
    pip_requirements: tuple[str, ...] = ()
    isolation: PythonIsolationSpec | None = None


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    provider: str
    version: str


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    accelerator: str
    gpu_count: int
    cuda_required: bool
    minimum_gpu_memory_bytes: int | None


@dataclass(frozen=True, slots=True)
class RunSpec:
    timeout_seconds: int
    seed: int
    environment: tuple[tuple[str, str], ...]
    capture_stdout: bool
    capture_stderr: bool


@dataclass(frozen=True, slots=True)
class OutputSpec:
    pattern: str
    required: bool
    maximum_bytes: int


@dataclass(frozen=True, slots=True)
class LimitSpec:
    maximum_file_count: int


@dataclass(frozen=True, slots=True)
class DurableCheckpointSpec:
    directory: str
    generator_prefix: str
    state_prefix: str
    filename_digits: int
    poll_seconds: int
    remote_retention: int


@dataclass(frozen=True, slots=True)
class ArtifactGroupSpec:
    marker: str
    files: tuple[str, ...]
    ready_when: tuple[tuple[str, str], ...]

    def mapping(self):
        return {"marker": self.marker, "files": list(self.files), "ready_when": dict(self.ready_when)}


@dataclass(frozen=True, slots=True)
class ManagedRecoverySpec:
    checkpoint_pattern: str
    log_patterns: tuple[str, ...]
    artifact_groups: tuple[ArtifactGroupSpec, ...] = ()


@dataclass(frozen=True, slots=True)
class JobSpec:
    schema_version: int
    name: str
    payload: tuple[PayloadEntry, ...]
    entrypoint: EntrypointSpec
    python: PythonSpec
    runtime: RuntimeSpec
    resources: ResourceSpec
    run: RunSpec
    outputs: tuple[OutputSpec, ...]
    limits: LimitSpec
    durable_checkpoints: DurableCheckpointSpec | None = None
    remote_inputs: tuple[RemoteInputSpec, ...] = ()
    managed_recovery: ManagedRecoverySpec | None = None

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, base_dir: Path | str = "."
    ) -> "JobSpec":
        root = _object(value, "JobSpec")
        _keys(
            root,
            required={
                "schema_version",
                "name",
                "payload",
                "entrypoint",
                "python",
                "runtime",
                "resources",
                "run",
                "outputs",
                "limits",
            },
            optional={"durable_checkpoints", "remote_inputs", "managed_recovery"},
            where="JobSpec",
        )

        schema_version = _integer(root["schema_version"], "JobSpec.schema_version")
        if schema_version != 1:
            raise SpecError("JobSpec.schema_version must equal 1")

        name = _string(root["name"], "JobSpec.name")
        if not _NAME_RE.fullmatch(name):
            raise SpecError(
                "JobSpec.name must be a lowercase hyphen-separated slug"
            )

        resolved_base = Path(os.path.abspath(os.fspath(base_dir)))
        payload_values = _array(root["payload"], "JobSpec.payload")
        payload = tuple(
            _parse_payload(entry, index, resolved_base)
            for index, entry in enumerate(payload_values)
        )
        if not payload:
            raise SpecError("JobSpec.payload must contain at least one entry")
        targets = [entry.target for entry in payload]
        duplicates = sorted(
            target for target, count in Counter(targets).items() if count > 1
        )
        if duplicates:
            raise SpecError(
                "JobSpec.payload contains duplicate targets: " + ", ".join(duplicates)
            )

        entrypoint = _parse_entrypoint(root["entrypoint"])
        python = _parse_python(root["python"], payload)
        runtime = _parse_runtime(root["runtime"])
        resources = _parse_resources(root["resources"])
        run = _parse_run(root["run"])
        if python.isolation is not None:
            forbidden = [key for key, _ in run.environment
                         if (key.startswith(("PYTHON", "PIP_")) and key != "PYTHONUNBUFFERED")
                         or key in ("PATH", "VIRTUAL_ENV", "__PYVENV_LAUNCHER__")]
            if forbidden:
                raise SpecError("isolated Python forbids environment overrides: " + ", ".join(forbidden))
        outputs = tuple(
            _parse_output(entry, index)
            for index, entry in enumerate(_array(root["outputs"], "JobSpec.outputs"))
        )
        limits = _parse_limits(root["limits"])
        durable_checkpoints = (
            None
            if "durable_checkpoints" not in root
            else _parse_durable_checkpoints(root["durable_checkpoints"])
        )
        remote_inputs = (
            ()
            if "remote_inputs" not in root
            else _parse_remote_inputs(root["remote_inputs"], payload)
        )

        return cls(
            schema_version=schema_version,
            name=name,
            payload=payload,
            entrypoint=entrypoint,
            python=python,
            runtime=runtime,
            resources=resources,
            run=run,
            outputs=outputs,
            limits=limits,
            durable_checkpoints=durable_checkpoints,
            remote_inputs=remote_inputs,
            managed_recovery=(
                _parse_managed_recovery(root["managed_recovery"])
                if "managed_recovery" in root else None
            ),
        )

    @classmethod
    def from_json(cls, path: Path | str) -> "JobSpec":
        return load_job_spec(path)


def load_job_spec(path: Path | str) -> JobSpec:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
        value = json.loads(text, object_pairs_hook=_unique_object)
    except UnicodeDecodeError as exc:
        raise SpecError(f"{source}: JobSpec must be UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"{source}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
        ) from exc
    return JobSpec.from_mapping(value, base_dir=source.parent)


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SpecError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _parse_payload(value: Any, index: int, base_dir: Path) -> PayloadEntry:
    where = f"JobSpec.payload[{index}]"
    item = _object(value, where)
    _keys(item, required={"role", "source", "target"}, optional={"exclude"}, where=where)

    role = _string(item["role"], f"{where}.role")
    if role not in _PAYLOAD_ROLES:
        raise SpecError(f"{where}.role must be one of {sorted(_PAYLOAD_ROLES)}")

    source_text = _string(item["source"], f"{where}.source")
    source = Path(source_text)
    if not source.is_absolute():
        source = base_dir / source
    source = Path(os.path.abspath(source))
    _validate_source_tree(source, f"{where}.source")

    target = _portable_path(item["target"], f"{where}.target")
    exclude = tuple(
        _nonempty_strings(item.get("exclude", []), f"{where}.exclude")
    )
    return PayloadEntry(role=role, source=source, target=target, exclude=exclude)


def _parse_entrypoint(value: Any) -> EntrypointSpec:
    where = "JobSpec.entrypoint"
    item = _object(value, where)
    _keys(item, required={"kind", "path", "argv", "cwd"}, optional=set(), where=where)
    kind = _string(item["kind"], f"{where}.kind")
    if kind != "python_script":
        raise SpecError(f"{where}.kind must equal 'python_script'")
    return EntrypointSpec(
        kind=kind,
        path=_portable_path(item["path"], f"{where}.path"),
        argv=tuple(_strings(item["argv"], f"{where}.argv", allow_empty=True)),
        cwd=_portable_path(item["cwd"], f"{where}.cwd", allow_dot=True),
    )


def _parse_python(value: Any, payload: Sequence[PayloadEntry]) -> PythonSpec:
    where = "JobSpec.python"
    item = _object(value, where)
    _keys(
        item,
        required={
            "requires_python",
            "runtime_requirements",
            "install_requirements",
            "embedded_wheels",
        },
        optional={"pip_requirements", "isolation"},
        where=where,
    )
    runtime = tuple(
        _nonempty_strings(item["runtime_requirements"], f"{where}.runtime_requirements")
    )
    for index, requirement in enumerate(runtime):
        if "://" in requirement or "git+" in requirement.lower() or "@" in requirement:
            raise SpecError(
                f"{where}.runtime_requirements[{index}] must be an installed-package constraint"
            )
    install = tuple(
        _nonempty_strings(item["install_requirements"], f"{where}.install_requirements")
    )
    for index, requirement in enumerate(install):
        if not _INSTALL_REQUIREMENT_RE.fullmatch(requirement):
            raise SpecError(
                f"{where}.install_requirements[{index}] must be an exact index pin with a sha256 hash"
            )

    embedded = tuple(
        _portable_path(entry, f"{where}.embedded_wheels[{index}]")
        for index, entry in enumerate(
            _strings(item["embedded_wheels"], f"{where}.embedded_wheels", allow_empty=True)
        )
    )
    wheel_targets = {entry.target for entry in payload if entry.role == "wheel"}
    missing_wheels = sorted(set(embedded) - wheel_targets)
    if missing_wheels:
        raise SpecError(
            f"{where}.embedded_wheels do not reference wheel payload targets: "
            + ", ".join(missing_wheels)
        )

    pip_requirements = tuple(_nonempty_strings(item.get("pip_requirements", []), f"{where}.pip_requirements"))
    if pip_requirements:
        from packaging.requirements import Requirement, InvalidRequirement
        for requirement in pip_requirements:
            try:
                parsed = Requirement(requirement)
            except InvalidRequirement:
                raise SpecError("pip_requirements must contain valid PyPI requirements") from None
            if parsed.url:
                raise SpecError("pip_requirements cannot contain URL or local-path requirements")

    return PythonSpec(
        requires_python=_string(item["requires_python"], f"{where}.requires_python"),
        runtime_requirements=runtime,
        install_requirements=install,
        embedded_wheels=embedded,
        pip_requirements=pip_requirements,
        isolation=_parse_isolation(item["isolation"], item["requires_python"]) if "isolation" in item else None,
    )


def _parse_isolation(value, requires_python):
    where = "JobSpec.python.isolation"
    item = _object(value, where)
    _keys(item, required={"kind", "version", "url", "bytes", "sha256"}, optional=set(), where=where)
    if item["kind"] != "cpython-standalone-linux-x86_64":
        raise SpecError(where + ".kind must equal cpython-standalone-linux-x86_64")
    version = _string(item["version"], where + ".version")
    if not re.fullmatch(r"3\.12\.[0-9]+", version):
        raise SpecError(where + ".version must be an exact Python 3.12 patch version")
    from packaging.specifiers import SpecifierSet, InvalidSpecifier
    try:
        if not SpecifierSet(requires_python).contains(version):
            raise SpecError("isolated Python does not satisfy requires_python")
    except InvalidSpecifier:
        raise SpecError("invalid requires_python constraint") from None
    url = _string(item["url"], where + ".url")
    if not _REMOTE_URL_RE.fullmatch(url):
        raise SpecError(where + ".url must be a credential-free HTTPS URL")
    size = _positive_integer(item["bytes"], where + ".bytes")
    if size > 256 * 1024 * 1024:
        raise SpecError("isolated Python archive exceeds 256 MiB")
    sha = _string(item["sha256"], where + ".sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise SpecError("isolated Python requires a lowercase SHA-256")
    return PythonIsolationSpec(item["kind"], version, url, size, sha)


def _parse_runtime(value: Any) -> RuntimeSpec:
    where = "JobSpec.runtime"
    item = _object(value, where)
    _keys(item, required={"provider", "version"}, optional=set(), where=where)
    provider = _string(item["provider"], f"{where}.provider")
    if provider != "google_colab":
        raise SpecError(f"{where}.provider must equal 'google_colab'")
    version = _string(item["version"], f"{where}.version")
    if not _RUNTIME_VERSION_RE.fullmatch(version):
        raise SpecError(f"{where}.version must be 'latest' or YYYY.MM")
    return RuntimeSpec(provider=provider, version=version)


def _parse_resources(value: Any) -> ResourceSpec:
    where = "JobSpec.resources"
    item = _object(value, where)
    _keys(
        item,
        required={
            "accelerator",
            "gpu_count",
            "cuda_required",
            "minimum_gpu_memory_bytes",
        },
        optional=set(),
        where=where,
    )
    accelerator = _string(item["accelerator"], f"{where}.accelerator")
    if accelerator not in _TERMINAL_ACCELERATORS:
        raise SpecError(f"{where}.accelerator must be 'cpu' or 'gpu'")
    gpu_count = _integer(item["gpu_count"], f"{where}.gpu_count")
    cuda_required = _boolean(item["cuda_required"], f"{where}.cuda_required")
    minimum = item["minimum_gpu_memory_bytes"]
    if minimum is not None:
        minimum = _positive_integer(minimum, f"{where}.minimum_gpu_memory_bytes")

    if accelerator == "gpu" and (gpu_count != 1 or not cuda_required):
        raise SpecError(f"{where}: version 1 GPU jobs require gpu_count=1 and cuda_required=true")
    if accelerator == "cpu" and (gpu_count != 0 or cuda_required or minimum is not None):
        raise SpecError(
            f"{where}: CPU jobs require gpu_count=0, cuda_required=false, and no GPU memory minimum"
        )
    return ResourceSpec(accelerator, gpu_count, cuda_required, minimum)


def _parse_run(value: Any) -> RunSpec:
    where = "JobSpec.run"
    item = _object(value, where)
    _keys(
        item,
        required={
            "timeout_seconds",
            "seed",
            "environment",
            "capture_stdout",
            "capture_stderr",
        },
        optional=set(),
        where=where,
    )
    environment = _object(item["environment"], f"{where}.environment")
    parsed_environment: list[tuple[str, str]] = []
    for key in sorted(environment):
        if not _ENV_NAME_RE.fullmatch(key):
            raise SpecError(f"{where}.environment has invalid variable name {key!r}")
        if key in {"COLABCOLA_SEED", "COLABCOLA_RESUME_MANIFEST"}:
            raise SpecError(
                f"{where}.environment.COLABCOLA_SEED is reserved for the configured run seed"
            )
        value_text = _string(environment[key], f"{where}.environment.{key}", allow_empty=True)
        if value_text and _SECRET_ENV_RE.search(key):
            raise SpecError(
                f"{where}.environment.{key} looks secret-bearing; secret values are forbidden"
            )
        parsed_environment.append((key, value_text))

    return RunSpec(
        timeout_seconds=_positive_integer(item["timeout_seconds"], f"{where}.timeout_seconds"),
        seed=_integer(item["seed"], f"{where}.seed"),
        environment=tuple(parsed_environment),
        capture_stdout=_boolean(item["capture_stdout"], f"{where}.capture_stdout"),
        capture_stderr=_boolean(item["capture_stderr"], f"{where}.capture_stderr"),
    )


def _parse_output(value: Any, index: int) -> OutputSpec:
    where = f"JobSpec.outputs[{index}]"
    item = _object(value, where)
    _keys(item, required={"pattern", "required", "maximum_bytes"}, optional=set(), where=where)
    return OutputSpec(
        pattern=_portable_path(item["pattern"], f"{where}.pattern"),
        required=_boolean(item["required"], f"{where}.required"),
        maximum_bytes=_positive_integer(item["maximum_bytes"], f"{where}.maximum_bytes"),
    )


def _parse_limits(value: Any) -> LimitSpec:
    where = "JobSpec.limits"
    item = _object(value, where)
    _keys(
        item,
        required={"maximum_file_count"},
        optional=set(),
        where=where,
    )
    return LimitSpec(
        maximum_file_count=_positive_integer(
            item["maximum_file_count"], f"{where}.maximum_file_count"
        ),
    )


def _parse_remote_inputs(
    value: Any, payload: Sequence[PayloadEntry]
) -> tuple[RemoteInputSpec, ...]:
    where = "JobSpec.remote_inputs"
    payload_targets = {entry.target for entry in payload}
    entries: list[RemoteInputSpec] = []
    for index, raw in enumerate(_array(value, where)):
        item_where = f"{where}[{index}]"
        item = _object(raw, item_where)
        _keys(
            item,
            required={"target", "url", "bytes", "sha256"},
            optional=set(),
            where=item_where,
        )
        target = _portable_path(item["target"], f"{item_where}.target")
        url = _string(item["url"], f"{item_where}.url")
        if not _REMOTE_URL_RE.fullmatch(url):
            raise SpecError(
                f"{item_where}.url must be an https URL without user information, "
                "whitespace, or a fragment"
            )
        size = _positive_integer(item["bytes"], f"{item_where}.bytes")
        digest = _string(item["sha256"], f"{item_where}.sha256")
        if not _SHA256_RE.fullmatch(digest):
            raise SpecError(f"{item_where}.sha256 must be 64 hexadecimal characters")
        if target in payload_targets:
            raise SpecError(f"{item_where}.target collides with a payload target: {target}")
        entries.append(
            RemoteInputSpec(target=target, url=url, bytes=size, sha256=digest.lower())
        )
    duplicates = sorted(
        target
        for target, count in Counter(entry.target for entry in entries).items()
        if count > 1
    )
    if duplicates:
        raise SpecError(f"{where} contains duplicate targets: " + ", ".join(duplicates))
    return tuple(entries)


def _parse_durable_checkpoints(value: Any) -> DurableCheckpointSpec:
    where = "JobSpec.durable_checkpoints"
    item = _object(value, where)
    _keys(
        item,
        required={
            "directory",
            "generator_prefix",
            "state_prefix",
            "filename_digits",
            "poll_seconds",
            "remote_retention",
        },
        optional=set(),
        where=where,
    )
    generator_prefix = _string(item["generator_prefix"], f"{where}.generator_prefix")
    state_prefix = _string(item["state_prefix"], f"{where}.state_prefix")
    for label, prefix in (
        ("generator_prefix", generator_prefix),
        ("state_prefix", state_prefix),
    ):
        if not _CHECKPOINT_PREFIX_RE.fullmatch(prefix):
            raise SpecError(f"{where}.{label} must be a portable filename prefix")
    if generator_prefix == state_prefix:
        raise SpecError(f"{where} checkpoint prefixes must be distinct")

    filename_digits = _positive_integer(item["filename_digits"], f"{where}.filename_digits")
    if filename_digits > 16:
        raise SpecError(f"{where}.filename_digits must not exceed 16")
    poll_seconds = _positive_integer(item["poll_seconds"], f"{where}.poll_seconds")
    if poll_seconds < 5 or poll_seconds > 3600:
        raise SpecError(f"{where}.poll_seconds must be between 5 and 3600")
    remote_retention = _positive_integer(
        item["remote_retention"], f"{where}.remote_retention"
    )
    if remote_retention < 2 or remote_retention > 10:
        raise SpecError(f"{where}.remote_retention must be between 2 and 10")
    return DurableCheckpointSpec(
        directory=_portable_path(item["directory"], f"{where}.directory"),
        generator_prefix=generator_prefix,
        state_prefix=state_prefix,
        filename_digits=filename_digits,
        poll_seconds=poll_seconds,
        remote_retention=remote_retention,
    )
def _validate_source_tree(path: Path, where: str) -> None:
    def visit(current: Path) -> None:
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as exc:
            raise SpecError(f"{where} does not exist: {path}") from exc
        except OSError as exc:
            raise SpecError(f"{where} cannot be inspected: {current}: {exc}") from exc

        attributes = getattr(metadata, "st_file_attributes", 0)
        if stat.S_ISLNK(metadata.st_mode) or attributes & _REPARSE_FLAG:
            raise SpecError(f"{where} contains a forbidden symlink or reparse point: {current}")
        if stat.S_ISREG(metadata.st_mode):
            return
        if not stat.S_ISDIR(metadata.st_mode):
            raise SpecError(f"{where} contains a forbidden special file: {current}")

        try:
            with os.scandir(current) as entries:
                children = sorted(entries, key=lambda entry: entry.name)
        except OSError as exc:
            raise SpecError(f"{where} cannot be enumerated: {current}: {exc}") from exc
        for child in children:
            visit(current / child.name)

    visit(path)


def _portable_path(value: Any, where: str, *, allow_dot: bool = False) -> str:
    text = _string(value, where)
    if allow_dot and text == ".":
        return text
    if (
        "\\" in text
        or "\x00" in text
        or text.startswith("/")
        or _DRIVE_PREFIX_RE.match(text)
    ):
        raise SpecError(f"{where} must be a normalized relative POSIX path")
    normalized = PurePosixPath(text).as_posix()
    if normalized != text or any(part in {"", ".", ".."} or ":" in part for part in PurePosixPath(text).parts):
        raise SpecError(f"{where} must be a normalized relative POSIX path")
    return text


def _keys(
    value: Mapping[str, Any], *, required: set[str], optional: set[str], where: str
) -> None:
    actual = set(value)
    missing = sorted(required - actual)
    unknown = sorted(actual - required - optional)
    if missing:
        raise SpecError(f"{where} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise SpecError(f"{where} has unknown fields: {', '.join(unknown)}")


def _object(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpecError(f"{where} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise SpecError(f"{where} keys must be strings")
    return value


def _array(value: Any, where: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise SpecError(f"{where} must be an array")
    return value


def _string(value: Any, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a nonempty string"
        raise SpecError(f"{where} must be {qualifier}")
    return value


def _strings(value: Any, where: str, *, allow_empty: bool) -> list[str]:
    return [
        _string(entry, f"{where}[{index}]", allow_empty=allow_empty)
        for index, entry in enumerate(_array(value, where))
    ]


def _nonempty_strings(value: Any, where: str) -> list[str]:
    return _strings(value, where, allow_empty=False)


def _integer(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecError(f"{where} must be an integer")
    return value


def _positive_integer(value: Any, where: str) -> int:
    result = _integer(value, where)
    if result <= 0:
        raise SpecError(f"{where} must be greater than zero")
    return result


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise SpecError(f"{where} must be a boolean")
    return value


def _parse_managed_recovery(value: Any) -> ManagedRecoverySpec:
    where = "JobSpec.managed_recovery"
    item = _object(value, where)
    _keys(item, required={"checkpoint_pattern", "log_patterns"}, optional={"artifact_groups"}, where=where)
    checkpoint = _portable_path(item["checkpoint_pattern"], where + ".checkpoint_pattern")
    logs = tuple(_portable_path(p, where + ".log_patterns")
                 for p in _array(item["log_patterns"], where + ".log_patterns"))
    if not checkpoint.startswith("outputs/") or not checkpoint.endswith(".json"):
        raise SpecError("managed checkpoint pattern must be an outputs/ JSON manifest")
    if any(not p.startswith("outputs/") for p in logs) or len(logs) > 16:
        raise SpecError("managed logs must be at most 16 outputs/ patterns")
    groups = []
    for raw in _array(item.get("artifact_groups", []), where + ".artifact_groups"):
        group = _object(raw, where + ".artifact_groups[]")
        _keys(group, required={"marker", "files", "ready_when"}, optional=set(), where=where)
        marker = _portable_path(group["marker"], where + ".marker")
        files = tuple(_portable_path(p, where + ".files") for p in _array(group["files"], where + ".files"))
        ready = _object(group["ready_when"], where + ".ready_when")
        if (not marker.startswith("outputs/") or not marker.endswith(".json")
                or not 1 <= len(files) <= 64 or len(set(files)) != len(files)
                or not 1 <= len(ready) <= 16
                or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in ready.items())):
            raise SpecError("completed artifact groups need an outputs/ JSON marker,1-64 relative files and string ready_when fields")
        groups.append(ArtifactGroupSpec(marker, files, tuple(sorted(ready.items()))))
    if len(groups) > 16:
        raise SpecError("at most 16 completed artifact groups are supported")
    return ManagedRecoverySpec(checkpoint, logs, tuple(groups))
