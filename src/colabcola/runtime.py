"""Deterministic source for the four notebook runtime cell groups."""

from __future__ import annotations

from typing import Final
from pathlib import Path

_ISOLATION_HELPERS = Path(__file__).with_name("isolation_runtime.py").read_text(encoding="utf-8-sig")

from .checkpoint_runtime import DURABLE_CHECKPOINT_SOURCE

_GPU_CHECK_SOURCE = Path(__file__).with_name("gpu_check.py").read_text(encoding="utf-8-sig")


_PREFLIGHT_SOURCE: Final[str] = r'''
# Runtime preflight and shared runtime helpers.
import datetime
import glob
import importlib.metadata
import re
import pathlib
import platform
import signal
import shutil
import subprocess
import threading
import time
import traceback

_COLABCOLA_RUNTIME_STARTED_AT = datetime.datetime.now(
    datetime.timezone.utc
).isoformat()
_COLABCOLA_RUNTIME_STARTED_MONOTONIC = time.monotonic()
_COLABCOLA_TERMINAL_STATE = None
_COLABCOLA_FAILURE = None
_COLABCOLA_PROCESS_EXIT_CODE = None
_COLABCOLA_OUTPUTS = []
_COLABCOLA_ENVIRONMENT_FACTS = {}
_COLABCOLA_PYTHON = sys.executable
_COLABCOLA_ISOLATION_FACTS = None
_COLABCOLA_DRIVE_AUTH_CONTEXT = None
_COLABCOLA_DURABLE_CHECKPOINT_FACTS = None
_COLABCOLA_TRANSPORT_FACTS = {
    "mode": str(globals().get("COLABCOLA_TRANSPORT_MODE", "embedded"))
}
_COLABCOLA_RUNTIME_FACTS = {
    "python": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "platform": platform.platform(),
    "requested_provider": COLABCOLA_MANIFEST["runtime"]["provider"],
    "requested_version": COLABCOLA_MANIFEST["runtime"]["version"],
    "packages": {},
    "cuda_available": False,
    "gpu_count": 0,
    "gpus": [],
}

COLABCOLA_CONTENT_ROOT = pathlib.Path(
    str(globals().get("COLABCOLA_CONTENT_ROOT_OVERRIDE", "/content"))
)
COLABCOLA_JOB_ROOT = (
    COLABCOLA_CONTENT_ROOT / "colabcola" / "jobs" / COLABCOLA_JOB_ID
)
COLABCOLA_RESULT_ROOT = (
    COLABCOLA_CONTENT_ROOT / "colabcola" / "results" / COLABCOLA_JOB_ID
)


def _colabcola_sanitize_exception(exc):
    text = " ".join(str(exc).split())
    return text[:2000] or exc.__class__.__name__


def _colabcola_fail(stage, terminal_state, exc):
    global _COLABCOLA_FAILURE, _COLABCOLA_TERMINAL_STATE
    if _COLABCOLA_TERMINAL_STATE is None:
        _COLABCOLA_TERMINAL_STATE = terminal_state
        _COLABCOLA_FAILURE = {
            "stage": stage,
            "exception_type": exc.__class__.__name__,
            "message": _colabcola_sanitize_exception(exc),
        }


def _colabcola_requirement_tools():
    try:
        from packaging.requirements import Requirement
        from packaging.specifiers import SpecifierSet
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
        from pip._vendor.packaging.specifiers import SpecifierSet
    return Requirement, SpecifierSet


def _colabcola_check_requirements(requirements):
    if _COLABCOLA_ISOLATION_FACTS is not None:
        facts = _colabcola_probe_python(_COLABCOLA_PYTHON, requirements)
        return facts["packages"], facts["problems"]
    Requirement, _ = _colabcola_requirement_tools()
    records = []
    problems = []
    for text in requirements:
        try:
            requirement = Requirement(text)
            if requirement.marker is not None and not requirement.marker.evaluate():
                records.append(
                    {"requirement": text, "status": "marker_skipped", "version": None}
                )
                continue
            try:
                version = importlib.metadata.version(requirement.name)
            except importlib.metadata.PackageNotFoundError:
                version = None
            satisfied = version is not None and requirement.specifier.contains(
                version, prereleases=True
            )
            records.append(
                {
                    "requirement": text,
                    "status": "satisfied" if satisfied else "unsatisfied",
                    "version": version,
                }
            )
            if not satisfied:
                problems.append(
                    f"runtime requirement {text!r} is not satisfied (installed={version!r})"
                )
        except Exception as exc:
            records.append(
                {"requirement": text, "status": "invalid", "version": None}
            )
            problems.append(f"invalid runtime requirement {text!r}: {exc}")
    return records, problems


try:
    if COLABCOLA_RESULT_ROOT.exists():
        shutil.rmtree(COLABCOLA_RESULT_ROOT)
    COLABCOLA_RESULT_ROOT.mkdir(parents=True, exist_ok=False)

    canonical_manifest_bytes = COLABCOLA_MANIFEST_JSON.encode("utf-8")
    if hashlib.sha256(canonical_manifest_bytes).hexdigest() != COLABCOLA_MANIFEST_SHA256:
        raise RuntimeError("canonical manifest SHA-256 does not match")
    regenerated_manifest = (
        json.dumps(
            COLABCOLA_MANIFEST,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    if regenerated_manifest != canonical_manifest_bytes:
        raise RuntimeError("manifest literal is not canonical JSON")

    resources = COLABCOLA_MANIFEST["resources"]
    if resources["accelerator"] == "gpu":
        cuda_available = False
        gpu_records = []
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                for index in range(torch.cuda.device_count()):
                    properties = torch.cuda.get_device_properties(index)
                    gpu_records.append(
                        {
                            "index": index,
                            "name": properties.name,
                            "memory_bytes": int(properties.total_memory),
                        }
                    )
        except Exception:
            pass

        if not gpu_records and shutil.which("nvidia-smi"):
            query = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True, timeout=30,
            )
            if query.returncode == 0:
                for line in query.stdout.splitlines():
                    fields = [field.strip() for field in line.split(",", 2)]
                    if len(fields) == 3:
                        gpu_records.append(
                            {
                                "index": int(fields[0]),
                                "name": fields[1],
                                "memory_bytes": int(fields[2]) * 1024 * 1024,
                            }
                        )
                cuda_available = bool(gpu_records)

        _COLABCOLA_RUNTIME_FACTS["cuda_available"] = cuda_available
        _COLABCOLA_RUNTIME_FACTS["gpu_count"] = len(gpu_records)
        _COLABCOLA_RUNTIME_FACTS["gpus"] = gpu_records
    hardware = _colabcola_gpu_check(globals().get("COLABCOLA_REQUESTED_GPU_TYPE"), resources,
        _COLABCOLA_RUNTIME_FACTS["gpus"], _COLABCOLA_RUNTIME_FACTS["cuda_available"])
    hardware.update(job_id=COLABCOLA_JOB_ID, run_id=globals().get("COLABCOLA_RUN_ID"),
                    attempt=globals().get("COLABCOLA_ATTEMPT"), allocation_id=globals().get("COLABCOLA_ALLOCATION_ID"))
    hardware["host_memory"] = _colabcola_boot_memory()
    _COLABCOLA_RUNTIME_FACTS["resource_check"] = hardware
    temporary = COLABCOLA_RESULT_ROOT / "hardware.json.tmp"
    temporary.write_text(json.dumps(hardware, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, COLABCOLA_RESULT_ROOT / "hardware.json")
    if hardware["state"] != "passed":
        raise RuntimeError("requested GPU model/count/memory check failed: " + hardware["reason"])


    Requirement, SpecifierSet = _colabcola_requirement_tools()
    python_constraint = SpecifierSet(COLABCOLA_MANIFEST["python"]["requires_python"])
    if not python_constraint.contains(COLABCOLA_MANIFEST["python"].get("isolation", {}).get("version", platform.python_version()), prereleases=True):
        raise RuntimeError(
            "Python "
            + platform.python_version()
            + " does not satisfy "
            + COLABCOLA_MANIFEST["python"]["requires_python"]
        )

    package_records, package_problems = _colabcola_check_requirements(
        [] if COLABCOLA_MANIFEST["python"].get("isolation") else COLABCOLA_MANIFEST["python"]["runtime_requirements"]
    )
    _COLABCOLA_RUNTIME_FACTS["packages"] = {
        record["requirement"]: record for record in package_records
    }
    if package_problems:
        raise RuntimeError("; ".join(package_problems))


except Exception as exc:
    _colabcola_fail("preflight", "preflight_failed", exc)
'''


_EXTRACTION_SOURCE: Final[str] = r'''
# Reassemble, verify, and safely extract the complete hard-coded archive.
if _COLABCOLA_TERMINAL_STATE is None:
    try:
        archive_bytes = b"".join(
            base64.b64decode(chunk, validate=True)
            for chunk in COLABCOLA_PAYLOAD_CHUNKS
        )
        if len(archive_bytes) != COLABCOLA_ARCHIVE_BYTES:
            raise RuntimeError("decoded archive byte count does not match")
        archive_digest = hashlib.sha256(archive_bytes).digest()
        if archive_digest.hex() != COLABCOLA_ARCHIVE_SHA256:
            raise RuntimeError("decoded archive SHA-256 does not match")
        framed_identity = (
            len(COLABCOLA_MANIFEST_JSON.encode("utf-8")).to_bytes(8, "big")
            + COLABCOLA_MANIFEST_JSON.encode("utf-8")
            + len(archive_digest).to_bytes(8, "big")
            + archive_digest
        )
        if hashlib.sha256(framed_identity).hexdigest() != COLABCOLA_JOB_ID:
            raise RuntimeError("job ID does not match manifest and archive identities")

        if COLABCOLA_JOB_ROOT.exists():
            shutil.rmtree(COLABCOLA_JOB_ROOT)
        COLABCOLA_JOB_ROOT.mkdir(parents=True, exist_ok=False)
        manifest_files = COLABCOLA_MANIFEST["files"]

        with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
            infos = archive.infolist()
            if [info.filename for info in infos] != [
                record["path"] for record in manifest_files
            ]:
                raise RuntimeError("archive entry order or paths do not match manifest")
            if len({info.filename for info in infos}) != len(infos):
                raise RuntimeError("archive contains duplicate entry paths")

            for info, record in zip(infos, manifest_files, strict=True):
                path = pathlib.PurePosixPath(info.filename)
                if (
                    path.is_absolute()
                    or path.as_posix() != info.filename
                    or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
                    or info.is_dir()
                ):
                    raise RuntimeError(f"unsafe archive entry: {info.filename!r}")
                archive_mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_IFMT(archive_mode) != stat.S_IFREG:
                    raise RuntimeError(f"archive entry is not a regular file: {info.filename}")

                content = archive.read(info)
                if len(content) != record["size"]:
                    raise RuntimeError(f"archive size does not match for {info.filename}")
                if hashlib.sha256(content).hexdigest() != record["sha256"]:
                    raise RuntimeError(f"archive SHA-256 does not match for {info.filename}")

                target = COLABCOLA_JOB_ROOT.joinpath(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.resolve().relative_to(COLABCOLA_JOB_ROOT.resolve())
                with target.open("xb") as output:
                    output.write(content)
                target.chmod(int(record["mode"], 8))

        COLABCOLA_PAYLOAD_CHUNKS.clear()
        archive_bytes = b""
    except Exception as exc:
        _colabcola_fail("integrity", "integrity_failed", exc)
'''


_DRIVE_ACQUISITION_SOURCE: Final[str] = r'''
# Authorize per-file Drive API access, then stream and verify one exact object.
if _COLABCOLA_TERMINAL_STATE is None:
    temporary_archive = None
    downloaded_source = None
    try:
        acquisition_started_monotonic = time.monotonic()
        authorization_and_metadata_seconds = 0.0
        download_seconds = 0.0
        import urllib.error
        import urllib.parse
        import urllib.request

        def _colabcola_unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise RuntimeError("duplicate JSON key in OAuth or Drive response")
                result[key] = value
            return result

        def _colabcola_json_response(request):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    status = response.status
                    content = response.read()
            except urllib.error.HTTPError as exc:
                status = exc.code
                content = exc.read()
            try:
                value = json.loads(
                    content.decode("utf-8"), object_pairs_hook=_colabcola_unique_object
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Google endpoint returned invalid JSON (HTTP {status})"
                ) from exc
            if not isinstance(value, dict):
                raise RuntimeError("Google endpoint returned a non-object JSON value")
            return status, value

        staging_root = (
            COLABCOLA_CONTENT_ROOT / "colabcola" / "staging" / COLABCOLA_JOB_ID
        )
        if staging_root.exists():
            shutil.rmtree(staging_root)
        staging_root.mkdir(parents=True, exist_ok=False)

        source_override = globals().get("COLABCOLA_STAGED_BUNDLE_PATH_OVERRIDE")
        if source_override is None:
            from google.colab import userdata

            secret_text = userdata.get(COLABCOLA_GOOGLE_OAUTH_SECRET_NAME)
            try:
                client_document = json.loads(
                    secret_text, object_pairs_hook=_colabcola_unique_object
                )
            except Exception as exc:
                raise RuntimeError(
                    "Colab OAuth Secret is not valid Google client JSON"
                ) from exc
            if not isinstance(client_document, dict) or len(client_document) != 1:
                raise RuntimeError("Colab OAuth Secret must contain one client object")
            client_type, client = next(iter(client_document.items()))
            if client_type not in {"device", "installed"} or not isinstance(client, dict):
                raise RuntimeError("Colab OAuth Secret has an unsupported client type")
            required_client_fields = {
                "client_id",
                "client_secret",
                "project_id",
                "token_uri",
            }
            if not required_client_fields.issubset(client):
                raise RuntimeError("Colab OAuth Secret is missing required fields")
            if client["project_id"] != COLABCOLA_GOOGLE_PROJECT_ID:
                raise RuntimeError("Colab OAuth project does not match controller binding")
            if client["client_id"] != COLABCOLA_GOOGLE_OAUTH_CLIENT_ID:
                raise RuntimeError("Colab OAuth client ID does not match controller binding")
            if client["token_uri"] != "https://oauth2.googleapis.com/token":
                raise RuntimeError("Colab OAuth token endpoint is unexpected")
            client_secret = client["client_secret"]
            if not isinstance(client_secret, str) or not client_secret:
                raise RuntimeError("Colab OAuth client secret is invalid")

            device_body = urllib.parse.urlencode(
                {
                    "client_id": COLABCOLA_GOOGLE_OAUTH_CLIENT_ID,
                    "scope": COLABCOLA_GOOGLE_OAUTH_SCOPE,
                }
            ).encode("ascii")
            device_request = urllib.request.Request(
                "https://oauth2.googleapis.com/device/code",
                data=device_body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )
            status, device = _colabcola_json_response(device_request)
            if status != 200:
                raise RuntimeError(
                    "Google device authorization could not start: "
                    + str(device.get("error", f"HTTP {status}"))
                )
            device_code = device.get("device_code")
            user_code = device.get("user_code")
            verification_url = device.get("verification_url")
            expires_in = device.get("expires_in")
            interval = device.get("interval", 5)
            if (
                not isinstance(device_code, str)
                or not isinstance(user_code, str)
                or not isinstance(verification_url, str)
                or not isinstance(expires_in, int)
                or not isinstance(interval, int)
                or expires_in <= 0
                or interval <= 0
            ):
                raise RuntimeError("Google device authorization response is incomplete")
            print("Authorize colabcola per-file Drive access:")
            print(verification_url)
            print("Code:", user_code)
            print("The controller will continue automatically after approval.")
            sys.stdout.flush()

            deadline = time.monotonic() + expires_in
            access_token = None
            while time.monotonic() < deadline:
                time.sleep(interval)
                token_body = urllib.parse.urlencode(
                    {
                        "client_id": COLABCOLA_GOOGLE_OAUTH_CLIENT_ID,
                        "client_secret": client_secret,
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    }
                ).encode("ascii")
                token_request = urllib.request.Request(
                    "https://oauth2.googleapis.com/token",
                    data=token_body,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    method="POST",
                )
                token_status, token = _colabcola_json_response(token_request)
                error = token.get("error")
                if token_status == 200 and isinstance(token.get("access_token"), str):
                    granted_scopes = set(str(token.get("scope", "")).split())
                    if COLABCOLA_GOOGLE_OAUTH_SCOPE not in granted_scopes:
                        raise RuntimeError("Google OAuth did not grant drive.file scope")
                    if token.get("token_type") != "Bearer":
                        raise RuntimeError("Google OAuth returned an unexpected token type")
                    access_token = token["access_token"]
                    access_token_expires_in = token.get("expires_in", 3600)
                    refresh_token = token.get("refresh_token")
                    break
                if error == "authorization_pending":
                    continue
                if error == "slow_down":
                    interval += 5
                    continue
                raise RuntimeError(
                    "Google device authorization failed: "
                    + str(error or f"HTTP {token_status}")
                )
            if access_token is None:
                raise RuntimeError("Google device authorization expired")

            durable_checkpoints = COLABCOLA_MANIFEST.get("durable_checkpoints")
            if durable_checkpoints is not None:
                if (
                    not isinstance(refresh_token, str)
                    or not refresh_token
                    or isinstance(access_token_expires_in, bool)
                    or not isinstance(access_token_expires_in, int)
                    or access_token_expires_in <= 0
                ):
                    raise RuntimeError(
                        "Google OAuth did not provide refreshable credentials required "
                        "for durable checkpoints"
                    )
                _COLABCOLA_DRIVE_AUTH_CONTEXT = {
                    "access_token": access_token,
                    "access_token_expires_monotonic": (
                        time.monotonic() + max(0, access_token_expires_in - 60)
                    ),
                    "client_id": COLABCOLA_GOOGLE_OAUTH_CLIENT_ID,
                    "client_secret": client_secret,
                    "lock": threading.Lock(),
                    "refresh_token": refresh_token,
                }

            authorization = {"Authorization": "Bearer " + access_token}
            quoted_file_id = urllib.parse.quote(COLABCOLA_DRIVE_FILE_ID, safe="")
            metadata_url = (
                "https://www.googleapis.com/drive/v3/files/"
                + quoted_file_id
                + "?supportsAllDrives=true&fields="
                + urllib.parse.quote(
                    "id,name,parents,size,mimeType,appProperties,trashed", safe=","
                )
            )
            metadata_request = urllib.request.Request(metadata_url, headers=authorization)
            metadata_status, metadata = _colabcola_json_response(metadata_request)
            expected_properties = {
                "colabcola_job_id": COLABCOLA_JOB_ID,
                "colabcola_staged_sha256": COLABCOLA_STAGED_SHA256,
            }
            if (
                metadata_status != 200
                or metadata.get("id") != COLABCOLA_DRIVE_FILE_ID
                or metadata.get("name") != COLABCOLA_DRIVE_FILE_NAME
                or metadata.get("parents") != [COLABCOLA_DRIVE_PARENT_ID]
                or metadata.get("mimeType") != "application/octet-stream"
                or metadata.get("appProperties") != expected_properties
                or metadata.get("trashed") is not False
            ):
                raise RuntimeError("Drive metadata does not match controller binding")
            try:
                metadata_bytes = int(metadata.get("size"))
            except (TypeError, ValueError) as exc:
                raise RuntimeError("Drive metadata size is invalid") from exc
            if metadata_bytes != COLABCOLA_STAGED_BYTES:
                raise RuntimeError("Drive metadata byte count does not match")

            source_path = staging_root / ".payload.colabcola.tmp"
            downloaded_source = source_path
            download_url = (
                "https://www.googleapis.com/drive/v3/files/"
                + quoted_file_id
                + "?alt=media&supportsAllDrives=true"
            )
            download_started_monotonic = time.monotonic()
            authorization_and_metadata_seconds = (
                download_started_monotonic - acquisition_started_monotonic
            )
            download_request = urllib.request.Request(download_url, headers=authorization)
            download_digest = hashlib.sha256()
            downloaded_bytes = 0
            with urllib.request.urlopen(download_request, timeout=120) as response:
                with source_path.open("xb") as destination:
                    while block := response.read(8 * 1024 * 1024):
                        destination.write(block)
                        download_digest.update(block)
                        downloaded_bytes += len(block)
                    destination.flush()
                    os.fsync(destination.fileno())
            download_seconds = time.monotonic() - download_started_monotonic
            access_token = None
            client_secret = None
            secret_text = None
            if downloaded_bytes != COLABCOLA_STAGED_BYTES:
                raise RuntimeError("downloaded Drive object byte count does not match")
            if download_digest.hexdigest() != COLABCOLA_STAGED_SHA256:
                raise RuntimeError("downloaded Drive object SHA-256 does not match")
        else:
            source_path = pathlib.Path(str(source_override))

        reconstruction_started_monotonic = time.monotonic()
        source_metadata = os.lstat(source_path)
        source_attributes = getattr(source_metadata, "st_file_attributes", 0)
        if (
            stat.S_ISLNK(source_metadata.st_mode)
            or source_attributes
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            or not stat.S_ISREG(source_metadata.st_mode)
        ):
            raise RuntimeError("staged bundle is unavailable or unsafe")
        if source_metadata.st_size != COLABCOLA_STAGED_BYTES:
            raise RuntimeError("staged bundle byte count does not match")

        archive_path = staging_root / "archive.zip"
        temporary_archive = staging_root / ".archive.zip.tmp"

        whole_digest = hashlib.sha256()
        archive_digest = hashlib.sha256()
        with source_path.open("rb") as source:
            header = source.read(COLABCOLA_STAGED_HEADER_BYTES)
            whole_digest.update(header)
            if len(header) != COLABCOLA_STAGED_HEADER_BYTES:
                raise RuntimeError("staged bundle header is truncated")
            if header[:16].hex() != COLABCOLA_STAGED_MAGIC_HEX:
                raise RuntimeError("staged bundle magic does not match")
            manifest_bytes = int.from_bytes(header[16:24], "big")
            archive_bytes = int.from_bytes(header[24:32], "big")
            if manifest_bytes != len(COLABCOLA_MANIFEST_JSON.encode("utf-8")):
                raise RuntimeError("staged manifest byte count does not match")
            if archive_bytes != COLABCOLA_ARCHIVE_BYTES:
                raise RuntimeError("staged archive byte count does not match")

            staged_manifest = source.read(manifest_bytes)
            whole_digest.update(staged_manifest)
            if staged_manifest != COLABCOLA_MANIFEST_JSON.encode("utf-8"):
                raise RuntimeError("staged canonical manifest does not match")

            remaining = archive_bytes
            with temporary_archive.open("xb") as destination:
                while remaining:
                    block = source.read(min(8 * 1024 * 1024, remaining))
                    if not block:
                        raise RuntimeError("staged archive is truncated")
                    whole_digest.update(block)
                    archive_digest.update(block)
                    destination.write(block)
                    remaining -= len(block)
                destination.flush()
                os.fsync(destination.fileno())
            if source.read(1):
                raise RuntimeError("staged bundle has trailing bytes")

        if whole_digest.hexdigest() != COLABCOLA_STAGED_SHA256:
            raise RuntimeError("staged bundle SHA-256 does not match")
        if archive_digest.hexdigest() != COLABCOLA_ARCHIVE_SHA256:
            raise RuntimeError("staged archive SHA-256 does not match")
        os.replace(temporary_archive, archive_path)
        temporary_archive = None
        if downloaded_source is not None:
            downloaded_source.unlink(missing_ok=True)
            downloaded_source = None
        COLABCOLA_ARCHIVE_PATH = archive_path
        _COLABCOLA_TRANSPORT_FACTS = {
            "acquisition_seconds": (
                time.monotonic() - acquisition_started_monotonic
            ),
            "archive_bytes": COLABCOLA_ARCHIVE_BYTES,
            "authorization_and_metadata_seconds": (
                authorization_and_metadata_seconds
            ),
            "download_seconds": download_seconds,
            "drive_file_id": COLABCOLA_DRIVE_FILE_ID,
            "drive_relative_path": COLABCOLA_DRIVE_RELATIVE_PATH,
            "mode": "google_drive",
            "oauth_scope": COLABCOLA_GOOGLE_OAUTH_SCOPE,
            "staged_reconstruction_seconds": (
                time.monotonic() - reconstruction_started_monotonic
            ),
            "staged_bytes": COLABCOLA_STAGED_BYTES,
            "staged_sha256": COLABCOLA_STAGED_SHA256,
        }
    except Exception as exc:
        if temporary_archive is not None:
            temporary_archive.unlink(missing_ok=True)
        if downloaded_source is not None:
            downloaded_source.unlink(missing_ok=True)
        _colabcola_fail("integrity", "integrity_failed", exc)
'''


_DRIVE_EXTRACTION_SOURCE: Final[str] = r'''
# Stream-verify and safely extract the staged archive without loading it into RAM.
if _COLABCOLA_TERMINAL_STATE is None:
    try:
        extraction_started_monotonic = time.monotonic()
        archive_metadata = os.lstat(COLABCOLA_ARCHIVE_PATH)
        if not stat.S_ISREG(archive_metadata.st_mode):
            raise RuntimeError("local staged archive is unavailable or unsafe")
        if archive_metadata.st_size != COLABCOLA_ARCHIVE_BYTES:
            raise RuntimeError("local staged archive byte count does not match")
        archive_digest = hashlib.sha256()
        with COLABCOLA_ARCHIVE_PATH.open("rb") as archive_stream:
            while block := archive_stream.read(8 * 1024 * 1024):
                archive_digest.update(block)
        if archive_digest.hexdigest() != COLABCOLA_ARCHIVE_SHA256:
            raise RuntimeError("local staged archive SHA-256 does not match")
        framed_identity = (
            len(COLABCOLA_MANIFEST_JSON.encode("utf-8")).to_bytes(8, "big")
            + COLABCOLA_MANIFEST_JSON.encode("utf-8")
            + len(archive_digest.digest()).to_bytes(8, "big")
            + archive_digest.digest()
        )
        if hashlib.sha256(framed_identity).hexdigest() != COLABCOLA_JOB_ID:
            raise RuntimeError("job ID does not match manifest and archive identities")

        if COLABCOLA_JOB_ROOT.exists():
            shutil.rmtree(COLABCOLA_JOB_ROOT)
        COLABCOLA_JOB_ROOT.mkdir(parents=True, exist_ok=False)
        manifest_files = COLABCOLA_MANIFEST["files"]
        with zipfile.ZipFile(COLABCOLA_ARCHIVE_PATH, "r") as archive:
            infos = archive.infolist()
            if [info.filename for info in infos] != [
                record["path"] for record in manifest_files
            ]:
                raise RuntimeError("archive entry order or paths do not match manifest")
            if len({info.filename for info in infos}) != len(infos):
                raise RuntimeError("archive contains duplicate entry paths")

            for info, record in zip(infos, manifest_files, strict=True):
                path = pathlib.PurePosixPath(info.filename)
                if (
                    path.is_absolute()
                    or path.as_posix() != info.filename
                    or any(
                        part in {"", ".", ".."} or ":" in part
                        for part in path.parts
                    )
                    or info.is_dir()
                    or info.file_size != record["size"]
                ):
                    raise RuntimeError(f"unsafe or mismatched archive entry: {info.filename!r}")
                archive_mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_IFMT(archive_mode) != stat.S_IFREG:
                    raise RuntimeError(
                        f"archive entry is not a regular file: {info.filename}"
                    )

                target = COLABCOLA_JOB_ROOT.joinpath(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.resolve().relative_to(COLABCOLA_JOB_ROOT.resolve())
                size = 0
                digest = hashlib.sha256()
                try:
                    with archive.open(info, "r") as source, target.open("xb") as output:
                        while block := source.read(8 * 1024 * 1024):
                            size += len(block)
                            if size > record["size"]:
                                raise RuntimeError(
                                    f"archive size exceeds manifest for {info.filename}"
                                )
                            digest.update(block)
                            output.write(block)
                    if size != record["size"]:
                        raise RuntimeError(
                            f"archive size does not match for {info.filename}"
                        )
                    if digest.hexdigest() != record["sha256"]:
                        raise RuntimeError(
                            f"archive SHA-256 does not match for {info.filename}"
                        )
                    target.chmod(int(record["mode"], 8))
                except Exception:
                    target.unlink(missing_ok=True)
                    raise

        COLABCOLA_ARCHIVE_PATH.unlink(missing_ok=True)
        _COLABCOLA_TRANSPORT_FACTS["extraction_seconds"] = (
            time.monotonic() - extraction_started_monotonic
        )
    except Exception as exc:
        _colabcola_fail("integrity", "integrity_failed", exc)
'''


_ENVIRONMENT_SOURCE: Final[str] = r'''
# Install only declared dependencies and recheck the resolved environment.
if _COLABCOLA_TERMINAL_STATE is None:
    try:
        python_manifest = COLABCOLA_MANIFEST["python"]
        embedded_wheels = []
        for relative in python_manifest["embedded_wheels"]:
            wheel = COLABCOLA_JOB_ROOT.joinpath(*pathlib.PurePosixPath(relative).parts)
            wheel.resolve().relative_to(COLABCOLA_JOB_ROOT.resolve())
            if not wheel.is_file() or wheel.is_symlink():
                raise RuntimeError(f"embedded wheel is unavailable or unsafe: {relative}")
            embedded_wheels.append(str(wheel))

        if embedded_wheels and _COLABCOLA_ISOLATION_FACTS is None:
            command = [
                _COLABCOLA_PYTHON,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-index",
                "--no-deps",
                *embedded_wheels,
            ]
            completed = subprocess.run(
                command,
                cwd=COLABCOLA_JOB_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "embedded-wheel installation failed: " + completed.stderr[-2000:]
                )

        install_requirements = python_manifest["install_requirements"]
        if install_requirements and _COLABCOLA_ISOLATION_FACTS is None:
            requirements_path = COLABCOLA_RESULT_ROOT / ".install-requirements.txt"
            requirements_path.write_text(
                "\n".join(install_requirements) + "\n", encoding="utf-8", newline="\n"
            )
            try:
                completed = subprocess.run(
                    [
                        _COLABCOLA_PYTHON,
                        "-m",
                        "pip",
                        "install",
                        "--disable-pip-version-check",
                        "--require-hashes",
                        "-r",
                        str(requirements_path),
                    ],
                    cwd=COLABCOLA_JOB_ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            finally:
                requirements_path.unlink(missing_ok=True)
            if completed.returncode != 0:
                raise RuntimeError(
                    "declared dependency installation failed: " + completed.stderr[-2000:]
                )

        pip_requirements = python_manifest.get("pip_requirements", [])
        if pip_requirements and _COLABCOLA_ISOLATION_FACTS is None:
            completed = subprocess.run(
                [_COLABCOLA_PYTHON, "-m", "pip", "install", "--disable-pip-version-check", *pip_requirements],
                cwd=COLABCOLA_JOB_ROOT, check=False, capture_output=True, text=True,
                timeout=COLABCOLA_MANIFEST["run"]["timeout_seconds"],
            )
            if completed.returncode != 0:
                raise RuntimeError("declared PyPI dependency installation failed: " + completed.stderr[-2000:])
        if _COLABCOLA_ISOLATION_FACTS is not None:
            setup_log = COLABCOLA_RESULT_ROOT / "environment.log"
            if embedded_wheels:
                _colabcola_isolated_install(_COLABCOLA_PYTHON,
                    ["install", "--disable-pip-version-check", "--no-index", "--no-deps", *embedded_wheels],
                    COLABCOLA_JOB_ROOT, setup_log)
            if install_requirements:
                requirements_path = COLABCOLA_RESULT_ROOT / ".install-requirements.txt"
                requirements_path.write_text("\n".join(install_requirements)+"\n", encoding="utf-8")
                try:
                    _colabcola_isolated_install(_COLABCOLA_PYTHON,
                        ["install", "--disable-pip-version-check", "--require-hashes", "-r", str(requirements_path)],
                        COLABCOLA_JOB_ROOT, setup_log)
                finally:
                    requirements_path.unlink(missing_ok=True)
            if pip_requirements:
                _colabcola_isolated_install(_COLABCOLA_PYTHON,
                    ["install", "--disable-pip-version-check", *pip_requirements], COLABCOLA_JOB_ROOT, setup_log)
            _colabcola_isolated_install(_COLABCOLA_PYTHON, ["check"], COLABCOLA_JOB_ROOT, setup_log)
        requirements_to_check = list(python_manifest["runtime_requirements"]) + list(pip_requirements)
        requirements_to_check.extend(
            line.split(" --hash=", 1)[0].strip() for line in install_requirements
        )
        package_records, package_problems = _colabcola_check_requirements(
            requirements_to_check
        )
        if package_problems:
            raise RuntimeError("; ".join(package_problems))
        _COLABCOLA_ENVIRONMENT_FACTS = {
            "isolation": _COLABCOLA_ISOLATION_FACTS,
            "embedded_wheels": list(python_manifest["embedded_wheels"]),
            "install_requirements": list(install_requirements),
            "pip_requirements": list(pip_requirements),
            "resolved_packages": {
                record["requirement"]: record["version"] for record in package_records
            },
        }
    except Exception as exc:
        _colabcola_fail("environment", "environment_failed", exc)
'''


_EXECUTION_SOURCE: Final[str] = r'''
# Execute without a shell, validate outputs, and make result evidence durable.
stdout_path = COLABCOLA_RESULT_ROOT / "stdout.log"
stderr_path = COLABCOLA_RESULT_ROOT / "stderr.log"
stdout_path.touch(exist_ok=True)
stderr_path.touch(exist_ok=True)


def _colabcola_kill_workload(process):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if process.poll() is None:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _colabcola_emit(stream, block):
    binary = getattr(stream, "buffer", None)
    if binary is not None:
        binary.write(block)
        binary.flush()
    else:
        stream.write(block.decode("utf-8", errors="replace"))
        stream.flush()


def _colabcola_drain(pipe, display, destination, capture):
    import codecs
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    binary = getattr(display, "buffer", None)
    while True:
        # BufferedReader.read(n) waits for n bytes or EOF. read1 forwards
        # currently available bytes, including tqdm's carriage-return updates.
        block = pipe.read1(64 * 1024)
        if not block:
            break
        if capture:
            destination.write(block)
            destination.flush()
        if binary is not None:
            _colabcola_emit(display, block)
        else:
            display.write(decoder.decode(block))
            display.flush()
    if binary is None:
        display.write(decoder.decode(b"", final=True))
        display.flush()


if _COLABCOLA_TERMINAL_STATE is None:
    try:
        checkpoint_config = _colabcola_checkpoint_config()
        checkpoint_stop = None
        checkpoint_thread = None
        checkpoint_errors = []
        checkpoint_records = []
        checkpoint_root = None
        checkpoint_folder_id = None
        if checkpoint_config is not None:
            checkpoint_root = COLABCOLA_JOB_ROOT.joinpath(
                *pathlib.PurePosixPath(checkpoint_config["directory"]).parts
            )
            checkpoint_root.resolve().relative_to(COLABCOLA_JOB_ROOT.resolve())
            checkpoint_folder_id = _colabcola_checkpoint_folder()
            checkpoint_records = _colabcola_committed_checkpoints(
                checkpoint_folder_id
            )
            restored_step = None
            if checkpoint_records:
                restored_step = _colabcola_restore_checkpoint(
                    checkpoint_root,
                    checkpoint_folder_id,
                    checkpoint_records[-1],
                )
            _COLABCOLA_DURABLE_CHECKPOINT_FACTS = {
                "checkpoint_folder_id": checkpoint_folder_id,
                "configured": True,
                "publications": [],
                "remote_retention": checkpoint_config["remote_retention"],
                "restored_step": restored_step,
                "retained": _colabcola_checkpoint_record_facts(checkpoint_records),
            }

        entrypoint = COLABCOLA_JOB_ROOT.joinpath(
            *pathlib.PurePosixPath(COLABCOLA_MANIFEST["entrypoint"]["path"]).parts
        )
        cwd_value = COLABCOLA_MANIFEST["entrypoint"]["cwd"]
        cwd = COLABCOLA_JOB_ROOT if cwd_value == "." else COLABCOLA_JOB_ROOT.joinpath(
            *pathlib.PurePosixPath(cwd_value).parts
        )
        entrypoint.resolve().relative_to(COLABCOLA_JOB_ROOT.resolve())
        cwd.resolve().relative_to(COLABCOLA_JOB_ROOT.resolve())
        if not entrypoint.is_file() or entrypoint.is_symlink():
            raise RuntimeError("entrypoint is unavailable or unsafe after extraction")
        if not cwd.is_dir() or cwd.is_symlink():
            raise RuntimeError("entrypoint working directory is unavailable or unsafe")

        command = [
            _COLABCOLA_PYTHON,
            str(entrypoint),
            *COLABCOLA_MANIFEST["entrypoint"]["argv"],
        ]
        environment = os.environ.copy()
        environment.setdefault("PYTHONUNBUFFERED", "1")
        environment.update(COLABCOLA_MANIFEST["run"]["environment"])
        if _COLABCOLA_ISOLATION_FACTS is not None:
            environment = _colabcola_isolated_env(_COLABCOLA_PYTHON, COLABCOLA_MANIFEST["run"]["environment"])
        environment["COLABCOLA_SEED"] = str(COLABCOLA_MANIFEST["run"]["seed"])

        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                shell=False,
                start_new_session=(os.name != "nt"),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assert process.stdout is not None and process.stderr is not None
            stdout_thread = threading.Thread(
                target=_colabcola_drain,
                args=(
                    process.stdout,
                    sys.stdout,
                    stdout_file,
                    COLABCOLA_MANIFEST["run"]["capture_stdout"],
                ),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=_colabcola_drain,
                args=(
                    process.stderr,
                    sys.stderr,
                    stderr_file,
                    COLABCOLA_MANIFEST["run"]["capture_stderr"],
                ),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()
            if checkpoint_config is not None:
                checkpoint_stop = threading.Event()
                checkpoint_thread = threading.Thread(
                    target=_colabcola_checkpoint_worker,
                    args=(
                        checkpoint_root,
                        checkpoint_config,
                        checkpoint_folder_id,
                        checkpoint_records,
                        checkpoint_stop,
                        checkpoint_errors,
                        _COLABCOLA_DURABLE_CHECKPOINT_FACTS,
                    ),
                    daemon=True,
                )
                checkpoint_thread.start()

            execution_error = None
            deadline = time.monotonic() + COLABCOLA_MANIFEST["run"][
                "timeout_seconds"
            ]
            lifecycle_deadline = globals().get("COLABCOLA_LIFECYCLE_DEADLINE")
            if lifecycle_deadline is not None:
                deadline = min(deadline, time.monotonic() + max(0, lifecycle_deadline - time.time() - 2))
            while process.poll() is None:
                if checkpoint_errors:
                    _colabcola_kill_workload(process)
                    execution_error = RuntimeError(
                        "durable checkpoint publication failed: "
                        + _colabcola_sanitize_exception(checkpoint_errors[0])
                    )
                    break
                if time.monotonic() >= deadline:
                    _colabcola_kill_workload(process)
                    execution_error = TimeoutError(
                        "job exceeded timeout_seconds="
                        + str(COLABCOLA_MANIFEST["run"]["timeout_seconds"])
                    )
                    break
                time.sleep(1)
            _COLABCOLA_PROCESS_EXIT_CODE = process.wait()

            if checkpoint_stop is not None:
                checkpoint_stop.set()
            if checkpoint_thread is not None:
                checkpoint_thread.join()
            if execution_error is None and checkpoint_errors:
                execution_error = RuntimeError(
                    "durable checkpoint publication failed: "
                    + _colabcola_sanitize_exception(checkpoint_errors[0])
                )
            stdout_thread.join()
            stderr_thread.join()
            process.stdout.close()
            process.stderr.close()
            if execution_error is not None:
                raise execution_error

        if _COLABCOLA_PROCESS_EXIT_CODE != 0:
            raise RuntimeError(
                f"job process exited with code {_COLABCOLA_PROCESS_EXIT_CODE}"
            )
    except Exception as exc:
        _colabcola_fail("execution", "job_failed", exc)


if _COLABCOLA_DRIVE_AUTH_CONTEXT is not None:
    with _COLABCOLA_DRIVE_AUTH_CONTEXT["lock"]:
        for sensitive_key in (
            "access_token",
            "client_id",
            "client_secret",
            "refresh_token",
        ):
            _COLABCOLA_DRIVE_AUTH_CONTEXT[sensitive_key] = None


if _COLABCOLA_TERMINAL_STATE is None:
    try:
        output_records = {}
        job_root_resolved = COLABCOLA_JOB_ROOT.resolve()
        for declaration in COLABCOLA_MANIFEST["outputs"]:
            pattern = declaration["pattern"]
            matches = sorted(
                pathlib.Path(value)
                for value in glob.glob(
                    str(COLABCOLA_JOB_ROOT / pathlib.PurePosixPath(pattern)),
                    recursive=True,
                )
            )
            matched_files = []
            for candidate in matches:
                if candidate.is_dir():
                    continue
                # Some Windows glob implementations yield the nonexistent
                # static directory prefix for a trailing '**' (outputs/**).
                # Ignore that placeholder only, not vanished output files or
                # dangling symlinks; those must still fail validation.
                if (pattern.endswith("/**")
                        and candidate == COLABCOLA_JOB_ROOT / pathlib.PurePosixPath(pattern).parent
                        and not os.path.lexists(candidate)):
                    continue
                metadata = os.lstat(candidate)
                attributes = getattr(metadata, "st_file_attributes", 0)
                if (
                    stat.S_ISLNK(metadata.st_mode)
                    or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                    or not stat.S_ISREG(metadata.st_mode)
                ):
                    raise RuntimeError(f"unsafe output match: {candidate}")
                candidate.resolve().relative_to(job_root_resolved)
                relative = candidate.relative_to(COLABCOLA_JOB_ROOT).as_posix()
                size = candidate.stat().st_size
                with candidate.open("rb") as output_stream:
                    digest = hashlib.file_digest(output_stream, "sha256").hexdigest()
                matched_files.append((relative, size))
                output_records[relative] = {
                    "path": relative,
                    "size": size,
                    "sha256": digest,
                }
            if declaration["required"] and not matched_files:
                raise RuntimeError(f"required output pattern matched no files: {pattern}")
            matched_bytes = sum(size for _, size in matched_files)
            if matched_bytes > declaration["maximum_bytes"]:
                raise RuntimeError(
                    f"output pattern {pattern!r} produced {matched_bytes} bytes, "
                    f"exceeding maximum_bytes={declaration['maximum_bytes']}"
                )
        _COLABCOLA_OUTPUTS = [output_records[path] for path in sorted(output_records)]
        _COLABCOLA_TERMINAL_STATE = "success"
    except Exception as exc:
        _colabcola_fail("output_validation", "output_validation_failed", exc)


def _colabcola_file_evidence(path, captured):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "captured": captured,
        "bytes": path.stat().st_size,
        "sha256": digest,
        "truncated": False,
    }


def _colabcola_result_value():
    ended_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    duration = max(0.0, time.monotonic() - _COLABCOLA_RUNTIME_STARTED_MONOTONIC)
    value = {
        "archive_sha256": COLABCOLA_ARCHIVE_SHA256,
        "artifacts_zip": "artifacts.zip",
        "compiler_contract_version": COLABCOLA_CONTRACT_VERSION,
        "compiler_version": COLABCOLA_COMPILER_VERSION,
        "duration_seconds": duration,
        "ended_at_utc": ended_at,
        "environment": _COLABCOLA_ENVIRONMENT_FACTS,
        "failure": _COLABCOLA_FAILURE,
        "job_id": COLABCOLA_JOB_ID,
        "manifest_sha256": COLABCOLA_MANIFEST_SHA256,
        "notebook_sha256": None,
        "notebook_sha256_status": "available from compiler or launcher receipt only",
        "outputs": _COLABCOLA_OUTPUTS,
        "process_exit_code": _COLABCOLA_PROCESS_EXIT_CODE,
        "runtime": _COLABCOLA_RUNTIME_FACTS,
        "started_at_utc": _COLABCOLA_RUNTIME_STARTED_AT,
        "stderr": _colabcola_file_evidence(
            stderr_path, COLABCOLA_MANIFEST["run"]["capture_stderr"]
        ),
        "stdout": _colabcola_file_evidence(
            stdout_path, COLABCOLA_MANIFEST["run"]["capture_stdout"]
        ),
        "terminal_state": _COLABCOLA_TERMINAL_STATE,
        "transport": _COLABCOLA_TRANSPORT_FACTS,
    }
    if _COLABCOLA_DURABLE_CHECKPOINT_FACTS is not None:
        value["durable_checkpoints"] = _COLABCOLA_DURABLE_CHECKPOINT_FACTS
    return value


def _colabcola_json_bytes(value):
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _colabcola_atomic_write(path, content):
    temporary = path.with_name("." + path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


result_path = COLABCOLA_RESULT_ROOT / "result.json"
artifacts_path = COLABCOLA_RESULT_ROOT / "artifacts.zip"
result_bytes = _colabcola_json_bytes(_colabcola_result_value())
_colabcola_atomic_write(result_path, result_bytes)

try:
    temporary_artifacts = COLABCOLA_RESULT_ROOT / ".artifacts.zip.tmp"
    with zipfile.ZipFile(
        temporary_artifacts,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as artifact_archive:
        artifact_archive.comment = b""
        result_info = zipfile.ZipInfo("result.json", (1980, 1, 1, 0, 0, 0))
        result_info.compress_type = zipfile.ZIP_DEFLATED
        result_info.create_system = 3
        result_info.external_attr = (stat.S_IFREG | 0o644) << 16
        artifact_archive.writestr(result_info, result_bytes)
        for record in _COLABCOLA_OUTPUTS:
            output_path = COLABCOLA_JOB_ROOT.joinpath(
                *pathlib.PurePosixPath(record["path"]).parts
            )
            output_info = zipfile.ZipInfo(record["path"], (1980, 1, 1, 0, 0, 0))
            output_info.compress_type = zipfile.ZIP_DEFLATED
            output_info.create_system = 3
            output_info.external_attr = (stat.S_IFREG | 0o644) << 16
            output_info.file_size = record["size"]
            streamed_size, streamed_digest = 0, hashlib.sha256()
            with output_path.open("rb") as source, artifact_archive.open(
                output_info, "w", force_zip64=record["size"] >= zipfile.ZIP64_LIMIT
            ) as target:
                while block := source.read(8 * 1024 * 1024):
                    streamed_size += len(block)
                    if streamed_size > record["size"]:
                        raise RuntimeError("validated output grew during packaging")
                    streamed_digest.update(block)
                    target.write(block)
            if streamed_size != record["size"] or streamed_digest.hexdigest() != record["sha256"]:
                raise RuntimeError(f"validated output changed before packaging: {record['path']}")
    os.replace(temporary_artifacts, artifacts_path)
except Exception as exc:
    temporary_artifacts.unlink(missing_ok=True)
    if _COLABCOLA_TERMINAL_STATE == "success":
        _COLABCOLA_TERMINAL_STATE = None
        _colabcola_fail("result_packaging", "output_validation_failed", exc)
        result_bytes = _colabcola_json_bytes(_colabcola_result_value())
        _colabcola_atomic_write(result_path, result_bytes)

print(f"colabcola result: {result_path}")
if artifacts_path.exists():
    artifacts_digest = hashlib.sha256()
    artifacts_bytes = 0
    with artifacts_path.open("rb") as artifacts_stream:
        while True:
            artifacts_block = artifacts_stream.read(1024 * 1024)
            if not artifacts_block:
                break
            artifacts_digest.update(artifacts_block)
            artifacts_bytes += len(artifacts_block)
    print(f"colabcola artifacts: {artifacts_path}")
    print(
        "colabcola artifacts evidence: "
        + json.dumps(
            {
                "name": artifacts_path.name,
                "sha256": artifacts_digest.hexdigest(),
                "size": artifacts_bytes,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
if _COLABCOLA_TERMINAL_STATE != "success":
    raise RuntimeError(
        "colabcola job ended in "
        + str(_COLABCOLA_TERMINAL_STATE)
        + "; inspect "
        + str(result_path)
    )
'''


_REMOTE_INPUTS_SOURCE: Final[str] = r'''
# Acquire declared remote inputs beneath the job root, verifying byte count and SHA-256 (contract 0.14).
if _COLABCOLA_TERMINAL_STATE is None and COLABCOLA_MANIFEST.get("remote_inputs"):
    import http.client
    import urllib.error
    import urllib.request

    class _ColabcolaRemoteDiskError(RuntimeError):
        pass

    _COLABCOLA_REMOTE_RETRYABLE = (
        urllib.error.URLError,
        http.client.HTTPException,
        ConnectionError,
        TimeoutError,
        OSError,
    )

    def _colabcola_remote_progress(state, *, force=False):
        global remote_last_progress
        now = time.monotonic()
        if not force and now - remote_last_progress < 2:
            return
        remote_last_progress = now
        received = remote_verified_bytes + (0 if state == "verified" else remote_have)
        file_bytes = remote_expected if state == "verified" else remote_have
        elapsed = max(now - remote_started_monotonic, 0.001)
        message = (
            "Remote inputs: " + str(len(remote_facts)) + "/" + str(len(remote_entries))
            + " files verified | " + str(received) + "/" + str(remote_total_bytes)
            + " bytes received | " + format(received / elapsed / (1024 ** 2), ".1f")
            + " MiB/s | file " + str(remote_index) + "/" + str(len(remote_entries))
            + " " + remote_entry["target"] + " " + str(file_bytes) + "/"
            + str(remote_expected) + " bytes | " + state
            + " (attempt " + str(remote_attempts) + ")"
        )
        # URLs (including signed query strings) and HTTP headers never enter this log.
        message = "".join(c if c.isprintable() else " " for c in message)
        COLABCOLA_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        with (COLABCOLA_RESULT_ROOT / "transfer.log").open("a", encoding="utf-8") as log:
            log.write(message + "\n")
            log.flush()
        print(message, flush=True)

    try:
        remote_started_monotonic = time.monotonic()
        remote_last_progress = float("-inf")
        remote_verified_bytes = 0
        remote_entries = COLABCOLA_MANIFEST["remote_inputs"]
        remote_total_bytes = sum(int(entry["bytes"]) for entry in remote_entries)
        remote_free_bytes = shutil.disk_usage(COLABCOLA_JOB_ROOT).free
        if remote_free_bytes < remote_total_bytes + 1024 ** 3:
            raise _ColabcolaRemoteDiskError(
                "insufficient free disk for remote inputs: need "
                + str(remote_total_bytes)
                + " bytes plus a 1 GiB margin, have "
                + str(remote_free_bytes)
            )
        remote_facts = []
        for remote_index, remote_entry in enumerate(remote_entries, 1):
            remote_url = str(remote_entry["url"])
            remote_expected = int(remote_entry["bytes"])
            remote_expected_sha256 = str(remote_entry["sha256"]).lower()
            if not remote_url.startswith("https://"):
                raise RuntimeError("remote input URL is not HTTPS: " + remote_entry["target"])
            remote_path = pathlib.PurePosixPath(remote_entry["target"])
            if (
                remote_path.is_absolute()
                or remote_path.as_posix() != remote_entry["target"]
                or any(
                    part in {"", ".", ".."} or ":" in part for part in remote_path.parts
                )
            ):
                raise RuntimeError("unsafe remote input target: " + repr(remote_entry["target"]))
            remote_target = COLABCOLA_JOB_ROOT.joinpath(*remote_path.parts)
            remote_target.parent.mkdir(parents=True, exist_ok=True)
            remote_target.resolve().relative_to(COLABCOLA_JOB_ROOT.resolve())
            if remote_target.exists():
                raise RuntimeError(
                    "remote input target already exists: " + remote_entry["target"]
                )
            remote_partial = remote_target.with_name(remote_target.name + ".partial")
            remote_entry_started = time.monotonic()
            remote_attempts = 0
            remote_status = None
            while True:
                remote_attempts += 1
                try:
                    remote_have = (
                        remote_partial.stat().st_size if remote_partial.exists() else 0
                    )
                    if remote_have > remote_expected:
                        remote_partial.unlink()
                        remote_have = 0
                    _colabcola_remote_progress("connecting", force=True)
                    remote_digest = hashlib.sha256()
                    if remote_have:
                        with remote_partial.open("rb") as remote_existing:
                            while remote_block := remote_existing.read(8 * 1024 * 1024):
                                remote_digest.update(remote_block)
                    remote_request = urllib.request.Request(
                        remote_url,
                        headers={"User-Agent": "colabcola/" + COLABCOLA_COMPILER_VERSION},
                    )
                    if remote_have:
                        remote_request.add_header("Range", "bytes=" + str(remote_have) + "-")
                    with urllib.request.urlopen(remote_request, timeout=120) as remote_response:
                        remote_status = getattr(remote_response, "status", None)
                        if remote_have and remote_status == 206:
                            remote_mode = "ab"
                        elif remote_status == 200:
                            remote_have = 0
                            remote_digest = hashlib.sha256()
                            remote_mode = "wb"
                        else:
                            raise RuntimeError(
                                "unexpected HTTP status "
                                + str(remote_status)
                                + " for remote input "
                                + remote_entry["target"]
                            )
                        with remote_partial.open(remote_mode) as remote_output:
                            while remote_block := remote_response.read(8 * 1024 * 1024):
                                remote_have += len(remote_block)
                                if remote_have > remote_expected:
                                    raise RuntimeError(
                                        "remote input exceeds its declared byte count: "
                                        + remote_entry["target"]
                                    )
                                remote_digest.update(remote_block)
                                remote_output.write(remote_block)
                                _colabcola_remote_progress("downloading")
                    if remote_have != remote_expected:
                        raise http.client.IncompleteRead(b"", remote_expected - remote_have)
                    if remote_digest.hexdigest() != remote_expected_sha256:
                        raise RuntimeError(
                            "remote input SHA-256 does not match: " + remote_entry["target"]
                        )
                    os.replace(remote_partial, remote_target)
                    remote_target.chmod(0o644)
                    break
                except _COLABCOLA_REMOTE_RETRYABLE as remote_exc:
                    if remote_attempts >= 3:
                        _colabcola_remote_progress("failed", force=True)
                        remote_partial.unlink(missing_ok=True)
                        raise RuntimeError(
                            "remote input download failed after "
                            + str(remote_attempts)
                            + " attempts: "
                            + remote_entry["target"]
                            + ": "
                            + _colabcola_sanitize_exception(remote_exc)
                        ) from remote_exc
                    _colabcola_remote_progress("retrying", force=True)
                    time.sleep(1.0)
                except Exception:
                    _colabcola_remote_progress("verification failed", force=True)
                    remote_partial.unlink(missing_ok=True)
                    raise
            remote_facts.append(
                {
                    "attempts": remote_attempts,
                    "bytes": remote_expected,
                    "http_status": remote_status,
                    "seconds": time.monotonic() - remote_entry_started,
                    "sha256": remote_expected_sha256,
                    "target": remote_entry["target"],
                }
            )
            remote_verified_bytes += remote_expected
            remote_have = 0
            _colabcola_remote_progress("verified", force=True)
        _COLABCOLA_TRANSPORT_FACTS["remote_inputs"] = {
            "bytes": remote_total_bytes,
            "count": len(remote_entries),
            "entries": remote_facts,
            "seconds": time.monotonic() - remote_started_monotonic,
        }
    except _ColabcolaRemoteDiskError as exc:
        _colabcola_fail("remote_inputs", "preflight_failed", exc)
    except Exception as exc:
        _colabcola_fail("remote_inputs", "remote_input_failed", exc)
'''


_ISOLATION_SETUP_SOURCE = r'''
if _COLABCOLA_TERMINAL_STATE is None and COLABCOLA_MANIFEST["python"].get("isolation"):
    try:
        _COLABCOLA_PYTHON, _COLABCOLA_ISOLATION_FACTS = _colabcola_prepare_python(
            COLABCOLA_MANIFEST["python"]["isolation"],
            COLABCOLA_CONTENT_ROOT / "colabcola/environments" / COLABCOLA_JOB_ID)
        _COLABCOLA_ENVIRONMENT_FACTS["isolation"] = _COLABCOLA_ISOLATION_FACTS
        _, SpecifierSet = _colabcola_requirement_tools()
        if not SpecifierSet(COLABCOLA_MANIFEST["python"]["requires_python"]).contains(_COLABCOLA_ISOLATION_FACTS["python"]):
            raise RuntimeError("isolated Python violates the workload constraint")
    except Exception as exc:
        _colabcola_fail("environment", "environment_failed", exc)
'''


def runtime_cell_sources(*, isolated=False) -> tuple[tuple[str, str], ...]:
    """Return ordered group labels and stable code-cell source strings."""

    return (
        ("preflight", ((_ISOLATION_HELPERS + "\n" if isolated else "") + _GPU_CHECK_SOURCE + "\n" + _PREFLIGHT_SOURCE).strip()),
        ("extraction", (_EXTRACTION_SOURCE + "\n" + (_ISOLATION_SETUP_SOURCE if isolated else _REMOTE_INPUTS_SOURCE)).strip()),
        ("environment", (_ENVIRONMENT_SOURCE + ("\n" + _REMOTE_INPUTS_SOURCE if isolated else "")).strip()),
        ("execution", (DURABLE_CHECKPOINT_SOURCE + "\n" + _EXECUTION_SOURCE).strip()),
    )


def drive_runtime_cell_sources(*, isolated=False) -> tuple[tuple[str, str], ...]:
    """Return the Drive controller's ordered runtime cell groups."""

    return (
        ("preflight", ((_ISOLATION_HELPERS + "\n" if isolated else "") + _GPU_CHECK_SOURCE + "\n" + _PREFLIGHT_SOURCE).strip()),
        ("acquisition", _DRIVE_ACQUISITION_SOURCE.strip()),
        ("extraction", (_DRIVE_EXTRACTION_SOURCE + "\n" + (_ISOLATION_SETUP_SOURCE if isolated else _REMOTE_INPUTS_SOURCE)).strip()),
        ("environment", (_ENVIRONMENT_SOURCE + ("\n" + _REMOTE_INPUTS_SOURCE if isolated else "")).strip()),
        ("execution", (DURABLE_CHECKPOINT_SOURCE + "\n" + _EXECUTION_SOURCE).strip()),
    )
