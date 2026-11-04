"""Command-line entry point for the implemented colabcola surface."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ._version import __version__
from .compiler import (
    compile_drive_controller_invocation,
    compile_invocation,
    prepare_drive_invocation,
)
from .controller import (
    build_drive_controller,
    controller_auth_from_settings,
    write_drive_controller,
)
from .drive import (
    drive_receipt_for_observed_upload,
    stage_drive_bundle,
    write_drive_receipt,
)
from .lifecycle_io import LifecycleError
from .profiles import add_profile_argument, selected
from .errors import BundleError, DriveTransportError, SpecError
from .google_drive import (
    DEFAULT_COLAB_SECRET_NAME,
    GOOGLE_DRIVE_SCOPE,
    create_google_drive_backend,
    google_drive_settings_from_clients,
    load_google_drive_settings,
    write_google_drive_settings,
)
from .invocation import JobInvocation, load_job_invocation
from .transport import build_staged_bundle
from .bundle import build_bundle


@dataclass(frozen=True, slots=True)
class SubmissionCommand:
    """A parsed request to prepare one job invocation."""

    job_file: Path
    job_args: tuple[str, ...]
    output: Path | None
    drive_config: Path | None
    mode: str = "submit"
    transport: str = "drive"
    execution_config: Path | None = None
    run_dir: Path | None = None
    prepare_only: bool = False
    resume_from: Path | None = None
    approve: str | None = None
    preflight_only: bool = False
    allow_active_assignments: tuple[str, ...] = ()
    profile: str | None = None


@dataclass(frozen=True, slots=True)
class ControllerCommand:
    """A parsed request to bind a tiny controller to one observed Drive upload."""

    job_file: Path
    job_args: tuple[str, ...]
    output: Path | None
    receipt: Path | None
    drive_file_id: str
    drive_parent_id: str
    drive_relative_path: str
    reused: bool
    drive_config: Path | None
    profile: str | None = None


@dataclass(frozen=True, slots=True)
class GoogleDriveAuthCommand:
    """Configure same-project desktop and Colab OAuth clients."""

    desktop_client_secrets: Path
    colab_client_secrets: Path
    colab_secret_name: str
    drive_config: Path | None
    profile: str | None = None


@dataclass(frozen=True, slots=True)
class ValidateCommand:
    """A parsed request to validate one job file."""

    job_file: Path
    profile: str | None = None


Command = SubmissionCommand | ControllerCommand | GoogleDriveAuthCommand | ValidateCommand


def _submission_parser(*, prog: str = "colabcola") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Run managed .colajobs with progress and recovery, stage other jobs for their "
            "Drive controller, or explicitly compile a self-contained notebook."
        ),
        epilog="Use colabcola --profile NAME <command> to select an account for one invocation. Set up account access with colabcola login. Run recipes with colabcola run --help. Create jobs with colabcola package --help. Arguments after JOB.colajob are forwarded unchanged to the job entrypoint.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "output path (defaults to JOB.colabcola for Drive or JOB.ipynb for embedded)"
        ),
    )
    parser.add_argument(
        "--drive-config",
        type=Path,
        help="Google Drive settings path (defaults to the per-user config location)",
    )
    parser.add_argument(
        "--transport",
        choices=("drive", "embedded"),
        default="drive",
        help="transport to prepare; Drive is the small-controller main line",
    )
    parser.add_argument("--execution-config", type=Path, help="managed execution settings; default JOB.run.json")
    parser.add_argument("--run-dir", type=Path, help="durable journal; repeat the same command to reconnect")
    parser.add_argument("--prepare-only", action="store_true", help="prepare managed execution without network or allocation")
    parser.add_argument("--resume-from", type=Path, help="continue a terminal same-job run in a new run directory with a new budget")
    parser.add_argument("--approve", help="optional exact prepared-plan SHA-256")
    parser.add_argument("--preflight-only", action="store_true", help="check prepared job, credentials, assignments and Drive without uploading or allocating")
    parser.add_argument("--allow-active-assignment", action="append", default=[], metavar="ENDPOINT", help="explicitly permit this existing assignment alongside a fresh run; repeat for each endpoint")
    parser.add_argument("job_file", type=Path, metavar="JOB.colajob")
    parser.add_argument(
        "job_args",
        nargs=argparse.REMAINDER,
        metavar="JOB_ARGS",
        help="arguments forwarded unchanged, without shell parsing",
    )
    add_profile_argument(parser)
    return parser


def _controller_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="colabcola controller",
        description=(
            "Generate a tiny controller after independently verifying an uploaded Drive object."
        ),
        epilog="Use colabcola --profile NAME <command> to select an account for one invocation. Set up account access with colabcola login. Run recipes with colabcola run --help. Create jobs with colabcola package --help. Arguments after JOB.colajob are forwarded unchanged to the job entrypoint.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--output", type=Path, help="controller .ipynb output path")
    parser.add_argument("--receipt", type=Path, help="staging receipt .json output path")
    parser.add_argument("--drive-config", type=Path, help="Google Drive settings path")
    parser.add_argument("--drive-file-id", required=True)
    parser.add_argument("--drive-parent-id", required=True)
    parser.add_argument("--drive-relative-path", required=True)
    parser.add_argument("--reused", action="store_true")
    parser.add_argument("job_file", type=Path, metavar="JOB.colajob")
    parser.add_argument("job_args", nargs=argparse.REMAINDER, metavar="JOB_ARGS")
    add_profile_argument(parser)
    return parser


def _google_drive_auth_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="colabcola auth google-drive",
        description=(
            "Authorize per-file Drive access and store only credential-free settings; "
            "user tokens are saved in the operating-system keyring."
        ),
    )
    parser.add_argument("--desktop-client-secrets", required=True, type=Path)
    parser.add_argument("--colab-client-secrets", required=True, type=Path)
    parser.add_argument(
        "--colab-secret-name",
        default=DEFAULT_COLAB_SECRET_NAME,
        help="Colab Secrets key containing the limited-input OAuth client JSON",
    )
    parser.add_argument("--drive-config", type=Path, help="settings output path")
    add_profile_argument(parser)
    return parser


def _validation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="colabcola validate",
        description="Validate a version 1 .colajob without compiling it.",
    )
    parser.add_argument("job_file", type=Path, metavar="JOB.colajob")
    add_profile_argument(parser)
    return parser


def _profile_prefix(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """Move only leading global selectors; never inspect workload arguments."""
    raw = list(argv)
    end = 0
    while end < len(raw):
        if raw[end] == "--profile":
            end = min(end + 2, len(raw))
        elif raw[end].startswith("--profile="):
            end += 1
        else:
            break
    return raw[:end], raw[end:]


def _parse_command(argv: Sequence[str]) -> Command:
    prefix, raw = _profile_prefix(argv)
    if raw[:2] == ["auth", "google-drive"]:
        args = _google_drive_auth_parser().parse_args(prefix + raw[2:])
        return GoogleDriveAuthCommand(
            desktop_client_secrets=args.desktop_client_secrets,
            colab_client_secrets=args.colab_client_secrets,
            colab_secret_name=args.colab_secret_name,
            drive_config=args.drive_config,
            profile=args.profile,
        )
    if raw and raw[0] == "validate":
        args = _validation_parser().parse_args(prefix + raw[1:])
        return ValidateCommand(job_file=args.job_file, profile=args.profile)

    if raw and raw[0] == "controller":
        args = _controller_parser().parse_args(prefix + raw[1:])
        return ControllerCommand(
            job_file=args.job_file,
            job_args=tuple(args.job_args),
            output=args.output,
            receipt=args.receipt,
            drive_file_id=args.drive_file_id,
            drive_parent_id=args.drive_parent_id,
            drive_relative_path=args.drive_relative_path,
            reused=args.reused,
            drive_config=args.drive_config,
            profile=args.profile,
        )

    mode = "submit"
    parser = _submission_parser()
    if raw and raw[0] == "compile":
        mode = "compile"
        raw = raw[1:]
        parser = _submission_parser(prog="colabcola compile")

    args = parser.parse_args(prefix + raw)
    if args.prepare_only and args.preflight_only:
        parser.error("--prepare-only and --preflight-only are mutually exclusive")
    return SubmissionCommand(
        job_file=args.job_file,
        job_args=tuple(args.job_args),
        output=args.output,
        drive_config=args.drive_config,
        mode=mode,
        transport=args.transport,
        execution_config=args.execution_config, run_dir=args.run_dir,
        prepare_only=args.prepare_only, resume_from=args.resume_from, approve=args.approve,
        preflight_only=args.preflight_only, allow_active_assignments=tuple(args.allow_active_assignment),
        profile=args.profile,
    )


def _validation_summary(invocation: JobInvocation) -> dict[str, object]:
    spec = invocation.spec
    return {
        "schema_version": spec.schema_version,
        "name": spec.name,
        "payload_entries": len(spec.payload),
        "outputs": len(spec.outputs),
        "accelerator": spec.resources.accelerator,
        "runtime_provider": spec.runtime.provider,
        "runtime_version": spec.runtime.version,
    }


def main(argv: Sequence[str] | None = None) -> int:
    prefix, arguments = _profile_prefix(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "login":
        from .onboarding import main as login_main
        return login_main(prefix + arguments[1:])
    if arguments and arguments[0] == "_drive-token":
        if prefix:
            _submission_parser().error("--profile is not supported for the internal token helper")
        from .onboarding import token_main
        return token_main(arguments[1:])
    if arguments and arguments[0] == "lifecycle":
        from .lifecycle import main as lifecycle_main
        global_parser = argparse.ArgumentParser(prog="colabcola")
        add_profile_argument(global_parser)
        options = global_parser.parse_args(prefix)
        try:
            with selected(options.profile):
                return lifecycle_main(arguments[1:])
        except LifecycleError as exc:
            print(f"colabcola: {exc}", file=sys.stderr)
            return 2
    if arguments and arguments[0] == "run":
        from .workflow import main as workflow_main
        return workflow_main(prefix + arguments[1:])
    if arguments and arguments[0] == "package":
        from .packager import main as package_main
        return package_main(prefix + arguments[1:])
    command = _parse_command(prefix + arguments)
    try:
        with selected(command.profile):
            return _execute(command)
    except LifecycleError as exc:
        print(f"colabcola: {exc}", file=sys.stderr)
        return 2


def _execute(command: Command) -> int:
    try:
        if isinstance(command, GoogleDriveAuthCommand):
            settings = google_drive_settings_from_clients(
                command.desktop_client_secrets,
                command.colab_client_secrets,
                colab_secret_name=command.colab_secret_name,
            )
            create_google_drive_backend(settings, force_authorize=True)
            settings_path = write_google_drive_settings(settings, command.drive_config)
            print(
                json.dumps(
                    {
                        "colab_oauth_client_id": settings.colab_oauth_client_id,
                        "colab_secret_name": settings.colab_secret_name,
                        "drive_scope": GOOGLE_DRIVE_SCOPE,
                        "google_cloud_project_id": settings.google_cloud_project_id,
                        "next_action": (
                            "Add the complete limited-input client JSON to Colab Secret "
                            + settings.colab_secret_name
                            + " and grant generated controllers access."
                        ),
                        "settings_path": settings_path.as_posix(),
                        "state": "google_drive_configured",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0

        if isinstance(command, ValidateCommand):
            invocation = load_job_invocation(command.job_file)
            print(
                json.dumps(
                    _validation_summary(invocation),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0

        invocation = load_job_invocation(command.job_file, command.job_args)
        if isinstance(command, SubmissionCommand):
            managed_options = (command.execution_config or command.run_dir or command.prepare_only
                               or command.resume_from or command.approve or command.preflight_only or command.allow_active_assignments)
            if invocation.spec.managed_recovery and command.mode == "submit" and command.transport == "drive":
                from .managed import run_managed
                return run_managed(command, invocation)
            if managed_options:
                raise SpecError("managed execution options require a managed_recovery job submitted with Drive")
        if isinstance(command, ControllerCommand):
            settings = load_google_drive_settings(command.drive_config)
            auth = controller_auth_from_settings(settings)
            staged = build_staged_bundle(build_bundle(invocation))
            receipt = drive_receipt_for_observed_upload(
                staged,
                remote_file_id=command.drive_file_id,
                remote_parent_id=command.drive_parent_id,
                remote_relative_path=command.drive_relative_path,
                reused=command.reused,
            )
            output = command.output or command.job_file.with_name(
                command.job_file.stem + "-controller.ipynb"
            )
            if output.resolve(strict=False) == command.job_file.resolve(strict=False):
                raise SpecError("controller output may not overwrite the source .colajob")
            result = compile_drive_controller_invocation(invocation, receipt, auth, output)
            receipt_path = command.receipt or output.with_suffix(".receipt.json")
            write_drive_receipt(receipt, receipt_path)
            response = {
                "controller_bytes": len(result.notebook_bytes),
                "controller_path": result.output_path.as_posix(),
                "controller_sha256": result.notebook_sha256,
                "drive_file_id": receipt.remote_file_id,
                "job_id": result.job_id,
                "receipt_path": receipt_path.resolve(strict=True).as_posix(),
                "state": "awaiting_manual_launch",
            }
        elif command.transport == "embedded":
            output = command.output or command.job_file.with_suffix(".ipynb")
            if output.resolve(strict=False) == command.job_file.resolve(strict=False):
                raise SpecError("notebook output may not overwrite the source .colajob")
            result = compile_invocation(invocation, output)
            response = {
                "job_id": result.job_id,
                "notebook_bytes": len(result.notebook_bytes),
                "notebook_path": result.output_path.as_posix(),
                "notebook_sha256": result.notebook_sha256,
                "state": (
                    "compiled" if command.mode == "compile" else "awaiting_manual_launch"
                ),
                "transport": "embedded",
                "warnings": list(result.warnings),
            }
        else:
            output = command.output or command.job_file.with_suffix(".colabcola")
            if output.resolve(strict=False) == command.job_file.resolve(strict=False):
                raise SpecError("bundle output may not overwrite the source .colajob")
            compilation_started = time.monotonic()
            result = prepare_drive_invocation(invocation, output)
            compilation_seconds = time.monotonic() - compilation_started
            if command.mode == "compile":
                response = {
                    "bundle_bytes": result.staged_bytes,
                    "bundle_path": result.output_path.as_posix(),
                    "bundle_sha256": result.staged_sha256,
                    "compile_seconds": compilation_seconds,
                    "job_id": result.job_id,
                    "state": "awaiting_drive_staging",
                    "transport": "google_drive",
                    "warnings": list(result.warnings),
                }
            else:
                settings = load_google_drive_settings(command.drive_config)
                backend = create_google_drive_backend(settings)
                staging_started = time.monotonic()
                receipt = stage_drive_bundle(result.artifact, result.output_path, backend)
                staging_seconds = time.monotonic() - staging_started
                controller_started = time.monotonic()
                auth = controller_auth_from_settings(settings)
                controller_artifact = build_drive_controller(result.artifact, receipt, auth)
                controller_output = result.output_path.with_name(
                    result.output_path.stem + "-controller.ipynb"
                )
                controller_path = write_drive_controller(
                    controller_artifact, controller_output
                )
                receipt_path = result.output_path.with_name(
                    result.output_path.stem + "-drive-receipt.json"
                )
                write_drive_receipt(receipt, receipt_path)
                controller_seconds = time.monotonic() - controller_started
                response = {
                    "bundle_bytes": result.staged_bytes,
                    "bundle_path": result.output_path.as_posix(),
                    "bundle_sha256": result.staged_sha256,
                    "compile_seconds": compilation_seconds,
                    "controller_bytes": len(controller_artifact.notebook_bytes),
                    "controller_path": controller_path.as_posix(),
                    "controller_sha256": controller_artifact.notebook_sha256,
                    "controller_seconds": controller_seconds,
                    "drive_file_id": receipt.remote_file_id,
                    "drive_staging_seconds": staging_seconds,
                    "job_id": result.job_id,
                    "receipt_path": receipt_path.resolve(strict=True).as_posix(),
                    "reused": receipt.reused,
                    "state": "awaiting_manual_launch",
                    "transport": "google_drive",
                    "warnings": list(result.warnings),
                }
    except (OSError, BundleError, DriveTransportError, SpecError, LifecycleError, ValueError) as exc:
        print(f"colabcola: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            response,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0
