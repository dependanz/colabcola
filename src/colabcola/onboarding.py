"""First-use account setup and private token protocol for managed jobs.

Public profile/helper files contain no credentials. Only explicit login may ask
for consent or create a Drive folder; package/status merely read configuration.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import stat
import sys

from .console import ConsoleUI, emit_json
from .lifecycle_io import LifecycleError, atomic
from . import profiles

class OnboardingError(LifecycleError):
    """Only authored, credential-free diagnostics may be displayed publicly."""

PROFILE_FILE = "profile.json"
HELPER_FILE = "drive-helper.json"
_FOLDER_MIME = "application/vnd.google-apps.folder"
_FOLDER_PROPERTIES = {"colabcola.role": "workspace", "colabcola.schema": "1"}
_FOLDER_FIELDS = "id,name,mimeType,ownedByMe,trashed,appProperties,capabilities(canAddChildren)"
_GOOGLE_ERROR_LIMIT = 65536
_GOOGLE_ERROR_REASONS = frozenset({
    "accessNotConfigured", "authError", "insufficientPermissions", "insufficientFilePermissions",
    "appNotAuthorizedToFile", "domainPolicy", "rateLimitExceeded", "userRateLimitExceeded",
    "dailyLimitExceeded", "storageQuotaExceeded", "forbidden", "notFound",
    "SERVICE_DISABLED", "ACCESS_TOKEN_SCOPE_INSUFFICIENT", "RATE_LIMIT_EXCEEDED", "QUOTA_EXCEEDED",
})
_GOOGLE_ERROR_DOMAINS = frozenset({"googleapis.com", "usageLimits", "global"})
_GOOGLE_ERROR_SERVICES = frozenset({"drive.googleapis.com", "colaboratory.googleapis.com"})


def _google_error_details(exc):
    """Read bounded machine fields only; HttpError's string and reason contain private text."""
    status = None
    reasons, domains, services = set(), set(), set()
    try:
        candidate = getattr(getattr(exc, "resp", None), "status", None)
        if type(candidate) is int and 100 <= candidate <= 599:
            status = candidate
        content = getattr(exc, "content", None)
        if not isinstance(content, bytes) or len(content) > _GOOGLE_ERROR_LIMIT:
            return status, reasons, domains, services
        payload = json.loads(content)
        error = payload.get("error") if isinstance(payload, dict) else None
        if not isinstance(error, dict):
            return status, reasons, domains, services
        entries = error.get("errors", [])
        if isinstance(entries, list):
            for item in entries:
                if not isinstance(item, dict):
                    continue
                reason, domain = item.get("reason"), item.get("domain")
                if isinstance(reason, str) and reason in _GOOGLE_ERROR_REASONS:
                    reasons.add(reason)
                if isinstance(domain, str) and domain in _GOOGLE_ERROR_DOMAINS:
                    domains.add(domain)
        entries = error.get("details", [])
        if isinstance(entries, list):
            for item in entries:
                if not isinstance(item, dict) or item.get("@type") != "type.googleapis.com/google.rpc.ErrorInfo":
                    continue
                reason, domain = item.get("reason"), item.get("domain")
                if isinstance(reason, str) and reason in _GOOGLE_ERROR_REASONS:
                    reasons.add(reason)
                if isinstance(domain, str) and domain in _GOOGLE_ERROR_DOMAINS:
                    domains.add(domain)
                metadata = item.get("metadata")
                service = metadata.get("service") if isinstance(metadata, dict) else None
                if isinstance(service, str) and service in _GOOGLE_ERROR_SERVICES:
                    services.add(service)
    except (ValueError, UnicodeError, TypeError, RecursionError, AttributeError):
        pass
    return status, reasons, domains, services


def _setup_failure(stage, exc):
    status, reasons, domains, services = _google_error_details(exc)
    details = ["setup failed at " + stage]
    if status is not None:
        details.append("HTTP " + str(status))
    for label, values in (("reason", reasons), ("domain", domains if reasons else set()),
                          ("service", services if reasons else set())):
        if values:
            details.append(label + "=" + ",".join(sorted(values)))
    message = "; ".join(details)
    if reasons & {"SERVICE_DISABLED", "accessNotConfigured"}:
        api = "Drive" if stage.startswith("Drive ") else "requested"
        message += (". The " + api + " API is disabled for the OAuth application's Google Cloud project. "
                    "Enable that API in the project belonging to the selected Desktop OAuth client; "
                    "colabcola login guides project setup, but repeating Google consent alone will not enable the API")
    elif reasons & {"ACCESS_TOKEN_SCOPE_INSUFFICIENT", "insufficientPermissions", "authError"} or status == 401:
        access = "per-file Google Drive" if stage.startswith("Drive ") else "Google Colab"
        message += ". Run colabcola login --force and grant " + access + " access"
    elif "domainPolicy" in reasons:
        message += ". The organization's policy blocks this operation; ask its administrator to approve access"
    elif "storageQuotaExceeded" in reasons:
        message += ". Drive storage quota is exhausted; free storage or increase the account's quota before retrying"
    elif stage == "Drive folder verification" and (status == 404 or "notFound" in reasons):
        message += (". The saved or selected folder is unavailable to this Drive application. "
                    "For an older profile, run colabcola login --config-home NEW_DIRECTORY to create a fresh workspace, "
                    "or select a folder already authorized to this application with --drive-parent. The saved profile was not changed")
    elif reasons & {"rateLimitExceeded", "userRateLimitExceeded", "dailyLimitExceeded", "RATE_LIMIT_EXCEEDED", "QUOTA_EXCEEDED"} or status == 429:
        message += ". Google reports a rate or quota limit; wait for it to reset before retrying; signing in again will not reset it"
    return message


@contextlib.contextmanager
def _setup_stage(stage):
    try:
        yield
    except OnboardingError:
        raise
    except Exception as exc:
        raise OnboardingError(_setup_failure(stage, exc)) from None


@contextlib.contextmanager
def _drive_setup_stage(stage):
    from . import drive_oauth
    with _setup_stage(stage):
        try:
            yield
        except drive_oauth.DriveOAuthError as exc:
            # This type contains only messages authored by our Drive OAuth module.
            raise OnboardingError("setup failed at " + stage + "; " + str(exc)) from None


def _no_links(path):
    for parent in (path, *path.parents):
        if parent.is_symlink() or (hasattr(parent, "is_junction") and parent.is_junction()):
            raise OnboardingError("Colabcola configuration cannot use links or junctions")


def config_home():
    """Resolve the selected profile without reading credentials or creating files."""
    try:
        return profiles.config_home()
    except profiles.ProfileError as exc:
        raise OnboardingError(str(exc)) from None


@contextlib.contextmanager
def _selected_home(value, profile=None):
    try:
        with profiles.selected(profile=profile, config_home=value) as root:
            yield root
    except profiles.ProfileError as exc:
        raise OnboardingError(str(exc)) from None


def _read_config(name, required=True):
    root = config_home()
    path = root / name
    _no_links(path)
    try:
        metadata = path.stat()
    except FileNotFoundError:
        if not required:
            return None
        raise OnboardingError("Colabcola is not configured; run colabcola login") from None
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 16384:
        raise OnboardingError("Invalid Colabcola configuration; run colabcola login")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        raise OnboardingError("Invalid Colabcola configuration; run colabcola login") from None


def load_profile(required=False):
    """Read only the credential-free account defaults."""
    value = _read_config(PROFILE_FILE, required=required)
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != {"schema", "drive_parent"}
            or type(value["schema"]) is not int or value["schema"] != 1
            or not isinstance(value["drive_parent"], str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,240}", value["drive_parent"])):
        raise OnboardingError("Invalid Colabcola profile; run colabcola login")
    return value


def _helper_argv():
    return [str(Path(sys.executable).absolute()), "-m", "colabcola", "_drive-token",
            "--config-home", str(config_home())]


def execution_defaults():
    """Resolve host settings for running; does not access OAuth or the network."""
    profile = load_profile(required=True)
    if _read_config(HELPER_FILE) != _helper_argv():
        raise OnboardingError("Colabcola login helper changed or Python moved; run colabcola login")
    return {"drive_parent": profile["drive_parent"], "token_command": config_home() / HELPER_FILE}


def _check_host():
    if not (sys.platform.startswith("linux") or sys.platform == "win32"):
        raise OnboardingError("Managed execution currently requires Linux/Ubuntu WSL or Windows. Native macOS support is not yet validated")


def _required_scopes():
    from . import lifecycle_provider as provider
    return list(provider.SCOPES)


def _expiry(credential, service="Colab"):
    value = credential.expiry
    if not isinstance(value, datetime):
        raise OnboardingError(service + " authorization has no usable expiry; run colabcola login --force")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).timestamp()


def _validate_credential(credential):
    scopes = _required_scopes()
    granted = getattr(credential, "granted_scopes", None)
    if isinstance(granted, str):
        granted = granted.split()
    if (not credential.valid or not isinstance(credential.token, str) or not credential.token
            or not credential.refresh_token or not credential.has_scopes(scopes)
            or (granted is not None and not set(scopes) <= set(granted))):
        raise OnboardingError("Google must grant Colab access with renewal; run colabcola login --force")
    _expiry(credential)
    return credential


def _renew(credential, minimum_seconds):
    _validate_credential(credential)
    if _expiry(credential) <= datetime.now(timezone.utc).timestamp() + minimum_seconds:
        from google.auth.transport.requests import Request
        from . import lifecycle_provider as provider
        credential.refresh(Request())
        _validate_credential(credential)
        if _expiry(credential) <= datetime.now(timezone.utc).timestamp() + minimum_seconds:
            raise OnboardingError("Google returned insufficient Colab access lifetime; run colabcola login --force")
        path = provider.auth_path()
        _no_links(path)
        atomic(path, json.loads(credential.to_json()))
        os.chmod(path, 0o600)
    return credential


def _login_credentials(force):
    from . import lifecycle_provider as provider
    _no_links(provider.auth_path())
    credential = None
    if not force:
        try:
            credential = _renew(provider.credentials(), 600)
        except Exception:
            credential = None
    if credential is None:
        ConsoleUI().say("Login: authorize Google Colab access; Drive authorization follows separately. Credentials stay on this computer.")
        _no_links(provider.auth_path())
        try:
            provider.login(scopes=_required_scopes())
        except provider.AuthorizationScopeError:
            raise OnboardingError("Google did not grant all requested permissions; run colabcola login --force and grant Colab access. Existing Colab authorization was not replaced") from None
        _no_links(provider.auth_path())
        credential = _renew(provider.credentials(), 600)
    return credential


def _login_drive_credentials(force, email, subject):
    from . import drive_oauth
    _no_links(config_home() / "drive-oauth.json")
    if not force:
        try:
            return drive_oauth.credentials(minimum_seconds=600)
        except drive_oauth.DriveOAuthError:
            pass
    ConsoleUI().say("Login: authorize per-file Google Drive access using the same Google account as Colab.")
    _no_links(config_home() / "drive-oauth.json")
    return drive_oauth.login(login_hint=email, expected_subject=subject)


def _drive_files(credential):
    from googleapiclient.discovery import build
    return build("drive", "v3", credentials=credential, cache_discovery=False).files()


def _folder(value, expected=None, *, marked=False):
    if (not isinstance(value, dict) or not isinstance(value.get("id"), str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,240}", value["id"])
            or (expected is not None and value["id"] != expected)
            or value.get("mimeType") != _FOLDER_MIME
            or value.get("ownedByMe") is not True or value.get("trashed") is not False
            or value.get("capabilities", {}).get("canAddChildren") is not True
            or (marked and any(value.get("appProperties", {}).get(k) != v for k, v in _FOLDER_PROPERTIES.items()))):
        raise OnboardingError("The saved Drive destination must be an owned, writable, untrashed Colabcola folder")
    return value["id"]


def _ensure_folder(files, profile, explicit_parent=None):
    if explicit_parent or profile:
        parent = explicit_parent or profile["drive_parent"]
        with _setup_stage("Drive folder verification"):
            value = files.get(fileId=parent, fields=_FOLDER_FIELDS).execute(num_retries=2)
        return _folder(value, parent)
    query = ("'root' in parents and trashed = false and mimeType = '" + _FOLDER_MIME + "' and "
             "appProperties has { key='colabcola.role' and value='workspace' } and "
             "appProperties has { key='colabcola.schema' and value='1' }")
    matches = []
    token = None
    seen = set()
    for _ in range(100):
        with _setup_stage("Drive folder lookup"):
            response = files.list(q=query, spaces="drive", fields="nextPageToken,files(" + _FOLDER_FIELDS + ")",
                                  pageSize=100, pageToken=token).execute(num_retries=2)
        if not isinstance(response, dict) or not isinstance(response.get("files", []), list):
            raise OnboardingError("Drive returned invalid folder metadata")
        matches.extend(response.get("files", []))
        if len(matches) > 1:
            raise OnboardingError("Multiple app-owned Colabcola folders exist; run colabcola login --drive-parent FOLDER_ID")
        token = response.get("nextPageToken")
        if not token:
            break
        if not isinstance(token, str) or token in seen:
            raise OnboardingError("Drive returned invalid folder pagination")
        seen.add(token)
    else:
        raise OnboardingError("Drive folder lookup exceeded its page limit")
    if matches:
        return _folder(matches[0], marked=True)
    with _setup_stage("Drive folder creation"):
        created = files.create(body={"name": "colabcola", "mimeType": _FOLDER_MIME,
                                     "parents": ["root"], "appProperties": _FOLDER_PROPERTIES},
                               fields="id").execute(num_retries=0)
    if not isinstance(created, dict) or not isinstance(created.get("id"), str):
        raise OnboardingError("Drive did not confirm folder creation; rerun colabcola login")
    with _setup_stage("Drive folder verification"):
        value = files.get(fileId=created["id"], fields=_FOLDER_FIELDS).execute(num_retries=2)
    return _folder(value, created["id"])


def _verify_colab():
    from . import lifecycle_provider as provider
    _no_links(provider.auth_path())
    client = provider.GoogleProvider()
    try:
        assignments = client.assignments()
        return len(assignments)
    finally:
        client.close()


def _login_client_configuration(root, profile, project=None, client_json=None, *, auto_client=True):
    """Fresh accounts use personal projects; preserve explicit existing account setups."""
    from . import drive_oauth, project_setup
    local = root / drive_oauth.CLIENT_FILE
    _no_links(local)
    _no_links(root / "project-setup.json")
    if (profile is not None or (root / PROFILE_FILE).is_file()) and project is None and client_json is None and not (root / "project-setup.json").exists():
        if local.exists():
            ConsoleUI().say("Login: reusing the existing local Drive OAuth client. Cloud project billing was not checked by this login.")
            client = drive_oauth.preflight_client()["installed"]
            return {"mode": "existing_user_client", "project_id": client["project_id"], "cloud_checks": "not_checked"}
        ConsoleUI().say("Login: reusing the existing shared-client profile (legacy). New installations use a personal Google Cloud project. "
                        "Use --config-home NEW_DIRECTORY to set up a separate personal profile.")
        return {"mode": "legacy_shared_client", "cloud_checks": "not_checked"}
    try:
        return project_setup.ensure_personal_client(root, project=project, client_json=client_json,
                                                    use_bundled_client=(auto_client and client_json is None and profile is None),
                                                    **({"auto_client": False} if not auto_client else {}))
    except project_setup.ProjectSetupError as exc:
        raise OnboardingError(str(exc)) from None




def main(argv=None):
    parser = argparse.ArgumentParser(prog="colabcola login", description="Set up your Google Cloud project, guide Desktop OAuth creation, and authorize Colab and Drive.")
    selection = parser.add_mutually_exclusive_group()
    profiles.add_profile_argument(selection)
    selection.add_argument("--config-home", type=Path, help="explicit configuration directory; preserves legacy logins")
    parser.add_argument("--drive-parent", help="explicit owned Drive folder ID; replaces the saved destination only after verification")
    parser.add_argument("--project", help="reuse this Google Cloud project instead of choosing interactively")
    parser.add_argument("--drive-client", type=Path, help="import the Desktop OAuth JSON downloaded from your project's Console")
    parser.add_argument("--no-auto-client", action="store_true", help="use manual OAuth client selection instead of the bundled default or Downloads detection")
    parser.add_argument("--force", action="store_true", help="request Google consent again; does not replace an existing project or client")
    parser.add_argument("--status", action="store_true", help="inspect configuration only; no credential reads or network")
    args = parser.parse_args(argv)
    if args.status and (args.force or args.drive_parent or args.project or args.drive_client or args.no_auto_client):
        parser.error("--status cannot be combined with --force, --drive-parent, --project, --drive-client or --no-auto-client")
    try:
        with _selected_home(args.config_home, args.profile) as root:
            if args.status:
                profile = load_profile()
                if profile is not None:
                    execution_defaults()
                from . import project_setup
                try:
                    setup_status = project_setup.status(root)
                except project_setup.ProjectSetupError as exc:
                    raise OnboardingError(str(exc)) from None
                if profile and setup_status["mode"] == "not_configured":
                    setup_status["mode"] = "existing_configuration"
                emit_json({"state": "configured" if profile else "not_configured", "config_home": str(root),
                                  "profile": profiles.profile_name(root),
                                  "drive_parent": profile["drive_parent"] if profile else None,
                                  "project_setup": setup_status,
                                  "credentials_checked": False, "network_accessed": False})
                return 0
            _check_host()
            if args.drive_parent is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,240}", args.drive_parent):
                raise OnboardingError("--drive-parent must be an exact Drive folder ID")
            from . import drive_oauth, lifecycle_provider as provider
            with _setup_stage("controller dependency check"):
                provider.dependency_report()
            _no_links(root / PROFILE_FILE)
            try:
                profile = load_profile()
            except OnboardingError:
                if not args.force:
                    raise OnboardingError("Invalid Colabcola profile; run colabcola login --force to rebuild it after verified setup") from None
                if (root / PROFILE_FILE).exists() and not (root / PROFILE_FILE).is_file():
                    raise OnboardingError("Colabcola profile must be a regular file; repair the configuration path before login") from None
                profile = None
            with _drive_setup_stage("Drive OAuth client configuration"):
                setup = _login_client_configuration(root, profile, args.project, args.drive_client,
                                                    **({"auto_client": False} if args.no_auto_client else {}))
                drive_oauth.preflight_client()
            with _setup_stage("Colab authorization"):
                credential = _login_credentials(args.force)
            with _setup_stage("Colab access verification"):
                assignments = _verify_colab()
            with _drive_setup_stage("Colab account verification"):
                colab_account = drive_oauth.identity(credential)
            management_account = setup.get("cloud_account_email")
            if management_account and management_account.casefold() != colab_account["email"].casefold():
                raise OnboardingError("Colab sign-in differs from the account used for Google Cloud project setup. "
                                      "Run colabcola login --force and choose the setup account; no Drive workspace was created")
            with _drive_setup_stage("Drive authorization"):
                drive_credential = _login_drive_credentials(args.force, colab_account["email"], colab_account["sub"])
            with _drive_setup_stage("Drive account verification"):
                drive_account = drive_oauth.identity(drive_credential)
            if colab_account["sub"] != drive_account["sub"]:
                raise OnboardingError("Colab and Drive authorizations belong to different Google accounts; run colabcola login --force and select the same account for both. The saved profile and Drive destination were not changed")
            with _setup_stage("Drive client initialization"):
                files = _drive_files(drive_credential)
            parent = _ensure_folder(files, profile, args.drive_parent)
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            _no_links(root / HELPER_FILE)
            _no_links(root / PROFILE_FILE)
            atomic(root / HELPER_FILE, _helper_argv())
            atomic(root / PROFILE_FILE, {"schema": 1, "drive_parent": parent})
            emit_json({"state": "configured", "config_home": str(root), "profile": profiles.profile_name(root), "drive_parent": parent,
                              "colab_access_verified": True, "active_assignments": assignments,
                              "account_email": colab_account["email"], "same_google_account_verified": True,
                              "runtime_allocated": False,
                              "project_setup": {key: setup[key] for key in ("mode", "project_id", "project_number", "cloud_checks", "stage") if key in setup},
                              "next_command": "colabcola package -- python YOUR_SCRIPT.py"})
            return 0
    except KeyboardInterrupt:
        ConsoleUI().error("interrupted; rerun login to finish setup", command="login")
        return 130
    except OnboardingError as exc:
        ConsoleUI().error(exc, command="login")
        return 2
    except Exception as exc:
        ConsoleUI().error(_setup_failure("local configuration", exc), command="login")
        return 2


def token_main(argv=None):
    """Internal short-lived token protocol; never invoke for public status."""
    parser = argparse.ArgumentParser(prog="colabcola _drive-token")
    parser.add_argument("--config-home", type=Path)
    args = parser.parse_args(argv)
    try:
        with _selected_home(args.config_home):
            execution_defaults()
            from . import drive_oauth
            _no_links(config_home() / "drive-oauth.json")
            with _drive_setup_stage("Drive token renewal"):
                credential = drive_oauth.credentials(minimum_seconds=1860)
            print(json.dumps({"access_token": credential.token, "expires_at": _expiry(credential, "Drive")}))
            return 0
    except OnboardingError as exc:
        ConsoleUI().error(exc)
        return 2
    except Exception as exc:
        ConsoleUI().error(_setup_failure("Drive token renewal", exc))
        return 2




