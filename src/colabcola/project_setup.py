"""Resumable personal Cloud project setup; never allocates compute or attaches billing."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
import sys
from urllib.parse import urlencode
import uuid

from .lifecycle_io import LifecycleError, atomic, digest, encoded

SETUP_FILE = "project-setup.json"
CLIENT_FILE = "drive-client.json"
_STAGES = {"selected", "approved", "project_ready", "drive_enabled", "awaiting_client", "ready"}
_CHECKS = {"not_checked", "online_verified", "self_reported"}
_FIELDS = {"schema", "mode", "project_id", "project_number", "stage", "method", "cloud_checks",
           "setup_approved", "create_requested", "billing_unlinked", "drive_enabled", "client_sha256",
           "cloud_account_email"}


class ProjectSetupError(LifecycleError):
    """Only authored, nonsecret guidance crosses the setup boundary."""


def _path_text(path):
    value = str(path).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    if sys.platform.startswith("linux") and re.match(r"^[A-Za-z]:[\\/]", value):
        value = "/mnt/" + value[0].lower() + "/" + value[3:].replace("\\", "/")
    return value


def _safe(path):
    path = Path(os.path.abspath(Path(_path_text(path)).expanduser()))
    if any(p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction()) for p in (path, *path.parents)):
        raise ProjectSetupError("Project setup paths cannot use symbolic links or junctions")
    return path


def _read(path):
    path = _safe(path)
    try:
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 16384:
            raise ValueError()
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, UnicodeError):
        raise ProjectSetupError("Setup configuration must be a valid regular JSON file no larger than 16 KiB") from None


def _project_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", value):
        raise ProjectSetupError("Use an exact Google Cloud project ID: 6-30 lowercase letters, digits or hyphens")
    return value


def _record(root):
    path = _safe(root / SETUP_FILE)
    if not path.exists():
        return None
    value = _read(path)
    if (not isinstance(value, dict) or type(value.get("schema")) is not int or value.get("schema") not in (1, 2)
            or set(value) != (_FIELDS if value["schema"] == 1 else _FIELDS | {"project_mode"})
            or value.get("mode") != "user_project"
            or not isinstance(value.get("stage"), str) or value["stage"] not in _STAGES
            or not isinstance(value.get("method"), str) or value["method"] not in {"gcloud", "manual", "imported"}
            or not isinstance(value.get("cloud_checks"), str) or value["cloud_checks"] not in _CHECKS
            or any(type(value.get(k)) is not bool for k in ("setup_approved", "create_requested", "billing_unlinked", "drive_enabled"))):
        raise ProjectSetupError("Invalid project setup record; preserve this configuration and use a new --config-home")
    _project_id(value["project_id"])
    number, sha, account = value["project_number"], value["client_sha256"], value["cloud_account_email"]
    if (number is not None and (not isinstance(number, str) or not re.fullmatch(r"[0-9]{1,30}", number))
            or sha is not None and (not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha))
            or account is not None and (not isinstance(account, str) or len(account) > 320
                                       or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", account))):
        raise ProjectSetupError("Invalid project setup binding; use a new --config-home")
    if value["stage"] in {"drive_enabled", "awaiting_client", "ready"} and (
            not value["setup_approved"] or not value["billing_unlinked"] or not value["drive_enabled"]
            or value["cloud_checks"] == "not_checked"):
        raise ProjectSetupError("Project setup checks are incomplete; preserve this configuration and use a new --config-home")
    if value["stage"] == "ready" and not sha:
        raise ProjectSetupError("Project setup has no client binding; use a new --config-home")
    if value["schema"] == 1:
        # Earlier selected records discarded whether the user chose "new". Do
        # not recover that authority from an ID prefix or a failed lookup.
        unknown = (value["stage"] == "selected" and value["method"] == "gcloud"
                   and not value["setup_approved"] and not value["create_requested"])
        value = {**value, "schema": 2, "project_mode": "new" if value["create_requested"] else "unknown" if unknown else "existing"}
    project_mode = value.get("project_mode")
    if (not isinstance(project_mode, str) or project_mode not in {"new", "existing", "unknown"}
            or value["create_requested"] and project_mode != "new"
            or value["create_requested"] and not value["setup_approved"]
            or project_mode == "new" and value["method"] == "gcloud" and value["setup_approved"] and not value["create_requested"]
            or project_mode == "unknown" and not (value["stage"] == "selected" and value["method"] == "gcloud"
                                                  and not value["setup_approved"] and not value["create_requested"])):
        raise ProjectSetupError("Invalid project creation intent; preserve this setup and use a new --config-home")
    return value


def _save(root, value, **updates):
    value.update(updates)
    _safe(root).mkdir(parents=True, exist_ok=True, mode=0o700)
    target = _safe(root / SETUP_FILE)
    atomic(target, value)
    os.chmod(target, 0o600)


def _client(path):
    from .drive_oauth import validate_client, DriveOAuthError
    try:
        value = validate_client(_read(path))
    except DriveOAuthError:
        raise ProjectSetupError("Use the downloaded Google Desktop OAuth JSON for this project; other client types or endpoints are not accepted") from None
    _project_id(value["installed"]["project_id"])
    return value, digest(encoded(value))


def status(root):
    """Read setup metadata only: no OAuth client/token files, Cloud calls or writes."""
    value = _record(_safe(root))
    if value is None:
        return {"mode": "not_configured", "project_id": None, "project_mode": None,
                "stage": "not_started", "cloud_checks": "not_checked"}
    return {key: value[key] for key in ("mode", "project_id", "project_mode", "stage", "cloud_checks")}


def _say(message):
    print("colabcola: " + message, file=sys.stderr, flush=True)


def _console_url(path, project, account=None):
    query = {"project": project}
    if account:
        query["authuser"] = account
    return "https://console.cloud.google.com/" + path + "?" + urlencode(query)


def _oauth_instructions(project, account=None):
    _say("Selected Cloud project ID: " + project)
    if account:
        _say("Verified Cloud setup account: " + account)
        _say("Your browser's default Google account may differ. Before changing anything, verify the top-right account is " +
             account + " and the project picker shows the exact project ID " + project + ".")
        _say("The links include a best-effort account hint; they do not guarantee which account Cloud Console selects. "
             "If needed, switch to the setup account, then reopen the exact project link below. Alternatively use a separate/private browser session signed in only to that account.")
    else:
        _say("The Cloud setup account was not independently verified in manual setup. Choose the Google account you intend to use for Colab, "
             "check it in the top-right account menu, and select the exact project ID " + project + " in the project picker.")
        _say("If the browser selected another account, switch to your chosen account and reopen the exact project link; "
             "a separate/private browser session signed in only to that account also avoids a different default account.")
    email = account or "the same Google account you selected above"
    _say("If this project's OAuth app is already configured, keep those settings and continue at the matching step below.")
    _say("1. Open Google Auth Platform: " + _console_url("auth/overview", project, account))
    _say("   If the page shows Get started, click it. In App information, enter App name: Colabcola personal; "
         "User support email: " + email + ". Click Next.")
    _say("   The support-email dropdown follows the signed-in browser account. If the intended email is missing, switch accounts "
         "and reopen the exact project link before continuing.")
    _say("2. On Audience, choose External, then Next. On Contact Information, enter " + email + " and click Next.")
    _say("3. Review Google's Data Policy. If you agree, select its agreement checkbox and choose Continue, then Create. "
         "This creates the initial OAuth configuration; it does not publish the app. Colabcola does not accept terms for you.")
    _say("4. Open Audience: " + _console_url("auth/audience", project, account))
    _say("   Keep publishing status Testing. Under Test users, click Add users, add " + email + ", and Save. "
         "Use that same account for both Colab and Drive sign-in.")
    _say("5. Open Data Access: " + _console_url("auth/scopes", project, account))
    _say("   Click Add or remove scopes. Select exactly these three requested scopes, then Update and Save:")
    _say("   openid")
    _say("   https://www.googleapis.com/auth/userinfo.email")
    _say("   https://www.googleapis.com/auth/drive.file")
    _say("6. Open Clients: " + _console_url("auth/clients", project, account))
    _say("   Click Create client. Application type: Desktop app. Name: Colabcola Desktop. Click Create, then immediately Download JSON "
         "and save it before closing the creation dialog; the client secret is shown only at creation.")
    _say("7. Save the downloaded JSON outside your code/project folder, preferably in Downloads using Google's original filename. "
         "Login can detect the matching Desktop client automatically. An explicit file path is also accepted; do not paste JSON contents. "
         "A quoted Windows C:\\... path is accepted in WSL.")


def ensure_personal_client(root: Path, *, project=None, client_json=None, input_fn=None, auto_client=True):
    """Configure the user's own Drive client; never silently select a publisher client."""
    from .cloud_project import CloudProjectClient, CloudProjectError
    root = _safe(root)
    ask_input = input if input_fn is None else input_fn
    record = None
    def ask(label, default=""):
        answer = ask_input(label + (" [" + default + "]" if default else "") + ": ").strip()
        return answer or default
    try:
        record = _record(root)
        local_path = _safe(root / CLIENT_FILE)
        existing, existing_sha = _client(local_path) if local_path.exists() else (None, None)
        supplied, supplied_sha = _client(client_json) if client_json is not None else (None, None)
        requested = _project_id(project) if project is not None else None
        protected = any(_safe(root / name).exists() for name in ("profile.json", "drive-oauth.json", "colab-oauth.json"))
        if supplied and requested and supplied["installed"]["project_id"] != requested:
            raise ProjectSetupError("Desktop OAuth JSON belongs to another project; choose JSON for the exact --project ID")
        selected = requested or (supplied["installed"]["project_id"] if supplied else None)
        previous_project = record["project_id"] if record else existing["installed"]["project_id"] if existing else None
        if ((selected and previous_project and selected != previous_project and protected)
                or (supplied and existing and supplied_sha != existing_sha and protected)
                or (supplied and record and record["client_sha256"] and supplied_sha != record["client_sha256"] and protected)
                or (not existing and (root / "profile.json").exists() and record is None)):
            raise ProjectSetupError("This changes a configured account's project or OAuth client; preserve it and use a new --config-home")
        if record and selected and selected != record["project_id"]:
            raise ProjectSetupError("A different project is already selected for this setup; resume it or use a new --config-home")
        if existing and record:
            if (existing["installed"]["project_id"] != record["project_id"]
                    or record["client_sha256"] not in (None, existing_sha)):
                raise ProjectSetupError("Local Drive client differs from the recorded setup binding; preserve it and use a new --config-home")
            if record["stage"] == "ready" and (supplied is None or supplied_sha == existing_sha):
                return dict(record)
        if existing and not record:
            if selected and selected != existing["installed"]["project_id"]:
                raise ProjectSetupError("A local Drive client already selects another project; use a new --config-home")
            record = {"schema": 2, "mode": "user_project", "project_mode": "existing", "project_id": existing["installed"]["project_id"],
                      "project_number": None, "stage": "selected", "method": "imported", "cloud_checks": "not_checked",
                      "setup_approved": False, "create_requested": False, "billing_unlinked": False,
                      "drive_enabled": False, "client_sha256": existing_sha, "cloud_account_email": None}
            _save(root, record)
        client = None
        if record is None:
            client = CloudProjectClient(config_root=root)
            method = "gcloud"
            default_project = "new"
            project_mode = "existing"
            while not client.available():
                _say("Google Cloud CLI is not installed. Installation: https://cloud.google.com/sdk/docs/install")
                _say("Windows: winget install --exact --id Google.CloudSDK. Ubuntu/WSL with snap: sudo snap install google-cloud-cli --classic. Reopen your terminal and check gcloud --version.")
                choice = ask("Choose retry after installation, manual Cloud Console setup, or stop", "manual").lower()
                if choice == "manual":
                    method = "manual"; break
                if choice != "retry":
                    raise ProjectSetupError("Setup paused before project changes; install Google Cloud CLI and rerun colabcola login, or choose manual setup")
            account = None
            if method == "gcloud":
                account = client.authenticate()
                if selected is None:
                    projects = client.projects()
                    ids = [_project_id(item["project_id"]) for item in projects[:20]]
                    if ids:
                        _say("Available project IDs: " + ", ".join(ids))
                    managed = [item["project_id"] for item in projects
                               if item.get("state") == "ACTIVE" and item.get("labels", {}).get("colabcola-managed") == "true"]
                    if len(managed) == 1:
                        default_project = _project_id(managed[0])
            if selected is None:
                selected = ask("Project ID to use, or new for a dedicated Colabcola project", default_project)
                if selected == "new":
                    selected = "colabcola-" + uuid.uuid4().hex[:12]
                    project_mode = "new"
            selected = _project_id(selected)
            record = {"schema": 2, "mode": "user_project", "project_mode": project_mode,
                      "project_id": selected, "project_number": None,
                      "stage": "selected", "method": method, "cloud_checks": "not_checked", "setup_approved": False,
                      "create_requested": False, "billing_unlinked": False, "drive_enabled": False,
                      "client_sha256": None, "cloud_account_email": account}
            _save(root, record)  # Persist stable identity before any Cloud mutation.
        selected = record["project_id"]
        if record["project_mode"] == "unknown":
            _say("Earlier setup saved project ID " + selected + " without recording whether to create it or reuse it. The saved ID will stay unchanged.")
            choice = ask("For this exact saved ID, choose create or reuse", "").lower()
            if choice not in {"create", "new", "reuse", "existing"}:
                raise ProjectSetupError("Project intent was not selected; rerun colabcola login and choose create or reuse for the saved ID")
            _save(root, record, project_mode="new" if choice in {"create", "new"} else "existing")
        if record["stage"] not in {"drive_enabled", "awaiting_client", "ready"}:
            if record["method"] in {"manual", "imported"}:
                _say("Manual setup for project " + selected + "; Cloud checks will be recorded as self-reported, not independently verified.")
                _say("Create/select exactly this project if needed: https://console.cloud.google.com/projectcreate")
                _say("Check that NO billing account is linked: https://console.cloud.google.com/billing/linkedaccount?project=" + selected)
                _say("Enable only Google Drive API: https://console.cloud.google.com/apis/library/drive.googleapis.com?project=" + selected)
                if ask("Confirm this exact project exists, has no linked billing account, and Drive API is enabled (yes/no)", "no").lower() != "yes":
                    raise ProjectSetupError("Setup paused: billing or Drive API checks are incomplete. Do not attach billing; use an unbilled project and rerun colabcola login")
                _save(root, record, stage="drive_enabled", setup_approved=True, billing_unlinked=True,
                      drive_enabled=True, cloud_checks="self_reported")
            else:
                if client is None:
                    client = CloudProjectClient(config_root=root)
                    if not client.available():
                        raise ProjectSetupError("Install Google Cloud CLI to resume this selected project, then rerun colabcola login; the project ID is preserved")
                    account = client.authenticate(account=record["cloud_account_email"])
                    _save(root, record, cloud_account_email=account)
                if record["project_mode"] == "new":
                    if not record["setup_approved"]:
                        _say("Create project " + selected + " and enable only Google Drive API. No billing account will be attached and no paid service will be enabled.")
                        if ask("Approve these setup changes (yes/no)", "no").lower() != "yes":
                            raise ProjectSetupError("Project setup was not approved; rerun colabcola login to resume the same selected project")
                        _save(root, record, stage="approved", setup_approved=True, create_requested=True)
                    if not record["create_requested"]:
                        raise ProjectSetupError("Project creation has no saved approval; preserve this setup and use a new --config-home")
                    # An ordinary user cannot reliably describe a nonexistent ID:
                    # gcloud may return ambiguous permission/not-found text. The
                    # approved create itself is exact-ID and reconciled by the adapter.
                    observed = client.create(selected)
                else:
                    observed = client.project(selected)
                    if observed is None:
                        raise ProjectSetupError("The selected existing project was not found; no project was created. Choose an existing project you can access, or use a new --config-home and select new")
                if observed.get("project_id") != selected or observed.get("state") != "ACTIVE":
                    raise ProjectSetupError("Selected project is not confirmed active; resolve project access before retrying")
                number = str(observed["project_number"])
                if not re.fullmatch(r"[0-9]{1,30}", number):
                    raise ProjectSetupError("Google did not confirm a valid project number")
                permissions = client.check_permissions(selected)
                if (permissions.get("missing") or not isinstance(permissions.get("missing"), list)
                        or not isinstance(permissions.get("required"), list) or not permissions["required"]
                        or not isinstance(permissions.get("granted"), list)
                        or not set(permissions["required"]) <= set(permissions["granted"])):
                    raise ProjectSetupError("Your Cloud account lacks required project permissions; ask the project administrator or use another project in a new --config-home")
                billing = client.billing(selected)
                if billing.get("linked") is not False or billing.get("enabled") is not False:
                    raise ProjectSetupError("This project has linked, enabled or unknown billing. Colabcola will not change billing; choose an unbilled project with a new --config-home")
                _save(root, record, stage="project_ready", project_number=number, billing_unlinked=True)
                if not record["setup_approved"]:
                    _say("Enable only Google Drive API in project " + selected + ". No billing changes or paid services.")
                    if ask("Approve these setup changes (yes/no)", "no").lower() != "yes":
                        raise ProjectSetupError("Drive API setup was not approved; rerun colabcola login to resume")
                    _save(root, record, setup_approved=True)
                enabled = client.ensure_drive(selected)
                if enabled.get("enabled") is not True:
                    raise ProjectSetupError("Drive API enablement could not be verified; rerun colabcola login to resume")
                _save(root, record, stage="drive_enabled", drive_enabled=True, cloud_checks="online_verified")
        _save(root, record, stage="awaiting_client")
        _say("Testing authorizations with Drive scope normally expire after seven days. Moving a personal app to Production is your manual choice; Colabcola does not publish the app or accept terms for you.")
        if supplied is None and existing is None:
            _oauth_instructions(selected, record["cloud_account_email"])
            from .client_handoff import choose_client
            from .client_downloads import Candidate
            selection = choose_client(selected, expected_sha256=record["client_sha256"] if protected else None,
                                      automatic=auto_client, input_fn=ask_input,
                                      interactive=input_fn is None and sys.stdin is not None and sys.stdin.isatty(), say=_say)
            if selection is None:
                raise ProjectSetupError("Cloud setup is saved; setup is intentionally paused and no setup progress was lost. "
                                        "After downloading the Desktop OAuth JSON, rerun colabcola login to resume this same setup, "
                                        "or use colabcola login --drive-client PATH. No runtime was started")
            if isinstance(selection, Candidate):
                supplied, supplied_sha = selection.client, selection.sha256
            else:
                supplied, supplied_sha = _client(selection)
        chosen, chosen_sha = (supplied, supplied_sha) if supplied is not None else (existing, existing_sha)
        if chosen["installed"]["project_id"] != selected:
            raise ProjectSetupError("Desktop OAuth JSON belongs to another project; use the exact project shown in the saved setup and retry --drive-client PATH")
        if existing and chosen_sha != existing_sha and protected:
            raise ProjectSetupError("Existing account credentials cannot be rebound to another OAuth client; use a new --config-home")
        if record["client_sha256"] and chosen_sha != record["client_sha256"] and protected:
            raise ProjectSetupError("The imported OAuth client differs from the saved account binding; use a new --config-home")
        if existing_sha != chosen_sha:
            atomic(_safe(root / CLIENT_FILE), chosen)
            os.chmod(root / CLIENT_FILE, 0o600)
        _save(root, record, stage="ready", client_sha256=chosen_sha)
        return dict(record)
    except (EOFError, KeyboardInterrupt):
        stage = record["stage"] if record else "not started"
        raise ProjectSetupError("Project setup paused at " + stage + "; rerun colabcola login to resume. No runtime was started") from None
    except CloudProjectError as exc:
        raise ProjectSetupError("Cloud project setup could not continue: " + str(exc) + ". The selected project is preserved; rerun colabcola login or use a new --config-home") from None
    except ProjectSetupError:
        raise
    except Exception:
        raise ProjectSetupError("Project setup could not be completed safely; keep its saved state and rerun colabcola login. No runtime was started") from None

