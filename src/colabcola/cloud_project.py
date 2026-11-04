"""Explicit, isolated Google Cloud project setup using the installed gcloud CLI.

This manages a Cloud application project, never Colab runtimes or billing links.
Management credentials belong to gcloud in a separate private configuration.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import requests

from .lifecycle_io import LifecycleError

LABEL = "colabcola-managed"
DRIVE_SERVICE = "drive.googleapis.com"
PERMISSIONS = (
    "resourcemanager.projects.get", "resourcemanager.projects.update",
    "serviceusage.services.get", "serviceusage.services.list", "serviceusage.services.enable",
)
_MAX_OUTPUT = 1024 * 1024
_CODES = ("NOT_FOUND", "PERMISSION_DENIED", "UNAUTHENTICATED", "ALREADY_EXISTS",
          "RESOURCE_EXHAUSTED", "FAILED_PRECONDITION", "UNAVAILABLE")


def _failure_code(error):
    """Classify bounded CLI prose without returning its dynamic/private content."""
    text = error.lower()
    if "terms of service" in text or "terms of use" in text:
        return "TERMS_REQUIRED"
    if any(part in text for part in ("organization policy", "organisation policy", "orgpolicy", "org policy",
                                     "constraints/", "constraints.")):
        return "ORG_POLICY"
    if (re.search(r"quota.{0,80}(exceed|exhaust)", text)
            or re.search(r"(exceed|exhaust).{0,80}quota", text)
            or "project creation limit" in text):
        return "RESOURCE_EXHAUSTED"
    if any(part in text for part in ("invalid_grant", "reauthentication required", "invalid authentication credentials")):
        return "UNAUTHENTICATED"
    if "does not have permission to access projects instance" in text and "or it may not exist" in text:
        # Resource visibility does not distinguish a nonexistent project from a
        # project this account cannot read. Never turn this into NOT_FOUND.
        return "ACCESS_DENIED_OR_NOT_FOUND"
    if re.search(r"project id.{0,100}already (in use|exists)", text) or "requested entity already exists" in text:
        return "ALREADY_EXISTS"
    code = next((code for code in _CODES if re.search(r"\b" + code + r"\b", error)), None)
    if code:
        return code
    if any(part in text for part in ("does not have permission", "permission denied", "not authorized")):
        return "PERMISSION_DENIED"
    return "FAILED"


def _failure_guidance(code):
    return {
        "TERMS_REQUIRED": "Google requires Cloud terms acceptance. Complete that step yourself in the Cloud Console for this account; Colabcola cannot accept terms",
        "ORG_POLICY": "An organization policy blocks this operation. Ask the organization administrator to allow this setup",
        "RESOURCE_EXHAUSTED": "Google reports an exhausted quota or project limit. Resolve that limit before retrying; Colabcola will not delete projects or choose another ID",
        "UNAUTHENTICATED": "The Cloud setup account needs sign-in again; rerun colabcola login",
        "ACCESS_DENIED_OR_NOT_FOUND": "Google could not confirm this project: it may be absent or inaccessible to the selected account. Existing projects require verified access",
        "ALREADY_EXISTS": "The exact project ID is already in use. It can be reused only after its identity, Colabcola label and account permissions are verified",
        "PERMISSION_DENIED": "The selected account is not allowed to perform this operation. Check project creation or access rights and organization policy",
        "UNAVAILABLE": "Google Cloud is temporarily unavailable; retry the same saved project selection",
    }.get(code, "Check the selected account, project access and organization policy")


class CloudProjectError(LifecycleError):
    """Authored, credential-free error suitable for the interactive guide."""
    def __init__(self, message, *, code="FAILED"):
        super().__init__(message)
        self.code = code


def _project_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", value):
        raise CloudProjectError("Cloud project ID must contain 6-30 lowercase letters, digits or hyphens and start with a letter")
    return value


def _account(value):
    if (not isinstance(value, str) or len(value) > 254
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*@[A-Za-z0-9][A-Za-z0-9.-]*\.[A-Za-z]{2,63}", value)
            or value.lower().endswith(".gserviceaccount.com")):
        raise CloudProjectError("Choose a Google user account email for Cloud project setup")
    return value.lower()


def _normalize(value, expected=None):
    if not isinstance(value, dict):
        raise CloudProjectError("Google Cloud returned invalid project metadata")
    project_id = _project_id(value.get("projectId"))
    number = value.get("projectNumber")
    if type(number) is int:
        number = str(number)
    labels = value.get("labels", {})
    state = value.get("lifecycleState", value.get("state"))
    if (not isinstance(number, str) or not re.fullmatch(r"[1-9][0-9]{0,29}", number)
            or not isinstance(labels, dict) or len(labels) > 1000
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in labels.items())
            or not isinstance(state, str) or not state or len(state) > 64
            or (expected is not None and project_id != expected)):
        raise CloudProjectError("Google Cloud returned invalid or mismatched project metadata")
    return {"project_id": project_id, "project_number": number, "labels": dict(labels), "state": state}


class CloudProjectClient:
    def __init__(self, config_root=None):
        from .onboarding import config_home, _no_links
        self.root = Path(config_root).expanduser().absolute() if config_root is not None else config_home()
        self.config = self.root / "cloud-setup" / "gcloud"
        _no_links(self.config)
        self.account = None

    def available(self):
        """Discovery only: never install, execute, authenticate or create files."""
        return shutil.which("gcloud") is not None

    def _environment(self):
        from .onboarding import _no_links
        _no_links(self.config)
        for directory in (self.root, self.config.parent, self.config):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt" and self.config.stat().st_mode & 0o077:
            raise CloudProjectError("Cloud setup configuration must be private (mode 0700)")
        # Ambient credentials, impersonation, endpoints and project overrides must
        # not redirect this explicitly selected setup account or its operations.
        env = {k: v for k, v in os.environ.items() if not k.upper().startswith("CLOUDSDK_")
               and k.upper() not in {"GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"}}
        env.update(CLOUDSDK_CONFIG=str(self.config), CLOUDSDK_CORE_DISABLE_USAGE_REPORTING="true",
                   CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK="true", CLOUDSDK_CORE_LOG_HTTP="false")
        return env

    def _run(self, args, *, stage, interactive=False, timeout=60, json_output=True, require_account=True):
        executable = shutil.which("gcloud")
        if executable is None:
            raise CloudProjectError("Google Cloud CLI is not installed. Install gcloud from https://cloud.google.com/sdk/docs/install, then repeat colabcola login; it will not be downloaded automatically")
        if require_account and self.account is None:
            raise CloudProjectError("Authenticate a Google account for Cloud setup before selecting or changing a project")
        argv = [executable, *args, "--verbosity=error"]
        if self.account is not None:
            argv.append("--account=" + _account(self.account))
        if not interactive:
            argv.append("--quiet")
            if json_output:
                argv.append("--format=json")
        try:
            result = subprocess.run(argv, shell=False, check=False, env=self._environment(),
                                    cwd=str(self.config), timeout=timeout, capture_output=not interactive,
                                    stdin=None if interactive else subprocess.DEVNULL,
                                    text=True, encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            raise CloudProjectError(stage + " timed out; retry the same saved project selection", code="TIMEOUT") from None
        except (OSError, ValueError):
            raise CloudProjectError(stage + " could not start; check the Google Cloud CLI installation") from None
        if result.returncode:
            error = result.stderr if isinstance(result.stderr, str) and len(result.stderr) <= _MAX_OUTPUT else ""
            code = _failure_code(error)
            raise CloudProjectError(stage + " failed (" + code + "). " + _failure_guidance(code), code=code)
        if interactive:
            return None
        output = result.stdout
        if not isinstance(output, str) or len(output.encode("utf-8")) > _MAX_OUTPUT:
            raise CloudProjectError(stage + " returned too much or invalid output")
        if not json_output:
            return output.strip()
        try:
            return json.loads(output)
        except (ValueError, RecursionError):
            raise CloudProjectError(stage + " returned invalid JSON") from None

    def _accounts(self):
        rows = self._run(["auth", "list", "--limit=101"], stage="Cloud account discovery", require_account=False)
        if not isinstance(rows, list) or len(rows) > 100:
            raise CloudProjectError("Cloud account discovery returned invalid or too many accounts")
        result = []
        for row in rows:
            if not isinstance(row, dict) or row.get("status") not in ("ACTIVE", "", None):
                raise CloudProjectError("Cloud account discovery returned invalid account metadata")
            result.append({"account": _account(row.get("account")), "active": row.get("status") == "ACTIVE"})
        return result

    @staticmethod
    def _choose(accounts):
        active = [row["account"] for row in accounts if row["active"]]
        choices = active or [row["account"] for row in accounts]
        if len(choices) > 1:
            raise CloudProjectError("Several Cloud setup accounts are available; choose one account explicitly before continuing")
        return choices[0] if choices else None

    def authenticate(self, account=None):
        """Reuse an isolated account or perform gcloud's interactive user login."""
        requested = _account(account) if account is not None else None
        self.account = requested
        accounts = self._accounts()
        selected = requested or self._choose(accounts)
        known = selected is not None and any(row["account"] == selected for row in accounts)
        self.account = selected
        force = False
        if known:
            try:
                self._token()  # Verify/refresh silently; never expose its output.
                return selected
            except CloudProjectError as exc:
                if exc.code != "UNAUTHENTICATED":
                    raise
                force = True
        args = ["auth", "login"] + ([selected] if selected else [])
        args += ["--no-launch-browser", "--brief", "--no-activate"]
        if force:
            args.append("--force")
        self._run(args, stage="Cloud account sign-in", interactive=True, timeout=300, require_account=False)
        accounts = self._accounts()
        if selected is None:
            selected = self._choose(accounts)
        if selected is None or not any(row["account"] == selected for row in accounts):
            raise CloudProjectError("Cloud sign-in did not authorize the selected account; repeat login with that account")
        self.account = selected
        self._token()
        return selected

    def _token(self):
        token = self._run(["auth", "print-access-token"], stage="Cloud account authorization", json_output=False)
        if not 1 <= len(token) <= 16384 or any(c.isspace() for c in token):
            raise CloudProjectError("Cloud account authorization returned an invalid access token")
        return token

    def projects(self):
        rows = self._run(["projects", "list", "--filter=labels." + LABEL + "=true", "--limit=1001"],
                         stage="Cloud project discovery")
        if not isinstance(rows, list) or len(rows) > 1000:
            raise CloudProjectError("Cloud project discovery returned invalid or too many projects; choose a project ID explicitly")
        result = [_normalize(row) for row in rows]
        # Validate the server filter again; a display name never proves ownership.
        return sorted((row for row in result if row["labels"].get(LABEL) == "true" and row["state"] == "ACTIVE"),
                      key=lambda row: row["project_id"])

    def project(self, project_id):
        project_id = _project_id(project_id)
        try:
            value = self._run(["projects", "describe", project_id], stage="Cloud project inspection")
        except CloudProjectError as exc:
            if exc.code == "NOT_FOUND":
                return None
            raise
        return _normalize(value, project_id)

    def check_permissions(self, project_id):
        project_id = _project_id(project_id)
        token = self._token()
        # gcloud has no projects test-iam-permissions command. This is the
        # documented read-only Resource Manager test, not policy enumeration.
        try:
            with requests.Session() as session:
                # No ambient .netrc credential may override the chosen account.
                session.trust_env = False
                with session.post("https://cloudresourcemanager.googleapis.com/v1/projects/" + project_id + ":testIamPermissions",
                                  headers={"Authorization": "Bearer " + token}, json={"permissions": list(PERMISSIONS)},
                                  timeout=30, allow_redirects=False, stream=True) as response:
                    if response.status_code != 200:
                        raise ValueError()
                    body = bytearray()
                    for chunk in response.iter_content(8192):
                        body.extend(chunk)
                        if len(body) > 16384:
                            raise ValueError()
                    value = json.loads(body)
            granted = value.get("permissions", []) if isinstance(value, dict) else None
            if not isinstance(granted, list) or any(not isinstance(p, str) or p not in PERMISSIONS for p in granted):
                raise ValueError()
        except Exception:
            raise CloudProjectError("Cloud project permissions could not be verified; check account access and retry") from None
        granted = sorted(set(granted))
        return {"required": list(PERMISSIONS), "granted": granted, "missing": sorted(set(PERMISSIONS) - set(granted))}

    def _managed(self, value):
        if value["labels"].get(LABEL) != "true" or value["state"] != "ACTIVE":
            raise CloudProjectError("The saved project is not an active Colabcola-managed project; select it explicitly in the setup guide instead of adopting it by name")
        if self.check_permissions(value["project_id"])["missing"]:
            raise CloudProjectError("The selected account lacks required project setup permissions; keep the saved project ID and resolve access before retrying")
        return value

    def create(self, project_id):
        """Attempt an approved exact-ID create, then verify it through direct reads.

        The guide must persist create intent and approval before calling this.
        Describe cannot be an existence gate: Google may deny reads of absent
        projects as well as inaccessible ones. Create never modifies a collision.
        """
        project_id = _project_id(project_id)
        failure = None
        try:
            self._run(["projects", "create", project_id, "--labels=" + LABEL + "=true", "--no-enable-cloud-apis"],
                      stage="Cloud project creation", timeout=120, json_output=False)
        except CloudProjectError as exc:
            failure = exc
        # Do not use eventually consistent project search to decide whether a
        # timed-out create succeeded, and never invent a replacement project ID.
        try:
            created = self.project(project_id)
        except CloudProjectError:
            created = None
        if created is None:
            reason = (str(failure) + ". ") if failure else ""
            raise CloudProjectError(reason + "Cloud project creation could not be confirmed; retry the same saved project ID",
                                    code=failure.code if failure else "PENDING") from None
        return self._managed(created)

    def billing(self, project_id):
        project_id = _project_id(project_id)
        value = self._run(["billing", "projects", "describe", project_id], stage="Cloud billing inspection")
        if (not isinstance(value, dict) or value.get("projectId") != project_id
                or type(value.get("billingEnabled")) is not bool
                or not isinstance(value.get("billingAccountName"), str)
                or (value["billingAccountName"] and not re.fullmatch(r"billingAccounts/[A-Za-z0-9-]+", value["billingAccountName"]))
                or (value["billingEnabled"] and not value["billingAccountName"])):
            raise CloudProjectError("Cloud project billing status is unknown; verify it before continuing")
        return {"linked": bool(value["billingAccountName"]), "enabled": value["billingEnabled"]}

    def _drive_enabled(self, project_id):
        # gcloud limits API iteration before applying this client-side filter.
        # A small --limit would miss Drive when other enabled services come first.
        rows = self._run(["services", "list", "--enabled", "--project=" + project_id,
                          "--filter=config.name=" + DRIVE_SERVICE], stage="Drive API status inspection")
        if not isinstance(rows, list) or len(rows) > 1:
            raise CloudProjectError("Drive API status could not be verified")
        if not rows:
            return False
        value = rows[0]
        if (not isinstance(value, dict) or not isinstance(value.get("config"), dict)
                or value["config"].get("name") != DRIVE_SERVICE or value.get("state") != "ENABLED"):
            raise CloudProjectError("Drive API status could not be verified")
        return True

    def ensure_drive(self, project_id):
        project_id = _project_id(project_id)
        if self._drive_enabled(project_id):
            return {"enabled": True, "changed": False}
        try:
            self._run(["services", "enable", DRIVE_SERVICE, "--project=" + project_id],
                      stage="Drive API enablement", timeout=120, json_output=False)
        except CloudProjectError:
            # A lost acknowledgement may follow successful enablement.
            if not self._drive_enabled(project_id):
                raise CloudProjectError("Drive API enablement could not be confirmed; retry the same project") from None
        if not self._drive_enabled(project_id):
            raise CloudProjectError("Drive API enablement is pending; retry the same project")
        return {"enabled": True, "changed": True}
