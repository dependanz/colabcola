"""Publisher-owned Desktop OAuth for Drive; independent of Colab authorization.

The desktop client identifies the application and is distributable. User tokens
are private, per-user files, never package resources or workload inputs.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.resources
import json
import logging
import os
from pathlib import Path
import re
import stat

from .google_drive import GOOGLE_DRIVE_SCOPE
from .lifecycle_io import LifecycleError, atomic

DRIVE_SCOPES = ["openid", "https://www.googleapis.com/auth/userinfo.email", GOOGLE_DRIVE_SCOPE]
CLIENT_FILE = "drive-client.json"
TOKEN_FILE = "drive-oauth.json"
BUNDLED_CLIENT = "publisher_oauth.json"
_TOKEN_URI = "https://oauth2.googleapis.com/token"
_AUTH_URIS = {"https://accounts.google.com/o/oauth2/auth", "https://accounts.google.com/o/oauth2/v2/auth"}
_CLIENT_KEYS = {"client_id", "client_secret", "project_id", "auth_uri", "token_uri", "redirect_uris", "auth_provider_x509_cert_url"}


class DriveOAuthError(LifecycleError):
    """Only authored credential-free errors cross this module's public boundary."""


def auth_path():
    from .onboarding import config_home
    return config_home() / TOKEN_FILE


def _file(path, *, private=False):
    from .onboarding import _no_links
    _no_links(path)
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 16384:
        raise DriveOAuthError("Invalid Drive authorization configuration; run colabcola login --force")
    if private and os.name != "nt" and metadata.st_mode & 0o077:
        raise DriveOAuthError("Drive credential file must have mode 0600")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_client(value):
    """Accept only a Google Desktop client and fixed Google OAuth endpoints."""
    if not isinstance(value, dict) or set(value) != {"installed"}:
        raise DriveOAuthError("Publisher OAuth configuration must be a Google Desktop app client")
    client = value["installed"]
    required = {"client_id", "client_secret", "project_id", "auth_uri", "token_uri"}
    if (not isinstance(client, dict) or not required <= set(client) or not set(client) <= _CLIENT_KEYS
            or any(not isinstance(client[k], str) or not client[k] or len(client[k]) > 1024 for k in required)
            or not re.fullmatch(r"[A-Za-z0-9._-]+\.apps\.googleusercontent\.com", client["client_id"])
            or not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", client["project_id"])
            or client["auth_uri"] not in _AUTH_URIS or client["token_uri"] != _TOKEN_URI):
        raise DriveOAuthError("Invalid publisher Desktop OAuth client; the publisher must repair the application configuration")
    # Redirect URIs in downloads are informational for Desktop clients. Always
    # construct our own loopback redirect; never trust an arbitrary supplied URL.
    return {"installed": {**{k: client[k] for k in required}, "redirect_uris": ["http://localhost"]}}


def preflight_client():
    """Validate application identity before any account consent or network call."""
    from .onboarding import config_home, _no_links
    path = config_home() / CLIENT_FILE
    _no_links(path)
    try:
        if path.exists():
            return validate_client(_file(path))
        if (config_home() / "project-setup.json").exists():
            raise DriveOAuthError("The personal project's Desktop OAuth client is missing. Run colabcola login to resume setup; shared publisher credentials will not be substituted")
        resource = importlib.resources.files("colabcola").joinpath(BUNDLED_CLIENT)
        if not resource.is_file():
            raise DriveOAuthError("This legacy profile has no bundled publisher Drive OAuth client. Run colabcola login --config-home NEW_DIRECTORY to set up your own Google Cloud project")
        body = resource.read_bytes()
        if len(body) > 16384:
            raise ValueError()
        return validate_client(json.loads(body))
    except DriveOAuthError:
        raise
    except Exception:
        raise DriveOAuthError("Publisher Drive OAuth configuration could not be read; reinstall a configured Colabcola build") from None


@contextmanager
def _private_exchange():
    # The upstream callback logger logs its GET query (authorization code), and
    # OAuth debug loggers can log tokens. Never put those into the CLI log stream.
    previous = logging.root.manager.disable
    logging.disable(max(previous, logging.CRITICAL))
    try:
        yield
    finally:
        logging.disable(previous)


def _expiry(credential):
    value = credential.expiry
    if not isinstance(value, datetime):
        raise DriveOAuthError("Drive authorization has no usable expiry; run colabcola login --force")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def _validate(credential, client):
    granted = credential.granted_scopes
    if isinstance(granted, str):
        granted = granted.split()
    if (not isinstance(credential.token, str) or not credential.token or not credential.refresh_token
            or not credential.has_scopes(DRIVE_SCOPES)
            or (granted is not None and not set(DRIVE_SCOPES) <= set(granted))):
        raise DriveOAuthError("Google must grant identity and per-file Drive access with renewal; run colabcola login --force")
    if (credential.client_id != client["client_id"] or credential.client_secret != client["client_secret"]
            or credential.token_uri != _TOKEN_URI):
        raise DriveOAuthError("Saved Drive authorization belongs to a different OAuth client; run colabcola login --force")
    _expiry(credential)
    return credential


def _save(credential):
    from .onboarding import _no_links
    path = auth_path()
    _no_links(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic(path, json.loads(credential.to_json()))
    os.chmod(path, 0o600)


def credentials(minimum_seconds=600, force_refresh=False):
    """Load/renew only publisher Drive credentials; never prompt or read Colab auth."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    client = preflight_client()["installed"]
    try:
        credential = Credentials.from_authorized_user_info(_file(auth_path(), private=True))
        _validate(credential, client)
        if force_refresh or not credential.valid or _expiry(credential) <= datetime.now(timezone.utc).timestamp() + minimum_seconds:
            with _private_exchange():
                credential.refresh(Request())
            _validate(credential, client)
            if not credential.valid or _expiry(credential) <= datetime.now(timezone.utc).timestamp() + minimum_seconds:
                raise DriveOAuthError("Google returned insufficient Drive access lifetime; run colabcola login --force")
            _save(credential)
        return credential
    except DriveOAuthError:
        raise
    except Exception:
        raise DriveOAuthError("Drive authorization is unavailable or expired; run colabcola login") from None


def login(login_hint=None, timeout_seconds=300, expected_subject=None):
    """Interactive Desktop consent with loopback callback, OAuth state and PKCE."""
    from google_auth_oauthlib.flow import InstalledAppFlow
    from .onboarding import _no_links
    config = preflight_client()
    _no_links(auth_path())
    try:
        flow = InstalledAppFlow.from_client_config(config, DRIVE_SCOPES, autogenerate_code_verifier=True)
        kwargs = {"prompt": "consent", "access_type": "offline"}
        if login_hint:
            kwargs["login_hint"] = login_hint
        with _private_exchange():
            credential = flow.run_local_server(
                host="127.0.0.1", bind_addr="127.0.0.1", port=0,
                open_browser=False, timeout_seconds=timeout_seconds,
                authorization_prompt_message="Open this URL in your browser to authorize Colabcola Drive access:\n{url}",
                success_message="Drive authorization received. Return to the terminal for verification. You may close this window.",
                **kwargs)
        _validate(credential, config["installed"])
        if not credential.valid or _expiry(credential) <= datetime.now(timezone.utc).timestamp() + 600:
            raise DriveOAuthError("Google returned insufficient Drive access lifetime; repeat login")
        if expected_subject is not None and identity(credential)["sub"] != expected_subject:
            raise DriveOAuthError("Colab and Drive sign-ins belong to different Google accounts. Run colabcola login again and choose the same account; existing Drive authorization was not replaced")
        # Validate granted scopes before to_json drops that information.
        _save(credential)
        return credential
    except DriveOAuthError:
        raise
    except Exception as exc:
        if type(exc).__name__ in {"WSGITimeoutError", "TimeoutError"}:
            raise DriveOAuthError("Drive sign-in timed out. Run colabcola login again and open its link in this computer's browser") from None
        raise DriveOAuthError("Drive browser authorization did not complete. Confirm that the selected account is permitted by the OAuth app and your organization and accepts the requested permissions. If your own app is in Testing, add your account as a test user, then run colabcola login again") from None


def identity(credential):
    """Fetch a Google-verified account identity for either authorization."""
    from google.auth.transport.requests import AuthorizedSession
    try:
        with _private_exchange(), AuthorizedSession(credential) as session:
            response = session.get("https://openidconnect.googleapis.com/v1/userinfo", timeout=30, allow_redirects=False)
            if response.status_code != 200 or len(response.content) > 16384:
                raise ValueError()
            value = response.json()
        if (not isinstance(value, dict) or not isinstance(value.get("sub"), str)
                or not 1 <= len(value["sub"]) <= 255
                or not isinstance(value.get("email"), str) or len(value["email"]) > 320
                or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value["email"])
                or value.get("email_verified") is not True):
            raise ValueError()
        return {"sub": value["sub"], "email": value["email"]}
    except Exception:
        raise DriveOAuthError("Google account identity could not be verified; run colabcola login again") from None
