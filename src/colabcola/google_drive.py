"""Production Google Drive OAuth settings and API adapter.

The deterministic transport lifecycle lives in :mod:`colabcola.drive`.  This
module supplies the optional Google client-library integration used by the
normal CLI.  User refresh tokens stay in the operating-system keyring; only
public client identity and a path to the user's desktop OAuth client file are
persisted in colabcola settings.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .drive import DRIVE_BINARY_MIME_TYPE, DriveFile, DriveFolder
from .errors import DriveTransportError


GOOGLE_DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.file"
GOOGLE_DRIVE_SETTINGS_SCHEMA_VERSION = 1
DEFAULT_COLAB_SECRET_NAME = "COLABCOLA_GOOGLE_DRIVE_OAUTH"
GOOGLE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"

_CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9._-]+\.apps\.googleusercontent\.com$")
_PROJECT_ID_RE = re.compile(r"^[a-z][a-z0-9-]{4,61}[a-z0-9]$")
_SECRET_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
_REPARSE_FLAG = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_SETTINGS_KEYS = frozenset(
    {
        "colab_oauth_client_id",
        "colab_secret_name",
        "desktop_client_secrets",
        "google_cloud_project_id",
        "schema_version",
    }
)
_OAUTH_CLIENT_REQUIRED_KEYS = frozenset(
    {"auth_uri", "client_id", "client_secret", "project_id", "token_uri"}
)
_DRIVE_FILE_FIELDS = "id,name,parents,size,mimeType,appProperties,trashed"
_UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024
_KEYRING_SERVICE = "colabcola.google-drive"


@dataclass(frozen=True, slots=True)
class GoogleOAuthClient:
    """Validated public identity and private secret from one Google client file."""

    project_id: str
    client_id: str
    client_secret: str
    client_type: str


@dataclass(frozen=True, slots=True)
class GoogleDriveSettings:
    """Credential-free settings needed by desktop staging and the controller."""

    schema_version: int
    google_cloud_project_id: str
    desktop_client_secrets: Path
    colab_oauth_client_id: str
    colab_secret_name: str = DEFAULT_COLAB_SECRET_NAME

    def as_value(self) -> dict[str, object]:
        return {
            "colab_oauth_client_id": self.colab_oauth_client_id,
            "colab_secret_name": self.colab_secret_name,
            "desktop_client_secrets": self.desktop_client_secrets.as_posix(),
            "google_cloud_project_id": self.google_cloud_project_id,
            "schema_version": self.schema_version,
        }


def default_google_drive_settings_path() -> Path:
    """Return the per-user settings location without creating it."""

    configured = os.environ.get("COLABCOLA_CONFIG_HOME")
    if configured:
        return Path(configured).expanduser() / "google-drive.json"
    if os.name == "nt" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "colabcola" / "google-drive.json"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "colabcola" / "google-drive.json"


def load_google_oauth_client(
    path: Path | str, *, expected_type: str | None = None
) -> GoogleOAuthClient:
    """Strictly load a Google-downloaded OAuth client JSON file."""

    source = _safe_regular_file(path, "Google OAuth client file")
    try:
        value = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except UnicodeDecodeError as exc:
        raise DriveTransportError("Google OAuth client file must be UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise DriveTransportError(
            f"Google OAuth client file is invalid JSON: {exc.msg}"
        ) from exc
    if not isinstance(value, dict) or len(value) != 1:
        raise DriveTransportError("Google OAuth client file must contain one client object")
    client_type, client_value = next(iter(value.items()))
    if client_type not in {"installed", "device"} or not isinstance(client_value, dict):
        raise DriveTransportError("Google OAuth client file has an unsupported client type")
    if expected_type is not None and client_type != expected_type:
        raise DriveTransportError(
            f"Google OAuth client must use type {expected_type!r}, observed {client_type!r}"
        )
    if not _OAUTH_CLIENT_REQUIRED_KEYS.issubset(client_value):
        raise DriveTransportError("Google OAuth client file is missing required fields")
    fields = {
        key: client_value[key]
        for key in ("project_id", "client_id", "client_secret", "auth_uri", "token_uri")
    }
    if any(not isinstance(item, str) or not item for item in fields.values()):
        raise DriveTransportError("Google OAuth client fields must be nonempty strings")
    if not _PROJECT_ID_RE.fullmatch(fields["project_id"]):
        raise DriveTransportError("Google OAuth project ID is invalid")
    if not _CLIENT_ID_RE.fullmatch(fields["client_id"]):
        raise DriveTransportError("Google OAuth client ID is invalid")
    if fields["auth_uri"] != "https://accounts.google.com/o/oauth2/auth":
        raise DriveTransportError("Google OAuth authorization endpoint is unexpected")
    if fields["token_uri"] != "https://oauth2.googleapis.com/token":
        raise DriveTransportError("Google OAuth token endpoint is unexpected")
    return GoogleOAuthClient(
        project_id=fields["project_id"],
        client_id=fields["client_id"],
        client_secret=fields["client_secret"],
        client_type=client_type,
    )


def google_drive_settings_from_clients(
    desktop_client_secrets: Path | str,
    colab_client_secrets: Path | str,
    *,
    colab_secret_name: str = DEFAULT_COLAB_SECRET_NAME,
) -> GoogleDriveSettings:
    """Build credential-free settings from two same-project OAuth clients."""

    desktop_path = _safe_regular_file(
        desktop_client_secrets, "desktop Google OAuth client file"
    )
    desktop = load_google_oauth_client(desktop_path, expected_type="installed")
    colab = load_google_oauth_client(colab_client_secrets)
    if desktop.project_id != colab.project_id:
        raise DriveTransportError(
            "desktop and Colab OAuth clients must belong to the same Google Cloud project"
        )
    if desktop.client_id == colab.client_id:
        raise DriveTransportError(
            "desktop and Colab OAuth clients must be separate platform clients"
        )
    settings = GoogleDriveSettings(
        schema_version=GOOGLE_DRIVE_SETTINGS_SCHEMA_VERSION,
        google_cloud_project_id=desktop.project_id,
        desktop_client_secrets=desktop_path,
        colab_oauth_client_id=colab.client_id,
        colab_secret_name=colab_secret_name,
    )
    verify_google_drive_settings(settings)
    return settings


def verify_google_drive_settings(settings: GoogleDriveSettings) -> None:
    if settings.schema_version != GOOGLE_DRIVE_SETTINGS_SCHEMA_VERSION:
        raise DriveTransportError("unsupported Google Drive settings schema version")
    if not _PROJECT_ID_RE.fullmatch(settings.google_cloud_project_id):
        raise DriveTransportError("Google Drive settings project ID is invalid")
    if not _CLIENT_ID_RE.fullmatch(settings.colab_oauth_client_id):
        raise DriveTransportError("Google Drive settings Colab client ID is invalid")
    if not _SECRET_NAME_RE.fullmatch(settings.colab_secret_name):
        raise DriveTransportError("Google Drive settings Colab secret name is invalid")
    desktop_path = _safe_regular_file(
        settings.desktop_client_secrets, "desktop Google OAuth client file"
    )
    if not desktop_path.is_absolute():
        raise DriveTransportError("desktop Google OAuth client path must be absolute")
    desktop = load_google_oauth_client(desktop_path, expected_type="installed")
    if desktop.project_id != settings.google_cloud_project_id:
        raise DriveTransportError("desktop OAuth client project does not match settings")


def load_google_drive_settings(path: Path | str | None = None) -> GoogleDriveSettings:
    """Strictly load credential-free Google Drive settings."""

    source = _safe_regular_file(
        default_google_drive_settings_path() if path is None else path,
        "Google Drive settings",
    )
    try:
        value = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except UnicodeDecodeError as exc:
        raise DriveTransportError("Google Drive settings must be UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise DriveTransportError(f"Google Drive settings are invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict) or set(value) != _SETTINGS_KEYS:
        raise DriveTransportError("Google Drive settings have missing or unknown fields")
    if isinstance(value["schema_version"], bool) or not isinstance(
        value["schema_version"], int
    ):
        raise DriveTransportError("Google Drive settings schema version must be an integer")
    string_keys = _SETTINGS_KEYS - {"schema_version"}
    if any(not isinstance(value[key], str) or not value[key] for key in string_keys):
        raise DriveTransportError("Google Drive settings fields must be nonempty strings")
    settings = GoogleDriveSettings(
        schema_version=value["schema_version"],
        google_cloud_project_id=value["google_cloud_project_id"],
        desktop_client_secrets=Path(value["desktop_client_secrets"]),
        colab_oauth_client_id=value["colab_oauth_client_id"],
        colab_secret_name=value["colab_secret_name"],
    )
    verify_google_drive_settings(settings)
    if _settings_json_bytes(settings) != source.read_bytes():
        raise DriveTransportError("Google Drive settings are not in canonical persisted form")
    return settings


def write_google_drive_settings(
    settings: GoogleDriveSettings, path: Path | str | None = None
) -> Path:
    """Atomically persist credential-free Google Drive settings."""

    verify_google_drive_settings(settings)
    destination = default_google_drive_settings_path() if path is None else Path(path)
    if destination.suffix.lower() != ".json":
        raise DriveTransportError("Google Drive settings path must use the .json extension")
    if destination.exists() and destination.is_dir():
        raise DriveTransportError("Google Drive settings path is a directory")
    if destination.is_symlink():
        raise DriveTransportError("Google Drive settings path may not be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = _settings_json_bytes(settings)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return destination.resolve(strict=True)


class GoogleDriveApiBackend:
    """DriveBackend implemented with Google's supported Python client libraries."""

    def __init__(self, settings: GoogleDriveSettings, *, force_authorize: bool = False):
        verify_google_drive_settings(settings)
        self.settings = settings
        credentials = _desktop_credentials(settings, force_authorize=force_authorize)
        try:
            from googleapiclient.discovery import build

            self._files = build(
                "drive", "v3", credentials=credentials, cache_discovery=False
            ).files()
        except Exception as exc:
            raise _google_client_error("construct Drive API client", exc) from exc

    def ensure_folder(self, *, name: str, parent_id: str) -> DriveFolder:
        query = (
            f"name = '{_drive_query_literal(name)}' and "
            f"'{_drive_query_literal(parent_id)}' in parents and "
            f"mimeType = '{GOOGLE_FOLDER_MIME_TYPE}' and trashed = false"
        )
        matches = self._list_files(query, "id,name,parents")
        if len(matches) > 1:
            raise DriveTransportError(
                f"Drive contains multiple app-visible folders named {name!r} under one parent"
            )
        if matches:
            value = matches[0]
        else:
            try:
                value = (
                    self._files.create(
                        body={
                            "name": name,
                            "mimeType": GOOGLE_FOLDER_MIME_TYPE,
                            "parents": [parent_id],
                        },
                        fields="id,name,parents",
                    )
                    .execute(num_retries=3)
                )
            except Exception as exc:
                raise _google_client_error("create Drive staging folder", exc) from exc
        return DriveFolder(
            file_id=_required_string(value, "id", "Drive folder"),
            name=_required_string(value, "name", "Drive folder"),
            parent_id=parent_id,
        )

    def find_payload(
        self,
        *,
        name: str,
        parent_id: str,
        app_properties: Mapping[str, str],
    ) -> Sequence[DriveFile]:
        clauses = [
            f"name = '{_drive_query_literal(name)}'",
            f"'{_drive_query_literal(parent_id)}' in parents",
            "trashed = false",
        ]
        clauses.extend(
            "appProperties has { key='"
            + _drive_query_literal(key)
            + "' and value='"
            + _drive_query_literal(value)
            + "' }"
            for key, value in sorted(app_properties.items())
        )
        return tuple(
            _drive_file(value, expected_parent_id=parent_id)
            for value in self._list_files(" and ".join(clauses), _DRIVE_FILE_FIELDS)
        )

    def upload_resumable(
        self,
        *,
        local_path: Path,
        name: str,
        parent_id: str,
        mime_type: str,
        app_properties: Mapping[str, str],
    ) -> DriveFile:
        if mime_type != DRIVE_BINARY_MIME_TYPE:
            raise DriveTransportError("unexpected Drive upload MIME type")
        source = _safe_regular_file(local_path, "Drive upload source")
        try:
            from googleapiclient.http import MediaFileUpload

            media = MediaFileUpload(
                os.fspath(source),
                mimetype=mime_type,
                chunksize=_UPLOAD_CHUNK_BYTES,
                resumable=True,
            )
            request = self._files.create(
                body={
                    "appProperties": dict(sorted(app_properties.items())),
                    "mimeType": mime_type,
                    "name": name,
                    "parents": [parent_id],
                },
                media_body=media,
                fields=_DRIVE_FILE_FIELDS,
            )
            response = None
            while response is None:
                _, response = request.next_chunk(num_retries=5)
        except Exception as exc:
            raise _google_client_error(
                "perform resumable upload of Drive payload", exc
            ) from exc
        return _drive_file(response, expected_parent_id=parent_id)

    def trash(self, *, file_id: str) -> None:
        try:
            response = (
                self._files.update(
                    fileId=file_id, body={"trashed": True}, fields="id,trashed"
                ).execute(num_retries=3)
            )
        except Exception as exc:
            raise _google_client_error("trash receipt-scoped Drive object", exc) from exc
        if response.get("id") != file_id or response.get("trashed") is not True:
            raise DriveTransportError("Drive trash response did not confirm the exact object")

    def _list_files(self, query: str, fields: str) -> tuple[Mapping[str, Any], ...]:
        values: list[Mapping[str, Any]] = []
        page_token = None
        try:
            while True:
                response = (
                    self._files.list(
                        q=query,
                        spaces="drive",
                        fields=f"nextPageToken,files({fields})",
                        pageSize=100,
                        pageToken=page_token,
                    ).execute(num_retries=3)
                )
                page = response.get("files", [])
                if not isinstance(page, list) or any(not isinstance(item, dict) for item in page):
                    raise DriveTransportError("Drive list response has an invalid shape")
                values.extend(page)
                page_token = response.get("nextPageToken")
                if not page_token:
                    break
        except DriveTransportError:
            raise
        except Exception as exc:
            raise _google_client_error("list app-visible Drive objects", exc) from exc
        return tuple(values)


def create_google_drive_backend(
    settings: GoogleDriveSettings, *, force_authorize: bool = False
) -> GoogleDriveApiBackend:
    """Public factory kept patchable for CLI and adapter tests."""

    return GoogleDriveApiBackend(settings, force_authorize=force_authorize)


def _desktop_credentials(
    settings: GoogleDriveSettings, *, force_authorize: bool, interactive: bool = True,
    minimum_lifetime_seconds: int = 0
) -> object:
    """Load/refresh a user token from the OS keyring or run desktop OAuth."""

    try:
        import keyring
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise DriveTransportError(
            "Google Drive support requires google-api-python-client, "
            "google-auth-oauthlib, and keyring"
        ) from exc

    desktop = load_google_oauth_client(
        settings.desktop_client_secrets, expected_type="installed"
    )
    token_key = hashlib.sha256(
        (settings.google_cloud_project_id + "\0" + desktop.client_id).encode("utf-8")
    ).hexdigest()
    credentials = None
    try:
        serialized = None if force_authorize else keyring.get_password(_KEYRING_SERVICE, token_key)
        if serialized:
            token_value = json.loads(serialized, object_pairs_hook=_unique_object)
            credentials = Credentials.from_authorized_user_info(
                token_value, scopes=[GOOGLE_DRIVE_SCOPE]
            )
        from datetime import datetime, timedelta, timezone
        needs_refresh = credentials and credentials.expired
        if credentials and minimum_lifetime_seconds:
            needs_refresh = (not credentials.expiry or credentials.expiry.replace(tzinfo=timezone.utc)
                             < datetime.now(timezone.utc) + timedelta(seconds=minimum_lifetime_seconds))
        if credentials and needs_refresh and credentials.refresh_token:
            credentials.refresh(Request())
        if credentials is None or not credentials.valid or not credentials.has_scopes(
            [GOOGLE_DRIVE_SCOPE]
        ):
            if not interactive:
                raise DriveTransportError("cached Drive authorization is unavailable")
            flow = InstalledAppFlow.from_client_secrets_file(
                os.fspath(settings.desktop_client_secrets), [GOOGLE_DRIVE_SCOPE]
            )
            credentials = flow.run_local_server(
                host="127.0.0.1",
                port=0,
                open_browser=True,
                authorization_prompt_message=(
                    "Open this URL to grant colabcola per-file Google Drive access:\n{url}"
                ),
                success_message="colabcola Google Drive authorization completed; you may close this tab.",
            )
        if not credentials.valid or not credentials.has_scopes([GOOGLE_DRIVE_SCOPE]):
            raise DriveTransportError("Google OAuth did not grant the required drive.file scope")
        keyring.set_password(_KEYRING_SERVICE, token_key, credentials.to_json())
        return credentials
    except DriveTransportError:
        raise
    except Exception as exc:
        raise _google_client_error("authorize Google Drive in the OS credential store", exc) from exc


def _settings_json_bytes(settings: GoogleDriveSettings) -> bytes:
    return (
        json.dumps(
            settings.as_value(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _safe_regular_file(path: Path | str, label: str) -> Path:
    source = Path(path).expanduser()
    try:
        metadata = os.lstat(source)
    except OSError as exc:
        raise DriveTransportError(f"{label} is unavailable") from exc
    attributes = getattr(metadata, "st_file_attributes", 0)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or attributes & _REPARSE_FLAG
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise DriveTransportError(f"{label} must be one regular non-link file")
    try:
        return source.resolve(strict=True)
    except OSError as exc:
        raise DriveTransportError(f"{label} is unavailable") from exc


def _drive_file(value: Mapping[str, Any], *, expected_parent_id: str) -> DriveFile:
    parents = value.get("parents")
    properties = value.get("appProperties", {})
    size = value.get("size")
    if value.get("trashed") is not False:
        raise DriveTransportError("Drive adapter observed a trashed payload")
    if parents != [expected_parent_id]:
        raise DriveTransportError("Drive payload parent identity is unexpected")
    if not isinstance(properties, dict) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in properties.items()
    ):
        raise DriveTransportError("Drive payload app properties are invalid")
    try:
        parsed_size = int(size)
    except (TypeError, ValueError) as exc:
        raise DriveTransportError("Drive payload size is invalid") from exc
    return DriveFile(
        file_id=_required_string(value, "id", "Drive payload"),
        name=_required_string(value, "name", "Drive payload"),
        parent_id=expected_parent_id,
        size=parsed_size,
        mime_type=_required_string(value, "mimeType", "Drive payload"),
        app_properties=tuple(sorted(properties.items())),
    )


def _required_string(value: Mapping[str, Any], key: str, label: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise DriveTransportError(f"{label} field {key!r} is invalid")
    return item


def _drive_query_literal(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise DriveTransportError("Drive query identity must be a nonempty string")
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _google_client_error(action: str, exc: Exception) -> DriveTransportError:
    status = getattr(getattr(exc, "resp", None), "status", None)
    suffix = f" (HTTP {status})" if isinstance(status, int) else ""
    return DriveTransportError(f"failed to {action}{suffix}: {exc.__class__.__name__}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DriveTransportError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result
