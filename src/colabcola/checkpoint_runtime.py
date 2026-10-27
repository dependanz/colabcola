"""Notebook source for durable Google Drive checkpoint publication."""

from __future__ import annotations

from typing import Final


DURABLE_CHECKPOINT_SOURCE: Final[str] = r'''
# Publish complete checkpoint pairs to Drive. The manifest uploaded last is the commit marker.
def _colabcola_checkpoint_config():
    value = COLABCOLA_MANIFEST.get("durable_checkpoints")
    if value is None:
        return None
    if _COLABCOLA_TRANSPORT_FACTS.get("mode") != "google_drive":
        raise RuntimeError("durable checkpoints require google_drive transport")
    if _COLABCOLA_DRIVE_AUTH_CONTEXT is None:
        raise RuntimeError("durable checkpoint Drive authorization is unavailable")
    return value


def _colabcola_checkpoint_json(value):
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8") + b"\n"


def _colabcola_checkpoint_token():
    context = _COLABCOLA_DRIVE_AUTH_CONTEXT
    if context is None:
        raise RuntimeError("durable checkpoint Drive authorization is unavailable")
    with context["lock"]:
        if time.monotonic() < context["access_token_expires_monotonic"]:
            return context["access_token"]
        body = urllib.parse.urlencode(
            {
                "client_id": context["client_id"],
                "client_secret": context["client_secret"],
                "refresh_token": context["refresh_token"],
                "grant_type": "refresh_token",
            }
        ).encode("ascii")
        request = urllib.request.Request(
            "https://oauth2.googleapis.com/token",
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        status, response = _colabcola_json_response(request)
        expires_in = response.get("expires_in", 3600)
        if (
            status != 200
            or not isinstance(response.get("access_token"), str)
            or isinstance(expires_in, bool)
            or not isinstance(expires_in, int)
            or expires_in <= 0
        ):
            raise RuntimeError("Google Drive checkpoint token refresh failed")
        context["access_token"] = response["access_token"]
        context["access_token_expires_monotonic"] = time.monotonic() + max(
            0, expires_in - 60
        )
        return context["access_token"]


def _colabcola_checkpoint_request(url, *, data=None, headers=None, method=None):
    request_headers = {"Authorization": "Bearer " + _colabcola_checkpoint_token()}
    request_headers.update(headers or {})
    request = urllib.request.Request(
        url, data=data, headers=request_headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()


def _colabcola_checkpoint_json_request(url, *, value=None, method=None):
    data = None if value is None else _colabcola_checkpoint_json(value)
    headers = {} if data is None else {"Content-Type": "application/json; charset=UTF-8"}
    status, response_headers, content = _colabcola_checkpoint_request(
        url, data=data, headers=headers, method=method
    )
    try:
        response = json.loads(
            content.decode("utf-8"), object_pairs_hook=_colabcola_unique_object
        )
    except Exception as exc:
        raise RuntimeError(
            f"Google Drive checkpoint endpoint returned invalid JSON (HTTP {status})"
        ) from exc
    if not isinstance(response, dict):
        raise RuntimeError("Google Drive checkpoint endpoint returned non-object JSON")
    return status, response_headers, response


def _colabcola_checkpoint_query_literal(value):
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


def _colabcola_checkpoint_list(role):
    query = (
        "trashed = false and "
        f"appProperties has {{ key='colabcola_job_id' and value='{_colabcola_checkpoint_query_literal(COLABCOLA_JOB_ID)}' }} and "
        f"appProperties has {{ key='colabcola_checkpoint_role' and value='{_colabcola_checkpoint_query_literal(role)}' }}"
    )
    fields = "nextPageToken,files(id,name,parents,size,md5Checksum,mimeType,appProperties,trashed)"
    files = []
    page_token = None
    while True:
        parameters = {
            "q": query,
            "fields": fields,
            "pageSize": "1000",
            "spaces": "drive",
            "supportsAllDrives": "true",
        }
        if page_token is not None:
            parameters["pageToken"] = page_token
        url = "https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode(
            parameters
        )
        status, _, response = _colabcola_checkpoint_json_request(url)
        if status != 200 or not isinstance(response.get("files"), list):
            raise RuntimeError("Google Drive checkpoint listing failed")
        files.extend(response["files"])
        page_token = response.get("nextPageToken")
        if page_token is None:
            return files
        if not isinstance(page_token, str) or not page_token:
            raise RuntimeError("Google Drive checkpoint listing returned invalid pagination")


def _colabcola_checkpoint_folder():
    matches = _colabcola_checkpoint_list("folder")
    if len(matches) > 1:
        raise RuntimeError("ambiguous durable checkpoint folder")
    if matches:
        folder = matches[0]
        if (
            folder.get("name") != "checkpoints"
            or folder.get("parents") != [COLABCOLA_DRIVE_PARENT_ID]
            or folder.get("mimeType") != "application/vnd.google-apps.folder"
        ):
            raise RuntimeError("durable checkpoint folder metadata is invalid")
        return folder["id"]
    metadata = {
        "name": "checkpoints",
        "parents": [COLABCOLA_DRIVE_PARENT_ID],
        "mimeType": "application/vnd.google-apps.folder",
        "appProperties": {
            "colabcola_job_id": COLABCOLA_JOB_ID,
            "colabcola_checkpoint_role": "folder",
        },
    }
    fields = urllib.parse.quote("id,name,parents,mimeType,appProperties,trashed", safe=",")
    url = "https://www.googleapis.com/drive/v3/files?supportsAllDrives=true&fields=" + fields
    status, _, folder = _colabcola_checkpoint_json_request(
        url, value=metadata, method="POST"
    )
    if status not in {200, 201} or not isinstance(folder.get("id"), str):
        raise RuntimeError("could not create durable checkpoint folder")
    return folder["id"]


def _colabcola_checkpoint_file_evidence(path):
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            sha256.update(block)
            md5.update(block)
            size += len(block)
    return {
        "name": path.name,
        "size": size,
        "sha256": sha256.hexdigest(),
        "md5": md5.hexdigest(),
    }


def _colabcola_local_checkpoint_pairs(root, config):
    digits = config["filename_digits"]
    prefixes = (config["generator_prefix"], config["state_prefix"])
    patterns = [
        re.compile(r"^" + re.escape(prefix) + r"_(\d{" + str(digits) + r"})$")
        for prefix in prefixes
    ]
    found = [{}, {}]
    if not root.exists():
        return []
    for candidate in root.iterdir():
        if not candidate.is_file() or candidate.is_symlink():
            continue
        for index, pattern in enumerate(patterns):
            match = pattern.fullmatch(candidate.name)
            if match:
                step = int(match.group(1))
                if step in found[index]:
                    raise RuntimeError(f"duplicate local checkpoint member for step {step}")
                found[index][step] = candidate
    return [
        (step, found[0][step], found[1][step])
        for step in sorted(set(found[0]) & set(found[1]))
    ]


def _colabcola_checkpoint_upload(path, role, step, folder_id, evidence):
    metadata = {
        "name": path.name,
        "parents": [folder_id],
        "mimeType": "application/octet-stream",
        "appProperties": {
            "colabcola_job_id": COLABCOLA_JOB_ID,
            "colabcola_checkpoint_role": role,
            "colabcola_checkpoint_step": str(step),
            "colabcola_checkpoint_sha256": evidence["sha256"],
        },
    }
    fields = urllib.parse.quote(
        "id,name,parents,size,md5Checksum,mimeType,appProperties,trashed", safe=","
    )
    url = (
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable"
        "&supportsAllDrives=true&fields=" + fields
    )
    status, response_headers, _ = _colabcola_checkpoint_request(
        url,
        data=_colabcola_checkpoint_json(metadata),
        headers={
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Type": "application/octet-stream",
            "X-Upload-Content-Length": str(evidence["size"]),
        },
        method="POST",
    )
    location = response_headers.get("Location")
    if status not in {200, 201} or not isinstance(location, str) or not location:
        raise RuntimeError(f"could not start {role} checkpoint upload")
    offset = 0
    result = None
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            end = offset + len(block) - 1
            upload_status, upload_headers, content = _colabcola_checkpoint_request(
                location,
                data=block,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(len(block)),
                    "Content-Range": f"bytes {offset}-{end}/{evidence['size']}",
                },
                method="PUT",
            )
            offset = end + 1
            if upload_status in {200, 201}:
                result = json.loads(
                    content.decode("utf-8"), object_pairs_hook=_colabcola_unique_object
                )
            elif upload_status == 308:
                if upload_headers.get("Range") != f"bytes=0-{end}":
                    raise RuntimeError(
                        f"{role} checkpoint upload acknowledged an unexpected range"
                    )
            else:
                raise RuntimeError(
                    f"{role} checkpoint upload failed (HTTP {upload_status})"
                )
    if offset != evidence["size"] or not isinstance(result, dict):
        raise RuntimeError(f"{role} checkpoint upload did not complete")
    if (
        result.get("name") != evidence["name"]
        or result.get("parents") != [folder_id]
        or int(result.get("size", -1)) != evidence["size"]
        or result.get("md5Checksum") != evidence["md5"]
        or result.get("appProperties") != metadata["appProperties"]
        or result.get("trashed") is not False
    ):
        raise RuntimeError(f"{role} checkpoint upload verification failed")
    return result


def _colabcola_checkpoint_json_request_raw(url, body, headers):
    status, response_headers, content = _colabcola_checkpoint_request(
        url, data=body, headers=headers, method="POST"
    )
    try:
        value = json.loads(
            content.decode("utf-8"), object_pairs_hook=_colabcola_unique_object
        )
    except Exception as exc:
        raise RuntimeError(
            f"Google Drive checkpoint endpoint returned invalid JSON (HTTP {status})"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError("Google Drive checkpoint endpoint returned non-object JSON")
    return status, response_headers, value


def _colabcola_checkpoint_upload_manifest(manifest, folder_id):
    content = _colabcola_checkpoint_json(manifest)
    digest = hashlib.sha256(content).hexdigest()
    metadata = {
        "name": f"checkpoint_{manifest['step']:016d}.json",
        "parents": [folder_id],
        "mimeType": "application/json",
        "appProperties": {
            "colabcola_job_id": COLABCOLA_JOB_ID,
            "colabcola_checkpoint_role": "manifest",
            "colabcola_checkpoint_step": str(manifest["step"]),
            "colabcola_checkpoint_sha256": digest,
        },
    }
    boundary = "colabcola-checkpoint-boundary"
    body = (
        ("--" + boundary + "\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n").encode("ascii")
        + _colabcola_checkpoint_json(metadata)
        + ("\r\n--" + boundary + "\r\nContent-Type: application/json\r\n\r\n").encode("ascii")
        + content
        + ("\r\n--" + boundary + "--\r\n").encode("ascii")
    )
    fields = urllib.parse.quote(
        "id,name,parents,size,md5Checksum,mimeType,appProperties,trashed", safe=","
    )
    url = (
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"
        "&supportsAllDrives=true&fields=" + fields
    )
    status, _, response = _colabcola_checkpoint_json_request_raw(
        url, body, {"Content-Type": "multipart/related; boundary=" + boundary}
    )
    if (
        status not in {200, 201}
        or response.get("appProperties") != metadata["appProperties"]
        or response.get("parents") != [folder_id]
        or response.get("trashed") is not False
    ):
        raise RuntimeError("durable checkpoint manifest commit failed")
    return response


def _colabcola_checkpoint_download(file_id):
    quoted = urllib.parse.quote(file_id, safe="")
    url = (
        "https://www.googleapis.com/drive/v3/files/"
        + quoted
        + "?alt=media&supportsAllDrives=true"
    )
    status, _, content = _colabcola_checkpoint_request(url)
    if status != 200:
        raise RuntimeError("durable checkpoint download failed")
    return content


def _colabcola_committed_checkpoints(folder_id):
    records = []
    seen_steps = set()
    for metadata in _colabcola_checkpoint_list("manifest"):
        properties = metadata.get("appProperties")
        if not isinstance(properties, dict) or metadata.get("parents") != [folder_id]:
            raise RuntimeError("durable checkpoint manifest metadata is invalid")
        try:
            step = int(properties["colabcola_checkpoint_step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("durable checkpoint manifest step is invalid") from exc
        if step in seen_steps:
            raise RuntimeError(f"ambiguous durable checkpoint manifest for step {step}")
        seen_steps.add(step)
        content = _colabcola_checkpoint_download(metadata["id"])
        if (
            hashlib.sha256(content).hexdigest()
            != properties.get("colabcola_checkpoint_sha256")
        ):
            raise RuntimeError("durable checkpoint manifest hash is invalid")
        try:
            manifest = json.loads(
                content.decode("utf-8"), object_pairs_hook=_colabcola_unique_object
            )
        except Exception as exc:
            raise RuntimeError("durable checkpoint manifest content is invalid") from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != 1
            or manifest.get("job_id") != COLABCOLA_JOB_ID
            or manifest.get("step") != step
            or step <= 0
            or not isinstance(manifest.get("created_at_utc"), str)
            or set(manifest)
            != {
                "schema_version",
                "job_id",
                "step",
                "created_at_utc",
                "generator",
                "state",
            }
        ):
            raise RuntimeError("durable checkpoint manifest contract is invalid")
        config = COLABCOLA_MANIFEST["durable_checkpoints"]
        if step >= 10 ** config["filename_digits"]:
            raise RuntimeError("durable checkpoint manifest step exceeds filename width")
        expected_names = {
            "generator": (
                config["generator_prefix"]
                + "_"
                + str(step).zfill(config["filename_digits"])
            ),
            "state": (
                config["state_prefix"]
                + "_"
                + str(step).zfill(config["filename_digits"])
            ),
        }
        for role in ("generator", "state"):
            member = manifest.get(role)
            if (
                not isinstance(member, dict)
                or set(member) != {"id", "name", "size", "sha256"}
                or not isinstance(member.get("id"), str)
                or not member["id"]
                or member.get("name") != expected_names[role]
                or isinstance(member.get("size"), bool)
                or not isinstance(member.get("size"), int)
                or member["size"] <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", str(member.get("sha256")))
            ):
                raise RuntimeError("durable checkpoint manifest member is invalid")
        records.append({"metadata": metadata, "manifest": manifest})
    return sorted(records, key=lambda record: record["manifest"]["step"])


def _colabcola_checkpoint_member_metadata(member, role, step, folder_id):
    quoted = urllib.parse.quote(member["id"], safe="")
    fields = urllib.parse.quote(
        "id,name,parents,size,mimeType,appProperties,trashed", safe=","
    )
    url = (
        "https://www.googleapis.com/drive/v3/files/"
        + quoted
        + "?supportsAllDrives=true&fields="
        + fields
    )
    status, _, metadata = _colabcola_checkpoint_json_request(url)
    expected_properties = {
        "colabcola_job_id": COLABCOLA_JOB_ID,
        "colabcola_checkpoint_role": role,
        "colabcola_checkpoint_step": str(step),
        "colabcola_checkpoint_sha256": member["sha256"],
    }
    if (
        status != 200
        or metadata.get("id") != member["id"]
        or metadata.get("name") != member["name"]
        or metadata.get("parents") != [folder_id]
        or int(metadata.get("size", -1)) != member["size"]
        or metadata.get("appProperties") != expected_properties
        or metadata.get("trashed") is not False
    ):
        raise RuntimeError("durable checkpoint member metadata is invalid")


def _colabcola_restore_checkpoint(root, folder_id, record):
    manifest = record["manifest"]
    root.mkdir(parents=True, exist_ok=True)
    for role in ("generator", "state"):
        member = manifest[role]
        _colabcola_checkpoint_member_metadata(
            member, role, manifest["step"], folder_id
        )
        content = _colabcola_checkpoint_download(member["id"])
        if (
            len(content) != member["size"]
            or hashlib.sha256(content).hexdigest() != member["sha256"]
        ):
            raise RuntimeError("durable checkpoint member content is invalid")
        target = root / member["name"]
        temporary = root / ("." + member["name"] + ".restore.tmp")
        target.resolve().relative_to(root.resolve())
        temporary.resolve().relative_to(root.resolve())
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    return manifest["step"]


def _colabcola_checkpoint_trash(file_id):
    quoted = urllib.parse.quote(file_id, safe="")
    fields = urllib.parse.quote("id,trashed", safe=",")
    url = (
        "https://www.googleapis.com/drive/v3/files/"
        + quoted
        + "?supportsAllDrives=true&fields="
        + fields
    )
    status, _, response = _colabcola_checkpoint_json_request(
        url, value={"trashed": True}, method="PATCH"
    )
    if (
        status != 200
        or response.get("id") != file_id
        or response.get("trashed") is not True
    ):
        raise RuntimeError("durable checkpoint retention cleanup failed")


def _colabcola_checkpoint_retain(records, limit):
    while len(records) > limit:
        obsolete = records.pop(0)
        # Remove the commit marker first. If interrupted, the remaining pair is an orphan
        # and can never be selected as a recovery point.
        _colabcola_checkpoint_trash(obsolete["metadata"]["id"])
        _colabcola_checkpoint_trash(obsolete["manifest"]["generator"]["id"])
        _colabcola_checkpoint_trash(obsolete["manifest"]["state"]["id"])


def _colabcola_checkpoint_record_facts(records):
    return [
        {
            "step": record["manifest"]["step"],
            "manifest_file_id": record["metadata"]["id"],
            "generator_file_id": record["manifest"]["generator"]["id"],
            "state_file_id": record["manifest"]["state"]["id"],
        }
        for record in records
    ]


def _colabcola_publish_checkpoint(
    step, generator_path, state_path, folder_id, records
):
    generator_evidence = _colabcola_checkpoint_file_evidence(generator_path)
    state_evidence = _colabcola_checkpoint_file_evidence(state_path)
    generator = _colabcola_checkpoint_upload(
        generator_path, "generator", step, folder_id, generator_evidence
    )
    state = _colabcola_checkpoint_upload(
        state_path, "state", step, folder_id, state_evidence
    )
    manifest = {
        "schema_version": 1,
        "job_id": COLABCOLA_JOB_ID,
        "step": step,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generator": {
            "id": generator["id"],
            "name": generator_evidence["name"],
            "size": generator_evidence["size"],
            "sha256": generator_evidence["sha256"],
        },
        "state": {
            "id": state["id"],
            "name": state_evidence["name"],
            "size": state_evidence["size"],
            "sha256": state_evidence["sha256"],
        },
    }
    manifest_metadata = _colabcola_checkpoint_upload_manifest(manifest, folder_id)
    records.append({"metadata": manifest_metadata, "manifest": manifest})
    return manifest


def _colabcola_checkpoint_worker(
    root, config, folder_id, records, stop, errors, facts
):
    published_steps = {record["manifest"]["step"] for record in records}
    try:
        while True:
            for step, generator_path, state_path in _colabcola_local_checkpoint_pairs(
                root, config
            ):
                if step not in published_steps:
                    manifest = _colabcola_publish_checkpoint(
                        step, generator_path, state_path, folder_id, records
                    )
                    published_steps.add(step)
                    facts["publications"].append(
                        {
                            "step": manifest["step"],
                            "manifest_file_id": records[-1]["metadata"]["id"],
                            "generator_file_id": manifest["generator"]["id"],
                            "state_file_id": manifest["state"]["id"],
                        }
                    )
                    _colabcola_checkpoint_retain(
                        records, config["remote_retention"]
                    )
                    facts["retained"] = _colabcola_checkpoint_record_facts(records)
            if stop.wait(config["poll_seconds"]):
                break
        # One final scan catches a pair written immediately before process exit.
        for step, generator_path, state_path in _colabcola_local_checkpoint_pairs(
            root, config
        ):
            if step not in published_steps:
                manifest = _colabcola_publish_checkpoint(
                    step, generator_path, state_path, folder_id, records
                )
                published_steps.add(step)
                facts["publications"].append(
                    {
                        "step": manifest["step"],
                        "manifest_file_id": records[-1]["metadata"]["id"],
                        "generator_file_id": manifest["generator"]["id"],
                        "state_file_id": manifest["state"]["id"],
                    }
                )
                _colabcola_checkpoint_retain(records, config["remote_retention"])
                facts["retained"] = _colabcola_checkpoint_record_facts(records)
    except Exception as exc:
        errors.append(exc)
        stop.set()
'''
