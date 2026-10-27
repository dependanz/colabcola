"""Read-only launch admission. Reports are outside the immutable approved plan."""
from __future__ import annotations
import ast
import base64
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys
import time
import uuid
from urllib.request import Request
from ._version import __version__
from .lifecycle_io import (LifecycleError, atomic, digest, encoded, verified_inputs,
                           read_regular, INPUT_BYTES, SNAPSHOT_BYTES, RESULT_METADATA_BYTES)


class PreflightError(LifecycleError):
    pass


def installation_integrity(package_root=None, distribution=None):
    """Check installed wheel RECORD, including missing/extra package code/resources.

    This detects installation damage; RECORD is not a publisher signature. Editable
    installs cannot establish this claim and must use a locally built wheel.
    """
    package_root = Path(package_root or Path(__file__).parent).resolve()
    distribution = distribution or importlib.metadata.distribution("colabcola")
    expected = {}
    for entry in distribution.files or ():
        name = str(entry).replace("\\", "/")
        if name.startswith("colabcola/") and name.endswith((".py", ".json")):
            relative = name.removeprefix("colabcola/")
            if not entry.hash or entry.hash.mode != "sha256":
                raise LifecycleError("controller wheel RECORD has no SHA256")
            expected[relative] = entry
    actual = {p.relative_to(package_root).as_posix() for p in package_root.rglob("*")
              if p.is_file() and p.suffix in (".py", ".json") and "__pycache__" not in p.parts}
    if not expected or actual != set(expected):
        raise LifecycleError("install a built colabcola wheel; package files differ from RECORD")
    for relative, entry in expected.items():
        path = package_root / relative
        if path.is_symlink() or Path(distribution.locate_file(entry)).resolve() != path.resolve():
            raise LifecycleError("controller is not the recorded wheel installation")
        with path.open("rb") as stream:
            actual_hash = base64.urlsafe_b64encode(hashlib.file_digest(stream, "sha256").digest()).rstrip(b"=").decode()
        if actual_hash != entry.hash.value:
            raise LifecycleError("controller wheel file integrity failure")
    return {"version": __version__, "files_verified": len(expected), "basis": "installed_wheel_RECORD"}


def frozen_job(directory, plan):
    """Recompute embedded job identity from literals, never execute notebook code."""
    document = json.loads(read_regular(directory, "notebook.ipynb", INPUT_BYTES))
    constants, chunks = {}, []
    for cell in document["cells"]:
        group = cell.get("metadata", {}).get("colabcola", {}).get("group")
        if group not in ("manifest", "payload"):
            continue
        source = cell["source"]
        tree = ast.parse(source if isinstance(source, str) else "".join(source))
        if group == "manifest":
            for node in tree.body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    if name in ("COLABCOLA_MANIFEST_JSON", "COLABCOLA_ARCHIVE_SHA256", "COLABCOLA_ARCHIVE_BYTES", "COLABCOLA_JOB_ID"):
                        if name in constants:
                            raise LifecycleError("duplicate frozen constant")
                        constants[name] = ast.literal_eval(node.value)
        else:
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
                raise LifecycleError("invalid frozen payload")
            call = tree.body[0].value
            if (not isinstance(call, ast.Call) or len(call.args) != 1 or call.keywords
                or not isinstance(call.func, ast.Attribute) or call.func.attr != "append"
                or not isinstance(call.func.value, ast.Name) or call.func.value.id != "COLABCOLA_PAYLOAD_CHUNKS"):
                raise LifecycleError("invalid frozen payload expression")
            chunks.append(ast.literal_eval(call.args[0]))
    sha, size = hashlib.sha256(), 0
    for chunk in chunks:
        body = base64.b64decode(chunk, validate=True)
        size += len(body)
        sha.update(body)
    if not chunks or size != constants["COLABCOLA_ARCHIVE_BYTES"] or sha.hexdigest() != constants["COLABCOLA_ARCHIVE_SHA256"]:
        raise LifecycleError("frozen archive identity mismatch")
    manifest = constants["COLABCOLA_MANIFEST_JSON"].encode("utf-8")
    archive_hash = sha.digest()
    identity = digest(len(manifest).to_bytes(8,"big") + manifest + len(archive_hash).to_bytes(8,"big") + archive_hash)
    if identity != plan["job_id"] or identity != constants["COLABCOLA_JOB_ID"]:
        raise LifecycleError("frozen job identity mismatch")
    return json.loads(manifest)


def drive_parent(parent, access, request=None):
    from urllib.parse import urlencode
    if not isinstance(parent, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", parent):
        raise LifecycleError("invalid Drive parent")
    fields = "id,mimeType,ownedByMe,trashed,capabilities(canAddChildren)"
    req = Request("https://www.googleapis.com/drive/v3/files/" + parent + "?" + urlencode({"fields":fields}),
                  headers={"Authorization":"Bearer " + access["access_token"]}, method="GET")
    # No redirects: never forward bearer credentials to another destination.
    if request is None:
        from urllib.request import HTTPRedirectHandler, build_opener
        class NoRedirect(HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        request = build_opener(NoRedirect).open
    with request(req, timeout=30) as response:
        if getattr(response, "status", 200) != 200:
            raise LifecycleError("Drive metadata request failed")
        body = response.read(65537)
    if len(body) > 65536:
        raise LifecycleError("Drive metadata too large")
    value = json.loads(body)
    if (value.get("id") != parent or value.get("mimeType") != "application/vnd.google-apps.folder"
        or value.get("ownedByMe") is not True or value.get("trashed") is not False
        or value.get("capabilities", {}).get("canAddChildren") is not True):
        raise LifecycleError("Drive parent must be an owned writable untrashed folder")
    return {"id":parent, "owned":True, "trashed":False, "can_add_children":True}


def run_preflight(directory, approval, *, validator=None, allow_active_assignments=(), observer=None, close_provider=False):
    """Caller holds the run lock. Return report and reusable provider/lease on success."""
    from .lifecycle import get_token
    from .lifecycle_auth import TokenLease
    from .lifecycle_provider import provider_for_plan, dependency_report
    directory = Path(directory).resolve()
    report = {"schema":1, "id":uuid.uuid4().hex, "started_at":time.time(), "state":"running",
              "controller_version":__version__, "platform":sys.platform, "checks":[],
              "network_accessed":False, "allocation_performed":False, "upload_performed":False}
    provider = None
    current = "journal"
    state = None
    def check(name, action):
        nonlocal current
        current = name
        item = {"name":name,"state":"running","started_at":time.time()}
        report["checks"].append(item)
        try:
            detail = action()
            item.update(state="passed", detail=detail, ended_at=time.time())
            if observer and hasattr(observer, "say"):
                observer.say("Preflight: " + name.replace("_"," ") + " passed")
            return detail
        except BaseException:
            item.update(state="failed", ended_at=time.time())
            raise
    def persist():
        report["ended_at"] = time.time()
        target = directory / "preflight" / (report["id"] + ".json")
        if target.parent.is_symlink():
            raise LifecycleError("preflight journal symlink rejected")
        atomic(target, report)
        if isinstance(state, dict):
            state["preflight"] = {"state":report["state"], "path":str(target), "sha256":digest(encoded(report))}
            state.setdefault("events", []).append({"sequence":len(state.get("events",[]))+1,
                "time":report["ended_at"], "event":"preflight_" + report["state"], "phase":state.get("phase")})
            atomic(directory / "state.json", state)
        return target
    try:
        state = json.loads((directory / "state.json").read_text())
        def fresh():
            if state.get("phase") != "prepared" or state.get("attempts") != [] or state.get("deadline") is not None:
                raise LifecycleError("preflight requires prepared phase and zero attempts")
            return {"phase":"prepared", "attempts":0}
        check("journal", fresh)
        plans = []
        def read_plan():
            plans.append(json.loads((directory / "plan.json").read_text()))
            return {"read":True}
        check("plan_read", read_plan)
        plan = plans[0]
        def binding():
            actual = digest(encoded(plan))
            if actual != approval or plan["directory"] != str(directory):
                raise LifecycleError("approval or run path mismatch")
            report.update(plan_sha256=actual, job_id=plan["job_id"], run_id=plan["run_id"])
            return {"plan_sha256":actual,"job_id":plan["job_id"]}
        check("approval", binding)
        check("controller_integrity", installation_integrity)
        check("provider_dependencies", dependency_report)
        check("prepared_files", lambda: {"count":len(verified_inputs(directory,plan["files"]))})
        check("frozen_job", lambda: _manifest_identity(directory,plan))
        report["hardware_policy"] = plan.get("hardware_policy", "legacy_model_check_unavailable")
        if validator:
            check("consumer_binding", lambda: validator(plan) or {"verified":True})
        def disk():
            # Chunks + reconstructed file + copy fallback; retain one additional snapshot reserve.
            required = 4 * SNAPSHOT_BYTES + RESULT_METADATA_BYTES
            free = shutil.disk_usage(directory).free
            report["disk"] = {"free_bytes":free,"required_bytes":required,
                              "policy":"four_snapshot_reserve; ongoing growth is rechecked during recovery"}
            if free < required:
                raise LifecycleError("insufficient recovery disk")
            return report["disk"]
        check("recovery_disk", disk)
        report["network_accessed"] = True
        def sign_in():
            nonlocal provider
            provider = provider_for_plan(plan)
            return {"authenticated":True}
        check("colab_sign_in", sign_in)
        # Provider and credential objects are never returned by check() into public reports.
        def assignments():
            items = provider.assignments()
            endpoints = [str(getattr(item,"endpoint", "")) for item in items]
            allowed = list(allow_active_assignments)
            if any(not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,240}", endpoint) for endpoint in endpoints + allowed):
                raise LifecycleError("invalid assignment identifier")
            report["assignments"] = {"observed":endpoints,"explicitly_allowed":allowed}
            if set(allowed) - set(endpoints) or set(endpoints) - set(allowed):
                raise LifecycleError("existing assignments need exact explicit override")
            return report["assignments"]
        check("active_assignments", assignments)
        token = TokenLease(lambda:get_token(plan["token_command"]))
        def access_check():
            access = token()
            if plan.get("authorization_policy") != "renewable-v1" and access["expires_at"] < time.time()+plan["max_seconds"]+60:
                raise LifecycleError("nonrenewable access must cover the entire run")
            return {"expires_at":access["expires_at"],"seconds_left":access["expires_at"]-time.time()}
        check("drive_access", access_check)
        check("drive_parent", lambda:drive_parent(plan["parent"],token()))
        if close_provider:
            def close_client():
                provider.close()
                return {"closed":True}
            check("client_cleanup", close_client)
        report["state"] = "passed"
        persist()
        return report, provider, token
    except BaseException as exc:
        report.update(state="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed", failed_check=current,
                      remediation=_REMEDIATION.get(current,"Inspect the prepared job and reinstall the verified wheel if needed."))
        if provider is not None:
            try:
                provider.close()
            except Exception:
                report["client_cleanup"] = "failed"
        target = persist()
        if isinstance(exc,KeyboardInterrupt):
            raise
        raise PreflightError("preflight " + current + " failed; " + report["remediation"] + " Report: " + str(target)) from None


def _manifest_identity(directory, plan):
    frozen_job(directory,plan)
    return {"job_id":plan["job_id"]}


_REMEDIATION = {
    "journal":"Use a prepared journal with zero attempts; reconnect or inspect an existing run without resetting it.",
    "approval":"Use the approval printed for this exact prepared run and the same run directory.",
    "controller_integrity":"Install the built development wheel; editable or modified installations are not launch-verified.",
    "provider_dependencies":"Install the pinned colab extra from the development wheelhouse.",
    "active_assignments":"Inspect the reported assignments; explicitly allow their IDs only if concurrent compute is intended.",
    "colab_sign_in":"Run colabcola lifecycle login in this controller environment, then retry.",
    "drive_access":"Renew host Drive authorization; the helper must return more than 600 seconds of access.",
    "drive_parent":"Select an owned, untrashed Drive folder that permits new children.",
    "recovery_disk":"Free the reported recovery reserve on the journal filesystem, then retry.",
}
