"""Pinned Google Colab provider. No interactive authorization during supervision."""
from __future__ import annotations
import base64
from datetime import datetime, timezone
import importlib.resources
import json
import os
from pathlib import Path
import uuid

from .lifecycle_io import LifecycleError, atomic, digest, verified_inputs
from ._version import __version__

SCOPES = ["openid", "https://www.googleapis.com/auth/userinfo.email",
          "https://www.googleapis.com/auth/colaboratory"]
UPSTREAM_COMMIT = "79cd126383e72b1e911f20b7030e4247f7861704"


def verify_provider():
    expected = {
        "client.py": "7081d2c9decbc9a4cafb4ab89202c7e7a644f72bd74a7e846bffce5344839a06",
        "runtime.py": "e95adf8dc98c9a08dce1fff035839863cbe80c9571d36634c3bc3d0cfcf06bcb",
        "state.py": "563232b3b0bc201245e9c35a79df052ead53b6b8c90348aebe3fe8d1d23c83d3",
    }
    try:
        package = importlib.resources.files("colab_cli")
    except ModuleNotFoundError:
        raise LifecycleError(f"Colab provider is not installed. Install the development wheelhouse with: python -m pip install --find-links /path/to/wheelhouse 'colabcola[colab]=={__version__}' (Python 3.12+ on Linux/WSL); see DEVELOPMENT.md") from None
    for name, sha in expected.items():
        if digest(package.joinpath(name).read_bytes().replace(b"\r\n", b"\n")) != sha:
            raise LifecycleError("Google provider differs from inspected commit " + UPSTREAM_COMMIT)


KERNEL_SOURCE_HASHES = {'client.py': 'd666344ceffd80d6d18a072e5a738069066e32471d97c238344edc108b8fda98', 'manager.py': '1006dd646953741cec569185075df88bb306efeb50e418a2de1ff44d972feecb', 'wsclient.py': '1a4456403b972990216012e079c5da0ff8a7c69fd3fcb72b19c20d74bd460bb1', 'utils.py': '98ffaef260265d0ced4cd7b849f5d9374515027acbe7a8db7b6feb614d7c49d4', 'constants.py': '63f874790a6c4cb8de5a4c4c08393e8ea91ddb4fbac2b1247993fe9dda19cb5e', 'models.py': '0ca21a2b00d1d2b657f52bd7113eee1cc1474094fb8b121eabb72b4b84b498fb'}


def dependency_report():
    """Offline installation checks: no credential reads or provider construction."""
    import importlib.metadata
    verify_provider()
    try:
        package = importlib.resources.files("jupyter_kernel_client")
        for name, expected in KERNEL_SOURCE_HASHES.items():
            if digest(package.joinpath(name).read_bytes().replace(b"\r\n", b"\n")) != expected:
                raise LifecycleError("Jupyter provider differs from inspected fork f18e982c; reinstall the development wheelhouse colab extra")
        from colab_cli.client import Client, Prod
        from colab_cli.state import SessionState
        from jupyter_kernel_client import KernelClient, JupyterSubprotocol
    except ImportError:
        raise LifecycleError("Provider dependencies are incomplete; reinstall the development wheelhouse colab extra and run python -m pip check") from None
    return {"provider_source_verified": True, "google_colab_cli": importlib.metadata.version("google-colab-cli"),
            "jupyter_kernel_client": importlib.metadata.version("jupyter-kernel-client"),
            "upstream_commit": UPSTREAM_COMMIT, "kernel_commit": "f18e982c3265df5e923aa9def101ab3fd737e139"}


def auth_path():
    return Path.home() / ".config/colabcola/colab-oauth.json"


def login():
    dependency_report()
    from google_auth_oauthlib.flow import InstalledAppFlow
    config = json.loads(importlib.resources.files("colab_cli").joinpath("oauth_config.json").read_text())
    flow = InstalledAppFlow.from_client_config(config, SCOPES)
    flow.redirect_uri = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"
    url, _ = flow.authorization_url(prompt="consent", token_usage="remote")
    print("Open this Google sign-in URL, then paste the code into this terminal:\n" + url)
    flow.fetch_token(code=input("Authorization code: ").strip())
    path = auth_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic(path, json.loads(flow.credentials.to_json()))
    os.chmod(path, 0o600)
    print("Colab authorization saved. No runtime was allocated.")


def credentials():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    path = auth_path()
    if not path.is_file():
        raise LifecycleError("Colab login required: colabcola lifecycle login")
    if os.name != "nt" and path.stat().st_mode & 0o077:
        raise LifecycleError("Colab credential file must have mode 0600")
    try:
        credential = Credentials.from_authorized_user_file(str(path), SCOPES)
        if not credential.valid:
            credential.refresh(Request())
        if not credential.valid or not credential.has_scopes(SCOPES):
            raise ValueError()
        atomic(path, json.loads(credential.to_json()))
        os.chmod(path, 0o600)
        return credential
    except Exception:
        raise LifecycleError("Colab credentials unavailable; run lifecycle login") from None


class GoogleProvider:
    high_mem = False
    """Uses the inspected upstream client; tokens never enter the supervisor journal."""
    def __init__(self):
        dependency_report()
        from colab_cli.client import Client, Prod
        from google.auth.transport.requests import AuthorizedSession
        verify_provider()
        os.environ["COLAB_CLI_REQUEST_TIMEOUT"] = "40"
        os.environ["COLAB_CLI_ASSIGN_RETRIES"] = "0"
        self.client = Client(Prod(), AuthorizedSession(credentials()))
        # Upstream debug logging includes authorization headers.
        self.client.logger.disabled = True

    def assignments(self):
        return self.client.list_assignments()

    def lookup(self, endpoint):
        return next((item for item in self.assignments() if item.endpoint == endpoint), None)

    def allocate(self, allocation_id, gpu):
        from colab_cli.client import Accelerator, Variant, resolve_assign_shape
        # Persist allocation_id BEFORE calling. Retrying uses the same notebook UUID;
        # upstream reconciles an uncertain POST with a GET before repeating allocation.
        assignment = self.client.assign(uuid.UUID(allocation_id),
                                        variant=Variant.GPU if gpu != "NONE" else Variant.DEFAULT,
                                        accelerator=Accelerator(gpu),
                                         shape=resolve_assign_shape(Accelerator(gpu), high_mem=self.high_mem))
        return assignment.endpoint

    def reconcile(self, allocation_id, gpu):
        from colab_cli.client import Accelerator, Variant, Assignment, resolve_assign_shape
        result = self.client._get_assignment(
            uuid.UUID(allocation_id),
            Variant.GPU if gpu != "NONE" else Variant.DEFAULT, Accelerator(gpu),
            resolve_assign_shape(Accelerator(gpu), high_mem=self.high_mem))
        return result.endpoint if isinstance(result, Assignment) else None

    def release(self, endpoint):
        # The endpoint is supplied only from the durable owned-attempt journal.
        # A missing list entry does not prove the runtime has stopped.
        status = "accepted"
        try:
            self.client.unassign(endpoint)
        except Exception as exc:
            code = getattr(getattr(exc, "response", None), "status_code", None)
            if code != 404:
                raise LifecycleError("owned runtime release unavailable: " + type(exc).__name__) from None
            status = 404
        if self.lookup(endpoint) is not None:
            raise LifecycleError("owned Colab CLI runtime release not yet confirmed")
        return {"state": "absent", "unassign_status": status}

    def _session(self, endpoint):
        from colab_cli.state import SessionState
        assignment = self.lookup(endpoint)
        if assignment is None:
            raise LifecycleError("owned runtime is absent")
        info = assignment.runtime_proxy_info
        return SessionState(name="colabcola", endpoint=endpoint, token=info.token, url=info.url)

    def _contents(self, session, method, path, payload=None):
        import requests
        from urllib.parse import quote
        response = requests.request(
            method, session.url.rstrip("/") + "/api/contents/" + quote(path.strip("/"), safe="/"),
            params={"authuser": "0", "colab-runtime-proxy-token": session.token,
                    **({"content": "1"} if method == "GET" else {})},
            headers={"X-Colab-Runtime-Proxy-Token": session.token},
            json=payload, timeout=(10, 40))
        if response.status_code == 404:
            raise FileNotFoundError(path)
        if response.status_code >= 400:
            raise LifecycleError(f"runtime transfer HTTP {response.status_code}")
        return response.json()

    def _upload(self, session, path, body):
        self._contents(session, "PUT", path, {"name": path.rsplit("/", 1)[-1], "path": path,
            "type": "file", "format": "base64", "content": base64.b64encode(body).decode(), "chunk": 1})

    def _execute(self, session, code):
        from .lifecycle_kernel import execute
        import requests
        response = requests.get(session.url.rstrip("/") + "/api/kernels",
                                params={"colab-runtime-proxy-token": session.token},
                                headers={"X-Colab-Runtime-Proxy-Token": session.token},
                                timeout=(10, 40))
        if response.status_code >= 400:
            raise LifecycleError(f"runtime kernel listing HTTP {response.status_code}")
        kernels = response.json()
        # Reuse the owned runtime's existing kernel, including after supervisor restart.
        execute(session, code, kernels[0]["id"] if kernels else None)

    def _json_file(self, session, path):
        try:
            value = self._contents(session, "GET", path)
        except FileNotFoundError:
            return None
        body = base64.b64decode(value["content"]) if value.get("format") == "base64" else value["content"]
        return json.loads(body)

    def _workload_connection(self, endpoint, session, kernel_id):
        from .lifecycle_kernel import WorkloadExecution
        connections = getattr(self, "_workloads", {})
        self._workloads = connections
        if endpoint not in connections:
            connections[endpoint] = WorkloadExecution(session, kernel_id)
        if connections[endpoint].kernel_id != kernel_id:
            raise LifecycleError("owned workload kernel changed;no replacement allowed")
        return connections[endpoint]

    def close(self):
        failures = []
        for connection in getattr(self, "_workloads", {}).values():
            try:
                connection.close()
            except Exception as exc:
                failures.append(type(exc).__name__)
        self._workloads = {}
        session = getattr(getattr(self,"client",None),"session",None) or getattr(self,"http",None)
        if session is not None:
            try:
                session.close()
            except Exception as exc:
                failures.append(type(exc).__name__)
        if failures:
            raise LifecycleError("workload connection cleanup failed: " + ",".join(failures))

    def _kernel(self, session, kernel_id=None):
        import requests
        response = requests.get(session.url.rstrip("/") + "/api/kernels",
            params={"colab-runtime-proxy-token": session.token},
            headers={"X-Colab-Runtime-Proxy-Token": session.token},timeout=(10,40))
        if response.status_code >= 400:
            raise LifecycleError(f"runtime kernel listing HTTP {response.status_code}")
        kernels = response.json()
        kernel = next((k for k in kernels if kernel_id is None or k["id"] == kernel_id), None)
        if kernel is None:
            raise LifecycleError("owned workload kernel is absent;no replacement allowed")
        return kernel

    def _execution_identity(self, plan, attempt, kernel_id):
        if not isinstance(kernel_id, str) or not kernel_id:
            raise LifecycleError("owned workload kernel identity missing")
        return {"run_id": plan["run_id"], "job_id": plan["job_id"],
                "attempt": attempt["number"], "kernel_id": kernel_id}

    def boot(self, endpoint, plan, attempt, token):
        session = self._session(endpoint)
        tracked = plan.get("execution_mode") == "kernel-v1"
        if tracked:
            remote = f"content/colabcola-control/{plan['run_id']}/{attempt['number']}"
            receipt = self._json_file(session, remote + "/kernel-execution.json")
            intent = self._json_file(session, remote + "/kernel-request.json")
            if receipt or intent:
                if not receipt:
                    raise LifecycleError("workload submission uncertain;no replay allowed")
                identity = self._execution_identity(plan, attempt, receipt.get("kernel_id"))
                if (any(receipt.get(k) != v for k,v in identity.items())
                        or intent is None or any(intent.get(k) != v for k,v in identity.items())):
                    raise LifecycleError("workload execution identity mismatch")
                kernel = self._kernel(session, receipt["kernel_id"])
                self._workload_connection(endpoint, session, kernel["id"])
                return
        remote = f"/content/colabcola-control/{plan['run_id']}/{attempt['number']}"
        self._execute(session, "import pathlib\npathlib.Path(" + repr(remote) +
                      ").mkdir(parents=True, exist_ok=True, mode=0o700)\n")
        local = Path(plan["directory"])
        paths = verified_inputs(local, plan["files"])
        for name in paths:
            if name == "notebook.ipynb" and plan.get("input_transport") == "drive_chunks":
                continue
            body = paths[name].read_bytes()
            if digest(body) != plan["files"][name]:
                raise LifecycleError("prepared source changed")
            self._upload(session, remote.lstrip("/") + "/" + name, body)
        configuration = {key: plan[key] for key in
                         ("run_id", "job_id", "parent", "checkpoint_pattern", "log_patterns", "files")}
        for key in ("artifact_groups", "artifact_output_rules", "hardware_policy", "gpu"):
            if key in plan:
                configuration[key] = plan[key]
        if plan.get("input_transport") == "drive_chunks":
            configuration["input_snapshot"] = attempt["input_snapshot"]
        if plan.get("authorization_policy") == "renewable-v1":
            configuration["authorization_policy"] = "renewable-v1"
        configuration.update(attempt=attempt["number"], deadline=attempt["deadline"],
                             resume=attempt.get("resume"), directory=remote,
                             allocation_id=attempt.get("allocation_id"), endpoint=endpoint)
        self._upload(session, remote.lstrip("/") + "/worker.json",
                     json.dumps(configuration).encode())
        self._upload(session, remote.lstrip("/") + "/drive-token.json", json.dumps(token).encode())
        if tracked:
            from .lifecycle_kernel import workload_code
            kernel = self._kernel(session)
            identity = self._execution_identity(plan, attempt, kernel["id"])
            connection = self._workload_connection(endpoint, session, kernel["id"])
            self._upload(session, remote.lstrip("/") + "/kernel-request.json", json.dumps(identity).encode())
            connection.submit(workload_code(remote, identity))
        else:
            # Historical prepared plans retain their original execution semantics.
            self._execute(session, "import subprocess,sys,pathlib\np=pathlib.Path(" + repr(remote) + ")\n"
                          "log=(p/'bootstrap.log').open('ab')\n"
                          "subprocess.Popen([sys.executable,str(p/'lifecycle_worker.py'),str(p/'worker.json')],"
                          "stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)\n")

    def refresh_token(self, endpoint, plan, attempt, token):
        session = self._session(endpoint)
        remote = f"content/colabcola-control/{plan['run_id']}/{attempt['number']}"
        value = {key: token[key] for key in ("access_token", "expires_at")}
        self._upload(session, remote + "/drive-token.next.json", json.dumps(value).encode())
        if plan.get("execution_mode") == "kernel-v1":
            return  # Runtime atomically validates/adopts pending access while its kernel is busy.
        self._execute(session, "import pathlib,os\np=pathlib.Path('/" + remote + "')\n"
                      "os.replace(p/'drive-token.next.json',p/'drive-token.json')\n")

    def observe(self, endpoint, plan, attempt, token):
        if plan.get("authorization_policy") != "renewable-v1":
            self.refresh_token(endpoint, plan, attempt, token)
        session = self._session(endpoint)
        remote = f"content/colabcola-control/{plan['run_id']}/{attempt['number']}"
        try:
            value = self._contents(session, "GET", remote + "/status.json")
        except FileNotFoundError:
            return {"state": "starting"}
        body = base64.b64decode(value["content"]) if value.get("format") == "base64" else value["content"]
        observed = json.loads(body)
        if plan.get("execution_mode") == "kernel-v1":
            receipt = self._json_file(session, remote + "/kernel-execution.json")
            intent = self._json_file(session, remote + "/kernel-request.json")
            record = receipt or intent
            if not record:
                raise LifecycleError("owned workload execution receipt missing")
            identity = self._execution_identity(plan, attempt, record.get("kernel_id"))
            if (any(record.get(k) != v for k,v in identity.items())
                    or intent is None or any(intent.get(k) != v for k,v in identity.items())):
                raise LifecycleError("workload execution identity mismatch")
            kernel = self._kernel(session, record["kernel_id"])
            connection = self._workload_connection(endpoint, session, kernel["id"])
            try:
                evidence = connection.poll()
            except LifecycleError:
                connection.close()
                self._workloads.pop(endpoint, None)
                connection = self._workload_connection(endpoint, session, kernel["id"])
                evidence = connection.poll()  # Reattach only;never send execute again.
            observed["kernel_execution"] = {**evidence,"execution_state":kernel.get("execution_state"),
                "worker_execution_state":(receipt or {}).get("state", "submitted")}
        return observed


def provider_for_plan(plan):
    """Select only the frozen backend; legacy never imports the beta REST adapter."""
    backend = plan.get("provider_backend", "legacy")
    if backend == "legacy":
        if plan.get("runtime_version", "latest") != "latest":
            raise LifecycleError("Colab CLI adapter cannot select a dated runtime; the job must not silently use Latest")
        provider = GoogleProvider()
        provider.high_mem = plan.get("high_mem", False)
        return provider
    if backend == "rest":
        from .lifecycle_rest import RestGoogleProvider
        return RestGoogleProvider(plan)
    raise LifecycleError("unknown prepared provider backend")
