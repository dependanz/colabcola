"""Owned kernel connections and genuine tracked workload execution."""
from __future__ import annotations
import logging
import re
import time
from .lifecycle_io import LifecycleError


def sanitized_exception_class(value):
    """Retain only a bounded exception class identifier, never its message."""
    if type(value) is str and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", value):
        return value
    return None


def safe_client_class(base):
    class OwnedClient(base):
        def __init__(self, **kwargs):
            # Upstream assigns these only after manager construction, which can
            # fail on a network timeout. Its destructor then calls stop().
            self._manager = None
            self._own_kernel = False
            super().__init__(**kwargs)
            self._own_kernel = False

        def stop(self, **kwargs):
            manager = self._manager
            if manager is None:
                return
            try:
                # Closing a control connection must never release compute.
                super().stop(**{**kwargs, "shutdown_kernel": False})
            finally:
                # manager.client is a lazy factory, not a nullable attribute.
                # Failed startup can leave no kernel; asking for its client then
                # raises RuntimeError and masks the original connection failure.
                if getattr(manager, "has_kernel", True):
                    socket = getattr(manager.client, "kernel_socket", None)
                    if socket is not None:
                        socket.close()
            self._manager = None  # Idempotent explicit teardown and destructor.

        def __del__(self):
            try:
                self.stop()
            except Exception as exc:
                # Preserve diagnostics without leaking URL tokens in tracebacks.
                logging.getLogger(__name__).warning(
                    "Kernel connection cleanup failed (%s)", type(exc).__name__)
    return OwnedClient


def execute(session, code, kernel_id):
    from ._vendor import jupyter_kernel_client as jkc
    import requests
    if hasattr(jkc, "ColabKernelClient"):
        base, token_key = jkc.ColabKernelClient, "proxy_token"
    elif hasattr(jkc, "KernelClient"):
        base, token_key = jkc.KernelClient, "token"
    else:
        base, token_key = jkc.JupyterKernelClient, "token"
    client_type = safe_client_class(base)
    for attempt in range(3):
        client = None
        failure = None
        try:
            client = client_type(server_url=session.url, **{token_key: session.token},
                kernel_id=kernel_id,
                client_kwargs={"subprotocol": jkc.JupyterSubprotocol.DEFAULT,
                               "extra_params": {"colab-runtime-proxy-token": session.token}},
                headers={"X-Colab-Client-Agent": "colab-cli",
                         "X-Colab-Runtime-Proxy-Token": session.token})
            client.start()
        except Exception as exc:
            failure = exc
        if failure is not None:
            if client is not None:
                try:
                    _close(client)
                except LifecycleError as exc:
                    raise LifecycleError("runtime kernel connection failed: " +
                        type(failure).__name__ + "; " + str(exc)) from None
            if isinstance(failure, (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout)) and attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            raise LifecycleError("runtime kernel connection failed: " + type(failure).__name__) from None
        execution_failure = None
        try:
            # Never retry execution: a lost reply may follow a successful boot.
            reply = client.execute(code, allow_stdin=False, timeout=120)
            if not reply or reply.get("status") != "ok" or any(
                    item.get("output_type") == "error" for item in reply.get("outputs", [])):
                raise LifecycleError("runtime bootstrap failed")
        except LifecycleError as exc:
            execution_failure = str(exc)
        except Exception as exc:
            execution_failure = "runtime kernel execution failed: " + type(exc).__name__
        finally:
            try:
                _close(client)
            except LifecycleError as exc:
                if execution_failure:
                    raise LifecycleError(execution_failure + "; " + str(exc)) from None
                raise
        if execution_failure:
            raise LifecycleError(execution_failure) from None
        return


def _close(client):
    try:
        client.stop()
    except Exception as exc:
        raise LifecycleError("runtime kernel cleanup failed: " + type(exc).__name__) from None


class WorkloadExecution:
    """One real execute_request; reconnecting never submits the workload again."""
    def __init__(self, session, kernel_id):
        from ._vendor import jupyter_kernel_client as jkc
        base = getattr(jkc, "KernelClient", None) or jkc.JupyterKernelClient
        self.client = safe_client_class(base)(server_url=session.url, token=session.token,
            kernel_id=kernel_id,
            client_kwargs={"subprotocol": jkc.JupyterSubprotocol.DEFAULT,
                           "extra_params": {"colab-runtime-proxy-token": session.token}},
            headers={"X-Colab-Client-Agent": "colab-cli",
                     "X-Colab-Runtime-Proxy-Token": session.token})
        self.request_id = None
        self.state = "attached"
        self.reply_error_type = None
        self.kernel_id = kernel_id
        try:
            self.client.start()
        except Exception as exc:
            _close(self.client)
            raise LifecycleError("workload kernel connection failed: " + type(exc).__name__) from None

    def submit(self, code):
        if self.state != "attached":
            raise LifecycleError("workload execution must not be replayed")
        self.state = "submission_unknown"  # A lost send acknowledgement is never retried.
        try:
            self.request_id = self.client._manager.client.execute(
                code, allow_stdin=False, store_history=False)
        except Exception as exc:
            raise LifecycleError("workload submission uncertain: " + type(exc).__name__) from None
        self.state = "submitted"

    def poll(self):
        import queue
        wire = self.client._manager.client
        if not wire.connection_ready.is_set():
            raise LifecycleError("workload kernel connection lost")
        # No output text or remote exception values enter the public journal.
        for channel in (wire.iopub_channel, wire.shell_channel):
            for _ in range(64):
                try:
                    message = channel.get_msg(timeout=0)
                except (queue.Empty, TimeoutError):
                    break
                if self.request_id and message.get("parent_header", {}).get("msg_id") != self.request_id:
                    continue
                kind = message.get("header", {}).get("msg_type")
                content = message.get("content", {})
                if kind == "status" and content.get("execution_state") == "busy":
                    self.state = "busy"
                elif kind == "execute_reply":
                    self.state = "completed" if content.get("status") == "ok" else "failed"
                    self.reply_error_type = (sanitized_exception_class(content.get("ename"))
                                             if self.state == "failed" else None)
        result = {"mode": "kernel-v1", "kernel_id": self.kernel_id,
                  "request_id": self.request_id, "request_state": self.state}
        if self.state == "failed" and getattr(self, "reply_error_type", None):
            result["reply_error_type"] = self.reply_error_type
        return result

    def close(self):
        _close(self.client)  # Never interrupt, shutdown or release compute here.


def workload_code(remote, identity):
    """Execute the real worker in the kernel for its complete lifetime."""
    return ("import pathlib,json,time,runpy,sys,os\n"
        "def _colabcola_workload():\n"
        " p=pathlib.Path(" + repr(remote) + ")\n"
        " identity=" + repr(identity) + "\n"
        " receipt=p/'kernel-execution.json'\n"
        " if receipt.exists(): raise RuntimeError('owned workload already started;no replay')\n"
        " def record(**fields):\n"
        "  value={**identity,**fields};tmp=receipt.with_suffix('.tmp')\n"
        "  tmp.write_text(json.dumps(value));os.replace(tmp,receipt)\n"
        " started=time.time()\n"
        " record(state='running',started_at=started)\n"
        " old_argv=sys.argv;old_path=list(sys.path)\n"
        " try:\n"
        "  sys.argv=[str(p/'lifecycle_worker.py'),str(p/'worker.json')]\n"
        "  sys.path.insert(0,str(p))\n"
        "  runpy.run_path(str(p/'lifecycle_worker.py'),run_name='__main__')\n"
        " except BaseException as exc:\n"
        "  record(state='failed',started_at=started,finished_at=time.time(),error_type=type(exc).__name__)\n"
        "  raise\n"
        " else: record(state='finished',started_at=started,finished_at=time.time())\n"
        " finally: sys.argv=old_argv;sys.path[:]=old_path\n"
        "_colabcola_workload()\n")
