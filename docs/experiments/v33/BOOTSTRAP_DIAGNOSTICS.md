# v33f: safe remote H.264 bootstrap diagnostics

Status: **offline diagnostic hardening only**. No remote GPU is allocated or
remote Python dependencies installed by this change.

The first managed v33e H.264 attempt failed during `creating_comm` and reported
`RemoteException`; its owned T4 was released and cleanup confirmed absence.
This is compatible with remote PyAV/NumPy/libx264 unavailability but does not
prove a cause. The `transport` journal field was absent because `running` was
never reached.

In H.264 mode, the optional encoder still initializes before the handshake is
accepted. A failure now opens an authenticated, run-ID-scoped Comm carrying a
sanitized **`video_preflight_failed`** message instead of allowing its exception
to be collapsed into a generic kernel error. The host checks the run identity,
T4 handshake, dimensions and Comm ID, then raises an allowlisted diagnostic,
closes the Comm and lets the existing run journal reconcile/release its owned T4.
A failure to open Comm itself still uses the established bootstrap error path.

Possible diagnostic codes are:

- `pyav_unavailable` — Python package `av` was not importable remotely
- `numpy_unavailable` — Python package `numpy` was not importable remotely
- `libx264_unavailable` — remote PyAV could not create or open `libx264`
- `source_path_missing` — the H.264 bootstrap source path was not supplied
- `video_initialization_failed` — unexpected source initialization failure
- `unknown` — an unrecognized diagnostic was received and suppressed by host

Codes are determined only from the exact fixed errors in Colab Cola's own
remote source. Exception values, arbitrary paths, tokens and tracebacks are
never sent in the new failure handshake. This does not establish whether Colab
has remote PyAV/NumPy/libx264 without another explicitly confirmed allocation.

PPM remains the default, and no video mode is enabled unless explicitly
selected. No change to batch work, released package requirements, or provider
allocation/cleanup semantics.

## Offline tests

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m pytest -q -rs tests/test_bridge_video_preflight_diagnostics.py tests/test_bridge_video_comm_session.py tests/test_bridge_video_colab_source.py tests/test_bridge_comm.py tests/test_bridge_colab.py tests/test_bridge_comm_local_kernel.py
```

Do not run another paid T4 allocation solely because this patch passes; review
its complete regression output and decide whether additional paid diagnosis is
worthwhile. The existing `bridge cleanup` command is sufficient to reconcile
only the owned allocation if an allocation occurs and its release is uncertain.
