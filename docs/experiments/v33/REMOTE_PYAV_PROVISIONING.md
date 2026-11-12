# v33g: explicit remote PyAV provisioning (opt-in)

**Status:** offline patch for the managed H.264 experiment. Remote successful
installation is not yet verified on an allocated Colab T4.

The v33f live diagnostic returned `pyav_unavailable`; the owned T4 was
confirmed released. This patch introduces a separate `--install-remote-pyav`
flag, valid **only** with `--transport comm --video-codec h264 --inflight 1`.
It never changes the default PPM route, local installation, batch jobs, or
owned allocation/cleanup lifecycle.

When the option is explicitly set for an allocated owned T4, the kernel checks
whether PyAV can be imported. If not, it attempts exactly one **wheel-only**
install from `https://pypi.org/simple` using the **remote kernel's own Python**:
`av==17.0.0`. Pip is isolated, uses no source builds or transitive dependency
resolution, and has a 40-second process timeout. Output is discarded to avoid
capturing package-manager metadata or credentials. The host bootstrap has an
extended bounded reply deadline only for this opt-in mode.

This operation **does use the runtime's network and can consume compute units**.
It does not guarantee PyPI egress, a compatible wheel, encoder availability,
or real-time performance. Remote NumPy is expected to already be present; the
installer does not alter NumPy or PyTorch. No automatic retries, fallbacks to
source compilation, or global host installation are attempted.

If pip fails or expires, the existing owned Jupyter Comm reports only fixed
`pyav_install_failed` or `pyav_install_timeout` codes and the managed owner
releases its allocation. The original `pyav_unavailable`, `numpy_unavailable`,
`libx264_unavailable`, and generic initialization diagnoses remain available.

## Offline checks

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m pytest -q -rs tests/test_bridge_video_remote_install.py tests/test_bridge_video_preflight_diagnostics.py tests/test_bridge_video_comm_session.py tests/test_bridge_colab.py tests/test_bridge_comm.py
& $py -m colabcola --profile pypi-new-account-test bridge colab --transport comm --video-codec h264 --inflight 1 --install-remote-pyav --preflight-only
```

The `--preflight-only` command never installs into a remote runtime or allocates
one. The next real T4 attempt still requires a new run directory and
`--confirm-allocation`, followed by a `phase: released` journal check.
