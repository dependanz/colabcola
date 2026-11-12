# v33e: opt-in managed H.264 Comm integration (offline validation first)

The new `colabcola bridge colab --transport comm --video-codec h264 --inflight 1`
route reuses the existing owned Jupyter kernel, authenticated Comm ID, remote
CUDA renderer, numbered `video_pull`, strict CCV1 receiver and Tk viewer.

**Status:** v33e adds an experimental managed remote-video route, **not live
T4 verified**. Do not claim any provider FPS, latency, throughput or codec
availability until a separately authorized, supervised trial has completed.

Defaults remain `--video-codec ppm`: the existing file-polling and Comm PPM
paths and two-credit default for ordinary Comm remain unchanged. Video is
allowed only with one outstanding pull, without benchmark or link-probe modes.
The owned allocation lifecycle, local journal and explicit confirmation are
unchanged. An unrecognized/missing remote PyAV/libx264 encoder fails bootstrap
before video `comm_open` and must release the owned allocation. No automatic
remote pip installs, new internet listeners, tunneling or public endpoints.

The local PyAV H.264 decoder is checked **before allocation**; that local
preflight does **not** establish remote codec availability. Remote codec
initialization is requested before declaring video Comm ready.

The remote source keeps a persistent CPU libx264 encoder. Each pull encodes
one most-recent CUDA PPM snapshot (not direct arbitrary GPU framebuffer
capture), and fresh video sequence is distinct from CUDA renderer sequence.
The v33c receiver gates out-of-order/dependent access units until an IDR.
Remote `capture_ns` belongs to the Colab kernel clock and is **never**
subtracted from host times. `video pull RTT` is request-to-response, the
`local decode` metric is host-only, input ACK is applied-input acknowledgement,
and `GUI configured` is not actual monitor scanout or input-to-photon latency.

The receiver publishes no frame when dependent frames are suppressed, so the
GUI holds its last image. Future work should add explicit IDR demand and
freshness-aware sender policy only after transport measurements.

## Allocation-free validation

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m pytest -q -rs tests/test_bridge_video_comm_session.py tests/test_bridge_video_colab_source.py tests/test_bridge_video_comm_wire.py tests/test_bridge_comm.py tests/test_bridge_colab.py tests/test_bridge_pipeline.py
& $python -m colabcola.cli --profile pypi-new-account-test bridge colab --transport comm --video-codec h264 --inflight 1 --preflight-only
```

No real T4 allocation or release claim is included in v33e's offline tests.
For an eventual authorized live run, use a **new** directory, explicit
`--confirm-allocation`, a bounded session, and verify the journal reports
`phase: released`. A failed/uncertain cleanup requires the existing
`bridge cleanup` command and ownership reconciliation before reuse.
