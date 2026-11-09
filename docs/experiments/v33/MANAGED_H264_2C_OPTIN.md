# v33j — opt-in managed H.264 two-credit selection

Status: **offline integration only**. The managed Colab route now permits
`--transport comm --video-codec h264 --inflight 2` explicitly, selecting the
v33i `stream_comm_h264_pipeline` worker. The default H.264 setting stays at
**one credit**; ordinary Comm PPM continues to default to two credits; file
polling and batch submission are unchanged.

The same owned Colab T4, run ID, authenticated Jupyter Comm ID, codec bootstrap,
`--install-remote-pyav` opt-in, journal and release routine are reused. No
additional sockets, GPUs, or package installs are introduced by two credits.
`video_pull` replies are correlated and committed by pull sequence before
H.264 decoder ingestion. A late/invalid/duplicate/unsolicited response fails
closed, never causes uncertain replay. The v33i window caps pending *plus*
out-of-order buffered replies at two.

**Caveat:** Two in-flight requests overlap host-to-kernel round trips, but
remote H.264 encoding currently occurs synchronously in the kernel Comm
callback. This does not guarantee doubled FPS or any improvement over v33g's
single-credit ~5.8 decoded FPS / ~172 ms pull RTT result. It may also change
ACK latency, CPU contention and frame freshness. Do not call local decode
cost or request RTT an input-to-photon measurement.

## Allocation-free tests

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m pytest -q -rs tests/test_bridge_video_comm_pipeline.py tests/test_bridge_video_comm_session.py
```

Then discover and run all `tests/test_bridge*.py` files using PowerShell
`Get-ChildItem` (wildcards are not expanded automatically for native commands).

## Authorized live-trial boundary

A real v33j T4 comparison remains **unperformed**. Only after full Windows
regression and preflight pass, and with separate user authorization, use a
new run directory, `--confirm-allocation`, explicit `--inflight 2`, and verify
`state.json` records `phase: released` after the run. Compare against the
single-credit v33g baseline using matched video configuration; results across
different runtime sessions are not a randomized or simultaneous comparison.
