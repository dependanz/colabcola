# v33l — opt-in managed single-runtime H.264 credit sweep

**Status: offline integration candidate. No new Tesla T4 allocation is implied.**

This adds an explicit `--benchmark-h264-credits` mode to the existing owned
managed-T4 bridge. It dispatches the v33k `run_h264_credit_sweep` worker with
one host decoder, one kernel Comm, one remote persistent encoder, and continuous
pull/input numbering across 1 → 2 → 1 credit phases. Pending and buffered
replies must fully drain before phase changes. The sweep owns no provider
allocation; the existing journal and release routine retain that responsibility.

## Guardrails

- Requires `--transport comm --video-codec h264 --inflight 1 --fps 12 --poll-hz 12`.
- Cannot combine with `--benchmark-sweep`, `--benchmark-credits`, or `--probe-link`.
- Phase duration defaults to 16 s, allowed range 6–40 s; warmup remains 2 s.
- Session time budget must be at least `3 × phase_seconds + 100` seconds.
- A fresh run directory and `--confirm-allocation` are still mandatory for
  any real managed GPU run. Remote PyAV installation remains separately opt-in.
- Standard PPM/file transport, single-credit H.264, and two-credit H.264
  choices are unchanged.

The controlled sweep reduces *between-allocation* variation but does not fully
isolate causal effects of credit count: phases occur at different times, and
video contents and workload conditions can change. Its ACK metric is not
input-to-photon, and application bytes exclude network/WebSocket overhead.

## Allocation-free verification

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m pytest -q -rs tests/test_bridge_video_credit_sweep_managed.py `
    tests/test_bridge_video_credit_sweep.py `
    tests/test_bridge_video_comm_session.py `
    tests/test_bridge_colab.py

& $py -m colabcola --profile pypi-new-account-test bridge colab `
    --transport comm --video-codec h264 --inflight 1 --fps 12 --poll-hz 12 `
    --benchmark-h264-credits --h264-credit-phase-seconds 16 --preflight-only
```

Do not perform a paid T4 trial until full Windows regression and the preflight
succeed. If a later trial is explicitly approved, preserve its terminal output,
check `phase: released`, and use `bridge cleanup` for the exact owned allocation
if release cannot be confirmed.
