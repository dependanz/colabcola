# v33k — single-runtime H.264 credit A/B/A′ sweep (offline implementation)

**Status: allocation-free implementation, not yet wired to the managed Colab CLI.**
The verified v33g T4 stream achieved approximately 5.8 decoded FPS with a
single credit; v33j achieved approximately 10.6 decoded FPS with two credits,
with no reported dependency skips and confirmed owned-runtime release. These
were separate Colab allocations, so that comparison is suggestive, not a
controlled causal estimate.

`bridge_video_credit_sweep.run_h264_credit_sweep` prepares a three-phase
**1-credit → 2-credit → 1-credit** measurement in the *same* owned kernel Comm:

- Exactly one host decoder and one `KernelH264PullWindow` persist across phases.
- Pull IDs, source encoder state, sequence gating and input numbering do not reset.
- The receiver window always has a hard maximum of two; the per-phase sender
  enforces the lower cap when the phase calls for one credit.
- At each phase boundary, the sender stops issuing requests and drains every
  pending or out-of-order buffered reply **before** changing the cap.
- Each phase excludes its first two seconds from measurement by default.
  Transition/drain time is also excluded. The normal phase duration is 16 s.
- Every request is unique and never replayed after an uncertain response.
  The existing owned-runtime cleanup path remains responsible for release.

The per-phase report includes decoded FPS, payload H.264/CCV1/Base64 application
rates, pull RTT medians, high-resolution host decode+RGB time, locally observed
input ACK, dependency suppressions, maximum outstanding and longest period with
no newly decoded output. No metric is input-to-photon latency or total network
traffic. The phases are not identical source material, and renderer dynamics,
CPU/Colab variability, sampling, and temporal drift remain confounders.

## Next integration stage

Add a separate `--benchmark-h264-credits` CLI option requiring:
`--transport comm --video-codec h264 --inflight 1` (one-credit startup),
`--fps 12 --poll-hz 12`, an explicit confirmed owned T4 allocation, and a
session time budget large enough to finish all three phases plus bootstrap.
A single authenticated Comm and owned T4 must be used, and the existing PPM,
ordinary H.264 one/two-credit modes and batch runner must remain unchanged.
Perform fake-provider lifecycle/cleanup regression before any real GPU trial.

## Local verification (no Colab allocation)

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m pytest -q -rs tests/test_bridge_video_credit_sweep.py `
    tests/test_bridge_video_comm_pipeline.py `
    tests/test_bridge_video_comm_session.py
```

Then run the full bridge test suite with `Get-ChildItem tests -Filter
"test_bridge*.py"` to resolve PowerShell paths before invoking pytest.
