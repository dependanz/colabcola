# Primary non-REST workflow and revalidation

The current product priority is ordinary script/module/sequence packaging, unattended
Colab execution, progress/logs, automatic checkpoint/final recovery and confirmed
runtime release without the beta Colab REST API. The primary adapter is Google's
official CLI session client, retaining the configuration value `legacy` for compatibility.
Google sign-in, Drive's API and Jupyter transport remain in use. This is not an API-free
implementation. No allowlist workaround, account switching, browser keepalive or quota
evasion is introduced. REST adoption is deferred until non-beta availability is reviewed.

## Historical evidence audit

Sixteen managed plans were inspected within registered colabcola output resources.
Seven completed plans use legacy selection; zero completed plans use REST. Historical
plans without provider_backend select legacy by the preserved default. No journal was
rewritten. Evidence and plan hashes: the nonrest_revalidation resource's
historical-backends.json. This is an audit of existing evidence, not a new cloud run.

| Case | Existing evidence | Revalidation on current implementation |
| --- | --- | --- |
| Hyperplus external library / deliberate recovery | Legacy, completed two-runtime recovery plus earlier one-runtime fit | Current local recovery tests pass; bounded GPU replay remains planned |
| Fixed-image SIREN | Legacy, completed | Local ordinary-script packaging passes; GPU replay remains planned |
| Kodak grid / restart | Legacy, completed two-runtime pilot | Passed current general-packager replay: two T4s, step-50 deliberate restart/restore, 24 fits, 353.400854 seconds; cold Drive and release verified |
| Kodak standalone general packager | Legacy, completed | Standalone run.py live restart/recovery replay passed; module and train/evaluate local executions also pass |
| GaussianAvatar training and render-only sequence | Legacy, both completed | Fresh 5000-step training plus matched200/5000 renders passed on one T4; cold Drive and release verified. Historical render-only evidence retained. |
| Dated CPU renewal probe | REST, failed before creation | Distinct legacy/Latest CPU renewal probe passed on one runtime; no dated acceptance claim |
| Full 5 GiB artifacts | Local filesystem proof, no runtime provider | Preserved local full-size proof plus passing transfer regressions; large live Drive transfer remains unmeasured |
| Personal LongCat eight-case/kernel-v1 | Historical five-case jobs retained;dev9/dev10 interruptions preserved | Passed dev11 on one A10080GB:8/8 videos,9468.904660 controller seconds/9674.901152 through independent checks within10800;3 natural renewals/post-expiry publication,8 pre-final clip commits,cold hashes/full decode and owned-runtime absence verified. Log delivery remains42.076150s median/90.132747s maximum in146 samples. |

## First new live transaction — verified 2026-10-15

Resource: `nonrest_revalidation`, dated20261015. Job9eb367cd651d5f3db16fc470793047c613b8fa1aa1f769f15d2ffbd38f00b6c2.
Approval b1279fd01c54e6c30442760f3071940ec627e5b3b50d4553f2ebf8937bd9b913.
CPU only, Latest, one runtime maximum,4500 shared seconds,3900 seconds of workload,
4200-second workload timeout,128 MiB outputs. The user separately authorized this allowance. The single command completed in
4014.337087 seconds (66 minutes 54 seconds), including runtime release. Its journal
is terminal: do not rerun the historical launch command in `LIVE_PROBE.md`.

Verified: one genuine token change; authenticated publication 165.899118 seconds
past initial expiry; automatic recovery of the 69,261,233-byte artifact ZIP with
135 outputs; all seven final snapshot members verified again from fresh Drive bytes;
and independent absence of the exact owned runtime. The workload ran 3900.005573
seconds and wrote 66 checkpoints. Latest reported Python 3.13.16; this does not
establish dated-image selection. The provider was legacy; no beta Colab REST calls,
replacement runtime, account changes or vdm_tracking access occurred.

Evidence in the nonrest_revalidation resource: `probe-live-verification.json`,
`probe.run/probe-verification.json`, `probe.run/events.ndjson`, `probe-launch.log`
and `probe-verifier.log`. The earlier `verification.json` remains preparation evidence.

The Kodak and GaussianAvatar replays below have now passed. Next, prepare new immutable
external-library/SIREN transactions from the ordinary examples, under
separately explicit runtime/time bounds. Preserve original evidence. Do not launch
all workloads under an unspecified cumulative budget. A live5GiB transfer needs its
own storage and transfer allowance. Local regressions are not substitutes for CUDA,
natural token expiry or cloud transfer/release acceptance.

## Kodak restart replay — verified 2026-10-15

The ordinary standalone `examples/kodak_grid/run.py` was packaged with the general
CLI into a fresh job cdc8f0208/approval eb7b6c67. One native 768x512 image, seed 1234,
24 fits (HOSC/SIREN, depths 1/2/3, widths 32/64/128/256), 250 steps per fit.
The user authorized two T4 runtimes maximum and 900 shared seconds.

Passed in 353.400854 seconds. Both allocations were independently observed as T4s.
The first runtime was deliberately released only after checkpoint 50 was verified;
the distinct second restored the exact manifest and completed the grid. Checkpoint
Adam counters, sampling RNG and final loss prefix match exactly; the first pair
performed 200 resumed updates. All 12 paired initializations, 3000 finite paired loss
rows and 24 native-resolution reconstruction PNGs passed local checks.

Inputs, the proof and final checkpoints, and final artifacts passed fresh Drive
readback. The artifact ZIP is 7,095,736 bytes with 16 outputs. Both owned endpoints
were independently absent; no active assignments remained. A bootstrap ReadTimeout
recovered on the same second runtime. No additional runtime or vdm_tracking access.
The GPU image reported Python 3.13.15 and Torch 2.11.0+cu130; the earlier CPU probe
reported Python 3.13.16. The explicit Python range >=3.13,<3.14 accommodates these
patch differences without asserting a stable Latest image.

Evidence under `nonrest_revalidation/kodak-restart/`: `preparation-verification.json`,
`allocation-1.json`, `allocation-2.json`, `live-verification.json`,
`continuation-verification.json`, `output-verification.json`, `launch.log` and
`verified-preview/comparison.png`. Local checkpoint tests: 2/2 passed. No claim of
convergence, an activation winner or independent uninterrupted GPU bitwise replay.
The GaussianAvatar replay below has also passed. Remaining external-library/SIREN live replays need separate bounds.

## GaussianAvatar longer training replay — verified 2026-10-15

Fresh general-packager job1cdc386e/approval8b7f9d31 used six ordinary modules:
validate, build/CUDA checks, unchanged upstream train, render200, render5000,
and gaussian_avatar.compare. One T4 maximum and3600 shared seconds were authorized.
Ten local example tests and the real CPU loader passed before launch;1674 payload
files,77,686,833 raw bytes and80,988,850 frozen input bytes were verified.

Passed on one T4 in1038.611695 seconds (17m19s). Native setup took472.602037 seconds;
all three extension/gradient checks passed. Training completed5000 iterations with
4999 optimizer updates and10144 finite Gaussians. Saved checkpoints at200/1000/5000,
PLY/FLAME files, renders and comparisons were automatically recovered. Fresh Drive
readback verified3 input/11 final snapshot members and63 output files in the
19,144,851-byte ZIP. Independent lookup found the owned endpoint absent and zero
active assignments. No replacement runtime or vdm_tracking access occurred.

All eight matched training views improved from step200 to5000. Independently
recomputed aggregate RGB PNG PSNR rose23.9423735 to36.3857754dB; mean RGB MSE fell
94.3028217%, and non-white-mask MSE fell93.8638744%.41 PNGs and exact comparison
pixels passed checks. These are eight training views of one pose at256x373; no
held-out, animation or convergence claim. The default densification start10000 was
not reached. Unrequested upstream MP4 exports are empty because yuv420p rejects
the odd image height; the requested PNGs are valid.

Evidence under nonrest_revalidation/gaussian-quality/: preparation-verification.json,
allocation-1.json, live-verification.json, training-verification.json,
comparison-verification.json, launch.log and verified-output/outputs/comparison/.
The ordinary reusable recipe is examples/gaussian_avatar/quality-commands.json;
there is no workload-specific packaging helper.

UX findings retained for follow-up: training stdout/stderr arrived in delayed large
chunks despite regular build-log publication, and the upstream Jupyter client
emitted ignored missing-_manager destructor tracebacks. Supervision and exact
runtime release still passed. No live-code patch was attempted during this run.

## Dated-runtime and LongCat boundary

The pinned CLI assignment interface and current upstream interface inspected 2026-10-15
accept runtime identity, accelerator and host-memory shape, but no image date. Dated
requests remain rejected on this adapter; it never silently changes 2026.07 to Latest.
Explicit job-owned Python 3.12 isolation was subsequently implemented and verified
with compatible Torch/CUDA and flash-attention, then used for the separately authorized
LongCat run on 2026-10-16 below. This does not reproduce the dated base image.
The original non-REST preparation excluded vdm_tracking; the later live run was
separately authorized and its partial workload result is recorded below.

## Initial local acceptance (historical)

Native:230 tests,225 passed,5 skipped. WSL:230 tests,228 passed,2 optional dependency
skips. Eight new cases cover no-REST provider selection, invalid backend/date rejection,
release confirmation, offline long-run packaging and renewal-evidence acceptance.
Full logs and source hashes are in the resource's verification.json and test logs.
No credentials, Drive calls or runtime operations were used in this preparation.

## Sources

- https://github.com/googlecolab/google-colab-cli (official session automation)
- https://raw.githubusercontent.com/googlecolab/google-colab-cli/main/src/colab_cli/client.py (current assignment interface)
- https://developers.google.com/colab/api/reference/rest (beta allowlist restriction)
- Existing colabcola managed plans and terminal journals, individually hashed in historical-backends.json.


## Live-progress and connection-cleanup follow-up (2026-10-15)

The delayed 64 KiB runtime pipe reads were replaced by read1 with immediate capture
flush and incremental UTF-8 display. Generic verified stdout/stderr and declared
.log snapshots now appear in the single command, with bounded tails and attempt/file
cursors. The local train/evaluate fixture waits for a host handshake: the host must
see training stdout and carriage-return stderr before allowing training to finish.

A local owned-kernel client adapter initializes manager/ownership state before the
upstream constructor, disconnects explicitly and preserves actual cleanup failures.
The installed WSL Jupyter client passes the injected constructor-failure regression
without the missing-_manager destructor warning. No installed upstream files were
patched. Explicit runtime release and confirmation remain separate and unchanged.

Evidence: nonrest_revalidation/live-progress-fix/{native-tests.log,wsl-tests.log,
verification.json}. This is local acceptance only; no new cloud allocation or
vdm_tracking access. The next authorized replay must use a new frozen journal.


## Live log/client acceptance (2026-10-15)

The fresh GaussianAvatar job retained the same ordinary six-command workload and
1674 payload files, but compiled the fixed runtime into a new immutable notebook.
Plan bd6428f7/job1cdc386e passed through the public command on one T4 in963.539729
seconds within3600. Training completed5000 iterations/4999 optimizer updates; all
native extension checks passed. Timestamped CLI output verified298 training updates
across15 snapshots, first step330 received337.943647 seconds before the completion
marker. The first training stderr snapshot was13333 bytes. No missing-manager or
client-cleanup warning occurred; CLI exit0.

Independent cold Drive readback verified3 input and11 final snapshot members;
63 ZIP outputs and41 PNGs passed integrity/pixel checks. The19142427-byte ZIP SHA-256
is4e512c1844e8cc608c456db1638d57810cccf5a7eee2111e7872fe5af0eb5911.
The checkpoint contains10144 Gaussians and47 finite tensors. Owned runtime
absence and zero active assignments were independently confirmed. All eight matched
training views improved (PSNR23.9423 to36.3784dB); no held-out claim. The known empty
optional upstream MP4 exports remain outside PNG acceptance.

Evidence: nonrest_revalidation/gaussian-live-progress/acceptance.json,
progress-verification.json, live-verification.json, training-verification.json,
comparison-verification.json, launch.log and console-events.ndjson. All authorized
compute is consumed; no additional runtime or vdm_tracking access occurred.

## LongCat guided-packager live replay — 2026-10-16

Exact recreated dev6 job c126bb28/plan6b136d83 ran through the bare public command.
One A100-SXM4-80GB;3685.999965seconds within10800;released and independently0 active.
All99 pins and47 remote inputs/60640344246bytes checked. Cold Drive3 input/17 final
members,27 outputs and3,352,480-byte ZIP verified. Automatic token rotation and final
publication84.111834seconds after initial expiry pass. One complete3.4s960x960/25fps
video decodes locally;4 longer clips fail at segment2 with CUDA OOM. No retries or
changes. Lifecycle gates pass;full-five workload acceptance remains blocked.
The job's exit0 preserves artifacts and must not be read as five-clip success.
Live logs show setup/sampling/outcomes,but model downloads are silent and snapshot
polling refetches old commits;one observed snapshot age81.8s at53 commits. Root-error
excerpts need improvement. Evidence:longcat-live-20261016/SUMMARY.md and JSON proofs.
No more compute is authorized by this result;no5GiB live-throughput claim.

## Dev7 transfer/log follow-up (offline, 2026-10-16)

The LongCat live findings now have local fixes: throttled remote-input transfer.log
publication, cached size/SHA-256-verified immutable commit bytes and bounded early
error excerpts with unchanged redaction. An in-flight fake download is observed
through the actual worker log collection, snapshot publication/recovery and host
Progress reporter before completion. A 53-commit catalog test needs no manifest
GETs on unchanged subsequent polls and one GET for one added commit. Tamper,
identity, caller mutation and attempt-isolation regressions pass. Full suite and
installed-wheel evidence: longcat-progress-oom-20261016. No new cloud latency,
LongCat success, runtime release or 5 GiB live-transfer result is claimed.


## Dev11 private eight-case LongCat live proof — 2026-10-16

The public command-first packager used a `.colaml` declaration for the ordinary
vdm_tracking entry point, with both portraits crossed with VOCASET s01 and three
full-length user recordings. No custom job builder or scientific-grid edit was
needed. Installed `0.1.0.dev11` and a fresh `kernel-v1` journal completed all 8
cases on one A100-SXM4-80GB. Controller time was 9468.904660 seconds; independent
checks finished at 9674.901152 seconds, within the shared 10800-second bound.
The CLI exited 0.

All 99 pins and 47 remote inputs (60640344246 bytes) were verified. Python 3.12.13
isolation retained Torch 2.8.0+cu128. The 5 GiB output allowance was preserved;
the actual 18091785-byte ZIP with 61 outputs does not measure 5 GiB live
throughput. Cold Drive readback verified 3 input and 24 final snapshot members.
All 8 canonical H264/AAC videos at 960x960 and 25 fps pass hashes, full CPU
decoding and separate video/audio duration checks. Peak GPU use reached
79904 MiB in the three-segment cases.

One kernel/request identity appears across 363 busy observations. Three natural
token rotations and authenticated publication beyond initial expiry + 120 seconds
were recorded during busy execution. All 8 video/manifest commits were observed
before final and independently cold-read. Drive server creation times also place
each before the final commit. Independent assignment listing and allocation GET
confirm 0 owned and 0 total active runtimes after cleanup. The original unassign
HTTP status was not persisted, so no specific status is inferred.

Download progress (253 counter lines) and sampling progress (171 positive lines)
were visible. Residual log delay remains: 146 source-stamped receipt samples,
42.076150-second median and 90.132747-second maximum. No client cleanup warning
was observed; upstream model warnings are separate. This is one bounded workflow
proof. It does not measure scientific quality, dated-runtime selection, future
runtime lifetime or the cause of the historical five-case failures.

Evidence is in the mapped `danzel_longcat_example` resource: `acceptance.json`,
`live-verification.json`, `incremental-verification.json`, `video-verification.json`,
`progress-verification.json`, `final-observability.json` and `consumer/longcat.colaml`.
Job: `29f04895b520ec9a622da098642f66bb0f06d3c4ef32f2ec02d69bd69895e1d0`.
Plan: `bb50268440b769f07158320e8f30046d2a8b41d9883b0bb40cc8f637efe560d3`.
The allowance is consumed. A new allocation requires a fresh explicit bound.
