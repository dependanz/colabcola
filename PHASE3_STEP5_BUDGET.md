# Phase 3 step 5 full-training budget

**Status:** local durability gate implemented and verified; calibration compilation/staging/run remain user-operated and separately authorized  
**Prepared:** 2026-08-26; engineering gate updated 2026-08-30  
**Finish line:** fresh step 0 through step 2,507,900 on the full 12,950+150 LJ Speech resource

## Outcome

Do not launch the retained 2,500-step full-resource JobSpec as a repeated full-training chain. Its
segment size was appropriate for transport validation, but repeating it to the configured finish
line would require 1,004 manually operated sessions, approximately 4.25 TB of staged uploads, and
approximately 1.02 TB of recovered checkpoint artifacts.

The viable Colab candidate is an A100-only chain with much longer segments. Atomic local checkpoint
writes, manifest-last Drive publication, strict same-job restoration, and two-pair remote retention
now pass local source and regression verification. They remain live-unverified until the first
calibration checkpoint is published and restored. The recommended next external experiment is one
fresh 0-to-100,000-step A100 calibration, not an unconditional 2,507,900-step run.

## Verified planning inputs

| Input | Value | Evidence boundary |
|---|---:|---|
| Configured finish line | 2,507,900 optimizer steps | pinned V1 schedule: 3,100 epochs × 809 batches/epoch |
| Pilot completion | 5,000 steps | operational split-run proof on 512+32 subset; full run must restart at 0 |
| T4 median rate | 1.8393968695 seconds/step | verified first 2,500-step segment |
| T4 raw finish estimate | 53.3915 GPU-days at the measured median | excludes validation, checkpoint, allocation, transfer, and interruption overhead; established T4 range remains 49.47-55.86 days |
| A100 median / mean rate | 0.232037236 / 0.264549568 seconds/step | verified resumed 2,500-step segment on one A100; subset result, not a full-run guarantee |
| A100 raw finish estimate | 161.646-184.296 GPU-hours / 6.735-7.679 GPU-days | median-to-mean extrapolation before overhead |
| A100 planning envelope | 221.155 GPU-hours / 9.215 GPU-days | explicit estimate: mean extrapolation plus 20% reserve; not measured usage |
| Current Colab account | Pro+ / 680.14 compute units available / approximately 0 CU/hour with no active session | user-supplied signed-in Resources screenshot on 2026-08-26; no accelerator was allocated |
| T4 planning envelope | 59.364-67.032 GPU-days | explicit estimate: established raw range plus 20% reserve |
| First full-resource staged object | 3,213,480,126 bytes | independently verified local bundle; intentionally not uploaded or run |
| Estimated resume object | approximately 4.23 GB | estimate: full-resource object plus the measured approximately 1.017 GB generator/state pair; must be measured before approval |
| Desktop first-upload throughput | 3.004 MiB/s | measured 3.213 GB probe upload; includes Drive/API overhead |
| Colab staged download throughput | 79.10 MiB/s | measured CPU probe; not a provider guarantee |
| Recovered checkpoint result per segment | approximately 1.017 GB | estimate from verified step-5,000 generator/state sizes before ZIP/container overhead |

## Session scenarios

All transfer totals below are planning estimates. They assume the current monolithic bundle repeats
the dataset and latest checkpoint pair in every resumed segment. They exclude failed retries,
controllers, receipts, metrics growth, and periodic mid-session durability uploads.

| Scenario | Target steps/session | Planned sessions | Training time/session from measured rate | Estimated staged uploads | Estimated recovered checkpoints | Decision |
|---|---:|---:|---:|---:|---:|---|
| Current 2,500-step segmentation | 2,500 | 1,004 | A100 mean: 11.0 minutes | ~4.25 TB | ~1.02 TB | **NO-GO**; orchestration and transfer dominate |
| T4, 10-hour managed-runtime envelope | 15,000 | 168 | 7.66 hours | ~710 GB | ~171 GB | **NO-GO by default**; excessive sessions and exposure |
| T4, 20-hour Pro+ envelope | 30,000 | 84 | 15.33 hours | ~354 GB | ~85 GB | **NO-GO by default**; A100 evidence is materially better |
| A100, 10-hour managed-runtime envelope | 100,000 | 26 | 7.35 hours at measured mean | ~109 GB | ~26 GB | viable only after durability and account-budget gates |
| A100, 20-hour Pro+ envelope | 200,000 | 13 | 14.70 hours at measured mean | ~54 GB | ~13 GB | best transfer/session count, but requires Pro+, units, and A100 availability |

The 10-hour and 20-hour envelopes deliberately leave two and four hours beneath the official
general 12-hour and Pro+ 24-hour maxima. They are safeguards, not guarantees: Colab states that
limits, availability, accelerator type, and termination behavior remain dynamic.

## Compute-unit and dollar budget

Google does not publish stable universal GPU availability or usage limits. The official FAQ says
paid availability depends on the account's compute-unit balance and that the backend may terminate
when units are exhausted. The signed-in purchase page is the authority for the user's current
price, included balance, and offers. Therefore a defensible dollar total cannot be inferred from
public documentation alone.

Current account values:

1. Colab plan: Pro+ (user-reported and visible in the signed-in Resources panel).
2. Current compute-unit balance: 680.14 CU.
3. The UI-displayed A100 compute-unit burn rate in CU/hour: not measured; zero shown while no
   session is active is not the A100 rate.
4. The signed-in effective purchase price in dollars per compute unit: not measured.

Then calculate:

```text
A100 planning compute units = 221.155 GPU-hours × displayed A100 CU/hour
Additional units to buy     = max(0, planning units - current balance)
Planning dollar ceiling     = additional units × current dollars/CU
```

Do not substitute a third-party or historical CU/hour figure. If the account cannot expose an A100
or its current burn rate, the A100 budget remains `not measured` and the full chain remains stopped.

## Required engineering gate

The prior controller returned checkpoints only at terminal packaging. Contract `0.13` / compiler
contract `0.10` now adds an opt-in durable-checkpoint policy for Drive controllers.

Before the 100,000-step calibration, implement and verify:

1. ⏳ An exact multi-segment planner remains required before continuing beyond the calibration; the
   single immutable 0→100,000 calibration JobSpec does not need it.
2. ✅ Periodic upload is implemented at 25,000-step workload intervals. Stable local members are
   SHA-256/MD5 measured, resumably uploaded and metadata-verified, and committed by a canonical
   hash-bound manifest uploaded last. Remote retention is exactly two committed pairs.
3. ✅ Resume discovery is same-job and fail-closed: it ignores orphan members and rejects duplicate
   step manifests, malformed contracts, unexpected filenames, metadata mismatch, absent members,
   and content hash/size mismatch before atomically restoring the newest committed pair.
4. ✅ Runtime result evidence distinguishes restored step, every publication ID, and the exact
   retained manifest/generator/state IDs. Existing terminal state still distinguishes success and
   failure; richer remote status polling remains Phase 4 work.
5. ✅ Retention cannot delete the sole pair: it begins only above two committed records and trashes
   the oldest manifest before its members. Final remote checkpoint cleanup remains intentionally
   deferred until terminal artifacts are recovered and verified.
6. ⏳ The checked-in 5,551-byte calibration JobSpec validates locally with 16 payload entries and
   four outputs. Its multi-gigabyte compilation, staging, and controller identity measurement are
   the next user-operated step and precede any Drive/run approval.

Separating the stable 3.21 GB dataset object from changing checkpoint/job objects could reduce
repeated uploads substantially, while keeping the union of fetched immutable objects fully
self-contained. That optimization is recommended but not required for the first 100,000-step
calibration if the approximately 4.23 GB resume transfer is accepted and measured.

## Recommended staged commitment

1. **Engineering only:** durable checkpoint transport is locally implemented and verified; the
   multi-segment planner remains deferred until the calibration establishes feasibility. No GPU or
   Drive action resulted from this implementation.
2. **Account budget capture:** retain the verified Pro+/680.14-CU starting state, then record the
   A100 CU/hour rate when the separately approved calibration allocation exists; obtain purchase
   pricing only if the measured projection exceeds the included balance.
3. **One A100 calibration:** after a separate exact-payload approval, run fresh step 0→100,000 with
   a 10-hour entrypoint ceiling and durable pairs at steps 25k/50k/75k/100k.
4. **Calibration acceptance:** recover and verify the final result, measure actual CU consumption,
   throughput, validation/checkpoint overhead, interruptions, and generated audio at step 100,000.
5. **Remainder decision:** authorize, resize, move providers, or stop the remaining 2,407,900 steps
   using measured cost and quality evidence.

## Stopping rules

1. Do not silently substitute a T4 when the approved budget assumes A100 throughput.
2. Stop and re-budget if the first 10,000 A100 steps average more than 0.35 seconds/step or projected
   compute-unit consumption exceeds the approved ceiling by more than 10%.
3. Stop if a periodic durable checkpoint cannot be independently recovered and resumed exactly.
4. Stop after two consecutive segment failures with the same unresolved cause.
5. Stop before continuing beyond 100,000 if generated audio shows no material improvement over the
   metallic/buzzy step-5,000 baseline, unless the user explicitly approves continuation as a
   training-method investigation rather than a quality-seeking run.
6. Never use multiple accounts, reconnect automation, hidden remote control, or another quota-evasion
   technique to manufacture continuity.

## Current external authority

- Google Colab FAQ, accessed 2026-08-26:
  `https://research.google.com/colaboratory/faq.html`
- Past runtime version FAQ, accessed 2026-08-26:
  `https://research.google.com/colaboratory/runtime-version-faq.html`
- Signed-in paid-services page for account-specific pricing and units:
  `https://colab.research.google.com/signup?upgrade=true`

The FAQ states that managed runtimes and GPU types are dynamic and not guaranteed, general
runtimes can last at most 12 hours depending on availability and usage, Pro+ can support up to 24
hours of continuous execution with sufficient compute units, and dedicated GCP Marketplace VMs are
the supported route for guaranteed resources without Colab-enforced runtime limits.
