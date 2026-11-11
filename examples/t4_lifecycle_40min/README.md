# Issue #18: bounded 40-minute live T4 reliability acceptance

The companion \`t4-40min.colaml\` requests **one** Tesla T4, one \`2400\`-second
workload, a \`2520\`-second command timeout, and a \`3300\`-second total
lifecycle cap. No deliberate fault/restart is requested. The script executes
small CUDA matrix multiplications every 10 seconds, emits progress every
30 seconds, and publishes a SHA-verified checkpoint manifest about every
300 seconds. Final success requires a verified \`outputs/acceptance.json\`
whose \`elapsed_seconds >= 2400\` and \`mode == "t4-live"\`.

Only an authenticated profile on the user's Windows host may start this.
Use a trusted **built wheel** of the feature-branch code, not an editable
installation. \`colabcola run --prepare-only\` and \`--preflight-only\`
do **not** allocate; the final \`colabcola run\` invocation does allocate and
must use the **same** \`--run-dir\` and account profile.

After the run inspect \`state.json\`, \`events.ndjson\`, verified
\`runtime/host-telemetry.json\` log snapshots, checkpoint receipts,
\`outputs/acceptance.json\`, and final owned-runtime release evidence.
A clean script exit alone does not prove the control-plane acceptance gate.

No Colab credentials or provider endpoints belong in this folder. The
\`--cpu-smoke\` option exists solely for local tests and explicitly refuses
durations beyond 10 seconds.

If the run fails, do **not** create a new workspace to bypass its journal.
Recover/inspect the existing run first. Do not assume a partial or stale
kernel receipt establishes a successful workload.
