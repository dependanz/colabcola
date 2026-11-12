# v33o — live 30 FPS diagnostic sweep receipt

**Evidence:** operator-provided PowerShell output and attached T4 runtime log,
November 11, 2026. This is an observation, not independent provider telemetry.
**Run:** `v33o-30fps-diagnostics-20261111-071212`.
**Configuration:** Tesla T4; authenticated kernel Comm; H.264; explicit
remote PyAV provisioning; `--fps 30 --poll-hz 30`;
`--benchmark-h264-capacity` with phases **1,2,3,4,8,16,32 credits**.
One kernel/encoder/decoder, continuous sequence IDs; 10 s phases, 2 s warmup
excluded. **All phases completed; process exit code 0; owned T4 journal
`phase: released` and explicit release confirmation.**

## Scored phase metrics

| Credits | Decoded FPS | Distinct renderer IDs in replies/s | Duplicate IDs | Host pull-submit p50 (ms) | Pull RTT p50 (ms) | Remote read p50 (ms) | Remote encode p50 (ms) | Max outstanding |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7.125 | 7.125 | 0 | 140 | 140.0 | 3.76 | 2.87 | 1 |
| 2 | 14.125 | 14.000 | 1 | 63 | 140.0 | 3.85 | 3.08 | 2 |
| 3 | 18.250 | 17.125 | 9 | 47 | 140.5 | 4.91 | 3.93 | 3 |
| 4 | 18.500 | 17.125 | 11 | 47 | 141.0 | 5.18 | 3.80 | 4 |
| 8 | 18.625 | 18.375 | 2 | 47 | 125.0 | 4.03 | 3.08 | 4 |
| 16 | 18.250 | 18.000 | 2 | 47 | 125.0 | 3.73 | 2.98 | 4 |
| 32 | 18.500 | 18.250 | 2 | 47 | 125.0 | 3.98 | 3.39 | 4 |

Every scored phase lasted 8 seconds. Each phase reported zero H.264 dependent-frame
suppression and zero missing remote timing samples. There were 1,134 pull requests
in the overall sweep, including unscored warmups and phase drains.
Remote read timing is remote file read plus zlib decode; remote encode timing is
source RGB/PyAV encoding, both measured on the *remote* clock. Their medians
cannot safely be added to represent a total or subtracted from host times.

## Interpretation

- The observed decoded rate increased from 7.125 (1 credit) to 18.25 (3)
  and saturated at about 18–19 FPS from 3 through 32 credits.
- At 3–32 credits, the **host's median pull-submit interval was 47 ms**,
  compared with the 33.3 ms target for 30 Hz. The existing receive loop
  blocks up to 30 ms in `window.poll` before reconsidering submission.
  This makes receive-wait/scheduling overhead a promising *hypothesis*.
- Medians for remote file reading plus zlib decoding and CPU H.264 encoding
  were separately ~3–5 ms each, and local decode+RGB stayed below 0.8 ms.
  These sampled operations alone do not explain a 47 ms host interval;
  kernel messaging, scheduling and renderer generation remain unisolated.
- Distinct renderer IDs observed in replies reached ~18/s. This does not
  measure the independent production rate of the CUDA renderer: its snapshots
  may be overwritten between host requests.
- Higher credit limits did not fill the window (maximum outstanding was 4),
  so simply increasing the cap cannot establish 30 unique FPS.
- This is a single non-randomized sweep with phases run in ascending order.
  Input ACK p50 was ~172–188 ms. No input-to-photon or monitor presentation
  latency and no complete network-byte accounting was measured.

## Next experiment

Keep the standard 12 FPS paths unchanged. Experiment offline with a deadline-
aware, shorter **host IOPub receive wait** only in the 30/60 Hz capacity sweep,
measure actual receive-wait duration and submission interval, and verify the
full Windows regression before another T4. If the improvement fails to increase
fresh renderer IDs, inspect independent `render-metrics.json` producer FPS and
kernel Comm dispatch scheduling next.
