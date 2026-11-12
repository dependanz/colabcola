# v33a: optional H.264 Jupyter Comm protocol contract

Status: **offline contract only**. No runtime integration, provider allocation, or throughput claim.

The existing `CCB1` (zlib PPM) Comm reply and file-polling routes are unchanged.
`bridge_video_comm_protocol.py` defines an optional, versioned Comm reply:

- `v = 1`, `kind = "video_access_unit"`, `codec = "h264-annexb"`
- `run_id`: same 32-character owned session identifier as the existing bridge
- `pull_seq`: correlated bounded request identifier; caller must track actual outstanding requests
- `input_ack`: source-observed input sequence number, not input-to-photon latency
- `data`: bounded canonical Base64 containing one existing `CCV1` access-unit record

The payload is fully bounded, checked for session identity, request correlation,
valid CCV1 header, complete body, and agreement between the keyframe flag and
presence of an IDR NAL marker. A sequence gate rejects duplicates/reordering
and suppresses dependent frames after loss until an IDR appears.

No sender/controller policy is included at this stage. No attempt is made to
subtract the remote producer's monotonic `capture_ns` from the local receiver's
clock. Actual decoder recovery, buffering behavior, codec dependencies, and
application display latency require the later live integration phase.

Run offline: `python -m pytest -q tests/test_bridge_video_comm_protocol.py`.
Next: integrate opt-in mode with one outstanding credit, owned-kernel lifecycle,
real remote encoding and local decoding; retain existing Comm PPM as the default.
