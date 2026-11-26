# Private input assets (issue #40)

## Status and scope

M0 implements the private content-addressed storage protocol in `lifecycle_io.py`.
M1 implements offline recipe packaging and frozen manifest identities. It is a
development intermediate: notebook and staged transport compilation explicitly
reject private-input jobs until managed acquisition is implemented in M2.
Controller staging, plan receipt binding, preflight and runtime materialization
remain pending. The 512 MiB prepared-input cap remains unchanged. No live Drive
or Colab allocation is required by M0/M1 tests.

## M0: storage contract

- A frozen asset identity is the pair `(bytes, sha256)`, with a positive size of
  at most 4 GiB. Use a versioned, domain-separated Drive namespace derived from
  that identity beneath the caller's explicit Drive parent. The full identity
  is bound in the immutable commit. Neither job ID nor target path is part of
  the storage namespace; separate jobs can reuse the same bytes.
- Require an owned, untrashed, writable and unshared folder. Check the scoped
  catalogue's parent, namespace, owner, sharing and trash metadata for each
  referenced object. Upload under that parent with no permission mutation and
  no public URLs. Fail closed if metadata is missing or disagrees. These checks
  are not encryption or a guarantee against subsequent sharing by the account
  owner; do not store assets in a folder that will be shared.
- On a miss, verify the source is a regular file of the frozen size, hash it
  before sending any bytes, and rehash during transfer. Use existing DriveStore
  chunk readback with 8 MiB chunks. Publish the commit only after every chunk is
  verified and the aggregate size/digest match. Recheck parent privacy before
  publishing the commit. Concurrent source mutation can leave uncommitted
  chunks; they are never an accepted asset. No automatic remote deletion.
- On a hit, read and verify the scoped commit and chunk catalogue. Reuse without
  any asset upload or chunk download by the host. This is commit/catalogue
  evidence, not full remote byte verification; the consuming runtime must
  verify all chunk bytes and the full asset SHA-256. A corrupt committed asset
  fails closed instead of being silently replaced. An incomplete publication
  can resume by content-deduplicating and reading back existing chunks.
- Restore into a private temporary file, within the finite caller deadline and
  a disk reserve of asset size plus 128 MiB. Verify each chunk and the aggregate
  digest. Expose the result through an exclusive hardlink in the destination
  filesystem, then remove the temporary file; never overwrite an existing target
  or expose partially verified bytes. A filesystem lacking hardlinks fails
  safely. The later runtime integration must validate the entire target path
  under the job root and check aggregate disk requirements before any restore.
- Exact, plan-bound commit receipts are validated by direct metadata lookup,
  so an immediate restore need not wait for Drive search indexing. Missing
  member catalogue entries may be checked by exact ID in this case, but only
  scoped owned/unshared references can be downloaded. Discovery of existing
  assets without an exact receipt still depends on Drive catalogue visibility.
- Existing run snapshots and environment caches retain their separate storage
  namespaces and behavior. Reuse the renewable authorization lease in the
  control process; never expose credentials or source-local paths to workload
  code or remote asset manifests.

## M1: offline recipe packaging and identity

Declare regular local files, not directories or remote URLs:

```toml
[[private_inputs]]
source = "./models/weights.pt"
target = "models/weights.pt"
```

Source paths are relative to the recipe (absolute paths are also accepted).
The packager streams the source to compute its exact bytes and SHA-256, checks
file metadata before and after hashing, and rejects links, Windows reparse
points, missing or special files and colabcola account configuration paths.
The source file and hardlinked aliases are excluded from implicit project
inclusion; attempts to explicitly embed the same file or reuse it as packaging
control data are rejected. This does not discover independent copies of a
private file elsewhere in the project. Choose project includes accordingly.
The initial limits are 64 files, 4 GiB per file and 8 GiB in aggregate.
Private targets are normalized relative POSIX paths with case-insensitive
collision checks against embedded, remote and other private input paths,
including ancestor/descendant conflicts. `_colabcola` and `outputs` are reserved.

The local `.colajob` stores source bindings alongside the frozen identities;
only target, byte count and SHA-256 enter the portable canonical manifest.
Local source paths are not part of the job ID. The packager rewrites exact
source-path arguments to their declared target relative to the step cwd and
preserves declarations in saved recipes. On a newly packaged job, changing
source bytes changes the identity. Loading a previously packaged job does not
rehash or silently refreeze that asset, including when the original source is
absent. M2 must verify the frozen identity at staging and only permit an absent
source when a verified committed asset is available. A changed or unavailable
source cannot be uploaded under the old digest.

## M2: planned managed integration

The controller must stage and journal exact asset receipts before allocating
compute; the runtime must check plan/manifest identity, per-file and aggregate
limits, disk headroom and the workload deadline before materializing any asset.
Restore before provisioning or executing a workload that needs the asset.
Standalone compilation must not silently discover a user's Drive account;
report that managed acquisition is required. Persist useful size/transfer/reuse/
integrity evidence without private URLs, credentials or local source paths.
Do not relax the notebook cap.

## M3/M4: acceptance

Local tests exercise the real DriveStore protocol against a simulated Drive
endpoint; include corruption, scope, missing/mutated inputs, partial commit,
restart/receipt visibility, bounds and no-overwrite checks. The optional 1 GiB
proof creates 128 distinct 8 MiB chunks on disk and verifies complete restore
and zero asset uploads on a second client:

```bash
COLABCOLA_PRIVATE_ASSET_LARGE_PROOF=1 python -m pytest \
    tests/test_private_assets.py -k real_1gib -s
```

The integration acceptance must use at least two independent managed jobs and a
synthetic 1 GiB local file, no public URL, verified runtime bytes and release,
no second asset upload, and preflight evidence of the total, bound and disk
headroom. A local fake endpoint is not proof of live Drive permission metadata,
throughput or API behavior. No production research assets are needed for tests.
