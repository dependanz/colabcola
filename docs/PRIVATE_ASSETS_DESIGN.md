# Private input assets (issue #40)

## Status and scope

M0 implements the private content-addressed storage protocol in `lifecycle_io.py`.
Recipe syntax, manifest/plan binding, controller staging, preflight and runtime
materialization are **not** implemented by M0. The 512 MiB prepared-input cap
remains unchanged. No live Drive or Colab allocation is required by M0 tests.

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

## M1/M2: planned integration

The recipe-level `[[private_inputs]]` declaration should map a local source to a
normalized job-relative target. Freeze bytes and digest without embedding the
large file in the notebook. Include size, digest and target in the job identity;
keep source bindings local. Reject target collisions (including ancestor/child
and reserved paths) with all bundled, remote and other private input targets.
Avoid also embedding the declared asset via broad project include patterns.
Validate the local source binding before staging or allow an explicit existing
asset receipt for an already-committed identity. The controller must stage and
journal exact asset receipts before allocating compute; the runtime must check
plan/manifest identity, per-file and aggregate limits, disk headroom and the
workload deadline before materializing any asset. Restore before provisioning
or executing a workload that needs the asset. Standalone compilation must not
silently discover a user's Drive account; report that managed acquisition is
required. Persist useful size/transfer/reuse/integrity evidence without private
URLs, credentials or local source paths. Do not relax the notebook cap.

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
