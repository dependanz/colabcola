# Environment cache design (issue #43)

Status: M0 implements named `pip.wheels`; M1/M2 implement opt-in,
same-prefix environment caching with managed Drive storage. Local integration
uses real Python/pip subprocesses, an injected environment manager, and an
in-memory Drive HTTP endpoint. Real micromamba and cold/warm Colab acceptance
remain gates. Legacy jobs and environments without opt-in do not use the cache.

## M1: identity and prefix safety

- Opt in per named environment, `cache = true`. No cache lookup or
  upload for legacy recipes or an environment that does not opt in.
- Use a versioned SHA-256 over a canonical JSON identity: normalized environment
  definition, exact manager pin, cache/launcher implementation version, declared
  dependency inputs and wheel SHA-256/size, actual runtime image release identity,
  Linux/CPU ABI and GPU compute capability/driver when relevant. `latest` alone
  is not an image identifier. Missing reliable compatibility facts cause a safe
  bypass; a request for a T4 alone is not proof of an actual GPU architecture.
- Exact CPU flags remain the default. Real CPU runs on the same Colab image
  produced different keys solely due to heterogeneous exposed CPU flags. An
  explicit `cache_cpu = "x86-64-v2"` declares that the entire installed
  environment is portable to that ISA baseline. Check the intersection of
  visible CPU flags against the baseline; project only the CPU identity into
  that target; constrain micromamba's `__archspec` to `x86_64_v2` on eligible
  portable cache builds. All other identity fields remain exact. Preserve the
  observed host fingerprint in result evidence. This is a user contract, not
  automatic native-code analysis: package scripts and prebuilt wheels may use
  additional ISA extensions, so keep exact matching for unverified native
  builds. No old cache key is silently reinterpreted under the new policy.
- Hash requirements and local package sources; conservatively bind the full frozen
  job identity for local source builds or other dependencies whose build input
  scope cannot be established. Wheel-only environments need not be invalidated by
  unrelated workload code changes. An explicit policy version invalidates older
  incompatible cache formats.
- A conda prefix can embed its absolute path in executables and installed
  metadata. Build and restore at the same stable, key-derived absolute prefix
  (under a managed runtime-owned cache root). Do not relocate arbitrary conda
  directories or replace unverified existing paths.
- Cache only a fully resolved, checked prefix before any workload step; never
  cache the job tree, controller configuration, token files, or the whole package
  manager download cache. Preserve executable permissions and internal links;
  reject unsafe paths, absolute/escaping links, special files, archive duplicates,
  unbounded expansion, and invalid identities. Stage extraction privately before
  atomically exposing the prefix. Recheck interpreter activation and version,
  package inventory, `pip check`, toolchain and CUDA compatibility after restore.

## M2: account-scoped persistent storage

- Reuse the managed lifecycle's explicit Drive parent, renewable access lease,
  hash-verified chunked object publication and commit-last model. A distinct
  versioned cache namespace must be independent of a random execution run ID;
  bind the full cache key in every accepted commit. Never locate artifacts by
  filename alone or scan unrelated account data. Never pass auth into step argv,
  environment variables, cache metadata, or packaged project files.
- The cache manifest records the canonical identity, prefix, archive byte count
  and digest, bounded expanded size, resolved package evidence and measured cold
  build duration. Publish the commit only after chunk readback. Verify references,
  identity, archive and restored environment before declaring a hit. Concurrent
  writers may publish equivalent verified commits; no destructive remote cleanup
  or overwriting an unrelated snapshot as part of a run.
- Treat cache availability as an optimization, not an execution requirement.
  Missing/incompatible/corrupt entries, storage failures, and insufficient disk or
  time must be visible in evidence and either trigger a clean cold build (when
  the remaining budget permits) or a bounded failure. Never execute a partially
  restored prefix. Limit transfers and decompression, reserve workload and final
  artifact/release time, and clean only the owned temporary cache workspace.
- Standalone notebooks without the managed storage boundary should explicitly
  report a cache bypass, not attempt to discover or mount a user's Drive account.
  Cache objects contain installed code and potentially proprietary wheels; keep
  them in the opted-in user's selected private storage scope.

## M3: acceptance and measurement

Use a single named CPU environment first and a small packaged wheel. Test cold
build/publish and a fresh-runtime warm restore, then changed wheel/declaration and
runtime compatibility invalidation. Verify that no conda or pip install stage is
run on a verified hit and that results match the cold run. Exercise unavailable
storage, invalid digest, unsafe archive, truncated upload, deadline, low disk, and
post-restore validation failure with local tests. Verify owned runtime release
and artifact recovery for every managed acceptance run.

Record lookup/download/extraction/validation times, cold build and publication
cost, and end-to-end environment setup time. Report signed savings relative to
uncached build time, not just a reduction in pip install time; a cache may be
slower for small environments. Measure a representative GPU/CUDA extension
workload only after CPU acceptance. A future cache for native wheels produced
inside a workload needs a separate explicit artifact/publish contract and must
not blindly preserve the post-workload environment.
