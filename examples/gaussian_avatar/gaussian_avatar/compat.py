"""Explicit, hash-bound compatibility patch for the pinned rasterizer source."""
import hashlib
from pathlib import Path

HEADER = "submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h"
BEFORE = "4ead5b624a1f36197a6da3cba9e032a651318c96bc8d1f856eb196a3120fa544"
AFTER = "33e3dd7ffbdafd3d0729e8d111a993ffcef071da65fd87bbf363b59adf522026"

def apply(root):
    path = Path(root) / HEADER
    body = path.read_bytes()
    before = hashlib.sha256(body).hexdigest()
    if before == AFTER:
        return {"path": HEADER, "before": BEFORE, "after": AFTER, "state": "already_applied"}
    if before != BEFORE:
        raise ValueError("Rasterizer header differs from the pinned source; refusing compatibility patch")
    updated = body.replace(b"#include <iostream>", b"#include <cstdint>\n#include <iostream>", 1)
    if hashlib.sha256(updated).hexdigest() != AFTER:
        raise ValueError("Unexpected rasterizer patch result")
    path.write_bytes(updated)
    return {"path": HEADER, "before": BEFORE, "after": AFTER, "state": "applied"}
