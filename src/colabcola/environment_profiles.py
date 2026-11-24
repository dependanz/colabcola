"""Reviewed bootstrap binaries for named workload environments."""

# The release asset is verified before it is made executable. Updating this pin
# changes the frozen manifest and therefore the job identity.
MICROMAMBA = {
    "kind": "micromamba-linux-x86_64",
    "version": "2.10.0-0",
    "url": "https://github.com/mamba-org/micromamba-releases/releases/download/2.10.0-0/micromamba-linux-64",
    "bytes": 19028608,
    "sha256": "701f39c06f8e6a670a0e186c8479b9e5c0502a2f0cda6cef65d83bdd2875e8bd",
}
