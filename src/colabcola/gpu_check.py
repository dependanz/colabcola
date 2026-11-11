"""Pure GPU-family validation, embedded verbatim into compiled runtime preflight."""

def _colabcola_gpu_check(requested, resources, gpus, cuda_available):
    import re
    # G4 family: https://docs.cloud.google.com/compute/docs/gpus
    selectors = {"T4": r"(?:TESLA )?T4(?:[- ]|$)", "L4": r"L4(?:[- ]|$)", "A100": r"A100(?:[- ]|$)",
                 "H100": r"H100(?:[- ]|$)", "G4": r"RTX PRO 6000 BLACKWELL SERVER EDITION$"}
    minimum = resources.get("minimum_gpu_memory_bytes")
    report = {"requested_gpu":requested, "minimum_gpu_memory_bytes":minimum,
              "requested_gpu_count":resources["gpu_count"], "gpus":gpus,
              "cuda_available":bool(cuda_available), "state":"passed", "reason":None}
    reason = None
    if resources["accelerator"] == "gpu":
        eligible = []
        for gpu in gpus:
            name = re.sub(r"^(?:NVIDIA|TESLA) +", "", gpu["name"].upper().strip())
            family_ok = requested is None or (requested in selectors and re.match(selectors[requested],name) is not None)
            if family_ok and (minimum is None or gpu["memory_bytes"] >= minimum):
                eligible.append(gpu)
        if resources["cuda_required"] and not cuda_available:
            reason = "cuda_unavailable"
        elif requested is not None and requested not in selectors:
            reason = "unsupported_requested_gpu"
        elif len(eligible) < resources["gpu_count"]:
            reason = "gpu_model_or_memory_mismatch"
    elif requested not in (None,"NONE"):
        reason = "gpu_request_conflicts_with_cpu_job"
    if reason:
        report.update(state="failed",reason=reason)
    return report


def _colabcola_boot_memory(proc_root="/proc", cgroup_root="/sys/fs/cgroup"):
    """Boot-time host memory snapshot; missing Linux metrics remain null."""
    import pathlib
    proc_root, cgroup_root = pathlib.Path(proc_root), pathlib.Path(cgroup_root)
    report = {"mem_total_bytes": None, "cgroup_memory_limit_bytes": None}
    try:
        for line in (proc_root / "meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                parts = line.partition(":")[2].split()
                if len(parts) == 2 and parts[1] == "kB":
                    report["mem_total_bytes"] = int(parts[0]) * 1024
                break
    except (OSError, ValueError):
        pass
    try:
        cgroup_value = ""
        if (cgroup_root / "cgroup.controllers").is_file():
            cgroup_value = (cgroup_root / "memory.max").read_text().strip()
            for line in (proc_root / "self/cgroup").read_text().splitlines():
                parts = line.split(":", 2)
                if len(parts) == 3 and parts[:2] == ["0", ""]:
                    relative = pathlib.PurePosixPath(parts[2])
                    if parts[2].startswith("/") and ".." not in relative.parts:
                        scoped = cgroup_root.joinpath(*relative.parts[1:], "memory.max")
                        if scoped.is_file():
                            cgroup_value = scoped.read_text().strip()
                    break
        elif (cgroup_root / "memory").is_dir():
            base = cgroup_root / "memory"
            cgroup_value = (base / "memory.limit_in_bytes").read_text().strip()
            for line in (proc_root / "self/cgroup").read_text().splitlines():
                parts = line.split(":", 2)
                if len(parts) == 3 and "memory" in parts[1].split(","):
                    relative = pathlib.PurePosixPath(parts[2])
                    if parts[2].startswith("/") and ".." not in relative.parts:
                        scoped = base.joinpath(*relative.parts[1:], "memory.limit_in_bytes")
                        if scoped.is_file():
                            cgroup_value = scoped.read_text().strip()
                    break
        if cgroup_value and cgroup_value != "max":
            amount = int(cgroup_value)
            if amount >= 0:
                report["cgroup_memory_limit_bytes"] = amount
    except (OSError, ValueError):
        pass
    return report
