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
