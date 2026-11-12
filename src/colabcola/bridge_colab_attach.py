"""Explicit, non-owning attachment to one user-provisioned Colab T4.

Never allocates or unassigns a runtime. An explicitly authorized live attach
does execute the bridge bootstrap inside the existing user-owned notebook.
"""
from __future__ import annotations

import time
import uuid

from .bridge_colab import save
from .bridge_demo import run_viewer
from .lifecycle import run_lock
from .lifecycle_io import LifecycleError


def select_single_t4(provider) -> str:
    """Fail closed for missing, multiple, or non-T4 assignments."""
    assignments = provider.assignments()
    if not isinstance(assignments, (list, tuple)) or len(assignments) != 1:
        raise LifecycleError("external attach requires exactly one active Colab assignment")
    assignment = assignments[0]
    accelerator = getattr(assignment, "accelerator", None)
    if getattr(accelerator, "value", accelerator) != "T4":
        raise LifecycleError("external attach requires an existing Tesla T4")
    endpoint = getattr(assignment, "endpoint", None)
    if not isinstance(endpoint, str) or not 1 <= len(endpoint) <= 256:
        raise LifecycleError("external T4 assignment identity invalid")
    return endpoint


def single_kernel_id(provider, session) -> str:
    """Read-only Jupyter kernel listing; never start, restart or replace one."""
    from urllib.parse import urlparse
    import requests

    url, token = getattr(session, "url", ""), getattr(session, "token", "")
    if (not isinstance(url, str) or urlparse(url).scheme != "https"
            or not urlparse(url).hostname or not isinstance(token, str)
            or not token):
        raise LifecycleError("external runtime proxy credentials unavailable")
    response = requests.get(
        url.rstrip("/") + "/api/kernels",
        params={"colab-runtime-proxy-token": token},
        headers={"X-Colab-Runtime-Proxy-Token": token},
        timeout=(10, 40),
    )
    if response.status_code != 200:
        raise LifecycleError("external kernel inventory unavailable")
    try:
        kernels = response.json()
    except (ValueError, TypeError):
        raise LifecycleError("external kernel inventory invalid") from None
    if not isinstance(kernels, list) or len(kernels) != 1:
        raise LifecycleError("external attach requires exactly one existing kernel")
    record = kernels[0]
    kernel_id = record.get("id") if isinstance(record, dict) else None
    if not isinstance(kernel_id, str) or not 1 <= len(kernel_id) <= 128:
        raise LifecycleError("external kernel identity invalid")
    return kernel_id


def validate_attach_options(args) -> None:
    if (getattr(args, "transport", None) != "comm"
            or getattr(args, "video_codec", None) != "h264"
            or not (getattr(args, "benchmark_h264_capacity", False)
                    or getattr(args, "push_video", False))
            or not (
                (getattr(args, "push_video", False) and
                 (getattr(args, "fps", None), getattr(args, "poll_hz", None))
                 in ((30, 30), (60, 60)))
                or (getattr(args, "benchmark_h264_capacity", False) and
                    (getattr(args, "fps", None), getattr(args, "poll_hz", None))
                    == (30, 30)))
            or getattr(args, "inflight", None) != 1
            or getattr(args, "confirm_allocation", False)):
        raise LifecycleError("external attach requires 30Hz pull sweep or 30/60Hz push H.264 Comm")
    if getattr(args, "push_video", False) and (
            not getattr(args, "sequence_buffered", False)
            or getattr(args, "benchmark_h264_capacity", False)
            or getattr(args, "benchmark_h264_credits", False)):
        raise LifecycleError("push video requires external buffered H.264 without pull benchmarks")
    if getattr(args, "producer_zlib_level0", False) and (
            not getattr(args, "push_video", False)
            or not getattr(args, "sequence_buffered", False)
            or (getattr(args, "fps", None), getattr(args, "poll_hz", None)) != (60, 60)
            or getattr(args, "benchmark_h264_capacity", False)):
        raise LifecycleError("level0 intermediate compression requires 60Hz external push")
    credits = getattr(args, "push_credits", 4)
    fps = getattr(args, "fps", None)
    if (type(credits) is not int or credits not in (4, 6, 8, 10, 12)
            or (not getattr(args, "push_video", False) and credits != 4)
            or (getattr(args, "push_video", False) and
                ((fps == 30 and credits not in (4, 6, 8))
                 or (fps == 60 and credits not in (8, 10, 12))))):
        raise LifecycleError("invalid external push FPS/credit window")
    if not getattr(args, "preflight_only", False) and not getattr(args, "confirm_attach", False):
        raise LifecycleError("external attach requires --confirm-attach")


def run_attached_colab(args, provider) -> int:
    """Use one existing T4 kernel without taking ownership of the runtime."""
    validate_attach_options(args)
    endpoint = select_single_t4(provider)
    session = provider._session(endpoint)
    if select_single_t4(provider) != endpoint:
        raise LifecycleError("external assignment changed before attachment")
    kernel_id = single_kernel_id(provider, session)
    if getattr(args, "preflight_only", False):
        print("[colabcola] Existing T4 and single kernel verified; "
              "no runtime allocated, no code uploaded or executed", flush=True)
        return 0

    if args.run_dir is None or args.run_dir.exists():
        raise LifecycleError("external attach requires a fresh run directory")

    from .bridge_comm import KernelCommBridge
    from .bridge_video_credit_sweep import run_h264_capacity_sweep

    args.run_dir.mkdir(parents=True, exist_ok=False)
    with run_lock(args.run_dir):
        state = {
            "schema": 1, "backend": "google-colab-external",
            "ownership": "external", "gpu": "T4",
            "profile": args.profile or "", "run_id": uuid.uuid4().hex,
            "created_at": time.time(),
        }
        save(args.run_dir, state, "attaching")
        deadline = time.monotonic() + args.max_seconds
        peer = KernelCommBridge(
            provider, session, state["run_id"], args.fps,
            video_codec="h264",
            install_remote_pyav=getattr(args, "install_remote_pyav", False),
            sequence_buffered=getattr(args, "sequence_buffered", False),
            max_buffer_lag_frames=getattr(args, "renderer_lag_frames", 2),
            video_push=getattr(args, "push_video", False),
            push_credits=getattr(args, "push_credits", 4),
            producer_zlib_level0=getattr(args, "producer_zlib_level0", False),
        )
        # Used exclusively to label benchmark ownership. Never a release token.
        peer.external_attachment = True
        opened = False
        try:
            # The expected kernel is reverified before remote file upload.
            details = peer.open(expected_kernel_id=kernel_id)
            opened = True
            save(args.run_dir, state, "running", gpu_name=details["gpu"],
                 transport="comm", video_codec="h264",
                 sequence_buffered=getattr(args, "sequence_buffered", False),
                 renderer_lag_frames=getattr(args, "renderer_lag_frames", 2),
                 target_fps=args.fps,
                 video_push=getattr(args, "push_video", False),
                 push_credits=getattr(args, "push_credits", 4),
                 producer_zlib_level=(
                     0 if getattr(args, "producer_zlib_level0", False) else 1))

            push = getattr(args, "push_video", False)
            if push:
                from .bridge_video_push_session import run_push_comm_h264
                def work(outgoing, frames, stopped):
                    run_push_comm_h264(
                        peer, details["gpu"], outgoing, frames, stopped,
                        deadline, credits=peer.push_credits)
            else:
                def work(outgoing, frames, stopped):
                    run_h264_capacity_sweep(
                        peer, details["gpu"], outgoing, frames, stopped,
                        deadline, args.poll_hz,
                        phase_seconds=getattr(args, "h264_capacity_phase_seconds", 10),
                    )

            run_viewer(
                "127.0.0.1", 0, "", width=640, height=360,
                title=("Colab Cola · T4 sender-driven H.264" if push
                       else "Colab Cola · existing T4 H.264 (kernel Comm)"),
                worker=work, viewer_stats=getattr(args, "viewer_stats", False),
                timestamp_local_input=push,
                poll_interval_ms=(5 if push and args.fps == 60
                                  else 10 if push else 30),
            )
        finally:
            try:
                peer.close()
            except Exception:
                save(args.run_dir, state, "needs_manual_review",
                     close_status="unconfirmed")
                raise LifecycleError(
                    "external Comm close unconfirmed; user-owned runtime not released"
                ) from None
            else:
                # Host disconnect is not proof that remote worker has stopped.
                save(args.run_dir, state,
                     "detached" if opened else "needs_manual_review",
                     close_status="host_disconnected")
                print("[colabcola] Host bridge detached; "
                      "user-owned Colab runtime NOT released", flush=True)
    return 0
