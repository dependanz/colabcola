"""Opt-in loopback demonstration of Colab Cola's bidirectional application bridge.

No Colab allocation is performed here. The remote-provider transport and the
unmodified-application capture layer remain future work.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hmac
import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .bridge_protocol import (
    BridgeProtocolError,
    MAX_FRAME_BYTES,
    PROTOCOL_VERSION,
    read_message,
    send_message,
)

WIDTH, HEIGHT = 192, 120
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


@dataclass
class Scene:
    x: float = 0.5
    y: float = 0.5
    phase: float = 0.0

    def apply(self, message: dict) -> None:
        if message["event"] == "mouse":
            self.x = float(message["x"])
            self.y = float(message["y"])
        elif message["event"] == "key":
            key = message["key"].lower()
            if key == "left":
                self.phase -= 0.3
            elif key == "right":
                self.phase += 0.3
            elif key == "up":
                self.y = max(0.0, self.y - 0.05)
            elif key == "down":
                self.y = min(1.0, self.y + 0.05)


def render_ppm(scene: Scene, seconds: float, device: str = "cpu") -> bytes:
    """Render a moving field; optionally perform the math with a CUDA tensor."""
    if device == "cuda":
        import torch

        xx = torch.linspace(0, 1, WIDTH, device="cuda").unsqueeze(0).expand(HEIGHT, -1)
        yy = torch.linspace(0, 1, HEIGHT, device="cuda").unsqueeze(1).expand(-1, WIDTH)
        distance = (xx - scene.x) ** 2 + (yy - scene.y) ** 2
        glow = torch.exp(-distance * 42)
        wave = torch.sin(xx * 10 + seconds * 2 + scene.phase) * 0.5 + 0.5
        r = (45 + 170 * glow + 35 * wave).clamp(0, 255)
        g = (30 + 100 * yy + 65 * glow).clamp(0, 255)
        b = (90 + 95 * xx + 50 * (1 - wave)).clamp(0, 255)
        pixels = torch.stack((r, g, b), dim=-1).to(torch.uint8).cpu().numpy().tobytes()
    else:
        image = bytearray()
        for iy in range(HEIGHT):
            y = iy / (HEIGHT - 1)
            for ix in range(WIDTH):
                x = ix / (WIDTH - 1)
                glow = math.exp(-42 * ((x - scene.x) ** 2 + (y - scene.y) ** 2))
                wave = math.sin(x * 10 + seconds * 2 + scene.phase) * 0.5 + 0.5
                image.extend((
                    min(255, int(45 + 170 * glow + 35 * wave)),
                    min(255, int(30 + 100 * y + 65 * glow)),
                    min(255, int(90 + 95 * x + 50 * (1 - wave))),
                ))
        pixels = bytes(image)
    result = f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii") + pixels
    if len(result) > MAX_FRAME_BYTES:
        raise BridgeProtocolError("rendered image too large")
    return result


async def handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    token: str,
    fps: int,
    device: str,
) -> None:
    inputs: asyncio.Task | None = None
    try:
        hello = await asyncio.wait_for(read_message(reader), timeout=10)
        if hello["type"] != "hello" or not hmac.compare_digest(hello["token"], token):
            await send_message(writer, {"v": PROTOCOL_VERSION, "type": "error", "message": "access denied"})
            return
        await send_message(writer, {
            "v": PROTOCOL_VERSION, "type": "ready", "width": WIDTH, "height": HEIGHT,
            "device": device,
        })
        scene = Scene()

        async def receive_inputs() -> None:
            while True:
                message = await read_message(reader)
                if message["type"] == "close":
                    return
                if message["type"] != "input":
                    raise BridgeProtocolError("expected input event")
                scene.apply(message)

        inputs = asyncio.create_task(receive_inputs())
        started = time.monotonic()
        sequence = 0
        while not inputs.done():
            frame = render_ppm(scene, time.monotonic() - started, device)
            await send_message(writer, {
                "v": PROTOCOL_VERSION, "type": "frame", "seq": sequence,
                "format": "ppm", "data": base64.b64encode(frame).decode("ascii"),
            })
            sequence += 1
            await asyncio.sleep(1 / fps)
    except (asyncio.IncompleteReadError, ConnectionError, BrokenPipeError, BridgeProtocolError, TimeoutError):
        pass
    finally:
        if inputs is not None:
            inputs.cancel()
            try:
                await inputs
            except (asyncio.CancelledError, asyncio.IncompleteReadError, ConnectionError, BridgeProtocolError):
                pass
        writer.close()
        try:
            await writer.wait_closed()
        except ConnectionError:
            pass


async def run_server(host: str, port: int, token: str, fps: int, device: str) -> None:
    if device == "cuda":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("CUDA demo requires PyTorch on the server") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but PyTorch reports no available GPU")

    async def serve_one(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await handle_connection(reader, writer, token=token, fps=fps, device=device)

    server = await asyncio.start_server(serve_one, host, port)
    address = server.sockets[0].getsockname()
    print(f"[colabcola] Interactive demo listening on {address[0]}:{address[1]} ({device})", flush=True)
    print("[colabcola] This is a loopback prototype, not a managed Colab session.", flush=True)
    async with server:
        await server.serve_forever()


async def stream_viewer(
    host: str,
    port: int,
    token: str,
    outgoing: queue.Queue,
    frames: queue.Queue,
    stopped: threading.Event,
) -> None:
    writer = None
    try:
        reader, writer = await asyncio.open_connection(host, port)
        await send_message(writer, {"v": PROTOCOL_VERSION, "type": "hello", "token": token})
        response = await asyncio.wait_for(read_message(reader), timeout=10)
        if response["type"] != "ready":
            raise RuntimeError(response.get("message", "bridge handshake rejected"))
        frames.put(("status", f"Connected · {response['device']} · {response['width']}x{response['height']}"))

        async def transmit() -> None:
            while not stopped.is_set():
                try:
                    event = outgoing.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.02)
                    continue
                await send_message(writer, {"v": PROTOCOL_VERSION, "type": "input", **event})

        task = asyncio.create_task(transmit())
        pending_read = asyncio.create_task(read_message(reader))
        sequence = -1
        try:
            while not stopped.is_set():
                # Keep one read task alive across timeouts: cancelling a partial
                # length-prefixed read could otherwise desynchronize the stream.
                done, _ = await asyncio.wait((pending_read,), timeout=0.3)
                if not done:
                    continue
                message = pending_read.result()
                pending_read = asyncio.create_task(read_message(reader))
                if message["type"] == "error":
                    raise RuntimeError(message.get("message", "remote error"))
                if message["type"] != "frame" or message["seq"] <= sequence:
                    continue
                sequence = message["seq"]
                payload = base64.b64decode(message["data"], validate=True)
                if len(payload) > MAX_FRAME_BYTES or not payload.startswith(b"P6\n"):
                    raise BridgeProtocolError("unsupported or oversized frame")
                # Only the freshest frame is needed for interactive presentation.
                while True:
                    try:
                        frames.get_nowait()
                    except queue.Empty:
                        break
                frames.put(("frame", payload))
        finally:
            task.cancel()
            pending_read.cancel()
            await asyncio.gather(task, pending_read, return_exceptions=True)
    except (OSError, ValueError, RuntimeError, asyncio.IncompleteReadError, BridgeProtocolError) as exc:
        frames.put(("status", f"Disconnected: {type(exc).__name__}: {exc}"))
    finally:
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass


def photoimage_from_ppm(master, data: bytes):
    """Decode raw PPM bytes for Tk; Base64 is only for network transport."""
    import tkinter as tk
    return tk.PhotoImage(master=master, data=data, format="PPM")


VIEWER_STATUS_HEIGHT = 28


def concise_viewer_status(value: str) -> str:
    """Keep verbose transport telemetry in stdout; show fixed-length GUI status."""
    value = str(value)
    pieces = value.split(" · ")
    if len(pieces) >= 2 and "kernel Comm " in pieces[1]:
        fps = pieces[1].replace("kernel Comm ", "")
        roundtrip = next((p.replace("pull RTT p50 ", "RTT ")
                          for p in pieces if p.startswith("pull RTT p50 ")), "")
        return " · ".join(p for p in (pieces[0], fps, roundtrip) if p)
    if len(pieces) >= 2 and " FPS" in pieces[1]:
        return " · ".join(pieces[:2])
    return value[:96] + ("…" if len(value) > 96 else "")


def create_viewer_layout(root, width: int, height: int):
    """Fix the image viewport and footer sizes independently of status text."""
    import tkinter as tk

    if not 1 <= width <= 1920 or not 1 <= height <= 1080:
        raise ValueError("invalid interactive viewport dimensions")
    root.geometry(f"{width}x{height + VIEWER_STATUS_HEIGHT}")
    root.resizable(False, False)
    container = tk.Frame(root, width=width, height=height + VIEWER_STATUS_HEIGHT)
    container.pack()
    container.pack_propagate(False)
    label = tk.Label(container, text="Waiting for frames", anchor="center")
    label.place(x=0, y=0, width=width, height=height)
    status = tk.StringVar(master=root, value="Connecting...")
    tk.Label(container, textvariable=status, anchor="w", padx=8).place(
        x=0, y=height, width=width, height=VIEWER_STATUS_HEIGHT
    )
    return label, status


def run_viewer(host: str, port: int, token: str, *, width: int = WIDTH,
               height: int = HEIGHT, title: str | None = None,
               worker=None, viewer_stats: bool = False,
               timestamp_local_input: bool = False,
               poll_interval_ms: int = 30) -> None:
    # Preserve the 30ms viewer default for existing transports; allow only
    # explicit local experiments to request a shorter bounded Tk poll.
    if (type(poll_interval_ms) is not int
            or not 5 <= poll_interval_ms <= 100):
        raise ValueError("Tk poll interval must be an integer between 5 and 100 ms")
    # Tk stays on the main thread; transport I/O stays on a dedicated thread.
    import tkinter as tk

    outgoing: queue.Queue = queue.Queue(maxsize=64)
    frames: queue.Queue = queue.Queue(maxsize=2)
    stopped = threading.Event()
    root = tk.Tk()
    root.title(title or "Colab Cola · Interactive Bridge Prototype")
    label, status = create_viewer_layout(root, width, height)
    from .bridge_viewer_stats import LocalFrameTiming, ViewerPresentationMeter
    gui_meter = ViewerPresentationMeter(started_at=time.monotonic()) if viewer_stats else None
    last_motion = [0.0]

    def enqueue(event: dict) -> None:
        if timestamp_local_input:
            # Only opt-in local experiments use this private metadata.
            # Do not alter the provider-facing input event protocol.
            event = {**event, "_ui_callback_ns": time.perf_counter_ns()}
        try:
            outgoing.put_nowait(event)
        except queue.Full:
            # Input events are transient; never block the GUI.
            pass

    def mouse(event) -> None:
        now = time.monotonic()
        if now - last_motion[0] < 0.03:
            return
        last_motion[0] = now
        enqueue({"event": "mouse", "x": max(0.0, min(1.0, event.x / width)),
                 "y": max(0.0, min(1.0, event.y / height))})

    def poll() -> None:
        if stopped.is_set():
            root.destroy()
            return
        try:
            while True:
                kind, data = frames.get_nowait()
                if kind == "status":
                    status.set(concise_viewer_status(data))
                elif kind in ("frame", "frame-timed"):
                    timing = None
                    if kind == "frame-timed":
                        if not isinstance(data, tuple) or len(data) != 2:
                            raise ValueError("malformed local timed frame")
                        data, timing = data
                        if not isinstance(timing, LocalFrameTiming):
                            raise ValueError("invalid local frame metadata")
                    image_started = time.perf_counter()
                    image = photoimage_from_ppm(root, data)
                    label.configure(image=image, text="")
                    label.image = image
                    configured_ns = time.perf_counter_ns()
                    if gui_meter is not None:
                        gui_meter.presented((time.perf_counter() - image_started) * 1000)
                        if timing is not None:
                            gui_meter.configured_timed_frame(timing, configured_ns)
                    if status.get() == "Connecting...":
                        status.set("Connected")
        except queue.Empty:
            pass
        if gui_meter is not None:
            summary = gui_meter.report(time.monotonic())
            if summary is not None:
                print("[colabcola] " + summary, flush=True)
        root.after(poll_interval_ms, poll)

    label.bind("<Motion>", mouse)
    root.bind("<KeyPress>", lambda e: enqueue({"event": "key", "key": e.keysym[:32]}))
    root.protocol("WM_DELETE_WINDOW", lambda: (stopped.set(), root.destroy()))
    failures = []

    def dispatch() -> None:
        try:
            if worker is not None:
                worker(outgoing, frames, stopped)
            else:
                asyncio.run(stream_viewer(host, port, token, outgoing, frames, stopped))
        except Exception as exc:
            failures.append(exc)
            stopped.set()

    thread = threading.Thread(target=dispatch, daemon=True)
    thread.start()
    root.after(poll_interval_ms, poll)
    root.mainloop()
    stopped.set()
    thread.join()  # Wait for in-flight I/O to finish before releasing a remote runtime.
    if failures:
        raise failures[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="colabcola bridge",
        description="Experimental interactive bridge: local loopback or one managed Colab T4.",
    )
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("serve", "view"):
        command = sub.add_parser(name)
        command.add_argument("--host", default="127.0.0.1")
        command.add_argument("--port", type=int, default=8765)
    server = sub.choices["serve"]
    server.add_argument("--fps", type=int, default=8)
    server.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    from .profiles import add_profile_argument
    add_profile_argument(parser)
    remote = sub.add_parser("colab", help="allocate one managed T4 for an interactive GPU scene")
    remote.add_argument("--run-dir", type=Path)
    remote.add_argument("--gpu", choices=("T4",), default="T4")
    remote.add_argument("--transport", choices=("files", "comm"), default="files",
                        help="files=proven authenticated polling, comm=experimental persistent Jupyter websocket")
    remote.add_argument("--video-codec", choices=("ppm", "h264"), default="ppm",
                        help="ppm=existing snapshots; h264=opt-in one- or two-credit experimental Comm video")
    remote.add_argument("--install-remote-pyav", action="store_true",
                        help="explicitly permit pinned PyPI PyAV wheel install in the owned T4 kernel")
    remote.add_argument("--fps", type=int, default=6)
    remote.add_argument("--poll-hz", type=int, default=3)
    remote.add_argument("--inflight", type=int, default=None,
                        help="Comm frame credits (default 2 for ordinary Comm streaming; 1 for file polling and benchmarks)")
    remote.add_argument("--viewer-stats", action="store_true",
                        help="report GUI frame configuration FPS and Tk processing time locally")
    remote.add_argument("--benchmark-credits", action="store_true",
                        help="single T4 sweep of 1, 2, 3, 4, then 1 frame credits at 12 FPS")
    remote.add_argument("--credit-phase-seconds", type=int, default=12)
    remote.add_argument("--probe-link", action="store_true",
                        help="measure tiny Jupyter Comm RTT before rendering; no extra allocation")
    remote.add_argument("--benchmark-sweep", action="store_true",
                        help="compare 6/6, 12/6, 6/12, 12/12, and repeated 6/6 in one Comm session")
    remote.add_argument("--benchmark-phase-seconds", type=int, default=12)
    remote.add_argument("--max-seconds", type=int, default=600)
    remote.add_argument("--confirm-allocation", action="store_true")
    remote.add_argument("--preflight-only", action="store_true")
    clean = sub.add_parser("cleanup", help="release a previously owned interactive T4")
    clean.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action in ("colab", "cleanup"):
        from .bridge_colab import cleanup_colab, run_colab
        from .lifecycle_io import LifecycleError
        try:
            if args.action == "cleanup":
                return cleanup_colab(args)
            if args.run_dir is None and not args.preflight_only:
                parser.error("colab requires --run-dir except for --preflight-only")
            # Two credits are live-verified for ordinary Comm streaming,
            # including local Tk configuration. Preserve single-credit
            # behavior for file polling, diagnostics and controlled sweeps.
            if args.inflight is None:
                args.inflight = (2 if args.transport == "comm" and args.video_codec == "ppm"
                                   and not (args.benchmark_sweep or args.benchmark_credits
                                            or args.probe_link) else 1)
            if args.video_codec == "h264" and (
                    args.transport != "comm" or args.inflight not in (1, 2)
                    or args.benchmark_sweep or args.benchmark_credits or args.probe_link):
                parser.error("H.264 video requires --transport comm --inflight 1 or 2, without benchmarks/probe")
            if args.install_remote_pyav and args.video_codec != "h264":
                parser.error("--install-remote-pyav requires --video-codec h264")
            max_poll_hz = 12 if args.transport == "comm" else 6
            if not 1 <= args.fps <= 12 or not 1 <= args.poll_hz <= max_poll_hz:
                parser.error(
                    f"colab FPS must be 1..12 and {args.transport} poll-hz must be 1..{max_poll_hz}"
                )
            if not 120 <= args.max_seconds <= 1800:
                parser.error("colab --max-seconds must be within 120..1800")
            if not 1 <= args.inflight <= 4:
                parser.error("--inflight must be in 1..4")
            if args.benchmark_credits:
                from .bridge_credit_sweep import MIN_PHASE_SECONDS, MAX_PHASE_SECONDS
                if (args.transport != "comm" or (args.fps, args.poll_hz) != (12, 12)
                        or args.inflight != 1 or args.benchmark_sweep or args.probe_link):
                    parser.error("credit sweep requires --transport comm --fps 12 --poll-hz 12"
                                 " and no other experimental transport mode")
                if not MIN_PHASE_SECONDS <= args.credit_phase_seconds <= MAX_PHASE_SECONDS:
                    parser.error("credit sweep phase duration must be 10..40 seconds")
                if args.max_seconds < args.credit_phase_seconds * 5 + 80:
                    parser.error("credit sweep time budget must cover all phases and startup")
            if args.inflight > 1 and (args.transport != "comm" or args.benchmark_sweep
                                       or args.benchmark_credits or args.probe_link):
                parser.error("--inflight >1 requires --transport comm, without --probe-link or --benchmark-sweep")
            if args.probe_link and (args.transport != "comm" or args.benchmark_sweep):
                parser.error("--probe-link requires --transport comm without --benchmark-sweep")
            if args.benchmark_sweep:
                from .bridge_sweep import MAX_PHASE_SECONDS, MIN_PHASE_SECONDS
                if args.transport != "comm" or (args.fps, args.poll_hz) != (6, 6):
                    parser.error("benchmark sweep requires --transport comm --fps 6 --poll-hz 6")
                if not MIN_PHASE_SECONDS <= args.benchmark_phase_seconds <= MAX_PHASE_SECONDS:
                    parser.error("benchmark phase duration must be between 10 and 40 seconds")
                if args.max_seconds < args.benchmark_phase_seconds * 5 + 80:
                    parser.error("benchmark max-seconds must cover all five phases plus startup")
            return run_colab(args)
        except LifecycleError as exc:
            print("[colabcola] " + str(exc))
            return 2
    if args.host not in LOOPBACK_HOSTS:
        parser.error("prototype transport is loopback-only; use an authenticated provider adapter for remote compute")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.action == "serve" and not 1 <= args.fps <= 30:
        parser.error("--fps must be between 1 and 30")
    token = os.environ.get("COLABCOLA_BRIDGE_TOKEN", "")
    if len(token) > 512:
        parser.error("COLABCOLA_BRIDGE_TOKEN is too long")
    if args.action == "serve":
        try:
            asyncio.run(run_server(args.host, args.port, token, args.fps, args.device))
        except KeyboardInterrupt:
            return 0
    else:
        run_viewer(args.host, args.port, token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
