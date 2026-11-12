# v36A — experimental Freedoom game application adapter (offline first)

**Status:** prototype source+authenticated input routing. This is not a claim of successful Colab gameplay. Keep the v34C 30FPS/C6 and v35C 60FPS/C10 synthetic CUDA paths intact. No game or Colab runtime was launched to author this code.

## Goal and architecture

Run freely redistributable Freedoom Phase 1 (or Phase 2 fallback) with the local package's Chocolate Doom source port inside one explicitly selected browser-owned Linux Colab T4 notebook. The engine is CPU-rendered, though the normal external T4 authorization checks remain in place. A supervised per-session Xvfb display at 640x360x24 provides the virtual screen. A supervised ffmpeg x11grab child streams 640x360 RGB24 frames to the game source, which publishes the existing bounded CCB1-zlib snapshots into an eight-slot ring and latest-frame file. The existing PyAV H.264 push source and authenticated kernel Comm remain unchanged; the Windows viewer displays the stream.

Remote kernel bootstrap chooses this source **only** when CLI explicitly requests --application freedoom under external, sequence-buffered, H.264 Comm, sender-driven **30FPS/30Hz**, with credits 4, 6 or 8. First trial uses six credits and one-frame lag. The default application is synthetic. The 60Hz zlib level-0 optimization is mutually exclusive with the game source; no normal/managed path can enable it. The app is not a new remote command execution interface.

## Remote environment and installation

No automatic apt, pip, game asset download, or arbitrary program execution is performed. The remote runtime must already contain the OS binaries Xvfb, chocolate-doom, ffmpeg, and xdotool. The adapter checks absolute executable paths and accepts a **fixed** Freedoom package IWAD path only: /usr/share/games/doom/freedoom1.wad, otherwise /usr/share/games/doom/freedoom2.wad. Both must be regular, non-symlink files with a valid IWAD signature and bounded size. Ubuntu's freedoom package normally installs these files. A separate explicit, disposable notebook provisioning step may install packages after the user reviews it; never run it silently. User remains responsible for the browser-connected runtime's credits and manual disconnect.

## Input and responsiveness

- The Tk viewer's opt-in game mode tracks key down/up for Up, Down, Left, Right, Control_L/R, Space, Return, Escape, Shift_L/R, and Tab; first-click focuses the viewer. It also forwards bounded relative mouse deltas and left/right button transitions.
- Unlike synthetic mouse updates, the host's event loop does **not coalesce game key transitions**. It transmits up to sixteen ordered input events per iteration on the established authenticated shell Comm.
- The kernel validates event names, values, sequence numbers, and identity, then enqueues them into a bounded 64-item game-source input queue. Queue overflow fails closed with a bounded error, not silent dropping of key releases.
- The game source applies input via fixed xdotool XTEST commands with no shell and no arbitrary key strings, and reports the last applied input seq in the CCB1 frame ACK. Held keys/buttons are released on loss of Tk focus and on application shutdown.
- Mouse motion may be limited at the edge of the Tk window in this first proof; keyboard turning (Left/Right) is the primary way to verify first gameplay controls. This is a non-physical input-to-Tk metric, not input-to-photon.
- The inherited render_copy_ms telemetry in this game mode measures blocking RGB capture-pipe reading, **not** CUDA render/copy. It must not be compared as the same stage with synthetic GPU scenes.
- Rendering at 30 FPS is an application experiment only; do not claim 60FPS or GPU acceleration from Chocolate Doom.

## Remote lifecycle constraints

- Xvfb uses an exclusive reserved local X display :98. If another process owns it, fail closed rather than killing/attaching to unrelated jobs. It listens on no network TCP socket.
- The adapter supervises only its own Xvfb, Doom, and ffmpeg children. It starts no general listener; the capture pipe is bounded by the RGB frame dimensions and an eight-slot publication ring. The child process group is not blindly killed; only started PIDs are terminated/killed on timeout.
- A Comm close requests the game worker stop, releases held keys and mouse buttons, and terminates its child processes. A failed/ambiguous bootstrap needs manual runtime review; an externally owned T4 is never allocated or released by the bridge.
- Colab notebook disconnection is still **manual**. The bridge's detached journal is not proof that the game subprocesses or T4 are stopped if a close packet was lost. Terminate the disposable runtime in the browser at end of the experiment.

## Validation gates

1. Full offline Windows+Ubuntu bridge CI, local Windows focused v36/v35/v34 and full bridge tests, pinned provider integrity, clean tracked worktree. Preserve the untracked v32 evidence.
2. Provision disposable Colab notebook with explicitly reviewed packages only if missing, and connect one active browser-owned T4. Test exact one-assignment/single-kernel read-only preflight for --application freedoom.
3. One supervised live run of <=120 seconds on 30FPS, C6, renderer lag 1. Observe actual game scene, movement via held Up/Left/Right, Ctrl firing, Space door opening; test closing the Tk viewer to exercise process cleanup.
4. Inspect remote renderer source FPS, decoded distinct FPS, Tk FPS, p95/p99 frame pacing and input callback-to-Tk latency. Ensure clean external journal and manually disconnect notebook.
5. If capture/startup fails, log sanitized stage/error class only. Do not repeatedly retry an uncertain kernel execution request on a paid notebook; inspect dependencies and errors offline first.

## Success criteria

We do **not** declare this milestone live-accepted until a user observes actual playable DOOM graphics and controls in the Windows viewer, confirms correct input ACK behavior, and verifies that only the adapter-owned processes terminate. Treat Linux Xvfb/SDL/ffmpeg focus, mouse relative control and session cleanup as unproven until that live test.
