# v37 — Colab Cola Interactive Input Lab

**Status:** experimental, offline-first app on its own feature branch. Not yet live accepted on Colab.

## Purpose

Freedoom streamed near 29 FPS and an input-bearing session recorded 679 input callback→Tk acknowledgments (median ~197ms), but we could not isolate whether specific buttons worked: the user saw no clear response from Space, and the Windows cursor did not stay centered/hidden. This app diagnoses input independently of the game.

## Isolation and requirements

Opt-in CLI application name: inputlab, allowed only on one explicitly attached browser-owned T4, sequence-buffered H.264 Comm push at 30FPS / 30Hz, credits 4, 6 or 8. Default synthetic and legacy Freedoom modes stay unchanged. Level-0 60Hz compression remains separate. Remote Input Lab uses Python/Pillow to draw 640x360 RGB frames (CPU rendering on a GPU-verified T4), publishing the existing bounded CCB1 ring to the existing H.264 sender. Requires Pillow in the notebook; no Xvfb or Chocolate Doom. No automatic apt/pip or external listeners.

## Interactive stages

1. Keyboard: on-screen key grid; letters, digits, common punctuation, F1-F12, arrows, modifiers, navigation and numeric keypad keysyms turn green as key-down arrives. They remain green to show all previously detected keys. Not all OS-reserved shortcuts can be captured by Tk/Windows.
2. Mouse: left, middle and right clicks illuminate each button indicator, with recent remote click locations shown as colored dots.
3. Scroll: signed wheel-up and wheel-down tick counters and progress bars.
4. Pointer-lock / voxel look: a software-rendered block world, center crosshair, yaw/pitch telemetry and an explicit LOCK CURSOR button. Clicking it hides and centers the Windows cursor and forwards real bounded relative pointer deltas, rotating the remote camera. Escape, focus loss and close restore the cursor and release held inputs. After measured look movement, unlocking permits FINISH TEST.
5. Completion: white expanding circular wipe, stylized linked-ring Colab Cola wordmark, exact caption: **interactive bridge test finished**. The final page stays visible until closed.

Each Next Test button is rendered by the remote app, with normalized left-button down/up delivered over the authenticated Jupyter Comm; the remote state decides stage transitions. Key-up is never intentionally coalesced. The remote accepts only allowlisted events, checked run ID, monotonically increasing input sequence, strict bounded fields, and a bounded FIFO. The existing CCB1 frame input-ACK field reports the last applied sequence.

## Windows pointer-lock safety

Never capture on viewer open. Only enable after a deliberate click on the stage-four control. First try a Tk no-cursor setting; because Tk 8.6 does not support that cursor on Windows, fall back to Win32 ShowCursor(FALSE) with strictly paired ShowCursor(TRUE) restoration and SetCursorPos to recenter. Fail closed if hiding/recentering is unavailable. Escape, focus-out, viewer close and session completion release pointer lock. No global keyboard hooks, no persistent pointer capture or arbitrary OS commands. Pointer warping is experimental; verify actual cursor behavior on the user's Windows desktop.

## Acceptance

Offline: test all stage transitions, keyboard highlighting, three buttons, signed scrolling, input bounds, FIFO preservation, focus/Escape cursor restore, module selection, CLI external-mode guards, and CCB1 frame parity and wipe (conditionally skip graphics tests when PIL unavailable). CI Windows/Ubuntu plus local full bridge. Check pinned provider source and clean tracked worktree; preserve v32 untracked files.

Live: one disposable browser-owned T4, read-only assignment/kernel preflight, explicit 30FPS/C6 120s run-dir, Pillow import available. Observe actual green keyboard feedback (including Space), all buttons and scrolling, deliberate lock with hidden/centered pointer, voxel camera rotation, Escape restore and final wipe. Report FPS, pacing and input callback→Tk percentiles (not physical input-to-photon). Manually disconnect Colab afterward; never bridge cleanup an external journal.

## Limitations

This is a diagnostic scene, not a full game engine. The voxel world is software-rendered with procedural blocks; the completion linked-ring lockup is a generated logo treatment, not a pixel-exact reproduction of the repository banner. Windows-reserved shortcuts may be unavailable. Pointer warping is not Raw Input and must be validated on the target desktop.
