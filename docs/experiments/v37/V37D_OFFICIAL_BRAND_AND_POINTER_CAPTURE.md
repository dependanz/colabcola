# v37D — Official website branding and bounded Windows pointer capture

## User feedback

At the 2026-11-12 v37C 30Hz/C6 Input Lab trial the user saw the finished screen but reported its invented ring logo did not match colabcola.com and that moving the mouse fast enough let the cursor escape the game viewport. The old controller only warped the cursor back to center on Tk Motion callbacks and never clipped it at OS level. This is a real capture defect, not a transport protocol limit.

v37C host report: 3996 distinct decoded H.264 frames / 197.219s = 20.262 FPS; 3674 Tk frames / 197.266s = 18.625 FPS; weighted producer rate 23.501 FPS; input callback to Tk p50/p95/p99 241.08/1310.36/2652.68ms, 1306 observations. Exit 0 with remote_stop_marker_uploaded true (request acknowledged by contents API, but status stopped must still be confirmed remotely).

## Approved website branding

The public website source repository dependanz/colabcola-site for https://colabcola.com/ renders assets/colabcola-banner.png as the homepage's 1584×396 hero image. The same approved asset is already in the application repository at docs/assets/colabcola-banner.png, Git blob SHA f15d2f9b68f0678df7f772d399bb11b45f23ebbe, size 600210. Copy that exact existing Git blob into src/colabcola/assets/colabcola-banner.png, retaining pixels. Bundle the asset in the Python wheel. Only when the user explicitly starts remote Input Lab, validate the packaged bytes against that immutable blob hash and upload them to the exact run directory. Pillow reads the fixed PNG and displays it as a resized 570×143 official website banner in the white circle-wipe ending, alongside exact caption 'interactive bridge test finished'.

This is the website-approved banner, not a hand-drawn two-ring icon. It includes more than an isolated icon; a dedicated approved standalone icon can later be curated from actual source art. No runtime HTTP fetch or arbitrary image URL.

## Bounded Win32 pointer capture

Only after the user explicitly clicks the stage-four LOCK CURSOR button, use GetClientRect/ClientToScreen on the native label HWND, GetClipCursor to remember the preexisting desktop clip, and ClipCursor to restrict mouse movement to a center rectangle fully inside the viewport. Rectangle half-side is min(96, width/4, height/4), retaining sufficient room for relative event deltas but blocking extremely fast exits. Hide cursor and SetCursorPos center; repeat recenter after Tk Motion. Explicit Escape, FocusOut, Unmap/minimize, actual toplevel move/resize and Destroy release clip and cursor visibility, restoring exactly previous clip rectangle. Do not release on child Configure. On any error, fail closed and restore. No global keyboard hooks. Tests inject an API fake rather than modifying the Windows desktop.

Safety: ClipCursor imposes a temporary OS-wide pointer boundary while lock is active. Normal Escape/focus-out/close restore it. Avoid force-killing an unresponsive test while clipped; switch focus or use secure attention if necessary. This is not Raw Input. The user should test restore on the actual Windows desktop before vigorous movement.

## Gates

Test official PNG Git hash and finish-frame pixels; simulate high-speed cursor movement under the clip and exact prior bound restoration; verify Escape, focus loss, Configure, and capture errors; ensure wheel, keyboard, game controls, synthetic default and authenticated remote CCB1 are unchanged. Run focused/full Windows plus Ubuntu CI and local Windows regression, pinned provider integrity. Before another paid Colab run, repeat the one-T4/single-kernel read-only preflight and confirm the prior remote Input Lab stopped. Use one fresh 30Hz/C6 bounded run, observe logo/cursor behavior and independently check stop marker and stopped status. Preserve untracked v32 files and never clean up external Colab runtime automatically.