"""Local Desktop-client handoff, with no background reader of OAuth input."""
from __future__ import annotations

import json
import os
import select
import sys
import time

from . import client_downloads

POLL_SECONDS = 1.0


def _say(message):
    print("colabcola: " + message, file=sys.stderr, flush=True)


def _identity(result):
    return tuple(sorted((str(c.path), c.sha256) for c in result.candidates))


def _pause_or_path(answer, candidates=()):
    answer = answer.strip()
    if not answer or answer.casefold() == "later":
        return None
    if len(answer) <= 4 and answer.isdecimal() and 1 <= int(answer) <= len(candidates):
        # Manual selection is revalidated by project_setup before import.
        return str(candidates[int(answer) - 1].path)
    return answer


def _manual(input_fn, say, candidates=(), reason=None):
    if reason:
        say(reason)
    for index, candidate in enumerate(candidates, 1):
        say(str(index) + ". " + json.dumps(str(candidate.path), ensure_ascii=True))
    options = "; choose a listed number or enter a path" if candidates else ""
    return _pause_or_path(input_fn("Path to the downloaded Desktop OAuth JSON" + options + " (Windows C:\\... or local path; Enter or later to finish later): "), candidates)


def _tty_descriptor(interactive):
    if not interactive or os.name != "posix" or sys.stdin is None:
        return None
    try:
        fd = sys.stdin.fileno()
        return fd if os.isatty(fd) else None
    except (OSError, ValueError, AttributeError):
        return None


def _input_ready(fd, timeout=0):
    return bool(select.select([fd], [], [], timeout)[0])


def _found(candidate, say):
    say("Detected Desktop OAuth JSON for the selected project: " + json.dumps(str(candidate.path), ensure_ascii=True))
    say("Validated its Desktop client format, Google endpoints and project ID; continuing login. Google access is verified by the authorization steps that follow.")
    return candidate


def choose_client(project_id, *, expected_sha256=None, automatic=True, input_fn=None, interactive=None, say=None):
    """Return a validated snapshot, an explicitly entered path, or an intentional pause.

    The watcher runs in the calling thread. It never leaves input() running behind
    the next Google authorization prompt. Completed input wins over detection;
    unfinished path text is discarded only when automatically advancing a real TTY.
    """
    ask = input if input_fn is None else input_fn
    say = _say if say is None else say
    if not automatic:
        return _manual(ask, say)
    directories = client_downloads.download_directories()
    if not directories:
        return _manual(ask, say, reason="Downloads could not be located; enter the Desktop OAuth JSON path below.")
    say("Checking current-user Downloads for Google client_secret_*.json files for project " + project_id + ".")
    for directory in directories:
        say("Downloads: " + json.dumps(str(directory), ensure_ascii=True))
    if interactive is None:
        interactive = sys.stdin is not None and sys.stdin.isatty()
    fd = _tty_descriptor(interactive)
    result = client_downloads.scan_downloads(directories, project_id=project_id, expected_sha256=expected_sha256)
    if result.incomplete:
        return _manual(ask, say, reason="Automatic selection is unavailable because the Downloads check was incomplete or a matching file is not ready. Enter an explicit path.")
    if len(result.candidates) > 1:
        return _manual(ask, say, result.candidates, "More than one different Desktop client matches this project. Choose a number or enter a path; no client was selected automatically.")
    if fd is None:
        # Noninteractive consumers never wait for a future file. Explicit paths
        # remain available through the prompt or --drive-client.
        if len(result.candidates) == 1:
            time.sleep(POLL_SECONDS)
            again = client_downloads.scan_downloads(directories, project_id=project_id, expected_sha256=expected_sha256)
            if not again.incomplete and _identity(again) == _identity(result):
                return _found(again.candidates[0], say)
            if not again.incomplete and len(again.candidates) > 1:
                return _manual(ask, say, again.candidates, "More than one different matching client was found; choose explicitly.")
        return _manual(ask, say, reason="No single stable matching Desktop client was detected; enter its path, or rerun login after downloading it.")

    say("Waiting for a matching Desktop OAuth download. Paste a path and press Enter, or type later to pause. Enter alone or Ctrl+C also pauses.")
    say("A detected download advances login automatically; unfinished path text is discarded before the next authorization prompt.")
    previous = _identity(result) if len(result.candidates) == 1 else None
    while True:
        try:
            if _input_ready(fd, POLL_SECONDS):
                return _pause_or_path(ask(""))
            current = client_downloads.scan_downloads(directories, project_id=project_id, expected_sha256=expected_sha256)
            if _input_ready(fd):
                return _pause_or_path(ask(""))
            if current.incomplete:
                previous = None
                continue
            if len(current.candidates) > 1:
                return _manual(ask, say, current.candidates, "More than one different Desktop client matches this project. Choose explicitly.")
            identity = _identity(current)
            if len(current.candidates) == 1 and identity == previous:
                # A partially typed path must not become an OAuth code. Check
                # once more for a completed answer, then drop only pending TTY
                # input before this prompt automatically advances.
                if _input_ready(fd):
                    return _pause_or_path(ask(""))
                import termios
                try:
                    termios.tcflush(fd, termios.TCIFLUSH)
                except (OSError, termios.error):
                    return _manual(ask, say, reason="This terminal cannot safely advance automatically; enter the client path.")
                return _found(current.candidates[0], say)
            previous = identity if len(current.candidates) == 1 else None
        except (OSError, ValueError):
            return _manual(ask, say, reason="Automatic terminal monitoring is unavailable; enter the Desktop OAuth JSON path.")
