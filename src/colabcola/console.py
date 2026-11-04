"""Presentation for public CLI commands; never mixes progress into JSON stdout.

This module owns *display*, not redaction or authorization. Call sites must retain
existing credential-safe diagnostics and lifecycle checks. Rich is optional at
runtime, and terminal styles never interpret workload or provider strings as markup.
"""
from __future__ import annotations

import json
import sys
from typing import Any, TextIO


class ConsoleUI:
    """A single, stream-bound stderr renderer shared by public commands."""

    def __init__(self, stream: TextIO | None = None):
        self.stream = sys.stderr if stream is None else stream
        self._rich = None
        if callable(getattr(self.stream, "isatty", None)) and self.stream.isatty():
            try:
                from rich.console import Console
                self._rich = Console(file=self.stream, highlight=False, markup=False)
            except ImportError:
                pass

    @staticmethod
    def _safe(message: Any) -> str:
        # Do not allow escape sequences or carriage-return rewriting in provider
        # messages, paths, errors or a workload's stdout/stderr.
        return "".join(char if char.isprintable() else " " for char in str(message))

    def say(self, message: Any) -> None:
        message = self._safe(message)
        if self._rich is None:
            print("[colabcola] " + message, file=self.stream, flush=True)
            return
        from rich.text import Text
        line = Text("[colabcola] ", style="dim cyan")
        if ": " in message and not message.startswith("["):
            stage, detail = message.split(": ", 1)
            style = ("bold red" if stage == "Error" else
                     "bold green" if stage == "Result" and detail.startswith("SUCCESS") else
                     "bold")
            line.append(stage + ": ", style=style)
            line.append(detail)
        else:
            line.append(message)
        self._rich.print(line)

    def error(self, message: Any, *, command: str | None = None) -> None:
        # The public output has one stable error prefix irrespective of parser.
        # Keep the original, actionable authored reason as the suffix.
        if command:
            self.say(f"Error: {command}: {message}")
        else:
            self.say(f"Error: {message}")


def emit_json(value: Any, *, indent: int | None = None, sort_keys: bool = False,
              separators: tuple[str, str] | None = None) -> None:
    """Write precisely one JSON document to stdout (no progress or styling)."""
    print(json.dumps(value, indent=indent, sort_keys=sort_keys, separators=separators))
