"""Read-only configuration selection for independent, explicitly named profiles."""
from __future__ import annotations

import argparse
import contextlib
from contextvars import ContextVar
import os
from pathlib import Path
import re

from .lifecycle_io import LifecycleError


class ProfileError(LifecycleError):
    """A profile name or selected configuration path is invalid."""


_PROFILE_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_WINDOWS_DEVICES = frozenset({"con", "prn", "aux", "nul"} | {
    prefix + str(number) for prefix in ("com", "lpt") for number in range(1, 10)
})
_SELECTION_VARIABLES = ("COLABCOLA_CONFIG_HOME", "COLABCOLA_PROFILE")
_SHADOWED_CONFIG_HOMES = ContextVar("colabcola_shadowed_config_homes", default=())


def _validate_name(name):
    if (not isinstance(name, str) or not _PROFILE_NAME.fullmatch(name)
            or name in _WINDOWS_DEVICES):
        raise ProfileError(
            "Invalid profile name: use 1-64 lowercase letters, digits, underscores or hyphens, "
            "starting with a letter or digit; Windows device names are reserved"
        )
    return name


def _absolute_home(value):
    try:
        if isinstance(value, str) and not value:
            raise ValueError("empty configuration path")
        # Do not resolve(): it follows links before we can reject them.
        path = Path(os.path.abspath(Path(value).expanduser()))
        for parent in (path, *path.parents):
            if parent.is_symlink() or (hasattr(parent, "is_junction") and parent.is_junction()):
                raise ProfileError("Colabcola configuration cannot use links or junctions")
        return path
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        if isinstance(exc, ProfileError):
            raise
        raise ProfileError("Invalid Colabcola configuration directory") from None


def named_home(name):
    """Return a validated named profile path without reading or creating files."""
    return _absolute_home(Path.home() / ".colabcola" / "configs" / _validate_name(name))


def profile_name(path):
    """Identify an exact named-profile directory; arbitrary homes have no name."""
    selected_path = _absolute_home(path)
    # The named-profile root is only a lexical comparison here. Inspecting an
    # unrelated root could block a valid legacy or custom configuration.
    profiles_root = Path(os.path.abspath(Path.home() / ".colabcola" / "configs"))
    if selected_path.parent != profiles_root:
        return None
    try:
        return _validate_name(selected_path.name)
    except ProfileError:
        return None


def _resolve_config_home():
    configured = os.environ.get("COLABCOLA_CONFIG_HOME")
    if configured:
        return _absolute_home(configured)
    profile = os.environ.get("COLABCOLA_PROFILE")
    if profile:
        return named_home(profile)
    legacy = Path.home() / ".config" / "colabcola"
    if (legacy.is_symlink()
            or (hasattr(legacy, "is_junction") and legacy.is_junction())
            or legacy.exists()):
        return _absolute_home(legacy)
    return named_home("default")


def config_home():
    """Select a configuration directory without accessing credentials or writing files.

    An explicit environment path wins over the environment profile. With neither,
    existing legacy configuration stays selected; fresh homes use configs/default.
    """
    return _resolve_config_home()


def private_config_homes():
    """Return active and shadowed config paths for private-file exclusion only."""
    return tuple(dict.fromkeys((config_home(), *_SHADOWED_CONFIG_HOMES.get())))


@contextlib.contextmanager
def selected(profile=None, config_home=None):
    """Temporarily select one explicit home, restoring both environment selectors."""
    if profile is not None and config_home is not None:
        raise ProfileError("--profile and --config-home cannot be used together")
    if profile is None and config_home is None:
        yield _resolve_config_home()
        return
    path = named_home(profile) if profile is not None else _absolute_home(config_home)
    previous = {name: os.environ.get(name) for name in _SELECTION_VARIABLES}
    shadowed = _SHADOWED_CONFIG_HOMES.get()
    previous_home = previous["COLABCOLA_CONFIG_HOME"]
    if previous_home:
        # Retain only lexical exclusion metadata. An unrelated old directory may
        # be linked or unavailable, and must not prevent explicit selection.
        previous_path = Path(os.path.abspath(os.path.expanduser(previous_home)))
        shadowed = (*shadowed, previous_path)
    shadow_token = _SHADOWED_CONFIG_HOMES.set(shadowed)
    try:
        os.environ["COLABCOLA_CONFIG_HOME"] = str(path)
        os.environ.pop("COLABCOLA_PROFILE", None)
        yield path
    finally:
        try:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        finally:
            _SHADOWED_CONFIG_HOMES.reset(shadow_token)


class _ProfileArgument(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            parser.error("--profile may be specified only once")
        setattr(namespace, self.dest, values)


def add_profile_argument(parser):
    """Add the consistent explicit-profile selector to a command parser."""
    return parser.add_argument(
        "--profile", metavar="NAME", default=None, action=_ProfileArgument,
        help="use the named Colabcola profile for this command",
    )
