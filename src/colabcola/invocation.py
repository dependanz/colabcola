"""Bind a source JobSpec to invocation-time arguments without shell interpretation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .errors import SpecError
from .spec import JobSpec, load_job_spec


JOB_FILE_SUFFIX = ".colajob"


@dataclass(frozen=True, slots=True)
class JobInvocation:
    """A validated JobSpec plus the exact arguments supplied for one invocation.

    This object deliberately has no job ID. The final identity also depends on the payload file
    records produced by the deterministic bundler in Phase 1, step 6.
    """

    spec: JobSpec
    job_args: tuple[str, ...]

    @property
    def effective_argv(self) -> tuple[str, ...]:
        """Return configured defaults followed by invocation arguments, unchanged."""

        return self.spec.entrypoint.argv + self.job_args


def bind_job_arguments(
    spec: JobSpec, job_args: Sequence[str] = ()
) -> JobInvocation:
    """Create an immutable invocation while preserving every argument exactly."""

    if isinstance(job_args, str):
        raise SpecError("job arguments must be a sequence of strings, not one string")

    normalized: list[str] = []
    for index, argument in enumerate(job_args):
        if not isinstance(argument, str):
            raise SpecError(f"job argument {index} must be a string")
        if "\x00" in argument:
            raise SpecError(f"job argument {index} contains a forbidden NUL byte")
        normalized.append(argument)
    return JobInvocation(spec=spec, job_args=tuple(normalized))


def load_job_invocation(
    path: Path | str, job_args: Sequence[str] = ()
) -> JobInvocation:
    """Load a strict-JSON ``.colajob`` file and bind invocation-time arguments."""

    source = Path(path)
    if source.suffix != JOB_FILE_SUFFIX:
        raise SpecError(f"job file must use the {JOB_FILE_SUFFIX} extension: {source}")
    return bind_job_arguments(load_job_spec(source), job_args)
