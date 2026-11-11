"""Allocation-timeout ownership handling; offline, no Google calls or GPU."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json

import pytest
from requests.exceptions import ReadTimeout

from colabcola import bridge_colab as bridge
from colabcola.lifecycle_io import LifecycleError


@pytest.mark.parametrize("accepted_before_timeout", [False, True])
def test_timeout_is_sanitized_reconciled_and_never_retried(
        tmp_path, monkeypatch, capsys, accepted_before_timeout):
    providers = []

    class TimeoutProvider:
        def __init__(self):
            self.endpoint = None
            self.requests = 0
            self.released = []
            self.closed = False
            providers.append(self)

        def assignments(self):
            return []

        def allocate(self, allocation_id, gpu):
            self.requests += 1
            assert gpu == "T4"
            if accepted_before_timeout:
                self.endpoint = "owned-test-endpoint"
            raise ReadTimeout("Authorization: Bearer SECRET_SHOULD_NOT_LEAK")

        def reconcile(self, allocation_id, gpu):
            assert gpu == "T4"
            return self.endpoint

        def lookup(self, endpoint):
            return self.endpoint if endpoint == self.endpoint else None

        def release(self, endpoint):
            assert endpoint == self.endpoint == "owned-test-endpoint"
            self.released.append(endpoint)
            self.endpoint = None
            return {"state": "absent"}

        def close(self):
            self.closed = True

    import colabcola.profiles
    monkeypatch.setattr(bridge, "GoogleProvider", TimeoutProvider)
    monkeypatch.setattr(colabcola.profiles, "selected", lambda _: nullcontext())
    opts = argparse.Namespace(profile="test", run_dir=tmp_path / "attempt",
                              preflight_only=False, confirm_allocation=True,
                              max_seconds=300, fps=6, poll_hz=3)

    with pytest.raises(LifecycleError, match="allocation outcome uncertain"):
        bridge.run_colab(opts)

    provider = providers[-1]
    state = json.loads((opts.run_dir / "state.json").read_text(encoding="utf-8"))
    output = capsys.readouterr().out
    assert provider.requests == 1
    assert provider.closed
    assert provider.released == (["owned-test-endpoint"] if accepted_before_timeout else [])
    assert state["phase"] == "released"
    assert state["allocation_failure"] == "provider_read_timeout"
    assert state["allocation_failure_stage"] == "assignment"
    assert isinstance(state["allocation_failed_at"], (int, float))
    assert "SECRET_SHOULD_NOT_LEAK" not in output
    assert "SECRET_SHOULD_NOT_LEAK" not in str(state)
    assert "no retry attempted" in output


def test_timeout_does_not_release_unknown_assignment(tmp_path, monkeypatch):
    providers = []

    class ForeignAssignmentProvider:
        def __init__(self):
            self.timed_out = False
            self.released = []
            providers.append(self)

        def assignments(self):
            return [object()] if self.timed_out else []

        def allocate(self, allocation_id, gpu):
            self.timed_out = True
            raise ReadTimeout("SECRET_SHOULD_NOT_LEAK")

        def reconcile(self, allocation_id, gpu):
            return None

        def lookup(self, endpoint):
            return None

        def release(self, endpoint):
            self.released.append(endpoint)
            raise AssertionError("must not release unknown assignment")

        def close(self):
            pass

    import colabcola.profiles
    monkeypatch.setattr(bridge, "GoogleProvider", ForeignAssignmentProvider)
    monkeypatch.setattr(colabcola.profiles, "selected", lambda _: nullcontext())
    opts = argparse.Namespace(profile="test", run_dir=tmp_path / "attempt",
                              preflight_only=False, confirm_allocation=True,
                              max_seconds=300, fps=6, poll_hz=3)
    with pytest.raises(LifecycleError, match="owned T4 release not confirmed"):
        bridge.run_colab(opts)

    provider = providers[-1]
    journal = json.loads((opts.run_dir / "state.json").read_text(encoding="utf-8"))
    assert provider.released == []
    assert journal["phase"] == "needs_cleanup"
    assert journal["allocation_failure"] == "provider_read_timeout"
    assert journal["cleanup_error"] == "LifecycleError"
