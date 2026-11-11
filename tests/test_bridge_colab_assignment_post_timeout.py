"""Scoped Colab assignment POST timeout; no network or Colab allocation."""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from colabcola._vendor.colab_cli.client import Client
from colabcola.lifecycle_provider import GoogleProvider


def test_bridge_provider_overrides_assignment_post_only():
    provider = GoogleProvider.__new__(GoogleProvider)
    observed = []

    class StubClient:
        def assign(self, notebook_hash, **options):
            observed.append((notebook_hash, options))
            return SimpleNamespace(endpoint="owned-runtime")

    provider.client = StubClient()
    allocation_id = str(uuid.uuid4())
    assert provider.allocate(allocation_id, "T4") == "owned-runtime"
    assert len(observed) == 1
    request_id, options = observed[0]
    assert request_id == uuid.UUID(allocation_id)
    assert options["post_timeout"] == 120.0
    assert options["accelerator"].value == "T4"


@pytest.mark.parametrize("override", [None, 120.0])
def test_client_assign_preserves_default_and_forwards_override(monkeypatch, override):
    monkeypatch.setenv("COLAB_CLI_ASSIGN_RETRIES", "0")
    client = Client.__new__(Client)
    observed = []
    client._get_assignment = lambda *args: SimpleNamespace(token="xsrf")
    client._post_assignment = lambda *args, **kwargs: (
        observed.append(kwargs) or SimpleNamespace(endpoint="owned-runtime")
    )
    if override is None:
        result = client.assign(uuid.uuid4())
        assert observed == [{}]
    else:
        result = client.assign(uuid.uuid4(), post_timeout=override)
        assert observed == [{"post_timeout": 120.0}]
    assert result.endpoint == "owned-runtime"


def test_only_assignment_post_http_request_receives_override():
    client = Client.__new__(Client)
    observed = []
    client._build_assign_url = lambda *args: "https://colab.research.google.com/tun/m/assign"
    client._issue_request = lambda url, **kwargs: (
        observed.append(kwargs) or SimpleNamespace(token="xsrf")
    )
    allocation_id = uuid.uuid4()
    client._get_assignment(allocation_id)
    client._post_assignment(allocation_id, "xsrf", post_timeout=120.0)
    client._post_assignment(allocation_id, "xsrf")
    assert "timeout" not in observed[0]
    assert observed[0]["schema"] is not None
    assert observed[1]["method"] == "POST"
    assert observed[1]["timeout"] == 120.0
    assert observed[2]["method"] == "POST"
    assert "timeout" not in observed[2]
