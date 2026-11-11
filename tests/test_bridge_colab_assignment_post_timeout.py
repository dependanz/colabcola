"""Scoped Colab assignment POST timeout, with audited vendored sources intact."""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from requests.exceptions import ReadTimeout

from colabcola._vendor.colab_cli.client import Assignment, COLAB_XSRF_TOKEN_HEADER
from colabcola.lifecycle_provider import GoogleProvider, verify_provider, _verify_bundle


def make_provider(assignment):
    calls = []

    class StubClient:
        def assign(self, *args, **kwargs):
            raise AssertionError("Vendored assign must not be invoked by bridge")

        def _get_assignment(self, *args, **kwargs):
            calls.append(("get", args, kwargs))
            return assignment

        def _build_assign_url(self, *args, **kwargs):
            calls.append(("url", args, kwargs))
            return "https://colab.research.google.com/fixture-only"

        def _issue_request(self, url, **kwargs):
            calls.append(("post", url, kwargs))
            return SimpleNamespace(endpoint="owned-t4")

    provider = GoogleProvider.__new__(GoogleProvider)
    provider.client = StubClient()
    return provider, calls


def test_vendor_source_and_provenance_still_verify():
    verify_provider()
    manifest = _verify_bundle()
    assert manifest["schema"] == 1


def test_assignment_get_uses_unmodified_timeout_and_posts_only_once():
    provider, calls = make_provider(SimpleNamespace(token="test-xsrf"))
    allocation_id = str(uuid.uuid4())
    assert provider.allocate(allocation_id, "T4") == "owned-t4"
    assert [c[0] for c in calls] == ["get", "url", "post"]
    identifier, variant, accelerator, shape = calls[0][1]
    assert identifier == uuid.UUID(allocation_id)
    assert variant.value == "GPU"
    assert accelerator.value == "T4"
    assert shape is None
    assert calls[0][2] == {}
    assert calls[1][1] == calls[0][1]
    request = calls[2][2]
    assert request["method"] == "POST"
    assert request["timeout"] == 120.0
    assert request["headers"][COLAB_XSRF_TOKEN_HEADER["key"]] == "test-xsrf"
    assert request["schema"].__name__ == "PostAssignmentResponse"


def test_preexisting_owned_assignment_never_posts():
    existing = Assignment.model_construct(endpoint="existing-owned-t4")
    provider, calls = make_provider(existing)
    assert provider.allocate(str(uuid.uuid4()), "T4") == "existing-owned-t4"
    assert len(calls) == 1
    assert calls[0][0] == "get"


def test_uncertain_assignment_post_is_not_retried():
    provider, calls = make_provider(SimpleNamespace(token="test-xsrf"))
    def fail_once(url, **kwargs):
        calls.append(("post", url, kwargs))
        raise ReadTimeout("SECRET_SHOULD_NOT_BE_LOGGED")
    provider.client._issue_request = fail_once
    with pytest.raises(ReadTimeout):
        provider.allocate(str(uuid.uuid4()), "T4")
    assert [c[0] for c in calls] == ["get", "url", "post"]


def test_assignment_get_failure_never_posts():
    provider, calls = make_provider(SimpleNamespace(token="ignored"))
    def fail_get(*args, **kwargs):
        calls.append(("get", args, kwargs))
        raise ReadTimeout("GET timed out")
    provider.client._get_assignment = fail_get
    with pytest.raises(ReadTimeout):
        provider.allocate(str(uuid.uuid4()), "T4")
    assert [c[0] for c in calls] == ["get"]
