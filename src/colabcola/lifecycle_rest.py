"""Opt-in documented Colab REST API adapter. No silent runtime-version fallback.

https://developers.google.com/colab/api/reference/rest/v1beta/runtimes
Creation is asynchronous and requestId/runtimeId bind uncertain calls to one intent.
Only operation identifiers and public request fields are persisted, never connection tokens.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from types import SimpleNamespace
import uuid
from urllib.parse import urlparse

from .lifecycle_io import LifecycleError, AllocationRejected, atomic
from .lifecycle_provider import GoogleProvider, credentials, verify_provider

BASE = 'https://colaboratory.googleapis.com'


class RestGoogleProvider(GoogleProvider):
    def __init__(self, plan, session=None):
        self.plan = plan
        self.root = Path(plan['directory']) / 'allocations'
        if session is None:
            from google.auth.transport.requests import AuthorizedSession
            verify_provider()  # Reuse only the inspected Jupyter client for bootstrap.
            session = AuthorizedSession(credentials())
        self.http = session

    def request(self, method, path, *, missing=False, **kwargs):
        try:
            response = self.http.request(method, BASE + path, timeout=(10, 40),
                                         allow_redirects=False, **kwargs)
        except Exception:
            raise LifecycleError('Colab REST connection unavailable') from None
        if response.status_code == 404 and missing:
            return None
        if response.status_code >= 400 or 300 <= response.status_code < 400:
            # Never serialize response bodies/URLs, which may contain credentials.
            error = LifecycleError('Colab REST HTTP ' + str(response.status_code))
            error.http_status = response.status_code
            raise error
        try:
            value = response.json()
            if not isinstance(value, dict):
                raise ValueError()
            return value
        except ValueError:
            raise LifecycleError('invalid Colab REST response') from None

    def name(self, allocation_id):
        value = uuid.UUID(allocation_id)
        if value.version != 4:
            raise LifecycleError('allocation id must be UUID4')
        return 'runtimes/cc-' + str(value)

    def record_path(self, allocation_id):
        self.name(allocation_id)
        return self.root / (allocation_id + '.json')

    def lookup(self, endpoint):
        if not re.fullmatch(r'runtimes/[a-z][a-z0-9-]{0,62}', endpoint):
            raise LifecycleError('invalid owned REST runtime name')
        value = self.request('GET', '/v1beta/' + endpoint, missing=True)
        if value is not None and value.get('name') != endpoint:
            raise LifecycleError('Colab REST runtime identity mismatch')
        return value

    def expected(self, gpu):
        return {'variant': 'VARIANT_CPU' if gpu == 'NONE' else 'VARIANT_GPU',
                'accelerator': gpu,
                'shape': 'SHAPE_HIGHMEM' if self.plan.get('high_mem') else 'SHAPE_STANDARD'}

    def validate_runtime(self, value, gpu):
        if value.get('runtimeSpec') != self.expected(gpu):
            raise AllocationRejected('allocated runtime specification differs from request')
        version = self.plan.get('runtime_version', 'latest')
        if version != 'latest' and value.get('version') != version:
            raise AllocationRejected('allocated runtime version differs from requested date')

    def operation(self, record, path):
        name = record.get('operation')
        if not name:
            return None
        if not re.fullmatch(r'operations/[A-Za-z0-9_./-]+', name) or '..' in name:
            raise LifecycleError('invalid Colab operation identity')
        op = self.request('GET', '/v1/' + name)
        if op.get('done') and op.get('error'):
            record['error_code'] = op['error'].get('code')
            record['state'] = 'unresolved' if record['error_code'] == 6 else 'rejected'
            atomic(path, record)
            if record['error_code'] == 6:
                raise LifecycleError('Colab REST existing resource not yet reconciled')
            raise AllocationRejected('Colab REST allocation rejected (code ' + str(record['error_code']) + ')')
        return op

    def assignments(self):
        values, token = [], None
        while True:
            params = {"pageSize":100}
            if token:
                params["pageToken"] = token
            response = self.request("GET", "/v1beta/runtimes", params=params)
            for item in response.get("runtimes", []):
                if item.get("state") not in ("DELETED", "STATE_DELETED"):
                    values.append(SimpleNamespace(endpoint=item["name"]))
            if len(values) > 10000:
                raise LifecycleError("too many Colab assignments")
            next_token = response.get("nextPageToken")
            if not next_token:
                return values
            if next_token == token:
                raise LifecycleError("Colab listing pagination stalled")
            token = next_token

    def allocate(self, allocation_id, gpu):
        name = self.name(allocation_id)
        path = self.record_path(allocation_id)
        prior_intent = path.exists()
        record = json.loads(path.read_text()) if prior_intent else {'name': name, 'state': 'intent'}
        try:
            value = self.lookup(name)
        except LifecycleError as exc:
            if not prior_intent and getattr(exc, 'http_status', None) in (400, 401, 403):
                atomic(path, {'name': name, 'state': 'not_created'})
                raise AllocationRejected(str(exc)) from None
            raise
        if value is not None:
            self.validate_runtime(value, gpu)
            return name
        if record['state'] in ('rejected', 'not_created'):
            raise AllocationRejected('Colab REST allocation previously rejected')
        if record.get('operation'):
            self.operation(record, path)
            raise LifecycleError('Colab REST allocation pending')
        body = {'runtimeSpec': self.expected(gpu)}
        if self.plan.get('runtime_version', 'latest') != 'latest':
            body['version'] = self.plan['runtime_version']
        atomic(path, record)
        try:
            op = self.request('POST', '/v1beta/runtimes',
                params={'requestId': allocation_id, 'runtimeId': name.split('/')[1]}, json=body)
        except LifecycleError as exc:
            if not prior_intent and getattr(exc, 'http_status', None) in (400, 401, 403):
                record['state'] = 'rejected'
                atomic(path, record)
                raise AllocationRejected(str(exc)) from None
            raise
        if op.get('done') and op.get('error'):
            record['error_code'] = op['error'].get('code')
            record['state'] = 'unresolved' if record['error_code'] == 6 else 'rejected'
            atomic(path, record)
            # An existing resource may follow an uncertain earlier request; reconcile by name.
            if record['error_code'] == 6:
                value = self.lookup(name)
                if value is not None:
                    self.validate_runtime(value, gpu)
                    return name
            if record['error_code'] == 6:
                raise LifecycleError('Colab REST existing resource not yet reconciled')
            raise AllocationRejected('Colab REST allocation rejected (code ' + str(record['error_code']) + ')')
        if not isinstance(op.get('name'), str):
            raise LifecycleError('Colab REST create omitted operation identity')
        record.update(operation=op['name'], state='pending')
        atomic(path, record)
        value = self.lookup(name)
        if value is not None:
            self.validate_runtime(value, gpu)
            return name
        raise LifecycleError('Colab REST allocation pending')

    def reconcile(self, allocation_id, gpu):
        name = self.name(allocation_id)
        path = self.record_path(allocation_id)
        if path.exists() and json.loads(path.read_text())['state'] == 'not_created':
            return None  # Definitive rejection before any POST, even if GET is denied.
        if self.lookup(name) is not None:
            return name  # Even mismatched owned allocations must be released.
        path = self.record_path(allocation_id)
        if not path.exists():
            return None  # No create intent was sent by this adapter.
        record = json.loads(path.read_text())
        if record['state'] == 'rejected':
            return None
        try:
            op = self.operation(record, path)
        except AllocationRejected:
            return None
        if op and op.get('done') and not op.get('error'):
            return name  # DELETE allowMissing reconciles a completed creation.
        raise LifecycleError('owned REST allocation outcome unresolved; cleanup must continue')

    def release(self, endpoint):
        self.lookup(endpoint)  # Validate exact resource name, never a broad deletion.
        self.request('DELETE', '/v1beta/' + endpoint, params={'allowMissing': 'true'})
        if self.lookup(endpoint) is not None:
            raise LifecycleError('owned REST runtime deletion not yet confirmed')

    def _session(self, endpoint):
        value = self.lookup(endpoint)
        if value is None:
            raise LifecycleError('owned runtime is absent')
        info = value.get('connectionInfo', {})
        try:
            expiry = datetime.fromisoformat(info['expireTime'].replace('Z', '+00:00'))
            url = urlparse(info['url'])
            if (expiry <= datetime.now(timezone.utc) or url.scheme != 'https'
                    or not url.hostname or url.username or url.password or not info['token']):
                raise ValueError()
            return SimpleNamespace(url=info['url'], token=info['token'], endpoint=endpoint)
        except (KeyError, ValueError, TypeError):
            raise LifecycleError('runtime connection information unavailable or expired') from None
