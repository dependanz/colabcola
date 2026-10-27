"""Host-only renewable access tokens; credentials never enter journals."""
import math
import threading
import time
from .lifecycle_io import LifecycleError


class TokenLease:
    """Cache short-lived access only; refresh well before runtime expiry protection."""
    def __init__(self, loader, clock=time.time, refresh_margin=600):
        self.loader, self.clock, self.margin = loader, clock, refresh_margin
        self._value = None
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            if self._value is None or self._value['expires_at'] <= self.clock() + self.margin:
                value = self.loader()
                if (not isinstance(value, dict) or not isinstance(value.get('access_token'), str)
                        or not value['access_token'] or type(value.get('expires_at')) not in (int, float)
                        or not math.isfinite(value['expires_at']) or not value['expires_at'] > self.clock() + self.margin):
                    raise LifecycleError('Drive helper must return more than 600 seconds of access')
                # Discard any unexpected helper fields, especially refresh credentials.
                self._value = {k: value[k] for k in ('access_token', 'expires_at')}
            return dict(self._value)


def _expiry(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _worker_confirms(worker, attempt, expiry, now):
    if not isinstance(worker, dict) or type(worker.get("attempt")) is not int:
        return False
    evidence = worker.get("authorization", {})
    if not isinstance(evidence, dict):
        return False
    publication = evidence.get("last_verified_publication", {})
    if not isinstance(publication, dict):
        return False
    at = _expiry(publication.get("at"))
    count = evidence.get("verified_publications")
    return (worker.get("attempt") == attempt and expiry is not None
            and _expiry(evidence.get("latest_expires_at")) == expiry
            and type(count) is int and count > 0 and at is not None
            and 0 < at <= now + 5 and at < expiry and expiry > now + 120)


def authorization_health(state, renewal=None, *, now=None):
    """Separate last confirmed runtime access from a host delivery/refresh retry.

    Worker evidence can confirm a lost acknowledgement only for an expiry already
    known to the host. It cannot invent or extend a credential's lifetime.
    """
    now = time.time() if now is None else now
    attempts = state.get("attempts", [])
    if not attempts:
        return dict(state.get("drive_authorization", {}))
    number = attempts[-1]["number"]
    old = state.get("drive_authorization", {})
    if old.get("attempt") not in (None, number):
        old = {}
    update = renewal or {}
    if update.get("attempt") != number:
        update = {}
    result = {k: v for k, v in old.items() if k not in ("state", "error_type")}
    result.update({k: v for k, v in update.items() if k not in ("state", "error_type", "expires_at")})
    confirmed = _expiry(old.get("expires_at"))
    delivered = _expiry(update.get("expires_at"))
    if delivered is not None:
        confirmed = max(confirmed or 0, delivered)
    candidate = _expiry(result.get("credential_expires_at")) or confirmed
    delivery = dict(result.get("delivery", {}))
    worker = state.get("worker", {})
    if _worker_confirms(worker, number, candidate, now):
        confirmed = max(confirmed or 0, candidate)
        if delivery.get("stage") != "credentials":
            if delivery.get("state") == "retrying":
                result["last_delivery_error"] = delivery
            delivery = {"state": "confirmed", "source": "worker",
                        "at": worker["authorization"]["last_verified_publication"]["at"]}
    if delivery:
        result["delivery"] = delivery
    result["attempt"] = number
    result["expires_at"] = confirmed if confirmed is not None else 0
    result["state"] = ("valid" if confirmed is not None and confirmed > now + 120
                       else "expiring" if confirmed is not None else "unavailable")
    return result


class RenewalPump:
    """Refresh independently; do not open a kernel connection for an unchanged lease."""
    def __init__(self, engine, provider, token, interval=60, *, clock=time.time):
        self.engine, self.provider, self.token, self.interval = engine, provider, token, interval
        self.clock = clock
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True, name='colabcola-drive-renewal')
        self.health = {'state': 'waiting'}
        self._delivered = None  # Private token identity, never copied into health/journals.
        self._stage = 'credentials'

    def once(self):
        state = self.engine.state
        if state['phase'] != 'monitoring' or not state.get('attempts'):
            return
        attempt = dict(state['attempts'][-1])
        if attempt.get('released') or not attempt.get('endpoint'):
            return
        if self.health.get('attempt') != attempt['number']:
            self.health = {'attempt': attempt['number']}
        self._stage = 'credentials'
        value = self.token()
        identity = (attempt['number'], attempt['endpoint'], value['access_token'], value['expires_at'])
        if identity == self._delivered:
            # A recovered credential helper may return the same confirmed lease.
            # Clear its transient refresh error without another runtime delivery.
            at = self.health.get("at", self.clock())
            self.health = {'state': 'renewed', 'expires_at': value['expires_at'],
                           'credential_expires_at': value['expires_at'], 'at': at,
                           'attempt': attempt['number'],
                           'delivery': {'state': 'confirmed', 'source': 'previous_delivery', 'at': at}}
            return
        self.health = {**self.health, 'credential_expires_at': value['expires_at']}
        if _worker_confirms(state.get('worker', {}), attempt['number'], value['expires_at'], self.clock()):
            source = 'worker'
        else:
            self._stage = 'delivery'
            self.provider.refresh_token(attempt['endpoint'], self.engine.plan, attempt, value)
            source = 'delivery'
        self._delivered = identity
        self.health = {'state': 'renewed', 'expires_at': value['expires_at'],
                       'credential_expires_at': value['expires_at'], 'at': self.clock(),
                       'attempt': attempt['number'],
                       'delivery': {'state': 'confirmed', 'source': source, 'at': self.clock()}}

    def failed(self, exc):
        self.health = {**self.health, 'state': 'retrying',
                       'delivery': {'state': 'retrying', 'stage': self._stage,
                                    'error_type': type(exc).__name__, 'at': self.clock()}}

    def run(self):
        while not self.stop_event.wait(self.interval):
            try:
                self.once()
            except Exception as exc:
                self.failed(exc)  # No exception text, URLs or credentials in public health.

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=5)
