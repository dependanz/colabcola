from __future__ import annotations
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from colabcola.lifecycle_io import (LifecycleError, atomic, digest, encoded, publish,
                                   recover, snapshots, read_regular)
from colabcola.lifecycle import Supervisor, run_lock, verify_final, recover_terminal_final, reported_final
from colabcola.lifecycle_worker import checkpoint_members
from colabcola import JobSpec, build_bundle, bind_job_arguments
from test_notebook import notebook_job


class MemoryStore:
    def __init__(self):
        self.objects = {}
        self.fail_kind = None
    def put(self, kind, name, body):
        if self.fail_kind == kind:
            raise LifecycleError("injected transfer failure")
        key = digest(body)
        self.objects[key] = (kind, name, body)
        return {"id": key, "sha256": key, "bytes": len(body)}
    def get(self, ref):
        body = self.objects[ref["id"]][2]
        if digest(body) != ref["sha256"] or len(body) != ref["bytes"]:
            raise LifecycleError("injected corrupt object")
        return body
    def list(self, kind):
        return [{"id": key, "size": str(len(body)), "appProperties": {"cc_sha": key}}
                for key, (k, name, body) in self.objects.items() if k == kind]


class SnapshotCatalogTests(unittest.TestCase):
    def setUp(self):
        self.store=MemoryStore()
        self.identity={'run_id':'a'*32,'job_id':'b'*64}

    def test_long_run_fetches_only_new_manifests_and_does_not_cache_mutable_commits(self):
        for index in range(53):
            publish(self.store,self.identity,'logs',{'stdout.log':str(index).encode()},attempt=1)
        with mock.patch.object(self.store,'get',wraps=self.store.get) as get:
            values=snapshots(self.store,self.identity,'logs')
            self.assertEqual(get.call_count,53)
            values[-1]['commit']['job_id']='caller mutation'
            again=snapshots(self.store,self.identity,'logs')
            self.assertTrue(all(v['commit']['job_id']==self.identity['job_id'] for v in again))
            self.assertEqual(get.call_count,53)
            publish(self.store,self.identity,'logs',{'stdout.log':b'new'},attempt=1)
            latest=snapshots(self.store,self.identity,'logs')
            self.assertEqual(len(latest),54)
            self.assertEqual(get.call_count,54)

    def test_changed_reference_and_wrong_identity_are_reverified(self):
        snapshot=publish(self.store,self.identity,'logs',{'stdout.log':b'ok'},attempt=1)
        snapshots(self.store,self.identity,'logs')
        records=self.store.list('logs');records[0]['appProperties']['cc_sha']='f'*64
        with mock.patch.object(self.store,'list',return_value=records), self.assertRaises(LifecycleError):
            snapshots(self.store,self.identity,'logs')
        with self.assertRaisesRegex(LifecycleError,'identity mismatch'):
            snapshots(self.store,{**self.identity,'job_id':'c'*64},'logs')

    def test_disappeared_catalog_entry_is_not_reused_or_retained(self):
        publish(self.store,self.identity,'logs',{'stdout.log':b'ok'},attempt=1)
        snapshots(self.store,self.identity,'logs')
        with mock.patch.object(self.store,'list',return_value=[]):
            self.assertEqual(snapshots(self.store,self.identity,'logs'),[])
        with mock.patch.object(self.store,'get',wraps=self.store.get) as get:
            snapshots(self.store,self.identity,'logs')
            self.assertEqual(get.call_count,1)

class Provider:
    def __init__(self):
        self.allocations, self.active, self.boots, self.releases = {}, set(), [], []
        self.timeout_after_allocation = False
        self.lookup_error = False
    def allocate(self, allocation_id, gpu):
        endpoint = self.allocations.setdefault(allocation_id, "vm-" + str(len(self.allocations) + 1))
        self.active.add(endpoint)
        if self.timeout_after_allocation:
            self.timeout_after_allocation = False
            raise TimeoutError("secret URL must not appear")
        return endpoint
    def reconcile(self, allocation_id, gpu):
        endpoint = self.allocations.get(allocation_id)
        return endpoint if endpoint in self.active else None
    def lookup(self, endpoint):
        if self.lookup_error:
            raise TimeoutError("do not allocate on timeout")
        return endpoint if endpoint in self.active else None
    def release(self, endpoint):
        self.active.discard(endpoint)
        self.releases.append(endpoint)
    def boot(self, endpoint, plan, attempt, token):
        if endpoint not in self.boots:
            self.boots.append(endpoint)
    def observe(self, endpoint, plan, attempt, token):
        return {"state": "running"}


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.identity = {"run_id": "a"*32, "job_id": "b"*64}
        self.store, self.provider = MemoryStore(), Provider()
        self.now = 1000.0

    def engine(self, fault=False):
        for name in ("notebook.ipynb", "lifecycle_worker.py", "lifecycle_io.py"):
            (self.root/name).write_bytes(b"fixture")
        self.plan = {**self.identity, "directory": str(self.root), "max_attempts": 2,
                     "max_seconds": 100, "fault_after_checkpoint": fault, "gpu": "NONE", "loss_grace_seconds": 0,
                     "files": {n: digest(b"fixture") for n in
                               ("notebook.ipynb", "lifecycle_worker.py", "lifecycle_io.py")}}
        atomic(self.root/"plan.json", self.plan)
        atomic(self.root/"state.json", {"phase":"prepared","attempts":[],"events":[]})
        return self.reload()

    def reload(self):
        return Supervisor(self.root, self.provider, self.store, lambda: {"access_token":"test"},
                          clock=lambda: self.now)

    def running(self, fault=False):
        engine = self.engine(fault)
        for _ in range(3):
            engine.advance()
        self.assertEqual(engine.state["phase"], "monitoring")
        return engine

    def test_commit_is_not_visible_when_member_upload_fails(self):
        self.store.fail_kind = "member"
        with self.assertRaises(LifecycleError):
            publish(self.store, self.identity, "checkpoint", {"state.pt": b"state"}, step=2)
        self.assertEqual(snapshots(self.store, self.identity, "checkpoint"), [])

    def test_commit_is_not_visible_when_commit_upload_fails(self):
        self.store.fail_kind = "checkpoint"
        with self.assertRaises(LifecycleError):
            publish(self.store, self.identity, "checkpoint", {"state.pt": b"state"}, step=2)
        self.assertTrue(self.store.list("member"))
        self.assertEqual(snapshots(self.store, self.identity, "checkpoint"), [])

    def test_corrupt_restore_does_not_publish_destination(self):
        snap = publish(self.store, self.identity, "checkpoint", {"state.pt": b"state"}, step=2)
        key = snap["commit"]["members"]["state.pt"]["id"]
        self.store.objects[key] = ("member","state.pt",b"corrupt")
        with self.assertRaises(LifecycleError):
            recover(self.store, snap, self.root/"restore")
        self.assertFalse((self.root/"restore").exists())

    def test_restore_rejects_traversal(self):
        snap = {"commit":{"members":{"../escape": {"id":"x"}}}}
        with self.assertRaises(LifecycleError):
            recover(self.store, snap, self.root/"restore")

    def test_snapshot_identity_is_bound(self):
        publish(self.store, self.identity, "checkpoint", {"s": b"x"}, step=2)
        with self.assertRaises(LifecycleError):
            snapshots(self.store, {**self.identity,"job_id":"wrong"}, "checkpoint")

    def test_unknown_allocation_retries_same_identity_after_restart(self):
        engine = self.engine()
        engine.advance()
        self.provider.timeout_after_allocation = True
        engine.advance()
        self.assertEqual(engine.state["phase"], "allocating")
        reloaded = self.reload()
        reloaded.advance()
        self.assertEqual(len(self.provider.allocations), 1)
        self.assertEqual(reloaded.state["phase"], "booting")
        self.assertNotIn("secret URL", json.dumps(reloaded.state))

    def test_monitor_restart_does_not_allocate_or_launch_again(self):
        self.running()
        self.reload().advance()
        self.assertEqual(len(self.provider.allocations), 1)
        self.assertEqual(len(self.provider.boots), 1)


    def test_kernel_reply_failure_releases_owned_runtime_and_retries_checkpoint(self):
        engine = self.running()
        engine.plan["execution_mode"] = "kernel-v1"
        atomic(self.root / "plan.json", engine.plan)
        snapshot = publish(self.store, self.identity, "checkpoint",
                           {"state.pt": b"verified"}, step=9)
        self.provider.active.add("unrelated")
        self.provider.observe = mock.Mock(return_value={
            "state": "running", "attempt": 1, "heartbeat_at": 998.0,
            "kernel_execution": {"request_state": "failed",
                                 "worker_execution_state": "running",
                                 "reply_error_type": "KeyboardInterrupt"}})
        engine.advance()
        self.assertEqual(engine.state["phase"], "recovering")
        self.assertEqual(self.provider.releases, ["vm-1"])
        self.assertEqual(self.provider.active, {"unrelated"})
        self.assertTrue(engine.state["attempts"][0]["released"])
        self.assertEqual(engine.state["worker"]["observed_at"], 1000.0)
        failure = engine.state["last_failure"]
        self.assertEqual(failure["cause"], "worker_failed")
        self.assertEqual(failure["error_type"], "KeyboardInterrupt")
        self.assertEqual(failure["last_heartbeat"], 998.0)
        self.assertEqual(failure["signal"], "kernel_reply_failed")
        event = [row for row in engine.state["events"]
                 if row["event"] == "worker_failed"]
        self.assertEqual(len(event), 1)
        self.assertEqual(event[0]["observed_at"], 1000.0)
        self.assertEqual(len(self.provider.boots), 1)
        restarted = self.reload()
        restarted.advance()
        self.assertEqual(restarted.state["phase"], "allocating")
        self.assertEqual(restarted.state["attempts"][-1]["resume"], snapshot)
        restarted.advance()
        restarted.advance()
        self.assertEqual(len(self.provider.boots), 2)
        self.assertEqual(len(self.provider.allocations), 2)

    def assert_owned_worker_receipt_failure(self, receipt, reply, signal):
        engine = self.running()
        engine.plan["execution_mode"] = "kernel-v1"
        engine.plan["max_attempts"] = 1
        self.provider.observe = mock.Mock(return_value={
            "state": "running", "attempt": 1,
            "kernel_execution": {"request_state": reply,
                                 "worker_execution_state": receipt,
                                 "worker_error_type": "RuntimeError",
                                 "worker_finished_at": 900.25}})
        engine.advance()
        self.assertEqual(engine.state["phase"], "failed")
        self.assertEqual(engine.state["reason"],
                         "worker_failed_attempt_budget_exhausted")
        self.assertEqual(engine.state["last_failure"]["signal"], signal)
        self.assertEqual(engine.state["last_failure"]["finished_at"], 900.25)
        self.assertFalse(self.provider.active)

    def test_missing_receipt_with_terminal_status_fails_owned_attempt(self):
        engine = self.running()
        engine.plan["execution_mode"] = "kernel-v1"
        engine.plan["max_attempts"] = 1
        atomic(self.root / "plan.json", engine.plan)
        self.provider.observe = mock.Mock(return_value={
            "state": "failed", "attempt": 1,
            "kernel_execution": {
                "request_state": "receipt_missing",
                "worker_execution_state": "unknown"}})
        engine.advance()
        self.assertEqual(engine.state["phase"], "failed")
        self.assertEqual(engine.state["reason"], "runtime_failed")
        self.assertEqual(self.provider.releases, ["vm-1"])
        self.assertFalse(self.provider.active)
        self.assertEqual(len(self.provider.allocations), 1)
        changes = [x for x in engine.state["events"]
                   if x["event"] == "kernel_execution_changed"]
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["kernel_execution"]["request_state"],
                         "receipt_missing")
        self.assertEqual(changes[0]["observed_at"], self.now)
        self.assertFalse(any(x["event"] == "worker_failed"
                             for x in engine.state["events"]))

    def test_failed_worker_receipt_releases_attempt(self):
        self.assert_owned_worker_receipt_failure(
            "failed", "busy", "worker_receipt_failed")

    def test_finished_without_final_commit_releases_attempt(self):
        self.assert_owned_worker_receipt_failure(
            "finished", "completed", "finished_without_final")

    def test_uncertain_worker_failure_release_cannot_allocate_replacement(self):
        engine = self.running()
        engine.plan["execution_mode"] = "kernel-v1"
        atomic(self.root / "plan.json", engine.plan)
        self.provider.observe = mock.Mock(return_value={
            "state": "running", "attempt": 1,
            "kernel_execution": {"request_state": "busy",
                                 "worker_execution_state": "failed"}})
        original = self.provider.release
        self.provider.release = mock.Mock(side_effect=LifecycleError("release uncertain"))
        engine.advance()
        self.assertEqual(engine.state["phase"], "releasing_failed_worker")
        self.assertEqual(len(self.provider.allocations), 1)
        self.assertIn("vm-1", self.provider.active)
        self.assertEqual(len([e for e in engine.state["events"]
                              if e["event"] == "worker_failed"]), 1)
        self.provider.release = original
        restored = self.reload()
        restored.advance()
        self.assertEqual(restored.state["phase"], "recovering")
        self.assertEqual(len(self.provider.allocations), 1)
        restored.advance()
        self.assertEqual(restored.state["phase"], "allocating")
        self.assertEqual(len(self.provider.allocations), 1)

    def test_worker_failure_requires_verified_release_absence(self):
        engine = self.running()
        engine.plan["execution_mode"] = "kernel-v1"
        self.provider.observe = mock.Mock(return_value={
            "state": "running", "attempt": 1,
            "kernel_execution": {"request_state": "failed",
                                 "worker_execution_state": "failed"}})
        original = self.provider.release
        self.provider.release = mock.Mock(return_value=None)
        engine.advance()
        self.assertEqual(engine.state["phase"], "releasing_failed_worker")
        self.assertEqual(len(self.provider.allocations), 1)
        self.assertIn("vm-1", self.provider.active)
        self.provider.release = original
        engine.advance()
        self.assertEqual(engine.state["phase"], "recovering")
        self.assertTrue(engine.state["attempts"][0]["released"])

    def test_kernel_failure_without_owned_evidence_does_not_release(self):
        engine = self.running()
        # Legacy plans have no tracked kernel execution contract.
        self.provider.observe = mock.Mock(return_value={
            "state": "running",
            "kernel_execution": {"request_state": "failed",
                                 "worker_execution_state": "failed"}})
        engine.advance()
        self.assertEqual(engine.state["phase"], "monitoring")
        self.assertFalse(self.provider.releases)
        engine.plan["execution_mode"] = "kernel-v1"
        self.provider.observe.return_value = {
            "state": "running",
            "kernel_execution": {"request_state": "attached",
                                 "worker_execution_state": "running"}}
        engine.advance()
        self.assertEqual(engine.state["phase"], "monitoring")
        self.assertFalse(self.provider.releases)

    def test_reported_final_precedes_failed_kernel_reply(self):
        engine = self.running()
        engine.plan["execution_mode"] = "kernel-v1"
        final = self.final_snapshot(1, 0)
        ref = {key: final[key] for key in ("id", "bytes", "sha256")}
        self.provider.observe = mock.Mock(return_value={
            "state": "running", "attempt": 1, "final_commit": ref,
            "kernel_execution": {"request_state": "failed",
                                 "worker_execution_state": "failed"}})
        with mock.patch.object(self.store, "list", return_value=[]):
            engine.advance()
            self.assertEqual(engine.state["phase"], "monitoring")
            self.assertFalse(self.provider.releases)
            engine.advance()
        self.assertEqual(engine.state["phase"], "completed")
        self.assertFalse(any(row["event"] == "worker_failed"
                             for row in engine.state["events"]))

    def test_kernel_absence_requires_repeated_observation_and_grace(self):
        engine = self.running()
        engine.plan["execution_mode"] = "kernel-v1"
        engine.plan["max_attempts"] = 1
        atomic(self.root / "plan.json", engine.plan)
        self.provider.observe = mock.Mock(return_value={
            "state": "running", "attempt": 1,
            "kernel_execution": {
                "request_state": "kernel_absent",
                "worker_execution_state": "running"}})
        engine.advance()
        self.assertEqual(engine.state["phase"], "monitoring")
        self.assertEqual(engine.state["kernel_absence"]["observations"], 1)
        self.assertEqual(self.provider.releases, [])
        self.now += 10
        second = self.reload()
        second.advance()
        self.assertEqual(second.state["phase"], "monitoring")
        self.assertEqual(second.state["kernel_absence"]["observations"], 2)
        self.assertEqual(self.provider.releases, [])
        self.now += 20
        restarted = self.reload()
        restarted.advance()
        self.assertEqual(restarted.state["phase"], "failed")
        self.assertEqual(restarted.state["reason"], "worker_failed_attempt_budget_exhausted")
        self.assertEqual(restarted.state["last_failure"]["signal"], "kernel_absent_confirmed")
        self.assertEqual(self.provider.releases, ["vm-1"])
        self.assertEqual(len(self.provider.allocations), 1)
        self.assertEqual(len([e for e in restarted.state["events"]
                              if e["event"] == "kernel_absent_observed"]), 1)

    def test_same_kernel_reappears_before_grace_without_release(self):
        engine = self.running()
        engine.plan["execution_mode"] = "kernel-v1"
        self.provider.observe = mock.Mock(return_value={
            "state": "running", "attempt": 1,
            "kernel_execution": {
                "request_state": "kernel_absent",
                "worker_execution_state": "running"}})
        engine.advance()
        self.assertIn("kernel_absence", engine.state)
        self.now += 15
        self.provider.observe.return_value = {
            "state": "running", "attempt": 1, "heartbeat_at": self.now,
            "kernel_execution": {
                "kernel_id": "kernel", "request_state": "attached",
                "worker_execution_state": "running"}}
        resumed = self.reload()
        resumed.advance()
        self.assertEqual(resumed.state["phase"], "monitoring")
        self.assertNotIn("kernel_absence", resumed.state)
        self.assertEqual(self.provider.releases, [])
        resumed.advance()
        self.assertEqual(len([e for e in resumed.state["events"]
                              if e["event"] == "kernel_absence_cleared"]), 1)

    def test_kernel_listing_failure_must_not_release_runtime(self):
        engine = self.running()
        engine.plan["execution_mode"] = "kernel-v1"
        self.provider.observe = mock.Mock(
            side_effect=LifecycleError("runtime kernel listing HTTP 503"))
        engine.advance()
        self.assertEqual(engine.state["phase"], "monitoring")
        self.assertNotIn("kernel_absence", engine.state)
        self.assertEqual(self.provider.releases, [])

    def test_kernel_execution_transitions_are_journaled_once_per_change(self):
        engine = self.running()
        engine.plan["execution_mode"] = "kernel-v1"
        atomic(self.root / "plan.json", engine.plan)
        self.provider.observe = mock.Mock(return_value={
            "state": "running", "attempt": 1,
            "kernel_execution": {"request_state": "submitted",
                                 "worker_execution_state": "running",
                                 "execution_state": "busy"}})
        engine.advance()
        self.assertEqual(len([x for x in engine.state["events"]
                              if x["event"] == "kernel_execution_changed"]), 1)
        self.reload().advance()
        self.assertEqual(len([x for x in engine.state["events"]
                              if x["event"] == "kernel_execution_changed"]), 1)
        self.provider.observe.return_value = {
            "state": "running", "attempt": 1,
            "kernel_execution": {"request_state": "busy",
                                 "worker_execution_state": "running",
                                 "execution_state": "busy"}}
        refreshed = self.reload()
        refreshed.advance()
        self.assertEqual(len([x for x in refreshed.state["events"]
                              if x["event"] == "kernel_execution_changed"]), 2)
        # Reattached to a restarted kernel with the same ID: not failure
        # evidence until a separate receipt/absence/stall signal exists.
        self.provider.observe.return_value = {
            "state": "running", "attempt": 1,
            "kernel_execution": {"request_state": "attached",
                                 "worker_execution_state": "running",
                                 "execution_state": "idle"}}
        refreshed.advance()
        changes = [x for x in refreshed.state["events"]
                   if x["event"] == "kernel_execution_changed"]
        self.assertEqual(len(changes), 3)
        self.assertEqual(changes[-1]["kernel_execution"]["request_state"], "attached")
        self.assertEqual(changes[-1]["observed_at"], self.now)
        self.assertEqual(refreshed.state["phase"], "monitoring")
        self.assertFalse(self.provider.releases)

    def test_heartbeat_stall_warns_once_and_never_releases(self):
        engine = self.running()
        self.provider.active.add("unrelated")
        self.provider.observe = mock.Mock(return_value={
            "state": "running", "attempt": 1, "heartbeat_at": 879.0})
        engine.advance()  # Age 121 s exceeds max(120 s, 10 * 5 s).
        self.assertEqual(engine.state["phase"], "monitoring")
        self.assertEqual(engine.state["worker_stall"]["heartbeat_age"], 121.0)
        rows = [e for e in engine.state["events"] if e["event"] == "worker_stalled"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["observed_at"], 1000.0)
        self.assertEqual(rows[0]["last_heartbeat"], 879.0)
        self.assertEqual(rows[0]["threshold_seconds"], 120.0)
        self.assertEqual(self.provider.releases, [])
        self.assertEqual(len(self.provider.allocations), 1)
        self.assertEqual(self.provider.active, {"vm-1", "unrelated"})

        # Repeat observations and supervisor restart must not repeat a warning.
        engine.advance()
        self.reload().advance()
        persisted = json.loads((self.root / "state.json").read_text())
        self.assertEqual(len([e for e in persisted["events"]
                              if e["event"] == "worker_stalled"]), 1)
        self.assertEqual(self.provider.releases, [])

    def test_stall_clears_only_when_fresh_heartbeat_resumes(self):
        engine = self.running()
        self.provider.observe = mock.Mock(return_value={
            "state": "running", "attempt": 1, "heartbeat_at": 850.0})
        engine.advance()
        self.assertIn("worker_stall", engine.state)

        # Unknown heartbeat does not falsely clear an existing warning.
        self.provider.observe.return_value = {"state": "running", "attempt": 1}
        engine.advance()
        self.assertIn("worker_stall", engine.state)

        self.provider.observe.return_value = {
            "state": "running", "attempt": 1, "heartbeat_at": 999.5}
        restarted = self.reload()
        restarted.advance()
        self.assertEqual(restarted.state["phase"], "monitoring")
        self.assertNotIn("worker_stall", restarted.state)
        self.assertEqual(len([e for e in restarted.state["events"]
                              if e["event"] == "worker_stall_cleared"]), 1)
        restarted.advance()
        self.assertEqual(len([e for e in restarted.state["events"]
                              if e["event"] == "worker_stall_cleared"]), 1)
        self.assertEqual(self.provider.releases, [])

        # The next stale interval begins a new warning episode.
        self.provider.observe.return_value = {
            "state": "running", "attempt": 1, "heartbeat_at": 850.0}
        restarted.advance()
        self.assertEqual(len([e for e in restarted.state["events"]
                              if e["event"] == "worker_stalled"]), 2)

    def test_stall_threshold_respects_declared_poll_interval(self):
        engine = self.running()
        engine.plan["poll_seconds"] = 30
        atomic(self.root / "plan.json", engine.plan)
        self.provider.observe = mock.Mock(return_value={
            "state": "running", "attempt": 1, "heartbeat_at": 700.0})
        engine.advance()
        self.assertNotIn("worker_stall", engine.state)  # Exactly 300s.
        self.provider.observe.return_value = {
            "state": "running", "attempt": 1, "heartbeat_at": 699.0}
        self.reload().advance()
        observed = json.loads((self.root / "state.json").read_text())
        self.assertEqual(observed["worker_stall"]["threshold_seconds"], 300.0)
        self.assertEqual(self.provider.releases, [])

    def test_malformed_or_future_heartbeat_does_not_claim_stall(self):
        engine = self.running()
        for heartbeat in (None, True, "old", float("nan"), float("inf"),
                          -1, 1001.0):
            with self.subTest(heartbeat=heartbeat):
                self.provider.observe = mock.Mock(return_value={
                    "state": "running", "attempt": 1,
                    "heartbeat_at": heartbeat})
                engine.advance()
                self.assertEqual(engine.state["phase"], "monitoring")
                self.assertNotIn("worker_stall", engine.state)
        self.assertFalse(any(e["event"] == "worker_stalled"
                             for e in engine.state["events"]))
        self.assertEqual(self.provider.releases, [])

    def test_terminal_worker_failure_preempts_stall_warning(self):
        engine = self.running()
        engine.plan["execution_mode"] = "kernel-v1"
        engine.plan["max_attempts"] = 1
        self.provider.observe = mock.Mock(return_value={
            "state": "running", "attempt": 1, "heartbeat_at": 700.0,
            "kernel_execution": {"request_state": "failed",
                                 "worker_execution_state": "failed"}})
        engine.advance()
        self.assertEqual(engine.state["phase"], "failed")
        self.assertEqual(len([e for e in engine.state["events"]
                              if e["event"] == "worker_failed"]), 1)
        self.assertFalse(any(e["event"] == "worker_stalled"
                             for e in engine.state["events"]))

    def test_lookup_timeout_does_not_replace_runtime(self):
        engine = self.running()
        self.provider.lookup_error = True
        engine.advance()
        self.assertEqual(engine.state["phase"], "monitoring")
        self.assertEqual(len(self.provider.allocations), 1)
        self.assertEqual(engine.state["connectivity"], "unknown")

    def test_confirmed_loss_requires_two_observations_then_restores_checkpoint(self):
        engine = self.running()
        snapshot = publish(self.store, self.identity, "checkpoint", {"state.pt": b"saved"}, step=3)
        self.provider.active.clear()
        engine.advance()
        self.assertEqual(engine.state["phase"], "monitoring")
        engine.advance()
        self.assertEqual(engine.state["phase"], "recovering")
        engine.advance()
        engine.advance()
        self.assertEqual(len(self.provider.allocations), 2)
        self.assertEqual(engine.state["attempts"][-1]["resume"], snapshot)

    def test_proof_fault_requires_verified_checkpoint_and_only_owned_runtime(self):
        engine = self.running(True)
        self.provider.active.add("unrelated")
        engine.advance()
        self.assertEqual(self.provider.releases, [])
        publish(self.store, self.identity, "checkpoint", {"s":b"checkpoint"}, step=3)
        engine.advance()
        self.assertEqual(self.provider.releases, ["vm-1"])
        self.assertIn("unrelated", self.provider.active)
        self.reload().advance()
        self.assertEqual(self.provider.releases, ["vm-1"])

    def test_budget_cleanup_and_cancel_are_owned_only(self):
        engine = self.running()
        self.provider.active.add("unrelated")
        self.now = 1101
        engine.advance()
        self.assertEqual(engine.state["phase"], "budget_exhausted")
        self.assertEqual(self.provider.active, {"unrelated"})

    def test_cancel_survives_restart(self):
        self.running()
        atomic(self.root/"cancel.request", {})
        engine = self.reload()
        engine.advance()
        self.assertEqual(engine.state["phase"], "cancelled")

    def test_lock_excludes_duplicate_supervisors(self):
        with run_lock(self.root):
            with self.assertRaises(LifecycleError):
                with run_lock(self.root):
                    pass

    def test_final_success_requires_all_archive_hashes(self):
        engine = self.running()
        result={"job_id":self.identity["job_id"],"terminal_state":"success",
                "outputs":[{"path":"outputs/test","size":2,"sha256":digest(b"ok")}]}
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,"w") as z:
            z.writestr("result.json",encoded(result))
            z.writestr("outputs/test",b"ok")
        publish(self.store,self.identity,"final",{"result.json":encoded(result),"artifacts.zip":buffer.getvalue()},
                step=6,attempt=1)
        engine.advance()
        self.assertEqual(engine.state["phase"],"completed")
        self.assertEqual(self.provider.active,set())
        self.assertTrue(Path(engine.state["final"]["local_directory"]).is_dir())

    def test_live_logs_only_use_the_current_attempt(self):
        engine=self.running()
        publish(self.store,self.identity,'logs',{'stdout.log':b'current'},attempt=1,step=0)
        publish(self.store,self.identity,'logs',{'stdout.log':b'other attempt'},attempt=2,step=999)
        engine.advance()
        self.assertEqual(engine.state['logs']['commit']['attempt'],1)
        root=Path(engine.state['logs']['local_directory'])
        self.assertEqual((root/'stdout.log').read_bytes(),b'current')

    def test_latest_log_uses_creation_time_not_random_drive_id(self):
        refs = [publish(self.store, self.identity, "logs", {"stdout.log": text}) for text in (b"first", b"second")]
        older, newer = sorted(refs, key=lambda ref: ref["id"], reverse=True)
        entries = [{"id": ref["id"], "size": str(ref["bytes"]), "appProperties": {"cc_sha": ref["sha256"]}, "createdTime": date}
                   for ref, date in [(newer, "2026-10-15T10:00:02.000Z"), (older, "2026-10-15T10:00:01.000Z")]]
        with mock.patch.object(self.store, "list", return_value=entries):
            self.assertEqual(snapshots(self.store, self.identity, "logs")[-1]["id"], newer["id"])

    def test_worker_final_is_recovered_when_drive_listing_lags(self):
        engine = self.running()
        final = self.final_snapshot(1, 0, terminal="job_failed")
        ref = {k: final[k] for k in ("id", "bytes", "sha256")}
        with mock.patch.object(self.store, "list", return_value=[]), mock.patch.object(self.provider, "observe", return_value={"state": "failed", "attempt": 1, "final_commit": ref}):
            engine.advance()
            self.assertEqual(engine.state["phase"], "monitoring")
            engine.advance()
        self.assertEqual(engine.state["phase"], "failed")
        self.assertEqual(engine.state["reason"], "job_failed")
        self.assertTrue(Path(engine.state["final"]["local_directory"]).is_dir())
        self.assertEqual(len(self.provider.allocations), 1)
        self.assertFalse(self.provider.active)

    def test_terminal_reopen_recovers_without_provider_or_new_attempt(self):
        engine = self.running()
        final = self.final_snapshot(1, 0, terminal="job_failed")
        engine.state["worker"] = {"attempt": 1, "state": "failed", "final_commit": {k: final[k] for k in ("id", "bytes", "sha256")}}
        engine.plan.update(parent="fixture", token_command=["fixture"])
        atomic(self.root/"plan.json", engine.plan)
        engine.finish("failed", "runtime_failed")
        before = list(engine.state["attempts"])
        with mock.patch("colabcola.lifecycle.DriveStore", return_value=self.store) as store, mock.patch("colabcola.lifecycle_provider.GoogleProvider", side_effect=AssertionError("must not allocate")):
            recovered = recover_terminal_final(self.root)
            again = recover_terminal_final(self.root)
            self.assertEqual(store.call_count, 1)
        self.assertEqual(recovered, again)
        self.assertEqual(recovered["attempts"], before)
        self.assertEqual(recovered["phase"], "failed")
        self.assertTrue(Path(recovered["final"]["local_directory"]).is_dir())

    def test_reported_final_rejects_another_attempt(self):
        engine = self.running()
        final = self.final_snapshot(2, 0)
        with self.assertRaises(LifecycleError):
            reported_final(self.store, engine.plan, engine.state["attempts"][0], final)

    def test_final_corrupt_archive_cannot_complete(self):
        engine=self.running()
        result={"job_id":self.identity["job_id"],"terminal_state":"success",
                "outputs":[{"path":"outputs/test","size":2,"sha256":digest(b"ok")}]}
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,"w") as z:
            z.writestr("result.json",encoded(result))
            z.writestr("outputs/test",b"no")
        publish(self.store,self.identity,"final",{"result.json":encoded(result),"artifacts.zip":buffer.getvalue()}, attempt=1)
        engine.advance()
        self.assertNotEqual(engine.state["phase"],"completed")
        self.assertEqual(self.provider.active,{"vm-1"})

    def test_budget_reconciles_allocation_whose_response_was_lost(self):
        engine = self.engine()
        engine.advance()
        self.provider.timeout_after_allocation = True
        engine.advance()
        self.now = 1101
        engine = self.reload()
        engine.advance()
        self.assertEqual(engine.state["phase"], "budget_exhausted")
        self.assertEqual(self.provider.releases, ["vm-1"])
        self.assertEqual(self.provider.active, set())

    def test_loss_during_boot_does_not_hang_or_duplicate(self):
        engine = self.engine()
        engine.advance(); engine.advance()
        self.provider.active.clear()
        engine.advance(); engine.advance()
        self.assertEqual(engine.state["phase"], "recovering")
        engine.advance(); engine.advance()
        self.assertEqual(len(self.provider.allocations), 2)

    def test_attempt_budget_stops_after_second_runtime_loss(self):
        engine = self.running()
        for _ in range(2):
            self.provider.active.clear()
            engine.advance(); engine.advance(); engine.advance()
            if engine.state["phase"] == "allocating":
                engine.advance(); engine.advance()
        self.assertEqual(engine.state["phase"], "failed")
        self.assertEqual(len(self.provider.allocations), 2)
        self.assertEqual(self.provider.active, set())

    def final_snapshot(self, attempt, step, evidence=None, terminal="success"):
        result = {"job_id": self.identity["job_id"], "terminal_state": terminal, "outputs": []}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z:
            z.writestr("result.json", encoded(result))
        members = {"result.json": encoded(result), "artifacts.zip": buffer.getvalue()}
        if evidence is not None:
            members["attempt.json"] = encoded(evidence)
        return publish(self.store, self.identity, "final", members, step=step, attempt=attempt)

    def test_finishing_before_fault_is_not_a_recovery_pass(self):
        engine = self.running(True)
        self.final_snapshot(1, 6)
        engine.advance()
        self.assertEqual(engine.state["phase"], "failed")
        self.assertEqual(engine.state["reason"], "recovery_proof_not_demonstrated")

    def test_recovery_pass_binds_replacement_to_verified_resume(self):
        engine = self.running(True)
        snap = publish(self.store, self.identity, "checkpoint", {"s": b"saved"}, step=3, attempt=1)
        engine.advance()
        for _ in range(5):
            engine.advance()
        self.assertEqual(engine.state["phase"], "monitoring")
        self.final_snapshot(2, 6, {"attempt": 2, "restored_step": 3, "resume_sha256": snap["sha256"]})
        engine.advance()
        self.assertEqual(engine.state["phase"], "completed")
        self.assertEqual(engine.state["restored_step"], 3)
        self.assertEqual(self.provider.active, set())

    def test_late_publications_cannot_replace_the_proof_checkpoint(self):
        engine = self.running(True)
        snap = publish(self.store, self.identity, "checkpoint", {"s": b"saved"}, step=3, attempt=1)
        engine.advance()
        self.assertEqual(engine.state["proof_checkpoint"], snap)
        publish(self.store, self.identity, "checkpoint", {"s": b"finished"}, step=6, attempt=1)
        self.final_snapshot(1, 6)
        engine = self.reload()
        for _ in range(5):
            engine.advance()
        self.assertEqual(engine.state["phase"], "monitoring")
        self.assertEqual(engine.state["attempts"][1]["resume"], snap)
        engine.advance()
        self.assertEqual(engine.state["phase"], "monitoring")
        self.final_snapshot(2, 6, {"attempt": 2, "restored_step": 3, "resume_sha256": snap["sha256"]})
        engine.advance()
        self.assertEqual(engine.state["phase"], "completed")
        self.assertEqual(len(self.provider.allocations), 2)

    def test_lost_fault_release_response_is_reconciled(self):
        engine = self.running(True)
        snap = publish(self.store, self.identity, "checkpoint", {"s": b"saved"}, step=3)
        real_release = self.provider.release
        def uncertain(endpoint):
            real_release(endpoint)
            raise TimeoutError()
        self.provider.release = uncertain
        engine.advance()
        self.provider.release = real_release
        engine.advance(); engine.advance()
        self.assertTrue(engine.state["fault_injected"])
        self.assertFalse(engine.state["fault_pending"])
        self.assertEqual(engine.state["proof_checkpoint"], snap)

    def test_workload_checkpoint_members_verify_bytes(self):
        outputs=self.root/"outputs"
        outputs.mkdir()
        (outputs/"state.pt").write_bytes(b"state")
        atomic(outputs/"state.pt.json",{"schema":1,"step":2,"file":"state.pt",
                                      "bytes":5,"sha256":digest(b"state")})
        self.assertEqual(checkpoint_members(self.root,"outputs/*.json")[0][0],2)
        (outputs/"state.pt").write_bytes(b"bad")
        with self.assertRaises(LifecycleError):
            checkpoint_members(self.root,"outputs/*.json")

    def test_recovery_policy_changes_job_identity_only_when_declared(self):
        source=self.root/"train.py"
        source.write_text("print('ok')")
        mapping=notebook_job(source)
        old=build_bundle(bind_job_arguments(JobSpec.from_mapping(mapping)))
        mapping["managed_recovery"]={"checkpoint_pattern":"outputs/state-*.json",
                                    "log_patterns":["outputs/log.ndjson"]}
        new=build_bundle(bind_job_arguments(JobSpec.from_mapping(mapping)))
        self.assertNotEqual(old.job_id,new.job_id)
        self.assertNotIn("managed_recovery",json.loads(old.canonical_manifest_bytes))
        for bad in ("../state.json","outputs/../state.json","secret.json"):
            mapping["managed_recovery"]["checkpoint_pattern"]=bad
            with self.assertRaises(Exception):
                JobSpec.from_mapping(mapping)


class DriveTransportTests(unittest.TestCase):
    def test_multipart_scopes_parent_run_and_verifies_uploaded_bytes(self):
        from colabcola.lifecycle_io import DriveStore
        from urllib.parse import urlparse, parse_qs
        calls = []
        def request(req, timeout):
            calls.append(req)
            self.assertEqual(timeout, 30)
            self.assertEqual(req.get_header("Authorization"), "Bearer fixture")
            if req.method == "POST":
                self.assertIn(b'"parents":["parent"]', req.data)
                self.assertIn(b'"cc_run":"' + b"a"*32 + b'"', req.data)
                self.assertIn(b"payload", req.data)
                return io.BytesIO(b'{"id":"uploaded"}')
            if parse_qs(urlparse(req.full_url).query).get("alt") == ["media"]:
                return io.BytesIO(b"payload")
            return io.BytesIO(b'{"files":[]}')
        store = DriveStore("parent", "a"*32, lambda: "fixture", request=request)
        ref = store.put("member", "state.pt", b"payload")
        self.assertEqual(ref["sha256"], digest(b"payload"))
        self.assertEqual([r.method for r in calls], ["GET", "POST", "GET"])

    def test_incomplete_listing_fails_closed(self):
        from colabcola.lifecycle_io import DriveStore
        store = DriveStore("parent", "a"*32, lambda: "fixture",
                           request=lambda *a, **kw: io.BytesIO(b'{"incompleteSearch":true}'))
        with self.assertRaises(LifecycleError):
            store.list("checkpoint")

    def test_retryable_http_failure_is_bounded_and_redacted(self):
        from colabcola.lifecycle_io import DriveStore
        import urllib.error
        request = mock.Mock(side_effect=urllib.error.HTTPError("secret-url", 503, "secret", {}, None))
        store = DriveStore("parent", "a"*32, lambda: "fixture", request=request)
        with mock.patch("colabcola.lifecycle_io.time.sleep"):
            with self.assertRaisesRegex(LifecycleError, "^Drive HTTP 503$"):
                store.list("checkpoint")
        self.assertEqual(request.call_count, 3)


if __name__ == "__main__":
    unittest.main()
