"""Private asset protocol tests: real DriveStore requests, local fake Drive.

No credentials, cloud calls or GPU are required. The optional 1 GiB proof uses
unique 8 MiB chunks and a disk-backed fake endpoint, not a whole-file bytes mock.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import unittest
import urllib.error
import urllib.parse
from unittest import mock

from colabcola import lifecycle_io as lio


class AssetDriveHTTP:
    def __init__(self, directory):
        self.directory = directory
        self.objects = {}
        self.events = []
        self.shared_parent = False
        self.fail_kind = None
        self.corrupt = set()
        self.outside = False
        self.hide_catalog = False
        self.max_payload = 0

    def request(self, req, timeout):
        assert 0 < timeout <= 30
        assert req.get_header("Authorization") == "Bearer test-lease"
        url = urllib.parse.urlparse(req.full_url)
        assert url.hostname == "www.googleapis.com"
        params = urllib.parse.parse_qs(url.query)
        if req.method == "POST":
            assert params["uploadType"] == ["multipart"]
            self.max_payload = max(self.max_payload, len(req.data))
            boundary = req.get_header("Content-type").split("boundary=", 1)[1].encode()
            metadata, payload = req.data.split(b"\r\n--" + boundary + b"\r\n", 1)
            meta = json.loads(metadata.split(b"\r\n\r\n", 1)[1])
            body = payload.split(b"\r\n\r\n", 1)[1].removesuffix(b"\r\n--" + boundary + b"--\r\n")
            kind = meta["appProperties"]["cc_kind"]
            if self.fail_kind == kind:
                raise urllib.error.HTTPError(req.full_url, 403, "fixture failure", {}, None)
            assert set(meta) == {"name", "parents", "appProperties"}
            assert meta["appProperties"]["cc_sha"] == lio.digest(body)
            file_id = f"object_{len(self.objects):05d}"
            path = self.directory / file_id
            path.write_bytes(body)
            self.objects[file_id] = {**meta, "id": file_id, "size": str(len(body)),
                                     "ownedByMe": True, "shared": False, "trashed": False,
                                     "createdTime": file_id}
            self.events.append(("put", kind, file_id))
            return io.BytesIO(lio.encoded({"id": file_id}))
        assert req.method == "GET"
        if params.get("alt") == ["media"]:
            file_id = url.path.rsplit("/", 1)[1]
            self.events.append(("get", self.objects[file_id]["appProperties"]["cc_kind"], file_id))
            if file_id in self.corrupt:
                return io.BytesIO(b"corrupted")
            return (self.directory / file_id).open("rb")
        if "q" not in params:
            parent = url.path.rsplit("/", 1)[1]
            assert "shared" in params["fields"][0]
            if parent in self.objects:
                self.events.append(("metadata", parent, None))
                return io.BytesIO(lio.encoded(self.objects[parent]))
            if parent not in ("private-parent", "other-parent"):
                raise urllib.error.HTTPError(req.full_url, 404, "not found", {}, None)
            self.events.append(("parent", parent, None))
            return io.BytesIO(lio.encoded({"id": parent, "mimeType": "application/vnd.google-apps.folder",
                                         "ownedByMe": True, "shared": self.shared_parent, "trashed": False,
                                         "capabilities": {"canAddChildren": True}}))
        query = params["q"][0]
        parent = re.search(r"'([^']+)' in parents", query).group(1)
        run = re.search(r"key='cc_run' and value='([^']+)'", query).group(1)
        kind = re.search(r"key='cc_kind' and value='([^']+)'", query).group(1)
        self.events.append(("list", kind, run))
        fields = params["fields"][0]
        assert all(field in fields for field in ("parents", "ownedByMe", "shared", "trashed"))
        items = [dict(obj) for obj in self.objects.values()
                 if obj["parents"] == [parent] and obj["appProperties"]["cc_run"] == run
                 and obj["appProperties"]["cc_kind"] == kind]
        if self.hide_catalog:
            items = []
        if self.outside and items:
            items[0]["parents"] = ["foreign-parent"]
        return io.BytesIO(lio.encoded({"files": items}))

    def storage(self, parent="private-parent"):
        return lio.PrivateAssetStorage(parent, lambda: "test-lease", store_factory=self.factory)

    def factory(self, *args, **kwargs):
        return lio.DriveStore(*args, request=self.request, **kwargs)

    def rewrite(self, file_id, value):
        """Simulate maliciously edited catalog + content; caller must still bind identity."""
        body = lio.encoded(value)
        (self.directory / file_id).write_bytes(body)
        self.objects[file_id]["size"] = str(len(body))
        self.objects[file_id]["appProperties"]["cc_sha"] = lio.digest(body)


class PrivateAssetTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        remote = self.root / "remote"
        remote.mkdir()
        self.drive = AssetDriveHTTP(remote)
        self.path = self.root / "asset.bin"
        self.body = bytes(range(195))
        self.path.write_bytes(self.body)
        self.identity = lio.private_asset_identity(len(self.body), lio.digest(self.body))
        self.deadline = time.time() + 60
        patch = mock.patch.object(lio, "CHUNK_BYTES", 64)
        patch.start()
        self.addCleanup(patch.stop)

    def publish(self):
        return self.drive.storage().ensure(self.path, self.identity, self.deadline)

    def restore(self, name="restored.bin", **kw):
        return self.drive.storage().restore(self.identity, self.root / name, self.deadline, **kw)

    def test_commit_last_reuse_across_independent_clients_and_restore(self):
        self.assertIsNone(self.drive.storage().inspect(self.identity, self.deadline))
        published = self.publish()
        self.assertEqual(published["state"], "published")
        self.assertEqual([r["bytes"] for r in published["commit"]["chunks"]], [64, 64, 64, 3])
        writes = [e for e in self.drive.events if e[0] == "put"]
        self.assertEqual([e[1] for e in writes], [lio.PRIVATE_ASSET_KIND_PREFIX + "member"] * 4 + [lio.PRIVATE_ASSET_KIND_PREFIX + "commit"])
        for event in writes[:-1]:
            self.assertLess(self.drive.events.index(event), self.drive.events.index(("get", event[1], event[2])))
            self.assertLess(self.drive.events.index(("get", event[1], event[2])), self.drive.events.index(writes[-1]))
        self.path.unlink()  # the frozen identity, not the local source, addresses a hit
        start = len(self.drive.events)
        reused = self.drive.storage().ensure(None, self.identity, self.deadline)
        self.assertEqual(reused["state"], "reused")
        self.assertEqual(reused["reference"], published["reference"])
        self.assertFalse(any(e[0] == "put" or e[:2] == ("get", lio.PRIVATE_ASSET_KIND_PREFIX + "member") for e in self.drive.events[start:]))
        self.assertEqual(self.restore(reference=reused["reference"]), {"state": "restored", **self.identity})
        self.assertEqual((self.root / "restored.bin").read_bytes(), self.body)

    def test_exact_receipt_survives_catalogue_lag(self):
        receipt = self.publish()
        self.drive.hide_catalog = True
        self.assertIsNone(self.drive.storage().inspect(self.identity, self.deadline))
        self.assertEqual(self.restore(reference=receipt["reference"]), {"state": "restored", **self.identity})
        self.assertEqual((self.root / "restored.bin").read_bytes(), self.body)
        self.drive.objects[receipt["commit"]["chunks"][0]["id"]]["shared"] = True
        with self.assertRaisesRegex(lio.LifecycleError, "reference is outside its scope"):
            self.restore("second.bin", reference=receipt["reference"])
        self.assertFalse((self.root / "second.bin").exists())

    def test_missing_and_changed_source_do_not_commit(self):
        with self.assertRaisesRegex(lio.LifecycleError, "no local source"):
            self.drive.storage().ensure(None, self.identity, self.deadline)
        self.path.write_bytes(b"z" * len(self.body))
        with self.assertRaisesRegex(lio.LifecycleError, "differs from frozen identity"):
            self.publish()
        self.assertIsNone(self.drive.storage().inspect(self.identity, self.deadline))
        self.assertFalse(any(e[0] == "put" for e in self.drive.events))

    def test_failed_commit_can_resume_without_reuploading_chunks(self):
        self.drive.fail_kind = lio.PRIVATE_ASSET_KIND_PREFIX + "commit"
        with self.assertRaisesRegex(lio.LifecycleError, "Drive HTTP 403"):
            self.publish()
        self.assertIsNone(self.drive.storage().inspect(self.identity, self.deadline))
        self.drive.fail_kind = None
        start = len(self.drive.events)
        self.assertEqual(self.publish()["state"], "published")
        self.assertEqual([e[1] for e in self.drive.events[start:] if e[0] == "put"], [lio.PRIVATE_ASSET_KIND_PREFIX + "commit"])

    def test_privacy_and_parent_scope(self):
        self.drive.shared_parent = True
        with self.assertRaisesRegex(lio.LifecycleError, "owned, unshared"):
            self.publish()
        self.assertFalse(self.drive.objects)
        self.drive.shared_parent = False
        receipt = self.publish()
        self.assertIsNone(self.drive.storage("other-parent").inspect(self.identity, self.deadline))
        self.drive.outside = True
        with self.assertRaisesRegex(lio.LifecycleError, "not owned, unshared and in scope"):
            self.restore()
        self.drive.outside = False
        self.drive.objects[receipt["commit"]["chunks"][0]["id"]]["shared"] = True
        with self.assertRaisesRegex(lio.LifecycleError, "not owned, unshared and in scope"):
            self.restore()
        self.assertFalse((self.root / "restored.bin").exists())

    def test_corruption_never_exposes_target(self):
        receipt = self.publish()
        self.drive.corrupt.add(receipt["commit"]["chunks"][1]["id"])
        with self.assertRaisesRegex(lio.LifecycleError, "integrity mismatch"):
            self.restore()
        self.assertFalse((self.root / "restored.bin").exists())
        self.assertFalse(list(self.root.glob(".private-asset-*")))
        self.drive.corrupt.clear()
        wrong = dict(receipt["commit"])
        wrong["chunks"] = list(reversed(wrong["chunks"]))
        self.drive.rewrite(receipt["reference"]["id"], wrong)
        with self.assertRaisesRegex(lio.LifecycleError, "chunk is invalid"):
            self.restore()
        self.assertFalse((self.root / "restored.bin").exists())

    def test_wrong_order_of_equal_sized_chunks_rejected_by_full_digest(self):
        receipt = self.publish()
        wrong = dict(receipt["commit"])
        chunks = list(wrong["chunks"])
        chunks[0], chunks[1] = chunks[1], chunks[0]
        wrong["chunks"] = chunks
        self.drive.rewrite(receipt["reference"]["id"], wrong)
        with self.assertRaisesRegex(lio.LifecycleError, "SHA-256 or size mismatch"):
            self.restore()
        self.assertFalse((self.root / "restored.bin").exists())
        self.assertFalse(list(self.root.glob(".private-asset-*")))

    def test_forged_commit_and_external_reference_rejected(self):
        receipt = self.publish()
        bad_ref = dict(receipt["reference"], id="foreign-object")
        with self.assertRaisesRegex(lio.LifecycleError, "reference is outside"):
            self.restore(reference=bad_ref)
        wrong = dict(receipt["commit"], asset={"bytes": len(self.body), "sha256": "f" * 64})
        self.drive.rewrite(receipt["reference"]["id"], wrong)
        with self.assertRaisesRegex(lio.LifecycleError, "commit identity mismatch"):
            self.restore()

    def test_foreign_member_id_rejected(self):
        receipt = self.publish()
        wrong = dict(receipt["commit"])
        wrong["chunks"] = [dict(c) for c in wrong["chunks"]]
        wrong["chunks"][0]["id"] = "unrelated-object"
        self.drive.rewrite(receipt["reference"]["id"], wrong)
        with self.assertRaisesRegex(lio.LifecycleError, "outside its scope"):
            self.restore()
        self.assertFalse(any(e[-1] == "unrelated-object" for e in self.drive.events))

    def test_bounds_disk_deadline_and_no_overwrite(self):
        for size, sha in ((0, "a"*64), (True, "a"*64), (lio.PRIVATE_ASSET_BYTES+1, "a"*64), (1, "x")):
            with self.subTest(size=size, sha=sha), self.assertRaises(lio.LifecycleError):
                lio.private_asset_identity(size, sha)
        with self.assertRaises(TimeoutError):
            self.drive.storage().ensure(self.path, self.identity, time.time()-1)
        self.assertFalse(self.drive.events)
        self.publish()
        target = self.root / "existing.bin"
        target.write_bytes(b"existing")
        with self.assertRaisesRegex(lio.LifecycleError, "destination must be new"):
            self.restore("existing.bin")
        self.assertEqual(target.read_bytes(), b"existing")
        with mock.patch.object(shutil, "disk_usage", return_value=mock.Mock(free=0)):
            with self.assertRaisesRegex(lio.LifecycleError, "disk space"):
                self.restore()
        self.assertFalse((self.root / "restored.bin").exists())
        # Simulate an unrelated destination appearing just before publication.
        real_link = os.link
        def compete(source, target):
            Path(target).write_bytes(b"other writer")
            real_link(source, target)
        with mock.patch.object(os, "link", side_effect=compete), self.assertRaises(FileExistsError):
            self.restore()
        self.assertEqual((self.root / "restored.bin").read_bytes(), b"other writer")
        self.assertFalse(list(self.root.glob(".private-asset-*")))

    def test_distinct_identity_and_namespace(self):
        self.publish()
        other = lio.private_asset_identity(len(self.body), "a"*64)
        self.assertNotEqual(lio.private_asset_namespace(self.identity), lio.private_asset_namespace(other))
        self.assertNotEqual(lio.private_asset_namespace(self.identity), lio.environment_cache_namespace(self.identity["sha256"]))
        self.assertIsNone(self.drive.storage().inspect(other, self.deadline))

    @unittest.skipUnless(os.environ.get("COLABCOLA_PRIVATE_ASSET_LARGE_PROOF") == "1", "opt-in 1 GiB disk-backed protocol proof")
    def test_real_1gib_streaming_proof(self):
        if shutil.disk_usage(self.root).free < 4 * 1024**3:
            self.skipTest("requires 4 GiB local free disk")
        with mock.patch.object(lio, "CHUNK_BYTES", 8 * 1024**2):
            size = 1024**3
            sha = hashlib.sha256()
            with self.path.open("wb") as stream:
                for i in range(size // lio.CHUNK_BYTES):
                    body = i.to_bytes(4, "little") + b"a" * (lio.CHUNK_BYTES - 4)
                    sha.update(body)
                    stream.write(body)
            identity = lio.private_asset_identity(size, sha.hexdigest())
            deadline = time.time() + 180
            receipt = self.drive.storage().ensure(self.path, identity, deadline)
            self.assertEqual(receipt["state"], "published")
            self.assertEqual(len(receipt["commit"]["chunks"]), 128)
            self.assertEqual(len({c["id"] for c in receipt["commit"]["chunks"]}), 128)
            start = len(self.drive.events)
            self.assertEqual(self.drive.storage().ensure(None, identity, deadline)["state"], "reused")
            self.assertFalse(any(e[0] == "put" or e[:2] == ("get", lio.PRIVATE_ASSET_KIND_PREFIX + "member") for e in self.drive.events[start:]))
            dest = self.root / "large-restored.bin"
            self.drive.storage().restore(identity, dest, deadline, reference=receipt["reference"])
            with dest.open("rb") as stream:
                self.assertEqual(hashlib.file_digest(stream, "sha256").hexdigest(), identity["sha256"])
            self.assertEqual(dest.stat().st_size, size)
            self.assertLess(self.drive.max_payload, lio.CHUNK_BYTES + 4096)
            print(f"1 GiB proof: {size} bytes; 128 unique verified chunks; peak HTTP request {self.drive.max_payload} bytes; zero asset uploads on reuse")
