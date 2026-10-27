from __future__ import annotations

import datetime
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

from colabcola.checkpoint_runtime import DURABLE_CHECKPOINT_SOURCE
from colabcola.runtime import drive_runtime_cell_sources


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError("duplicate JSON key")
        result[key] = value
    return result


def _helpers() -> dict[str, object]:
    namespace: dict[str, object] = {
        "hashlib": hashlib,
        "datetime": datetime,
        "json": json,
        "re": re,
        "COLABCOLA_JOB_ID": "a" * 64,
        "COLABCOLA_MANIFEST": {
            "durable_checkpoints": {
                "filename_digits": 8,
                "generator_prefix": "g",
                "state_prefix": "do",
            }
        },
        "_colabcola_unique_object": _unique_object,
    }
    exec(DURABLE_CHECKPOINT_SOURCE, namespace)
    return namespace


class DurableCheckpointRuntimeTests(unittest.TestCase):
    def test_runtime_cell_embeds_checkpoint_protocol(self) -> None:
        execution = dict(drive_runtime_cell_sources())["execution"]
        self.assertIn("def _colabcola_checkpoint_worker", execution)
        self.assertIn("manifest uploaded last is the commit marker", execution)

    def test_local_discovery_uses_only_complete_exact_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "g_00000010").write_bytes(b"generator")
            (root / "do_00000010").write_bytes(b"state")
            (root / "g_00000020").write_bytes(b"orphan")
            (root / ".g_00000030.tmp").write_bytes(b"temporary")
            config = {
                "filename_digits": 8,
                "generator_prefix": "g",
                "state_prefix": "do",
            }

            pairs = _helpers()["_colabcola_local_checkpoint_pairs"](root, config)  # type: ignore[operator]

            self.assertEqual([(step, g.name, s.name) for step, g, s in pairs], [(10, "g_00000010", "do_00000010")])

    def test_retention_removes_commit_marker_first_and_keeps_two_pairs(self) -> None:
        namespace = _helpers()
        trashed = []
        namespace["_colabcola_checkpoint_trash"] = trashed.append
        records = [
            {
                "metadata": {"id": f"manifest-{step}"},
                "manifest": {
                    "step": step,
                    "generator": {"id": f"generator-{step}"},
                    "state": {"id": f"state-{step}"},
                },
            }
            for step in (10, 20, 30)
        ]

        namespace["_colabcola_checkpoint_retain"](records, 2)  # type: ignore[operator]

        self.assertEqual(trashed, ["manifest-10", "generator-10", "state-10"])
        self.assertEqual([record["manifest"]["step"] for record in records], [20, 30])

    def test_pair_members_are_uploaded_before_commit_manifest(self) -> None:
        namespace = _helpers()
        events = []
        evidence = {
            "generator": {
                "name": "g_00000010",
                "size": 1,
                "sha256": "1" * 64,
                "md5": "a" * 32,
            },
            "state": {
                "name": "do_00000010",
                "size": 1,
                "sha256": "2" * 64,
                "md5": "b" * 32,
            },
        }
        namespace["_colabcola_checkpoint_file_evidence"] = (
            lambda path: evidence["generator" if path.name.startswith("g_") else "state"]
        )

        def upload(path, role, step, folder_id, member_evidence):
            events.append(role)
            return {"id": role + "-id"}

        def commit(manifest, folder_id):
            events.append("manifest")
            return {"id": "manifest-id"}

        namespace["_colabcola_checkpoint_upload"] = upload
        namespace["_colabcola_checkpoint_upload_manifest"] = commit
        records = []

        manifest = namespace["_colabcola_publish_checkpoint"](  # type: ignore[operator]
            10, Path("g_00000010"), Path("do_00000010"), "folder-id", records
        )

        self.assertEqual(events, ["generator", "state", "manifest"])
        self.assertEqual(manifest["step"], 10)
        self.assertEqual(records[0]["metadata"]["id"], "manifest-id")

    def test_duplicate_committed_step_is_rejected(self) -> None:
        namespace = _helpers()
        manifest = {
            "schema_version": 1,
            "job_id": "a" * 64,
            "step": 10,
            "created_at_utc": "2026-08-30T00:00:00+00:00",
            "generator": {
                "id": "g-id",
                "name": "g_00000010",
                "size": 1,
                "sha256": "1" * 64,
            },
            "state": {
                "id": "do-id",
                "name": "do_00000010",
                "size": 1,
                "sha256": "2" * 64,
            },
        }
        content = json.dumps(
            manifest, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        metadata = {
            "id": "manifest-id",
            "parents": ["folder-id"],
            "appProperties": {
                "colabcola_checkpoint_step": "10",
                "colabcola_checkpoint_sha256": digest,
            },
        }
        namespace["_colabcola_checkpoint_list"] = lambda role: [metadata, dict(metadata, id="duplicate-id")]
        namespace["_colabcola_checkpoint_download"] = lambda file_id: content

        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            namespace["_colabcola_committed_checkpoints"]("folder-id")  # type: ignore[operator]


if __name__ == "__main__":
    unittest.main()
