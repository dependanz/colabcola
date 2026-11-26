"""Stream and hash the restored private input without retaining its contents."""
import hashlib
import json
from pathlib import Path

path = Path("models/weights.bin")
with path.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
report = {"bytes": path.stat().st_size, "sha256": digest}
Path("outputs").mkdir(exist_ok=True)
Path("outputs/check.json").write_text(json.dumps(report) + "\n", encoding="utf-8")
Path("outputs/complete.json").write_text('{"state":"complete"}\n', encoding="utf-8")
print(json.dumps(report), flush=True)
