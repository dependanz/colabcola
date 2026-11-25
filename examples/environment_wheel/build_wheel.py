"""Create a deterministic, dependency-free wheel for the named-wheel example."""
import base64
import csv
import hashlib
import io
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parent
wheel = root / "wheels" / "cc_wheel_probe-1.0-py3-none-any.whl"
dist = "cc_wheel_probe-1.0.dist-info"
members = {
    "cc_wheel_probe.py": b"VALUE = 17\n",
    dist + "/METADATA": b"Metadata-Version: 2.1\nName: cc-wheel-probe\nVersion: 1.0\n\n",
    dist + "/WHEEL": b"Wheel-Version: 1.0\nGenerator: colabcola-example\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
}
record = io.StringIO(newline="")
writer = csv.writer(record, lineterminator="\n")
for name, body in sorted(members.items()):
    sha = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).decode().rstrip("=")
    writer.writerow([name, "sha256=" + sha, len(body)])
writer.writerow([dist + "/RECORD", "", ""])
members[dist + "/RECORD"] = record.getvalue().encode()
wheel.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(wheel, "w") as archive:
    for name, body in sorted(members.items()):
        archive.writestr(zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)), body)
print(wheel)
print("SHA-256:", hashlib.sha256(wheel.read_bytes()).hexdigest())
