#!/bin/bash
set -euo pipefail
mkdir -p outputs
printf 'prepare stdout\n'
printf 'prepare stderr\n' >&2
python - "$@" <<'PY'
import json, sys, os
from pathlib import Path
assert sys.version_info[:2] == (3, 10), sys.version
assert Path(sys.prefix) == Path(os.environ['CONDA_PREFIX'])
assert sys.argv[1:] == ['two words', '$HOME', '$(touch unexpected)', '; literal'], sys.argv
Path('outputs/prepared.json').write_text(json.dumps({'python': sys.version.split()[0], 'arguments': sys.argv[1:]}) + '\n')
PY
