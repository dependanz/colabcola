#!/bin/bash
set -euo pipefail
printf 'evaluate stdout\n'
printf 'evaluate stderr\n' >&2
python - <<'PY'
import json, sys, os
from pathlib import Path
assert sys.version_info[:2] == (3, 12), sys.version
assert Path(sys.prefix) == Path(os.environ['CONDA_PREFIX'])
assert Path.cwd().name == 'scripts'
assert not Path('../unexpected').exists()
root = Path('../outputs')
prepared = json.loads((root / 'prepared.json').read_text())
assert prepared['python'].startswith('3.10.')
assert prepared['arguments'] == ['two words', '$HOME', '$(touch unexpected)', '; literal']
(root / 'acceptance.json').write_text(json.dumps({'prepare_python': prepared['python'], 'evaluate_python': sys.version.split()[0], 'literal_arguments': prepared['arguments'], 'cwd': Path.cwd().name, 'state': 'passed'}) + '\n')
(root / 'complete.json').write_text('{"state":"complete"}\n')
PY
