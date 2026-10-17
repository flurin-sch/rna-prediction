#!/bin/bash
# Freeze the exact tested sources, then submit CPU preparation -> GPU array -> summary.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.."
mkdir -p logs
if [[ -n "$(squeue --me --name=rna-chromatin-prepare,rna-chromatin-train,rna-chromatin-summary -h -o '%i')" ]]; then
    echo "A chromatin comparison job is already pending/running; refusing duplicate submission." >&2
    exit 1
fi
SNAPSHOT=$("$PWD/.conda/bin/python" - <<'PY'
from datetime import datetime
from pathlib import Path
import hashlib
import json
import shutil
import yaml

root = Path.cwd()
out = root / 'outputs/experiments/chromatin_representation'
out.mkdir(parents=True, exist_ok=True)
snapshot = out / 'sources' / datetime.now().strftime('%Y%m%dT%H%M%S%f')
snapshot.mkdir(parents=True)
shutil.copytree(root / 'code/rna_pred', snapshot / 'code/rna_pred', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
shutil.copytree(root / 'scripts', snapshot / 'scripts')
shutil.copytree(root / 'docs', snapshot / 'docs')
shutil.copy2(root / 'README.md', snapshot / 'README.md')
config = yaml.safe_load((root / 'code/rna_pred/experiments/chromatin_representation/configs/default.yaml').read_text())
config['project_root'] = str(root)
(snapshot / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
for name in ['pyproject.toml', 'environment.yml']:
    shutil.copy2(root / name, snapshot / name)
files = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(snapshot.rglob('*')) if p.is_file()}
(snapshot / 'sha256.json').write_text(json.dumps(files, indent=2) + '\n')
print(snapshot)
PY
)
PREP_JOB=$(sbatch --parsable --output="$PWD/logs/%x-%j.out" code/rna_pred/experiments/chromatin_representation/scripts/euler_chromatin_prepare.sh "$SNAPSHOT")
TRAIN_JOB=$(sbatch --parsable --dependency="afterok:$PREP_JOB" --kill-on-invalid-dep=yes \
    --output="$PWD/logs/%x-%A_%a.out" code/rna_pred/experiments/chromatin_representation/scripts/euler_chromatin_train.sh "$SNAPSHOT")
SUMMARY_JOB=$(sbatch --parsable --dependency="afterok:$TRAIN_JOB" --kill-on-invalid-dep=yes \
    --output="$PWD/logs/%x-%j.out" code/rna_pred/experiments/chromatin_representation/scripts/euler_chromatin_summary.sh "$SNAPSHOT")
"$PWD/.conda/bin/python" - "$SNAPSHOT" "$PREP_JOB" "$TRAIN_JOB" "$SUMMARY_JOB" <<'PY'
from pathlib import Path
import json
import sys
snapshot, prep, train, summary = sys.argv[1:]
data = dict(source_snapshot=snapshot, preparation_job=prep, training_array=train, summary_job=summary)
out = Path('outputs/experiments/chromatin_representation')
(out / 'submission.json').write_text(json.dumps(data, indent=2) + '\n')
print(json.dumps(data, indent=2))
PY
