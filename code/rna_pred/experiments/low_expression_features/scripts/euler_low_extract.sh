#!/bin/bash
#SBATCH --job-name=rna-low-extract
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=6G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="${1:?source snapshot}/code"
"$PWD/.conda/bin/python" -u - "$1/config.yaml" "$PWD/outputs/experiments/low_expression_features" <<'PY'
import sys
from pathlib import Path
from rna_pred.common.config import load_config
from rna_pred.experiments.low_expression_features.features import prepare_extra
config = load_config(sys.argv[1])
prepare_extra(config, Path(sys.argv[2]), Path(config['project_root'])/'outputs/experiments/chromatin_representation/features')
PY
