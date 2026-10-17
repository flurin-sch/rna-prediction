#!/bin/bash
#SBATCH --job-name=rna-low-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
cd "${SLURM_SUBMIT_DIR:?}"
SNAPSHOT=${1:?source snapshot}
COMMAND=${2:?explore, screen or summarize}
export PYTHONPATH="$SNAPSHOT/code"
"$PWD/.conda/bin/python" -u -m rna_pred.experiments.low_expression_features.run "$COMMAND" --config "$SNAPSHOT/config.yaml"
