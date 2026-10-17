#!/bin/bash
#SBATCH --job-name=rna-chromatin-summary
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
SNAPSHOT=${1:?Pass the source snapshot directory}
export PYTHONPATH="$SNAPSHOT/code${PYTHONPATH:+:$PYTHONPATH}"
"$PWD/.conda/bin/python" -u -m rna_pred.experiments.chromatin_representation.run summarize \
    --config "$SNAPSHOT/config.yaml" --output-dir "$PWD/outputs/experiments/chromatin_representation"
