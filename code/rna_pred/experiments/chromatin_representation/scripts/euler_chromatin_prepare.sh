#!/bin/bash
#SBATCH --job-name=rna-chromatin-prepare
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
SNAPSHOT=${1:?Pass the source snapshot directory}
export PYTHONPATH="$SNAPSHOT/code${PYTHONPATH:+:$PYTHONPATH}"
"$PWD/.conda/bin/python" -u -m rna_pred.experiments.chromatin_representation.run prepare \
    --config "$SNAPSHOT/config.yaml" --feature-dir "$PWD/outputs/experiments/chromatin_representation/features"
