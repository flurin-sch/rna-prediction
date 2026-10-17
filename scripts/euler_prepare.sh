#!/bin/bash
#SBATCH --job-name=rna-pred-full-prepare
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8g
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
exec "$PWD/.conda/bin/rna-pred" --config "${1:-code/rna_pred/baseline/configs/default.yaml}" prepare

