#!/bin/bash
# Submit with an afterok dependency on the full preprocessing job.
#SBATCH --job-name=rna-pred-full-cache
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8g
#SBATCH --gpus=rtx_4090:1
#SBATCH --gres=gpumem:24g
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
CONFIG_PATH=${1:-code/rna_pred/baseline/configs/default.yaml}
"$PWD/.conda/bin/rna-pred" --config "$CONFIG_PATH" cache --device cuda
"$PWD/.conda/bin/rna-pred" --config "$CONFIG_PATH" validate-cache
