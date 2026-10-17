#!/bin/bash
#SBATCH --job-name=rna-low-train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --gres=gpumem:24G
#SBATCH --tmp=12G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${SLURM_SUBMIT_DIR:?}"
SNAPSHOT=${1:?source snapshot}
export PYTHONPATH="$SNAPSHOT/code"
SEEDS=(42 43)
"$PWD/.conda/bin/python" -u -m rna_pred.experiments.low_expression_features.run train \
    --config "$SNAPSHOT/config.yaml" --seed "${SEEDS[$SLURM_ARRAY_TASK_ID]}" --stage
