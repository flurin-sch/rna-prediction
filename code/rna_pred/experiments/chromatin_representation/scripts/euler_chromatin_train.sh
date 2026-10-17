#!/bin/bash
#SBATCH --job-name=rna-chromatin-train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --gres=gpumem:24G
#SBATCH --tmp=12G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
SNAPSHOT=${1:?Pass the source snapshot directory}
export PYTHONPATH="$SNAPSHOT/code${PYTHONPATH:+:$PYTHONPATH}"
SEEDS=(42 43 44)
"$PWD/.conda/bin/python" -u -m rna_pred.experiments.chromatin_representation.run train \
    --config "$SNAPSHOT/config.yaml" --seed "${SEEDS[$SLURM_ARRAY_TASK_ID]}" \
    --feature-dir "$PWD/outputs/experiments/chromatin_representation/features" \
    --output-dir "$PWD/outputs/experiments/chromatin_representation" --stage --resume
