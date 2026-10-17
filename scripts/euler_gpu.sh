#!/bin/bash
# Submit from the repository: mkdir -p logs; sbatch scripts/euler_gpu.sh --config code/rna_pred/baseline/configs/smoke.yaml cache
# Override GPU model/resources with sbatch flags if required.
#SBATCH --job-name=rna-pred-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8g
#SBATCH --gpus=rtx_4090:1
#SBATCH --gres=gpumem:24g
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PROJECT_DIR=${SLURM_SUBMIT_DIR:?Submit this script from the repository root}
cd "$PROJECT_DIR"
# Assets should be downloaded in advance; the job needs no network.
exec "$PROJECT_DIR/.conda/bin/rna-pred" "$@"

