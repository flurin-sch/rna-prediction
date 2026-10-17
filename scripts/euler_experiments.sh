#!/bin/bash
#SBATCH --job-name=rna-pred-experiments
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --gres=gpumem:24G
#SBATCH --tmp=10G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${TMPDIR:?Slurm must provide node-local storage; request --tmp}"
CONFIG_PATH=${1:-code/rna_pred/baseline/configs/default.yaml}
if (( $# > 0 )); then shift; fi
"$PWD/.conda/bin/python" -m rna_pred.common.job --config "$CONFIG_PATH" "$@"
