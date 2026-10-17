#!/bin/bash
# Gate for prerequisite jobs that have aged out of the Slurm controller.
#SBATCH --job-name=rna-pred-experiment-prerequisites
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8g
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
CONFIG_PATH=${1:?Configuration required}
shift
if (( $# == 0 )); then
    echo "At least one prerequisite job ID is required" >&2
    exit 1
fi
for prerequisite_job in "$@"; do
    if [[ ! "$prerequisite_job" =~ ^[0-9]+$ ]]; then
        echo "Invalid prerequisite job ID: $prerequisite_job" >&2
        exit 1
    fi
    record=$(sacct --noheader --allocations --jobs="$prerequisite_job" --format=JobIDRaw,State,ExitCode --parsable2)
    if [[ "$record" != "$prerequisite_job|COMPLETED|0:0" ]]; then
        echo "Prerequisite $prerequisite_job has not completed successfully: $record" >&2
        exit 1
    fi
    echo "Confirmed prerequisite $record"
done
"$PWD/.conda/bin/rna-pred" --config "$CONFIG_PATH" validate-cache
