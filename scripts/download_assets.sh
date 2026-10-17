#!/bin/bash
# Download native hg38 and the pinned converted AlphaGenome weights on a login node.
set -euo pipefail
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"
mkdir -p data/reference data/checkpoints
STAGING_DIR=${RNA_PRED_DOWNLOAD_STAGE:-${TMPDIR:-/tmp}/rna-pred-downloads}
mkdir -p "$STAGING_DIR"
if [[ ! -f data/reference/hg38.fa ]]; then
    curl --fail --location --retry 3 --continue-at - \
        https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz \
        --output "$STAGING_DIR/hg38.fa.gz"
    gzip -dc "$STAGING_DIR/hg38.fa.gz" > "$STAGING_DIR/hg38.fa"
    cp "$STAGING_DIR/hg38.fa" data/reference/hg38.fa.tmp
    mv data/reference/hg38.fa.tmp data/reference/hg38.fa
fi
if [[ ! -f data/checkpoints/model_all_folds.safetensors ]]; then
    curl --fail --location --retry 3 --continue-at - \
        https://huggingface.co/gtca/alphagenome_pytorch/resolve/b01c0ffa73e07c053491f3b5ea8bcf67d93b9920/model_all_folds.safetensors \
        --output "$STAGING_DIR/model_all_folds.safetensors.part"
    cp "$STAGING_DIR/model_all_folds.safetensors.part" data/checkpoints/model_all_folds.safetensors.part
    mv data/checkpoints/model_all_folds.safetensors.part data/checkpoints/model_all_folds.safetensors
fi

