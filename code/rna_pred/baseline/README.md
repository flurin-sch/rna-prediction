# Common baseline workflow

## Purpose

Provide a stable CAGE predictor and reproducible evaluation controls for new
experiments. The baseline uses frozen AlphaGenome contextual sequence embeddings
plus five measured tracks: DNase, H3K27ac, H3K4me3, H3K9me3, and H3K27me3.

## Model and data conventions

Each gene has a 2,048-bp promoter and up to 60 nearest 512-bp DNase candidate
regions within ±100 kb. Promoter-overlapping candidates are excluded. Candidates
are potential regulatory regions, not validated enhancer–gene assignments.
Coordinates are 0-based, half-open; the TSS is the midpoint of the supplied interval.

Each distinct 262,144-bp TSS-centered reference window is encoded once. Regions
receive overlap-weighted means of 128-bp contextual embeddings (3,072 dimensions).
Chromosome boundaries use N padding. Frozen extraction runs the transformer;
prediction heads and the 1-bp decoder are excluded. Reference orientation is kept.

Track means divide by full region width, with uncovered bases contributing zero.
Signed log1p and training-only z-scores normalize valid regional signals. Signed
strand-relative distance and the promoter indicator complete the token features.
The model projects sequence features to width 64, fuses measured inputs, applies
three four-head interaction blocks, and predicts from the promoter token. Distinct
candidate-to-candidate attention is masked. Padding cannot influence predictions.

Training uses log1p expression, Smooth L1, AdamW (learning rate 1e-4, weight decay
1e-6), batch 256, seed 42, and at most 50 epochs. The default checkpoint minimizes
validation loss. Early stopping requires both a training plateau and sustained
validation deterioration for eight epochs, using 0.1% relative thresholds.

## Run

```bash
rna-pred prepare
rna-pred cache --device cuda
rna-pred validate-cache
rna-pred train
rna-pred predict
rna-pred evaluate --output-dir outputs/baseline-evaluation
rna-pred report --output-dir outputs/baseline-evaluation
```

`configs/default.yaml` is the default; `configs/smoke.yaml` uses separate artifact
paths. A smoke verification starts with:

```bash
rna-pred --config code/rna_pred/baseline/configs/smoke.yaml prepare --limit 2
rna-pred --config code/rna_pred/baseline/configs/smoke.yaml cache --device cuda --verify-repeat
rna-pred --config code/rna_pred/baseline/configs/smoke.yaml train
rna-pred --config code/rna_pred/baseline/configs/smoke.yaml predict
```

A limited preparation produces an incomplete competition CSV. Resume training with
`train --resume`. Evaluation supports `--resume`, `--overwrite` (archives the prior
output directory), `--no-plots`, and `--allow-limited` for software verification.

## Evaluation and existing results

A is chr2–22 excluding chr14/chr19; B is chr14/chr19. X3 C contains unlabelled chr1.
The three definitions live in `common/designs.py`. Transfer fits use A+B in both
cells and therefore share loci. Only A→B predicts C.

The saved `outputs/baseline-evaluation/` run selected by validation loss obtained
combined validation Spearman 0.8031 (A→B), 0.7578 (X1→X2), and 0.7559 (X2→X1).
A compact record is in `results.csv`. Later studies use their own matched controls
and often different checkpoint selection. Their baseline rows must not be
interchanged with these original values.

## Shared implementation

`common/data/` handles preparation, regional normalization and frozen caches;
`common/training.py` handles optimization, deterministic RNG, predictions and
checkpoint IO. `workflow.py` retains the baseline checkpoint contract and selection
policy. `evaluation.py` coordinates the three designs. `plots.py` reads saved
metrics/predictions and writes one `summary.pdf` without loading a model.
