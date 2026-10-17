# Chromatin representation comparison

## Problem

The original model averages five assays over selected regions and fuses them into
sequence tokens. Spatial chromatin patterns, additional assays, and gene-body
signals may be lost or difficult to use through that architecture. Low-expression
ranking is a particular weakness.

## Proposed solution and reasoning

Build a label-free sidecar with 911 measurements per gene: seven assays, 52
strand-oriented TSS bins across ±100 kb, nested promoter means, ten gene-body bins,
coverage fractions, and gene length. The two added assays are H3K4me1/H3K36me3.
Directly exposing spatial signals may help separate active from inactive genes.

Compare four variants:

- The original sequence/five-track model, freshly trained as a matched control.
- Histogram gradient boosting on the richer measured features.
- A two-layer chromatin MLP with direct expression output.
- A dual branch combining the original sequence encoder, chromatin MLP, direct
  branch outputs, and a joint predictor.

All models predict log1p expression; there is no hurdle head. Input sidecars and
normalizers contain no expression-derived feature or cell-identity embedding.
Shared extraction/loading utilities live in `common/data/chromatin.py`; the feature
recipe and comparison settings are in this topic's config. The later feature
investigation reuses this sidecar and the dual-branch runner.

## Evaluation

Three seeds (42/43/44), all three shared designs, four variants: 36 fits. Neural
variants use Smooth L1, AdamW, batch 256, up to 50 epochs, and patience eight,
selecting mean per-cell validation Spearman. Neural feature normalization uses
training examples only. Trees use transformed inputs, squared-error loss, up to
500 rounds evaluated every 25 rounds, 15 leaves, and patience eight evaluations.

Read overall Spearman, low-positive Spearman, and zero-versus-low-positive AUROC
separately. Low-positive thresholds use positive source-training Q25. A model that
improves overall ranking may still worsen the relevant low-expression separation.
Validation selects checkpoints; transfer designs share loci. This comparison did
not specify a formal statistical success threshold.

## Reproduce

Prepare the common dataset and complete sequence cache, then:

```bash
rna-pred experiment chromatin_representation prepare
rna-pred experiment chromatin_representation train --seed 42
rna-pred experiment chromatin_representation train --seed 43
rna-pred experiment chromatin_representation train --seed 44
rna-pred experiment chromatin_representation summarize
# Report only, reading existing results:
rna-pred experiment chromatin_representation report
```

Use `--resume` for interrupted fits. `--config`, `--feature-dir`, and `--output-dir`
override defaults. Existing outputs are under `outputs/experiments/chromatin_representation/`, with
`seed-<seed>/<design>/<variant>/` runs. Summarization writes numerical tables and
one multipage `summary.pdf`. For Euler submission, see `scripts/` and `docs/euler.md`
from the repository root.

## Results

`results.csv` records means, standard deviations, and defined counts across seeds.
Mean per-cell overall validation Spearman:

| Variant | A→B X1 | A→B X2 | X1→X2 | X2→X1 |
| --- | ---: | ---: | ---: | ---: |
| Matched baseline | 0.8046 | 0.8067 | 0.7592 | 0.7600 |
| Boosted trees | 0.8180 | 0.8285 | 0.7865 | 0.7835 |
| Chromatin MLP | 0.8091 | 0.8223 | 0.7880 | 0.7724 |
| Dual branch | 0.8184 | 0.8292 | 0.7971 | 0.7776 |

For A→B X2, dual-branch low-positive Spearman rose from 0.1800 to 0.2263,
while zero-versus-low-positive AUROC fell from 0.7760 to 0.7516. Other cells/designs
show different tradeoffs. X2→X1 low-positive Spearman is undefined because the
source-defined subgroup has one distinct label. The deterministic tree fits gave
identical results across the three seeds; their zero standard deviation is not
three independent estimates of uncertainty.

## Conclusion

**Mixed evidence; keep separate from the default baseline.** Richer measured
representations improve overall validation ranking. They do not consistently solve
low-expression discrimination, so describing the entire experiment as having no
numerical gains would be inaccurate. The dual branch is a useful experimental
control for the subsequent feature investigation, but is not promoted into the
stable baseline by this cleanup. Validation reuse limits independent generalization
claims; across-seed variation does not quantify uncertainty across held-out loci.
