# Additional features for low-expression errors

## Problem

The richer dual branch still struggles with low-positive expression. It might lack
an explicit description of inactive candidates, gene-body chromatin structure, or
promoter context. Descriptive signal associations alone do not establish that more
features improve prediction.

## Proposed solution and reasoning

Investigate three added feature families against the existing 911-feature sidecar:

- Enhancer/candidate summaries: all eligible versus retained DNase candidates,
  distances, counts/absence, histone peak support, and activity scarcity.
- Gene-body features: promoter-excluded signals, peak breadth, and strand-oriented
  body contrasts that might expose patterns obscured by uniform bins.
- Promoter/reference features: GC/CpG composition, chromatin contrasts, and supplied
  promoter neighborhoods that might explain residual expression differences.

Activity support is an operational BED-overlap signature, not a validated enhancer
assignment. Feature extraction uses no expression-fitted thresholds.

The staged workflow is descriptive exploration → source-training screens → selected
neural comparisons → paired genomic uncertainty. A later activity-count screen
isolates histone-supported count/absence features. After all screens were negative,
a bounded neural diagnostic tested a separate representation hypothesis: explicit
cardinality scalars might help an attention model even when they do not help trees.
This diagnostic adds only five DNase count/absence/truncation features.

## Evaluation

Screens use three chromosome-grouped folds within A, keeping both cells for a
chromosome together. Fixed histogram gradient boosting (200 rounds) compares
families, combinations, ablations and count controls. The acceptance criterion is
positive `0.5 * (2 * delta zero-low AUROC + delta low-positive Spearman)` and overall
Spearman change ≥ −0.005, averaged equally across folds/cells. Q50 is sensitivity
analysis; Q25 is primary. Thresholds use source-training labels.

No family passed. The initial negative selection was preserved and the later
five-feature diagnostic was documented in `training_decision.md`. It compared the
unchanged dual branch with added counts, seeds 42/43, all three designs: 12 fits.
Smooth L1 and mean per-cell validation-Spearman selection remained unchanged.
Uncertainty uses 300 paired resamples of 5-Mb genomic blocks, keeping matched cells
and predictions together, conditional on the fitted seeds.

## Reproduce

The chromatin representation topic supplies `outputs/experiments/chromatin_representation/features/`
and the earlier validation predictions used by exploration. Keep the original
prepared/cache artifacts and chromatin results available.

```bash
rna-pred experiment low_expression_features extract
rna-pred experiment low_expression_features explore
rna-pred experiment low_expression_features screen
rna-pred experiment low_expression_features activity-screen
# After a frozen selection/explicit diagnostic decision:
rna-pred experiment low_expression_features bundles
rna-pred experiment low_expression_features train --seed 42
rna-pred experiment low_expression_features train --seed 43
rna-pred experiment low_expression_features summarize
# Replot existing tables without extraction, screening, or training:
rna-pred experiment low_expression_features report
```

The completed run lives in `outputs/experiments/low_expression_features/`; use `--root`
for another investigation and `--chromatin-dir` for its upstream results. Its
recorded selection is frozen after training submission; screening a new root
reproduces the negative selection, not an automatic decision to run the diagnostic.
Config, selection, and source snapshots in the original outputs record the actual
run. Topic `scripts/` contains optional Euler wrappers.

Reports write one multipage `summary.pdf` from saved tables, without model loading,
custom fonts, or plotting-library checkouts. Numerical screens do not require PDF
creation. Exploration and predictive stages remain separate.

## Results

`screen_results.csv` contains the complete extended ranking; every added condition
failed the acceptance criterion. `results.csv` gives neural deltas from the matched
dual-branch control, averaged over two seeds. `intervals.csv` records paired block
intervals and defined bootstrap counts.

| Neural diagnostic | Overall Spearman change | Low-positive Spearman change | Zero-low AUROC change |
| --- | ---: | ---: | ---: |
| A→B X1 | +0.0013 | −0.0012 | +0.0063 |
| A→B X2 | −0.0066 | −0.0309 | −0.0088 |
| X1→X2 | −0.0064 | +0.0002 | −0.0097 |
| X2→X1 | −0.0013 | undefined | −0.0047 |

The undefined subgroup contains 889 low-positive examples with one distinct
expression label. Preserving missing correlations is essential.

## Conclusion

**Reject the added features as a default change.** Neither the primary feature
screens, the activity-count extension, nor the neural count diagnostic established
a consistent low-expression benefit. Small gains in A→B X1 did not transfer across
cells/designs, and A→B X2/X1→X2 overall changes exceeded the negative guardrail.
Do not promote these features or train the larger failed bundles automatically.

Common feature discovery used A labels from both cells. Transfer cells were held
out of neural optimization, but not feature discovery. Validation also selected
checkpoints. Results are exploratory; intervals describe genomic variation
conditional on two seeds and are not corrected for feature selection. Biological
associations in the exploration are not evidence of incremental predictive value.
