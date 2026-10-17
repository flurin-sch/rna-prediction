# Low-expression training losses

## Problem

Overall expression prediction is substantially stronger than ranking among
low-positive genes. Smooth L1 optimizes prediction magnitude, which may give
insufficient emphasis to distinctions near zero. Exact-zero labels are tied;
no ordering within those ties is identifiable.

## Proposed solution and reasoning

Compare three objectives with the same sequence/track architecture:

- Smooth L1 control.
- Weighted Smooth L1: twice the weight on low-positive training examples, normalized
  by the sum of weights. This increases their contribution while keeping a bounded
  weight and retaining magnitude supervision.
- Hybrid ranking: Smooth L1 plus 0.1 times within-cell pairwise logistic loss. This
  explicitly trains ordering without requiring expression distances to be accurate.
  Unequal-label pairs are grouped into zero/positive, low/low, and other pairs;
  low/low groups have weight two and each group is sampled up to 1,024 pairs/batch.

Low-positive means 0 < expression ≤ the 25th percentile of positive training labels.
Thresholds are fitted per source training cell; unseen validation cells use the
pooled source threshold. Ranking temperature is one. `losses.py` contains the
experimental objectives; baseline Smooth L1 and shared diagnostics remain common.

## Evaluation

One seed (42), the three shared A→B/X1→X2/X2→X1 designs, AdamW, batch 256,
maximum 50 epochs, and patience eight. All three arms select the highest mean
per-cell validation Spearman. This matched control differs from the original
loss-selected baseline evaluation. Objectives change optimization; reported losses
remain unweighted Smooth L1 for comparison.

The scientific question is consistent low-positive ranking improvement while
retaining overall performance. This suite did not define a statistical acceptance
threshold. Small changes from one seed do not establish an improvement.
Validation selects checkpoints, and cell-transfer designs share gene loci.

## Reproduce

Use the common prepared dataset and complete AlphaGenome cache. Configs are in
`configs/{baseline,weighted,ranking}.yaml`.

```bash
# Fresh matched comparison, all three objectives and all three designs:
rna-pred experiment low_expression_losses run
# Resume compatible runs:
rna-pred experiment low_expression_losses run --resume
# Summarize/replot that comparison:
rna-pred experiment low_expression_losses report
```

Saved and new suite outputs use `outputs/experiments/low_expression_losses/`, with
one folder per objective, `results.csv`, and one multipage `summary.pdf`.
For a single objective, the baseline `train` command also accepts its config.

## Results

Original runs are complete. `results.csv` records the validation values, subgroup
sizes, and source directories. Combined validation Spearman:

| Objective | A→B | X1→X2 | X2→X1 |
| --- | ---: | ---: | ---: |
| Smooth L1 control | 0.8054 | 0.7584 | 0.7577 |
| Weighted Smooth L1 | 0.7981 | 0.7499 | 0.7518 |
| Hybrid ranking | 0.8076 | 0.7592 | 0.7552 |

A→B low-positive Spearman provides a clearer view of the intended target:

| Objective | X1 | X2 |
| --- | ---: | ---: |
| Smooth L1 control | 0.1239 | 0.2095 |
| Weighted Smooth L1 | 0.1446 | 0.1462 |
| Hybrid ranking | 0.1140 | 0.1841 |

X2→X1 low-positive Spearman is undefined: the source-defined target subgroup has
one distinct expression label. It must remain missing rather than be scored zero.

## Conclusion

**Reject as a default change.** Weighting reduced overall ranking in all designs
and helped low-positive ranking in only one A→B cell. Hybrid ranking produced small
overall gains in two designs, but reduced low-positive A→B ranking in both cells.
Neither objective consistently solved the intended low-expression problem.
Keep unweighted Smooth L1 in the common baseline. These are single-seed,
validation-selected comparisons, so small differences remain exploratory.
