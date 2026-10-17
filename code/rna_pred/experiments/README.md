# Experiment index

The stable reference workflow is in `../baseline/`. Completed investigations are
kept by topic so collaborators can understand what was tested and avoid repeating
unsuccessful approaches. Each README records the problem, hypothesis, exact
control, evaluation, results, and adoption decision.

| Topic | Outputs | Decision |
| --- | --- | --- |
| [Low-expression losses](low_expression_losses/README.md) | `outputs/experiments/low_expression_losses/` | No consistent low-positive ranking improvement; retain Smooth L1. |
| [Chromatin representation](chromatin_representation/README.md) | `outputs/experiments/chromatin_representation/` | Overall improvement, mixed low-expression behavior; retain as a separate comparison. |
| [Low-expression features](low_expression_features/README.md) | `outputs/experiments/low_expression_features/` | Negative screens and no consistent neural benefit; reject added features. |

The three loss variants live in `low_expression_losses/{baseline,weighted,ranking}/`.
`outputs/baseline-evaluation/` contains the original three-design baseline run.
`outputs/archive/` holds the previous baseline evaluation and historical plotting
support. Original frozen source snapshots stay with their topics. Historical
checkpoint configs and snapshots retain their original paths; the archive's
relocation record maps those paths to their current locations.

The checked-in CSVs contain compact numerical evidence and current source paths;
large outputs remain ignored. Each topic's `report` command rebuilds one
`summary.pdf` with Matplotlib. Undefined subgroup correlations remain missing,
not zero. Results are exploratory because validation selects checkpoints; transfer
designs share loci.

For new topics, use `docs/experiment-template.md` from the repository root and add
a row here. State the matched control and success criterion before training.
