# RNA prediction from sequence and chromatin

Predict CAGE expression using frozen AlphaGenome sequence features and measured
chromatin signals. The default workflow is the original promoter/candidate-region
interaction model. Completed experiments live in separate topic folders, with their
hypotheses, reproduction commands, results, and conclusions.

The baseline is inspired by EPInformer's promoter–enhancer interaction architecture,
representing each gene as a promoter token plus up to 60 nearby DNase-defined
candidate regulatory regions. Each region combines frozen, cached AlphaGenome
sequence embeddings with measured DNase and histone signals, strand-relative
distance, and a promoter indicator through a trainable projection and fusion MLP.
Three masked attention blocks model promoter–candidate interactions while blocking
direct attention between distinct candidates, and an MLP reads the final promoter
token to predict log1p CAGE expression. Using pretrained contextual sequence
features lets us reuse sequence representations across cells while learning how
cell-specific chromatin signals contribute to expression.

## Setup

Use Python 3.12 or newer. For training with an existing feature cache:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

To extract AlphaGenome features, also install the pinned extraction dependency:

```bash
python -m pip install -e '.[extraction]'
```

Alternatively, `environment.yml` installs the complete environment:

```bash
mamba env create -p "$PWD/.conda" -f environment.yml
mamba activate "$PWD/.conda"
```

Tests use synthetic data and a fake sequence extractor on CPU. They verify software
behavior; predictive performance comes from the saved real-data comparisons.

## Baseline workflow

The supplied competition data belongs under `data/ml-4-g-project-1/`, containing
`CAGE-train/CAGE-train/*.tsv`, `DNase-bed/X*.bed`, and `<assay>-bigwig/X*.bw`
(or `.bigwig`) for X1/X2/X3. Obtain competition access and download the data first.
`scripts/download_kaggle_competition.py` is an optional per-file Kaggle downloader;
it requires the Kaggle client and credentials separately.

Run commands from the repository root:

```bash
bash scripts/download_assets.sh
rna-pred prepare
rna-pred cache --device cuda
rna-pred validate-cache
rna-pred train
rna-pred predict
```

The asset script downloads hg38 and the pinned converted AlphaGenome weights.
AlphaGenome extraction requires GPU memory; use an allocation on a cluster.
The weights are subject to AlphaGenome's non-commercial terms.

The default config is
[`code/rna_pred/baseline/configs/default.yaml`](code/rna_pred/baseline/configs/default.yaml).
Override it with `rna-pred --config PATH ...`. YAML paths are relative to
`project_root`, which is relative to the config file. Large data/environment
folders can be placed on shared scratch; see [Euler execution](docs/euler.md).

`prepare` writes checksummed example/region/signal tables. `cache` resumes missing
frozen features. `train --resume` continues a compatible checkpoint; a fresh run
requires a new `run_dir`. `predict` writes `outputs/default/submission.csv` in
original X3 row order. Predictions are never uploaded automatically.

For the three standard evaluation designs and their summary PDF:

```bash
rna-pred evaluate --output-dir outputs/baseline-evaluation
rna-pred report --output-dir outputs/baseline-evaluation
```

A→B trains on X1+X2 chromosome set A and validates on set B. X1→X2 and X2→X1
train on one cell's A+B and validate on the other cell's A+B. Validation selects
checkpoints; transfer designs share loci across cells. These are exploratory
validation results, not independent test estimates. See the
[baseline README](code/rna_pred/baseline/README.md) for model and data conventions.

## Experiments

| Topic | Question | Recorded conclusion |
| --- | --- | --- |
| [Low-expression losses](code/rna_pred/experiments/low_expression_losses/README.md) | Can weighting or ranking improve low-positive expression ordering? | No consistent benefit; keep Smooth L1. |
| [Chromatin representation](code/rna_pred/experiments/chromatin_representation/README.md) | Do richer measured features and alternative models help? | Overall Spearman improved, but low-expression discrimination was mixed; keep this separate from the default baseline. |
| [Low-expression features](code/rna_pred/experiments/low_expression_features/README.md) | Do activity scarcity, gene-body, and promoter features address remaining errors? | No added family passed screening; the count-feature neural diagnostic showed no consistent benefit. |

Every topic contains its code, configuration, README, and small versioned results
tables. Full predictions, checkpoints, and frozen source snapshots remain under
`outputs/` and are not included in Git. The [experiment index](code/rna_pred/experiments/README.md)
lists their output locations and conclusions.

Use `rna-pred experiment TOPIC --help` for topic commands. Regenerate reports from
the existing saved results without training:

```bash
rna-pred report
rna-pred experiment low_expression_losses report
rna-pred experiment chromatin_representation report
rna-pred experiment low_expression_features report
```

Each workflow writes one multipage `summary.pdf` in its output directory, using
default Matplotlib styling. Loss comparisons use
`outputs/experiments/low_expression_losses/summary.pdf`. No caption or figure-legend
files are generated. Reports need only saved results and the declared dependencies.

## Saved output layout

```text
outputs/
  baseline-evaluation/                 three-design baseline results and summary.pdf
  experiments/
    low_expression_losses/             baseline/, weighted/, ranking/, summary.pdf
    chromatin_representation/           features/, seed-*/ runs, summary.pdf
    low_expression_features/            features, screens, neural runs, tables, summary.pdf
  smoke/                               small pipeline verification run
  archive/                             previous evaluations and historical plotting support
```

Each topic keeps its checkpoints, predictions, source snapshots, and numerical
summaries together. Obsolete individual figures, previews, and legend files have
been removed from active runs. The archive contains a checksum-verified relocation
record; historical checkpoint configs and frozen snapshots retain their original
paths as provenance. A fresh single-model baseline run writes `outputs/default/`.

## Code organization and contributing

```text
code/rna_pred/
  cli.py
  common/                         configuration, data/cache, metrics, training, PDFs
  baseline/                       original model, workflow, evaluation, configs
  experiments/
    low_expression_losses/        weighted/ranking objectives and matched controls
    chromatin_representation/     alternative models and comparison runner
    low_expression_features/      feature extraction, screens, neural follow-up
scripts/                          shared downloads and baseline Euler wrappers
tests/                            CPU regression and integration tests
docs/                             Euler instructions and verification records
```

Experiments reuse the common data contracts and neural optimization loop. The
baseline retains loss-based checkpoint selection; experimental comparisons retain
their recorded mean per-cell Spearman selection. The richer chromatin sidecar is
shared by the latter two topics. Model settings, normalization, and input/cache
identities are checked on resume.

For a new experiment, copy the [README template](docs/experiment-template.md), place
its recipe and code under a new topic folder, identify the exact matched control,
and define success before running it. Afterward, add numerical results and an
adopt/reject/mixed conclusion to its README and update the experiment index.
Preserve undefined subgroup correlations as missing values. Run `python -m pytest -q`
after code changes. See [verification](docs/verification.md) for historical and
migration checks.
