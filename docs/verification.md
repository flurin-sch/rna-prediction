# Verification records

The historical record below describes the initial implementation before the topic
reorganization. Its old command/config paths are historical; current commands are
in the root and topic READMEs. Migration validation is recorded above that history.

## Reorganization checks

Verified on 2026-10-17:

- `python -m pytest -q`: 59 passed, including cache recovery, normalization,
  checkpoint resume, objective selection, all model variants, negative screens,
  report-only entry points, and moved config paths.
- Exact CPU parity against the saved pre-migration source on the same synthetic
  pipeline: baseline predictions, model parameters, and histories; dual-branch
  model parameters and selected epoch. This verifies implementation parity, not
  biological predictive performance.
- Existing real baseline and dual-branch checkpoints load into the moved models
  and give identical predictions to the original classes on the same inputs.
  The chromatin checkpoint's resume signature matches the moved config.
- Rebuilt baseline and all three experiment-topic PDFs from real saved results,
  without training or sequence extraction. Each report is a single multipage PDF;
  report tests check page counts and missing subgroup metrics.
- Built a wheel and checked it contains moved configs, topic READMEs, results CSVs,
  and submission scripts. Checked shell syntax and current documentation links.

No production training was repeated and no cluster jobs were submitted.
Original numerical outputs, checkpoints, caches, and frozen source snapshots were
preserved. Generated reports are under their existing output roots. A subsequent output cleanup removed obsolete individual plots and legend files
from active runs; current commands write only `summary.pdf`.

## Output cleanup (2026-10-17)

- Moved the baseline evaluation to `outputs/baseline-evaluation/` and all topic
  results to `outputs/experiments/<topic>/`; loss variants live inside their topic.
  Updated CLI defaults, configs, scripts, README commands, and compact source paths.
- Checksum-verified 1,070 retained files. Checkpoints, prediction tables, feature
  sidecars, and frozen sources are unchanged. The previous baseline evaluation and
  historical plotting support are archived. Removed 46 obsolete plots, previews,
  legends, or byte-identical duplicate artifacts from active outputs.
- Regenerated all four current `summary.pdf` files with default commands; page
  counts are baseline 3, losses 8, chromatin representations 4, and added features 6.
- Revalidated moved sidecars and an existing chromatin checkpoint resume signature.
  `python -m pytest -q`: 59 passed. Shell syntax and current documentation links pass.
- `outputs/archive/layout-migration-20261017.json` records relocations, removed
  artifacts, preservation checksums, and regenerated reports. Historical records
  retain their original paths as provenance. No training or cluster jobs were run.

## Historical implementation verification

# First implementation verification

Verified on 14 October 2026. This records software/inference checks, not a model
comparison or an estimate of predictive performance.

## Automated checks

`pytest -q`: **10 passed**. Tests cover deterministic candidate selection and summit
fallback, promoter exclusion, chromosome-edge padding, partial-bin pooling,
training-only signal normalization, attention-mask structure, padded-input isolation,
enhancer permutation invariance, promoter-only examples, cache recovery/invalidation
and corruption detection, training/prediction, and exact training resume.

The end-to-end automated tests use an explicitly fake sequence extractor. Real
AlphaGenome extraction was checked separately below. Python compilation, shell
syntax, `git diff --check`, and `pip check` also passed. All supplied gene/label
tables were loaded and validated; X1/X2 labels are finite and nonnegative.

## Real AlphaGenome smoke check

- Inputs: first two genes from each supplied table, ten gene/cell examples total.
- Context: six unique native 262,144-bp hg38 windows, reference orientation.
- Regions: 221 pooled promoter/candidate vectors; cached shape **[221, 3072]**,
  float32, all finite. Different regions within a shared window have distinct vectors.
- Extractor: pinned AlphaGenome PyTorch commit
  `72268c011d37cf2d387506493614952d85abe21b`, all-folds weights from Hugging Face
  revision `b01c0ffa73e07c053491f3b5ea8bcf67d93b9920`. Transformer retained,
  1-bp decoder and prediction heads skipped; frozen evaluation/inference mode,
  bfloat16 compute and float32 pooling/storage.
- Hardware: NVIDIA GeForce RTX 4090, PyTorch 2.8.0+cu128, Python 3.12.14.
- Peak allocated GPU memory: **6,337,147,904 bytes (5.90 GiB)**. This is PyTorch
  allocated memory, not total device reservation.
- Cache-stage time: **25.42 seconds**, including model loading, one repeat check,
  pooling and shard writes. Initial imports and reference/checkpoint hashing occur
  before that timer. Total Slurm batch duration was 82 seconds.
- Repeated first extraction: maximum absolute embedding difference **0.0**.
- Euler verification job **17089118** completed with exit code **0:0**.

Details and checksums are retained in `data/cache/alphagenome-smoke/`; the extraction
log is `logs/rna-pred-smoke-17089118.out`. The reference and checkpoint SHA-256 hashes
are in the cache manifest. These six windows are a functional pilot, not a reliable
throughput estimate for the complete project.

## Real end-to-end check

The smoke configuration trained for two epochs on the four X1/X2 training examples,
validated on four examples, and exported two X3 predictions. Losses were finite
(3.69564 and 3.66042). Checkpoint reload produced a **byte-identical CSV**. Artifacts
are under `outputs/smoke/`.

The smoke CSV is intentionally incomplete and must not be submitted to Kaggle.
Its tiny validation set provides no useful accuracy conclusion. Full cache production,
full training, experimental comparisons, and competition submission were not run.

## Environment and storage

`environment.yml` defines the environment; `environment.resolved.yml` records the
installed package versions, retaining the exact AlphaGenome git dependency and a
portable editable-project path. Project-generated large files live under
`/cluster/scratch/fschindele/rna-pred-implementation/`, linked through `.conda`,
`data/reference`, `data/checkpoints`, `data/prepared`, and `data/cache`. Supplied raw
data were not modified. Scratch assets can be recreated using the documented setup.


## Experimental-design and convergence update, 15 October 2026

Implemented pooled A-to-B training with C export and both A+B cell-transfer
directions. Training uses batch 256, complete end-of-epoch training/validation
loss and combined Spearman, minimum-validation-loss checkpoint selection, and
independent training-plateau/validation-deterioration stopping counters.

Verification: `python -m pytest -q` completed with **24 passed**, with the expected
Arial-unavailable warning and approved editable DejaVu Sans fallback. Tests include
exact interrupted/resumed training, loss rather than Spearman checkpoint selection,
threshold boundaries, sustained stopping and resume, three-design selections,
training-only normalization, C prediction ordering, symlink-aware input staging,
corruption/capacity rejection, identical staged predictions, batch fallback and RNG
restoration, safe result archiving, and complete job startup/resume with different
node-local paths. These synthetic tests establish no biological performance.

Three PDF drafts were generated from synthetic saved results. Independent layout
and legend reviewers followed the sci-figures review procedure. All actionable
findings were addressed: legend/selection-line clearance, sparse-point visibility,
panel lettering and alignment, shared labels, useful performance limits, actual
chromosome definitions, explicit synthetic provenance, panel sample counts/scales,
coincident-mark explanations, abbreviation definitions, and redundant wording.
No findings were declined. The final revised layouts were inspected visually.

Python compilation, shell syntax, and whitespace checks passed. Slurm's
`--test-only` accepted the production request. Requested resources are one RTX 4090,
24 GiB GPU memory, four CPUs, 32 GiB RAM, 10 GiB node-local disk, and 36 hours.
The job stages and verifies approximately 6.1 GiB of inputs before checking the
real batch and archiving prior results. Production outputs remain on persistent
storage. Full-data results and final figures will be produced by the submitted job.

Submitted retraining job **17140512**; submission provenance: `logs/experiment-retrain-17140512.json`.
