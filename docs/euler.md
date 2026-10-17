# Euler execution

Create the environment from `environment.yml` and obtain the data/assets before
submitting jobs. GPU extraction and training run in allocations; lightweight CPU
tests can use `python -m pytest -q`. Large environments, raw data, prepared tables,
caches and checkpoints should live on shared scratch accessible to compute nodes.
Symlink those directories into the configured project paths. Scratch may expire.

From the repository root:

```bash
mkdir -p logs
bash scripts/download_assets.sh
PREP_JOB=$(sbatch --parsable scripts/euler_prepare.sh)
sbatch --dependency="afterok:$PREP_JOB" --kill-on-invalid-dep=yes scripts/euler_full_cache.sh
```

After a complete cache exists, submit baseline evaluation:

```bash
sbatch scripts/euler_experiments.sh code/rna_pred/baseline/configs/default.yaml
# Interrupted run:
sbatch scripts/euler_experiments.sh code/rna_pred/baseline/configs/default.yaml --resume
```

The wrapper uses `rna_pred.common.job` to stage prepared tables/cache into node-local
`TMPDIR`, validate copies, and probe an actual forward/backward/optimizer step.
CUDA memory fallback retains the effective batch size on resume. All results are
written persistently. Check wrapper resource directives against the intended run.
Use `--overwrite` only when an archived replacement run is intended.

Experiment-specific submission scripts now live with their topics:

```bash
bash code/rna_pred/experiments/chromatin_representation/scripts/submit_chromatin_comparison.sh
python code/rna_pred/experiments/low_expression_features/scripts/submit_low_training.py
```

These commands submit jobs. The chromatin submitter freezes code/config and creates
preparation → per-seed GPU array → summary dependencies. The feature submitter
requires a recorded selection/decision and materialized prerequisites. It preserves
selection hashes and freezes source/config. Read each topic README first: the
completed feature screens were negative, and the existing neural run was an
explicitly documented diagnostic follow-up.

Submission scripts preserve full source snapshots, including the moved topic
configs/scripts. No personal plotting checkout or external font directory is
required. Pass Slurm notification settings explicitly if desired; no account-specific
mail recipient is built into the scripts.
