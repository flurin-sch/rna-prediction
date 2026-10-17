"""Euler job startup: stage immutable inputs, check the batch, then replace results."""
import argparse
import copy
import gc
from pathlib import Path
import os
import shutil
import subprocess
import time

import torch
from torch.utils.data import DataLoader

from rna_pred.common.data.cache import validate_cache
from rna_pred.common.config import atomic_json, load_config, read_json
from rna_pred.common.data.dataset import PreparedData, RegionDataset, fit_normalizer
from rna_pred.baseline.evaluation import run_experiments
from rna_pred.baseline.model import ExpressionModel
from rna_pred.common.training import resolve_device, restore_rng, rng_state, seed_everything


def stage_inputs(config, temporary_dir):
    began = time.perf_counter()
    temporary = Path(temporary_dir).resolve()
    if not temporary.is_dir():
        raise ValueError("TMPDIR must be an existing job-local directory")
    sources = {key: Path(config[key]).resolve(strict=True) for key in ("prepared_dir", "cache_dir")}
    if any(temporary == source or temporary.is_relative_to(source) for source in sources.values()):
        raise ValueError("Staging directory must be separate from source inputs")
    required = sum(p.stat().st_size for source in sources.values() for p in source.rglob("*") if p.is_file())
    if shutil.disk_usage(temporary).free < required + 512 * 1024 ** 2:
        raise ValueError("Insufficient node-local disk space for input staging")
    stage = temporary / ("rna-pred-" + os.environ.get("SLURM_JOB_ID", str(os.getpid())))
    stage.mkdir(exist_ok=False)
    staged = copy.deepcopy(config)
    for key, source in sources.items():
        target = stage / key
        target.mkdir()
        subprocess.run(["rsync", "-aL", "--", str(source) + "/", str(target) + "/"], check=True)
        staged[key] = str(target)
    # These checks include every prepared file, every cache vector/index/metadata hash,
    # full coverage, and finite vectors. Copied manifests keep their original identities.
    validation = validate_cache(staged)
    staged["runtime"] = {"slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                         "persistent_inputs": {k: str(v) for k, v in sources.items()},
                         "staged_inputs": {k: staged[k] for k in sources},
                         "staging_seconds": time.perf_counter() - began,
                         "input_bytes": required, "prepared_identity": validation["prepared_identity"],
                         "cache_identity": validation["cache_identity"]}
    print(f"Staged and verified {required / 1024**3:.2f} GiB in {staged['runtime']['staging_seconds']:.1f}s at {stage}", flush=True)
    return staged


def _probe_batch(config, prepared, batch_size, device):
    settings = config["train"]
    examples = prepared.select("train", ("X1", "X2"))
    # Dense examples test the full padded region shape used by every design.
    counts = prepared.mapping.groupby("example_id").size()
    order = examples.example_id.map(counts).sort_values(ascending=False, kind="stable").index
    examples = examples.loc[order].reset_index(drop=True)
    normalizer = fit_normalizer(prepared.mapping, examples.example_id, config["tracks"])
    dataset = RegionDataset(prepared, examples, normalizer)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    began = time.perf_counter()
    model = ExpressionModel(dataset.feature_dim, **config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    batch = next(iter(DataLoader(dataset, batch_size=batch_size, num_workers=0)))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch["sequence"].to(device), batch["features"].to(device), batch["valid"].to(device))
    from rna_pred.common.metrics import fit_thresholds
    from rna_pred.common.objectives import objective_settings, expression_loss
    objective = objective_settings(settings)
    thresholds = fit_thresholds(examples, objective["low_quantile"])
    cells = examples.iloc[batch["index"].numpy()].cell.tolist()
    loss = expression_loss(output, batch["target"].to(device), cells, thresholds, objective)
    if not torch.isfinite(loss):
        raise ValueError("Batch preflight produced nonfinite loss")
    loss.backward()
    optimizer.step()  # Includes lazy optimizer-state allocation in the memory check.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {"batch_size": batch_size, "examples_checked": len(output),
              "seconds": time.perf_counter() - began,
              "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
    # This isolated model and optimizer never enter real training or its checkpoints.
    return result


def check_batch(config, prepared, allow_fallback=True):
    generator = torch.Generator().manual_seed(config["train"]["seed"])
    original_rng = rng_state(generator)
    requested = config["train"]["batch_size"]
    candidates = [requested] + ([n for n in (128, 64) if n < requested] if allow_fallback else [])
    device = resolve_device(config["train"]["device"])
    attempts = []
    try:
        for candidate in candidates:
            seed_everything(config["train"]["seed"], config["train"]["cpu_threads"])
            try:
                measured = _probe_batch(config, prepared, candidate, device)
            except torch.cuda.OutOfMemoryError:
                attempts.append({"batch_size": candidate, "result": "CUDA out of memory"})
                print(f"Batch {candidate} exceeded GPU memory; trying a smaller batch", flush=True)
            else:
                attempts.append({**measured, "result": "passed"})
                return {"requested_batch_size": requested, "effective_batch_size": candidate, "attempts": attempts}
            finally:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        raise RuntimeError("No candidate batch fits GPU memory")
    finally:
        restore_rng(original_rng, generator)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("TMPDIR"):
        raise RuntimeError("Run this entrypoint inside a Slurm job with TMPDIR and --tmp requested")
    config = load_config(args.config)
    root = Path(args.output_dir or Path(config["run_dir"]).parent / "baseline-evaluation").resolve()
    # Resuming a run whose initial preflight fell back must retain its effective batch.
    if args.resume and (root / "runtime.json").exists():
        runtime = read_json(root / "runtime.json")
        config["train"]["batch_size"] = runtime["batch_preflight"]["effective_batch_size"]
    staged = stage_inputs(config, os.environ["TMPDIR"])
    prepared = PreparedData(staged)
    report = check_batch(staged, prepared, allow_fallback=not args.resume)
    staged["train"]["batch_size"] = report["effective_batch_size"]
    staged["runtime"]["batch_preflight"] = report
    print(f"Batch preflight: {report}", flush=True)
    # Persist preflight diagnostics outside the result directory before any archiving.
    log_dir = Path(config["project_root"]) / "logs"
    atomic_json(log_dir / f"experiment-preflight-{os.environ['SLURM_JOB_ID']}.json", staged["runtime"])
    run_experiments(staged, root, resume=args.resume, plots=not args.no_plots,
                    overwrite=args.overwrite, prepared_data=prepared)


if __name__ == "__main__":
    main()
