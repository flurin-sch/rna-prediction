"""Three explicit designs, with C predictions only from pooled A-to-B training."""
import copy
from datetime import datetime
import fcntl
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rna_pred.common.config import atomic_json
from rna_pred.common.data.dataset import PreparedData, RegionDataset
from rna_pred.baseline.model import ExpressionModel
from rna_pred.common.metrics import low_expression_metrics
from rna_pred.common.objectives import objective_settings
from rna_pred.baseline.workflow import predict, train, validate_checkpoint
from rna_pred.common.training import resolve_device, run_predictions, seed_everything


from rna_pred.common.designs import EXPERIMENTS
from rna_pred.common.metrics import regression_metrics


def evaluate(config, checkpoint_path, prepared, split, cell, output_path):
    splits = (split,) if isinstance(split, str) else tuple(split)
    cells = (cell,) if isinstance(cell, str) else tuple(cell)
    if not set(splits).issubset({"train", "val"}) or not set(cells).issubset({"X1", "X2"}):
        raise ValueError("Evaluation requires labelled X1/X2 train/val data")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_checkpoint(checkpoint, config, prepared)
    examples = prepared.select(splits, cells)
    if not np.isfinite(examples.gex).all() or (examples.gex < 0).any():
        raise ValueError("Evaluation requires finite nonnegative expression labels")
    dataset = RegionDataset(prepared, examples, checkpoint["normalizer"])
    loader = DataLoader(dataset, batch_size=config["train"]["batch_size"], num_workers=config["train"]["num_workers"])
    device = resolve_device(config["train"]["device"])
    model = ExpressionModel(dataset.feature_dim, **checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    predictions = run_predictions(model, loader, device)
    result = examples[["example_id", "cell", "split", "chr", "gene_name", "gex"]].copy()
    result["observed_log1p"] = np.log1p(result.gex)
    result["prediction_log1p"] = predictions
    with np.errstate(over="raise", invalid="raise"):
        result["prediction_gex"] = np.expm1(predictions)
    result.to_csv(output_path, index=False)
    return {**regression_metrics(examples, predictions), "selected_epoch": checkpoint["epoch"] + 1,
            "low_expression": low_expression_metrics(examples, predictions, checkpoint["expression_thresholds"])
            if "expression_thresholds" in checkpoint else None}


def run_experiments(config, output_dir=None, resume=False, plots=True, allow_limited=False,
                    overwrite=False, prepared_data=None):
    if resume and overwrite:
        raise ValueError("Cannot combine resume and overwrite")
    root = Path(output_dir or Path(config["run_dir"]).parent / "baseline-evaluation").resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    # The lock survives archiving the output directory and protects all three designs.
    with (root.parent / f".{root.name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another experiment writer holds {root}") from exc
        prepared = prepared_data if prepared_data is not None else PreparedData(config)
        if prepared.manifest["spec"]["limit_per_table"] is not None and not allow_limited:
            raise ValueError("Limited preparation requires --allow-limited; use full data for experiments")
        if root.exists() and any(root.iterdir()):
            if overwrite:
                archive = root.with_name(root.name + ".previous-" + datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z"))
                root.rename(archive)
                print(f"Archived previous experiments to {archive}", flush=True)
            elif not resume:
                raise ValueError("Experiment output exists; use --resume, --overwrite, or another output directory")
        root.mkdir(parents=True, exist_ok=True)
        if config.get("runtime"):
            atomic_json(root / "runtime.json", config["runtime"])
        rows = []
        for name, spec in EXPERIMENTS.items():
            current = copy.deepcopy(config)
            current["run_dir"] = str(root / name)
            folder = Path(current["run_dir"])
            atomic_json(folder / "design.json", {
                "experiment": name, **spec, "selection_metric": current["train"].get("selection_metric", "val_loss"),
                "objective": objective_settings(current["train"]),
                "spearman_aggregation": ("mean_per_cell" if current["train"].get("selection_metric") == "val_mean_spearman"
                                         else "combined_gene_cell_examples"),
                "all_chromosomes": "all supplied labelled chromosomes, chr2-22",
                "selection_caveat": "Validation participates in checkpoint selection; transfer designs share gene loci across cells",
                "predict_C": name == "a_to_b", "limited_data": prepared.manifest["spec"]["limit_per_table"] is not None,
            })
            best = train(current, resume=resume and (folder / "last.pt").exists(), prepared_data=prepared, **spec)
            seed_everything(current["train"]["seed"], current["train"]["cpu_threads"])
            for role, prefix in (("training", "train"), ("validation", "val")):
                metrics = evaluate(current, best, prepared, spec[f"{prefix}_splits"], spec[f"{prefix}_cells"],
                                   folder / f"{role}_predictions.csv")
                rows.append({"experiment": name, "role": role, "cells": "+".join(spec[f"{prefix}_cells"]),
                             "source_splits": "+".join(spec[f"{prefix}_splits"]),
                             **{k: v for k, v in metrics.items() if k != "spearman_per_cell"}})
            atomic_json(folder / "evaluation.json", rows[-2:])
            if name == "a_to_b":
                predict(current, best, folder / "C_predictions.csv", prepared_data=prepared)
            # Persist completed designs even when a later design is interrupted.
            pd.DataFrame(rows).to_csv(root / "metrics.csv", index=False)
            atomic_json(root / "experiments.json", {"schema": 2, "runs": EXPERIMENTS, "metrics": rows,
                       "test_predictions_created": True,
                       "data_provenance": config.get("data_provenance", {"kind": "supplied", "description": "Supplied CAGE labels and cached AlphaGenome embeddings"}),
                       "completed_designs": list(dict.fromkeys(r["experiment"] for r in rows)),
                       "limited_data": prepared.manifest["spec"]["limit_per_table"] is not None})
        if plots:
            from rna_pred.baseline.plots import plot_all
            plot_all(root)
    print(f"Experiment results saved to {root}", flush=True)
    return root
