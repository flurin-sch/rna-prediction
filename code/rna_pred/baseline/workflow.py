from pathlib import Path
import warnings
import time
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rna_pred.common.config import atomic_json
from rna_pred.common.data.dataset import PreparedData, RegionDataset, fit_normalizer
from rna_pred.baseline.model import ExpressionModel
from rna_pred.common.objectives import objective_settings, expression_loss
from rna_pred.common.metrics import fit_thresholds, low_expression_metrics, validation_metrics, prediction_loss, spearman_score
from rna_pred.common.training import (seed_everything, resolve_device, atomic_torch_save, rng_state, restore_rng,
                                      run_predictions, initial_stopping_state, update_stopping, neural_epochs)


def train(config, resume=False, train_cells=("X1", "X2"), val_cells=("X1", "X2"),
          train_splits=("train",), val_splits=("val",), prepared_data=None):
    settings = config["train"]
    objective = objective_settings(settings)
    selection_metric = settings.get("selection_metric", "val_loss")
    if selection_metric not in {"val_loss", "val_mean_spearman", "val_spearman"}:
        raise ValueError("Unknown selection_metric")
    if settings["epochs"] < 1 or settings["patience"] < 1:
        raise ValueError("Training epochs and patience must be positive")
    for key in ("train_min_relative_improvement", "val_min_relative_rise"):
        if not 0 <= settings.get(key, 0.001) < 1:
            raise ValueError(f"Invalid {key}")
    selection = {"train_cells": list(train_cells), "val_cells": list(val_cells),
                 "train_splits": list(train_splits), "val_splits": list(val_splits)}
    if not set(train_splits).issubset({"train", "val"}) or not set(val_splits).issubset({"train", "val"}):
        raise ValueError("Training and validation require labelled train/val splits")
    seed_everything(settings["seed"], settings["cpu_threads"])
    prepared = prepared_data if prepared_data is not None else PreparedData(config)
    train_examples = prepared.select(train_splits, train_cells)
    val_examples = prepared.select(val_splits, val_cells)
    if set(train_examples.example_id) & set(val_examples.example_id):
        raise ValueError("Training and validation examples overlap")
    for examples in (train_examples, val_examples):
        if not np.isfinite(examples.gex).all() or (examples.gex < 0).any():
            raise ValueError("Training and validation require finite nonnegative labels")
    out = Path(config["run_dir"])
    out.mkdir(parents=True, exist_ok=True)
    last_path, best_path = out / "last.pt", out / "best.pt"
    if (last_path.exists() or best_path.exists()) and not resume:
        raise ValueError("Run already has checkpoints; use --resume or choose another run_dir")
    checkpoint = torch.load(last_path, map_location="cpu", weights_only=True) if resume else None
    if checkpoint:
        if checkpoint.get("schema") != 2:
            raise ValueError("Resume requires a schema-2 checkpoint; start a fresh run")
        validate_checkpoint(checkpoint, config, prepared)
        if objective_settings(checkpoint["config"]["train"]) != objective:
            raise ValueError("Cannot change objective during training resume")
        if checkpoint.get("selection_metric", "val_loss") != selection_metric:
            raise ValueError("Cannot change selection_metric during training resume")
        for key in ("seed", "batch_size", "learning_rate", "weight_decay", "num_workers", "cpu_threads",
                    "patience", "train_min_relative_improvement", "val_min_relative_rise"):
            default = 0.001 if key in ("train_min_relative_improvement", "val_min_relative_rise") else None
            if checkpoint["config"]["train"].get(key, default) != settings.get(key, default):
                raise ValueError(f"Cannot change {key} during training resume")
        if any(checkpoint[key] != value for key, value in selection.items()):
            raise ValueError("Cannot change selected cells or splits during training resume")
        if not best_path.exists():
            raise ValueError("Resume requires the saved best checkpoint alongside last.pt")
        normalizer = checkpoint["normalizer"]
    else:
        normalizer = fit_normalizer(prepared.mapping, train_examples.example_id, config["tracks"])
    thresholds = fit_thresholds(train_examples, objective["low_quantile"])
    if checkpoint and checkpoint.get("expression_thresholds", thresholds) != thresholds:
        raise ValueError("Training expression thresholds changed during resume")
    train_data, val_data = (RegionDataset(prepared, examples, normalizer) for examples in (train_examples, val_examples))
    generator = torch.Generator().manual_seed(settings["seed"])
    train_loader = DataLoader(train_data, batch_size=settings["batch_size"], shuffle=True,
                              num_workers=settings["num_workers"], generator=generator)
    train_eval_loader = DataLoader(train_data, batch_size=settings["batch_size"], num_workers=settings["num_workers"])
    val_loader = DataLoader(val_data, batch_size=settings["batch_size"], num_workers=settings["num_workers"])
    device = resolve_device(settings["device"])
    model = ExpressionModel(train_data.feature_dim, **config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    start, state, history = 0, initial_stopping_state(), []
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, state, history = checkpoint["epoch"] + 1, checkpoint["stopping_state"], checkpoint["history"]
        restore_rng(checkpoint["rng"], generator)
    stopped = (min(state["train_plateau_epochs"], state["val_rise_epochs"]) >= settings["patience"]
               if selection_metric == "val_loss" else state.get("selection_plateau_epochs", 0) >= settings["patience"])
    def loss_fn(batch, output):
        cells = train_examples.iloc[batch["index"].numpy()].cell.tolist()
        return expression_loss(output, batch["target"].to(device), cells, thresholds, objective)
    for epoch, total_loss, train_predictions, predictions, began in neural_epochs(
            model, train_loader, train_eval_loader, val_loader, optimizer, device, loss_fn,
            start=start, epochs=settings["epochs"], stopped=lambda: stopped, predict_fn=run_predictions):
        train_loss = prediction_loss(train_examples, train_predictions)
        val_loss = prediction_loss(val_examples, predictions)
        train_metrics = validation_metrics(train_examples, train_predictions)
        metrics = validation_metrics(val_examples, predictions)
        state, improved, stopped = update_stopping(state, train_loss, val_loss, settings)
        if selection_metric != "val_loss":
            score = metrics["mean_spearman" if selection_metric == "val_mean_spearman" else "combined_spearman"]
            improved = state.get("best_selection_score") is None or score > state["best_selection_score"]
            state["best_selection_score"] = score if improved else state["best_selection_score"]
            state["selection_plateau_epochs"] = 0 if improved else state.get("selection_plateau_epochs", 0) + 1
            stopped = state["selection_plateau_epochs"] >= settings["patience"]
        record = {"epoch": epoch + 1, "optimization_loss": total_loss / len(train_data),
                  "train_loss": train_loss, "val_loss": val_loss,
                  "train_spearman": train_metrics["combined_spearman"], "val_spearman": metrics["combined_spearman"],
                  "val_mean_spearman": metrics["mean_spearman"],
                  "train_low_expression": low_expression_metrics(train_examples, train_predictions, thresholds),
                  "val_low_expression": low_expression_metrics(val_examples, predictions, thresholds),
                  "train_spearman_per_cell": train_metrics["spearman"], "val_spearman_per_cell": metrics["spearman"], **state}
        history.append(record)
        checkpoint = {"schema": 2, "epoch": epoch, "selection_metric": selection_metric, "stopping_state": state,
                      "expression_thresholds": thresholds, "objective": objective,
                      "model": model.state_dict(), "optimizer": optimizer.state_dict(), "normalizer": normalizer,
                      "config": config, "feature_columns": prepared.manifest["feature_columns"],
                      "prepared_identity": prepared.manifest["identity"], "cache_identity": prepared.cache.manifest["identity"],
                      "cache_index_sha256": prepared.cache.manifest["index_sha256"], **selection,
                      "rng": rng_state(generator), "history": history}
        atomic_torch_save(last_path, checkpoint)
        if improved:
            atomic_torch_save(best_path, checkpoint)
            val_output = val_examples[["example_id", "cell", "split", "chr", "gene_name", "gex"]].copy()
            val_output["prediction_log1p"] = predictions
            val_output.to_csv(out / "validation_predictions.csv", index=False)
        atomic_json(out / "history.json", history)
        elapsed = time.perf_counter() - began
        performance = {"epoch": epoch + 1, "epoch_seconds": elapsed,
                       "examples_processed": 2 * len(train_data) + len(val_data),
                       "examples_per_second": (2 * len(train_data) + len(val_data)) / elapsed,
                       "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                       "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
        with (out / "performance.jsonl").open("a") as handle:
            handle.write(json.dumps(performance) + "\n")
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}, "
              f"train_spearman={record['train_spearman']:.5f}, val_spearman={record['val_spearman']:.5f}, "
              f"val_mean_spearman={record['val_mean_spearman']:.5f}, selection={selection_metric}, "
              f"seconds={elapsed:.1f}, examples/s={performance['examples_per_second']:.1f}, "
              f"peak_gpu_bytes={performance['peak_gpu_allocated_bytes']}", flush=True)
    atomic_json(out / "run.json", {"config": config, **selection, "selection_metric": selection_metric, **state,
                                   "expression_thresholds": thresholds, "objective": objective,
                                   "termination_reason": "early_stopping" if stopped else "epoch_cap",
                                   "epochs_completed": len(history), "prepared_identity": prepared.manifest["identity"],
                                   "cache_identity": prepared.cache.manifest["identity"]})
    return best_path


def validate_checkpoint(checkpoint, config, prepared):
    if checkpoint["config"]["model"] != config["model"] or checkpoint["feature_columns"] != prepared.manifest["feature_columns"]:
        raise ValueError("Model or feature configuration does not match checkpoint")
    if checkpoint["prepared_identity"] != prepared.manifest["identity"]:
        raise ValueError("Prepared data does not match checkpoint")
    if checkpoint["cache_identity"] != prepared.cache.manifest["identity"]:
        raise ValueError("Sequence extraction settings do not match checkpoint")
    if checkpoint["cache_index_sha256"] != prepared.cache.manifest["index_sha256"]:
        raise ValueError("Sequence cache index changed since training")


def predict(config, checkpoint_path=None, output_path=None, prepared_data=None):
    seed_everything(config["train"]["seed"], config["train"]["cpu_threads"])
    prepared = prepared_data if prepared_data is not None else PreparedData(config)
    path = Path(checkpoint_path or Path(config["run_dir"]) / "best.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    validate_checkpoint(checkpoint, config, prepared)
    examples = prepared.select("test", ["X3"]).sort_values("row_order").reset_index(drop=True)
    dataset = RegionDataset(prepared, examples, checkpoint["normalizer"])
    loader = DataLoader(dataset, batch_size=config["train"]["batch_size"], num_workers=config["train"]["num_workers"])
    device = resolve_device(config["train"]["device"])
    model = ExpressionModel(dataset.feature_dim, **checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    values = run_predictions(model, loader, device)
    with np.errstate(over="raise", invalid="raise"):
        predictions = np.expm1(values)
    result = pd.DataFrame({"gene_name": examples.gene_name.to_numpy(), "gex_predicted": predictions})
    if not result.gene_name.is_unique or not np.isfinite(predictions).all() or len(result) != len(examples):
        raise ValueError("Prediction output failed format checks")
    output_path = Path(output_path or Path(config["run_dir"]) / "submission.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    if prepared.manifest["spec"]["limit_per_table"] is not None:
        warnings.warn("Prepared data was limited for smoke testing; this CSV is not a complete competition submission")
    print(f"Wrote {len(result)} predictions to {output_path}", flush=True)
    return result

