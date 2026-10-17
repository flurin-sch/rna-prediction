"""Controlled chromatin representation comparisons, run one reproducible seed per job."""
import argparse
import copy
import fcntl
import gc
import importlib.metadata
import os
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rna_pred.common.data.chromatin import feature_matrix, load_features, prepare_features
from rna_pred.common.config import atomic_json, digest, load_config, read_json
from rna_pred.common.data.dataset import PreparedData, RegionDataset, fit_normalizer
from rna_pred.common.designs import EXPERIMENTS
from rna_pred.common.metrics import regression_metrics
from rna_pred.baseline.model import ExpressionModel
from rna_pred.experiments.chromatin_representation.models import ChromatinModel
from rna_pred.common.metrics import fit_thresholds
from rna_pred.common.training import (atomic_torch_save, resolve_device, restore_rng, rng_state,
                                      seed_everything, neural_epochs, run_predictions)
from rna_pred.common.metrics import validation_metrics, diagnostics


VARIANTS = ("baseline", "boosted_trees", "chromatin_only", "dual_branch")


class ComparisonDataset(Dataset):
    def __init__(self, examples, chromatin, regions=None):
        self.examples, self.chromatin, self.regions = examples, chromatin, regions

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        item = self.regions[index] if self.regions is not None else {
            "index": index, "target": np.float32(np.log1p(self.examples.gex.iloc[index]))}
        item["chromatin"] = torch.from_numpy(self.chromatin[index])
        return item


def forward(model, batch, device, variant):
    if variant == "baseline":
        return model(batch["sequence"].to(device), batch["features"].to(device), batch["valid"].to(device))
    if variant == "chromatin_only":
        return model(batch["chromatin"].to(device))
    return model(batch["chromatin"].to(device), batch["sequence"].to(device),
                 batch["features"].to(device), batch["valid"].to(device))


def predictions(model, loader, device, variant):
    return run_predictions(model, loader, device, lambda m, b, d: forward(m, b, d, variant))


def write_predictions(folder, name, examples, scores):
    frame = examples[["example_id", "cell", "split", "chr", "gene_name", "gex"]].copy()
    frame["observed_log1p"] = np.log1p(frame.gex)
    frame["prediction_log1p"] = scores
    with np.errstate(over="raise", invalid="raise"):
        frame["prediction_gex"] = np.expm1(scores)
    frame.to_csv(Path(folder) / f"{name}_predictions.csv", index=False)
    if name == "C":
        frame[["gene_name", "prediction_gex"]].rename(columns={"prediction_gex": "gex_predicted"}).to_csv(
            Path(folder) / "submission.csv", index=False)


def train_neural(config, prepared, table, train_examples, val_examples, variant, folder, resume, signature):
    settings = config["train"]
    seed_everything(settings["seed"], settings["cpu_threads"])
    device = resolve_device(settings["device"])
    generator = torch.Generator().manual_seed(settings["seed"])
    checkpoint_path = folder / "last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True) if resume and checkpoint_path.exists() else None
    if checkpoint and checkpoint["signature"] != signature:
        raise ValueError("Cannot resume changed comparison inputs/settings")
    raw_train, raw_val = feature_matrix(table, train_examples), feature_matrix(table, val_examples)
    if checkpoint:
        normalizer, region_normalizer = checkpoint["normalizer"], checkpoint["region_normalizer"]
    else:
        mean, scale = raw_train.mean(0, dtype=np.float64), raw_train.std(0, dtype=np.float64)
        scale[scale < 1e-8] = 1.
        normalizer = {"mean": mean.tolist(), "scale": scale.tolist()}
        region_normalizer = fit_normalizer(prepared.mapping, train_examples.example_id, config["tracks"])
    def normalize(values):
        return ((values - np.asarray(normalizer["mean"])) / np.asarray(normalizer["scale"])).astype(np.float32)
    def dataset(examples, raw):
        regions = RegionDataset(prepared, examples, region_normalizer) if variant != "chromatin_only" else None
        return ComparisonDataset(examples, normalize(raw), regions)
    train_data, val_data = dataset(train_examples, raw_train), dataset(val_examples, raw_val)
    train_loader = DataLoader(train_data, batch_size=settings["batch_size"], shuffle=True, generator=generator)
    eval_train = DataLoader(train_data, batch_size=settings["batch_size"])
    eval_val = DataLoader(val_data, batch_size=settings["batch_size"])
    sequence_model = ExpressionModel(len(config["tracks"]) + 2, **config["model"]) if variant != "chromatin_only" else None
    model = (sequence_model if variant == "baseline" else ChromatinModel(
        raw_train.shape[1], sequence_model, **config["comparison"]["neural"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    start, best_score, plateau, history = 0, None, 0, []
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, best_score, plateau, history = checkpoint["epoch"], checkpoint["best_score"], checkpoint["plateau"], checkpoint["history"]
        restore_rng(checkpoint["rng"], generator)
    thresholds = fit_thresholds(train_examples, .25)
    def loss_fn(batch, score):
        return nn.functional.smooth_l1_loss(score, batch["target"].to(device))
    for epoch, _, train_scores, val_scores, began in neural_epochs(
            model, train_loader, eval_train, eval_val, optimizer, device, loss_fn,
            start=start, epochs=settings["epochs"], stopped=lambda: plateau >= settings["patience"],
            forward=lambda m, b, d: forward(m, b, d, variant)):
        value = validation_metrics(val_examples, val_scores)["mean_spearman"]
        improved = best_score is None or value > best_score
        best_score, plateau = (value, 0) if improved else (best_score, plateau + 1)
        record = {"epoch": epoch + 1, "val_mean_spearman": value,
                  "training": regression_metrics(train_examples, train_scores),
                  "validation": regression_metrics(val_examples, val_scores),
                  "train_low_expression": diagnostics(train_examples, train_scores, thresholds),
                  "val_low_expression": diagnostics(val_examples, val_scores, thresholds),
                  "seconds": time.perf_counter() - began,
                  "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
        history.append(record)
        payload = {"schema": 1, "signature": signature, "variant": variant, "config": config,
                   "epoch": epoch + 1, "best_score": best_score, "plateau": plateau, "history": history,
                   "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "normalizer": normalizer, "region_normalizer": region_normalizer,
                   "expression_thresholds": thresholds, "rng": rng_state(generator)}
        atomic_torch_save(checkpoint_path, payload)
        if improved:
            atomic_torch_save(folder / "best.pt", {k: v for k, v in payload.items() if k not in ("optimizer", "rng", "history")})
        atomic_json(folder / "history.json", history)
        print(f"{folder.name} epoch {epoch + 1}: val mean rho={value:.4f}, seconds={record['seconds']:.1f}", flush=True)
    best = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    def predict_examples(examples):
        data = dataset(examples, feature_matrix(table, examples))
        return predictions(model, DataLoader(data, batch_size=settings["batch_size"]), device, variant)
    return predict_examples, best["epoch"], thresholds


def train_trees(config, table, train_examples, val_examples, folder, resume, signature):
    settings = config["comparison"]["trees"]
    train_x, val_x = feature_matrix(table, train_examples), feature_matrix(table, val_examples)
    targets = np.log1p(train_examples.gex.to_numpy())
    last_path = folder / "last.joblib"
    checkpoint = joblib.load(last_path) if resume and last_path.exists() else None
    if checkpoint and checkpoint["signature"] != signature:
        raise ValueError("Cannot resume changed tree inputs/settings")
    tree = HistGradientBoostingRegressor(
        learning_rate=settings["learning_rate"], max_leaf_nodes=settings["max_leaf_nodes"],
        min_samples_leaf=settings["min_samples_leaf"], l2_regularization=settings["l2_regularization"],
        early_stopping=False, warm_start=True, random_state=config["train"]["seed"])
    history, best_score, plateau, start = [], None, 0, 0
    if checkpoint:
        tree, history, best_score, plateau = (checkpoint[k] for k in ("model", "history", "best_score", "plateau"))
        start = checkpoint["iteration"]
    for iteration in range(start + settings["eval_every"], settings["max_iter"] + 1, settings["eval_every"]):
        if plateau >= settings["patience"]:
            break
        began = time.perf_counter()
        tree.set_params(max_iter=iteration)
        tree.fit(train_x, targets)
        scores = tree.predict(val_x)
        value = validation_metrics(val_examples, scores)["mean_spearman"]
        improved = best_score is None or value > best_score
        best_score, plateau = (value, 0) if improved else (best_score, plateau + 1)
        history.append({"iteration": iteration, "val_mean_spearman": value, "seconds": time.perf_counter() - began})
        state = {"signature": signature, "model": tree, "history": history, "best_score": best_score,
                 "plateau": plateau, "iteration": iteration, "config": config}
        for target in ([last_path, folder / "best.joblib"] if improved else [last_path]):
            temporary = target.with_suffix(".tmp.joblib")
            joblib.dump(state, temporary)
            os.replace(temporary, target)
        atomic_json(folder / "history.json", history)
        print(f"{folder.name} trees={iteration}: val mean rho={value:.4f}", flush=True)
    best = joblib.load(folder / "best.joblib")
    return lambda examples: best["model"].predict(feature_matrix(table, examples)), best["iteration"], fit_thresholds(train_examples, .25)


def run_comparison(config, feature_dir, output_dir, seed, resume=False, prepared=None):
    config = copy.deepcopy(config)
    config["train"]["seed"] = seed
    if config["train"].get("objective", {}).get("name", "smooth_l1") != "smooth_l1":
        raise ValueError("Chromatin comparisons require the unweighted smooth_l1 objective")
    if config["train"].get("selection_metric", "val_mean_spearman") != "val_mean_spearman":
        raise ValueError("Chromatin comparisons select val_mean_spearman")
    seed_everything(seed, config["train"]["cpu_threads"])
    prepared = prepared if prepared is not None else PreparedData(config)
    if prepared.manifest["spec"]["limit_per_table"] is not None and not config["comparison"].get("allow_limited", False):
        raise ValueError("Limited data requires comparison.allow_limited")
    table, manifest = load_features(feature_dir, prepared.manifest["identity"])
    if set(table.index) != set(prepared.examples.example_id):
        raise ValueError("Chromatin sidecar example coverage differs from prepared examples")
    root = Path(output_dir) / f"seed-{seed}"
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = []
        for design in config["comparison"]["designs"]:
            spec = EXPERIMENTS[design]
            train_examples = prepared.select(spec["train_splits"], spec["train_cells"])
            val_examples = prepared.select(spec["val_splits"], spec["val_cells"])
            if set(train_examples.example_id) & set(val_examples.example_id):
                raise ValueError("Overlapping training and validation examples")
            for variant in config["comparison"]["variants"]:
                if variant not in VARIANTS:
                    raise ValueError(f"Unknown comparison variant {variant}")
                folder = root / design / variant
                folder.mkdir(parents=True, exist_ok=True)
                training_spec = {k: v for k, v in config["train"].items() if k != "device"}
                # Staged paths and Slurm IDs may change during resume; data identities may not.
                signature = digest({"schema": 1, "train": training_spec, "model": config["model"],
                                    "comparison": config["comparison"], "variant": variant, "design": spec,
                                    "prepared": prepared.manifest["identity"], "features": manifest["identity"],
                                    "cache": prepared.cache.manifest["identity"],
                                    "cache_index": prepared.cache.manifest["index_sha256"]})
                design_record = {"signature": signature, "variant": variant, "design": design, "seed": seed,
                                 "selection": spec, "feature_identity": manifest["identity"],
                                 "selection_metric": "val_mean_spearman", "config": config,
                                 "caveat": "Validation selects checkpoints; transfer designs share loci across cells",
                                 "versions": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "scikit-learn")}}
                if (folder / "design.json").exists():
                    if not resume:
                        raise ValueError(f"Existing run {folder}; use --resume")
                    if read_json(folder / "design.json")["signature"] != signature:
                        raise ValueError(f"Changed run specification: {folder}")
                atomic_json(folder / "design.json", design_record)
                if resume and (folder / "evaluation.json").exists():
                    rows.extend(read_json(folder / "evaluation.json"))
                    continue
                print(f"Training seed={seed} design={design} variant={variant}", flush=True)
                if variant == "boosted_trees":
                    predict_fn, selected, thresholds = train_trees(config, table, train_examples, val_examples, folder, resume, signature)
                else:
                    predict_fn, selected, thresholds = train_neural(config, prepared, table, train_examples, val_examples,
                                                                   variant, folder, resume, signature)
                evaluated = []
                for role, examples in [("training", train_examples), ("validation", val_examples)]:
                    scores = predict_fn(examples)
                    write_predictions(folder, role, examples, scores)
                    evaluated.append({"design": design, "variant": variant, "seed": seed, "role": role,
                                      "selected_step": selected, **regression_metrics(examples, scores),
                                      "low_expression": diagnostics(examples, scores, thresholds)})
                if design == "a_to_b":
                    test = prepared.select("test", ("X3",)).sort_values("row_order").reset_index(drop=True)
                    write_predictions(folder, "C", test, predict_fn(test))
                atomic_json(folder / "evaluation.json", evaluated)
                rows.extend(evaluated)
                # Drop model closures and optimizer references between runs.
                del predict_fn
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                pd.DataFrame(rows).to_csv(root / "metrics.csv", index=False)
        atomic_json(root / "results.json", rows)
        pd.DataFrame(rows).to_csv(root / "metrics.csv", index=False)
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "summarize", "report"])
    parser.add_argument("--config", default=Path(__file__).parent / "configs/default.yaml")
    parser.add_argument("--feature-dir", default="outputs/experiments/chromatin_representation/features")
    parser.add_argument("--output-dir", default="outputs/experiments/chromatin_representation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "report":
        from .plots import plot_all
        print(plot_all(args.output_dir), flush=True)
        return
    config = load_config(args.config)
    if args.command == "prepare":
        path = Path(args.feature_dir).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with (path.parent / f".{path.name}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            prepare_features(config, path)
    elif args.command == "summarize":
        summarize(config, args.output_dir)
        from .plots import plot_all
        print(plot_all(args.output_dir), flush=True)
    else:
        if args.stage:
            from rna_pred.common.job import stage_inputs
            config = stage_inputs(config, os.environ["TMPDIR"])
        run_comparison(config, args.feature_dir, args.output_dir, args.seed, args.resume)


def summarize(config, output_dir):
    root = Path(output_dir)
    rows, flat = [], []
    for seed in config["comparison"]["seeds"]:
        rows.extend(read_json(root / f"seed-{seed}" / "results.json"))
    atomic_json(root / "results.json", rows)
    pd.DataFrame(rows).to_csv(root / "metrics.csv", index=False)
    for row in rows:
        if row["role"] != "validation":
            continue
        for cell, sub in row["low_expression"].items():
            flat.append({"design": row["design"], "variant": row["variant"], "seed": row["seed"], "cell": cell,
                         "spearman": row["spearman_per_cell"][cell],
                         "low_positive_spearman": sub["low_positive"]["spearman"],
                         "zero_positive_auroc": sub["zero_positive_auroc"],
                         "zero_low_positive_auroc": sub["zero_low_positive_auroc"],
                         "low_positive_n": sub["low_positive"]["n"],
                         "low_positive_distinct_labels": sub["low_positive"]["n_distinct_labels"]})
    frame = pd.DataFrame(flat)
    frame.to_csv(root / "validation_by_cell.csv", index=False)
    scores = ["spearman", "low_positive_spearman", "zero_positive_auroc", "zero_low_positive_auroc"]
    frame.groupby(["design", "variant", "cell"])[scores].agg(["mean", "std", "count"]).to_csv(root / "seed_summary.csv")


if __name__ == "__main__":
    main()
