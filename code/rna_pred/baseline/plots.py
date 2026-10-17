"""Baseline diagnostics from saved evaluation outputs, without model loading."""
from pathlib import Path
import numpy as np
import pandas as pd

from rna_pred.common.config import read_json
from rna_pred.common.designs import EXPERIMENTS, LABELS
from rna_pred.common.reporting import pdf_report, save_page
import matplotlib.pyplot as plt


def plot_all(root, output=None):
    root = Path(root)
    output = Path(output) if output else root / "summary.pdf"
    metrics = pd.read_csv(root / "metrics.csv")
    if set(metrics.experiment) != set(EXPERIMENTS) or len(metrics) != 6:
        raise ValueError("Report requires training/validation results for all three designs")
    provenance = read_json(root / "experiments.json").get("data_provenance", {})
    note = provenance.get("description", "Supplied CAGE validation data")
    with pdf_report(output) as pdf:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(EXPERIMENTS))
        for offset, role in [(-.18, "training"), (.18, "validation")]:
            frame = metrics[metrics.role.eq(role)].set_index("experiment").loc[list(EXPERIMENTS)]
            ax.bar(x + offset, frame.spearman, width=.36, label=role)
        ax.set(xticks=x, xticklabels=list(LABELS.values()), ylabel="Combined Spearman")
        ax.legend()
        fig.text(.5, .01, note, ha="center", fontsize=8)
        save_page(pdf, fig, "Baseline evaluation")
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
        for ax, name in zip(axes.flat, EXPERIMENTS):
            for role in ("training", "validation"):
                frame = pd.read_csv(root / name / f"{role}_predictions.csv")
                ax.scatter(frame.observed_log1p, frame.prediction_log1p, s=3, alpha=.2,
                           rasterized=True, label=role)
            ax.set(title=LABELS[name], xlabel="Observed log1p expression", ylabel="Predicted log1p expression")
        axes.flat[0].legend()
        save_page(pdf, fig, "Predictions by evaluation design")
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)
        for col, name in enumerate(EXPERIMENTS):
            history = pd.DataFrame(read_json(root / name / "history.json"))
            for role in ("train", "val"):
                axes[0, col].plot(history.epoch, history[role + "_loss"], label=role)
                axes[1, col].plot(history.epoch, history[role + "_spearman"], label=role)
            axes[0, col].set(title=LABELS[name], ylabel="Smooth L1")
            axes[1, col].set(xlabel="Epoch", ylabel="Combined Spearman")
        axes[0, 0].legend()
        save_page(pdf, fig, "Learning curves")
    return output
