"""Saved loss comparisons in one multipage PDF."""
from pathlib import Path
import pandas as pd
from rna_pred.common.reporting import pdf_report, save_page
import matplotlib.pyplot as plt
from rna_pred.common.config import read_json
from rna_pred.common.designs import LABELS


def plot_all(root, output=None):
    root = Path(root)
    results = pd.read_csv(root / "results.csv")
    output = Path(output) if output else root / "summary.pdf"
    with pdf_report(output) as pdf:
        for metric in ("combined_spearman", "low_positive_spearman", "zero_positive_auroc", "low_positive_mae_log1p"):
            frame = results.copy()
            frame["case"] = frame.design + " / " + frame.cell
            if metric == "combined_spearman":
                frame = frame.drop_duplicates(["design", "variant"])
                frame["case"] = frame.design
            table = frame.pivot(index="case", columns="variant", values=metric)
            fig, ax = plt.subplots(figsize=(10, 5))
            table.plot.bar(ax=ax)
            ax.set(ylabel=metric.replace("_", " "), xlabel="Evaluation design / cell")
            ax.tick_params(axis="x", labelrotation=25)
            undefined = table.isna().any(axis=1)
            if undefined.any():
                ax.text(.01, .98, "Missing bars: undefined ranking", transform=ax.transAxes, va="top")
            save_page(pdf, fig, "Low-expression losses: " + metric.replace("_", " "))
        sources = results[["variant", "source"]].drop_duplicates()
        for variant, source in sources.itertuples(index=False, name=None):
            source = Path(source)
            if not all((source / design / "history.json").exists() for design in LABELS):
                continue
            fig, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)
            for col, design in enumerate(LABELS):
                history = pd.DataFrame(read_json(source / design / "history.json"))
                for role in ("train", "val"):
                    axes[0, col].plot(history.epoch, history[role + "_loss"], label=role)
                    axes[1, col].plot(history.epoch, history[role + "_spearman"], label=role)
                axes[0, col].set(title=LABELS[design], ylabel="Unweighted Smooth L1")
                axes[1, col].set(xlabel="Epoch", ylabel="Combined Spearman")
            axes[0, 0].legend()
            save_page(pdf, fig, variant + ": learning curves")
        if all((Path(source) / design / "validation_predictions.csv").exists()
               for source in sources.source for design in LABELS):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
            for ax, design in zip(axes.flat, LABELS):
                for variant, source in sources.itertuples(index=False, name=None):
                    frame = pd.read_csv(Path(source) / design / "validation_predictions.csv")
                    ax.scatter(frame.observed_log1p, frame.prediction_log1p, s=3, alpha=.15,
                               rasterized=True, label=variant)
                ax.set(title=LABELS[design], xlabel="Observed log1p expression", ylabel="Predicted log1p expression")
            axes.flat[0].legend()
            save_page(pdf, fig, "Validation predictions by objective")
    return output
