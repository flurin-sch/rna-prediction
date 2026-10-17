"""Compare saved chromatin runs without loading features or checkpoints."""
from pathlib import Path
import numpy as np
import pandas as pd

from rna_pred.common.config import read_json
from rna_pred.common.designs import LABELS
from rna_pred.common.reporting import pdf_report, save_page
import matplotlib.pyplot as plt


def plot_all(root, output=None):
    root = Path(root)
    rows = read_json(root / "results.json")
    records = []
    for row in rows:
        if row["role"] != "validation":
            continue
        for cell, metrics in row["low_expression"].items():
            records.append({"design": row["design"], "variant": row["variant"], "seed": row["seed"],
                            "cell": cell, "spearman": row["spearman_per_cell"][cell],
                            "low_positive_spearman": metrics["low_positive"]["spearman"],
                            "zero_low_positive_auroc": metrics["zero_low_positive_auroc"]})
    frame = pd.DataFrame(records)
    output = Path(output) if output else root / "summary.pdf"
    with pdf_report(output) as pdf:
        for metric in ("spearman", "low_positive_spearman", "zero_low_positive_auroc"):
            cases = frame[["design", "cell"]].drop_duplicates()
            variants = frame.variant.unique()
            fig, axes = plt.subplots(1, len(cases), figsize=(4 * len(cases), 5), squeeze=False)
            for ax, case in zip(axes.flat, cases.itertuples()):
                subset = frame[frame.design.eq(case.design) & frame.cell.eq(case.cell)]
                for x, variant in enumerate(variants):
                    values = subset.loc[subset.variant.eq(variant), metric].dropna().to_numpy(dtype=float)
                    if len(values):
                        ax.bar(x, values.mean(), alpha=.5)
                        ax.scatter(np.full(len(values), x), values, color="black", s=15)
                    else:
                        ax.text(x, 0, "undefined", rotation=90, ha="center", va="bottom")
                ax.set(title=f"{LABELS[case.design]} / {case.cell}", ylabel=metric.replace("_", " "),
                       xticks=range(len(variants)), xticklabels=variants)
                ax.tick_params(axis="x", labelrotation=45)
            save_page(pdf, fig, "Chromatin representations: " + metric.replace("_", " "))
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
        for ax, design in zip(axes.flat, LABELS):
            for folder in sorted(root.glob(f"seed-*/{design}/*")):
                history_path = folder / "history.json"
                if history_path.exists():
                    history = pd.DataFrame(read_json(history_path))
                    steps = history["epoch"] if "epoch" in history else history["iteration"]
                    ax.plot(steps, history.val_mean_spearman, label=f"{folder.name} / {folder.parents[1].name}")
            ax.set(title=LABELS[design], xlabel="Epoch / boosting round", ylabel="Mean per-cell Spearman")
        if axes.flat[0].get_legend_handles_labels()[0]:
            axes.flat[0].legend(fontsize=6)
        save_page(pdf, fig, "Validation learning curves")
    return output
