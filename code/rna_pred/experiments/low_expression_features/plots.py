"""Descriptive groups, feature screens, and neural deltas in one Matplotlib PDF."""
from pathlib import Path
import numpy as np
import pandas as pd

from rna_pred.common.reporting import pdf_report, save_page
import matplotlib.pyplot as plt


def plot_all(root, output=None):
    root = Path(root)
    output = Path(output) if output else root / "summary.pdf"
    groups_path = root / "tables/expression_group_summary.csv"
    rank_path = root / "screens/extended_ranking.csv"
    if not rank_path.exists():
        rank_path = root / "screens/ranking.csv"
    delta_path = root / "tables/neural_deltas.csv"
    if not any(p.exists() for p in (groups_path, rank_path, delta_path)):
        raise ValueError("No saved exploration, screen, or neural results to report")
    with pdf_report(output) as pdf:
        if groups_path.exists():
            groups = pd.read_csv(groups_path)
            fig, axes = plt.subplots(1, groups.cell.nunique(), figsize=(10, 4), squeeze=False)
            for ax, (cell, frame) in zip(axes.flat, groups.groupby("cell")):
                ax.bar(frame["group"], frame.n)
                ax.set(title=cell, ylabel="Genes", xlabel="Expression group")
            save_page(pdf, fig, "Expression groups (training-derived thresholds)")
        examples_path = root / "tables/exploration_examples.parquet"
        if examples_path.exists():
            examples = pd.read_parquet(examples_path)
            columns = ["enhancer/all/count", "enhancer/all/activity_supported_count", "body/H3K36me3/excluded_signal"]
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for ax, column in zip(axes, columns):
                frame = examples[examples.gex.notna()].groupby(["cell", "group"])[column].median().unstack("cell")
                frame.plot.bar(ax=ax)
                ax.set(title=column, ylabel="Median feature", xlabel="Expression group")
            save_page(pdf, fig, "Candidate scarcity and gene-body signal")
        profiles_path = root / "tables/signal_profiles.csv"
        if profiles_path.exists():
            profiles = pd.read_csv(profiles_path)
            cells = sorted(profiles.cell.unique())
            views = [("H3K36me3", "body"), ("H3K27me3", "body"), ("H3K4me3", "tss_-2048_2048")]
            fig, axes = plt.subplots(len(cells), len(views), figsize=(12, 4 * len(cells)), squeeze=False)
            for row, cell in enumerate(cells):
                for col, (track, area) in enumerate(views):
                    ax = axes[row, col]
                    subset = profiles[profiles.cell.eq(cell) & profiles.track.eq(track) & profiles.area.eq(area)]
                    for group, frame in subset.groupby("group"):
                        frame = frame.sort_values("bin")
                        x = (frame.bin.to_numpy() + .5) / 10 if area == "body" else (-2048 + (frame.bin.to_numpy() + .5) * 128) / 1000
                        ax.plot(x, frame.mean_log_signal, label=group)
                    ax.set(title=f"{cell}: {track}", ylabel="Mean log1p signal",
                           xlabel="Gene-body fraction" if area == "body" else "Distance to TSS (kb)")
            axes[0, 0].legend()
            save_page(pdf, fig, "Chromatin profiles by training-derived expression group")
        previous_path = root / "tables/previous_validation_predictions.parquet"
        if previous_path.exists():
            previous = pd.read_parquet(previous_path)
            previous = previous[previous.design.eq("a_to_b") & previous.seed.eq(42)]
            groups = ["zero", "low", "intermediate", "high"]
            cells = sorted(previous.cell.unique())
            fig, axes = plt.subplots(1, len(cells), figsize=(10, 5), squeeze=False)
            for ax, cell in zip(axes.flat, cells):
                subset = previous[previous.cell.eq(cell)]
                ax.boxplot([subset.loc[subset.group.eq(group), "residual"].to_numpy() for group in groups],
                           tick_labels=groups, whis=(0, 100), showfliers=False)
                ax.axhline(0, color="black", linewidth=.8)
                ax.set(title=cell, ylabel="Predicted minus observed log1p expression", xlabel="Expression group")
            save_page(pdf, fig, "Prior dual-branch validation errors (A→B, seed 42)")
        if rank_path.exists():
            ranking = pd.read_csv(rank_path)
            ranking = ranking[~ranking.condition.eq("baseline")].sort_values("composite")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(ranking.condition, ranking.composite)
            ax.axvline(0, color="black", linewidth=.8)
            ax.set(xlabel="Low-expression composite change", ylabel="Feature condition")
            save_page(pdf, fig, "Source-training screens")
        if delta_path.exists():
            deltas = pd.read_csv(delta_path)
            deltas = deltas[~deltas.arm.eq("baseline")]
            interval_path = root / "tables/paired_block_intervals.csv"
            intervals = pd.read_csv(interval_path) if interval_path.exists() else pd.DataFrame()
            if not intervals.empty:
                intervals = intervals[intervals["quantile"].eq(.25)]
            fig, axes = plt.subplots(1, 3, figsize=(13, 5))
            for ax, metric in zip(axes, ("spearman", "low_spearman", "zero_low_auc")):
                for x, row in enumerate(deltas.itertuples()):
                    value = getattr(row, "delta_" + metric)
                    if np.isfinite(value):
                        ax.plot(x, value, "o")
                        if not intervals.empty:
                            ci = intervals[intervals.arm.eq(row.arm) & intervals.design.eq(row.design)
                                           & intervals.cell.eq(row.cell) & intervals.metric.eq(metric)]
                            if len(ci) and np.isfinite(ci[["ci_low", "ci_high"]].to_numpy()).all():
                                ax.vlines(x, ci.ci_low.iloc[0], ci.ci_high.iloc[0], color="black", linewidth=1)
                    else:
                        ax.text(x, 0, "undefined", rotation=90, ha="center", va="bottom")
                labels = [f"{r.design}\n{r.cell}\n{r.arm}" for r in deltas.itertuples()]
                ax.axhline(0, color="black", linewidth=.8)
                ax.set(title=metric.replace("_", " "), ylabel="Change from matched dual-branch control",
                       xticks=range(len(deltas)), xticklabels=labels, xlim=(-.5, len(deltas) - .5))
                ax.tick_params(axis="x", labelrotation=45, labelsize=7)
            save_page(pdf, fig, "Neural diagnostic: paired changes and 95% block intervals")
    return output
