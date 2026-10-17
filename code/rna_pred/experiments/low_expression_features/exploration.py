"""Descriptive exploration; produces saved plotting data before predictive screens."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from rna_pred.common.data.chromatin import load_features
from rna_pred.common.config import atomic_json, atomic_parquet, read_json
from rna_pred.experiments.low_expression_features.features import TRACKS
from rna_pred.common.metrics import fit_thresholds, threshold_for

LEVELS = ("zero", "low", "intermediate", "high")


def markdown_table(frame, decimals=3):
    def value(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.{decimals}f}" if np.isfinite(v) else "undefined"
        return str(v)
    return "\n".join(["| " + " | ".join(map(str,frame.columns)) + " |",
                      "| " + " | ".join(["---"]*len(frame.columns)) + " |"] +
                     ["| " + " | ".join(value(v) for v in row) + " |" for row in frame.itertuples(index=False,name=None)])


def expression_groups(examples):
    result = examples.copy()
    result["group"] = "unlabelled"
    thresholds = {}
    for cell, g in examples.groupby("cell"):
        positive = g.loc[g.split.eq("train") & g.gex.gt(0), "gex"]
        if positive.empty:
            continue
        lo, hi = positive.quantile([.25, .75])
        thresholds[cell] = {"low": float(lo), "high": float(hi)}
        y = g.gex
        result.loc[g.index, "group"] = np.select(
            [y.eq(0), y.gt(0)&y.le(lo), y.gt(lo)&y.le(hi), y.gt(hi)], LEVELS, default="unlabelled")
    return result, thresholds


def flat_metrics(examples, scores, thresholds):
    """Ties remain tied, undefined rankings remain missing."""
    rows = []
    for cell, g in examples.groupby("cell", sort=True):
        s = np.asarray(scores)[examples.cell.eq(cell).to_numpy()]
        y = g.gex.to_numpy()
        limit = threshold_for(thresholds, cell)
        low, subset = (y>0)&(y<=limit), y<=limit
        def rho(a,b):
            return float(spearmanr(a,b).statistic) if len(a)>1 and np.unique(a).size>1 and np.unique(b).size>1 else np.nan
        def auc(a,b):
            return float(roc_auc_score(a,b)) if np.unique(a).size>1 else np.nan
        rows.append({"cell":cell,"n":len(g),"threshold":limit,"n_zero":int((y==0).sum()),
                     "n_low":int(low.sum()),"distinct_low":int(np.unique(y[low]).size),
                     "spearman":rho(y,s),"low_spearman":rho(y[low],s[low]),
                     "zero_low_auc":auc(y[subset]>0,s[subset]),"zero_positive_auc":auc(y>0,s),
                     "low_mae":float(np.abs(s[low]-np.log1p(y[low])).mean()) if low.any() else np.nan})
    return pd.DataFrame(rows)


def explore(config, root, base_dir, previous_root):
    root, previous_root = Path(root), Path(previous_root)
    tables = root/"tables"
    tables.mkdir(parents=True, exist_ok=True)
    examples = pd.read_parquet(Path(config["prepared_dir"])/"examples.parquet")
    examples, thresholds = expression_groups(examples)
    base, _ = load_features(base_dir, read_json(Path(config["prepared_dir"])/"manifest.json")["identity"])
    extra, _ = load_features(root/"features-extra-v1", read_json(Path(config["prepared_dir"])/"manifest.json")["identity"])
    data = examples.join(extra,on="example_id",validate="one_to_one")
    data["block"] = data.chr + ":" + (((data.TSS_start+data.TSS_end)//2)//5000000).astype(str)
    atomic_parquet(data,tables/"exploration_examples.parquet")
    atomic_json(tables/"expression_thresholds.json",thresholds)
    labelled = data[data.gex.notna()]
    mapping=pd.read_parquet(Path(config["prepared_dir"])/"mapping.parquet",columns=["example_id","slot"])
    old_counts=mapping.groupby("example_id").size()-1
    if not np.array_equal(data["enhancer/retained/count"].to_numpy(),old_counts.reindex(data.example_id).to_numpy()):
        raise ValueError("Retained candidate counts disagree with the original model inputs")
    counts = []
    for (cell,group),g in labelled.groupby(["cell","group"]):
        counts.append({"cell":cell,"group":group,"n":len(g),"distinct_labels":g.gex.nunique(),
                       "median_retained":g["enhancer/retained/count"].median(),
                       "median_all":g["enhancer/all/count"].median(),
                       "no_candidates_fraction":g["enhancer/all/none"].mean(),
                       "no_supported_activity_fraction":g["enhancer/all/activity_supported_none"].mean(),
                       "median_active":g["enhancer/all/activity_supported_count"].median(),
                       "truncated_fraction":g["enhancer/truncated_count"].gt(0).mean(),
                       "tss_outside_fraction":g["body/tss_outside"].mean(),
                       "overlapping_gene_fraction":g["body/overlapping_gene_count"].gt(0).mean()})
    summary = pd.DataFrame(counts)
    summary.to_csv(tables/"expression_group_summary.csv",index=False)
    labelled.groupby(["cell","gex"]).size().rename("n").reset_index().to_csv(tables/"label_frequencies.csv",index=False)
    profiles = []
    for (cell,group),g in labelled.groupby(["cell","group"]):
        for track in TRACKS:
            for area, n in [("body",10),("tss_-2048_2048",32)]:
                values = base.loc[g.example_id,[f"{track}/{area}/{i}/signal" for i in range(n)]].to_numpy()
                transformed = np.sign(values)*np.log1p(np.abs(values))
                for i in range(n):
                    profiles.append({"cell":cell,"group":group,"track":track,"area":area,"bin":i,
                                     "mean_log_signal":transformed[:,i].mean(),"n":len(g)})
    pd.DataFrame(profiles).to_csv(tables/"signal_profiles.csv",index=False)
    effects=[]
    columns=["enhancer/all/count","enhancer/all/activity_supported_count","enhancer/all/acetylated_count",
             "enhancer/all/promoter_marked_count","body/H3K36me3/excluded_signal","body/H3K36me3/peak_breadth",
             "body/H3K27me3/excluded_signal","body/H3K9me3/excluded_signal","promoter/gc","promoter/cpg_density"]
    for cell,g in labelled.groupby("cell"):
        for contrast, keep, positive in [("zero_vs_low",g.group.isin(["zero","low"]),"low"),
                                         ("zero_vs_high",g.group.isin(["zero","high"]),"high")]:
            for stratum,mask in [("all",np.ones(len(g),dtype=bool)),
                                  ("aligned_nonoverlapping",g["body/tss_outside"].eq(0)&g["body/overlapping_gene_count"].eq(0))]:
                sub=g[keep&mask]
                if sub.group.nunique()<2:
                    continue
                for col in columns:
                    # An association summary, not a fitted or held-out predictor.
                    value=roc_auc_score(sub.group.eq(positive),sub[col])
                    effects.append({"cell":cell,"contrast":contrast,"stratum":stratum,"feature":col,"descriptive_auc":value,"n":len(sub)})
    pd.DataFrame(effects).to_csv(tables/"descriptive_feature_associations.csv",index=False)
    pred_rows=[]
    metrics=[]
    for seed in (42,43,44):
        for design in ("a_to_b","x1_to_x2","x2_to_x1"):
            path=previous_root/f"seed-{seed}"/design/"dual_branch"/"validation_predictions.csv"
            p=pd.read_csv(path).merge(data[["example_id","group","block",*extra.columns]],on="example_id",validate="one_to_one")
            p["residual"]=p.prediction_log1p-p.observed_log1p
            p["seed"],p["design"]=seed,design
            pred_rows.append(p)
            train_cells = ("X1","X2") if design=="a_to_b" else (("X1",) if design=="x1_to_x2" else ("X2",))
            train=examples[examples.cell.isin(train_cells)&(examples.split.eq("train") if design=="a_to_b" else examples.split.isin(["train","val"]))]
            for q in (.25,.5):
                m=flat_metrics(p,p.prediction_log1p.to_numpy(),fit_thresholds(train,q))
                m["quantile"],m["seed"],m["design"]=q,seed,design
                metrics.append(m)
    predictions=pd.concat(pred_rows,ignore_index=True)
    atomic_parquet(predictions,tables/"previous_validation_predictions.parquet")
    pd.concat(metrics,ignore_index=True).to_csv(tables/"previous_validation_metrics.csv",index=False)
    residuals=[]
    for (design,seed,cell,group),g in predictions.groupby(["design","seed","cell","group"]):
        for name, mask in [("no_candidates",g["enhancer/all/none"].eq(1)),
                           ("candidates_without_support",g["enhancer/all/count"].gt(0)&g["enhancer/all/activity_supported_none"].eq(1)),
                           ("supported_candidates",g["enhancer/all/activity_supported_count"].gt(0))]:
            subset=g[mask]
            residuals.append({"design":design,"seed":seed,"cell":cell,"group":group,"candidate_class":name,
                              "n":len(subset),"median_residual":subset.residual.median(),"mae":subset.residual.abs().mean()})
    pd.DataFrame(residuals).to_csv(tables/"residuals_by_candidate_class.csv",index=False)
    # Matched gene differences remove the shared gene identity from this descriptive comparison.
    paired=labelled.pivot(index="gene_name",columns="cell",values=["gex",*columns]).dropna()
    paired_rows=[]
    dy=np.log1p(paired["gex"]["X2"])-np.log1p(paired["gex"]["X1"])
    for col in columns:
        dx=np.log1p(paired[col]["X2"])-np.log1p(paired[col]["X1"])
        paired_rows.append({"feature":col,"n_genes":len(paired),"delta_spearman":float(spearmanr(dx,dy).statistic) if np.ptp(dx)>0 else np.nan})
    pd.DataFrame(paired_rows).to_csv(tables/"paired_cell_differences.csv",index=False)
    (root/"exploration_report.md").write_text(
        "# Expression group summary\n\n" + markdown_table(summary) +
        "\n\nInterpretation and experimental rationale: see the low_expression_features README.\n")
    atomic_json(root/"exploration_complete.json",{"examples":len(labelled),"thresholds":thresholds,"report":"exploration_report.md"})
    print(f"Exploration saved to {root/'exploration_report.md'}",flush=True)
