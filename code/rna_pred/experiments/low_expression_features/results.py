"""Paired genomic-block uncertainty and the final investigation report."""
from pathlib import Path

import numpy as np
import pandas as pd

from rna_pred.common.config import atomic_json, atomic_parquet, read_json
from rna_pred.experiments.low_expression_features.exploration import flat_metrics, markdown_table
from rna_pred.common.metrics import fit_thresholds


def bootstrap_indices(blocks, generator):
    """Resample genomic blocks; every example/cell in a block moves together."""
    unique=np.unique(blocks)
    groups={b:np.flatnonzero(blocks==b) for b in unique}
    sampled=generator.choice(unique,size=len(unique),replace=True)
    return np.concatenate([groups[b] for b in sampled])


def paired_intervals(examples, baseline, candidate, thresholds, repeats=300):
    """Scores have shape (seeds, examples); CI covers loci, conditional on these seeds."""
    rng=np.random.default_rng(20261016)
    anchors=(examples.TSS_start+examples.TSS_end)//2
    # Assign matched gene/cell observations one common locus even if supplied TSSs differ.
    if examples.groupby("gene_name").chr.nunique().max()>1:
        raise ValueError("Matched gene names disagree on chromosome")
    anchors=anchors.groupby(examples.gene_name).transform("median").astype(np.int64)
    blocks=(examples.chr+":"+(anchors//5000000).astype(str)).to_numpy()
    records=[]
    for iteration in range(repeats):
        idx=bootstrap_indices(blocks,rng)
        frame=examples.iloc[idx].reset_index(drop=True)
        deltas=[]
        for a,b in zip(baseline,candidate):
            ma=flat_metrics(frame,a[idx],thresholds).set_index("cell")
            mb=flat_metrics(frame,b[idx],thresholds).set_index("cell")
            deltas.append(mb[["spearman","low_spearman","zero_low_auc"]]-ma[["spearman","low_spearman","zero_low_auc"]])
        mean=pd.concat(deltas).groupby(level=0).mean()
        for cell,row in mean.iterrows():
            records.append({"repeat":iteration,"cell":cell,**row.to_dict()})
    draws=pd.DataFrame(records)
    rows=[]
    for cell,g in draws.groupby("cell"):
        for metric in ("spearman","low_spearman","zero_low_auc"):
            values=g[metric].dropna()
            rows.append({"cell":cell,"metric":metric,"ci_low":values.quantile(.025) if len(values) else np.nan,
                         "ci_high":values.quantile(.975) if len(values) else np.nan,
                         "bootstrap_defined":len(values),"bootstrap_repeats":repeats,"n_blocks":len(np.unique(blocks))})
    return pd.DataFrame(rows),draws


def summarize(config, root):
    root=Path(root)
    selection=read_json(root/"selection.json")
    if not selection["selected"]:
        (root/"results_report.md").write_text(
            "# Negative feature screens\n\nNo bundle passed the recorded screening criterion. "
            "No neural fits are required. See screens/ranking.csv and screen_report.md.\n")
        atomic_json(root/"complete.json", {"fits": 0, "selected": [], "stage": "negative_screens"})
        return
    examples=pd.read_parquet(Path(config["prepared_dir"])/"examples.parquet")
    arms=["baseline",*selection["selected"]]
    records=[]
    prediction_records=[]
    uncertainty=[]
    for design in selection["designs"]:
        source=("X1","X2") if design=="a_to_b" else (("X1",) if design=="x1_to_x2" else ("X2",))
        training=examples[examples.cell.isin(source)&(examples.split.eq("train") if design=="a_to_b" else examples.split.isin(["train","val"]))]
        frames={}
        for arm in arms:
            for seed in selection["seeds"]:
                folder=root/"neural"/arm/f"seed-{seed}"/design/"dual_branch"
                if not (folder/"evaluation.json").exists():
                    raise ValueError(f"Incomplete neural fit: {folder}")
                p=pd.read_csv(folder/"validation_predictions.csv")
                p=p.merge(examples[["example_id","TSS_start","TSS_end"]],on="example_id",validate="one_to_one")
                frames[arm,seed]=p
                tagged=p.copy()
                tagged["arm"],tagged["seed"],tagged["design"]=arm,seed,design
                prediction_records.append(tagged)
                for quantile in (.25,.5):
                    m=flat_metrics(p,p.prediction_log1p.to_numpy(),fit_thresholds(training,quantile))
                    m["arm"],m["seed"],m["design"],m["quantile"]=arm,seed,design,quantile
                    records.append(m)
        reference=frames["baseline",selection["seeds"][0]]
        for p in frames.values():
            if p.example_id.tolist()!=reference.example_id.tolist():
                raise ValueError("Paired uncertainty requires identical prediction row order")
        for quantile in (.25,.5):
            threshold=fit_thresholds(training,quantile)
            for arm in selection["selected"]:
                a=np.array([frames["baseline",s].prediction_log1p.to_numpy() for s in selection["seeds"]])
                b=np.array([frames[arm,s].prediction_log1p.to_numpy() for s in selection["seeds"]])
                ci,draws=paired_intervals(reference,a,b,threshold)
                ci["design"],ci["arm"],ci["quantile"]=design,arm,quantile
                uncertainty.append(ci)
                atomic_parquet(draws,root/"tables"/f"bootstrap-{design}-{arm.replace('+','_')}-q{quantile}.parquet")
                print(f"Uncertainty {design}/{arm}/Q{quantile}",flush=True)
    metrics=pd.concat(records,ignore_index=True)
    metrics.to_csv(root/"tables/neural_metrics.csv",index=False)
    atomic_parquet(pd.concat(prediction_records,ignore_index=True),root/"tables/neural_predictions.parquet")
    ci=pd.concat(uncertainty,ignore_index=True)
    ci.to_csv(root/"tables/paired_block_intervals.csv",index=False)
    primary=metrics[metrics["quantile"].eq(.25)]
    measures=["spearman","low_spearman","zero_low_auc","zero_positive_auc","low_mae"]
    mean=primary.groupby(["design","arm","cell"])[measures].mean().reset_index()
    mean.to_csv(root/"tables/neural_summary.csv",index=False)
    paired=primary.merge(primary[primary.arm.eq("baseline")],on=["design","seed","cell"],suffixes=("","_baseline"))
    for measure in measures:
        paired["delta_"+measure]=paired[measure]-paired[measure+"_baseline"]
    delta_columns=["delta_"+m for m in measures]
    deltas=paired.groupby(["design","arm","cell"])[delta_columns].mean().reset_index()
    deltas.to_csv(root/"tables/neural_deltas.csv",index=False)
    (root/"results_report.md").write_text(
        "# Neural validation summary\n\n" + markdown_table(mean) +
        "\n\n## Paired changes\n\n" + markdown_table(deltas) +
        "\n\nInterpretation and limitations: see the low_expression_features README.\n")
    atomic_json(root/"complete.json",{"fits":selection["fits"],"selected":selection["selected"],"seeds":selection["seeds"],"bootstrap_repeats":300})
