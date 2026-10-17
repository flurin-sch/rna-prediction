"""Fixed, chromosome-grouped CPU screens; neural selection never sees B labels."""
from itertools import combinations
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from rna_pred.common.data.chromatin import feature_matrix, load_features
from rna_pred.common.config import atomic_json, atomic_parquet, digest, file_sha256, read_json
from rna_pred.experiments.low_expression_features.exploration import flat_metrics, markdown_table
from rna_pred.experiments.low_expression_features.features import FAMILIES
from rna_pred.common.metrics import fit_thresholds


def screen_conditions(base_columns, extra_columns):
    conditions={"baseline":list(base_columns)}
    for n in (1,2,3):
        for families in combinations(FAMILIES,n):
            conditions["+".join(families)] = list(base_columns)+[c for c in extra_columns if c.split("/")[0] in families]
    conditions["enhancer_counts"] = list(base_columns)+[c for c in extra_columns if c.startswith("enhancer/") and
         (c.endswith("/count") or c.endswith("/none") or c.endswith("truncated_count"))]
    conditions["enhancer_retained"] = list(base_columns)+[c for c in extra_columns if c.startswith("enhancer/retained/")]
    conditions["remove_existing_body"] = [c for c in base_columns if "/body/" not in c]
    return conditions


def chromosome_folds(examples):
    chroms=sorted(examples.chr.unique(),key=lambda c:int(c.removeprefix("chr")))
    if len(chroms)<3:
        raise ValueError("Screen needs at least three training chromosomes")
    mapping={c:i%3 for i,c in enumerate(chroms)}
    return examples.chr.map(mapping).to_numpy(),mapping


def run_screens(config, root, base_dir):
    root=Path(root)
    if not (root/"exploration_complete.json").exists():
        raise ValueError("Finish the descriptive analysis before predictive screens")
    prepared_identity=read_json(Path(config["prepared_dir"])/"manifest.json")["identity"]
    base,bm=load_features(base_dir,prepared_identity)
    extra,em=load_features(root/"features-extra-v1",prepared_identity)
    examples=pd.read_parquet(Path(config["prepared_dir"])/"examples.parquet")
    examples=examples[examples.split.eq("train")].reset_index(drop=True)
    folds,mapping=chromosome_folds(examples)
    table=pd.concat([base,extra.loc[base.index]],axis=1)
    x=feature_matrix(table,examples)
    column_indices={c:i for i,c in enumerate(table.columns)}
    conditions=screen_conditions(base.columns,extra.columns)
    settings={"max_iter":200,"learning_rate":.05,"max_leaf_nodes":15,"min_samples_leaf":30,
              "l2_regularization":1.,"early_stopping":False,"random_state":42}
    spec={"schema":1,"base_identity":bm["identity"],"extra_identity":em["identity"],
          "conditions":conditions,"settings":settings,"folds":mapping,"source_sha256":file_sha256(__file__),
          "ranking":"0.5 * (2 * delta zero-low AUROC + delta low-positive Spearman); equal fold/cell weights",
          "promising":"composite > 0 AND delta overall mean-per-cell Spearman >= -0.005"}
    signature=digest(spec)
    folder=root/"screens"
    folder.mkdir(exist_ok=True)
    if (folder/"spec.json").exists() and read_json(folder/"spec.json")["signature"]!=signature:
        raise ValueError("Changed screen specification; use a new investigation root")
    atomic_json(folder/"spec.json",{"signature":signature,**spec})
    rows=[]
    for name,columns in conditions.items():
        subset=x[:,[column_indices[c] for c in columns]]
        for fold in range(3):
            path=folder/f"{name.replace('+','_')}-fold-{fold}.parquet"
            val=examples.loc[folds==fold].reset_index(drop=True)
            train=examples.loc[folds!=fold].reset_index(drop=True)
            if path.exists():
                p=pd.read_parquet(path)
                if p.example_id.tolist()!=val.example_id.tolist():
                    raise ValueError("Changed screen prediction row order")
            else:
                began=time.monotonic()
                model=HistGradientBoostingRegressor(**settings)
                model.fit(subset[folds!=fold],np.log1p(train.gex.to_numpy()))
                p=val.copy()
                p["prediction_log1p"]=model.predict(subset[folds==fold])
                atomic_parquet(p,path)
                print(f"Screen {name} fold {fold}: {time.monotonic()-began:.1f}s",flush=True)
            for quantile in (.25,.5):
                metrics=flat_metrics(val,p.prediction_log1p.to_numpy(),fit_thresholds(train,quantile))
                metrics["condition"],metrics["fold"],metrics["quantile"]=name,fold,quantile
                rows.append(metrics)
            pd.concat(rows,ignore_index=True).to_csv(folder/"metrics.csv",index=False)
    metrics=pd.concat(rows,ignore_index=True)
    primary=metrics[metrics["quantile"].eq(.25)]
    means=primary.groupby("condition")[["spearman","low_spearman","zero_low_auc","low_mae"]].mean()
    baseline=means.loc["baseline"]
    for column in ("spearman","low_spearman","zero_low_auc","low_mae"):
        means["delta_"+column]=means[column]-baseline[column]
    means["composite"]=.5*(2*means.delta_zero_low_auc+means.delta_low_spearman)
    means["promising"]=(means.composite>0)&(means.delta_spearman>=-.005)
    means=means.sort_values("composite",ascending=False).reset_index()
    means.to_csv(folder/"ranking.csv",index=False)
    eligible=means[means.promising & ~means.condition.isin(["baseline","remove_existing_body"])]
    selected=eligible.head(2).condition.tolist()
    # Exact columns, fold assignments, identities and selection are frozen together.
    selection={"screen_signature":signature,"selected":selected,
               "columns":{name:conditions[name] for name in ["baseline",*selected]},
               "seeds":[42,43],"designs":["a_to_b","x1_to_x2","x2_to_x1"],
               "fits":6*(1+len(selected)) if selected else 0}
    atomic_json(root/"selection.json",selection)
    report=["# Predictive feature screens","","Three chromosome-grouped folds within A only. Both cells from a chromosome stay together; no B labels select feature bundles.","",
            "Fixed histogram gradient boosting: 200 iterations, 15 leaves, learning rate 0.05, minimum leaf size 30, L2=1, squared error on log1p expression. No early stopping or fitted per-cell rescaling.","",
            markdown_table(means),"",
            "Composite = 0.5 × (2 × AUROC improvement + low-positive Spearman improvement), averaged equally across folds and cells. Promising requires composite > 0 and overall Spearman change ≥ −0.005. These are source cross-validation screens, not independent tests.","",
            f"Selected bundles: {', '.join(selected) if selected else 'none'}. Planned neural fits: {selection['fits']}.","",
            "The full factorial includes family-removal comparisons: compare all three with each pair, and each pair with its single-family components. `enhancer_counts` isolates explicit counts/absence; `enhancer_retained` excludes unretained regions. `remove_existing_body` removes the original body bins; it is diagnostic and cannot be selected as an added-feature bundle.","",
            "Feature selection uses A labels from both X1 and X2. Consequently, the subsequent cell-transfer fits hold a cell out of neural optimization, but not out of feature selection. They are exploratory transfer comparisons, not fully cell-held-out tests.","",
            "Undefined subgroup correlations remain missing. Q50 sensitivity results and cell/fold metrics are in `screens/metrics.csv`. Fold variation and feature selection make these exploratory rankings. No screen changes neural loss or checkpoint selection."]
    (root/"screen_report.md").write_text("\n".join(report)+"\n")
    print("SCREEN SELECTION",selection["selected"],"FITS",selection["fits"],flush=True)
    return selection
