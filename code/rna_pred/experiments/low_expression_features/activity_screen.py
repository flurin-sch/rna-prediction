"""Isolate activity scarcity from the larger enhancer feature family."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from rna_pred.common.data.chromatin import feature_matrix, load_features
from rna_pred.common.config import atomic_json, atomic_parquet, digest, file_sha256, read_json
from rna_pred.experiments.low_expression_features.exploration import flat_metrics, markdown_table
from rna_pred.experiments.low_expression_features.screens import chromosome_folds
from rna_pred.common.metrics import fit_thresholds


def run(config, root, base_dir):
    root=Path(root)
    parent=read_json(root/"screens/spec.json")
    pid=read_json(Path(config["prepared_dir"])/"manifest.json")["identity"]
    base,_=load_features(base_dir,pid)
    extra,_=load_features(root/"features-extra-v1",pid)
    # The initial counts control contains raw DNase count/absence, not histone support counts.
    cols=[c for c in extra.columns if c.startswith("enhancer/") and
          (c.endswith("/count") or c.endswith("/none") or c.endswith("truncated_count") or
           any(c.endswith(f"/{s}_{stat}") for s in ("activity_supported","acetylated") for stat in ("count","none","fraction")))]
    columns=base.columns.tolist()+cols
    spec={"parent_screen_signature":parent["signature"],"columns":columns,"settings":parent["settings"],
          "source_sha256":file_sha256(__file__),"reason":"isolate histone-supported activity scarcity from continuous enhancer summaries"}
    folder=root/"screens/activity-counts-extension"
    folder.mkdir(exist_ok=True)
    signature=digest(spec)
    if (folder/"spec.json").exists() and read_json(folder/"spec.json")["signature"]!=signature:
        raise ValueError("Changed activity-scarcity screen")
    atomic_json(folder/"spec.json",{"signature":signature,**spec})
    examples=pd.read_parquet(Path(config["prepared_dir"])/"examples.parquet")
    examples=examples[examples.split.eq("train")].reset_index(drop=True)
    folds,mapping=chromosome_folds(examples)
    if mapping!=parent["folds"]:
        raise ValueError("Changed screening folds")
    table=pd.concat([base,extra.loc[base.index,cols]],axis=1)
    values=feature_matrix(table,examples)
    rows=[]
    for fold in range(3):
        train=examples[folds!=fold].reset_index(drop=True)
        val=examples[folds==fold].reset_index(drop=True)
        path=folder/f"fold-{fold}.parquet"
        if path.exists():
            p=pd.read_parquet(path)
            if p.example_id.tolist()!=val.example_id.tolist():
                raise ValueError("Activity screen rows changed")
        else:
            model=HistGradientBoostingRegressor(**parent["settings"])
            model.fit(values[folds!=fold],np.log1p(train.gex.to_numpy()))
            p=val.copy();p["prediction_log1p"]=model.predict(values[folds==fold])
            atomic_parquet(p,path)
        for q in (.25,.5):
            m=flat_metrics(val,p.prediction_log1p.to_numpy(),fit_thresholds(train,q))
            m["condition"],m["fold"],m["quantile"]="enhancer_activity_counts",fold,q
            rows.append(m)
        print(f"Activity-scarcity screen fold {fold} complete",flush=True)
    metrics=pd.concat(rows,ignore_index=True)
    metrics.to_csv(folder/"metrics.csv",index=False)
    means=metrics[metrics["quantile"].eq(.25)][["spearman","low_spearman","zero_low_auc","low_mae"]].mean()
    ranking=pd.read_csv(root/"screens/ranking.csv")
    baseline=ranking[ranking.condition.eq("baseline")].iloc[0]
    result={"condition":"enhancer_activity_counts",**means.to_dict()}
    for key in means.index:
        result["delta_"+key]=float(means[key]-baseline[key])
    result["composite"]=.5*(2*result["delta_zero_low_auc"]+result["delta_low_spearman"])
    result["promising"]=result["composite"]>0 and result["delta_spearman"]>=-.005
    combined=pd.concat([ranking,pd.DataFrame([result])],ignore_index=True).sort_values("composite",ascending=False)
    combined.to_csv(root/"screens/extended_ranking.csv",index=False)
    atomic_json(folder/"result.json",result)
    (root/"activity_screen_report.md").write_text("# Activity-scarcity isolation screen\n\n"
        "The initial raw-count control did not contain activity-supported counts. This additional condition isolates H3K27ac-supported and strict H3K27ac/H3K4me1-supported counts, absence indicators and fractions from the continuous enhancer features. It uses the original folds, settings, baseline and acceptance criterion. It was added after the initial negative screens and remains exploratory.\n\n"+
        markdown_table(combined)+"\n")
    print(result,flush=True)
