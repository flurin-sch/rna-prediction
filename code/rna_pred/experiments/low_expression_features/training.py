"""Freeze screened feature bundles and run controlled dual-branch comparisons."""
import copy
from pathlib import Path

import pandas as pd

from rna_pred.common.data.chromatin import load_features
from rna_pred.experiments.chromatin_representation.run import run_comparison
from rna_pred.common.config import atomic_json, atomic_parquet, digest, file_sha256, read_json


def materialize_bundles(config, root, base_dir):
    root=Path(root)
    selection=read_json(root/"selection.json")
    pid=read_json(Path(config["prepared_dir"])/"manifest.json")["identity"]
    base,bm=load_features(base_dir,pid)
    extra,em=load_features(root/"features-extra-v1",pid)
    all_features=pd.concat([base,extra.loc[base.index]],axis=1)
    examples=pd.read_parquet(Path(config["prepared_dir"])/"examples.parquet")
    for name,columns in selection["columns"].items():
        folder=root/"feature-bundles"/name
        spec={"schema":1,"prepared_identity":pid,"base_identity":bm["identity"],"extra_identity":em["identity"],
              "columns":columns,"screen_signature":selection["screen_signature"]}
        identity=digest(spec)
        if (folder/"manifest.json").exists():
            if read_json(folder/"manifest.json")["identity"]!=identity:
                raise ValueError("Changed frozen feature bundle")
            load_features(folder,pid)
            continue
        folder.mkdir(parents=True,exist_ok=True)
        files={}
        for cell,g in examples.groupby("cell"):
            path=folder/f"{cell}.parquet"
            atomic_parquet(all_features.loc[g.example_id,columns].reset_index(),path)
            files[cell]=file_sha256(path)
        atomic_json(folder/"manifest.json",{"identity":identity,"spec":spec,"columns":columns,"files":files})


def train_selected(config, root, seed, stage=False):
    root=Path(root)
    selection=read_json(root/"selection.json")
    if not selection["selected"]:
        raise ValueError("No promising screen: review the screen report before choosing further training")
    if stage:
        import os
        from rna_pred.common.job import stage_inputs
        config=stage_inputs(config,os.environ["TMPDIR"])
    from rna_pred.common.data.dataset import PreparedData
    prepared=PreparedData(config)
    for arm in ("baseline",*selection["selected"]):
        cfg=copy.deepcopy(config)
        cfg["comparison"]["variants"]=["dual_branch"]
        cfg["comparison"]["seeds"]=selection["seeds"]
        cfg["comparison"]["designs"]=selection["designs"]
        cfg["run_dir"]=str(root/"neural"/arm)
        run_comparison(cfg,root/"feature-bundles"/arm,root/"neural"/arm,seed,resume=True,prepared=prepared)
