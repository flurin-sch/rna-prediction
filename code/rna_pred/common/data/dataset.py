from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rna_pred.common.data.cache import CacheStore
from rna_pred.common.data.prepare import verify_prepared


def transform_signals(values):
    return np.sign(values) * np.log1p(np.abs(values))


def fit_normalizer(mapping, train_ids, tracks):
    selected = mapping[mapping.example_id.isin(train_ids)]
    if selected.empty:
        raise ValueError("No training regions for fitting normalization")
    values = transform_signals(selected[tracks].to_numpy(dtype=np.float64))
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite epigenomic features")
    mean, scale = values.mean(axis=0), values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return {"tracks": list(tracks), "mean": mean.tolist(), "scale": scale.tolist(),
            "transform": "signed-log1p"}


class PreparedData:
    def __init__(self, config):
        folder = Path(config["prepared_dir"])
        self.manifest = verify_prepared(folder)
        if self.manifest["spec"]["regions"] != config["regions"] or self.manifest["spec"]["tracks"] != config["tracks"]:
            raise ValueError("Prepared region/track settings do not match configuration")
        self.examples = pd.read_parquet(folder / "examples.parquet")
        self.mapping = pd.read_parquet(folder / "mapping.parquet").sort_values(["example_id", "slot"])
        self.cache = CacheStore(config["cache_dir"], config["model"]["embedding_dim"])
        if not set(self.mapping.region_id).issubset(self.cache.lookup):
            raise ValueError("Sequence cache is incomplete for this prepared dataset; run cache first")
        self.max_regions = self.manifest["max_regions"]
        if self.mapping.duplicated(["example_id", "slot"]).any():
            raise ValueError("Duplicate example/slot mappings")
        self.groups = {key: frame for key, frame in self.mapping.groupby("example_id", sort=False)}
        if set(self.examples.example_id) != set(self.groups):
            raise ValueError("Examples and region mappings do not agree")
        for key, frame in self.groups.items():
            if frame.slot.tolist() != list(range(len(frame))) or len(frame) > self.max_regions:
                raise ValueError(f"Invalid region slots for {key}")

    def select(self, split, cells):
        splits = (split,) if isinstance(split, str) else tuple(split)
        cells = tuple(cells)
        if not splits or not cells or len(set(splits)) != len(splits) or len(set(cells)) != len(cells):
            raise ValueError("Selections must contain unique, nonempty splits and cells")
        if not set(splits).issubset({"train", "val", "test"}) or not set(cells).issubset({"X1", "X2", "X3"}):
            raise ValueError("Unknown split or cell")
        selected = self.examples[self.examples.split.isin(splits) & self.examples.cell.isin(cells)].copy()
        if set(zip(selected.cell, selected.split)) != {(c, s) for c in cells for s in splits}:
            raise ValueError(f"Missing examples for {splits}/{cells}")
        if not selected.example_id.is_unique:
            raise ValueError("Duplicate example IDs in selection")
        return selected.reset_index(drop=True)


class RegionDataset(Dataset):
    def __init__(self, prepared, examples, normalizer):
        self.prepared, self.examples, self.normalizer = prepared, examples, normalizer
        self.mean = np.asarray(normalizer["mean"], dtype=np.float64)
        self.scale = np.asarray(normalizer["scale"], dtype=np.float64)
        tracks = normalizer["tracks"]
        if tracks != prepared.manifest["spec"]["tracks"]:
            raise ValueError("Normalizer track order is incompatible with prepared data")
        self.feature_dim = len(tracks) + 2

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        row = self.examples.iloc[index]
        regions = self.prepared.groups[row.example_id]
        count, maximum = len(regions), self.prepared.max_regions
        sequence = np.zeros((maximum, self.prepared.cache.expected_dim), dtype=np.float32)
        features = np.zeros((maximum, self.feature_dim), dtype=np.float32)
        valid = np.zeros(maximum, dtype=bool)
        sequence[:count] = self.prepared.cache.get_many(regions.region_id.tolist())
        signals = transform_signals(regions[self.normalizer["tracks"]].to_numpy(dtype=np.float64))
        features[:count, :-2] = (signals - self.mean) / self.scale
        features[:count, -2:] = regions[["distance", "is_promoter"]].to_numpy(dtype=np.float32)
        valid[:count] = True
        if not np.isfinite(features).all():
            raise ValueError("Nonfinite normalized features")
        return {"sequence": torch.from_numpy(sequence), "features": torch.from_numpy(features),
                "valid": torch.from_numpy(valid), "target": np.float32(np.log1p(row.gex)) if pd.notna(row.gex) else np.float32(0),
                "index": index}

