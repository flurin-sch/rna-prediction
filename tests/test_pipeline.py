import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from rna_pred.common.data.cache import CacheStore, build_cache, validate_cache
from rna_pred.common.config import read_json
from rna_pred.common.data.dataset import PreparedData
from rna_pred.common.data.prepare import prepare, signal_summary
from rna_pred.baseline.workflow import predict, train


def test_preparation_labels_strand_and_reuse(synthetic_project):
    config = synthetic_project
    manifest = prepare(config)
    assert manifest["files"]["examples"]["rows"] == 15
    mapping = pd.read_parquet(Path(config["prepared_dir"]) / "mapping.parquet")
    regions = pd.read_parquet(Path(config["prepared_dir"]) / "regions.parquet")
    promoter = mapping[mapping.slot.eq(0)]
    assert len(promoter) == 15 and promoter.distance.eq(0).all()
    minus = mapping[mapping.example_id.eq("X1/train/1")].merge(regions, on="region_id")
    assert minus.loc[minus.start > 110000, "distance"].lt(0).all()
    assert prepare(config)["identity"] == manifest["identity"]
    config["regions"]["candidate_bp"] = 1024
    with pytest.raises(ValueError, match="changed"):
        prepare(config)


def test_signal_uncovered_bases_and_chromosome_errors():
    class Track:
        def chroms(self):
            return {"chr1": 5}

        def values(self, chrom, start, end):
            return [2., np.nan, -1., 4., 0.][start:end]

    assert signal_summary(Track(), "chr1", -1, 6) == pytest.approx((5 / 7, 4 / 7))
    with pytest.raises(ValueError, match="absent"):
        signal_summary(Track(), "chr2", 0, 1)


def test_cache_resume_recovery_and_invalidation(synthetic_project, fake_alphagenome):
    config = synthetic_project
    prepare(config)
    build_cache(config, device="cpu", limit_windows=1)
    assert fake_alphagenome.calls == 1
    with pytest.raises(ValueError, match="incomplete"):
        PreparedData(config)
    build_cache(config, device="cpu")
    calls = fake_alphagenome.calls
    regions = pd.read_parquet(Path(config["prepared_dir"]) / "regions.parquet")
    assert calls == regions.window_id.nunique() < 15
    # Recover committed shards even if the global index was not committed.
    (Path(config["cache_dir"]) / "index.parquet").unlink()
    build_cache(config, device="cpu")
    assert fake_alphagenome.calls == calls
    store = CacheStore(config["cache_dir"])
    assert store.get_many(regions.region_id.iloc[:2].tolist()).shape == (2, 3072)
    assert read_json(Path(config["cache_dir"]) / "manifest.json")["complete"]
    validation = validate_cache(config)
    assert validation["examples"] == 15
    assert validation["requested_regions"] == len(regions)
    assert validation["all_finite"] and validation["checksums_verified"]
    config["regions"]["window_bp"] = 524288
    with pytest.raises(ValueError, match="do not match"):
        build_cache(config, device="cpu")
    config["regions"]["window_bp"] = 262144
    config["extraction"]["port_revision"] = "different-test-revision"
    with pytest.raises(ValueError, match="Incompatible cache"):
        build_cache(config, device="cpu")


def test_training_export_and_exact_resume(synthetic_project, fake_alphagenome):
    config = synthetic_project
    prepare(config)
    build_cache(config, device="cpu")
    full_path = train(config)
    first = predict(config)
    second = predict(config)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert first.gene_name.tolist() == ["gene0", "gene1", "gene2"]
    assert first.columns.tolist() == ["gene_name", "gex_predicted"]
    assert np.isfinite(first.gex_predicted).all()
    full = torch.load(Path(config["run_dir"]) / "last.pt", weights_only=True)
    resumed_config = copy.deepcopy(config)
    resumed_config["run_dir"] = str(Path(config["run_dir"]).parent / "resumed")
    resumed_config["train"]["epochs"] = 1
    train(resumed_config)
    resumed_config["train"]["epochs"] = 2
    train(resumed_config, resume=True)
    resumed = torch.load(Path(resumed_config["run_dir"]) / "last.pt", weights_only=True)
    for key in full["model"]:
        torch.testing.assert_close(full["model"][key], resumed["model"][key], rtol=0, atol=0)
    assert full["history"] == resumed["history"]
    assert full["normalizer"] == resumed["normalizer"]
    assert full_path.is_file()


def test_corrupted_cache_is_rejected(synthetic_project, fake_alphagenome):
    config = synthetic_project
    prepare(config)
    build_cache(config, device="cpu")
    shard = next(Path(config["cache_dir"]).glob("shard-*.npy"))
    with shard.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"x")
    with pytest.raises(ValueError, match="corrupted"):
        CacheStore(config["cache_dir"])


def test_preparation_reuses_relocated_raw_data(synthetic_project):
    config = synthetic_project
    original = prepare(config)
    source = Path(config["data_dir"])
    destination = source.parent / "scratch" / "raw"
    destination.parent.mkdir()
    source.rename(destination)
    source.symlink_to(destination, target_is_directory=True)
    relocated = copy.deepcopy(config)
    relocated["data_dir"] = str(source.resolve())
    assert prepare(relocated)["identity"] == original["identity"]
    # Relocation must not weaken input-change detection.
    labels = destination / "CAGE-train" / "CAGE-train" / "X1_train_y.tsv"
    labels.write_text(labels.read_text().replace("5.0", "6.0"))
    with pytest.raises(ValueError, match="changed"):
        prepare(relocated)
