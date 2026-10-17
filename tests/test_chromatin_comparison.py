import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig
import pytest
import torch

from rna_pred.common.data.cache import build_cache
from rna_pred.common.data.chromatin import (DEFAULT_FEATURES, binned_signal, feature_matrix, gene_features,
                               load_features, prepare_features)
from rna_pred.experiments.chromatin_representation.run import diagnostics, run_comparison, summarize
from rna_pred.common.config import load_config, read_json
from rna_pred.baseline.model import ExpressionModel
from rna_pred.experiments.chromatin_representation.models import ChromatinModel
from rna_pred.common.data.prepare import prepare


def test_exact_bin_means_padding_missing_and_strand(tmp_path):
    path = tmp_path / "test.bw"
    with pyBigWig.open(str(path), "w") as b:
        b.addHeader([("chr2", 1000)])
        b.addEntries(["chr2", "chr2"], [100, 200], ends=[150, 300], values=[2., 4.])
    with pyBigWig.open(str(path)) as b:
        values, coverage = binned_signal(b, "chr2", 0, 400, 4)
        np.testing.assert_allclose(values, [0, 1, 4, 0])
        np.testing.assert_allclose(coverage, [0, .5, 1, 0])
        values, coverage = binned_signal(b, "chr2", -100, 300, 2)
        np.testing.assert_allclose(values, [0, 2.5])
        np.testing.assert_allclose(coverage, [0, .75])
        settings = {"segments": [[-200, 200, 4]], "promoter_widths": [200], "body_bins": 4}
        row = pd.Series(dict(TSS_start=190, TSS_end=210, gene_start=0, gene_end=400, strand="+", chr="chr2"))
        positive = gene_features(row, {"DNase": b}, settings)
        row.strand = "-"
        negative = gene_features(row, {"DNase": b}, settings)
        for name in ("body", "tss_-200_200"):
            assert [positive[f"DNase/{name}/{i}/signal"] for i in range(4)] == [
                negative[f"DNase/{name}/{i}/signal"] for i in range(3, -1, -1)]
        assert positive["DNase/promoter_200/0/signal"] == negative["DNase/promoter_200/0/signal"]


def comparison_config(synthetic_project):
    config = synthetic_project
    settings = load_config(Path(__file__).parents[1] / "code/rna_pred/experiments/chromatin_representation/configs/default.yaml")
    config["comparison"] = copy.deepcopy(settings["comparison"])
    config["comparison"].update(seeds=[42], allow_limited=True)
    config["comparison"]["trees"].update(max_iter=4, eval_every=2, patience=2, min_samples_leaf=2)
    config["chromatin"] = {**DEFAULT_FEATURES, "tracks": config["tracks"]}
    return config


def test_features_and_all_model_roundtrip_resume(synthetic_project, fake_alphagenome, tmp_path):
    config = comparison_config(synthetic_project)
    prepare(config)
    build_cache(config, device="cpu")
    features, output = tmp_path / "features", tmp_path / "comparison"
    manifest = prepare_features(config, features)
    assert prepare_features(config, features) == manifest
    table, _ = load_features(features, manifest["spec"]["prepared_identity"])
    assert len(table) == 15
    assert not any("gex" in col or "cell" in col for col in table.columns)
    root = run_comparison(config, features, output, 42)
    rows = read_json(root / "results.json")
    assert len(rows) == 24
    for variant in config["comparison"]["variants"]:
        folder = root / "a_to_b" / variant
        submission = pd.read_csv(folder / "submission.csv")
        assert submission.gene_name.tolist() == ["gene0", "gene1", "gene2"]
        assert np.isfinite(submission.gex_predicted).all()
        if variant != "boosted_trees":
            checkpoint = torch.load(folder / "best.pt", weights_only=True)
            train_ids = ["X1/train/0", "X1/train/1", "X1/train/2", "X2/train/0", "X2/train/1", "X2/train/2"]
            train_examples = pd.DataFrame({"example_id": train_ids})
            expected = feature_matrix(table, train_examples).mean(0, dtype=np.float64)
            np.testing.assert_allclose(checkpoint["normalizer"]["mean"], expected)
    old_predictions = pd.read_csv(root / "a_to_b" / "dual_branch" / "validation_predictions.csv")
    # Exercise checkpoint-based continuation, rather than merely skipping finished runs.
    folder = root / "a_to_b" / "dual_branch"
    (folder / "evaluation.json").unlink()
    run_comparison(config, features, output, 42, resume=True)
    pd.testing.assert_frame_equal(old_predictions, pd.read_csv(folder / "validation_predictions.csv"), check_exact=True)
    summarize(config, output)
    assert (output / "seed_summary.csv").exists()
    by_cell = pd.read_csv(output / "validation_by_cell.csv")
    assert "low_positive_distinct_labels" in by_cell
    assert len(by_cell) == 16
    changed = copy.deepcopy(config)
    changed["comparison"]["neural"]["width"] = 256
    with pytest.raises(ValueError, match="Changed run specification"):
        run_comparison(changed, features, output, 42, resume=True)
    with (features / "X1.parquet").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="Corrupted"):
        load_features(features, manifest["spec"]["prepared_identity"])


def test_direct_chromatin_and_sequence_paths_receive_gradients():
    torch.set_num_threads(1)
    sequence_model = ExpressionModel(7, embedding_dim=12)
    model = ChromatinModel(10, sequence_model, dropout=0)
    chromatin = torch.randn(2, 10, requires_grad=True)
    sequence = torch.randn(2, 3, 12, requires_grad=True)
    features = torch.randn(2, 3, 7)
    valid = torch.tensor([[True, True, False], [True, False, False]])
    model(chromatin, sequence, features, valid).sum().backward()
    assert chromatin.grad.abs().sum() > 0
    assert sequence.grad[valid].abs().sum() > 0
    assert sequence.grad[~valid].abs().sum() == 0
    assert model.direct.weight.grad.abs().sum() > 0


def test_low_positive_diagnostics_expose_constant_labels():
    examples = pd.DataFrame({"cell": ["X2"] * 4, "gex": [0., .5, .5, 100.]})
    metrics = diagnostics(examples, np.arange(4.), {"pooled": 1., "per_cell": {}})["X2"]
    assert metrics["low_positive"]["spearman"] is None
    assert metrics["low_positive"]["n_distinct_labels"] == 1
    assert metrics["zero_low_positive_auroc"] == 1
