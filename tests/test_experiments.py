import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import spearmanr

from rna_pred.common.data.cache import build_cache
from rna_pred.common.config import read_json
from rna_pred.common.data.dataset import PreparedData, fit_normalizer
from rna_pred.baseline.evaluation import EXPERIMENTS, evaluate, run_experiments
from rna_pred.common.data.prepare import prepare
from rna_pred.baseline.workflow import prediction_loss


def test_experiment_selections_predictions_and_resume(synthetic_project, fake_alphagenome):
    config = synthetic_project
    prepare(config)
    build_cache(config, device="cpu")
    root = run_experiments(config, Path(config["run_dir"]) / "experiments", plots=True)
    metrics = pd.read_csv(root / "metrics.csv")
    assert len(metrics) == 6
    assert metrics.n.eq(6).all()
    prepared = PreparedData(config)
    for name, spec in EXPERIMENTS.items():
        checkpoint = torch.load(root / name / "best.pt", weights_only=True)
        assert checkpoint["schema"] == 2 and checkpoint["selection_metric"] == "val_loss"
        for key, value in spec.items():
            assert checkpoint[key] == list(value)
        train_examples = prepared.select(spec["train_splits"], spec["train_cells"])
        val_examples = prepared.select(spec["val_splits"], spec["val_cells"])
        assert not set(train_examples.example_id) & set(val_examples.example_id)
        expected = fit_normalizer(prepared.mapping, train_examples.example_id, config["tracks"])
        assert checkpoint["normalizer"] == expected
        history = read_json(root / name / "history.json")
        assert checkpoint["epoch"] + 1 == min(history, key=lambda r: r["val_loss"])["epoch"]
        assert read_json(root / name / "run.json")["termination_reason"] == "epoch_cap"
        assert len((root / name / "performance.jsonl").read_text().splitlines()) == 2
        for role, examples in (("training", train_examples), ("validation", val_examples)):
            frame = pd.read_csv(root / name / f"{role}_predictions.csv")
            assert frame.example_id.tolist() == examples.example_id.tolist()
            assert np.isfinite(frame.prediction_log1p).all()
            row = metrics[metrics.experiment.eq(name) & metrics.role.eq(role)].iloc[0]
            assert row.spearman == pytest.approx(spearmanr(frame.gex, frame.prediction_log1p).statistic)
            assert row.loss == pytest.approx(prediction_loss(frame, frame.prediction_log1p.to_numpy()))
    c = pd.read_csv(root / "a_to_b" / "C_predictions.csv")
    assert c.columns.tolist() == ["gene_name", "gex_predicted"]
    assert c.gene_name.tolist() == ["gene0", "gene1", "gene2"]
    assert np.isfinite(c.gex_predicted).all()
    assert len(list(root.rglob("C_predictions.csv"))) == 1
    assert read_json(root / "experiments.json")["test_predictions_created"] is True
    assert (root / "summary.pdf").read_bytes().startswith(b"%PDF-")
    assert not list(root.rglob("*_legend.md"))
    resumed = run_experiments(config, root, resume=True, plots=False)
    pd.testing.assert_frame_equal(metrics, pd.read_csv(resumed / "metrics.csv"), check_exact=True)
    with pytest.raises(ValueError, match="labelled"):
        evaluate(config, root / "a_to_b" / "best.pt", prepared, "test", "X3", root / "forbidden.csv")
    with pytest.raises(ValueError, match="exists"):
        run_experiments(config, root, plots=False)
    # Explicit replacement archives the entire previous run and leaves fresh three-design outputs.
    run_experiments(config, root, overwrite=True, plots=False)
    archives = list(root.parent.glob("experiments.previous-*"))
    assert len(archives) == 1 and (archives[0] / "a_to_b" / "best.pt").is_file()


def test_limited_experiments_rejected(synthetic_project, fake_alphagenome):
    config = synthetic_project
    prepare(config, limit=2)
    build_cache(config, device="cpu")
    with pytest.raises(ValueError, match="Limited"):
        run_experiments(config, plots=False)
