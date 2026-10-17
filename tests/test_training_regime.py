import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
import torch

from rna_pred.baseline import workflow as training
from rna_pred.common.data.cache import build_cache
from rna_pred.common.data.dataset import PreparedData
from rna_pred.common.data.prepare import prepare


def test_combined_spearman_and_loss():
    examples = pd.DataFrame({"cell": ["X1", "X1", "X2", "X2"], "gex": [0., 1., 10., 20.]})
    predictions = np.array([10., 20., 0., 1.])
    metrics = training.validation_metrics(examples, predictions)
    assert metrics["mean_spearman"] == pytest.approx(1)
    assert metrics["combined_spearman"] == pytest.approx(spearmanr(examples.gex, predictions).statistic)
    assert metrics["combined_spearman"] < 0
    target = torch.tensor(np.log1p(examples.gex.to_numpy()))
    expected = torch.nn.functional.smooth_l1_loss(torch.tensor(predictions), target).item()
    assert training.prediction_loss(examples, predictions) == pytest.approx(expected)
    assert training.spearman_score([0, 0], [1, 2]) == 0
    with pytest.raises(ValueError, match="nonfinite"):
        training.prediction_loss(examples, np.full(4, np.nan))


def test_stopping_requires_both_and_resets_validation_rise():
    settings = {"patience": 2}
    state = training.initial_stopping_state()
    for t, v in [(1., 1.), (.99, 1.1), (.98, 1.1)]:
        state, improved, stopped = training.update_stopping(state, t, v, settings)
        assert not stopped
    assert state["val_rise_epochs"] == 2 and state["train_plateau_epochs"] == 0
    state, _, stopped = training.update_stopping(state, .98, 1.1, settings)
    assert not stopped
    state, _, stopped = training.update_stopping(state, .98, 1.1, settings)
    assert stopped
    # A plateau in validation, rather than deterioration, must not trigger stopping.
    for _ in range(3):
        state, _, stopped = training.update_stopping(state, .98, 1., settings)
        assert not stopped and state["val_rise_epochs"] == 0
    state, improved, _ = training.update_stopping(state, .98, .9999, settings)
    assert improved and state["best_val_loss"] == .9999


def test_stopping_thresholds_and_near_zero():
    settings = {"patience": 1, "train_min_relative_improvement": .125, "val_min_relative_rise": .125}
    state, _, _ = training.update_stopping(training.initial_stopping_state(), 1., 1., settings)
    state, improved, stopped = training.update_stopping(state, .875, 1.125, settings)
    assert state["train_plateau_epochs"] == 1 and state["val_rise_epochs"] == 0
    assert not improved and not stopped
    state, _, _ = training.update_stopping(state, .874, 1.126, settings)
    assert state["train_plateau_epochs"] == 0
    state, _, _ = training.update_stopping(training.initial_stopping_state(), 0., 0., settings)
    state, _, stopped = training.update_stopping(state, 0., 1e-15, settings)
    assert not stopped
    with pytest.raises(ValueError, match="finite"):
        training.update_stopping(state, float("nan"), 1, settings)


def test_selections_reject_overlap_and_unlabelled_and_resume_changes(synthetic_project, fake_alphagenome):
    config = synthetic_project
    prepare(config)
    build_cache(config, device="cpu")
    prepared = PreparedData(config)
    assert len(prepared.select(("train", "val"), ("X1",))) == 6
    with pytest.raises(ValueError, match="unique"):
        prepared.select(("train", "train"), ("X1",))
    with pytest.raises(ValueError, match="Missing"):
        prepared.select("test", ("X1",))
    with pytest.raises(ValueError, match="overlap"):
        training.train(config, train_splits=("train", "val"))
    with pytest.raises(ValueError, match="labelled"):
        training.train(config, val_splits=("test",), val_cells=("X3",))
    training.train(config)
    changed = copy.deepcopy(config)
    changed["train"]["val_min_relative_rise"] = .002
    with pytest.raises(ValueError, match="val_min_relative_rise"):
        training.train(changed, resume=True)
    with pytest.raises(ValueError, match="selected cells or splits"):
        training.train(config, resume=True, train_cells=("X1",), val_cells=("X2",),
                       train_splits=("train", "val"), val_splits=("train", "val"))
    path = Path(config["run_dir"]) / "last.pt"
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["schema"] = 1
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="schema-2"):
        training.train(config, resume=True)


def test_checkpoint_selection_is_loss_not_spearman(synthetic_project, fake_alphagenome, monkeypatch):
    config = synthetic_project
    prepare(config)
    build_cache(config, device="cpu")
    original = training.run_predictions
    calls = 0
    def controlled(model, loader, device):
        nonlocal calls
        calls += 1
        if calls % 2:  # Actual complete-training evaluation.
            return original(model, loader, device)
        target = np.log1p(loader.dataset.examples.gex.to_numpy(dtype=float))
        if calls == 2:
            return target + 10  # Perfect ranking, very poor magnitude/loss.
        return target[::-1]  # Worse ranking, smaller loss.
    monkeypatch.setattr(training, "run_predictions", controlled)
    best_path = training.train(config)
    checkpoint = torch.load(best_path, weights_only=True)
    assert checkpoint["epoch"] == 1
    assert checkpoint["history"][1]["val_spearman"] < checkpoint["history"][0]["val_spearman"]
    assert checkpoint["history"][1]["val_loss"] < checkpoint["history"][0]["val_loss"]


def test_early_stopping_state_survives_resume(synthetic_project, fake_alphagenome, monkeypatch):
    config = synthetic_project
    config["train"].update(epochs=5, patience=1, train_min_relative_improvement=.99)
    prepare(config)
    build_cache(config, device="cpu")
    original = training.run_predictions
    calls = 0
    def controlled(model, loader, device):
        nonlocal calls
        calls += 1
        if calls % 2:
            return original(model, loader, device)
        return np.log1p(loader.dataset.examples.gex.to_numpy(dtype=float)) + calls
    monkeypatch.setattr(training, "run_predictions", controlled)
    training.train(config)
    last = torch.load(Path(config["run_dir"]) / "last.pt", weights_only=True)
    assert len(last["history"]) == 2
    assert last["stopping_state"]["val_rise_epochs"] == 1
    config["train"]["epochs"] = 8
    training.train(config, resume=True)
    assert calls == 4
    from rna_pred.common.config import read_json
    assert read_json(Path(config["run_dir"]) / "run.json")["termination_reason"] == "early_stopping"
