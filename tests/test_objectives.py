import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from rna_pred.experiments.low_expression_losses.losses import expression_loss
from rna_pred.common.objectives import objective_settings
from rna_pred.common.metrics import fit_thresholds, low_expression_metrics
from rna_pred.baseline import workflow as training
from rna_pred.common.data.prepare import prepare
from rna_pred.common.data.cache import build_cache


def settings(name, **kwargs):
    return objective_settings({"objective": {"name": name, **kwargs}})


def test_weighted_loss_upweights_only_low_positive():
    examples = pd.DataFrame({"cell": ["X1"] * 4, "gex": [0., 1., 2., 100.]})
    thresholds = fit_thresholds(examples, .25)
    y = torch.tensor(np.log1p(examples.gex.to_numpy()))
    s = y + torch.tensor([0., 1., 0., 0.])
    loss = expression_loss(s, y, examples.cell.tolist(), thresholds, settings('weighted_smooth_l1', low_weight=4))
    assert loss.item() == pytest.approx(4 * .5 / 7)
    assert thresholds['per_cell']['X1'] == 1.5


def test_ranking_rewards_ordering_and_has_finite_gradients():
    examples = pd.DataFrame({"cell": ["X1"] * 4, "gex": [0., .01, .02, 10.]})
    thresholds = fit_thresholds(examples, 1)
    y = torch.tensor(np.log1p(examples.gex.to_numpy()), dtype=torch.float32)
    cfg = settings('hybrid_ranking')
    good = torch.tensor([0., 1., 2., 3.], requires_grad=True)
    bad = good.detach().flip(0).requires_grad_()
    def ranking(s):
        return expression_loss(s, y, examples.cell.tolist(), thresholds, cfg) - torch.nn.functional.smooth_l1_loss(s, y)
    assert ranking(good) < ranking(bad)
    ranking(bad).backward()
    assert torch.isfinite(bad.grad).all()
    assert bad.grad[0] > 0 and bad.grad[-1] < 0


def test_ties_and_cross_cell_pairs_do_not_contribute():
    thresholds = {'pooled': 1., 'per_cell': {}}
    scores = torch.tensor([1., 2., 3.], requires_grad=True)
    targets = torch.zeros(3)
    for cells in [['X1'] * 3, ['X1', 'X2', 'X3']]:
        loss = expression_loss(scores, targets, cells, thresholds, settings('hybrid_ranking'))
        assert loss == torch.nn.functional.smooth_l1_loss(scores, targets)
        loss.backward()
    assert torch.isfinite(scores.grad).all()


def test_metrics_transfer_thresholds_and_undefined_subsets():
    source = pd.DataFrame({'cell': ['X1'] * 3, 'gex': [0., 1., 3.]})
    thresholds = fit_thresholds(source, .5)
    target = pd.DataFrame({'cell': ['X2'] * 4, 'gex': [0., .5, 1., 100.]})
    metrics = low_expression_metrics(target, np.array([0., 1., 2., 3.]), thresholds)['X2']
    assert metrics['threshold'] == 2
    assert metrics['threshold_source'] == 'pooled_training'
    assert metrics['zero_positive_auroc'] == 1
    assert metrics['low_positive']['spearman'] == pytest.approx(1)
    assert metrics['high_positive']['spearman'] is None
    tied = low_expression_metrics(source, np.ones(3), thresholds)['X1']
    assert tied['zero_positive_auroc'] == .5
    assert tied['low_positive']['spearman'] is None


@pytest.mark.parametrize('overrides', [{'temperature': 0}, {'low_weight': float('inf')},
                                      {'low_quantile': 0}, {'pairs_per_group': 0}, {'name': 'oops'}])
def test_invalid_settings(overrides):
    with pytest.raises(ValueError):
        objective_settings({'objective': overrides})


@pytest.mark.parametrize('name', ['weighted_smooth_l1', 'hybrid_ranking'])
def test_new_objectives_select_spearman_and_resume(synthetic_project, fake_alphagenome, monkeypatch, name):
    config = synthetic_project
    config['train'].update(objective={'name': name}, selection_metric='val_mean_spearman')
    prepare(config)
    build_cache(config, device='cpu')
    original = training.run_predictions
    calls = 0
    def controlled(model, loader, device):
        nonlocal calls
        calls += 1
        if calls % 2:
            return original(model, loader, device)
        target = np.log1p(loader.dataset.examples.gex.to_numpy(dtype=float))
        return target + 10 if calls == 2 else target[::-1]
    monkeypatch.setattr(training, 'run_predictions', controlled)
    best = training.train(config)
    checkpoint = torch.load(best, weights_only=True)
    assert checkpoint['epoch'] == 0
    assert checkpoint['selection_metric'] == 'val_mean_spearman'
    assert checkpoint['history'][0]['val_low_expression']['X1']['n_zero'] == 1
    assert checkpoint['expression_thresholds']['per_cell']['X1'] == 11.25
    changed = copy.deepcopy(config)
    changed['train']['objective']['low_weight'] = 4
    with pytest.raises(ValueError, match='objective'):
        training.train(changed, resume=True)
    changed = copy.deepcopy(config)
    changed['train']['selection_metric'] = 'val_loss'
    with pytest.raises(ValueError, match='selection_metric'):
        training.train(changed, resume=True)
    config['train']['epochs'] = 3
    training.train(config, resume=True)
    assert len(torch.load(Path(config['run_dir']) / 'last.pt', weights_only=True)['history']) == 3


def test_hybrid_balances_pair_groups_and_applies_bounded_weight():
    # One zero, two low-positive labels, one high label: all three groups exist.
    y = torch.tensor([0., .1, .2, 2.])
    s = torch.tensor([0., .3, -.2, 1.], requires_grad=True)
    cfg = settings('hybrid_ranking', low_weight=4, ranking_weight=.5)
    thresholds = {'pooled': float(np.expm1(.5)), 'per_cell': {}}
    actual = expression_loss(s, y, ['X1'] * 4, thresholds, cfg)
    def pair(a, b):
        return torch.nn.functional.softplus(s[a] - s[b])
    zero_group = torch.stack([pair(0, 1), pair(0, 2), pair(0, 3)]).mean()
    low_group = pair(1, 2)
    other_group = torch.stack([pair(1, 3), pair(2, 3)]).mean()
    expected = torch.nn.functional.smooth_l1_loss(s, y) + .5 * (zero_group + 4 * low_group + other_group) / 6
    torch.testing.assert_close(actual, expected)


def test_sampling_is_reproducible_and_bounded():
    y = torch.linspace(0, 4, 20)
    s = y.flip(0).requires_grad_()
    cfg = settings('hybrid_ranking', pairs_per_group=2)
    thresholds = {'pooled': 2., 'per_cell': {}}
    torch.manual_seed(17)
    first = expression_loss(s, y, ['X1'] * 20, thresholds, cfg)
    torch.manual_seed(17)
    second = expression_loss(s, y, ['X1'] * 20, thresholds, cfg)
    torch.testing.assert_close(first, second)
    first.backward()
    assert torch.isfinite(s.grad).all()


def test_spearman_plateau_stops_and_stays_stopped_on_resume(synthetic_project, fake_alphagenome, monkeypatch):
    config = synthetic_project
    config['train'].update(epochs=5, patience=1, selection_metric='val_mean_spearman',
                           objective={'name': 'hybrid_ranking'})
    prepare(config)
    build_cache(config, device='cpu')
    original = training.run_predictions
    calls = 0
    def controlled(model, loader, device):
        nonlocal calls
        calls += 1
        if calls % 2:
            return original(model, loader, device)
        return np.log1p(loader.dataset.examples.gex.to_numpy(dtype=float))
    monkeypatch.setattr(training, 'run_predictions', controlled)
    training.train(config)
    last = torch.load(Path(config['run_dir']) / 'last.pt', weights_only=True)
    assert len(last['history']) == 2
    assert last['stopping_state']['selection_plateau_epochs'] == 1
    config['train']['epochs'] = 8
    training.train(config, resume=True)
    assert calls == 4
