"""Weighted regression and within-cell pairwise ranking experiments."""
import numpy as np
import torch
from torch.nn import functional as F
from rna_pred.common.metrics import threshold_for

def expression_loss(scores, targets, cells, thresholds, settings):
    residual_loss = F.smooth_l1_loss(scores, targets, reduction="none")
    if settings["name"] == "smooth_l1":
        return residual_loss.mean()
    limits = scores.new_tensor([np.log1p(threshold_for(thresholds, c)) for c in cells])
    low = (targets > 0) & (targets <= limits)
    weights = torch.where(low, settings["low_weight"], 1.0)
    if settings["name"] == "weighted_smooth_l1":
        return (weights * residual_loss).sum() / weights.sum()
    cell_losses = []
    for cell in sorted(set(cells)):
        indices = torch.tensor([i for i, c in enumerate(cells) if c == cell], device=scores.device)
        s, y, lo = scores[indices], targets[indices], low[indices]
        i, j = torch.triu_indices(len(indices), len(indices), offset=1, device=scores.device)
        unequal = y[i] != y[j]
        i, j = i[unequal], j[unequal]
        zero_positive = (y[i] == 0) | (y[j] == 0)
        low_low = lo[i] & lo[j]
        groups = [zero_positive, low_low, ~(zero_positive | low_low)]
        numerator, denominator = s.sum() * 0, 0.0
        for group, weight in zip(groups, [1.0, settings["low_weight"], 1.0]):
            selected = torch.where(group)[0]
            if not len(selected):
                continue
            if len(selected) > settings["pairs_per_group"]:
                selected = selected[torch.randperm(len(selected), device=scores.device)[:settings["pairs_per_group"]]]
            a, b = i[selected], j[selected]
            margin = torch.sign(y[a] - y[b]) * (s[a] - s[b]) / settings["temperature"]
            # Equal budget per available pair group; bounded extra weight for low-low.
            numerator = numerator + weight * F.softplus(-margin).mean()
            denominator += weight
        if denominator:
            cell_losses.append(numerator / denominator)
    ranking = torch.stack(cell_losses).mean() if cell_losses else scores.sum() * 0
    return residual_loss.mean() + settings["ranking_weight"] * ranking

