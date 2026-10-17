"""Objective settings and default Smooth L1; optional losses load on demand."""
import numpy as np
from torch.nn import functional as F

DEFAULTS = {"name": "smooth_l1", "low_quantile": 0.25, "low_weight": 2.0,
            "ranking_weight": 0.1, "temperature": 1.0, "pairs_per_group": 1024}


def objective_settings(settings):
    supplied = settings.get("objective", {})
    if set(supplied) - set(DEFAULTS):
        raise ValueError("Unknown objective setting")
    result = {**DEFAULTS, **supplied}
    if result["name"] not in {"smooth_l1", "weighted_smooth_l1", "hybrid_ranking"}:
        raise ValueError("Unknown training objective")
    if not 0 < result["low_quantile"] <= 1:
        raise ValueError("low_quantile must be in (0, 1]")
    for key in ("low_weight", "ranking_weight", "temperature"):
        if not np.isfinite(result[key]) or result[key] < (1 if key == "low_weight" else 0):
            raise ValueError(f"Invalid {key}")
    if result["temperature"] == 0:
        raise ValueError("temperature must be positive")
    if not isinstance(result["pairs_per_group"], int) or result["pairs_per_group"] < 1:
        raise ValueError("pairs_per_group must be a positive integer")
    return result



def expression_loss(scores, targets, cells, thresholds, settings):
    if settings["name"] == "smooth_l1":
        return F.smooth_l1_loss(scores, targets, reduction="none").mean()
    from rna_pred.experiments.low_expression_losses.losses import expression_loss as experimental_loss
    return experimental_loss(scores, targets, cells, thresholds, settings)
