"""Shared regression metrics and training-derived low-expression diagnostics."""
import numpy as np
from scipy.stats import rankdata, spearmanr

def spearman_score(target, predictions):
    if len(predictions) < 2 or np.unique(predictions).size < 2 or np.unique(target).size < 2:
        return 0.0
    return float(spearmanr(target, predictions).statistic)


def validation_metrics(examples, predictions):
    per_cell = {}
    for cell in sorted(examples.cell.unique()):
        mask = examples.cell.eq(cell).to_numpy()
        per_cell[cell] = spearman_score(examples.loc[mask, "gex"].to_numpy(), predictions[mask])
    return {"spearman": per_cell, "mean_spearman": float(np.mean(list(per_cell.values()))),
            "combined_spearman": spearman_score(examples.gex.to_numpy(), predictions)}


def prediction_loss(examples, predictions):
    residual = np.abs(predictions - np.log1p(examples.gex.to_numpy(dtype=float)))
    loss = float(np.mean(np.where(residual < 1, 0.5 * residual ** 2, residual - 0.5)))
    if not np.isfinite(loss):
        raise ValueError("Evaluation loss became nonfinite")
    return loss


def fit_thresholds(examples, quantile):
    def threshold(values):
        positive = np.asarray(values)[np.asarray(values) > 0]
        return float(np.quantile(positive, quantile)) if len(positive) else 0.0
    return {"pooled": threshold(examples.gex),
            "per_cell": {cell: threshold(frame.gex) for cell, frame in examples.groupby("cell")}}


def threshold_for(thresholds, cell):
    # Transfer validation cells have no training labels: use pooled source labels.
    return thresholds["per_cell"].get(cell, thresholds["pooled"])


def low_expression_metrics(examples, predictions, thresholds):
    result = {}
    for cell, frame in examples.groupby("cell", sort=True):
        mask = examples.cell.eq(cell).to_numpy()
        y, scores = frame.gex.to_numpy(), predictions[mask]
        limit = threshold_for(thresholds, cell)
        groups = {"low_positive": (y > 0) & (y <= limit), "high_positive": y > limit}
        metrics = {"threshold": limit, "threshold_source": "cell_training" if cell in thresholds["per_cell"] else "pooled_training",
                   "n_zero": int((y == 0).sum())}
        for name, subset in groups.items():
            a, b = y[subset], scores[subset]
            defined = len(a) >= 2 and np.unique(a).size > 1 and np.unique(b).size > 1
            metrics[name] = {"n": len(a), "spearman": float(spearmanr(a, b).statistic) if defined else None,
                             "mae_log1p": float(np.abs(b - np.log1p(a)).mean()) if len(a) else None}
        n_positive, n_zero = int((y > 0).sum()), int((y == 0).sum())
        metrics["zero_positive_auroc"] = (float((rankdata(scores)[y > 0].sum() - n_positive * (n_positive + 1) / 2)
                                                    / (n_positive * n_zero)) if n_positive and n_zero else None)
        result[cell] = metrics
    return result


def regression_metrics(examples, predictions):
    correlations = validation_metrics(examples, predictions)
    observed = np.log1p(examples.gex.to_numpy(dtype=float))
    residual = predictions - observed
    variance = np.sum((observed - observed.mean()) ** 2)
    constant = len(observed) < 2 or np.ptp(observed) == 0 or np.ptp(predictions) == 0
    return {"spearman": correlations["combined_spearman"], "spearman_per_cell": correlations["spearman"],
            "n": len(observed), "loss": prediction_loss(examples, predictions),
            "rmse_log1p": float(np.sqrt(np.mean(residual ** 2))),
            "mae_log1p": float(np.mean(np.abs(residual))),
            "pearson_log1p": None if constant else float(np.corrcoef(observed, predictions)[0, 1]),
            "r2_log1p": None if variance == 0 else float(1 - np.sum(residual ** 2) / variance)}


def diagnostics(examples, scores, thresholds):
    result = low_expression_metrics(examples, scores, thresholds)
    from scipy.stats import rankdata
    for cell, frame in examples.groupby("cell"):
        y = frame.gex.to_numpy()
        s = scores[examples.cell.eq(cell).to_numpy()]
        limit = threshold_for(thresholds, cell)
        subset = y <= limit
        a, b = y[subset], s[subset]
        n_pos, n_zero = int((a > 0).sum()), int((a == 0).sum())
        result[cell]["zero_low_positive_auroc"] = (
            float((rankdata(b)[a > 0].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_zero))
            if n_pos and n_zero else None)
        for name, mask in [("low_positive", (y > 0) & (y <= limit)), ("high_positive", y > limit)]:
            result[cell][name]["n_distinct_labels"] = int(np.unique(y[mask]).size)
    return result


