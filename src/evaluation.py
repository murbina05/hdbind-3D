from __future__ import annotations
"""
All evaluation metrics in one place. Import from here exclusively — never reimplement inline.
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, fbeta_score,
    matthews_corrcoef, balanced_accuracy_score,
)


def enrichment_factor(y_true: np.ndarray, y_score: np.ndarray, top_frac: float) -> float:
    n = len(y_true)
    n_top = max(1, int(n * top_frac))
    top_idx = np.argsort(y_score)[::-1][:n_top]
    hit_rate_top = y_true[top_idx].mean()
    hit_rate_all = y_true.mean()
    return float(hit_rate_top / hit_rate_all) if hit_rate_all > 0 else 0.0


def hit_rate_at_k(y_true: np.ndarray, y_score: np.ndarray, top_frac: float) -> float:
    n = len(y_true)
    n_top = max(1, int(n * top_frac))
    top_idx = np.argsort(y_score)[::-1][:n_top]
    return float(y_true[top_idx].mean())


def bedroc(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 20.0) -> float:
    from rdkit.ML.Scoring.Scoring import CalcBEDROC
    scores_labels = sorted(zip(y_score, y_true), key=lambda x: -x[0])
    return float(CalcBEDROC(scores_labels, col=1, alpha=alpha))


def optimal_threshold_mcc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: list[float],
) -> float:
    """Return threshold that maximises MCC on the provided data."""
    best_t, best_mcc = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)
        if mcc > best_mcc:
            best_mcc, best_t = mcc, t
    return best_t


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    ef_fracs: list[float] = (0.01, 0.05),
    bedroc_alpha: float = 20.0,
) -> dict[str, float]:
    """
    Compute the full metric suite for one fold.
    All threshold-independent metrics use y_score (probabilities/scores).
    Threshold-dependent metrics use the provided threshold.
    """
    y_pred = (y_score >= threshold).astype(int)
    metrics: dict[str, float] = {}

    # Threshold-independent
    metrics["roc_auc"]   = roc_auc_score(y_true, y_score)
    metrics["pr_auc"]    = average_precision_score(y_true, y_score)
    for frac in ef_fracs:
        key = f"ef_{int(frac * 100)}pct"
        metrics[key] = enrichment_factor(y_true, y_score, frac)
        metrics[f"hit_rate_{int(frac * 100)}pct"] = hit_rate_at_k(y_true, y_score, frac)
    metrics["bedroc"]    = bedroc(y_true, y_score, bedroc_alpha)

    # Threshold-dependent
    metrics["precision"]         = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"]            = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"]                = f1_score(y_true, y_pred, zero_division=0)
    metrics["f2"]                = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    metrics["mcc"]               = matthews_corrcoef(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    return metrics


def aggregate_fold_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Aggregate list of per-fold dicts into {metric: {mean, std}}."""
    keys = fold_metrics[0].keys()
    return {
        k: {
            "mean": float(np.mean([m[k] for m in fold_metrics])),
            "std":  float(np.std([m[k] for m in fold_metrics])),
        }
        for k in keys
    }
