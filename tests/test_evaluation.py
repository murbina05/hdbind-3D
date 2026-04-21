"""Smoke tests for src/evaluation.py."""
from __future__ import annotations
import numpy as np
import pytest
from src.evaluation import (
    enrichment_factor,
    hit_rate_at_k,
    bedroc,
    optimal_threshold_mcc,
    compute_all_metrics,
    aggregate_fold_metrics,
)

RNG = np.random.default_rng(42)
N = 200
Y_TRUE = np.zeros(N, dtype=int)
Y_TRUE[:10] = 1  # 10 actives, 190 decoys
Y_SCORE = RNG.uniform(0, 1, N).astype(np.float32)
# Make actives score slightly higher
Y_SCORE[:10] += 0.3
Y_SCORE = np.clip(Y_SCORE, 0, 1)


def test_enrichment_factor_range():
    ef = enrichment_factor(Y_TRUE, Y_SCORE, 0.05)
    assert ef >= 0.0


def test_enrichment_factor_perfect():
    y_true = np.array([1, 1, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    ef = enrichment_factor(y_true, y_score, 0.4)
    assert ef > 1.0


def test_hit_rate_at_k():
    hr = hit_rate_at_k(Y_TRUE, Y_SCORE, 0.05)
    assert 0 <= hr <= 1


def test_bedroc():
    score = bedroc(Y_TRUE, Y_SCORE, alpha=20.0)
    assert 0 <= score <= 1


def test_optimal_threshold_mcc():
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    t = optimal_threshold_mcc(Y_TRUE, Y_SCORE, thresholds)
    assert t in thresholds


def test_compute_all_metrics_keys():
    metrics = compute_all_metrics(Y_TRUE, Y_SCORE)
    for key in ("roc_auc", "pr_auc", "bedroc", "ef_1pct", "ef_5pct",
                "precision", "recall", "f1", "f2", "mcc", "balanced_accuracy"):
        assert key in metrics


def test_compute_all_metrics_ranges():
    metrics = compute_all_metrics(Y_TRUE, Y_SCORE)
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["pr_auc"] <= 1
    assert 0 <= metrics["f1"] <= 1


def test_aggregate_fold_metrics():
    fold1 = {"roc_auc": 0.7, "pr_auc": 0.4}
    fold2 = {"roc_auc": 0.8, "pr_auc": 0.5}
    agg = aggregate_fold_metrics([fold1, fold2])
    assert "roc_auc" in agg
    assert abs(agg["roc_auc"]["mean"] - 0.75) < 1e-6
    assert "std" in agg["roc_auc"]
