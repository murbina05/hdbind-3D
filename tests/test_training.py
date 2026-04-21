"""Smoke tests for src/training.py."""
from __future__ import annotations
import numpy as np
import pytest
from src.training import make_classifier, cross_validate

RNG = np.random.default_rng(0)
N = 300
X = RNG.standard_normal((N, 20)).astype(np.float32)
Y = np.zeros(N, dtype=int)
Y[:15] = 1  # ~5% actives — similar to DUDE-Z ratio


def test_make_classifier_returns_estimator():
    clf = make_classifier(X, use_xgb=False)
    assert hasattr(clf, "fit")
    assert hasattr(clf, "predict_proba")


def test_cross_validate_returns_dict():
    agg = cross_validate(X, Y, n_folds=2, use_xgb=False)
    assert isinstance(agg, dict)
    assert "roc_auc" in agg
    assert "mean" in agg["roc_auc"]
    assert "std" in agg["roc_auc"]


def test_cross_validate_roc_auc_range():
    agg = cross_validate(X, Y, n_folds=2, use_xgb=False)
    assert 0 <= agg["roc_auc"]["mean"] <= 1


def test_cross_validate_with_xgboost():
    try:
        import xgboost  # noqa: F401
        agg = cross_validate(X, Y, n_folds=2, use_xgb=True)
        assert "roc_auc" in agg
    except ImportError:
        pytest.skip("XGBoost not installed")
