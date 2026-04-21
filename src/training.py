from __future__ import annotations
"""
Cross-validation training loop. Reused identically across all Tier 1-2 variants.
Uses GPU-accelerated XGBoost as default; falls back to sklearn LogisticRegression on CPU.
cuML LogisticRegression used when RAPIDS is available and dataset is large.
"""
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from config import CV_FOLDS, CV_SEED, THRESHOLD_SCAN, XGBOOST_PARAMS, GPU_AVAILABLE, CUML_AVAILABLE
from src.evaluation import compute_all_metrics, aggregate_fold_metrics, optimal_threshold_mcc

log = logging.getLogger(__name__)


def make_classifier(X: np.ndarray, use_xgb: bool = True):
    """
    Return the best available classifier for this hardware and dataset size.
    Priority: XGBoost GPU > cuML LogReg GPU > sklearn LogReg CPU.
    """
    n_samples = X.shape[0]

    if use_xgb:
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(**XGBOOST_PARAMS, verbosity=0, use_label_encoder=False,
                                eval_metric="aucpr")
            log.info("Classifier: XGBoost (device=%s)", XGBOOST_PARAMS["device"])
            return clf
        except ImportError:
            log.warning("XGBoost not available, falling back to LogisticRegression")

    if CUML_AVAILABLE and GPU_AVAILABLE and n_samples > 50_000:
        from cuml.linear_model import LogisticRegression as cuLogReg
        clf = cuLogReg(C=1.0, max_iter=2000, class_weight="balanced")
        log.info("Classifier: cuML LogisticRegression (GPU)")
        return clf

    clf = LogisticRegression(
        C=1.0, max_iter=2000,
        class_weight="balanced",
        solver="lbfgs", n_jobs=1,
    )
    log.info("Classifier: sklearn LogisticRegression (CPU)")
    return clf


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model=None,
    n_folds: int = CV_FOLDS,
    seed: int = CV_SEED,
    use_xgb: bool = True,
) -> dict[str, dict[str, float]]:
    """
    Run stratified k-fold CV and return aggregated metrics.

    Args:
        X       : Feature matrix (n_samples, n_features)
        y       : Binary labels (n_samples,)
        model   : sklearn-compatible estimator; if None, selects best available
        n_folds : Number of CV folds
        seed    : Random seed for reproducibility
        use_xgb : Prefer XGBoost GPU if available (recommended)

    Returns:
        Aggregated metrics dict: {metric_name: {mean, std}}
    """
    if model is None:
        model = make_classifier(X, use_xgb=use_xgb)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics = []

    for fold_i, (train_idx, test_idx) in enumerate(tqdm(skf.split(X, y), total=n_folds, desc="folds", leave=False)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model.fit(X_tr, y_tr)
        y_score = model.predict_proba(X_te)[:, 1]
        if hasattr(y_score, "to_numpy"):   # cuML returns cudf Series
            y_score = y_score.to_numpy()

        threshold = optimal_threshold_mcc(y_te, y_score, THRESHOLD_SCAN)
        fold_metrics.append(compute_all_metrics(y_te, y_score, threshold=threshold))
        log.debug("Fold %d/%d — threshold=%.2f", fold_i + 1, n_folds, threshold)

    return aggregate_fold_metrics(fold_metrics)
