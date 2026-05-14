"""
Test 2 — Permutation feature importance for pocket_featurizer.

For each target, trains pocket_featurizer classifiers using the same 5-fold
CV splits as cross_validate() (StratifiedKFold, seed=CV_SEED), then measures
how much BEDROC drops when each of the 15 features is independently shuffled
on the held-out test fold.

A shortcut classifier will show one dominant feature (expected: MolWt or
HeavyAtoms); a truly pocket-learning classifier will show distributed
importance.

Feature order (matches PocketFeaturizer.feature_names):
  [0] pocket_hydrophobic  [1] pocket_polar    [2] pocket_positive
  [3] pocket_negative     [4] pocket_aromatic [5] pocket_size
  [6] MolWt               [7] MolLogP         [8] NumHBD
  [9] NumHBA              [10] TPSA            [11] NumRotBonds
  [12] NumAromaticRings   [13] NumHeavyAtoms   [14] FractionCSP3

Outputs:
  results/diagnostics/pocket_mw_shortcut/test2_permutation_importance.csv
  results/diagnostics/pocket_mw_shortcut/test2_permutation_importance_aggregated.csv

Usage:
    python scripts/diag_02_pocket_permutation_importance.py \\
        --decoy-set default \\
        --n-shuffles 10 \\
        [--targets AA2AR ADRB2 EGFR]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import (
    RESULTS_DIR, CV_FOLDS, CV_SEED, THRESHOLD_SCAN, get_all_targets,
)
from src.data_loading import load_labels
from src.evaluation import bedroc, optimal_threshold_mcc
from src.featurizers.pocket import PocketFeaturizer
from src.training import make_classifier
from src.utils import load_pickle, setup_logging, timer

log = logging.getLogger(__name__)

DIAG_DIR = RESULTS_DIR / "diagnostics" / "pocket_mw_shortcut"
_LOG_FILE = RESULTS_DIR / "logs" / "diag_02_pocket_permutation_importance.log"

# Where to find precomputed feature vectors per decoy set
_FEAT_DIRS: dict[str, Path] = {
    "default":    RESULTS_DIR / "tier2" / "pocket_featurizer",
    "DUDE_Z":     RESULTS_DIR / "decoy_sets" / "DUDE_Z" / "pocket_featurizer",
    "Goldilocks": RESULTS_DIR / "decoy_sets" / "Goldilocks" / "pocket_featurizer",
    "Extrema":    RESULTS_DIR / "decoy_sets" / "Extrema" / "pocket_featurizer",
}

FEATURE_NAMES: list[str] = PocketFeaturizer().feature_names


def _load_X_y(target: str, feat_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load feature matrix and labels for one target.
    Returns (X, y, valid_ids).  Raises on missing data.
    """
    pkl = feat_dir / f"{target}.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"Feature file not found: {pkl}")

    feat_dict: dict[str, np.ndarray] = load_pickle(pkl)
    if not feat_dict:
        raise ValueError(f"Empty feature dict for {target}")

    ids, labels = load_labels(target)
    label_map = dict(zip(ids, labels))
    valid_ids = [i for i in feat_dict if i in label_map]

    if len(valid_ids) < CV_FOLDS * 2:
        raise ValueError(
            f"{target}: only {len(valid_ids)} labeled complexes — too few for CV"
        )

    X = np.stack([feat_dict[i] for i in valid_ids]).astype(np.float32)
    y = np.array([label_map[i] for i in valid_ids], dtype=int)
    return X, y, valid_ids


def _pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_true, y_score))


def _bedroc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return bedroc(y_true, y_score)


def permutation_importance_target(
    target: str,
    X: np.ndarray,
    y: np.ndarray,
    n_shuffles: int,
    target_idx: int,
) -> list[dict]:
    """
    Run per-feature permutation importance for one target.

    For each CV fold:
      1. Train classifier on train split.
      2. Compute baseline BEDROC/PR-AUC on test split.
      3. For each feature i, repeat n_shuffles times:
         - Shuffle column i with fixed RNG(seed per target×feature).
         - Score → BEDROC drop, PR-AUC drop.

    Returns list of dicts, one per (fold, feature, shuffle).
    """
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    n_features = X.shape[1]
    all_drops: list[dict] = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if y_te.sum() == 0 or (1 - y_te).sum() == 0:
            log.warning("%s fold %d: single class in test fold, skipping", target, fold_i)
            continue

        clf = make_classifier(X_tr, use_xgb=True)
        clf.fit(X_tr, y_tr)

        raw_base = clf.predict_proba(X_te)
        if hasattr(raw_base, "to_numpy"):
            raw_base = raw_base.to_numpy()
        if raw_base.ndim == 2:
            y_base = raw_base[:, 1]
        else:
            y_base = raw_base

        bedroc_base  = _bedroc_score(y_te, y_base)
        pr_auc_base  = _pr_auc(y_te, y_base)

        for feat_i in range(n_features):
            # Fixed seed per (target, feature) — same across folds for reproducibility
            seed = int(CV_SEED + target_idx * n_features * 7919 + feat_i * 97 + fold_i * 31)
            rng  = np.random.default_rng(seed)

            for shuffle_j in range(n_shuffles):
                X_perm = X_te.copy()
                X_perm[:, feat_i] = rng.permutation(X_perm[:, feat_i])

                raw_perm = clf.predict_proba(X_perm)
                if hasattr(raw_perm, "to_numpy"):
                    raw_perm = raw_perm.to_numpy()
                if raw_perm.ndim == 2:
                    y_perm = raw_perm[:, 1]
                else:
                    y_perm = raw_perm

                bedroc_perm = _bedroc_score(y_te, y_perm)
                pr_auc_perm = _pr_auc(y_te, y_perm)

                all_drops.append({
                    "target":        target,
                    "fold":          fold_i,
                    "shuffle":       shuffle_j,
                    "feature_index": feat_i,
                    "feature_name":  FEATURE_NAMES[feat_i],
                    "bedroc_base":   bedroc_base,
                    "pr_auc_base":   pr_auc_base,
                    "bedroc_drop":   bedroc_base - bedroc_perm,
                    "pr_auc_drop":   pr_auc_base - pr_auc_perm,
                })

        log.debug("%s fold %d: baseline BEDROC=%.4f PR-AUC=%.4f",
                  target, fold_i, bedroc_base, pr_auc_base)

    return all_drops


def aggregate_target_drops(drops: list[dict]) -> list[dict]:
    """
    Collapse raw (fold × shuffle) drops into mean ± std per (target, feature).
    """
    df = pd.DataFrame(drops)
    if df.empty:
        return []

    rows = []
    for (target, feat_name, feat_idx), sub in df.groupby(
        ["target", "feature_name", "feature_index"]
    ):
        rows.append({
            "target":          target,
            "feature_name":    feat_name,
            "feature_index":   feat_idx,
            "mean_bedroc_drop": sub["bedroc_drop"].mean(),
            "std_bedroc_drop":  sub["bedroc_drop"].std(ddof=1) if len(sub) > 1 else 0.0,
            "mean_pr_auc_drop": sub["pr_auc_drop"].mean(),
            "std_pr_auc_drop":  sub["pr_auc_drop"].std(ddof=1) if len(sub) > 1 else 0.0,
            "n_obs":            len(sub),
        })
    return rows


def aggregate_across_targets(per_target_df: pd.DataFrame) -> pd.DataFrame:
    """Per-feature median and IQR of mean BEDROC drop across all targets."""
    rows = []
    for feat_idx in sorted(per_target_df["feature_index"].unique()):
        sub  = per_target_df[per_target_df["feature_index"] == feat_idx]["mean_bedroc_drop"]
        name = per_target_df[per_target_df["feature_index"] == feat_idx]["feature_name"].iloc[0]
        rows.append({
            "feature_index":            feat_idx,
            "feature_name":             name,
            "n_targets":                len(sub),
            "median_bedroc_drop":       sub.median(),
            "iqr_bedroc_drop":          sub.quantile(0.75) - sub.quantile(0.25),
            "mean_bedroc_drop_overall": sub.mean(),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("median_bedroc_drop", ascending=False)
        .reset_index(drop=True)
    )


def flag_shortcuts(per_target_df: pd.DataFrame) -> None:
    """
    Print targets where a single feature's mean BEDROC drop exceeds 50% of the
    total BEDROC drop sum for that target — these are shortcut targets.
    """
    flagged = []
    for target, sub in per_target_df.groupby("target"):
        total = sub["mean_bedroc_drop"].sum()
        if total <= 0:
            continue
        dominant = sub.loc[sub["mean_bedroc_drop"].idxmax()]
        if dominant["mean_bedroc_drop"] > 0.5 * total:
            flagged.append({
                "target":           target,
                "dominant_feature": dominant["feature_name"],
                "dominant_drop":    dominant["mean_bedroc_drop"],
                "total_drop":       total,
                "fraction":         dominant["mean_bedroc_drop"] / total,
            })

    if flagged:
        print("\nShortcut targets (dominant feature > 50% of total BEDROC drop):")
        for row in sorted(flagged, key=lambda r: -r["fraction"]):
            print(f"  {row['target']:10s}  dominant={row['dominant_feature']:22s}"
                  f"  drop={row['dominant_drop']:.4f}  fraction={row['fraction']:.1%}")
    else:
        print("\nNo single-feature shortcut targets detected.")


def print_summary(agg_df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("TEST 2 — Permutation Feature Importance (pocket_featurizer)")
    print("=" * 72)
    print("\nFeatures ranked by median BEDROC drop across targets:")
    print(agg_df[["feature_name", "n_targets", "median_bedroc_drop",
                  "iqr_bedroc_drop"]].to_string(index=False))
    print("=" * 72 + "\n")


def main(
    targets: list[str],
    decoy_set: str,
    n_shuffles: int,
    feat_dir: Path | None,
) -> None:
    setup_logging(log_file=_LOG_FILE)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    if feat_dir is None:
        if decoy_set not in _FEAT_DIRS:
            raise ValueError(
                f"Unknown decoy set '{decoy_set}'. Known: {list(_FEAT_DIRS)} "
                "or supply --feat-dir explicitly."
            )
        feat_dir = _FEAT_DIRS[decoy_set]

    log.info("Feature dir: %s", feat_dir)
    log.info("n_shuffles=%d, CV_FOLDS=%d, CV_SEED=%d", n_shuffles, CV_FOLDS, CV_SEED)

    all_sorted_targets = get_all_targets()  # fixed ordering for stable seed

    all_per_target_rows: list[dict] = []

    with timer("Test 2 — permutation importance"):
        for target in targets:
            log.info("Processing target: %s", target)
            target_idx = (
                all_sorted_targets.index(target)
                if target in all_sorted_targets
                else hash(target) % 10000
            )
            try:
                X, y, _ = _load_X_y(target, feat_dir)
            except (FileNotFoundError, ValueError) as e:
                log.error("%s: skipping — %s", target, e)
                continue

            log.info("  %s: X=%s  positives=%d  negatives=%d",
                     target, X.shape, y.sum(), (1 - y).sum())

            raw_drops = permutation_importance_target(
                target, X, y, n_shuffles, target_idx
            )
            if not raw_drops:
                log.warning("%s: no drops computed — skipping", target)
                continue

            all_per_target_rows.extend(aggregate_target_drops(raw_drops))

    if not all_per_target_rows:
        log.error("No results — check that feature pkl files exist at %s", feat_dir)
        return

    per_target_df = pd.DataFrame(all_per_target_rows)
    out_per = DIAG_DIR / "test2_permutation_importance.csv"
    per_target_df.to_csv(out_per, index=False)
    log.info("Saved per-target importance: %s (%d rows)", out_per, len(per_target_df))

    agg_df = aggregate_across_targets(per_target_df)
    out_agg = DIAG_DIR / "test2_permutation_importance_aggregated.csv"
    agg_df.to_csv(out_agg, index=False)
    log.info("Saved aggregated importance: %s (%d rows)", out_agg, len(agg_df))

    print_summary(agg_df)
    flag_shortcuts(per_target_df)

    log.info("Test 2 complete — outputs:\n  %s\n  %s", out_per, out_agg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decoy-set",
        default="default",
        choices=list(_FEAT_DIRS),
        help="Which feature set to use (default: DUDE-Z default from tier2/)",
    )
    parser.add_argument(
        "--feat-dir",
        type=Path,
        default=None,
        help="Override feature directory (pocket_featurizer pkl files)",
    )
    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=10,
        help="Number of shuffle repetitions per feature per fold (default: 10)",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=get_all_targets(),
        help="Subset of targets (default: all 43)",
    )
    args = parser.parse_args()
    main(args.targets, args.decoy_set, args.n_shuffles, args.feat_dir)
