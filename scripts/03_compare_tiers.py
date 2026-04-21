"""
Cross-tier analysis: load all featurizer results, run 5-fold CV, compare with
statistical tests, and save figures and CSVs.

Usage:
    python scripts/03_compare_tiers.py [--targets TARGET [TARGET ...]]
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import get_all_targets, RESULTS_DIR, CV_FOLDS, CV_SEED, EF_FRACTIONS, BEDROC_ALPHA
from src.utils import setup_logging, load_pickle, timer
from src.training import cross_validate
from src.evaluation import aggregate_fold_metrics

log = logging.getLogger(__name__)

VARIANTS = {
    "tier1/plif":              "PLIF",
    "tier1/plec":              "PLEC",
    "tier1/grim":              "GRIM",
    "tier2/pocket_featurizer": "Pocket",
    "tier2/distance_matrix":   "Distance",
    "tier2/distance_chemistry":"DistChem",
}

PRIMARY_METRICS = ["roc_auc", "pr_auc", "bedroc", "ef_1pct", "ef_5pct", "mcc", "f1"]


def load_features_labels(variant_path: Path, target: str) -> tuple[np.ndarray, np.ndarray] | None:
    pkl = variant_path / f"{target}.pkl"
    if not pkl.exists():
        return None
    feats: dict[str, np.ndarray] = load_pickle(pkl)
    if not feats:
        return None

    # Labels are inferred from mol_id naming convention (ligands_ prefix → active)
    # We need to load labels from data_loading; use the feature keys as guide
    from src.data_loading import load_labels
    try:
        ids, labels = load_labels(target)
    except Exception as e:
        log.error("%s: load_labels failed — %s", target, e)
        return None

    label_map = dict(zip(ids, labels))
    valid_ids = [i for i in feats if i in label_map]
    if not valid_ids:
        return None

    X = np.stack([feats[i] for i in valid_ids]).astype(np.float32)
    y = np.array([label_map[i] for i in valid_ids], dtype=int)
    return X, y


def run_all_variants(targets: list[str]) -> pd.DataFrame:
    rows = []
    for rel_path, label in VARIANTS.items():
        variant_path = RESULTS_DIR / rel_path
        log.info("Evaluating variant: %s (%s)", label, variant_path)
        for target in targets:
            data = load_features_labels(variant_path, target)
            if data is None:
                log.warning("%s/%s: no data, skipping", label, target)
                continue
            X, y = data
            if y.sum() == 0 or (1 - y).sum() == 0:
                log.warning("%s/%s: single class, skipping", label, target)
                continue
            try:
                agg = cross_validate(X, y)
                row = {"variant": label, "target": target}
                for metric, vals in agg.items():
                    row[f"{metric}_mean"] = vals["mean"]
                    row[f"{metric}_std"] = vals["std"]
                rows.append(row)
            except Exception as e:
                log.error("%s/%s: CV failed — %s", label, target, e)

    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Saved CSV: %s", path)


def plot_metric_comparison(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    col = f"{metric}_mean"
    if col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    order = list(VARIANTS.values())
    sns.boxplot(data=df, x="variant", y=col, order=order, ax=ax)
    ax.set_title(f"{metric} across 43 targets")
    ax.set_xlabel("Featurizer variant")
    ax.set_ylabel(metric)
    plt.tight_layout()
    fig.savefig(out_dir / f"{metric}_comparison.png", dpi=150)
    plt.close(fig)


def statistical_tests(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests between all variant pairs."""
    col = f"{metric}_mean"
    if col not in df.columns:
        return pd.DataFrame()

    variants = df["variant"].unique().tolist()
    rows = []
    for i, v1 in enumerate(variants):
        for v2 in variants[i + 1:]:
            s1 = df[df["variant"] == v1][col].dropna()
            s2 = df[df["variant"] == v2][col].dropna()
            common = set(df[df["variant"] == v1]["target"]) & set(df[df["variant"] == v2]["target"])
            if len(common) < 5:
                continue
            s1 = df[(df["variant"] == v1) & df["target"].isin(common)].set_index("target")[col]
            s2 = df[(df["variant"] == v2) & df["target"].isin(common)].set_index("target")[col]
            s1, s2 = s1.align(s2, join="inner")
            stat, p = stats.wilcoxon(s1.values, s2.values)
            rows.append({"v1": v1, "v2": v2, "metric": metric, "statistic": stat, "p_value": p})
    return pd.DataFrame(rows)


def main(targets: list[str]) -> None:
    setup_logging()
    out_dir = RESULTS_DIR / "comparison"
    fig_dir = out_dir / "figures"

    log.info("Cross-tier comparison: %d targets, %d variants", len(targets), len(VARIANTS))

    with timer("Cross-tier analysis total"):
        df = run_all_variants(targets)

    if df.empty:
        log.error("No results collected — check that all featurizer scripts have been run first.")
        return

    # Per-target metrics CSV
    save_csv(df, out_dir / "per_target_metrics.csv")

    # Aggregate summary
    agg_rows = []
    for variant in df["variant"].unique():
        sub = df[df["variant"] == variant]
        row = {"variant": variant}
        for metric in PRIMARY_METRICS:
            col = f"{metric}_mean"
            if col in sub.columns:
                row[f"{metric}_mean"] = sub[col].mean()
                row[f"{metric}_std"]  = sub[col].std()
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows).sort_values("roc_auc_mean", ascending=False)
    save_csv(agg_df, out_dir / "aggregate_metrics.csv")
    log.info("\n%s", agg_df.to_string(index=False))

    # Figures
    for metric in PRIMARY_METRICS:
        plot_metric_comparison(df, metric, fig_dir)

    # Statistical tests
    stat_rows = []
    for metric in PRIMARY_METRICS:
        stat_rows.append(statistical_tests(df, metric))
    if stat_rows:
        stat_df = pd.concat(stat_rows, ignore_index=True)
        save_csv(stat_df, out_dir / "wilcoxon_tests.csv")

    log.info("All outputs saved to %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", default=get_all_targets())
    args = parser.parse_args()
    main(args.targets)
