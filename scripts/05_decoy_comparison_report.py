"""
Aggregate and compare featurizer performance across alternative decoy sets.
Reads results/decoy_sets/per_target_metrics.csv produced by 04_eval_decoy_sets.py.

Outputs:
  results/decoy_sets/report/aggregate_metrics.csv
  results/decoy_sets/report/wilcoxon_tests.csv
  results/decoy_sets/report/figures/{metric}_comparison.png
  results/logs/05_decoy_comparison_report.log

Usage:
    python scripts/05_decoy_comparison_report.py [--targets TARGET [...]]
                                                  [--decoy-sets SET [...]]
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

from config import RESULTS_DIR
from src.utils import setup_logging, timer

log = logging.getLogger(__name__)

DECOY_SETS_DIR = RESULTS_DIR / "decoy_sets"
REPORT_DIR = DECOY_SETS_DIR / "report"
_LOG_FILE = RESULTS_DIR / "logs" / f"{Path(__file__).stem}.log"
PRIMARY_METRICS = ["roc_auc", "pr_auc", "bedroc", "ef_1pct", "ef_5pct", "mcc", "f1"]

FEATURIZER_ORDER = ["plif", "plec", "grim", "pocket_featurizer",
                    "distance_matrix", "distance_chemistry"]
FEATURIZER_LABELS = {
    "plif": "PLIF", "plec": "PLEC", "grim": "GRIM",
    "pocket_featurizer": "Pocket",
    "distance_matrix": "Distance", "distance_chemistry": "DistChem",
}


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {path}\n"
            "Run 04_eval_decoy_sets.py first."
        )
    df = pd.read_csv(path)
    log.info("Loaded %d rows from %s", len(df), path)
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of each primary metric across targets, grouped by (decoy_set, featurizer)."""
    rows = []
    for (ds, feat), sub in df.groupby(["decoy_set", "featurizer"]):
        row: dict = {"decoy_set": ds, "featurizer": feat, "n_targets": len(sub)}
        for metric in PRIMARY_METRICS:
            col = f"{metric}_mean"
            if col in sub.columns:
                row[f"{metric}_mean"] = sub[col].mean()
                row[f"{metric}_std"] = sub[col].std()
        rows.append(row)
    agg_df = pd.DataFrame(rows).sort_values(
        ["featurizer", "roc_auc_mean"], ascending=[True, False]
    )
    return agg_df


def plot_metric(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    col = f"{metric}_mean"
    if col not in df.columns:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Map internal names to display labels
    plot_df = df.copy()
    plot_df["featurizer"] = plot_df["featurizer"].map(
        lambda x: FEATURIZER_LABELS.get(x, x)
    )
    order = [FEATURIZER_LABELS.get(f, f) for f in FEATURIZER_ORDER
             if FEATURIZER_LABELS.get(f, f) in plot_df["featurizer"].values]

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(
        data=plot_df,
        x="featurizer", y=col,
        hue="decoy_set",
        order=order,
        ax=ax,
    )
    ax.set_title(f"{metric} by featurizer and decoy set")
    ax.set_xlabel("Featurizer")
    ax.set_ylabel(metric)
    ax.legend(title="Decoy set", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(out_dir / f"{metric}_comparison.png", dpi=150)
    plt.close(fig)
    log.info("Saved figure: %s", out_dir / f"{metric}_comparison.png")


def wilcoxon_tests(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pairwise Wilcoxon tests between decoy sets, within each featurizer."""
    col = f"{metric}_mean"
    if col not in df.columns:
        return pd.DataFrame()

    rows = []
    for featurizer, sub in df.groupby("featurizer"):
        decoy_sets = sub["decoy_set"].unique().tolist()
        for i, ds1 in enumerate(decoy_sets):
            for ds2 in decoy_sets[i + 1:]:
                s1 = sub[sub["decoy_set"] == ds1].set_index("target")[col]
                s2 = sub[sub["decoy_set"] == ds2].set_index("target")[col]
                common = s1.index.intersection(s2.index)
                if len(common) < 5:
                    continue
                s1_vals = s1[common].values
                s2_vals = s2[common].values
                if np.allclose(s1_vals, s2_vals):
                    continue
                try:
                    stat, p = stats.wilcoxon(s1_vals, s2_vals)
                    rows.append({
                        "featurizer": featurizer,
                        "ds1": ds1, "ds2": ds2,
                        "metric": metric,
                        "n_common": len(common),
                        "statistic": stat,
                        "p_value": p,
                    })
                except Exception as e:
                    log.warning("Wilcoxon failed for %s/%s vs %s: %s", featurizer, ds1, ds2, e)
    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Saved CSV: %s (%d rows)", path, len(df))


def main(targets: list[str] | None, decoy_sets: list[str] | None) -> None:
    setup_logging(log_file=_LOG_FILE)

    with timer("Decoy comparison report total"):
        df = load_metrics(DECOY_SETS_DIR / "per_target_metrics.csv")

        if targets:
            df = df[df["target"].isin(targets)]
        if decoy_sets:
            df = df[df["decoy_set"].isin(decoy_sets)]

        if df.empty:
            log.error("No data after filtering — check --targets / --decoy-sets")
            return

        log.info("Report covers %d decoy_sets × %d featurizers × %d targets",
                 df["decoy_set"].nunique(),
                 df["featurizer"].nunique(),
                 df["target"].nunique())

        # Aggregate summary
        agg_df = aggregate(df)
        save_csv(agg_df, REPORT_DIR / "aggregate_metrics.csv")
        log.info("\n%s", agg_df.to_string(index=False))

        # Figures
        fig_dir = REPORT_DIR / "figures"
        for metric in PRIMARY_METRICS:
            plot_metric(df, metric, fig_dir)

        # Wilcoxon tests
        stat_frames = [wilcoxon_tests(df, m) for m in PRIMARY_METRICS]
        stat_frames = [f for f in stat_frames if not f.empty]
        if stat_frames:
            stat_df = pd.concat(stat_frames, ignore_index=True)
            save_csv(stat_df, REPORT_DIR / "wilcoxon_tests.csv")
        else:
            log.warning("No Wilcoxon tests produced (need ≥2 decoy sets with ≥5 common targets)")

    log.info("All outputs saved to %s", REPORT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--decoy-sets", nargs="+", default=None)
    args = parser.parse_args()
    main(args.targets, args.decoy_sets)
