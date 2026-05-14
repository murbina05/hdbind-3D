"""
Test 1 — Goldilocks collapse diagnostic.

Checks whether pocket_featurizer's BEDROC collapses on Goldilocks decoys
(evidence of property-driven inflation) the way distance features did, or holds
up like PLEC (the control).

For each featurizer, computes per-target:
    ΔBEDROC      = BEDROC(DUDE-Z default) − BEDROC(Goldilocks)
    ΔPR-AUC      = PR-AUC(DUDE-Z default) − PR-AUC(Goldilocks)

If pocket's median ΔBEDROC is substantially larger than PLEC's, the pocket
featurizer is inflated by property matching.

Reads:
  --comparison-csv  : results/comparison/per_target_metrics.csv
                      (produced by 03_compare_tiers.py — default DUDE-Z run)
  --results-dir     : results/decoy_sets/per_target_metrics.csv
                      (produced by 04_eval_decoy_sets.py — Goldilocks/Extrema)

Outputs:
  results/diagnostics/pocket_mw_shortcut/test1_goldilocks_delta.csv

Usage:
    python scripts/diag_01_pocket_goldilocks_collapse.py \\
        --results-dir results/decoy_sets \\
        [--targets AA2AR ADRB2 EGFR]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import RESULTS_DIR, get_all_targets
from src.utils import setup_logging, timer

log = logging.getLogger(__name__)

DIAG_DIR = RESULTS_DIR / "diagnostics" / "pocket_mw_shortcut"
_LOG_FILE = RESULTS_DIR / "logs" / "diag_01_pocket_goldilocks_collapse.log"

# Maps display-name (comparison CSV) → internal name (decoy_sets CSV)
DISPLAY_TO_INTERNAL: dict[str, str] = {
    "PLIF":    "plif",
    "PLEC":    "plec",
    "GRIM":    "grim",
    "Pocket":  "pocket_featurizer",
    "Distance": "distance_matrix",
    "DistChem": "distance_chemistry",
}
INTERNAL_TO_DISPLAY: dict[str, str] = {v: k for k, v in DISPLAY_TO_INTERNAL.items()}


def _load_default_metrics(comparison_csv: Path) -> pd.DataFrame:
    """Load DUDE-Z default metrics from 03_compare_tiers output."""
    if not comparison_csv.exists():
        raise FileNotFoundError(
            f"Default metrics CSV not found: {comparison_csv}\n"
            "Run 03_compare_tiers.py first."
        )
    df = pd.read_csv(comparison_csv)
    # Rename 'variant' → 'featurizer' and map display names to internal names
    df = df.rename(columns={"variant": "featurizer"})
    df["featurizer"] = df["featurizer"].map(
        lambda x: DISPLAY_TO_INTERNAL.get(x, x)
    )
    df["decoy_set"] = "DUDE_Z"
    log.info("Default metrics: %d rows, %d targets, %d featurizers",
             len(df), df["target"].nunique(), df["featurizer"].nunique())
    return df


def _load_goldilocks_metrics(results_dir: Path) -> pd.DataFrame:
    """Load Goldilocks metrics from 04_eval_decoy_sets output."""
    csv = results_dir / "per_target_metrics.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"Decoy set metrics CSV not found: {csv}\n"
            "Run 04_eval_decoy_sets.py first."
        )
    df = pd.read_csv(csv)
    goldilocks = df[df["decoy_set"] == "Goldilocks"].copy()
    if goldilocks.empty:
        raise ValueError(
            "No 'Goldilocks' rows found in per_target_metrics.csv. "
            "Re-run 04_eval_decoy_sets.py with --decoy-sets Goldilocks."
        )
    log.info("Goldilocks metrics: %d rows, %d targets, %d featurizers",
             len(goldilocks), goldilocks["target"].nunique(),
             goldilocks["featurizer"].nunique())
    return goldilocks


def compute_deltas(
    default_df: pd.DataFrame,
    goldilocks_df: pd.DataFrame,
    targets: list[str],
) -> pd.DataFrame:
    """
    Join default and Goldilocks metrics on (featurizer, target) and compute
    ΔBEDROC and ΔPR-AUC per row.
    """
    # Restrict to requested targets
    default_df = default_df[default_df["target"].isin(targets)]
    goldilocks_df = goldilocks_df[goldilocks_df["target"].isin(targets)]

    merged = default_df.merge(
        goldilocks_df,
        on=["featurizer", "target"],
        suffixes=("_default", "_goldilocks"),
    )
    if merged.empty:
        raise ValueError(
            "No rows after joining default and Goldilocks metrics. "
            "Check that featurizer names align between the two CSVs."
        )

    merged["delta_bedroc"] = merged["bedroc_mean_default"] - merged["bedroc_mean_goldilocks"]
    merged["delta_pr_auc"] = merged["pr_auc_mean_default"] - merged["pr_auc_mean_goldilocks"]

    out = merged[[
        "target", "featurizer",
        "bedroc_mean_default", "bedroc_mean_goldilocks", "delta_bedroc",
        "pr_auc_mean_default", "pr_auc_mean_goldilocks", "delta_pr_auc",
    ]].rename(columns={
        "bedroc_mean_default": "bedroc_default",
        "bedroc_mean_goldilocks": "bedroc_goldilocks",
        "pr_auc_mean_default": "pr_auc_default",
        "pr_auc_mean_goldilocks": "pr_auc_goldilocks",
    })
    return out.sort_values(["featurizer", "target"]).reset_index(drop=True)


def _summary_stats(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Compute median/mean/IQR of ΔBEDROC per featurizer, sorted by median descending."""
    rows = []
    for feat, sub in delta_df.groupby("featurizer"):
        d = sub["delta_bedroc"].dropna()
        rows.append({
            "featurizer": feat,
            "n_targets": len(d),
            "median_delta_bedroc": d.median(),
            "mean_delta_bedroc": d.mean(),
            "iqr_delta_bedroc": d.quantile(0.75) - d.quantile(0.25),
            "median_delta_pr_auc": sub["delta_pr_auc"].dropna().median(),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("median_delta_bedroc", ascending=False)
        .reset_index(drop=True)
    )


def _pocket_plec_ratios(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Per-target ratio: pocket delta_bedroc / plec delta_bedroc."""
    pocket = delta_df[delta_df["featurizer"] == "pocket_featurizer"].set_index("target")
    plec   = delta_df[delta_df["featurizer"] == "plec"].set_index("target")
    common = pocket.index.intersection(plec.index)
    rows = []
    for t in common:
        pocket_d = pocket.loc[t, "delta_bedroc"]
        plec_d   = plec.loc[t, "delta_bedroc"]
        ratio    = pocket_d / plec_d if plec_d != 0 else float("nan")
        rows.append({"target": t, "pocket_delta": pocket_d,
                     "plec_delta": plec_d, "ratio_pocket_plec": ratio})
    return pd.DataFrame(rows).sort_values("ratio_pocket_plec", ascending=False)


def print_summary(delta_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("TEST 1 — Goldilocks Collapse Diagnostic")
    print("=" * 72)
    print("\nMedian ΔBEDROC per featurizer (DUDE-Z default − Goldilocks):")
    print(summary.to_string(index=False))

    # Interpretation
    pocket_row = summary[summary["featurizer"] == "pocket_featurizer"]
    plec_row   = summary[summary["featurizer"] == "plec"]
    if not pocket_row.empty and not plec_row.empty:
        pm = pocket_row.iloc[0]["median_delta_bedroc"]
        lm = plec_row.iloc[0]["median_delta_bedroc"]
        ratio = pm / lm if lm != 0 else float("nan")
        print(f"\nInterpretation:")
        print(f"  pocket median ΔBEDROC = {pm:.4f}")
        print(f"  plec   median ΔBEDROC = {lm:.4f}  (control)")
        print(f"  pocket / plec ratio   = {ratio:.2f}")
        if ratio > 2.0:
            print("  ⚠ POCKET BEDROC IS SUBSTANTIALLY INFLATED relative to PLEC.")
            print("  ⚠ Goldilocks collapse is >2× PLEC — strong evidence of property shortcut.")
        elif ratio > 1.2:
            print("  ! Pocket shows moderate Goldilocks collapse relative to PLEC.")
        else:
            print("  ✓ Pocket Goldilocks collapse is comparable to PLEC (control).")

    print("\nPer-target pocket/plec ΔBEDROC ratio:")
    ratios = _pocket_plec_ratios(delta_df)
    print(ratios.to_string(index=False))
    print("=" * 72 + "\n")


def main(
    targets: list[str],
    results_dir: Path,
    comparison_csv: Path,
) -> None:
    setup_logging(log_file=_LOG_FILE)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    with timer("Test 1 — Goldilocks collapse"):
        default_df     = _load_default_metrics(comparison_csv)
        goldilocks_df  = _load_goldilocks_metrics(results_dir)
        delta_df       = compute_deltas(default_df, goldilocks_df, targets)

        out_csv = DIAG_DIR / "test1_goldilocks_delta.csv"
        delta_df.to_csv(out_csv, index=False)
        log.info("Saved ΔBEDROC table: %s (%d rows)", out_csv, len(delta_df))

        summary = _summary_stats(delta_df)
        print_summary(delta_df, summary)

    log.info("Test 1 complete — output: %s", out_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR / "decoy_sets",
        help="Directory produced by 04_eval_decoy_sets.py (contains per_target_metrics.csv)",
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=RESULTS_DIR / "comparison" / "per_target_metrics.csv",
        help="CSV produced by 03_compare_tiers.py (DUDE-Z default metrics)",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=get_all_targets(),
        help="Subset of targets to analyse (default: all 43)",
    )
    args = parser.parse_args()
    main(args.targets, args.results_dir, args.comparison_csv)
