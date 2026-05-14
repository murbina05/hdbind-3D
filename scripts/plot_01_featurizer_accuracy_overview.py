"""
Plot 1 — Featurizer accuracy overview.

4-row × 3-column grid of horizontal boxplots + strip plots.
Rows  : BEDROC, PR-AUC, ROC-AUC, MCC
Columns: DUDE-Z default, Extrema, Goldilocks

Featurizers sorted by median BEDROC on DUDE-Z default (consistent across all
subplots). Per-target dots overlaid on each box.

Outputs (both vector and raster):
  results/figures/summary/plot_01_featurizer_accuracy_overview.pdf
  results/figures/summary/plot_01_featurizer_accuracy_overview.png

Usage:
    python scripts/plot_01_featurizer_accuracy_overview.py \\
        [--results-dir results/decoy_sets] \\
        [--comparison-csv results/comparison/per_target_metrics.csv] \\
        [--output-dir results/figures/summary]
"""
from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from config import RESULTS_DIR
from src.plotting.palette import (
    FEATURIZER_COLORS, FEATURIZER_LABELS, DISPLAY_TO_INTERNAL,
    apply_publication_style,
)
from src.utils import setup_logging, timer

log = logging.getLogger(__name__)

_LOG_FILE = RESULTS_DIR / "logs" / "plot_01_featurizer_accuracy_overview.log"

METRICS: list[tuple[str, str]] = [
    ("bedroc_mean",   "BEDROC"),
    ("pr_auc_mean",   "PR-AUC"),
    ("roc_auc_mean",  "ROC-AUC"),
    ("mcc_mean",      "MCC"),
]
DECOY_SETS: list[str] = ["DUDE-Z default", "Extrema", "Goldilocks"]


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_comparison(comparison_csv: Path) -> pd.DataFrame:
    if not comparison_csv.exists():
        raise FileNotFoundError(
            f"Default metrics CSV not found: {comparison_csv}\n"
            "Run scripts/03_compare_tiers.py first."
        )
    df = pd.read_csv(comparison_csv)
    df = df.rename(columns={"variant": "featurizer"})
    df["featurizer"] = df["featurizer"].map(lambda x: DISPLAY_TO_INTERNAL.get(x, x))
    df["decoy_set"] = "DUDE-Z default"
    return df


def _load_decoy_sets(results_dir: Path) -> pd.DataFrame:
    csv = results_dir / "per_target_metrics.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"Decoy set metrics not found: {csv}\n"
            "Run scripts/04_eval_decoy_sets.py first."
        )
    return pd.read_csv(csv)


def build_unified_df(comparison_csv: Path, results_dir: Path) -> pd.DataFrame:
    """Merge default + Extrema/Goldilocks metrics into one tidy DataFrame."""
    default_df = _load_comparison(comparison_csv)
    decoy_df   = _load_decoy_sets(results_dir)
    combined   = pd.concat([default_df, decoy_df], ignore_index=True, sort=False)

    # Ensure featurizer column is consistent (internal names)
    missing_feat = set(combined["featurizer"].unique()) - set(FEATURIZER_LABELS)
    if missing_feat:
        log.warning("Unknown featurizers in data: %s", missing_feat)

    log.info(
        "Combined: %d rows | decoy_sets=%s | featurizers=%s | targets=%d",
        len(combined),
        sorted(combined["decoy_set"].unique()),
        sorted(combined["featurizer"].unique()),
        combined["target"].nunique(),
    )
    return combined


def _featurizer_order(df: pd.DataFrame) -> list[str]:
    """Sort by median BEDROC on DUDE-Z default, descending."""
    sub = df[df["decoy_set"] == "DUDE-Z default"]
    order = (
        sub.groupby("featurizer")["bedroc_mean"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    # Append any featurizer not in default (shouldn't happen, but be safe)
    all_feats = df["featurizer"].unique().tolist()
    for f in all_feats:
        if f not in order:
            order.append(f)
    return order


def _active_fraction(df: pd.DataFrame, decoy_set: str) -> float:
    """Estimate actives/(actives+decoys) from EF and hit_rate columns."""
    sub = df[df["decoy_set"] == decoy_set]
    if sub.empty or "ef_1pct_mean" not in sub.columns:
        return 0.017
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ratios = np.where(
            sub["ef_1pct_mean"] > 0,
            sub["hit_rate_1pct_mean"] / sub["ef_1pct_mean"],
            np.nan,
        )
    v = float(np.nanmedian(ratios))
    return v if np.isfinite(v) else 0.017


# ── Subplot drawing ────────────────────────────────────────────────────────────

def _draw_cell(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric_col: str,
    feat_order: list[str],
    feat_colors: dict[str, tuple],
    show_labels: bool,
    ref_line: float | None,
    ref_label: str | None,
) -> None:
    """Draw one boxplot + strip cell."""
    if data.empty or metric_col not in data.columns:
        ax.text(
            0.5, 0.5, "data not available",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=10, color="gray",
        )
        ax.axis("off")
        return

    # Only keep featurizers present in this decoy_set's data
    present_fts = data["featurizer"].unique().tolist()
    order = [f for f in feat_order if f in present_fts]
    palette = {f: feat_colors.get(f, "#888888") for f in order}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.boxplot(
            data=data,
            x=metric_col,
            y="featurizer",
            order=order,
            palette=palette,
            ax=ax,
            width=0.55,
            showfliers=False,
            linewidth=0.8,
            boxprops=dict(alpha=0.75),
        )
        sns.stripplot(
            data=data,
            x=metric_col,
            y="featurizer",
            order=order,
            palette=palette,
            ax=ax,
            size=2.8,
            alpha=0.55,
            jitter=True,
            zorder=3,
        )

    # Reference line
    if ref_line is not None:
        ax.axvline(ref_line, color="black", linestyle="--", linewidth=0.9, zorder=4)
        if ref_label:
            ax.text(
                ref_line + 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                len(order) - 0.5,
                ref_label,
                fontsize=7.5,
                color="black",
                va="bottom",
            )

    # Annotate small-n featurizers
    n_per_feat = data.groupby("featurizer")[metric_col].count()
    for fi, feat in enumerate(order):
        n = n_per_feat.get(feat, 0)
        if n < 10:
            ax.text(
                ax.get_xlim()[1], fi, f"n={n}",
                ha="right", va="center", fontsize=7.5, color="#cc2222",
            )

    # Y-axis labels (fix FixedLocator requirement)
    ax.set_yticks(range(len(order)))
    if show_labels:
        ax.set_yticklabels(
            [FEATURIZER_LABELS.get(f, f) for f in order],
            fontsize=9,
        )
    else:
        ax.set_yticklabels([""] * len(order))
    ax.set_ylabel("")
    ax.set_xlabel("")


# ── Figure assembly ────────────────────────────────────────────────────────────

def make_plot(df: pd.DataFrame, output_dir: Path) -> None:
    apply_publication_style()

    feat_order  = _featurizer_order(df)
    feat_colors = FEATURIZER_COLORS

    active_fracs: dict[str, float] = {
        ds: _active_fraction(df, ds) for ds in DECOY_SETS
    }
    log.info("Active fractions: %s", {k: f"{v:.4f}" for k, v in active_fracs.items()})

    n_rows = len(METRICS)
    n_cols = len(DECOY_SETS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(16, 12),
        sharey="row",
    )

    fig.suptitle(
        "Featurizer accuracy across 43 DUDE-Z targets and three decoy sets",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for col_i, ds_name in enumerate(DECOY_SETS):
        # Column title
        axes[0, col_i].set_title(ds_name, fontsize=12, fontweight="bold", pad=6)

        sub_ds = df[df["decoy_set"] == ds_name].copy()

        for row_i, (metric_col, metric_label) in enumerate(METRICS):
            ax = axes[row_i, col_i]

            # Reference line config
            if metric_col == "roc_auc_mean":
                ref_line, ref_label = 0.5, "random (0.5)"
            elif metric_col == "pr_auc_mean":
                frac = active_fracs.get(ds_name, 0.017)
                ref_line  = frac
                ref_label = f"random ({frac:.4f})"
            elif metric_col == "mcc_mean":
                ref_line, ref_label = 0.0, "random (0)"
            else:
                ref_line, ref_label = None, None

            _draw_cell(
                ax=ax,
                data=sub_ds,
                metric_col=metric_col,
                feat_order=feat_order,
                feat_colors=feat_colors,
                show_labels=(col_i == 0),
                ref_line=ref_line,
                ref_label=ref_label,
            )

            # Row label on left edge
            if col_i == 0:
                ax.set_ylabel(metric_label, fontsize=11, fontweight="bold", labelpad=8)

            # X-axis label on bottom row only
            if row_i == n_rows - 1:
                ax.set_xlabel(metric_label, fontsize=10)
            else:
                ax.tick_params(labelbottom=False)

    fig.tight_layout(rect=[0, 0.045, 1, 0.97])

    # Figure-level legend — one colored patch per featurizer, bottom centre
    legend_handles = [
        mpatches.Patch(
            facecolor=feat_colors.get(f, "#888888"),
            edgecolor="white",
            linewidth=0.5,
            label=FEATURIZER_LABELS.get(f, f),
            alpha=0.85,
        )
        for f in feat_order
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(feat_order),
        fontsize=10,
        title="Featurizer",
        title_fontsize=10,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 0.0),
    )

    _save(fig, output_dir, "plot_01_featurizer_accuracy_overview")

    # Summary stats to stdout
    n_feats = df["featurizer"].nunique()
    n_tgt   = df["target"].nunique()
    print(
        f"\nPlot 1: {n_feats} featurizers × {len(DECOY_SETS)} decoy sets × "
        f"{len(METRICS)} metrics, up to {n_tgt} targets each"
    )
    for ds in DECOY_SETS:
        n = df[df["decoy_set"] == ds]["target"].nunique()
        feats = sorted(df[df["decoy_set"] == ds]["featurizer"].unique())
        print(f"  {ds}: {n} targets, featurizers={feats}")
    print(f"  Featurizer order (by median BEDROC on default): {feat_order}")


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", pdf_path)
    log.info("Saved: %s", png_path)
    print(f"  → {pdf_path}")
    print(f"  → {png_path}")


def main(results_dir: Path, comparison_csv: Path, output_dir: Path) -> None:
    setup_logging(log_file=_LOG_FILE)

    with timer("Plot 1 — featurizer accuracy overview"):
        df = build_unified_df(comparison_csv, results_dir)
        make_plot(df, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR / "decoy_sets",
        help="Directory with 04_eval_decoy_sets.py output (contains per_target_metrics.csv)",
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=RESULTS_DIR / "comparison" / "per_target_metrics.csv",
        help="CSV from 03_compare_tiers.py (DUDE-Z default metrics)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "figures" / "summary",
    )
    args = parser.parse_args()
    main(args.results_dir, args.comparison_csv, args.output_dir)
