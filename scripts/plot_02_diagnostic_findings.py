"""
Plot 2 — Diagnostic findings summary.

2-row × 2-column grid:
  Panel A (top-left)  : Goldilocks collapse per featurizer
  Panel B (top-right) : Per-target pocket/PLEC ΔBEDROC ratio
  Panel C (bottom-left): Permutation feature importance (15 features)
  Panel D (bottom-right): Verdict summary text card

Data sources (all read-only):
  results/diagnostics/pocket_mw_shortcut/test1_goldilocks_delta.csv
  results/diagnostics/pocket_mw_shortcut/test2_permutation_importance.csv
  results/diagnostics/pocket_mw_shortcut/test2_permutation_importance_aggregated.csv
  results/diagnostics/pocket_mw_shortcut/REPORT.md

Outputs:
  results/figures/summary/plot_02_diagnostic_findings.pdf
  results/figures/summary/plot_02_diagnostic_findings.png

Usage:
    python scripts/plot_02_diagnostic_findings.py \\
        [--diagnostics-dir results/diagnostics/pocket_mw_shortcut] \\
        [--output-dir results/figures/summary]
"""
from __future__ import annotations

import argparse
import logging
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RESULTS_DIR
from src.plotting.palette import (
    FEATURIZER_LABELS, FEATURIZER_COLORS, POCKET_FEATURES, LIGAND_FEATURES,
    FEATURE_COLORS, apply_publication_style,
)
from src.utils import setup_logging, timer

log = logging.getLogger(__name__)

_LOG_FILE = RESULTS_DIR / "logs" / "plot_02_diagnostic_findings.log"
DIAG_DIR  = RESULTS_DIR / "diagnostics" / "pocket_mw_shortcut"

_CLIP_RATIO = 100.0  # visual clip for panel B


# ── Data loading ───────────────────────────────────────────────────────────────

def _require(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input missing: {path}\n"
            "Run the diagnostic scripts first:\n"
            "  diag_01_pocket_goldilocks_collapse.py\n"
            "  diag_02_pocket_permutation_importance.py"
        )
    return pd.read_csv(path)


def _parse_verdict(report_md: Path) -> str:
    """Extract verdict string from REPORT.md bold verdict line."""
    if not report_md.exists():
        return "UNKNOWN"
    text = report_md.read_text()
    m = re.search(r"\*\*Evidence of MolWt shortcut:\s*(\w+)\*\*", text)
    return m.group(1) if m else "UNKNOWN"


def _parse_shortcut_targets(report_md: Path) -> list[str]:
    """Extract shortcut target names from REPORT.md bullet list."""
    if not report_md.exists():
        return []
    text = report_md.read_text()
    return re.findall(r"\*\*([A-Z0-9]+)\*\*:", text)


# ── Panel A: Goldilocks collapse per featurizer ────────────────────────────────

def _panel_a(ax: plt.Axes, t1: pd.DataFrame) -> None:
    summary = (
        t1.groupby("featurizer")["delta_bedroc"]
        .agg(median="median", q25=lambda x: x.quantile(0.25),
             q75=lambda x: x.quantile(0.75), n="count")
        .reset_index()
    )
    summary["iqr_lo"] = summary["median"] - summary["q25"]
    summary["iqr_hi"] = summary["q75"]  - summary["median"]
    summary = summary.sort_values("median", ascending=True)  # worst at bottom

    plec_median = float(
        summary.loc[summary["featurizer"] == "plec", "median"].iloc[0]
    ) if "plec" in summary["featurizer"].values else None

    # Color by shortcut severity (greener = smaller abs delta)
    # Map delta to [0,1]: delta_bedroc is ≤ 0 (default ≥ goldilocks).
    # Use a diverging colormap: green (0 delta) → red (most negative)
    deltas = summary["median"].values
    max_abs = max(abs(deltas.min()), 1e-6)
    norm_vals = np.clip(abs(deltas) / max_abs, 0, 1)
    cmap = plt.cm.RdYlGn_r  # green=0 risk, red=high risk
    colors = [cmap(v) for v in norm_vals]

    bars = ax.barh(
        y=range(len(summary)),
        width=summary["median"].values,
        xerr=[summary["iqr_lo"].values, summary["iqr_hi"].values],
        error_kw=dict(ecolor="#444444", capsize=3, lw=0.8),
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    # Annotate n_targets at end of each bar
    for i, (_, row) in enumerate(summary.iterrows()):
        x_pos = min(row["median"] - row["iqr_lo"], row["median"]) - 0.005
        ax.text(x_pos, i, f"n={int(row['n'])}", ha="right", va="center",
                fontsize=7.5, color="#333333")

    # PLEC control line
    if plec_median is not None:
        ax.axvline(plec_median, color="#2266aa", linestyle="--",
                   linewidth=1.1, zorder=5, label=f"PLEC (control) {plec_median:.3f}")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(
        [FEATURIZER_LABELS.get(f, f) for f in summary["featurizer"]],
        fontsize=9,
    )
    ax.axvline(0, color="black", linewidth=0.6, zorder=3)
    ax.set_xlabel(
        "Median ΔBEDROC (default − Goldilocks)\nMore negative = more shortcut-driven",
        fontsize=10,
    )
    ax.set_title("A  Goldilocks collapse — property-shortcut diagnostic",
                 fontsize=12, fontweight="bold", loc="left")


# ── Panel B: per-target pocket/PLEC ratio ─────────────────────────────────────

def _panel_b(ax: plt.Axes, t1: pd.DataFrame) -> None:
    pocket = t1[t1["featurizer"] == "pocket_featurizer"].set_index("target")["delta_bedroc"]
    plec   = t1[t1["featurizer"] == "plec"].set_index("target")["delta_bedroc"]
    common = pocket.index.intersection(plec.index)

    if common.empty:
        ax.text(0.5, 0.5, "No pocket/plec overlap", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.axis("off")
        return

    ratios = pd.Series(
        {t: (pocket[t] / plec[t] if plec[t] != 0 else np.nan) for t in common},
        name="ratio",
    ).dropna().sort_values(ascending=False)

    true_vals = ratios.copy()
    clipped   = ratios.clip(lower=-_CLIP_RATIO, upper=_CLIP_RATIO)
    was_clipped = (true_vals != clipped).values

    # Colors: red = STRONG (>2), orange = MODERATE (1-2), gray = low (<1 or neg)
    def _color(r: float) -> str:
        if r > 2.0:   return "#d62728"    # strong shortcut — red
        if r > 1.0:   return "#ff7f0e"    # moderate — orange
        return "#aaaaaa"                  # low / negative — gray

    colors = [_color(r) for r in clipped.values]
    bars = ax.barh(
        y=range(len(clipped)),
        width=clipped.values,
        color=colors,
        edgecolor="none",
        height=0.75,
    )

    # Annotate clipped bars with true value
    for i, (target, clipped_val, true_val, was_c) in enumerate(
        zip(clipped.index, clipped.values, true_vals.values, was_clipped)
    ):
        if was_c:
            x_text = clipped_val + 1.5
            ax.text(x_text, i, f"{true_val:.1f}", ha="left", va="center",
                    fontsize=6.5, color="#444444")

    ax.axvline(2.0, color="#d62728", linestyle="--", linewidth=1.0, zorder=5,
               label="STRONG threshold (×2)")
    ax.axvline(1.0, color="#ff7f0e", linestyle=":",  linewidth=1.0, zorder=5,
               label="Parity (×1)")
    ax.axvline(0.0, color="black",   linewidth=0.6,  zorder=3)

    ax.set_yticks(range(len(clipped)))
    ax.set_yticklabels(clipped.index, fontsize=7)
    ax.set_xlabel("pocket_featurizer ΔBEDROC / PLEC ΔBEDROC ratio", fontsize=10)
    ax.set_title("B  Per-target shortcut severity (pocket vs PLEC Goldilocks collapse)",
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    # Legend patches
    patches = [
        mpatches.Patch(color="#d62728", label="STRONG (ratio > 2)"),
        mpatches.Patch(color="#ff7f0e", label="MODERATE (1–2)"),
        mpatches.Patch(color="#aaaaaa", label="Low / negative"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="lower right", framealpha=0.9)


# ── Panel C: permutation feature importance ────────────────────────────────────

def _panel_c(ax: plt.Axes, agg: pd.DataFrame) -> None:
    agg = agg.sort_values("median_bedroc_drop", ascending=True).copy()
    agg["iqr_lo"] = (agg["iqr_bedroc_drop"] / 2).clip(lower=0)
    agg["iqr_hi"] = (agg["iqr_bedroc_drop"] / 2).clip(lower=0)

    colors = [
        FEATURE_COLORS["pocket"] if fn in POCKET_FEATURES else FEATURE_COLORS["ligand"]
        for fn in agg["feature_name"]
    ]

    ax.barh(
        y=range(len(agg)),
        width=agg["median_bedroc_drop"].values,
        xerr=[agg["iqr_lo"].values, agg["iqr_hi"].values],
        error_kw=dict(ecolor="#555555", capsize=3, lw=0.8),
        color=colors,
        edgecolor="white",
        linewidth=0.4,
    )

    # Annotate the constant pocket features prominently
    for i, (_, row) in enumerate(agg.iterrows()):
        if row["feature_name"] in POCKET_FEATURES:
            ax.text(
                0.003, i,
                "0.000 (constant within target)",
                ha="left", va="center", fontsize=7.5,
                color="#888888", style="italic",
            )

    ax.set_yticks(range(len(agg)))
    ax.set_yticklabels(agg["feature_name"], fontsize=9)
    ax.set_xlabel("Median BEDROC drop under feature permutation", fontsize=10)
    ax.set_title("C  Pocket_featurizer: feature importance (43 targets)",
                 fontsize=12, fontweight="bold", loc="left")

    # Legend
    legend_handles = [
        mpatches.Patch(color=FEATURE_COLORS["ligand"], label="Ligand descriptor"),
        mpatches.Patch(color=FEATURE_COLORS["pocket"], label="Pocket composition"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right", framealpha=0.9)


# ── Panel D: verdict text card ─────────────────────────────────────────────────

def _panel_d(
    ax: plt.Axes,
    verdict: str,
    agg: pd.DataFrame,
    t1: pd.DataFrame,
    shortcut_targets: list[str],
) -> None:
    ax.axis("off")

    # Background box
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.02",
        linewidth=1.5,
        edgecolor="#444444",
        facecolor="#f8f8f8",
        transform=ax.transAxes,
        zorder=0,
    ))

    # Verdict color
    verdict_colors = {"STRONG": "#d62728", "MODERATE": "#ff7f0e",
                      "WEAK": "#2ca02c", "NONE": "#1f77b4", "UNKNOWN": "#888888"}
    vc = verdict_colors.get(verdict, "#444444")

    # Pocket/PLEC ratio
    pocket_d = t1[t1["featurizer"] == "pocket_featurizer"]["delta_bedroc"]
    plec_d   = t1[t1["featurizer"] == "plec"]["delta_bedroc"]
    ratio_str = "N/A"
    if not pocket_d.empty and not plec_d.empty and plec_d.median() != 0:
        ratio_str = f"{pocket_d.median() / plec_d.median():.2f}×"

    # Top features
    top5 = agg.sort_values("median_bedroc_drop", ascending=False).head(5)
    top5_lines = "\n".join(
        f"  {i+1}. {row['feature_name']}: {row['median_bedroc_drop']:.4f}"
        for i, (_, row) in enumerate(top5.iterrows())
    )

    n_pocket_zero = int((agg[agg["feature_name"].isin(POCKET_FEATURES)]["median_bedroc_drop"] == 0).sum())
    n_pocket_total = len(POCKET_FEATURES)

    text = (
        f"VERDICT\n"
        f"{'─' * 38}\n"
        f"Evidence of MolWt shortcut:  {verdict}\n\n"
        f"Goldilocks collapse ratio\n"
        f"  pocket / PLEC = {ratio_str}\n\n"
        f"Top-5 informative features\n"
        f"  (by median BEDROC drop)\n"
        f"{top5_lines}\n\n"
        f"Pocket feature contribution\n"
        f"  {n_pocket_zero}/{n_pocket_total} pocket features: 0.000 BEDROC\n"
        f"  drop on all targets\n"
        f"  (structurally constant within target)\n\n"
        f"Shortcut targets ({len(shortcut_targets)})\n"
        + (("\n".join(f"  {t}" for t in shortcut_targets[:8])
            + ("\n  ..." if len(shortcut_targets) > 8 else ""))
           if shortcut_targets else "  none") + "\n\n"
        f"Recommendation\n"
        f"  Exclude from Tier 3 or relabel\n"
        f"  as 'ligand_descriptors_baseline'"
    )

    ax.text(
        0.08, 0.95,
        text,
        transform=ax.transAxes,
        fontsize=9.5,
        va="top", ha="left",
        family="monospace",
        linespacing=1.45,
        zorder=2,
    )

    # Verdict highlight
    ax.text(
        0.08, 0.965,
        f"Evidence of MolWt shortcut:  {verdict}",
        transform=ax.transAxes,
        fontsize=10.5,
        va="top", ha="left",
        family="monospace",
        color=vc,
        fontweight="bold",
        zorder=3,
    )

    ax.set_title("D  Shortcut verdict summary",
                 fontsize=12, fontweight="bold", loc="left")


# ── Figure assembly ────────────────────────────────────────────────────────────

def make_plot(diagnostics_dir: Path, output_dir: Path) -> None:
    apply_publication_style()

    t1    = _require(diagnostics_dir / "test1_goldilocks_delta.csv")
    t2    = _require(diagnostics_dir / "test2_permutation_importance.csv")
    agg   = _require(diagnostics_dir / "test2_permutation_importance_aggregated.csv")

    report_md       = diagnostics_dir / "REPORT.md"
    verdict         = _parse_verdict(report_md)
    shortcut_targets = _parse_shortcut_targets(report_md)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    _panel_a(axes[0, 0], t1)
    _panel_b(axes[0, 1], t1)
    _panel_c(axes[1, 0], agg)
    _panel_d(axes[1, 1], verdict, agg, t1, shortcut_targets)

    fig.suptitle(
        "Pocket_featurizer diagnostic summary: property-shortcut evidence",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "plot_02_diagnostic_findings"
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", pdf_path)
    log.info("Saved: %s", png_path)
    print(f"  → {pdf_path}")
    print(f"  → {png_path}")

    # Summary stats
    print(f"\nPlot 2 summary stats:")
    print(f"  Verdict: {verdict}")
    print(f"  Test 1: {t1['featurizer'].nunique()} featurizers × {t1['target'].nunique()} targets")
    print(f"  Test 2: {t2['target'].nunique()} targets × {t2['feature_name'].nunique()} features")
    print(f"  Shortcut targets ({len(shortcut_targets)}): {shortcut_targets}")
    pocket_zero = agg[agg["feature_name"].isin(POCKET_FEATURES)]["median_bedroc_drop"] == 0
    print(f"  Pocket features with 0.000 drop: {pocket_zero.sum()}/{len(POCKET_FEATURES)}")
    pocket_d = t1[t1["featurizer"] == "pocket_featurizer"]["delta_bedroc"]
    plec_d   = t1[t1["featurizer"] == "plec"]["delta_bedroc"]
    if not pocket_d.empty and not plec_d.empty and plec_d.median() != 0:
        print(f"  pocket/plec Goldilocks ratio: {pocket_d.median() / plec_d.median():.2f}×")


def main(diagnostics_dir: Path, output_dir: Path) -> None:
    setup_logging(log_file=_LOG_FILE)
    with timer("Plot 2 — diagnostic findings"):
        make_plot(diagnostics_dir, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=DIAG_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "figures" / "summary",
    )
    args = parser.parse_args()
    main(args.diagnostics_dir, args.output_dir)
