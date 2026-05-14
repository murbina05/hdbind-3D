"""
Combined shortcut report — merges Test 1 and Test 2 outputs into a markdown
summary at results/diagnostics/pocket_mw_shortcut/REPORT.md.

Reads:
  results/diagnostics/pocket_mw_shortcut/test1_goldilocks_delta.csv
  results/diagnostics/pocket_mw_shortcut/test2_permutation_importance.csv
  results/diagnostics/pocket_mw_shortcut/test2_permutation_importance_aggregated.csv

Outputs:
  results/diagnostics/pocket_mw_shortcut/REPORT.md

Usage:
    python scripts/diag_03_pocket_shortcut_report.py
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from config import RESULTS_DIR
from src.utils import setup_logging, timer

log = logging.getLogger(__name__)

DIAG_DIR = RESULTS_DIR / "diagnostics" / "pocket_mw_shortcut"
_LOG_FILE = RESULTS_DIR / "logs" / "diag_03_pocket_shortcut_report.log"

TEST1_CSV = DIAG_DIR / "test1_goldilocks_delta.csv"
TEST2_CSV = DIAG_DIR / "test2_permutation_importance.csv"
TEST2_AGG_CSV = DIAG_DIR / "test2_permutation_importance_aggregated.csv"
REPORT_MD = DIAG_DIR / "REPORT.md"


# ── Verdict logic ─────────────────────────────────────────────────────────────

def _molwt_rank(agg_df: pd.DataFrame) -> int:
    """1-based rank of MolWt by median BEDROC drop (lower rank = higher drop)."""
    ranked = agg_df.sort_values("median_bedroc_drop", ascending=False).reset_index(drop=True)
    mask = ranked["feature_name"] == "MolWt"
    if not mask.any():
        return 999
    return int(ranked[mask].index[0]) + 1  # 1-based


def _pocket_plec_delta_ratio(t1_df: pd.DataFrame) -> float:
    """
    Ratio of pocket's median ΔBEDROC to PLEC's median ΔBEDROC.
    Returns nan if either featurizer has no data.
    """
    pocket = t1_df[t1_df["featurizer"] == "pocket_featurizer"]["delta_bedroc"]
    plec   = t1_df[t1_df["featurizer"] == "plec"]["delta_bedroc"]
    if pocket.empty or plec.empty or plec.median() == 0:
        return float("nan")
    return float(pocket.median() / plec.median())


def determine_verdict(
    molwt_rank: int,
    ratio: float,
) -> str:
    """
    STRONG  : MolWt is top-ranked AND pocket Goldilocks ΔBEDROC > 2× PLEC's
    MODERATE: one of the two conditions holds
    WEAK    : MolWt is in top 3 but Goldilocks delta comparable to PLEC
    NONE    : importance is distributed AND Goldilocks delta matches PLEC
    """
    molwt_top1 = molwt_rank == 1
    molwt_top3 = molwt_rank <= 3
    ratio_high = (not pd.isna(ratio)) and ratio > 2.0

    if molwt_top1 and ratio_high:
        return "STRONG"
    if molwt_top1 or ratio_high:
        return "MODERATE"
    if molwt_top3:
        return "WEAK"
    return "NONE"


# ── Shortcut targets ──────────────────────────────────────────────────────────

def find_shortcut_targets(
    t1_df: pd.DataFrame,
    t2_df: pd.DataFrame,
) -> list[str]:
    """
    Union of:
      (A) pocket Goldilocks ΔBEDROC > 2× PLEC's for that specific target
      (B) MolWt mean BEDROC drop > 50% of total BEDROC drop for that target
    """
    # Condition A
    pocket_d = t1_df[t1_df["featurizer"] == "pocket_featurizer"].set_index("target")["delta_bedroc"]
    plec_d   = t1_df[t1_df["featurizer"] == "plec"].set_index("target")["delta_bedroc"]
    common   = pocket_d.index.intersection(plec_d.index)
    cond_a: set[str] = set()
    for t in common:
        if plec_d[t] != 0 and pocket_d[t] / plec_d[t] > 2.0:
            cond_a.add(t)

    # Condition B
    cond_b: set[str] = set()
    for target, sub in t2_df.groupby("target"):
        total = sub["mean_bedroc_drop"].sum()
        if total <= 0:
            continue
        mw_row = sub[sub["feature_name"] == "MolWt"]
        if not mw_row.empty:
            mw_drop = float(mw_row["mean_bedroc_drop"].iloc[0])
            if mw_drop > 0.5 * total:
                cond_b.add(str(target))

    return sorted(cond_a | cond_b)


# ── Markdown formatting ───────────────────────────────────────────────────────

def _md_table(df: pd.DataFrame, cols: list[str], fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    rows   = []
    for _, row in df[cols].iterrows():
        cells = []
        for c in cols:
            v = row[c]
            f = fmt.get(c, "{}")
            try:
                cells.append(f.format(v))
            except (ValueError, TypeError):
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def build_report(
    t1_df: pd.DataFrame,
    t2_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    molwt_rank: int,
    ratio: float,
    verdict: str,
    shortcut_targets: list[str],
) -> str:
    ratio_str = f"{ratio:.2f}" if not pd.isna(ratio) else "N/A"

    # Summary per featurizer for Goldilocks test
    feat_summary = (
        t1_df.groupby("featurizer")["delta_bedroc"]
        .agg(["median", "mean"])
        .rename(columns={"median": "median_delta_bedroc", "mean": "mean_delta_bedroc"})
        .reset_index()
        .sort_values("median_delta_bedroc", ascending=False)
    )

    sections: list[str] = []

    # Title + verdict
    sections.append("# Pocket Featurizer MolWt Shortcut Diagnostic")
    sections.append("")
    sections.append(
        f"**Evidence of MolWt shortcut: {verdict}**"
    )
    sections.append("")
    sections.append("## Verdict criteria")
    sections.append(
        "| Criterion | Value | Threshold |"
        "\n| --- | --- | --- |"
        f"\n| MolWt rank by median BEDROC drop | {molwt_rank} | top-1 (STRONG), top-3 (WEAK) |"
        f"\n| pocket / plec Goldilocks ΔBEDROC ratio | {ratio_str} | > 2× (STRONG) |"
    )
    sections.append("")

    # Test 1 table
    sections.append("## Test 1 — Goldilocks ΔBEDROC by featurizer")
    sections.append("")
    sections.append(
        "ΔBEDROC = BEDROC(DUDE-Z default) − BEDROC(Goldilocks). "
        "Large Δ means performance collapses on property-matched decoys — "
        "evidence of shortcut learning."
    )
    sections.append("")
    sections.append(
        _md_table(
            feat_summary,
            cols=["featurizer", "median_delta_bedroc", "mean_delta_bedroc"],
            fmt={
                "median_delta_bedroc": "{:.4f}",
                "mean_delta_bedroc":   "{:.4f}",
            },
        )
    )
    sections.append("")

    # Test 2 table
    sections.append("## Test 2 — Per-feature median BEDROC drop (permutation importance)")
    sections.append("")
    sections.append(
        "Median BEDROC drop across all targets when each feature is shuffled. "
        "A dominant single feature signals shortcut behaviour."
    )
    sections.append("")
    sections.append(
        _md_table(
            agg_df,
            cols=["feature_name", "median_bedroc_drop", "iqr_bedroc_drop", "n_targets"],
            fmt={
                "median_bedroc_drop": "{:.4f}",
                "iqr_bedroc_drop":    "{:.4f}",
            },
        )
    )
    sections.append("")

    # Shortcut targets
    sections.append("## Targets showing strongest shortcut behaviour")
    sections.append("")
    if shortcut_targets:
        sections.append(
            "Targets where (A) pocket Goldilocks ΔBEDROC > 2× PLEC's "
            "**or** (B) MolWt drop > 50% of total BEDROC drop:"
        )
        sections.append("")
        for t in shortcut_targets:
            pocket_row = t1_df[
                (t1_df["target"] == t) & (t1_df["featurizer"] == "pocket_featurizer")
            ]
            plec_row = t1_df[
                (t1_df["target"] == t) & (t1_df["featurizer"] == "plec")
            ]
            mw_row = t2_df[
                (t2_df["target"] == t) & (t2_df["feature_name"] == "MolWt")
            ]
            pocket_d = pocket_row["delta_bedroc"].iloc[0] if not pocket_row.empty else float("nan")
            plec_d   = plec_row["delta_bedroc"].iloc[0]   if not plec_row.empty   else float("nan")
            mw_drop  = mw_row["mean_bedroc_drop"].iloc[0] if not mw_row.empty     else float("nan")
            sections.append(
                f"- **{t}**: pocket ΔBEDROC={pocket_d:.4f}, "
                f"plec ΔBEDROC={plec_d:.4f}, "
                f"MolWt BEDROC drop={mw_drop:.4f}"
            )
    else:
        sections.append("No targets meet the shortcut criteria.")
    sections.append("")

    # Interpretation
    sections.append("## Interpretation")
    sections.append("")
    if verdict == "STRONG":
        sections.append(
            "Both diagnostic tests agree: the pocket featurizer is learning a "
            "**MolWt shortcut**. MolWt is the top-ranked feature by permutation "
            f"importance and the featurizer's BEDROC collapses {ratio_str}× more than "
            "PLEC's on Goldilocks decoys. The reported DUDE-Z performance is inflated "
            "by property matching, not by genuine pocket shape learning."
        )
    elif verdict == "MODERATE":
        sections.append(
            "One of the two diagnostic tests is positive. Either MolWt dominates "
            "permutation importance **or** the Goldilocks collapse exceeds PLEC's by "
            f">2× (ratio={ratio_str}). Moderate evidence of property shortcut; "
            "investigate the flagged targets before drawing conclusions."
        )
    elif verdict == "WEAK":
        sections.append(
            f"MolWt is in the top-3 features by permutation importance (rank={molwt_rank}) "
            f"but the Goldilocks collapse ratio ({ratio_str}) does not exceed 2×. "
            "Weak evidence of shortcut; importance is partially distributed."
        )
    else:
        sections.append(
            "No shortcut detected. Feature importance is distributed and the "
            "Goldilocks collapse is comparable to PLEC's. The pocket featurizer "
            "appears to learn genuine binding signal."
        )

    return "\n".join(sections) + "\n"


def main() -> None:
    setup_logging(log_file=_LOG_FILE)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    for path in (TEST1_CSV, TEST2_CSV, TEST2_AGG_CSV):
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}\n"
                "Run diag_01 and diag_02 first."
            )

    with timer("Test 3 — shortcut report"):
        t1_df  = pd.read_csv(TEST1_CSV)
        t2_df  = pd.read_csv(TEST2_CSV)
        agg_df = pd.read_csv(TEST2_AGG_CSV)

        molwt_rank = _molwt_rank(agg_df)
        ratio      = _pocket_plec_delta_ratio(t1_df)
        verdict    = determine_verdict(molwt_rank, ratio)
        shortcut_targets = find_shortcut_targets(t1_df, t2_df)

        log.info("MolWt rank=%d  pocket/plec ratio=%.2f  verdict=%s",
                 molwt_rank, ratio if not pd.isna(ratio) else -1, verdict)

        report = build_report(t1_df, t2_df, agg_df, molwt_rank, ratio,
                              verdict, shortcut_targets)

        REPORT_MD.write_text(report)
        log.info("Report saved: %s", REPORT_MD)

        print(f"\nVerdict: Evidence of MolWt shortcut = {verdict}")
        print(f"MolWt rank: {molwt_rank}  |  pocket/plec Goldilocks ratio: "
              f"{'N/A' if pd.isna(ratio) else f'{ratio:.2f}'}")
        print(f"Shortcut targets ({len(shortcut_targets)}): "
              f"{', '.join(shortcut_targets) if shortcut_targets else 'none'}")
        print(f"Full report: {REPORT_MD}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # No required args — reads from fixed DIAG_DIR paths
    parser.parse_args()
    main()
