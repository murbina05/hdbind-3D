#!/usr/bin/env python
"""Phase 6 step (e) — aggregate metrics.json into tidy CSV + grouped bar chart.

Reads every metrics.json under outputs/phase6_eval/<config>/<decoy_set>/fold_K_seed_S/
and produces:
  - outputs/phase6_eval/_aggregated/tidy_metrics.csv
        long format: target, decoy_set, feature_config, classifier, fold, seed, metric, value
  - outputs/phase6_eval/_aggregated/summary_by_config.csv
        per-(config, decoy_set, classifier): mean and std of each metric across folds/seeds
  - outputs/phase6_eval/_aggregated/grouped_bar.png
        2 rows (classifiers) × 3 cols (BEDROC, ROC-AUC, PR-AUC); each subplot has
        feature_configs on x with decoy_set-grouped bars. Empty decoy_sets are
        skipped gracefully — Scenario A produces only DUDE-Z bars; the figure
        is incrementally extensible to Extrema/Goldilocks post-Threadripper.

Scenario A: only DUDE-Z bars will be visible until Extrema/Goldilocks runs
land in the same directory tree.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HDBIND = Path("/home/maurbina/hdbind-3D")
EVAL_ROOT = _HDBIND / "outputs" / "phase6_eval"
OUT_DIR = EVAL_ROOT / "_aggregated"

CONFIG_ORDER = [
    "PLEC", "EGNN-M2-pre", "EGNN-M2-post",
    "FLOWR-pre", "FLOWR-post",
    "PLEC+FLOWR-post", "EGNN-M2-pre+FLOWR-post",
]
DECOY_ORDER = ["DUDE-Z", "Extrema", "Goldilocks"]
DECOY_COLORS = {"DUDE-Z": "#1f77b4", "Extrema": "#ff7f0e", "Goldilocks": "#2ca02c"}
CLASSIFIERS = ["linear_probe", "mlp_2layer"]
METRICS = ["bedroc", "roc_auc", "pr_auc"]
METRIC_LABELS = {"bedroc": "BEDROC (α=20)", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}


def build_tidy(eval_root: Path) -> pd.DataFrame:
    """Walk metrics.json files and emit long-format rows."""
    rows: list[dict] = []
    for mp in sorted(eval_root.glob("*/*/fold_*_seed_*/metrics.json")):
        m = json.loads(mp.read_text())
        # path: <eval_root>/<config>/<decoy_set>/fold_K_seed_S/metrics.json
        feature_config = mp.parents[2].name
        decoy_set = mp.parents[1].name
        fold = int(m["fold"])
        seed = int(m["seed"])
        for clf, clf_data in m["classifiers"].items():
            for target, tmet in clf_data["per_target"].items():
                for metric in ("roc_auc", "pr_auc", "bedroc", "ef_1pct", "ef_5pct"):
                    v = tmet.get(metric, float("nan"))
                    rows.append({
                        "target": target,
                        "decoy_set": decoy_set,
                        "feature_config": feature_config,
                        "classifier": clf,
                        "fold": fold,
                        "seed": seed,
                        "metric": metric,
                        "value": float(v),
                        "n_poses": tmet.get("n_poses"),
                        "n_actives": tmet.get("n_actives"),
                    })
    return pd.DataFrame(rows)


def build_summary(tidy: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across folds, seeds, and targets within each (config, decoy_set, classifier, metric)."""
    g = (tidy
         .dropna(subset=["value"])
         .groupby(["feature_config", "decoy_set", "classifier", "metric"])
         ["value"]
         .agg(["mean", "std", "count"])
         .reset_index())
    return g


def render_bar_chart(summary: pd.DataFrame, out_png: Path) -> None:
    # Use only configs we actually have data for; preserve canonical order
    have_configs = set(summary["feature_config"].unique())
    configs = [c for c in CONFIG_ORDER if c in have_configs]
    have_decoys = set(summary["decoy_set"].unique())
    decoys = [d for d in DECOY_ORDER if d in have_decoys]

    n_classifiers = len(CLASSIFIERS)
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(n_classifiers, n_metrics, figsize=(5.5 * n_metrics, 4.5 * n_classifiers))
    if n_classifiers == 1:
        axes = np.array([axes])
    if n_metrics == 1:
        axes = axes.reshape(-1, 1)

    x_idx = np.arange(len(configs))
    bar_width = 0.8 / max(len(decoys), 1)

    for ri, clf in enumerate(CLASSIFIERS):
        for ci, metric in enumerate(METRICS):
            ax = axes[ri, ci]
            for di, decoy in enumerate(decoys):
                means = []
                stds = []
                for cfg in configs:
                    row = summary[(summary["feature_config"] == cfg)
                                  & (summary["decoy_set"] == decoy)
                                  & (summary["classifier"] == clf)
                                  & (summary["metric"] == metric)]
                    if len(row) == 1:
                        means.append(float(row["mean"].values[0]))
                        stds.append(float(row["std"].values[0]) if not np.isnan(row["std"].values[0]) else 0.0)
                    else:
                        means.append(np.nan)
                        stds.append(0.0)
                positions = x_idx + (di - (len(decoys) - 1) / 2) * bar_width
                ax.bar(positions, means, bar_width, yerr=stds, capsize=2,
                       label=decoy, color=DECOY_COLORS.get(decoy, None),
                       edgecolor="black", linewidth=0.4, alpha=0.9)
            ax.set_xticks(x_idx)
            ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
            ax.set_title(f"{clf} — {METRIC_LABELS[metric]}", fontsize=11)
            ax.grid(axis="y", alpha=0.3)
            if metric in ("roc_auc",):
                ax.set_ylim(0.4, 1.0)
            if ri == 0 and ci == n_metrics - 1:
                ax.legend(title="decoy set", fontsize=9)

    fig.suptitle(
        "Phase 6 representation benchmark — fixed classifier on frozen features\n"
        f"(decoys present: {', '.join(decoys)}; per-target metrics aggregated over folds × seeds)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", default=str(EVAL_ROOT))
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"scanning {eval_root}/*/*/fold_*_seed_*/metrics.json ...")
    tidy = build_tidy(eval_root)
    print(f"  loaded {len(tidy)} rows from "
          f"{tidy.groupby(['feature_config','decoy_set','classifier','fold','seed']).ngroups} runs")
    tidy_csv = out_dir / "tidy_metrics.csv"
    tidy.to_csv(tidy_csv, index=False)
    print(f"wrote {tidy_csv}")

    summary = build_summary(tidy)
    summary_csv = out_dir / "summary_by_config.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"wrote {summary_csv}")

    out_png = out_dir / "grouped_bar.png"
    render_bar_chart(summary, out_png)

    print()
    print("=== Headline summary (BEDROC, mean across folds × seeds × targets, both classifiers) ===")
    pivot = (summary
             [summary["metric"] == "bedroc"]
             .pivot_table(index="feature_config", columns=["decoy_set", "classifier"],
                          values="mean", aggfunc="first"))
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
