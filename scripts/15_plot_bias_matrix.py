"""15_plot_bias_matrix.py — produce plots for outputs/15_decoy_bias_matrix/.

- bias_matrix_bars.png — grouped bars of PR-AUC, ROC-AUC, BEDROC, EF@1%
  across 4 decoy backgrounds × 2 training regimes, with per-fold error bars.
- per_target_scatter.png — per-target PR-AUC, m2-groupkfold vs m2-ad-mixed,
  one panel per decoy background.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/maurbina/hdbind-3D")
OUT = ROOT / "outputs" / "15_decoy_bias_matrix" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

# (cell-key, csv-path, fold-metric-prefix, target-metric-cols)
GKF_CELLS = {
    "DUDE-Z":     ROOT / "outputs/07_train_egnn/20260505-000911-1662f0b-m2/per_fold_metrics.csv",
    "Goldilocks": ROOT / "outputs/11_eval_goldilocks/m2-goldilocks-fixed/per_fold_metrics.csv",
    "Extrema":    ROOT / "outputs/13_eval_decoys/m2gkf-bias-matrix-20260508/extrema/per_fold_metrics.csv",
    "AD":         ROOT / "outputs/17_eval_ad_decoys/m2gkf-on-ad-20260508/per_fold_metrics.csv",
}
GKF_PREFIXES = {"DUDE-Z": "val_", "Goldilocks": "gold_", "Extrema": "eval_", "AD": "ad_"}

ADMIX_CELLS = {
    "DUDE-Z":     ROOT / "outputs/13_eval_decoys/m2admix-bias-matrix-20260508/dudez/per_fold_metrics.csv",
    "Goldilocks": ROOT / "outputs/13_eval_decoys/m2admix-bias-matrix-20260508/goldilocks/per_fold_metrics.csv",
    "Extrema":    ROOT / "outputs/13_eval_decoys/m2admix-bias-matrix-20260508/extrema/per_fold_metrics.csv",
    "AD":         ROOT / "outputs/13_eval_decoys/m2admix-bias-matrix-20260508/ad/per_fold_metrics.csv",
}
ADMIX_PREFIX = "eval_"

# Per-target CSVs (only available for cells that ran via 13/17, not the
# DUDE-Z eval-only cell which uses training-time per_fold metrics — we
# reconstruct DUDE-Z per-target from the training-time val_predictions.csv
# only for completeness; for the scatter we'll skip DUDE-Z for m2-groupkfold
# and only compare the 3 cells where both have per-target).
GKF_TARGET = {
    # Goldilocks legacy m2-goldilocks-fixed has no per_target_metrics.csv; skip its scatter.
    "Extrema":    ROOT / "outputs/13_eval_decoys/m2gkf-bias-matrix-20260508/extrema/per_target_metrics.csv",
    "AD":         ROOT / "outputs/17_eval_ad_decoys/m2gkf-on-ad-20260508/per_target_metrics.csv",
}
ADMIX_TARGET = {
    k: ROOT / f"outputs/13_eval_decoys/m2admix-bias-matrix-20260508/{k.lower()}/per_target_metrics.csv"
    for k in ("Extrema", "AD")
}

DECOYS = ["DUDE-Z", "Goldilocks", "Extrema", "AD"]
PLEC_GKF = 0.023
GATE1_THRESHOLD = PLEC_GKF + 0.05

# ── Load fold-level metrics ───────────────────────────────────────────────
def _read_metric(csv_path: Path, prefix: str, name: str) -> tuple[float, float, np.ndarray]:
    df = pd.read_csv(csv_path)
    col = f"{prefix}{name}"
    if col not in df.columns:
        raise KeyError(f"{col} not in {csv_path}")
    vals = df[col].values.astype(float)
    return float(vals.mean()), float(vals.std(ddof=0)), vals

METRICS = [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("bedroc", "BEDROC"), ("ef_1pct", "EF@1%")]
records = []
for decoy in DECOYS:
    for regime, cells, prefixes in (
        ("M2-groupkfold (DUDE-Z trained)", GKF_CELLS, GKF_PREFIXES),
        ("M2-ad-mixed (50/50)",            ADMIX_CELLS, {k: ADMIX_PREFIX for k in ADMIX_CELLS}),
    ):
        for short, _ in METRICS:
            mean, std, vals = _read_metric(cells[decoy], prefixes[decoy], short)
            records.append({
                "decoy": decoy, "regime": regime, "metric": short,
                "mean": mean, "std": std, "vals": vals,
            })
df = pd.DataFrame(records)

# ── Plot 1: grouped bars, 4 metrics × 4 decoys × 2 regimes ────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    "Decoy bias matrix — M2 GroupKFold vs M2 AD-mixed (5-fold mean ± std)",
    fontsize=14, weight="bold", y=0.995,
)
x = np.arange(len(DECOYS))
width = 0.36
COLORS = {"M2-groupkfold (DUDE-Z trained)": "#3a6ea5",
          "M2-ad-mixed (50/50)":            "#c0392b"}

for ax, (short, label) in zip(axes.flat, METRICS):
    sub = df[df["metric"] == short]
    for i, regime in enumerate(COLORS):
        means = [sub[(sub.decoy == d) & (sub.regime == regime)]["mean"].iloc[0] for d in DECOYS]
        stds  = [sub[(sub.decoy == d) & (sub.regime == regime)]["std"].iloc[0]  for d in DECOYS]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                      label=regime, color=COLORS[regime], edgecolor="black", linewidth=0.5)
        for j, (m, s) in enumerate(zip(means, stds)):
            ax.text(x[j] + offset, m + s + max(means) * 0.015, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(DECOYS)
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    if short == "pr_auc":
        ax.axhline(PLEC_GKF, ls="--", color="gray", lw=1, alpha=0.7,
                   label=f"PLEC GroupKFold = {PLEC_GKF:.3f}")
        ax.axhline(GATE1_THRESHOLD, ls=":", color="forestgreen", lw=1.2, alpha=0.8,
                   label=f"Gate-1 threshold = {GATE1_THRESHOLD:.3f}")
    if short == "roc_auc":
        ax.axhline(0.5, ls="--", color="gray", lw=1, alpha=0.7, label="random = 0.5")
    if short in ("pr_auc", "roc_auc"):
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.legend(fontsize=8, loc="upper right")

plt.tight_layout(rect=(0, 0, 1, 0.97))
out_bars = OUT / "bias_matrix_bars.png"
plt.savefig(out_bars, dpi=150, bbox_inches="tight")
plt.close()
print(f"wrote {out_bars}")

# ── Plot 2: AD eval delta highlight (one bar per decoy bg) ────────────────
fig, ax = plt.subplots(figsize=(7.5, 4.5))
sub = df[df["metric"] == "pr_auc"]
gkf_means  = [sub[(sub.decoy == d) & (sub.regime == "M2-groupkfold (DUDE-Z trained)")]["mean"].iloc[0] for d in DECOYS]
mix_means  = [sub[(sub.decoy == d) & (sub.regime == "M2-ad-mixed (50/50)")]["mean"].iloc[0] for d in DECOYS]
deltas = np.array(mix_means) - np.array(gkf_means)
colors = ["#27ae60" if d > 0 else "#c0392b" for d in deltas]
bars = ax.bar(DECOYS, deltas, color=colors, edgecolor="black", linewidth=0.5)
for b, m, g, x_, d_ in zip(bars, mix_means, gkf_means, DECOYS, deltas):
    sign = "+" if d_ > 0 else ""
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + (0.005 if d_ > 0 else -0.012),
            f"{sign}{d_:.4f}\n({g:.3f}→{m:.3f})",
            ha="center", va="bottom" if d_ > 0 else "top", fontsize=9)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Δ PR-AUC (m2-ad-mixed − m2-groupkfold)")
ax.set_title("Phase 6 retraining effect by decoy background — improvement is AD-specific",
             fontsize=11, weight="bold")
ax.grid(axis="y", linestyle=":", alpha=0.4)
plt.tight_layout()
out_delta = OUT / "phase6_delta_pr_auc.png"
plt.savefig(out_delta, dpi=150, bbox_inches="tight")
plt.close()
print(f"wrote {out_delta}")

# ── Plot 3: per-target scatter for the 2 cells with per-target data ───────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
for ax, decoy in zip(axes, ("Extrema", "AD")):
    a = pd.read_csv(GKF_TARGET[decoy])
    b = pd.read_csv(ADMIX_TARGET[decoy])
    merged = a.merge(b, on="target", suffixes=("_gkf", "_mix"))
    ax.scatter(merged["pr_auc_gkf"], merged["pr_auc_mix"], s=28,
               c="#3a6ea5", alpha=0.75, edgecolors="black", linewidth=0.4)
    lim = max(merged["pr_auc_gkf"].max(), merged["pr_auc_mix"].max(), 0.5) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5, label="y = x")
    # Highlight 3 best-improved targets (by Δ)
    merged["delta"] = merged["pr_auc_mix"] - merged["pr_auc_gkf"]
    top = merged.nlargest(3, "delta")
    for _, r in top.iterrows():
        ax.annotate(r["target"], (r["pr_auc_gkf"], r["pr_auc_mix"]),
                    fontsize=8, xytext=(4, 2), textcoords="offset points")
    n = len(merged)
    mean_d = merged["delta"].mean()
    pos = int((merged["delta"] > 0).sum())
    ax.set_xlabel("PR-AUC (m2-groupkfold)")
    if decoy == "Extrema":
        ax.set_ylabel("PR-AUC (m2-ad-mixed)")
    ax.set_title(f"{decoy}  (n={n}, mean Δ = {mean_d:+.4f}, +{pos}/{n} improved)")
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, loc="lower right")
fig.suptitle("Per-target PR-AUC — m2-groupkfold vs m2-ad-mixed (each dot = 1 target)",
             fontsize=12, weight="bold", y=1.02)
plt.tight_layout()
out_scat = OUT / "per_target_scatter.png"
plt.savefig(out_scat, dpi=150, bbox_inches="tight")
plt.close()
print(f"wrote {out_scat}")

# ── Plot 4: AD per-target by rmsd_bucket ──────────────────────────────────
ad_gkf = pd.read_csv(GKF_TARGET["AD"])
ad_mix = pd.read_csv(ADMIX_TARGET["AD"])
ad_merged = ad_gkf.merge(ad_mix, on="target", suffixes=("_gkf", "_mix"))
# rmsd_bucket lives in the gkf table only; admix table doesn't have it
if "rmsd_bucket" in ad_gkf.columns:
    ad_merged["rmsd_bucket"] = ad_gkf.set_index("target").loc[ad_merged["target"], "rmsd_bucket"].values
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    bucket_colors = {"numerical_precision": "#3a6ea5", "sub_angstrom": "#27ae60", "major": "#c0392b"}
    for b, sub in ad_merged.groupby("rmsd_bucket"):
        ax.scatter(sub["pr_auc_gkf"], sub["pr_auc_mix"], s=42,
                   c=bucket_colors.get(b, "gray"), label=f"{b} (n={len(sub)})",
                   alpha=0.85, edgecolors="black", linewidth=0.5)
    lim = max(ad_merged["pr_auc_gkf"].max(), ad_merged["pr_auc_mix"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
    for _, r in ad_merged.iterrows():
        ax.annotate(r["target"], (r["pr_auc_gkf"], r["pr_auc_mix"]),
                    fontsize=7, xytext=(3, 2), textcoords="offset points", alpha=0.7)
    ax.set_xlabel("AD PR-AUC (m2-groupkfold)")
    ax.set_ylabel("AD PR-AUC (m2-ad-mixed)")
    ax.set_title(f"AD per-target PR-AUC by rmsd_bucket  (40 targets)",
                 fontsize=11, weight="bold")
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    out_ad = OUT / "ad_per_target_by_bucket.png"
    plt.savefig(out_ad, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_ad}")
