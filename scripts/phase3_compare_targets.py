#!/usr/bin/env python
"""Phase 3 cross-target feature comparison.

Loads all .npz from ~/hdbind-3D/features/flowr_root/<target>/ for the smoke
targets (AA2AR, ADRB2, EGFR by default), computes per-target feature stats,
and produces:

  - notes/phase3_target_comparison.md  — markdown table of per-target stats
                                           plus a verbal sanity check
  - notes/phase3_target_comparison.png — 7-panel histogram (one panel per
                                           extraction point), 3 overlapping
                                           histograms (one per target)

The point is NOT to publish science — it's to give the user a fast visual
that confirms feature distributions look comparable across targets. Wildly
divergent shapes would indicate a pocket-cutoff bug or an input-handling
discrepancy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HDBIND = Path("/home/maurbina/hdbind-3D")
HOOK_NAMES = [
    "f_lig_pre", "f_pocket_pre", "eij_pooled_pre",
    "z_lig_post", "z_pocket_post", "z_int_post",
    "combined_at_pic50_head",
]


def load_target(target_dir: Path) -> dict:
    """Load all .npz under a target dir into a stack-by-tensor-name dict."""
    files = sorted(target_dir.glob("*.npz"))
    assert files, f"No .npz files in {target_dir}"
    stacks: dict[str, list[np.ndarray]] = {n: [] for n in HOOK_NAMES}
    labels: list[int] = []
    kinds: list[str] = []
    for f in files:
        d = np.load(f, allow_pickle=False)
        for n in HOOK_NAMES:
            stacks[n].append(d[n])
        if "label" in d.files:
            labels.append(int(d["label"]))
        if "kind" in d.files:
            kinds.append(str(d["kind"]))
    stacks_arr = {n: np.stack(v, axis=0) for n, v in stacks.items()}
    return {
        "n": len(files),
        "tensors": stacks_arr,
        "labels": np.array(labels) if labels else None,
        "kinds": kinds,
    }


def per_target_stats(target_dir: Path, target_name: str, manifest: dict | None) -> dict:
    """Return a stats dict for the target."""
    bundle = load_target(target_dir)
    stats = {"target": target_name, "n_poses": bundle["n"]}
    if bundle["labels"] is not None:
        stats["n_actives"] = int((bundle["labels"] == 1).sum())
        stats["n_decoys"] = int((bundle["labels"] == 0).sum())
    stats["per_tensor"] = {}
    for n, arr in bundle["tensors"].items():
        flat = arr.reshape(arr.shape[0], -1)
        stats["per_tensor"][n] = {
            "shape": list(arr.shape),
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "pct_zero": float((flat == 0).mean()),
            "any_nan": bool(np.isnan(flat).any()),
            "any_inf": bool(np.isinf(flat).any()),
        }
    if manifest:
        stats["ms_per_ligand_avg"] = manifest.get("ms_per_ligand_avg")
        stats["elapsed_seconds"] = manifest.get("elapsed_seconds")
        stats["validation"] = manifest.get("validation")
    stats["_tensors"] = bundle["tensors"]  # for plotting; stripped from JSON output
    return stats


def render_md(all_stats: list[dict]) -> str:
    L: list[str] = []
    L.append("# Phase 3 cross-target feature comparison\n")
    L.append("Sanity inspection — confirms feature distributions are comparable "
             "across the three smoke targets (AA2AR, ADRB2, EGFR). Pose-count, "
             "throughput, and validation outcomes per target follow.\n")
    L.append("## Per-target summary\n")
    L.append("| Target | Poses | Actives | Decoys | Throughput (ms/lig) | Elapsed (s) | Fire-count | concat==combined | manual==pred |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for s in all_stats:
        v = s.get("validation") or {}
        L.append(
            f"| {s['target']} | {s['n_poses']} | {s.get('n_actives', '—')} | "
            f"{s.get('n_decoys', '—')} | "
            f"{s.get('ms_per_ligand_avg', 0):.1f} | "
            f"{s.get('elapsed_seconds', 0):.1f} | "
            f"{v.get('fire_count_assertion', '—')} | "
            f"{'OK' if (v.get('concat_eq_combined') or {}).get('passed') else 'FAIL'} | "
            f"{'OK' if (v.get('manual_head_eq_predict') or {}).get('passed') else 'FAIL'} |"
        )
    L.append("")

    L.append("## Per-tensor stats by target\n")
    for tn in HOOK_NAMES:
        L.append(f"### `{tn}`\n")
        L.append("| Target | shape | mean | std | min | max | any NaN | any Inf |")
        L.append("|---|---|---|---|---|---|---|---|")
        for s in all_stats:
            t = s["per_tensor"][tn]
            L.append(
                f"| {s['target']} | {t['shape']} | "
                f"{t['mean']:+.4f} | {t['std']:.4f} | "
                f"{t['min']:+.4f} | {t['max']:+.4f} | "
                f"{t['any_nan']} | {t['any_inf']} |"
            )
        L.append("")

    # Verbal sanity check
    L.append("## Sanity verdict\n")
    issues: list[str] = []
    for tn in HOOK_NAMES:
        stds = [s["per_tensor"][tn]["std"] for s in all_stats]
        # Heuristic: distributions are "comparable" if max(std)/min(std) < ~2.
        # Beyond ~3x indicates a pocket-size or input-handling discrepancy.
        if min(stds) > 0:
            ratio = max(stds) / min(stds)
            if ratio > 3.0:
                issues.append(
                    f"- `{tn}` std varies {ratio:.1f}x across targets: "
                    f"{[f'{s:.4f}' for s in stds]}. Investigate — could indicate "
                    f"a pocket-size or input-handling discrepancy across targets."
                )
        if any(s["per_tensor"][tn]["any_nan"] for s in all_stats):
            issues.append(f"- `{tn}` has NaN in at least one target. Investigate before Phase 4.")
        if any(s["per_tensor"][tn]["any_inf"] for s in all_stats):
            issues.append(f"- `{tn}` has Inf in at least one target. Investigate before Phase 4.")
    if issues:
        L.append("Observed issues:\n")
        for i in issues:
            L.append(i)
        L.append("")
    else:
        L.append("All tensor distributions are within ~3x std range across targets, "
                 "no NaN/Inf detected. Safe to proceed to Phase 4 throughput review.\n")
    L.append("\nSee accompanying figure: `phase3_target_comparison.png` (one panel "
             "per extraction point, three overlapping histograms — one per target).\n")
    return "\n".join(L)


def render_png(all_stats: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    colors = {"AA2AR": "tab:blue", "ADRB2": "tab:orange", "EGFR": "tab:green"}
    for i, tn in enumerate(HOOK_NAMES):
        ax = axes[i]
        for s in all_stats:
            t = s["_tensors"][tn].flatten()
            ax.hist(t, bins=80, alpha=0.45, density=True,
                    label=f"{s['target']} (N={s['n_poses']})",
                    color=colors.get(s["target"], None))
        ax.set_title(tn, fontsize=11)
        ax.set_xlabel("feature value")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    # Hide unused 8th panel
    axes[-1].axis("off")
    fig.suptitle(
        "Phase 3 cross-target FLOWR.ROOT feature distributions (per extraction point)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["AA2AR", "ADRB2", "EGFR"])
    ap.add_argument("--features_root",
                    default=str(_HDBIND / "features" / "flowr_root"))
    ap.add_argument("--out_md",
                    default=str(_HDBIND / "notes" / "phase3_target_comparison.md"))
    ap.add_argument("--out_png",
                    default=str(_HDBIND / "notes" / "phase3_target_comparison.png"))
    args = ap.parse_args()

    feat_root = Path(args.features_root)
    all_stats: list[dict] = []
    for t in args.targets:
        tdir = feat_root / t
        manifest_p = tdir / "manifest.json"
        manifest = json.loads(manifest_p.read_text()) if manifest_p.exists() else None
        s = per_target_stats(tdir, t, manifest)
        all_stats.append(s)
        print(
            f"{t}: n_poses={s['n_poses']}  "
            f"ms_per_lig={s.get('ms_per_ligand_avg', 0):.1f}  "
            f"elapsed_s={s.get('elapsed_seconds', 0):.1f}"
        )

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    render_png(all_stats, Path(args.out_png))
    md = render_md(all_stats)
    Path(args.out_md).write_text(md)
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
