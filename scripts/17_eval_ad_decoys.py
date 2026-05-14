"""
17_eval_ad_decoys.py — TIER3_DECOY_BIAS_PLAN.md Phase 5 §2 evaluation of M2
against the AD off-target-active decoy background.

Mirrors `scripts/11_eval_goldilocks.py` (per-fold replay of the train manifest's
GroupKFold/stratified split, eval against an alternate decoy LMDB) and adds an
`rmsd_bucket` per-target column derived from the Phase 3 alignment manifest so
the Phase 6 subset analysis (full-40 vs 26-same-crystal) can split on it without
re-deriving.

Default train run is `outputs/07_train_egnn/m2-stratified` per the bias plan,
even though that manifest's `cv: stratified` means every fold's val-target set
is all 43 — under stratified replay all folds will end up evaluating the full
AD index, so per-fold std reflects ensemble-disagreement rather than
disjoint-target generalization. To compare apples-to-apples against the
GroupKFold M2 baselines (DUDE-Z 0.150 / Goldilocks 0.062 / PLEC 0.023), pass
`--train-run outputs/07_train_egnn/<groupkfold-m2-run>` explicitly.

Outputs to outputs/17_eval_ad_decoys/<run_id>/:
  per_fold_metrics.csv   one row per fold; metrics + Δ vs DUDE-Z (where available)
  per_target_metrics.csv per-target metrics with rmsd_bucket column
  predictions.csv        (target, complex_id, fold, label, score) — for Phase 6
  manifest.json
  config.yaml
  REPORT.md
  HANDOFF.md             one-paragraph headline vs M2/PLEC reference baselines
  eval.log

Usage:
  python scripts/17_eval_ad_decoys.py --train-run outputs/07_train_egnn/m2-stratified
  # Auto-discovers latest decoy_source=ad 06 run for --ad-dir if not given.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import PROJECT_ROOT
from src.utils import load_canonical_groupkfold, setup_logging
from src.evaluation import compute_all_metrics

try:
    import torch
    from torch.utils.data import Dataset
    from torch_geometric.loader import DataLoader as PyGDataLoader
except ImportError as e:
    sys.stderr.write(f"Missing torch/torch_geometric: {e}\n"); sys.exit(1)

try:
    import lmdb
except ImportError:
    sys.stderr.write("Missing 'lmdb'\n"); sys.exit(1)

from src.models.egnn import EGNN

log = logging.getLogger("17_eval_ad_decoys")

# Reference values for the headline comparison (TIER3_DECOY_BIAS_PLAN Phase 5
# §4). These are GroupKFold M2 numbers; if --train-run is the stratified
# checkpoint, the apples-to-apples comparison is approximate — see REPORT.md.
M2_DUDEZ_PR_AUC = 0.150       # M2 GroupKFold on DUDE-Z (m2-stratified report)
M2_GOLDILOCKS_PR_AUC = 0.062  # M2 GroupKFold on Goldilocks (corrected)
PLEC_GROUPKFOLD_PR_AUC = 0.023  # PLEC under GroupKFold protocol

# Gate 1 with the corrected PLEC GroupKFold reference (memory file Tier 3 state)
PLEC_GATE1_REFERENCE = PLEC_GROUPKFOLD_PR_AUC
GATE_1_THRESHOLD = PLEC_GATE1_REFERENCE + 0.05  # 0.073

# RMSD-bucket boundaries (Å) for Phase 6 subset analysis. Same buckets used in
# CAVEATS.md: numerical_precision <= 0.001; sub_angstrom <= 1.0; major > 1.0.
RMSD_BUCKETS = [
    ("numerical_precision", 0.0, 0.001),
    ("sub_angstrom",        0.001, 1.0),
    ("major",               1.0, float("inf")),
]

NODE_FEATURE_DIM = 10
EDGE_ATTR_DIM = 3

_LMDB_ENV_CACHE: dict[str, "lmdb.Environment"] = {}


def _get_lmdb_env(path: str) -> "lmdb.Environment":
    env = _LMDB_ENV_CACHE.get(path)
    if env is None:
        env = lmdb.open(path, readonly=True, lock=False, subdir=False,
                        readahead=True, meminit=False, max_readers=512)
        _LMDB_ENV_CACHE[path] = env
    return env


def _to_lig_only(data):
    """M1 variant — drop protein atoms, keep ligand subgraph. Mirrors 07/11."""
    lig_mask = data.x[:, -1] == 0
    return data.subgraph(lig_mask)


_VARIANT_TRANSFORMS = {"full": None, "lig_only": _to_lig_only}


class LMDBPyGDataset(Dataset):
    def __init__(self, lmdb_path: str | Path, keys: list[str], variant: str = "full"):
        self.lmdb_path = str(lmdb_path)
        self.keys = [k.encode("ascii") if isinstance(k, str) else k for k in keys]
        if variant not in _VARIANT_TRANSFORMS:
            raise ValueError(f"Unknown variant {variant!r}")
        self.transform = _VARIANT_TRANSFORMS[variant]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        env = _get_lmdb_env(self.lmdb_path)
        with env.begin() as txn:
            payload = txn.get(self.keys[idx])
        if payload is None:
            raise KeyError(f"Missing LMDB key {self.keys[idx]!r}")
        d = pickle.loads(payload)
        if self.transform is not None:
            d = self.transform(d)
        return d


def _ligand_mask(batch_x: torch.Tensor) -> torch.Tensor:
    return batch_x[:, -1] == 0


@contextlib.contextmanager
def _nullcm():
    yield


def _evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, scores = [], []
    with torch.no_grad():
        cm = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
              if device.type == "cuda" else _nullcm())
        with cm:
            for batch in loader:
                batch = batch.to(device)
                logits = model(
                    batch.x, batch.pos, batch.edge_index,
                    batch.edge_attr, _ligand_mask(batch.x), batch.batch,
                )
                ys.append(batch.y.detach().float().cpu().numpy().ravel())
                scores.append(torch.sigmoid(logits.float()).cpu().numpy().ravel())
    return np.concatenate(ys), np.concatenate(scores)


def _git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short=7", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"


def _make_run_id(user_id: str | None) -> str:
    if user_id is not None:
        return user_id
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{_git_short_sha()}"


def _latest_ad_dataset_dir() -> Path:
    """Find newest 06 run with decoy_source=ad. Searches both flat (legacy)
    and nested (<run_id>/<source>/) layouts."""
    base = PROJECT_ROOT / "outputs" / "06_build_egnn_dataset"
    candidates = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        # Prefer nested ad/ subdir (new layout); fall back to flat (legacy).
        for sub in (d / "ad", d):
            m = sub / "manifest.json"
            if not m.exists():
                continue
            try:
                mj = json.loads(m.read_text())
                if mj.get("decoy_source") == "ad" or mj.get("decoy_set") == "AD":
                    candidates.append((sub.stat().st_mtime, sub))
                    break
            except (OSError, json.JSONDecodeError):
                continue
    if not candidates:
        raise FileNotFoundError(
            f"No 06 run with decoy_source=ad under {base}. Run: "
            f"python scripts/06_build_egnn_dataset.py --decoy-source ad --targets ..."
        )
    candidates.sort()
    return candidates[-1][1]


def _bucket_for(rmsd: float) -> str:
    for name, lo, hi in RMSD_BUCKETS:
        if lo <= rmsd < hi or (hi == float("inf") and rmsd >= lo):
            return name
    return "unknown"


def _load_rmsd_buckets(align_csv: Path) -> dict[str, dict]:
    """Return {target_upper: {post_rmsd, post_centroid_offset, rmsd_bucket}}."""
    df = pd.read_csv(align_csv)
    out = {}
    for _, r in df.iterrows():
        t = str(r["target"]).upper()
        rmsd = float(r["post_rmsd_a"])
        out[t] = {
            "post_rmsd_a": rmsd,
            "post_centroid_offset_a": float(r["post_centroid_offset_a"]),
            "rmsd_bucket": _bucket_for(rmsd),
        }
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--train-run", type=Path, default=None,
                   help="07_train_egnn run dir. Default: "
                        "outputs/07_train_egnn/m2-stratified (per Phase 5 §2).")
    p.add_argument("--ad-dir", type=Path, default=None,
                   help="06 run dir with decoy_source=ad (default: latest).")
    p.add_argument("--alignment-csv", type=Path, default=None,
                   help="Phase 3 per_target.csv with post_rmsd_a (default: "
                        "outputs/16_align_ad_decoys/all41-20260508/per_target.csv).")
    p.add_argument("--variant", choices=list(_VARIANT_TRANSFORMS), default=None,
                   help="Graph variant; defaults to the train manifest's variant.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.train_run is None:
        args.train_run = PROJECT_ROOT / "outputs" / "07_train_egnn" / "m2-stratified"
    args.train_run = args.train_run.resolve()
    if not (args.train_run / "manifest.json").exists():
        sys.stderr.write(f"No manifest.json in {args.train_run}\n")
        return 2

    if args.ad_dir is None:
        args.ad_dir = _latest_ad_dataset_dir()
    args.ad_dir = args.ad_dir.resolve()
    ad_manifest = json.loads((args.ad_dir / "manifest.json").read_text())
    if not (ad_manifest.get("decoy_source") == "ad"
            or ad_manifest.get("decoy_set") == "AD"):
        sys.stderr.write(
            f"--ad-dir {args.ad_dir} has decoy_source="
            f"{ad_manifest.get('decoy_source')!r}, expected 'ad'\n"
        )
        return 2

    if args.alignment_csv is None:
        args.alignment_csv = (PROJECT_ROOT / "outputs" / "16_align_ad_decoys"
                              / "all41-20260508" / "per_target.csv")
    args.alignment_csv = args.alignment_csv.resolve()
    if not args.alignment_csv.exists():
        sys.stderr.write(f"alignment per_target.csv missing: {args.alignment_csv}\n")
        return 2

    run_id = _make_run_id(args.run_id)
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "outputs" / "17_eval_ad_decoys" / run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=logging.INFO, log_file=args.output_dir / "eval.log")
    log.setLevel(logging.INFO)

    train_manifest = json.loads((args.train_run / "manifest.json").read_text())
    train_variant = train_manifest.get("variant", "full")
    if args.variant is None:
        args.variant = train_variant
    is_gate1_eligible = (args.variant == "full")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    git_sha = _git_short_sha()

    log.info("=" * 70)
    log.info("17_eval_ad_decoys starting")
    log.info("  run_id: %s", run_id)
    log.info("  git_sha: %s", git_sha)
    log.info("  train_run: %s", args.train_run)
    log.info("  train_cv: %s  variant: %s",
             train_manifest.get("cv", "?"), train_variant)
    log.info("  ad_dir: %s", args.ad_dir)
    log.info("  alignment_csv: %s", args.alignment_csv)
    log.info("  device: %s  PLEC GroupKFold ref: %.3f  gate-1 threshold: %.3f",
             device, PLEC_GATE1_REFERENCE, GATE_1_THRESHOLD)

    # DUDE-Z per-fold metrics for the side-by-side delta (when available).
    dude_metrics = pd.read_csv(args.train_run / "per_fold_metrics.csv")

    train_targets = list(train_manifest["targets"])
    n_folds = train_manifest["n_folds"]

    train_idx = pd.read_csv(Path(train_manifest["dataset_dir"]) / "index.csv",
                            dtype={"key": str})
    train_idx = train_idx[train_idx["target"].isin(set(train_targets))].reset_index(drop=True)
    train_targets_arr = train_idx["target"].values

    ad_idx = pd.read_csv(args.ad_dir / "index.csv", dtype={"key": str})
    ad_targets_present = set(ad_idx["target"].unique())
    missing_in_ad = sorted(set(train_targets) - ad_targets_present)
    if missing_in_ad:
        log.warning(
            "Train targets not present in AD LMDB (decoys missing): %s. "
            "These contribute zero rows to their fold's AD val set. "
            "(ABL1 expected — see CAVEATS.md.)",
            missing_in_ad,
        )

    fold_splits = load_canonical_groupkfold(args.train_run / "manifest.json")
    fold_paths = sorted(args.train_run.glob("fold_*/model.pt"))
    if len(fold_paths) != n_folds:
        log.warning("Found %d fold_K/model.pt files but train manifest says n_folds=%d",
                    len(fold_paths), n_folds)

    rmsd_lookup = _load_rmsd_buckets(args.alignment_csv)

    fold_rows = []
    pred_rows = []  # (target, complex_id, fold, label, score)
    t0 = time.perf_counter()

    for fold_idx, (_train_i, val_i) in enumerate(fold_splits):
        if fold_idx >= len(fold_paths):
            break
        model_path = fold_paths[fold_idx]
        val_targets = sorted(set(train_targets_arr[val_i]))

        # Filter AD index to this fold's val targets.
        sub = ad_idx[ad_idx["target"].isin(val_targets)]
        val_keys = list(sub["key"].astype(str).values)
        val_meta = sub[["target", "complex_id", "label"]].reset_index(drop=True)

        log.info("─" * 60)
        log.info("Fold %d  n_val_targets=%d  n_val_keys=%d  ckpt=%s",
                 fold_idx, len(val_targets), len(val_keys),
                 model_path.relative_to(args.train_run))

        if not val_keys:
            log.warning("Fold %d has 0 val keys in AD; skipping.", fold_idx)
            continue

        model = EGNN(
            in_node_dim=NODE_FEATURE_DIM, edge_attr_dim=EDGE_ATTR_DIM,
            h_dim=train_manifest["h_dim"], n_layers=train_manifest["n_layers"],
        ).to(device)
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)

        ds = LMDBPyGDataset(args.ad_dir / "dataset.lmdb", val_keys, variant=args.variant)
        loader = PyGDataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        t_a = time.perf_counter()
        y_true, y_score = _evaluate(model, loader, device)
        wall = time.perf_counter() - t_a

        n_pos = int(y_true.sum())
        n_neg = len(y_true) - n_pos
        if n_pos == 0:
            log.warning("Fold %d: 0 positives in AD val — skipping metrics.", fold_idx)
            continue
        metrics = compute_all_metrics(y_true.astype(int), y_score, threshold=0.5)

        # Side-by-side DUDE-Z baseline for the same fold (m2-stratified val_pr_auc).
        # Under stratified CV this is the inflated leakage probe; under
        # GroupKFold this is the canonical val_pr_auc.
        dude_row = dude_metrics[dude_metrics["fold"] == fold_idx]
        dude_pr_auc = float(dude_row["val_pr_auc"].iloc[0]) if not dude_row.empty else float("nan")
        dude_bedroc = float(dude_row["val_bedroc"].iloc[0]) if not dude_row.empty else float("nan")

        log.info("  AD          pr_auc=%.4f  roc_auc=%.4f  bedroc=%.4f  ef_1pct=%.2f  ef_5pct=%.2f  "
                 "n_pos=%d  n_neg=%d  %.1fs",
                 metrics["pr_auc"], metrics["roc_auc"], metrics["bedroc"],
                 metrics["ef_1pct"], metrics["ef_5pct"], n_pos, n_neg, wall)
        log.info("  DUDE-Z (same fold)  pr_auc=%.4f  bedroc=%.4f  Δ_pr_auc=%+.4f",
                 dude_pr_auc, dude_bedroc, metrics["pr_auc"] - dude_pr_auc)

        fold_rows.append({
            "fold": fold_idx,
            "n_val_targets": len(val_targets),
            "n_val": len(val_keys),
            "n_val_pos": n_pos,
            "n_val_neg": n_neg,
            "wall_s": wall,
            **{f"ad_{k}": v for k, v in metrics.items()},
            "dude_pr_auc": dude_pr_auc,
            "dude_bedroc": dude_bedroc,
            "delta_pr_auc": metrics["pr_auc"] - dude_pr_auc,
            "delta_bedroc": metrics["bedroc"] - dude_bedroc,
        })

        for j in range(len(val_keys)):
            pred_rows.append({
                "target": val_meta.at[j, "target"],
                "complex_id": val_meta.at[j, "complex_id"],
                "fold": fold_idx,
                "label": int(y_true[j]),
                "score": float(y_score[j]),
            })

    wall_total = time.perf_counter() - t0
    log.info("All folds done in %.1fs", wall_total)

    df_fold = pd.DataFrame(fold_rows)
    df_fold.to_csv(args.output_dir / "per_fold_metrics.csv", index=False)

    df_pred = pd.DataFrame(pred_rows)
    df_pred.to_csv(args.output_dir / "predictions.csv", index=False)

    # ── Per-target metrics (aggregate score across folds, then compute) ──
    # For (target, complex_id) appearing in multiple folds (stratified case),
    # average the scores; for single-fold targets the mean is identity.
    agg = (df_pred.groupby(["target", "complex_id", "label"], as_index=False)
                  ["score"].mean())
    per_target_rows = []
    for tgt, sub in agg.groupby("target"):
        y_t = sub["label"].values.astype(int)
        y_s = sub["score"].values.astype(float)
        n_pos = int(y_t.sum())
        n_neg = len(y_t) - n_pos
        if n_pos == 0:
            continue
        m = compute_all_metrics(y_t, y_s, threshold=0.5)
        info = rmsd_lookup.get(tgt, {"post_rmsd_a": np.nan,
                                     "post_centroid_offset_a": np.nan,
                                     "rmsd_bucket": "unknown"})
        per_target_rows.append({
            "target": tgt,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "post_rmsd_a": info["post_rmsd_a"],
            "post_centroid_offset_a": info["post_centroid_offset_a"],
            "rmsd_bucket": info["rmsd_bucket"],
            **{k: float(v) for k, v in m.items()},
        })
    df_target = pd.DataFrame(per_target_rows).sort_values("target").reset_index(drop=True)
    df_target.to_csv(args.output_dir / "per_target_metrics.csv", index=False)

    pr_auc_mean = float(df_fold["ad_pr_auc"].mean()) if not df_fold.empty else float("nan")
    gate1_pass = bool(is_gate1_eligible and pr_auc_mean >= GATE_1_THRESHOLD)

    manifest = {
        "run_id": run_id,
        "git_sha": git_sha,
        "venv": os.environ.get("VIRTUAL_ENV", "(none)"),
        "script": "scripts/17_eval_ad_decoys.py",
        "train_run": str(args.train_run),
        "train_cv": train_manifest.get("cv"),
        "train_variant": train_variant,
        "ad_dir": str(args.ad_dir),
        "alignment_csv": str(args.alignment_csv),
        "variant": args.variant,
        "n_folds": int(len(fold_rows)),
        "missing_in_ad": missing_in_ad,
        "wall_seconds_total": wall_total,
        "plec_groupkfold_pr_auc": PLEC_GATE1_REFERENCE,
        "gate1_threshold": GATE_1_THRESHOLD,
        "gate1_eligible": is_gate1_eligible,
        "gate1_pr_auc_mean": pr_auc_mean,
        "gate1_pass": gate1_pass,
        "rows": fold_rows,
        "rmsd_buckets": [
            {"name": n, "lo_a": lo, "hi_a": (None if hi == float("inf") else hi)}
            for n, lo, hi in RMSD_BUCKETS
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    cfg = {"args": {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}}
    (args.output_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    write_report(args.output_dir, manifest, df_fold, df_target)
    write_handoff(args.output_dir, manifest, df_fold, df_target)
    log.info("Outputs at %s", args.output_dir)
    if is_gate1_eligible:
        log.info("Gate 1: %s  (M2 mean PR-AUC %.4f vs threshold %.4f)",
                 "PASS" if gate1_pass else "FAIL", pr_auc_mean, GATE_1_THRESHOLD)
    return 0


def write_report(output_dir: Path, manifest: dict,
                 df_fold: pd.DataFrame, df_target: pd.DataFrame) -> None:
    is_eligible = manifest.get("gate1_eligible", True)
    gate1 = ("PASS" if manifest["gate1_pass"] else "FAIL") if is_eligible else "N/A (variant != full)"
    train_cv = manifest.get("train_cv", "?")
    cv_note = ""
    if train_cv == "stratified":
        cv_note = (
            "\n> **CV note:** train_run is stratified-CV (every fold's val targets = "
            "all 43). Each fold therefore evaluates the full AD index, and "
            "per-fold std reflects ensemble disagreement between the 5 "
            "checkpoints rather than disjoint-target generalization. The "
            "headline `mean PR-AUC` is the 5-checkpoint average. For "
            "apples-to-apples comparison with the GroupKFold M2 baselines, "
            "rerun with `--train-run` pointing at a `cv: groupkfold` 07 run."
        )

    lines = [
        "# 17_eval_ad_decoys — Phase 5 §2 evaluation on AD decoy background",
        "",
        f"- run_id: `{manifest['run_id']}`",
        f"- git_sha: `{manifest['git_sha']}`",
        f"- train_run: `{manifest['train_run']}`  (cv=`{train_cv}`, variant=`{manifest['train_variant']}`)",
        f"- variant evaluated: `{manifest.get('variant', 'full')}`",
        f"- ad_dir: `{manifest['ad_dir']}`",
        f"- alignment_csv: `{manifest['alignment_csv']}`",
        f"- folds evaluated: {manifest['n_folds']}",
        f"- wall: **{manifest['wall_seconds_total']:.1f}s**",
        f"- missing-in-AD targets: {manifest['missing_in_ad'] or 'none'}",
        cv_note,
        "",
        "## Per fold (AD vs DUDE-Z, same checkpoint)",
        "",
        "| fold | n_val | n_pos | AD pr_auc | AD roc_auc | AD bedroc | AD ef_1pct | AD ef_5pct | dude pr_auc | Δ pr_auc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df_fold.iterrows():
        lines.append(
            f"| {r['fold']} | {int(r['n_val'])} | {int(r['n_val_pos'])} | "
            f"{r['ad_pr_auc']:.4f} | {r['ad_roc_auc']:.4f} | {r['ad_bedroc']:.4f} | "
            f"{r['ad_ef_1pct']:.2f} | {r['ad_ef_5pct']:.2f} | "
            f"{r['dude_pr_auc']:.4f} | {r['delta_pr_auc']:+.4f} |"
        )
    if not df_fold.empty:
        std_lo = "ddof=0"  # population std, matches 11
        lines += [
            "",
            "## Aggregate (mean ± std across folds)",
            "",
            f"- **AD PR-AUC**: {df_fold['ad_pr_auc'].mean():.4f} ± {df_fold['ad_pr_auc'].std(ddof=0):.4f}",
            f"- AD ROC-AUC:  {df_fold['ad_roc_auc'].mean():.4f} ± {df_fold['ad_roc_auc'].std(ddof=0):.4f}",
            f"- AD BEDROC:   {df_fold['ad_bedroc'].mean():.4f} ± {df_fold['ad_bedroc'].std(ddof=0):.4f}",
            f"- AD EF@1%:    {df_fold['ad_ef_1pct'].mean():.2f} ± {df_fold['ad_ef_1pct'].std(ddof=0):.2f}",
            f"- AD EF@5%:    {df_fold['ad_ef_5pct'].mean():.2f} ± {df_fold['ad_ef_5pct'].std(ddof=0):.2f}",
            f"- DUDE-Z PR-AUC (same folds): {df_fold['dude_pr_auc'].mean():.4f} ± {df_fold['dude_pr_auc'].std(ddof=0):.4f}",
            f"- ΔPR-AUC (AD − DUDE-Z): {df_fold['delta_pr_auc'].mean():+.4f} ± {df_fold['delta_pr_auc'].std(ddof=0):.4f}",
            "",
            f"## §5 gate 1 — beat PLEC + 0.05 = {manifest['gate1_threshold']:.3f}",
            "",
            f"- mean AD PR-AUC: **{manifest['gate1_pr_auc_mean']:.4f}**",
            f"- PLEC GroupKFold reference (used as proxy; AD-specific PLEC TBD): {manifest['plec_groupkfold_pr_auc']:.3f}",
            f"- Gate 1 threshold: ≥ {manifest['gate1_threshold']:.3f}",
            f"- **Gate 1: {gate1}**",
            "",
            "## Gates 2/3/4 — pointers",
            "",
            "These three gates are **not** computed inline; they require additional runs:",
            "- **Gate 2** (receptor matters): "
            "`python scripts/10_bias_controls.py --train-run <train> --eval-dataset-dir <ad-dir>` "
            "and read the ΔBEDROC `none − receptor` row from its REPORT.md.",
            "- **Gate 3** (M2 lig-ablated ≤ M1 lig-only): same `10_bias_controls.py` "
            "invocation as gate 2 (its `ligand` ablation row), plus an M1 (lig_only) "
            "eval against the same AD dir for the comparison baseline.",
            "- **Gate 4** (random vs GroupKFold gap on AD): rerun this script with "
            "`--train-run` pointing at the GroupKFold M2 (`cv: groupkfold`) and at "
            "the stratified M2 (`cv: stratified`); the gap between the two AD PR-AUCs "
            "is the gate-4 readout.",
        ]
    else:
        lines.append("\n_No folds evaluated successfully._")

    # Per-target table grouped by RMSD bucket.
    if not df_target.empty:
        lines += [
            "",
            "## Per-target metrics (aggregated across folds, with rmsd_bucket)",
            "",
            "| target | rmsd_bucket | post_rmsd Å | n_pos | n_neg | pr_auc | roc_auc | bedroc | ef_1pct | ef_5pct |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        bucket_order = {n: i for i, (n, _, _) in enumerate(RMSD_BUCKETS)}
        df_t_sorted = df_target.copy()
        df_t_sorted["_b"] = df_t_sorted["rmsd_bucket"].map(bucket_order).fillna(99)
        df_t_sorted = df_t_sorted.sort_values(["_b", "target"]).reset_index(drop=True)
        for _, r in df_t_sorted.iterrows():
            lines.append(
                f"| {r['target']} | {r['rmsd_bucket']} | {r['post_rmsd_a']:.4f} | "
                f"{int(r['n_pos'])} | {int(r['n_neg'])} | "
                f"{r['pr_auc']:.4f} | {r['roc_auc']:.4f} | {r['bedroc']:.4f} | "
                f"{r['ef_1pct']:.2f} | {r['ef_5pct']:.2f} |"
            )

        # Bucket aggregates — Phase 6 readout pre-computed.
        lines += ["", "### Bucket aggregates (Phase 6 hook)", ""]
        lines += ["| bucket | n_targets | mean pr_auc | std pr_auc | mean bedroc |",
                  "|---|---:|---:|---:|---:|"]
        for n, _, _ in RMSD_BUCKETS:
            sub = df_target[df_target["rmsd_bucket"] == n]
            if sub.empty:
                continue
            lines.append(
                f"| {n} | {len(sub)} | {sub['pr_auc'].mean():.4f} | "
                f"{sub['pr_auc'].std(ddof=0):.4f} | {sub['bedroc'].mean():.4f} |"
            )

    lines += ["", "## Notes", "", "_Filled in after review by the operator._", ""]
    (output_dir / "REPORT.md").write_text("\n".join(lines))


def write_handoff(output_dir: Path, manifest: dict,
                  df_fold: pd.DataFrame, df_target: pd.DataFrame) -> None:
    """One-paragraph headline placing AD PR-AUC against the three reference
    baselines from TIER3_DECOY_BIAS_PLAN Phase 5 §4."""
    if df_fold.empty:
        (output_dir / "HANDOFF.md").write_text("# HANDOFF — no folds evaluated\n")
        return

    ad_pr = float(df_fold["ad_pr_auc"].mean())
    ad_pr_std = float(df_fold["ad_pr_auc"].std(ddof=0))
    ad_bedroc = float(df_fold["ad_bedroc"].mean())
    ad_roc = float(df_fold["ad_roc_auc"].mean())
    train_cv = manifest.get("train_cv", "?")
    apples_caveat = (
        " — note that the train run is `cv=stratified` (the leakage-probe "
        "checkpoint, not GroupKFold M2), so the comparison vs GroupKFold "
        "baselines is qualitative; rerun against a GroupKFold M2 checkpoint "
        "for an apples-to-apples Gate-4 readout"
        if train_cv == "stratified" else ""
    )

    # Compose comparison phrasing
    def _vs(ref_name, ref_val):
        delta = ad_pr - ref_val
        sign = "+" if delta >= 0 else ""
        if abs(delta) < 0.005:
            phrase = f"essentially matches {ref_name} ({ref_val:.3f})"
        elif delta < 0:
            phrase = f"falls below {ref_name} ({ref_val:.3f}, {sign}{delta:.3f})"
        else:
            phrase = f"exceeds {ref_name} ({ref_val:.3f}, {sign}{delta:.3f})"
        return phrase

    n_targets = len(df_target)
    n_buckets_str = ", ".join(
        f"{n}={int((df_target['rmsd_bucket'] == n).sum())}"
        for n, _, _ in RMSD_BUCKETS
        if (df_target['rmsd_bucket'] == n).any()
    )

    para = (
        f"# HANDOFF — Phase 5 §2 AD-decoy evaluation (`{manifest['run_id']}`)\n"
        f"\n"
        f"M2 AD-decoy mean PR-AUC across {manifest['n_folds']} folds is "
        f"**{ad_pr:.4f} ± {ad_pr_std:.4f}** (BEDROC {ad_bedroc:.4f}, "
        f"ROC-AUC {ad_roc:.4f}) over {n_targets} targets ({n_buckets_str}). "
        f"This {_vs('M2 GroupKFold DUDE-Z PR-AUC', M2_DUDEZ_PR_AUC)}, "
        f"{_vs('M2 Goldilocks PR-AUC', M2_GOLDILOCKS_PR_AUC)}, and "
        f"{_vs('PLEC GroupKFold PR-AUC', PLEC_GROUPKFOLD_PR_AUC)}{apples_caveat}. "
        f"Gate 1 (M2 ≥ PLEC + 0.05 = {manifest['gate1_threshold']:.3f}) "
        f"{'PASSES' if manifest['gate1_pass'] else 'FAILS'} on AD using the "
        f"GroupKFold-PLEC reference (no AD-specific PLEC computed yet). "
        f"Gates 2/3/4 against the AD background are not computed inline — "
        f"see REPORT.md §Gates 2/3/4 for the exact follow-up commands. The "
        f"per-target table in REPORT.md and `per_target_metrics.csv` carries "
        f"`rmsd_bucket ∈ {{numerical_precision, sub_angstrom, major}}` for "
        f"the Phase 6 subset analysis (full-40 vs 26-same-crystal).\n"
    )
    (output_dir / "HANDOFF.md").write_text(para)


if __name__ == "__main__":
    sys.exit(main())
