"""
12_verify_plec_groupkfold.py — re-evaluate PLEC under GroupKFold on Goldilocks.

Plan §5 gates require GroupKFold cross-target evaluation. Tier 1-2's published
PLEC PR-AUC of 0.374 on Goldilocks was computed with StratifiedKFold *within
each target* (per src/training.py). Those two protocols are not comparable, so
the gate-1 threshold (PLEC + 0.05 = 0.424) needs PLEC re-measured under the
same protocol used for M2.

Pipeline:
  1. Load Tier 1-2's locked PLEC features:
       results/decoy_sets/Goldilocks/plec/<target>.pkl  → {complex_id: 4096-d vec}
  2. Read the Goldilocks LMDB index.csv to recover (target, complex_id, label)
     for the rows used in M2 evaluation.
  3. Match PLEC features to those rows, drop missing.
  4. GroupKFold k=5 over targets (same as 07_train_egnn.py) — for each fold,
     train XGBoost on the train fold's PLEC vectors, score the val fold.
  5. Report per-fold and mean PR-AUC etc. Compare to the StratifiedKFold-within
     reference (0.374) so we know how much of the gap is protocol vs. signal.

Output: outputs/12_verify_plec_groupkfold/<run_id>/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
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

from config import PROJECT_ROOT, RESULTS_DIR, XGBOOST_PARAMS
from src.utils import load_canonical_groupkfold, load_pickle, setup_logging
from src.evaluation import compute_all_metrics

from sklearn.model_selection import GroupKFold

log = logging.getLogger("12_verify_plec_groupkfold")


PLEC_PKL_DIR = RESULTS_DIR / "decoy_sets" / "Goldilocks" / "plec"
PLEC_STRATIFIED_REFERENCE = 0.374  # Tier 1-2 number per project memory


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


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--goldilocks-dir", type=Path, required=True,
                   help="06 run dir with decoy_set=Goldilocks (need its index.csv).")
    p.add_argument("--plec-dir", type=Path, default=PLEC_PKL_DIR,
                   help="Dir containing one <target>.pkl per target.")
    p.add_argument("--train-run", type=Path, default=None,
                   help="If given, mirror this 07 run's GroupKFold split for "
                        "an apples-to-apples M2-vs-PLEC comparison.")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    args.goldilocks_dir = args.goldilocks_dir.resolve()
    if not (args.goldilocks_dir / "index.csv").exists():
        sys.stderr.write(f"No index.csv in {args.goldilocks_dir}\n"); return 2
    if not args.plec_dir.is_dir():
        sys.stderr.write(f"No PLEC dir at {args.plec_dir}\n"); return 2

    run_id = _make_run_id(args.run_id)
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "outputs" / "12_verify_plec_groupkfold" / run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=logging.INFO, log_file=args.output_dir / "verify.log")
    log.setLevel(logging.INFO)

    git_sha = _git_short_sha()
    log.info("=" * 70)
    log.info("12_verify_plec_groupkfold starting")
    log.info("  run_id: %s", run_id)
    log.info("  goldilocks_dir: %s", args.goldilocks_dir)
    log.info("  plec_dir: %s", args.plec_dir)
    log.info("  n_folds: %d", args.n_folds)

    # Goldilocks index → (target, complex_id, label) rows
    gold_idx = pd.read_csv(args.goldilocks_dir / "index.csv", dtype={"key": str})
    log.info("  goldilocks index rows: %d  targets: %d",
             len(gold_idx), gold_idx["target"].nunique())

    # Build feature matrix by joining with PLEC pkls per target.
    feature_dim = None
    feats = []
    keep_target = []
    keep_label = []
    keep_complex_id = []
    n_missing = 0
    targets_seen = set()
    for target, group in gold_idx.groupby("target"):
        pkl_path = args.plec_dir / f"{target}.pkl"
        if not pkl_path.exists():
            log.warning("  no PLEC pkl for %s; skipping its %d rows", target, len(group))
            n_missing += len(group)
            continue
        plec_dict = load_pickle(pkl_path)
        targets_seen.add(target)
        for _, r in group.iterrows():
            cid = r["complex_id"]
            vec = plec_dict.get(cid)
            if vec is None:
                n_missing += 1
                continue
            if feature_dim is None:
                feature_dim = len(vec)
            feats.append(vec)
            keep_target.append(target)
            keep_label.append(int(r["label"]))
            keep_complex_id.append(cid)
    if not feats:
        log.error("No PLEC features matched Goldilocks rows."); return 2

    X = np.stack(feats).astype(np.float32)
    y = np.asarray(keep_label, dtype=np.int64)
    groups = np.asarray(keep_target)
    log.info("  matched %d/%d rows  feature_dim=%d  n_targets=%d  n_missing=%d",
             len(X), len(gold_idx), feature_dim, len(targets_seen), n_missing)

    # XGBoost per-fold (GPU per CLAUDE.md XGBOOST_PARAMS).
    try:
        from xgboost import XGBClassifier
    except ImportError:
        sys.stderr.write("xgboost not installed — needed for PLEC verification\n")
        return 2

    # If --train-run given, mirror its DUDE-Z GroupKFold composition: derive
    # val_targets from the train manifest's groupkfold over its targets list,
    # then for each fold compute (train_idx, val_idx) into our matched X/y array.
    splits_iter = None
    if args.train_run is not None:
        train_manifest = json.loads(
            (args.train_run / "manifest.json").read_text()
        )
        train_targets = list(train_manifest["targets"])
        train_idx_csv = pd.read_csv(
            Path(train_manifest["dataset_dir"]) / "index.csv", dtype={"key": str}
        )
        train_idx_csv = train_idx_csv[
            train_idx_csv["target"].isin(set(train_targets))
        ].reset_index(drop=True)
        train_groups = train_idx_csv["target"].values
        canonical_splits = load_canonical_groupkfold(args.train_run / "manifest.json")
        per_fold_val_targets = [sorted(set(train_groups[vi])) for _ti, vi in canonical_splits]

        def _splits_from_train_run():
            our_groups_set = np.asarray(groups)
            for vt in per_fold_val_targets:
                val_mask = np.isin(our_groups_set, list(vt))
                yield np.where(~val_mask)[0], np.where(val_mask)[0]
        splits_iter = _splits_from_train_run()
        log.info("  mirroring train_run=%s GroupKFold split (n_folds=%d)",
                 args.train_run, train_manifest["n_folds"])
    else:
        splits_iter = GroupKFold(n_splits=args.n_folds).split(X, y, groups)

    rows = []
    t0 = time.perf_counter()
    for fold_idx, (train_i, val_i) in enumerate(splits_iter):
        val_targets = sorted(set(groups[val_i]))
        n_pos = int(y[train_i].sum())
        n_neg = len(train_i) - n_pos
        scale_pos_weight = float(n_neg / max(n_pos, 1))

        clf_params = dict(XGBOOST_PARAMS)
        clf_params["scale_pos_weight"] = scale_pos_weight  # per-fold imbalance
        clf = XGBClassifier(**clf_params, verbosity=0, eval_metric="aucpr")

        t_fit = time.perf_counter()
        clf.fit(X[train_i], y[train_i])
        y_score = clf.predict_proba(X[val_i])[:, 1]
        wall = time.perf_counter() - t_fit
        metrics = compute_all_metrics(y[val_i], y_score, threshold=0.5)
        log.info("Fold %d  val=%s  n_train=%d  n_val=%d  pr_auc=%.4f  roc_auc=%.4f  bedroc=%.4f  ef_1pct=%.2f  %.1fs",
                 fold_idx, val_targets, len(train_i), len(val_i),
                 metrics["pr_auc"], metrics["roc_auc"],
                 metrics["bedroc"], metrics["ef_1pct"], wall)
        rows.append({
            "fold": fold_idx,
            "held_out_targets": ",".join(val_targets),
            "n_train": len(train_i),
            "n_val": len(val_i),
            "wall_s": wall,
            **{f"val_{k}": v for k, v in metrics.items()},
        })

    wall_total = time.perf_counter() - t0

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "per_fold_metrics.csv", index=False)
    pr_auc_mean = float(df["val_pr_auc"].mean())
    pr_auc_std = float(df["val_pr_auc"].std(ddof=0))

    manifest = {
        "run_id": run_id,
        "git_sha": git_sha,
        "venv": os.environ.get("VIRTUAL_ENV", "(none)"),
        "script": "scripts/12_verify_plec_groupkfold.py",
        "goldilocks_dir": str(args.goldilocks_dir),
        "plec_dir": str(args.plec_dir),
        "n_folds": args.n_folds,
        "n_rows_matched": len(X),
        "n_rows_missing": n_missing,
        "n_targets": len(targets_seen),
        "feature_dim": feature_dim,
        "wall_seconds_total": wall_total,
        "stratified_within_target_reference_pr_auc": PLEC_STRATIFIED_REFERENCE,
        "groupkfold_pr_auc_mean": pr_auc_mean,
        "groupkfold_pr_auc_std": pr_auc_std,
        "rows": rows,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    cfg = {"args": {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}}
    (args.output_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    write_report(args.output_dir, manifest, df)
    log.info("Outputs at %s", args.output_dir)
    log.info("PLEC GroupKFold mean PR-AUC: %.4f ± %.4f  (Tier 1-2 StratifiedKFold-within ref: %.3f)",
             pr_auc_mean, pr_auc_std, PLEC_STRATIFIED_REFERENCE)
    return 0


def write_report(output_dir: Path, manifest: dict, df: pd.DataFrame) -> None:
    new_threshold = manifest["groupkfold_pr_auc_mean"] + 0.05
    lines = [
        "# 12_verify_plec_groupkfold — PLEC under cross-target GroupKFold",
        "",
        f"- run_id: `{manifest['run_id']}`",
        f"- goldilocks_dir: `{manifest['goldilocks_dir']}`",
        f"- plec_dir: `{manifest['plec_dir']}`",
        f"- folds: {manifest['n_folds']}",
        f"- rows matched: {manifest['n_rows_matched']} (missing: {manifest['n_rows_missing']})",
        f"- targets: {manifest['n_targets']}",
        f"- feature dim: {manifest['feature_dim']}",
        f"- wall: **{manifest['wall_seconds_total']:.1f}s**",
        "",
        "## Per fold (PLEC + XGBoost, GroupKFold over targets)",
        "",
        "| fold | held out | n_train | n_val | pr_auc | roc_auc | bedroc | ef_1pct |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['fold']} | {r['held_out_targets']} | {int(r['n_train'])} | {int(r['n_val'])} | "
            f"{r['val_pr_auc']:.4f} | {r['val_roc_auc']:.4f} | {r['val_bedroc']:.4f} | {r['val_ef_1pct']:.2f} |"
        )
    lines += [
        "",
        "## Aggregate",
        "",
        f"- **GroupKFold PR-AUC**: {manifest['groupkfold_pr_auc_mean']:.4f} ± {manifest['groupkfold_pr_auc_std']:.4f}",
        f"- StratifiedKFold-within-target reference (Tier 1-2): {manifest['stratified_within_target_reference_pr_auc']:.3f}",
        f"- Δ (GroupKFold − StratifiedKFold-within): {manifest['groupkfold_pr_auc_mean'] - manifest['stratified_within_target_reference_pr_auc']:+.4f}",
        "",
        "## Implications for §5 gate 1",
        "",
        f"- The plan's gate-1 threshold (PLEC + 0.05 = 0.424) was derived from Tier 1-2's StratifiedKFold-within-target number (0.374).",
        f"- Under the GroupKFold protocol the plan actually mandates, PLEC achieves **{manifest['groupkfold_pr_auc_mean']:.3f}**.",
        f"- Updated gate-1 threshold (PLEC GroupKFold + 0.05): **≥ {new_threshold:.3f}**.",
        "",
        "## Notes",
        "",
        "_Filled in after review by the operator._",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
