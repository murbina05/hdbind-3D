"""
13_eval_decoys.py — Unified TIER3_DECOY_BIAS_PLAN Phase 5 eval against any of
the four decoy backgrounds: dudez | goldilocks | extrema | ad.

Subsumes (does NOT replace; legacy callers in 12_verify_plec_groupkfold and
manual workflows still target 11/17 directly):
  * scripts/11_eval_goldilocks.py — Goldilocks-only Gate 1
  * scripts/17_eval_ad_decoys.py   — AD-only Gate 1 with rmsd_bucket join
  * the implicit "M2-on-DUDE-Z" eval that's recorded in 07_train_egnn's
    per_fold_metrics.csv

Pipeline (mirrors 11/17):
  1. Replay the train-time fold composition via load_canonical_groupkfold.
  2. For each fold: load fold_K/model.pt, filter the eval LMDB's index.csv to
     fold-K's val_targets, evaluate.
  3. Aggregate per-fold metrics + per-target metrics. For --decoy-source ad,
     attach `rmsd_bucket` from the Phase 3 alignment manifest.
  4. Write per_fold_metrics.csv, per_target_metrics.csv, predictions.csv,
     manifest.json, REPORT.md.

Outputs to outputs/13_eval_decoys/<run_id>/<source>/.

Usage:
  # AD with the GroupKFold M2 — the headline Phase 5 §4 number:
  python scripts/13_eval_decoys.py --decoy-source ad \
      --train-run outputs/07_train_egnn/20260505-000911-1662f0b-m2 \
      --run-id m2gkf-on-ad

  # Extrema, ditto:
  python scripts/13_eval_decoys.py --decoy-source extrema \
      --train-run outputs/07_train_egnn/20260505-000911-1662f0b-m2

  # DUDE-Z self-eval (re-derivation of the 0.150 number):
  python scripts/13_eval_decoys.py --decoy-source dudez \
      --train-run outputs/07_train_egnn/20260505-000911-1662f0b-m2
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

log = logging.getLogger("13_eval_decoys")

# Per-source metadata: manifest field used to identify the LMDB, the legacy
# `decoy_set` CamelCase that auto-discovery should match, and the gate-1 PLEC
# reference (where known). All three property-unmatched decoy sets share the
# corrected GroupKFold-PLEC reference; DUDE-Z's gate-1 threshold uses the
# stratified-CV PLEC reference (0.374) per Tier 1–2 — but since DUDE-Z eval
# under GroupKFold is itself the leakage probe, the gate-1 reading on DUDE-Z
# is informational only.
SOURCES = {
    "dudez":      {"decoy_set": "DUDE_Z",
                   "plec_ref": 0.374, "plec_ref_kind": "stratified-CV (Tier 1–2)"},
    "goldilocks": {"decoy_set": "Goldilocks",
                   "plec_ref": 0.023, "plec_ref_kind": "GroupKFold (corrected)"},
    "extrema":    {"decoy_set": "Extrema",
                   "plec_ref": 0.023, "plec_ref_kind": "GroupKFold (proxy; AD-specific TBD)"},
    "ad":         {"decoy_set": "AD",
                   "plec_ref": 0.023, "plec_ref_kind": "GroupKFold (proxy; AD-specific TBD)"},
}

NODE_FEATURE_DIM = 10
EDGE_ATTR_DIM = 3

# RMSD-bucket boundaries (Å) — applied only when decoy_source == 'ad'.
RMSD_BUCKETS = [
    ("numerical_precision", 0.0, 0.001),
    ("sub_angstrom",        0.001, 1.0),
    ("major",               1.0, float("inf")),
]

_LMDB_ENV_CACHE: dict[str, "lmdb.Environment"] = {}


def _get_lmdb_env(path: str) -> "lmdb.Environment":
    env = _LMDB_ENV_CACHE.get(path)
    if env is None:
        env = lmdb.open(path, readonly=True, lock=False, subdir=False,
                        readahead=True, meminit=False, max_readers=512)
        _LMDB_ENV_CACHE[path] = env
    return env


def _to_lig_only(data):
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


def _bucket_for(rmsd: float) -> str:
    for name, lo, hi in RMSD_BUCKETS:
        if lo <= rmsd < hi or (hi == float("inf") and rmsd >= lo):
            return name
    return "unknown"


def _load_rmsd_buckets(align_csv: Path) -> dict[str, dict]:
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


def _resolve_eval_dir(decoy_source: str, override: Path | None) -> Path:
    """Auto-discover the latest 06 run for `decoy_source`, or use override."""
    if override is not None:
        return override.resolve()
    base = PROJECT_ROOT / "outputs" / "06_build_egnn_dataset"
    expected_legacy = SOURCES[decoy_source]["decoy_set"]
    candidates = []
    for d in base.iterdir():
        if not d.is_dir() and not d.is_symlink():
            continue
        # Walk both the run dir itself (legacy flat layout) and run/<source>/
        # (current nested layout).
        for sub in (d / decoy_source, d):
            m = sub / "manifest.json"
            if not m.exists():
                continue
            try:
                mj = json.loads(m.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            ok = (mj.get("decoy_source") == decoy_source
                  or mj.get("decoy_set") == expected_legacy)
            if ok:
                candidates.append((sub.stat().st_mtime, sub))
                break
    if not candidates:
        raise FileNotFoundError(
            f"No 06 run with decoy_source={decoy_source} (or decoy_set={expected_legacy}) "
            f"under {base}. Build with: "
            f"python scripts/06_build_egnn_dataset.py --decoy-source {decoy_source} ..."
        )
    candidates.sort()
    return candidates[-1][1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--decoy-source", required=True, choices=list(SOURCES.keys()),
                   help="Which decoy background to evaluate against.")
    p.add_argument("--train-run", type=Path, required=True,
                   help="07_train_egnn run dir (has manifest.json + fold_K/model.pt).")
    p.add_argument("--eval-dir", type=Path, default=None,
                   help="06 run dir for the eval LMDB (default: latest matching --decoy-source).")
    p.add_argument("--alignment-csv", type=Path, default=None,
                   help="Phase 3 per_target.csv with post_rmsd_a (only used for --decoy-source ad). "
                        "Default: outputs/16_align_ad_decoys/all41-20260508/per_target.csv")
    p.add_argument("--variant", choices=list(_VARIANT_TRANSFORMS), default=None,
                   help="Graph variant; defaults to the train manifest's variant.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    args.train_run = args.train_run.resolve()
    if not (args.train_run / "manifest.json").exists():
        sys.stderr.write(f"No manifest.json in {args.train_run}\n")
        return 2

    eval_dir = _resolve_eval_dir(args.decoy_source, args.eval_dir)
    eval_manifest = json.loads((eval_dir / "manifest.json").read_text())

    run_id = _make_run_id(args.run_id)
    if args.output_dir is None:
        args.output_dir = (PROJECT_ROOT / "outputs" / "13_eval_decoys"
                           / run_id / args.decoy_source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=logging.INFO, log_file=args.output_dir / "eval.log")
    log.setLevel(logging.INFO)

    train_manifest = json.loads((args.train_run / "manifest.json").read_text())
    train_variant = train_manifest.get("variant", "full")
    if args.variant is None:
        args.variant = train_variant
    is_gate1_eligible = (args.variant == "full")

    src_meta = SOURCES[args.decoy_source]
    plec_ref = src_meta["plec_ref"]
    gate1_threshold = plec_ref + 0.05

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    git_sha = _git_short_sha()

    log.info("=" * 70)
    log.info("13_eval_decoys starting")
    log.info("  run_id: %s", run_id)
    log.info("  decoy_source: %s   (eval_dir=%s)", args.decoy_source, eval_dir)
    log.info("  train_run: %s  (cv=%s, variant=%s)",
             args.train_run, train_manifest.get("cv", "?"), train_variant)
    log.info("  device: %s   PLEC ref (%s): %.3f   gate-1 threshold: %.3f",
             device, src_meta["plec_ref_kind"], plec_ref, gate1_threshold)

    # Optional rmsd_bucket lookup (AD only).
    rmsd_lookup: dict[str, dict] = {}
    if args.decoy_source == "ad":
        if args.alignment_csv is None:
            args.alignment_csv = (PROJECT_ROOT / "outputs" / "16_align_ad_decoys"
                                  / "all41-20260508" / "per_target.csv")
        if args.alignment_csv.exists():
            rmsd_lookup = _load_rmsd_buckets(args.alignment_csv.resolve())
        else:
            log.warning("Alignment CSV %s not found; rmsd_bucket column will be 'unknown'.",
                        args.alignment_csv)

    dude_metrics = pd.read_csv(args.train_run / "per_fold_metrics.csv")

    train_targets = list(train_manifest["targets"])
    n_folds = train_manifest["n_folds"]

    train_idx = pd.read_csv(Path(train_manifest["dataset_dir"]) / "index.csv",
                            dtype={"key": str})
    train_idx = train_idx[train_idx["target"].isin(set(train_targets))].reset_index(drop=True)
    train_targets_arr = train_idx["target"].values

    eval_idx = pd.read_csv(eval_dir / "index.csv", dtype={"key": str})
    eval_targets_present = set(eval_idx["target"].unique())
    missing_in_eval = sorted(set(train_targets) - eval_targets_present)
    if missing_in_eval:
        log.warning(
            "Train targets not present in %s LMDB (no decoys/actives): %s. "
            "These contribute zero rows to their fold's val set.",
            args.decoy_source, missing_in_eval,
        )

    fold_splits = load_canonical_groupkfold(args.train_run / "manifest.json")
    fold_paths = sorted(args.train_run.glob("fold_*/model.pt"))
    if len(fold_paths) != n_folds:
        log.warning("Found %d fold_K/model.pt files but train manifest says n_folds=%d",
                    len(fold_paths), n_folds)

    fold_rows = []
    pred_rows = []
    t0 = time.perf_counter()

    for fold_idx, (_train_i, val_i) in enumerate(fold_splits):
        if fold_idx >= len(fold_paths):
            break
        model_path = fold_paths[fold_idx]
        val_targets = sorted(set(train_targets_arr[val_i]))

        sub = eval_idx[eval_idx["target"].isin(val_targets)]
        val_keys = list(sub["key"].astype(str).values)
        val_meta = sub[["target", "complex_id", "label"]].reset_index(drop=True)

        log.info("─" * 60)
        log.info("Fold %d  n_val_targets=%d  n_val_keys=%d  ckpt=%s",
                 fold_idx, len(val_targets), len(val_keys),
                 model_path.relative_to(args.train_run))

        if not val_keys:
            log.warning("Fold %d: 0 val keys — skipping.", fold_idx)
            continue

        model = EGNN(
            in_node_dim=NODE_FEATURE_DIM, edge_attr_dim=EDGE_ATTR_DIM,
            h_dim=train_manifest["h_dim"], n_layers=train_manifest["n_layers"],
        ).to(device)
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)

        ds = LMDBPyGDataset(eval_dir / "dataset.lmdb", val_keys, variant=args.variant)
        loader = PyGDataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        t_a = time.perf_counter()
        y_true, y_score = _evaluate(model, loader, device)
        wall = time.perf_counter() - t_a

        n_pos = int(y_true.sum())
        n_neg = len(y_true) - n_pos
        if n_pos == 0:
            log.warning("Fold %d: 0 positives — skipping metrics.", fold_idx)
            continue
        metrics = compute_all_metrics(y_true.astype(int), y_score, threshold=0.5)

        dude_row = dude_metrics[dude_metrics["fold"] == fold_idx]
        dude_pr_auc = float(dude_row["val_pr_auc"].iloc[0]) if not dude_row.empty else float("nan")
        dude_bedroc = float(dude_row["val_bedroc"].iloc[0]) if not dude_row.empty else float("nan")

        log.info("  %-10s pr_auc=%.4f roc_auc=%.4f bedroc=%.4f ef_1pct=%.2f ef_5pct=%.2f n_pos=%d n_neg=%d %.1fs",
                 args.decoy_source, metrics["pr_auc"], metrics["roc_auc"],
                 metrics["bedroc"], metrics["ef_1pct"], metrics["ef_5pct"],
                 n_pos, n_neg, wall)

        fold_rows.append({
            "fold": fold_idx,
            "n_val_targets": len(val_targets),
            "n_val": len(val_keys),
            "n_val_pos": n_pos,
            "n_val_neg": n_neg,
            "wall_s": wall,
            **{f"eval_{k}": v for k, v in metrics.items()},
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

    # Per-target aggregation: average score across folds per (target, complex_id).
    per_target_rows = []
    if not df_pred.empty:
        agg = (df_pred.groupby(["target", "complex_id", "label"], as_index=False)
                      ["score"].mean())
        for tgt, sub in agg.groupby("target"):
            y_t = sub["label"].values.astype(int)
            y_s = sub["score"].values.astype(float)
            n_pos = int(y_t.sum())
            n_neg = len(y_t) - n_pos
            if n_pos == 0:
                continue
            m = compute_all_metrics(y_t, y_s, threshold=0.5)
            row = {
                "target": tgt,
                "n_pos": n_pos,
                "n_neg": n_neg,
                **{k: float(v) for k, v in m.items()},
            }
            if rmsd_lookup:
                info = rmsd_lookup.get(tgt, {"post_rmsd_a": np.nan,
                                             "post_centroid_offset_a": np.nan,
                                             "rmsd_bucket": "unknown"})
                row.update({
                    "post_rmsd_a": info["post_rmsd_a"],
                    "post_centroid_offset_a": info["post_centroid_offset_a"],
                    "rmsd_bucket": info["rmsd_bucket"],
                })
            per_target_rows.append(row)
    df_target = pd.DataFrame(per_target_rows).sort_values("target").reset_index(drop=True)
    df_target.to_csv(args.output_dir / "per_target_metrics.csv", index=False)

    pr_auc_mean = float(df_fold["eval_pr_auc"].mean()) if not df_fold.empty else float("nan")
    gate1_pass = bool(is_gate1_eligible and pr_auc_mean >= gate1_threshold)

    manifest = {
        "run_id": run_id,
        "git_sha": git_sha,
        "venv": os.environ.get("VIRTUAL_ENV", "(none)"),
        "script": "scripts/13_eval_decoys.py",
        "decoy_source": args.decoy_source,
        "decoy_set_legacy": src_meta["decoy_set"],
        "train_run": str(args.train_run),
        "train_cv": train_manifest.get("cv"),
        "train_variant": train_variant,
        "eval_dir": str(eval_dir),
        "alignment_csv": (str(args.alignment_csv) if args.alignment_csv else None),
        "variant": args.variant,
        "n_folds": int(len(fold_rows)),
        "n_targets_eval": int(len(df_target)),
        "missing_in_eval": missing_in_eval,
        "wall_seconds_total": wall_total,
        "plec_ref": plec_ref,
        "plec_ref_kind": src_meta["plec_ref_kind"],
        "gate1_threshold": gate1_threshold,
        "gate1_eligible": is_gate1_eligible,
        "gate1_pr_auc_mean": pr_auc_mean,
        "gate1_pass": gate1_pass,
        "rows": fold_rows,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    cfg = {"args": {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}}
    (args.output_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    write_report(args.output_dir, manifest, df_fold, df_target)
    log.info("Outputs at %s", args.output_dir)
    if is_gate1_eligible:
        log.info("Gate 1 (%s): %s  (M2 mean PR-AUC %.4f vs threshold %.4f)",
                 args.decoy_source, "PASS" if gate1_pass else "FAIL",
                 pr_auc_mean, gate1_threshold)
    return 0


def write_report(output_dir: Path, manifest: dict,
                 df_fold: pd.DataFrame, df_target: pd.DataFrame) -> None:
    is_eligible = manifest.get("gate1_eligible", True)
    gate1 = ("PASS" if manifest["gate1_pass"] else "FAIL") if is_eligible else "N/A (variant != full)"
    src = manifest["decoy_source"]
    cv = manifest.get("train_cv", "?")
    cv_note = ""
    if cv == "stratified":
        cv_note = (
            "\n> **CV note:** train_run is stratified-CV (every fold's val targets = "
            "all train targets). Each fold therefore evaluates the full eval index, "
            "and per-fold std reflects ensemble disagreement among the 5 checkpoints "
            "rather than disjoint-target generalization.\n"
        )

    lines = [
        f"# 13_eval_decoys — Phase 5 evaluation against `{src}` background",
        "",
        f"- run_id: `{manifest['run_id']}`",
        f"- git_sha: `{manifest['git_sha']}`",
        f"- decoy_source: **{src}**  (legacy decoy_set: `{manifest['decoy_set_legacy']}`)",
        f"- train_run: `{manifest['train_run']}`  (cv=`{cv}`, variant=`{manifest['train_variant']}`)",
        f"- variant evaluated: `{manifest.get('variant', 'full')}`",
        f"- eval_dir: `{manifest['eval_dir']}`",
        f"- folds evaluated: {manifest['n_folds']}",
        f"- targets in per-target table: {manifest['n_targets_eval']}",
        f"- wall: **{manifest['wall_seconds_total']:.1f}s**",
        f"- missing-in-eval targets: {manifest['missing_in_eval'] or 'none'}",
        cv_note,
        "## Per fold (eval vs DUDE-Z, same checkpoint)",
        "",
        "| fold | n_val | n_pos | eval pr_auc | eval roc_auc | eval bedroc | eval ef_1pct | eval ef_5pct | dude pr_auc | Δ pr_auc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df_fold.iterrows():
        lines.append(
            f"| {int(r['fold'])} | {int(r['n_val'])} | {int(r['n_val_pos'])} | "
            f"{r['eval_pr_auc']:.4f} | {r['eval_roc_auc']:.4f} | {r['eval_bedroc']:.4f} | "
            f"{r['eval_ef_1pct']:.2f} | {r['eval_ef_5pct']:.2f} | "
            f"{r['dude_pr_auc']:.4f} | {r['delta_pr_auc']:+.4f} |"
        )
    if not df_fold.empty:
        lines += [
            "",
            "## Aggregate (mean ± std across folds)",
            "",
            f"- **{src} PR-AUC**: {df_fold['eval_pr_auc'].mean():.4f} ± {df_fold['eval_pr_auc'].std(ddof=0):.4f}",
            f"- {src} ROC-AUC:   {df_fold['eval_roc_auc'].mean():.4f} ± {df_fold['eval_roc_auc'].std(ddof=0):.4f}",
            f"- {src} BEDROC:    {df_fold['eval_bedroc'].mean():.4f} ± {df_fold['eval_bedroc'].std(ddof=0):.4f}",
            f"- {src} EF@1%:     {df_fold['eval_ef_1pct'].mean():.2f} ± {df_fold['eval_ef_1pct'].std(ddof=0):.2f}",
            f"- {src} EF@5%:     {df_fold['eval_ef_5pct'].mean():.2f} ± {df_fold['eval_ef_5pct'].std(ddof=0):.2f}",
            f"- DUDE-Z PR-AUC (same folds): {df_fold['dude_pr_auc'].mean():.4f} ± {df_fold['dude_pr_auc'].std(ddof=0):.4f}",
            f"- ΔPR-AUC ({src} − DUDE-Z): {df_fold['delta_pr_auc'].mean():+.4f} ± {df_fold['delta_pr_auc'].std(ddof=0):.4f}",
            "",
            f"## §5 gate 1 — beat PLEC + 0.05 = {manifest['gate1_threshold']:.3f}  (PLEC ref: {manifest['plec_ref_kind']})",
            "",
            f"- mean {src} PR-AUC: **{manifest['gate1_pr_auc_mean']:.4f}**",
            f"- PLEC reference: {manifest['plec_ref']:.3f}",
            f"- Gate 1 threshold: ≥ {manifest['gate1_threshold']:.3f}",
            f"- **Gate 1: {gate1}**",
        ]
    else:
        lines.append("\n_No folds evaluated successfully._\n")

    if not df_target.empty:
        has_buckets = "rmsd_bucket" in df_target.columns
        lines += [
            "",
            "## Per-target metrics (aggregated across folds)" + (
                " — with `rmsd_bucket`" if has_buckets else ""),
            "",
        ]
        if has_buckets:
            lines.append("| target | rmsd_bucket | post_rmsd Å | n_pos | n_neg | pr_auc | roc_auc | bedroc | ef_1pct | ef_5pct |")
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
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
            lines += ["", "### Bucket aggregates", "",
                      "| bucket | n_targets | mean pr_auc | std pr_auc | mean bedroc |",
                      "|---|---:|---:|---:|---:|"]
            for n, _, _ in RMSD_BUCKETS:
                sub = df_target[df_target["rmsd_bucket"] == n]
                if sub.empty:
                    continue
                lines.append(
                    f"| {n} | {len(sub)} | {sub['pr_auc'].mean():.4f} | "
                    f"{sub['pr_auc'].std(ddof=0):.4f} | {sub['bedroc'].mean():.4f} |"
                )
        else:
            lines.append("| target | n_pos | n_neg | pr_auc | roc_auc | bedroc | ef_1pct | ef_5pct |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for _, r in df_target.iterrows():
                lines.append(
                    f"| {r['target']} | {int(r['n_pos'])} | {int(r['n_neg'])} | "
                    f"{r['pr_auc']:.4f} | {r['roc_auc']:.4f} | {r['bedroc']:.4f} | "
                    f"{r['ef_1pct']:.2f} | {r['ef_5pct']:.2f} |"
                )

    lines += ["", "## Notes", "", "_Filled in after review by the operator._", ""]
    (output_dir / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
