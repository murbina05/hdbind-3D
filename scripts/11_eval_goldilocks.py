"""
11_eval_goldilocks.py — TIER3_EGNN_PLAN.md §5 gate-1 evaluation on Goldilocks.

Plan §5: "all gates evaluated on Goldilocks decoys, GroupKFold protocol."
Gate 1: M2 PR-AUC ≥ PLEC + 0.05 = 0.424 on Goldilocks. PLEC reference 0.374
is from project memory (Tier 1–2). The actives are identical to DUDE-Z; only
the decoys differ. Goldilocks decoys are property-unmatched and harder, so
this is the honest cross-target benchmark.

Pipeline:
  1. Load M2 from a 07_train_egnn run dir (variant=full).
  2. Replay the GroupKFold split using the train manifest's `targets` and
     `n_folds` (deterministic — sklearn.GroupKFold has no seed).
  3. For each fold: load fold_K/model.pt, identify val_targets, filter the
     Goldilocks LMDB's index.csv to those targets, evaluate.
  4. Compare to PLEC and to the same fold's DUDE-Z metrics (read from the
     train manifest's per-fold metrics).

Outputs to outputs/11_eval_goldilocks/<run_id>/:
  per_fold_metrics.csv   one row per fold; metrics + Δ vs DUDE-Z
  manifest.json
  config.yaml
  REPORT.md
  eval.log

Usage:
  python scripts/11_eval_goldilocks.py --train-run <07-run-dir>
  # Auto-discovers the latest decoy_set=Goldilocks 06 run for --goldilocks-dir
  # if not given.
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

log = logging.getLogger("11_eval_goldilocks")

# Reference PR-AUC for PLEC on Goldilocks (project memory; verify before signoff)
PLEC_GOLDILOCKS_PR_AUC = 0.374
GATE_1_THRESHOLD = PLEC_GOLDILOCKS_PR_AUC + 0.05  # 0.424

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
    """M1 variant — drop protein atoms, keep ligand subgraph. Mirrors 07."""
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
        self._cache: list | None = (
            [None] * len(self.keys) if self.transform is not None else None
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self._cache is not None and self._cache[idx] is not None:
            return self._cache[idx]
        env = _get_lmdb_env(self.lmdb_path)
        with env.begin() as txn:
            payload = txn.get(self.keys[idx])
        if payload is None:
            raise KeyError(f"Missing LMDB key {self.keys[idx]!r}")
        d = pickle.loads(payload)
        if self.transform is not None:
            d = self.transform(d)
            self._cache[idx] = d
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


def _latest_goldilocks_dataset_dir() -> Path:
    base = PROJECT_ROOT / "outputs" / "06_build_egnn_dataset"
    candidates = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        m = d / "manifest.json"
        if not m.exists():
            continue
        try:
            mj = json.loads(m.read_text())
            if mj.get("decoy_set") == "Goldilocks":
                candidates.append((d.stat().st_mtime, d))
        except (OSError, json.JSONDecodeError):
            continue
    if not candidates:
        raise FileNotFoundError(
            f"No 06 run with decoy_set=Goldilocks under {base}. "
            f"Run: python scripts/06_build_egnn_dataset.py --all-targets --decoy-set Goldilocks"
        )
    candidates.sort()
    return candidates[-1][1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--train-run", type=Path, required=True,
                   help="Path to a 07_train_egnn run dir (M2 / variant=full).")
    p.add_argument("--goldilocks-dir", type=Path, default=None,
                   help="Path to a 06 run dir with decoy_set=Goldilocks (default: latest).")
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

    if args.goldilocks_dir is None:
        args.goldilocks_dir = _latest_goldilocks_dataset_dir()
    args.goldilocks_dir = args.goldilocks_dir.resolve()
    gold_manifest = json.loads((args.goldilocks_dir / "manifest.json").read_text())
    if gold_manifest.get("decoy_set") != "Goldilocks":
        sys.stderr.write(
            f"--goldilocks-dir {args.goldilocks_dir} has decoy_set="
            f"{gold_manifest.get('decoy_set')!r}, expected 'Goldilocks'\n"
        )
        return 2

    run_id = _make_run_id(args.run_id)
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "outputs" / "11_eval_goldilocks" / run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=logging.INFO, log_file=args.output_dir / "eval.log")
    log.setLevel(logging.INFO)

    train_manifest = json.loads((args.train_run / "manifest.json").read_text())
    train_variant = train_manifest.get("variant", "full")
    if args.variant is None:
        args.variant = train_variant
    if args.variant != train_variant:
        log.warning("--variant=%s but train_run variant=%s; results may be misleading.",
                    args.variant, train_variant)
    is_gate1_eligible = (args.variant == "full")
    if not is_gate1_eligible:
        log.info("variant=%s — this is M1-style; gate 1 (M2 vs PLEC) does not apply here.",
                 args.variant)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    git_sha = _git_short_sha()

    log.info("=" * 70)
    log.info("11_eval_goldilocks starting")
    log.info("  run_id: %s", run_id)
    log.info("  git_sha: %s", git_sha)
    log.info("  train_run: %s", args.train_run)
    log.info("  goldilocks_dir: %s", args.goldilocks_dir)
    log.info("  device: %s  PLEC ref: %.3f  gate-1 threshold: %.3f",
             device, PLEC_GOLDILOCKS_PR_AUC, GATE_1_THRESHOLD)

    # Load DUDE-Z per-fold metrics for the side-by-side delta.
    dude_metrics = pd.read_csv(args.train_run / "per_fold_metrics.csv")
    # The training script stored val_targets as a stringified list; we replay
    # the GroupKFold ourselves below so we don't depend on its format.

    # Replay the *train-time* GroupKFold split (over DUDE-Z's 43 targets).
    # For each fold's val_targets, filter the Goldilocks index to those
    # targets. This preserves the original train/val target boundaries:
    # we evaluate fold_K/model.pt only on targets that fold_K never saw.
    train_targets = list(train_manifest["targets"])
    n_folds = train_manifest["n_folds"]

    train_idx = pd.read_csv(Path(train_manifest["dataset_dir"]) / "index.csv",
                            dtype={"key": str})
    train_idx = train_idx[train_idx["target"].isin(set(train_targets))].reset_index(drop=True)
    train_targets_arr = train_idx["target"].values

    gold_idx = pd.read_csv(args.goldilocks_dir / "index.csv", dtype={"key": str})
    gold_targets_present = set(gold_idx["target"].unique())
    missing_in_gold = sorted(set(train_targets) - gold_targets_present)
    if missing_in_gold:
        log.warning(
            "Train targets not present in Goldilocks LMDB (decoys missing): %s. "
            "These contribute zero rows to their fold's Goldilocks val set.",
            missing_in_gold,
        )

    fold_splits = load_canonical_groupkfold(args.train_run / "manifest.json")
    if len(fold_splits) != n_folds:
        log.warning("load_canonical_groupkfold returned %d folds but manifest says n_folds=%d",
                    len(fold_splits), n_folds)

    fold_paths = sorted(args.train_run.glob("fold_*/model.pt"))
    if len(fold_paths) != n_folds:
        log.warning("Found %d fold_K/model.pt files but train manifest says n_folds=%d",
                    len(fold_paths), n_folds)

    rows = []
    t0 = time.perf_counter()
    for fold_idx, (_train_i, val_i) in enumerate(fold_splits):
        if fold_idx >= len(fold_paths):
            break
        model_path = fold_paths[fold_idx]
        val_targets = sorted(set(train_targets_arr[val_i]))
        val_keys = list(
            gold_idx[gold_idx["target"].isin(val_targets)]["key"].astype(str).values
        )

        log.info("─" * 60)
        log.info("Fold %d  val=%s  n_val=%d  ckpt=%s",
                 fold_idx, val_targets, len(val_keys),
                 model_path.relative_to(args.train_run))

        model = EGNN(
            in_node_dim=NODE_FEATURE_DIM, edge_attr_dim=EDGE_ATTR_DIM,
            h_dim=train_manifest["h_dim"], n_layers=train_manifest["n_layers"],
        ).to(device)
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)

        ds = LMDBPyGDataset(args.goldilocks_dir / "dataset.lmdb", val_keys, variant=args.variant)
        loader = PyGDataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        t_a = time.perf_counter()
        y_true, y_score = _evaluate(model, loader, device)
        wall = time.perf_counter() - t_a
        n_pos = int(y_true.sum())
        n_neg = len(y_true) - n_pos
        if n_pos == 0:
            log.warning("Fold %d val has 0 positives — skipping metrics.", fold_idx)
            continue
        metrics = compute_all_metrics(y_true.astype(int), y_score, threshold=0.5)

        # DUDE-Z baseline for the same fold (for delta).
        dude_row = dude_metrics[dude_metrics["fold"] == fold_idx]
        dude_pr_auc = float(dude_row["val_pr_auc"].iloc[0]) if not dude_row.empty else float("nan")
        dude_bedroc = float(dude_row["val_bedroc"].iloc[0]) if not dude_row.empty else float("nan")

        log.info("  Goldilocks  pr_auc=%.4f  roc_auc=%.4f  bedroc=%.4f  ef_1pct=%.2f  "
                 "n_pos=%d  n_neg=%d  %.1fs",
                 metrics["pr_auc"], metrics["roc_auc"], metrics["bedroc"],
                 metrics["ef_1pct"], n_pos, n_neg, wall)
        log.info("  DUDE-Z (same fold)  pr_auc=%.4f  bedroc=%.4f  Δ_pr_auc=%+.4f  Δ_bedroc=%+.4f",
                 dude_pr_auc, dude_bedroc,
                 metrics["pr_auc"] - dude_pr_auc, metrics["bedroc"] - dude_bedroc)

        rows.append({
            "fold": fold_idx,
            "held_out_targets": ",".join(val_targets),
            "n_val": len(val_keys),
            "n_val_pos": n_pos,
            "n_val_neg": n_neg,
            "wall_s": wall,
            **{f"gold_{k}": v for k, v in metrics.items()},
            "dude_pr_auc": dude_pr_auc,
            "dude_bedroc": dude_bedroc,
            "delta_pr_auc": metrics["pr_auc"] - dude_pr_auc,
            "delta_bedroc": metrics["bedroc"] - dude_bedroc,
        })

    wall_total = time.perf_counter() - t0
    log.info("All folds done in %.1fs", wall_total)

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "per_fold_metrics.csv", index=False)

    pr_auc_mean = float(df["gold_pr_auc"].mean())
    gate1_pass = bool(is_gate1_eligible and pr_auc_mean >= GATE_1_THRESHOLD)

    manifest = {
        "run_id": run_id,
        "git_sha": git_sha,
        "venv": os.environ.get("VIRTUAL_ENV", "(none)"),
        "script": "scripts/11_eval_goldilocks.py",
        "train_run": str(args.train_run),
        "goldilocks_dir": str(args.goldilocks_dir),
        "variant": args.variant,
        "n_folds": int(len(rows)),
        "missing_in_goldilocks": missing_in_gold,
        "wall_seconds_total": wall_total,
        "plec_goldilocks_pr_auc": PLEC_GOLDILOCKS_PR_AUC,
        "gate1_threshold": GATE_1_THRESHOLD,
        "gate1_eligible": is_gate1_eligible,
        "gate1_pr_auc_mean": pr_auc_mean,
        "gate1_pass": gate1_pass,
        "rows": rows,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    cfg = {"args": {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}}
    (args.output_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    write_report(args.output_dir, manifest, df)
    log.info("Outputs at %s", args.output_dir)
    if is_gate1_eligible:
        log.info("Gate 1: %s  (M2 mean PR-AUC %.4f vs threshold %.4f)",
                 "PASS" if gate1_pass else "FAIL",
                 pr_auc_mean, GATE_1_THRESHOLD)
    else:
        log.info("variant=%s — gate 1 not applicable; mean PR-AUC %.4f",
                 args.variant, pr_auc_mean)
    return 0


def write_report(output_dir: Path, manifest: dict, df: pd.DataFrame) -> None:
    is_eligible = manifest.get("gate1_eligible", True)
    gate1 = ("PASS" if manifest["gate1_pass"] else "FAIL") if is_eligible else "N/A (variant != full)"
    lines = [
        "# 11_eval_goldilocks — §5 gate-1 evaluation on Goldilocks decoys",
        "",
        f"- run_id: `{manifest['run_id']}`",
        f"- git_sha: `{manifest['git_sha']}`",
        f"- train_run: `{manifest['train_run']}`",
        f"- variant: `{manifest.get('variant', 'full')}`",
        f"- goldilocks_dir: `{manifest['goldilocks_dir']}`",
        f"- folds evaluated: {manifest['n_folds']}",
        f"- wall: **{manifest['wall_seconds_total']:.1f}s**",
        f"- missing-in-Goldilocks targets: {manifest['missing_in_goldilocks'] or 'none'}",
        "",
        "## Per fold (Goldilocks vs DUDE-Z, same M2 checkpoint)",
        "",
        "| fold | held out | n_val | n_pos | gold pr_auc | gold roc_auc | gold bedroc | gold ef_1pct | dude pr_auc | Δ pr_auc | Δ bedroc |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['fold']} | {r['held_out_targets']} | {int(r['n_val'])} | {int(r['n_val_pos'])} | "
            f"{r['gold_pr_auc']:.4f} | {r['gold_roc_auc']:.4f} | {r['gold_bedroc']:.4f} | "
            f"{r['gold_ef_1pct']:.2f} | {r['dude_pr_auc']:.4f} | "
            f"{r['delta_pr_auc']:+.4f} | {r['delta_bedroc']:+.4f} |"
        )
    lines += [
        "",
        "## Aggregate (mean ± std across folds)",
        "",
        f"- **Goldilocks PR-AUC**: {df['gold_pr_auc'].mean():.4f} ± {df['gold_pr_auc'].std(ddof=0):.4f}",
        f"- Goldilocks ROC-AUC:  {df['gold_roc_auc'].mean():.4f} ± {df['gold_roc_auc'].std(ddof=0):.4f}",
        f"- Goldilocks BEDROC:   {df['gold_bedroc'].mean():.4f} ± {df['gold_bedroc'].std(ddof=0):.4f}",
        f"- Goldilocks EF@1%:    {df['gold_ef_1pct'].mean():.2f} ± {df['gold_ef_1pct'].std(ddof=0):.2f}",
        f"- DUDE-Z PR-AUC:       {df['dude_pr_auc'].mean():.4f} ± {df['dude_pr_auc'].std(ddof=0):.4f}",
        f"- ΔPR-AUC (Goldilocks − DUDE-Z): {df['delta_pr_auc'].mean():+.4f} ± {df['delta_pr_auc'].std(ddof=0):.4f}",
        f"- ΔBEDROC (Goldilocks − DUDE-Z): {df['delta_bedroc'].mean():+.4f} ± {df['delta_bedroc'].std(ddof=0):.4f}",
        "",
        f"## §5 gate 1 — beat PLEC + 0.05 = {manifest['gate1_threshold']:.3f}",
        "",
        f"- mean Goldilocks PR-AUC: **{manifest['gate1_pr_auc_mean']:.4f}**",
        f"- PLEC Goldilocks PR-AUC reference: {manifest['plec_goldilocks_pr_auc']:.3f}",
        f"- Gate 1 threshold: ≥ {manifest['gate1_threshold']:.3f}",
        f"- **Gate 1: {gate1}**",
        "",
        "## Notes",
        "",
        "_Filled in after review by the operator._",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
