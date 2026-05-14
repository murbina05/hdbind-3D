"""
07_train_egnn.py — Tier 3 EGNN trainer (M2: full complex).

Loads the LMDB built by scripts/06_build_egnn_dataset.py and trains a vanilla
EGNN under the GroupKFold-over-targets protocol from TIER3_EGNN_PLAN.md §3.2.

Smoke (default): --targets AA2AR ADRB2 EGFR → GroupKFold k=min(5, n_targets) =
3-fold leave-one-target-out. 5 epochs per fold.

Optimizer: Adam, lr 1e-4. Loss: BCEWithLogitsLoss with pos_weight = n_neg/n_pos
on the train fold. Mixed precision: BF16 via torch.amp on Blackwell.

Outputs to outputs/07_train_egnn/<run_id>/:
  fold_K/{model.pt, val_predictions.csv, train_curve.csv}
  manifest.json
  per_fold_metrics.csv
  config.yaml
  REPORT.md
  train.log

Hard rules respected:
  - Single-GPU baseline only (no torch.distributed yet).
  - num_workers=0 in DataLoader (no CUDA + multiprocessing).
"""
from __future__ import annotations

import argparse
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

from config import PROJECT_ROOT, RANDOM_SEED, TARGETS_DEBUG, get_all_targets
from src.utils import canonical_split_indices, setup_logging
from src.evaluation import compute_all_metrics

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
    from torch_geometric.loader import DataLoader as PyGDataLoader
except ImportError as e:
    sys.stderr.write(f"Missing torch/torch_geometric: {e}\n")
    sys.exit(1)

try:
    import lmdb
except ImportError:
    sys.stderr.write("Missing 'lmdb' in venv\n")
    sys.exit(1)

from src.models.egnn import EGNN

log = logging.getLogger("07_train_egnn")


# ── LMDB-backed PyG Dataset ────────────────────────────────────────────────

# LMDB only allows one env per file per process; cache so train + val datasets
# (and successive folds) share a single handle.
_LMDB_ENV_CACHE: dict[str, "lmdb.Environment"] = {}


def _get_lmdb_env(path: str) -> "lmdb.Environment":
    env = _LMDB_ENV_CACHE.get(path)
    if env is None:
        env = lmdb.open(
            path, readonly=True, lock=False, subdir=False,
            readahead=True, meminit=False, max_readers=512,
        )
        _LMDB_ENV_CACHE[path] = env
    return env


def _to_lig_only(data):
    """M1 variant: drop all protein atoms, leaving the ligand subgraph.
    Edges between the kept ligand atoms (intra-lig) are preserved; edges to
    protein atoms are dropped. PyG's Data.subgraph handles re-indexing.
    """
    lig_mask = data.x[:, -1] == 0
    return data.subgraph(lig_mask)


_VARIANT_TRANSFORMS = {
    "full": None,
    "lig_only": _to_lig_only,
}


class LMDBPyGDataset(Dataset):
    """Lazy-read PyG Data objects from a single-file LMDB.

    For non-`full` variants the transform is applied once per item and the
    transformed Data is cached in RAM. data.subgraph() is O(E) per call, so
    avoiding the per-epoch retransform turned out to be a 5x speedup on lig_only.
    Cache size scales with dataset size; ~10 KB per ligand-only complex
    (~110 MB for the 11 k-row smoke, ~1 GB for the 200 k-row full sweep).
    """

    def __init__(self, lmdb_path: str | Path, keys: list[str], variant: str = "full"):
        self.lmdb_path = str(lmdb_path)
        self.keys = [k.encode("ascii") if isinstance(k, str) else k for k in keys]
        if variant not in _VARIANT_TRANSFORMS:
            raise ValueError(f"Unknown variant {variant!r}; choose from {list(_VARIANT_TRANSFORMS)}")
        self.transform = _VARIANT_TRANSFORMS[variant]
        self._cache: list | None = None
        if self.transform is not None:
            self._cache = [None] * len(self.keys)

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
        data = pickle.loads(payload)
        if self.transform is not None:
            data = self.transform(data)
            self._cache[idx] = data
        return data


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


def _latest_dataset_dir() -> Path:
    base = PROJECT_ROOT / "outputs" / "06_build_egnn_dataset"
    if not base.is_dir():
        raise FileNotFoundError(f"No 06_build_egnn_dataset outputs at {base}")
    runs = sorted([p for p in base.iterdir() if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs under {base}")
    return runs[-1]


# ── Train / eval one fold ─────────────────────────────────────────────────

def _ligand_mask(batch_x: torch.Tensor) -> torch.Tensor:
    """Last column of x is is_protein bit; ligand mask = (is_protein == 0)."""
    return batch_x[:, -1] == 0


def _evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, scores = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(
                    batch.x, batch.pos, batch.edge_index,
                    batch.edge_attr, _ligand_mask(batch.x), batch.batch,
                )
            ys.append(batch.y.detach().float().cpu().numpy().ravel())
            scores.append(torch.sigmoid(logits.float()).cpu().numpy().ravel())
    return np.concatenate(ys), np.concatenate(scores)


def _train_fold(
    fold_idx: int,
    train_keys: list[str],
    val_keys: list[str],
    train_targets: list[str],
    val_targets: list[str],
    train_labels: np.ndarray,
    lmdb_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    fold_dir: Path,
) -> dict:
    fold_dir.mkdir(parents=True, exist_ok=True)
    log.info("─" * 60)
    log.info("Fold %d: train targets=%s val targets=%s",
             fold_idx, train_targets, val_targets)
    log.info("  train n=%d (pos=%d, neg=%d), val n=%d",
             len(train_keys), int(train_labels.sum()),
             len(train_keys) - int(train_labels.sum()), len(val_keys))

    n_pos = int(train_labels.sum())
    n_neg = len(train_labels) - n_pos
    pos_weight_val = float(n_neg / max(n_pos, 1))
    log.info("  pos_weight: %.2f", pos_weight_val)

    torch.manual_seed(RANDOM_SEED + fold_idx)

    model = EGNN(
        in_node_dim=10, edge_attr_dim=3,
        h_dim=args.h_dim, n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("  model params: %d", n_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_val, device=device)
    )

    train_ds = LMDBPyGDataset(lmdb_path, train_keys, variant=args.variant)
    val_ds = LMDBPyGDataset(lmdb_path, val_keys, variant=args.variant)
    train_loader = PyGDataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = PyGDataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    curve_rows = []
    best_pr_auc = -1.0
    best_epoch = -1
    best_state = None
    no_improve_evals = 0
    t0 = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        t_ep = time.perf_counter()
        ep_loss_sum = 0.0
        ep_n = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(
                    batch.x, batch.pos, batch.edge_index,
                    batch.edge_attr, _ligand_mask(batch.x), batch.batch,
                )
                target = batch.y.float().view(-1)
                loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            ep_loss_sum += float(loss.item()) * target.size(0)
            ep_n += target.size(0)
        train_loss = ep_loss_sum / max(ep_n, 1)

        do_eval = (
            (epoch + 1) % args.eval_every == 0
            or epoch == args.epochs - 1
            or epoch == 0  # always eval first epoch for early sanity
        )
        if not do_eval:
            ep_wall = time.perf_counter() - t_ep
            log.info("  epoch %d/%d  train_loss=%.4f  (no eval; --eval-every=%d)  %.1fs",
                     epoch + 1, args.epochs, train_loss, args.eval_every, ep_wall)
            curve_rows.append({"fold": fold_idx, "epoch": epoch,
                               "train_loss": train_loss, "wall_s": ep_wall})
            continue

        y_val, s_val = _evaluate(model, val_loader, device)
        try:
            metrics = compute_all_metrics(y_val.astype(int), s_val, threshold=0.5)
        except Exception as e:
            log.warning("metrics failed at epoch %d: %s", epoch, e)
            metrics = {"pr_auc": float("nan"), "roc_auc": float("nan"),
                       "bedroc": float("nan")}

        ep_wall = time.perf_counter() - t_ep
        log.info("  epoch %d/%d  train_loss=%.4f  val_pr_auc=%.4f  val_roc_auc=%.4f  "
                 "val_bedroc=%.4f  %.1fs",
                 epoch + 1, args.epochs, train_loss,
                 metrics.get("pr_auc", float("nan")),
                 metrics.get("roc_auc", float("nan")),
                 metrics.get("bedroc", float("nan")), ep_wall)
        curve_rows.append({"fold": fold_idx, "epoch": epoch, "train_loss": train_loss,
                           **metrics, "wall_s": ep_wall})

        if metrics.get("pr_auc", -1.0) > best_pr_auc:
            best_pr_auc = metrics["pr_auc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve_evals = 0
        else:
            no_improve_evals += 1
            if (args.early_stop_patience > 0
                    and no_improve_evals >= args.early_stop_patience):
                log.info("  early stop at epoch %d/%d (no val PR-AUC improvement for %d evals)",
                         epoch + 1, args.epochs, no_improve_evals)
                break

    fold_wall = time.perf_counter() - t0
    log.info("  fold done in %.1fs; best epoch=%d pr_auc=%.4f",
             fold_wall, best_epoch, best_pr_auc)

    if best_state is not None:
        torch.save(best_state, fold_dir / "model.pt")
    pd.DataFrame(curve_rows).to_csv(fold_dir / "train_curve.csv", index=False)

    # final val predictions using best model
    if best_state is not None:
        model.load_state_dict(best_state)
    y_val, s_val = _evaluate(model, val_loader, device)
    pd.DataFrame({
        "key": [k.decode("ascii") if isinstance(k, bytes) else k for k in val_keys],
        "y_true": y_val.astype(int),
        "y_score": s_val,
    }).to_csv(fold_dir / "val_predictions.csv", index=False)
    final_metrics = compute_all_metrics(y_val.astype(int), s_val, threshold=0.5)

    return {
        "fold": fold_idx,
        "train_targets": list(train_targets),
        "val_targets": list(val_targets),
        "n_train": len(train_keys),
        "n_val": len(val_keys),
        "n_train_pos": n_pos,
        "n_train_neg": n_neg,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_pr_auc": float(best_pr_auc),
        "fold_wall_s": fold_wall,
        **{f"val_{k}": float(v) for k, v in final_metrics.items()},
    }


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset-dir", type=Path, default=None,
                   help="Path to a 06_build_egnn_dataset run dir. Default: latest.")
    p.add_argument("--targets", nargs="+", default=None,
                   help="Restrict training to these targets (default: smoke set; pass --all-targets for all 43).")
    p.add_argument("--all-targets", action="store_true",
                   help="Train on all 43 DUDE-Z targets (mutually exclusive with --targets).")
    p.add_argument("--n-folds", type=int, default=None,
                   help="GroupKFold n_splits (default: min(5, n_targets)).")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--eval-every", type=int, default=1,
                   help="Run val eval every K epochs (default 1; e.g. 5 for long sweeps).")
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="Stop a fold if val PR-AUC hasn't improved across this many eval points. "
                        "0 disables (default).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--h-dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", type=str, default=None,
                   help="Override device (default: cuda if available else cpu).")
    p.add_argument("--variant", choices=list(_VARIANT_TRANSFORMS), default="full",
                   help="full = M2 (lig+pocket); lig_only = M1 (ligand atoms only).")
    p.add_argument("--cv", choices=["groupkfold", "stratified"], default="groupkfold",
                   help="GroupKFold over targets (default; cross-target eval) or "
                        "StratifiedKFold over (key, label) (random; for §5 gate 4).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.all_targets and args.targets is not None:
        sys.stderr.write("--targets and --all-targets are mutually exclusive\n")
        return 2
    if args.all_targets:
        args.targets = get_all_targets()
    elif args.targets is None:
        args.targets = list(TARGETS_DEBUG)

    run_id = _make_run_id(args.run_id)
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "outputs" / "07_train_egnn" / run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=logging.INFO, log_file=args.output_dir / "train.log")
    log.setLevel(logging.INFO)

    if args.dataset_dir is None:
        args.dataset_dir = _latest_dataset_dir()
    args.dataset_dir = args.dataset_dir.resolve()
    if not (args.dataset_dir / "dataset.lmdb").exists():
        log.error("Missing dataset.lmdb in %s", args.dataset_dir)
        return 2
    if not (args.dataset_dir / "index.csv").exists():
        log.error("Missing index.csv in %s", args.dataset_dir)
        return 2

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    git_sha = _git_short_sha()
    log.info("=" * 70)
    log.info("07_train_egnn starting")
    log.info("  run_id: %s", run_id)
    log.info("  git_sha: %s", git_sha)
    log.info("  venv: %s", os.environ.get("VIRTUAL_ENV", "(none)"))
    log.info("  dataset_dir: %s", args.dataset_dir)
    log.info("  output_dir: %s", args.output_dir)
    log.info("  targets: %s", args.targets)
    log.info("  device: %s (bf16=%s)",
             device, torch.cuda.is_bf16_supported() if device.type == "cuda" else "n/a")
    log.info("  epochs: %d  batch_size: %d  lr: %g  h_dim: %d  n_layers: %d  variant: %s",
             args.epochs, args.batch_size, args.lr, args.h_dim, args.n_layers, args.variant)
    log.info("  eval_every: %d  early_stop_patience: %d",
             args.eval_every, args.early_stop_patience)

    # Load index, filter to requested targets.
    # dtype={"key": str} preserves 9-digit zero-padded LMDB keys.
    full_index = pd.read_csv(args.dataset_dir / "index.csv", dtype={"key": str})
    idx = full_index[full_index["target"].isin(args.targets)].reset_index(drop=True)
    if len(idx) == 0:
        log.error("No rows in index.csv match --targets %s", args.targets)
        return 2
    log.info("  filtered index: %d rows across %d targets",
             len(idx), idx["target"].nunique())
    log.info("  per-target counts:")
    for t, cnt in idx.groupby("target")["label"].agg(["size", "sum"]).iterrows():
        log.info("    %s: n=%d pos=%d", t, cnt["size"], cnt["sum"])

    # GroupKFold (leave-one-target-out when n_targets <= n_folds).
    n_folds = args.n_folds or min(5, idx["target"].nunique())
    if n_folds > idx["target"].nunique():
        log.warning("n_folds=%d > n_targets=%d; clamping to n_targets",
                    n_folds, idx["target"].nunique())
        n_folds = idx["target"].nunique()
    log.info("  n_folds: %d (cv=%s)", n_folds, args.cv)

    keys_arr = idx["key"].astype(str).values
    targets_arr = idx["target"].values
    labels_arr = idx["label"].values.astype(int)

    splits = canonical_split_indices(
        keys=keys_arr, labels=labels_arr, targets=targets_arr,
        n_folds=n_folds, cv=args.cv,
    )

    fold_results = []
    t_total = time.perf_counter()
    for fold_idx, (train_i, val_i) in enumerate(splits):
        train_keys = list(keys_arr[train_i])
        val_keys = list(keys_arr[val_i])
        train_targets = sorted(set(targets_arr[train_i]))
        val_targets = sorted(set(targets_arr[val_i]))
        train_labels = labels_arr[train_i]
        fold_dir = args.output_dir / f"fold_{fold_idx}"
        result = _train_fold(
            fold_idx, train_keys, val_keys,
            train_targets, val_targets, train_labels,
            args.dataset_dir / "dataset.lmdb",
            args, device, fold_dir,
        )
        fold_results.append(result)

    wall_total = time.perf_counter() - t_total
    log.info("=" * 70)
    log.info("All folds done in %.1fs", wall_total)

    # Aggregate
    rows = pd.DataFrame(fold_results)
    rows.to_csv(args.output_dir / "per_fold_metrics.csv", index=False)
    metric_cols = [c for c in rows.columns
                   if c.startswith("val_") and rows[c].dtype.kind == "f"]
    aggregate = {f"{c}_mean": float(rows[c].mean()) for c in metric_cols}
    aggregate.update({f"{c}_std": float(rows[c].std(ddof=0)) for c in metric_cols})

    manifest = {
        "run_id": run_id,
        "git_sha": git_sha,
        "venv": os.environ.get("VIRTUAL_ENV", "(none)"),
        "script": "scripts/07_train_egnn.py",
        "dataset_dir": str(args.dataset_dir),
        "targets": list(args.targets),
        "n_folds": n_folds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "h_dim": args.h_dim,
        "n_layers": args.n_layers,
        "variant": args.variant,
        "cv": args.cv,
        "eval_every": args.eval_every,
        "early_stop_patience": args.early_stop_patience,
        "device": str(device),
        "wall_seconds_total": wall_total,
        "aggregate": aggregate,
        "folds": fold_results,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    cfg = {
        "args": {**{k: v for k, v in vars(args).items() if k not in ("dataset_dir", "output_dir")},
                  "dataset_dir": str(args.dataset_dir),
                  "output_dir": str(args.output_dir)},
    }
    cfg["args"]["targets"] = list(cfg["args"]["targets"])
    (args.output_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    write_report(args.output_dir, manifest)
    log.info("Outputs at %s", args.output_dir)
    return 0


def write_report(output_dir: Path, manifest: dict) -> None:
    folds = manifest["folds"]
    agg = manifest["aggregate"]
    lines = [
        "# 07_train_egnn — Smoke run report",
        "",
        f"- run_id: `{manifest['run_id']}`",
        f"- git_sha: `{manifest['git_sha']}`",
        f"- dataset_dir: `{manifest['dataset_dir']}`",
        f"- targets: {', '.join(manifest['targets'])}",
        f"- folds: {manifest['n_folds']}, epochs/fold: {manifest['epochs']}",
        f"- device: {manifest['device']}, batch_size: {manifest['batch_size']}, lr: {manifest['lr']}",
        f"- model: EGNN h_dim={manifest['h_dim']} n_layers={manifest['n_layers']}",
        f"- total wall: **{manifest['wall_seconds_total']:.1f}s**",
        "",
        "## Per-fold (best-epoch metrics on val)",
        "",
        "| fold | val targets | n_train | n_val | best epoch | pr_auc | roc_auc | bedroc | ef_1pct | ef_5pct | wall (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for f in folds:
        lines.append(
            f"| {f['fold']} | {','.join(f['val_targets'])} | "
            f"{f['n_train']} | {f['n_val']} | {f['best_epoch']} | "
            f"{f.get('val_pr_auc', float('nan')):.4f} | "
            f"{f.get('val_roc_auc', float('nan')):.4f} | "
            f"{f.get('val_bedroc', float('nan')):.4f} | "
            f"{f.get('val_ef_1pct', float('nan')):.2f} | "
            f"{f.get('val_ef_5pct', float('nan')):.2f} | "
            f"{f['fold_wall_s']:.1f} |"
        )
    lines += [
        "",
        "## Aggregate (mean ± std across folds)",
        "",
        f"- pr_auc:  {agg.get('val_pr_auc_mean', float('nan')):.4f} ± {agg.get('val_pr_auc_std', float('nan')):.4f}",
        f"- roc_auc: {agg.get('val_roc_auc_mean', float('nan')):.4f} ± {agg.get('val_roc_auc_std', float('nan')):.4f}",
        f"- bedroc:  {agg.get('val_bedroc_mean', float('nan')):.4f} ± {agg.get('val_bedroc_std', float('nan')):.4f}",
        f"- ef_1pct: {agg.get('val_ef_1pct_mean', float('nan')):.2f} ± {agg.get('val_ef_1pct_std', float('nan')):.2f}",
        "",
        "## Notes",
        "",
        "_Filled in after review by the operator._",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
