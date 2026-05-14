#!/usr/bin/env python
"""Phase 6 step (a) — fold-aware frozen-EGNN feature extraction.

The Tier 3 vanilla M2 EGNN was trained under GroupKFold-by-target at
`outputs/07_train_egnn/20260505-000911-1662f0b-m2/`. For each fold k the
model was held out from a specific subset of targets (k's val_targets).

For Phase 6's representation benchmark, **every target's features must come
from a model that did NOT see that target during training**. This script
enforces that:

    target T → fold k where T ∈ val_targets[k]   (exactly one such k)
             → load fold_k/model.pt
             → extract features for all T's poses
             → save per-pose .npz

Hard assertions catch any cross-fold contamination at extraction time
(every target appears in exactly one fold's val_targets; loaded model.pt
matches the expected fold-k state_dict shape; etc.).

Output: outputs/phase6_egnn_features/<TARGET>/<complex_id>.npz with keys
    egnn_emb_pre   (128,) float32  — input of head[0] = graph_emb
    egnn_emb_post  (128,) float32  — output of head[0] (post-Linear, pre-SiLU is fine
                                                       — captured at forward_hook)
    label          int32
    target         str
    complex_id     str
    fold_used      int            — diagnostic, asserts T ∈ fold_used.val_targets

Plus a top-level manifest.json with extraction metadata.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import json
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_HDBIND = Path("/home/maurbina/hdbind-3D")
if str(_HDBIND) not in sys.path:
    sys.path.insert(0, str(_HDBIND))

from src.models.egnn import EGNN

EGNN_RUN_DIR = _HDBIND / "outputs" / "07_train_egnn" / "20260505-000911-1662f0b-m2"
LMDB_PATH = _HDBIND / "outputs" / "06_build_egnn_dataset" / "20260505-000911-1662f0b" / "dataset.lmdb"
INDEX_CSV = _HDBIND / "outputs" / "06_build_egnn_dataset" / "20260505-000911-1662f0b" / "index.csv"
DEFAULT_OUT = _HDBIND / "outputs" / "phase6_egnn_features"


# ---------------------------------------------------------------------------
# Cross-fold contamination guard
# ---------------------------------------------------------------------------

def load_fold_map(run_dir: Path) -> tuple[dict[str, int], dict[int, list[str]]]:
    """Build target → fold_k mapping from per_fold_metrics.csv.

    Asserts that every target appears in EXACTLY ONE fold's val_targets.
    Returns (target_to_fold, fold_to_val_targets).
    """
    csv = run_dir / "per_fold_metrics.csv"
    df = pd.read_csv(csv)
    target_to_fold: dict[str, int] = {}
    fold_to_val_targets: dict[int, list[str]] = {}
    counter: Counter = Counter()
    for _, row in df.iterrows():
        fold = int(row["fold"])
        val_targets = ast.literal_eval(row["val_targets"])
        fold_to_val_targets[fold] = val_targets
        for t in val_targets:
            counter[t] += 1
            if t in target_to_fold:
                raise AssertionError(
                    f"Target {t!r} appears in val_targets of both fold "
                    f"{target_to_fold[t]} and fold {fold} — fold map is not "
                    f"a partition. Cross-fold contamination would be possible."
                )
            target_to_fold[t] = fold
    # Belt-and-suspenders
    dupes = [t for t, c in counter.items() if c != 1]
    if dupes:
        raise AssertionError(f"Targets appearing in != 1 fold: {dupes}")
    return target_to_fold, fold_to_val_targets


def load_fold_train_targets(run_dir: Path) -> dict[int, list[str]]:
    """For each fold, the train_targets list — used to assert the model
    didn't see a given target."""
    csv = run_dir / "per_fold_metrics.csv"
    df = pd.read_csv(csv)
    return {int(row["fold"]): ast.literal_eval(row["train_targets"])
            for _, row in df.iterrows()}


# ---------------------------------------------------------------------------
# Per-fold model loading
# ---------------------------------------------------------------------------

def load_fold_model(run_dir: Path, fold: int, device: torch.device) -> torch.nn.Module:
    """Load fold_k/model.pt = best-epoch checkpoint (per 07_train_egnn.py:298-316)."""
    ckpt = run_dir / f"fold_{fold}" / "model.pt"
    assert ckpt.exists(), f"Missing checkpoint: {ckpt}"
    state = torch.load(ckpt, map_location=device, weights_only=False)
    # The training script saves state_dict directly (no wrapper).
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model = EGNN(in_node_dim=10, edge_attr_dim=3, h_dim=128, n_layers=4)
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing, f"Missing keys when loading fold {fold}: {missing}"
    assert not unexpected, f"Unexpected keys for fold {fold}: {unexpected}"
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Hook capture (head[0] = first Linear of the 2-layer MLP head)
# ---------------------------------------------------------------------------

class _Capture:
    """Captures input (pre-MLP graph_emb) and output (post first Linear)
    of `model.head[0]`. Stores the most recent forward's values.

    head = Sequential(Linear, SiLU, Linear)
    head[0] = first Linear (h_dim → head_dim, both 128)
    - input  = graph_emb (post ligand-pool, pre-head)  ← egnn_emb_pre
    - output = head[0] output before SiLU              ← egnn_emb_post
    """
    def __init__(self):
        self.pre = None   # [B, 128]
        self.post = None  # [B, 128]
        self.n_pre = 0
        self.n_post = 0

    def reset(self):
        self.pre = None
        self.post = None
        self.n_pre = 0
        self.n_post = 0


def attach_hooks(model: torch.nn.Module, cap: _Capture):
    first_linear = model.head[0]

    def pre_hook(module, args):
        t = args[0] if isinstance(args, (tuple, list)) else args
        cap.pre = t.detach().cpu()
        cap.n_pre += 1

    def post_hook(module, inputs, output):
        cap.post = output.detach().cpu()
        cap.n_post += 1

    h1 = first_linear.register_forward_pre_hook(pre_hook)
    h2 = first_linear.register_forward_hook(post_hook)
    return [h1, h2]


# ---------------------------------------------------------------------------
# LMDB iteration per target
# ---------------------------------------------------------------------------

def _lmdb_env():
    return lmdb.open(str(LMDB_PATH), readonly=True, lock=False, subdir=False,
                     max_readers=512, readahead=False, meminit=False)


def load_data_for_keys(env, keys: list[bytes]) -> list:
    """Load a list of pyg Data objects from LMDB."""
    out = []
    with env.begin() as txn:
        for k in keys:
            payload = txn.get(k)
            if payload is None:
                out.append(None)
            else:
                out.append(pickle.loads(payload))
    return out


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name)[:120] or "ligand"


# ---------------------------------------------------------------------------
# Forward + capture per pose (B=1, matches FLOWR convention)
# ---------------------------------------------------------------------------

def _ligand_mask(x: torch.Tensor) -> torch.Tensor:
    return x[:, -1] == 0


def process_target(
    target: str, fold: int, model: torch.nn.Module, env, index_df: pd.DataFrame,
    out_dir: Path, device: torch.device, cap: _Capture,
    fold_train_targets: list[str], dry_run: bool = False,
) -> dict:
    # Assertion #1 — model.pt for fold k must NOT have seen this target during training.
    assert target not in fold_train_targets, (
        f"CROSS-FOLD CONTAMINATION: target {target} is in fold {fold}'s "
        f"train_targets. Refusing to extract features."
    )
    sub = index_df[index_df["target"] == target]
    out_dir.mkdir(parents=True, exist_ok=True)
    n_done = 0
    n_skipped_existing = 0
    n_skipped_missing = 0
    t_start = time.perf_counter()

    keys_bytes = [f"{k:09d}".encode("ascii") for k in sub["key"].tolist()]
    rows = sub.to_dict("records")
    # Load all data objects for this target — small enough to fit in RAM
    datas = load_data_for_keys(env, keys_bytes)

    for row, data in zip(rows, datas):
        if data is None:
            n_skipped_missing += 1
            continue
        complex_id = row["complex_id"]
        label = int(row["label"])
        out_path = out_dir / f"{_sanitize_filename(complex_id)}.npz"
        if out_path.exists():
            n_skipped_existing += 1
            continue
        if dry_run:
            n_done += 1
            continue

        # Single-pose forward (B=1, matches FLOWR convention)
        x = data.x.to(device)
        pos = data.pos.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        lmask = _ligand_mask(x)

        cap.reset()
        with torch.no_grad():
            _ = model(x, pos, edge_index, edge_attr, lmask, batch)

        # Sanity assertions
        assert cap.n_pre == 1 and cap.n_post == 1, (
            f"hooks fired n_pre={cap.n_pre} n_post={cap.n_post} for {complex_id}"
        )
        assert cap.pre is not None and cap.post is not None
        assert cap.pre.shape == (1, 128) and cap.post.shape == (1, 128), (
            f"unexpected shapes pre={cap.pre.shape} post={cap.post.shape}"
        )
        if not torch.isfinite(cap.pre).all() or not torch.isfinite(cap.post).all():
            raise AssertionError(f"NaN/Inf in EGNN capture for {complex_id}")

        np.savez_compressed(
            out_path,
            egnn_emb_pre=cap.pre.squeeze(0).numpy().astype(np.float32),
            egnn_emb_post=cap.post.squeeze(0).numpy().astype(np.float32),
            label=np.int32(label),
            target=target,
            complex_id=complex_id,
            fold_used=np.int32(fold),
        )
        n_done += 1

    elapsed = time.perf_counter() - t_start
    return {
        "target": target,
        "fold_used": fold,
        "n_poses": len(rows),
        "n_extracted_this_run": n_done,
        "n_skipped_existing": n_skipped_existing,
        "n_skipped_missing": n_skipped_missing,
        "elapsed_seconds": elapsed,
        "ms_per_pose": elapsed * 1000.0 / max(n_done, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=str(EGNN_RUN_DIR))
    ap.add_argument("--lmdb", default=str(LMDB_PATH))
    ap.add_argument("--index_csv", default=str(INDEX_CSV))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--targets", nargs="+", default=None,
                    help="restrict to these targets (default: all 43)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dry_run", action="store_true",
                    help="don't write .npz; just print the fold map and counts")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"run_dir: {run_dir}")
    print(f"lmdb:    {args.lmdb}")
    print(f"out_dir: {out_root}")
    print(f"device:  {device}")

    # ── Fold map + contamination guard ────────────────────────────────────
    target_to_fold, fold_to_val = load_fold_map(run_dir)
    fold_to_train = load_fold_train_targets(run_dir)
    n_folds = len(fold_to_val)
    all_targets = sorted(target_to_fold.keys())
    print(f"\nFold map ({n_folds} folds, {len(all_targets)} targets):")
    for fold in sorted(fold_to_val.keys()):
        vs = fold_to_val[fold]
        ts = fold_to_train[fold]
        print(f"  fold {fold}: {len(ts)} train_targets, {len(vs)} val_targets "
              f"-> {vs}")

    if args.targets:
        sel = [t for t in args.targets if t in target_to_fold]
        missing = [t for t in args.targets if t not in target_to_fold]
        if missing:
            print(f"WARNING: targets not in fold map: {missing}")
    else:
        sel = all_targets
    print(f"\nExtracting features for {len(sel)} targets")

    # ── Index ─────────────────────────────────────────────────────────────
    index_df = pd.read_csv(args.index_csv)
    print(f"index.csv: {len(index_df)} pose rows across "
          f"{index_df['target'].nunique()} targets")

    # ── Loop ──────────────────────────────────────────────────────────────
    env = _lmdb_env()
    results: list[dict] = []
    cap = _Capture()
    cur_fold = None
    model = None
    hooks: list = []

    t_total_start = time.perf_counter()
    for target in tqdm(sel, desc="EGNN extract", unit="t", position=0):
        fold = target_to_fold[target]
        # Lazy fold-model switching (group targets by fold to amortize model load)
        if cur_fold != fold:
            if hooks:
                for h in hooks:
                    h.remove()
            model = load_fold_model(run_dir, fold, device)
            hooks = attach_hooks(model, cap)
            cur_fold = fold
            tqdm.write(f"  loaded fold_{fold}/model.pt for target group "
                       f"{sorted([t for t,f in target_to_fold.items() if f == fold])}")

        target_out = out_root / target
        try:
            r = process_target(
                target=target, fold=fold, model=model, env=env,
                index_df=index_df, out_dir=target_out, device=device, cap=cap,
                fold_train_targets=fold_to_train[fold], dry_run=args.dry_run,
            )
            results.append(r)
        except AssertionError as exc:
            tqdm.write(f"TARGET FAILED {target}: {exc!r}")
            raise

    for h in hooks:
        h.remove()
    env.close()
    elapsed_total = time.perf_counter() - t_total_start

    # ── Manifest ──────────────────────────────────────────────────────────
    import socket
    manifest = {
        "egnn_run_dir": str(run_dir),
        "egnn_run_id": run_dir.name,
        "lmdb_path": str(args.lmdb),
        "extraction_host": socket.gethostname(),
        "extraction_device": str(device),
        "extraction_timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "n_folds": n_folds,
        "fold_to_val_targets": {str(k): v for k, v in fold_to_val.items()},
        "target_to_fold": target_to_fold,
        "n_targets_extracted": len(results),
        "per_target": results,
        "elapsed_total_seconds": elapsed_total,
        "total_poses_extracted": sum(r["n_extracted_this_run"] for r in results),
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n=== EGNN extraction summary "
          f"({len(results)} targets, {elapsed_total:.1f}s) ===")
    total_poses = sum(r["n_extracted_this_run"] for r in results)
    total_skipped_exist = sum(r["n_skipped_existing"] for r in results)
    total_missing = sum(r["n_skipped_missing"] for r in results)
    print(f"  total extracted this run: {total_poses}")
    print(f"  total skipped (already had .npz): {total_skipped_exist}")
    print(f"  total LMDB-missing: {total_missing}")
    print(f"  manifest: {manifest_path}")
    print(f"\n{'target':<10s}  {'fold':>4s}  {'poses':>6s}  {'ms/pose':>8s}")
    for r in results:
        print(f"  {r['target']:<10s}  {r['fold_used']:>4d}  "
              f"{r['n_poses']:>6d}  {r['ms_per_pose']:>8.2f}")


if __name__ == "__main__":
    main()
