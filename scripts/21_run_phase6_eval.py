#!/usr/bin/env python
"""Phase 6 — fixed-classifier representation benchmark.

For each (feature_config, decoy_set, fold, seed):
  - Load features for train_targets[fold] and val_targets[fold]
  - Apply per-stream StandardScaler (fit on train, transform val)
       — PLEC uses with_mean=False to preserve sparsity
  - Concat the streams that the feature_config requests
  - Train two classifiers: linear probe (LogisticRegression) and 2-layer MLP
  - Compute per-target ROC-AUC, PR-AUC, BEDROC, EF@1%, EF@5%
  - Write metrics.json under outputs/phase6_eval/<config>/<decoy_set>/fold_K_seed_S/

Scenario A scope: decoy_set = "DUDE-Z" only (Extrema/Goldilocks deferred to
Threadripper). The directory structure leaves <decoy_set>/{Extrema,Goldilocks}
slots empty so the aggregator picks them up when they're filled in later.

The 7 feature configs (per user direction — do not add an 8th without
justification):
  PLEC                          — Tier 1-2 baseline (4096-dim sparse)
  EGNN-M2-pre                   — graph_emb (128-dim, before head MLP)
  EGNN-M2-post                  — head[0] output (128-dim, post first Linear)
  FLOWR-pre                     — cat(f_lig_pre, f_pocket_pre, eij_pooled_pre) — 1664-dim
  FLOWR-post                    — cat(z_lig_post, z_pocket_post, z_int_post) — 384-dim
  PLEC+FLOWR-post               — concat
  EGNN-M2-pre+FLOWR-post        — concat
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
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

_HDBIND = Path("/home/maurbina/hdbind-3D")
if str(_HDBIND) not in sys.path:
    sys.path.insert(0, str(_HDBIND))

from src.evaluation import (  # noqa: E402
    enrichment_factor, hit_rate_at_k, bedroc,
)
from sklearn.metrics import roc_auc_score, average_precision_score


# ---------------------------------------------------------------------------
# Feature stream loaders (all return dict[(target, complex_id)] → np.ndarray)
# ---------------------------------------------------------------------------

FLOWR_ROOT = _HDBIND / "features" / "flowr_root"
EGNN_ROOT = _HDBIND / "outputs" / "phase6_egnn_features"
PLEC_DIR = _HDBIND / "results" / "tier1" / "plec"

# Stream names — these are the units that get standardized independently.
STREAMS = (
    "plec",
    "egnn_emb_pre", "egnn_emb_post",
    "f_lig_pre", "f_pocket_pre", "eij_pooled_pre",
    "z_lig_post", "z_pocket_post", "z_int_post",
)


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name)[:120] or "ligand"


def load_plec_for_target(target: str) -> dict[str, np.ndarray]:
    """PLEC dict {complex_id → ndarray(4096)}."""
    p = PLEC_DIR / f"{target}.pkl"
    with open(p, "rb") as f:
        return pickle.load(f)


def load_egnn_for_target(target: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    """Returns (egnn_emb_pre, egnn_emb_post, labels) dicts keyed by complex_id."""
    pre: dict[str, np.ndarray] = {}
    post: dict[str, np.ndarray] = {}
    labels: dict[str, int] = {}
    for npz in (EGNN_ROOT / target).glob("*.npz"):
        d = np.load(npz, allow_pickle=False)
        cid = str(d["complex_id"])
        pre[cid] = d["egnn_emb_pre"].astype(np.float32)
        post[cid] = d["egnn_emb_post"].astype(np.float32)
        labels[cid] = int(d["label"])
    return pre, post, labels


def load_flowr_for_target(target: str) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int]]:
    """Returns ({complex_id → {stream → ndarray}}, {complex_id → label})."""
    feats: dict[str, dict[str, np.ndarray]] = {}
    labels: dict[str, int] = {}
    flowr_streams = ("f_lig_pre", "f_pocket_pre", "eij_pooled_pre",
                     "z_lig_post", "z_pocket_post", "z_int_post")
    for npz in (FLOWR_ROOT / target).glob("*.npz"):
        d = np.load(npz, allow_pickle=False)
        cid = str(d["ligand_name"])
        feats[cid] = {s: d[s].astype(np.float32) for s in flowr_streams}
        labels[cid] = int(d["label"])
    return feats, labels


# ---------------------------------------------------------------------------
# Build the per-target feature/label matrix
# ---------------------------------------------------------------------------

def build_target_matrices(targets: list[str]) -> tuple[
    dict[str, np.ndarray],   # streams → (N, D) where N is total poses across all `targets`
    np.ndarray,              # y
    np.ndarray,              # target_index (one row per pose)
    list[str],               # complex_id per pose
]:
    """Load all 8 streams + labels for the given list of targets and stack into matrices.

    Pose ordering: target-by-target; within a target, by complex_id sorted alphabetically.
    Poses where ANY stream is missing are dropped (rare; logged).
    """
    per_target_records: list[tuple[str, str, int, dict[str, np.ndarray]]] = []  # (target, cid, label, stream-dict)
    for t in tqdm(targets, desc="  load", unit="t", leave=False, position=2):
        plec = load_plec_for_target(t)
        egnn_pre, egnn_post, egnn_labels = load_egnn_for_target(t)
        flowr_feats, flowr_labels = load_flowr_for_target(t)
        common = sorted(set(plec) & set(egnn_pre) & set(flowr_feats))
        for cid in common:
            lab_e = egnn_labels[cid]
            lab_f = flowr_labels[cid]
            assert lab_e == lab_f, (
                f"Label mismatch {t}/{cid}: EGNN={lab_e} FLOWR={lab_f}"
            )
            d = {
                "plec":           plec[cid].astype(np.float32),
                "egnn_emb_pre":   egnn_pre[cid],
                "egnn_emb_post":  egnn_post[cid],
                **flowr_feats[cid],
            }
            per_target_records.append((t, cid, lab_e, d))

    if not per_target_records:
        raise RuntimeError(f"No common poses across feature sources for targets {targets}")

    streams_data: dict[str, list[np.ndarray]] = {s: [] for s in STREAMS}
    y: list[int] = []
    target_idx: list[str] = []
    complex_ids: list[str] = []
    for t, cid, lab, d in per_target_records:
        for s in STREAMS:
            streams_data[s].append(d[s])
        y.append(lab)
        target_idx.append(t)
        complex_ids.append(cid)
    streams_arr = {s: np.stack(streams_data[s], axis=0) for s in STREAMS}
    return streams_arr, np.array(y, dtype=np.int32), np.array(target_idx), complex_ids


# ---------------------------------------------------------------------------
# Per-stream standardization
# ---------------------------------------------------------------------------

def per_stream_standardize(
    train_streams: dict[str, np.ndarray],
    val_streams: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, StandardScaler]]:
    """Fit StandardScaler on train fold, apply same scaler to val fold.

    PLEC uses with_mean=False to preserve its sparsity structure
    (subtracting the mean would densify a 99.6%-sparse feature).
    Other streams use full standardization (mean+std).
    """
    scalers: dict[str, StandardScaler] = {}
    train_out: dict[str, np.ndarray] = {}
    val_out: dict[str, np.ndarray] = {}
    for s in train_streams.keys():
        # PLEC kept sparse: with_mean=False prevents densifying a 99.6%-sparse
        # bit vector. Other streams get full (mean, std) normalization so that
        # all input features land in roughly the same scale before they're
        # concatenated — without this, the high-variance streams (pocket_pre
        # std ≈ 0.82) dominate a linear probe over the low-variance ones
        # (f_lig_pre std ≈ 0.18) by SCALE rather than signal.
        if s == "plec":
            sc = StandardScaler(with_mean=False, with_std=True)
        else:
            sc = StandardScaler(with_mean=True, with_std=True)
        sc.fit(train_streams[s])
        scalers[s] = sc
        train_out[s] = sc.transform(train_streams[s]).astype(np.float32)
        val_out[s] = sc.transform(val_streams[s]).astype(np.float32)
    return train_out, val_out, scalers


# ---------------------------------------------------------------------------
# Feature config assembly — per user direction, 7 configs only
# ---------------------------------------------------------------------------

FEATURE_CONFIGS = {
    "PLEC":                    ["plec"],
    "EGNN-M2-pre":             ["egnn_emb_pre"],
    "EGNN-M2-post":            ["egnn_emb_post"],
    "FLOWR-pre":               ["f_lig_pre", "f_pocket_pre", "eij_pooled_pre"],
    "FLOWR-post":              ["z_lig_post", "z_pocket_post", "z_int_post"],
    "PLEC+FLOWR-post":         ["plec", "z_lig_post", "z_pocket_post", "z_int_post"],
    "EGNN-M2-pre+FLOWR-post":  ["egnn_emb_pre", "z_lig_post", "z_pocket_post", "z_int_post"],
}


def concat_streams(streams: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    parts = [streams[n] for n in names]
    return np.concatenate(parts, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

def train_linear_probe(X_tr, y_tr, X_va, seed: int) -> tuple[np.ndarray, dict]:
    clf = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced",
        solver="lbfgs", n_jobs=1, random_state=seed,
    )
    clf.fit(X_tr, y_tr)
    scores = clf.predict_proba(X_va)[:, 1]
    return scores.astype(np.float32), {"n_iter": int(clf.n_iter_[0])}


class _MLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_2layer(X_tr, y_tr, X_va, seed: int, epochs: int = 40,
                     batch_size: int = 1024, lr: float = 1e-3,
                     device: torch.device | None = None) -> tuple[np.ndarray, dict]:
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    in_dim = X_tr.shape[1]
    model = _MLP(in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # Class-imbalance via pos_weight in BCE-with-logits
    n_pos = max(int(y_tr.sum()), 1)
    n_neg = max(len(y_tr) - n_pos, 1)
    pos_weight = torch.tensor([n_neg / n_pos], device=device, dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    X_va_t = torch.from_numpy(X_va).to(device)

    n = len(X_tr_t)
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            b = idx[s:s + batch_size]
            opt.zero_grad(set_to_none=True)
            logits = model(X_tr_t[b])
            loss = loss_fn(logits, y_tr_t[b])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(X_va_t)).cpu().numpy().astype(np.float32)
    return scores, {"epochs": epochs, "in_dim": in_dim}


# ---------------------------------------------------------------------------
# Per-target metrics
# ---------------------------------------------------------------------------

def per_target_metrics(y: np.ndarray, scores: np.ndarray,
                       targets: np.ndarray) -> dict:
    """Return {target: {metric: value}} for all targets in `targets`."""
    out: dict[str, dict[str, float]] = {}
    for t in np.unique(targets):
        m = targets == t
        if m.sum() < 2:
            continue
        yt, st = y[m], scores[m]
        if yt.sum() == 0 or yt.sum() == len(yt):
            # No actives or all actives → metrics undefined
            out[t] = {"n_poses": int(m.sum()), "n_actives": int(yt.sum()),
                      "n_decoys": int(len(yt) - yt.sum()),
                      "roc_auc": float("nan"), "pr_auc": float("nan"),
                      "bedroc": float("nan"), "ef_1pct": float("nan"),
                      "ef_5pct": float("nan")}
            continue
        out[t] = {
            "n_poses": int(m.sum()),
            "n_actives": int(yt.sum()),
            "n_decoys": int(len(yt) - yt.sum()),
            "roc_auc": float(roc_auc_score(yt, st)),
            "pr_auc": float(average_precision_score(yt, st)),
            "bedroc": float(bedroc(yt, st, alpha=20.0)),
            "ef_1pct": float(enrichment_factor(yt, st, 0.01)),
            "ef_5pct": float(enrichment_factor(yt, st, 0.05)),
        }
    return out


# ---------------------------------------------------------------------------
# One (config, fold, seed) → both classifiers, write metrics.json
# ---------------------------------------------------------------------------

def aggregate_metrics(per_target: dict) -> dict:
    metrics = ["roc_auc", "pr_auc", "bedroc", "ef_1pct", "ef_5pct"]
    agg = {}
    for m in metrics:
        vals = [v[m] for v in per_target.values() if not np.isnan(v[m])]
        if vals:
            agg[f"{m}_mean"] = float(np.mean(vals))
            agg[f"{m}_std"] = float(np.std(vals))
            agg[f"{m}_n"] = len(vals)
    return agg


def run_one(
    config: str,
    fold: int,
    seed: int,
    train_streams: dict[str, np.ndarray],
    train_y: np.ndarray,
    train_targets: np.ndarray,
    val_streams: dict[str, np.ndarray],
    val_y: np.ndarray,
    val_targets_arr: np.ndarray,
    val_complex_ids: list[str],
    out_dir: Path,
    device: torch.device,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    streams_used = FEATURE_CONFIGS[config]

    # Standardize ALL streams (we only need the ones in this config, but standardizing
    # all is cheap and lets us not pre-select). Caller can pre-select to save memory.
    train_std, val_std, _ = per_stream_standardize(
        {s: train_streams[s] for s in streams_used},
        {s: val_streams[s] for s in streams_used},
    )
    X_tr = concat_streams(train_std, streams_used)
    X_va = concat_streams(val_std, streams_used)
    in_dim = X_tr.shape[1]

    result = {
        "feature_config": config,
        "streams_used": streams_used,
        "in_dim": int(in_dim),
        "fold": fold,
        "seed": seed,
        "n_train_poses": int(len(train_y)),
        "n_val_poses": int(len(val_y)),
        "n_train_pos": int(train_y.sum()),
        "n_val_pos": int(val_y.sum()),
        "classifiers": {},
    }

    for clf_name, fn in (("linear_probe", train_linear_probe),
                        ("mlp_2layer", train_mlp_2layer)):
        t0 = time.perf_counter()
        if clf_name == "mlp_2layer":
            scores, meta = fn(X_tr, train_y, X_va, seed, device=device)
        else:
            scores, meta = fn(X_tr, train_y, X_va, seed)
        per_t = per_target_metrics(val_y, scores, val_targets_arr)
        agg = aggregate_metrics(per_t)
        elapsed = time.perf_counter() - t0
        result["classifiers"][clf_name] = {
            "per_target": per_t,
            "aggregate": agg,
            "elapsed_seconds": elapsed,
            "meta": meta,
        }

    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--configs", nargs="+", default=list(FEATURE_CONFIGS.keys()))
    ap.add_argument("--out_root", default=str(_HDBIND / "outputs" / "phase6_eval"))
    ap.add_argument("--decoy_set", default="DUDE-Z",
                    help="Scenario A scope only — Extrema/Goldilocks deferred")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Read fold map from the EGNN training run (canonical GroupKFold splits)
    egnn_run = _HDBIND / "outputs" / "07_train_egnn" / "20260505-000911-1662f0b-m2"
    df = pd.read_csv(egnn_run / "per_fold_metrics.csv")
    fold_to_train_targets = {int(r["fold"]): ast.literal_eval(r["train_targets"])
                              for _, r in df.iterrows()}
    fold_to_val_targets = {int(r["fold"]): ast.literal_eval(r["val_targets"])
                            for _, r in df.iterrows()}

    print(f"folds: {args.folds}  seeds: {args.seeds}")
    print(f"configs ({len(args.configs)}): {args.configs}")
    print(f"decoy_set: {args.decoy_set}")
    print(f"out_root: {out_root}\n")

    t_total_start = time.perf_counter()
    n_runs = 0
    n_skipped = 0

    # Per-fold loading — load each fold's train+val features once,
    # then loop over (config, seed) cheaply against them.
    for fold in tqdm(args.folds, desc="fold", unit="f", position=0):
        tt = fold_to_train_targets[fold]
        vt = fold_to_val_targets[fold]
        print(f"\n=== fold {fold}: train_n={len(tt)} val_n={len(vt)} ===")
        print(f"  val_targets: {vt}")
        t_load = time.perf_counter()
        train_streams, train_y, train_targets, _ = build_target_matrices(tt)
        val_streams, val_y, val_targets_arr, val_complex_ids = build_target_matrices(vt)
        print(f"  loaded {len(train_y)} train / {len(val_y)} val poses "
              f"in {time.perf_counter()-t_load:.1f}s")

        for config in args.configs:
            for seed in args.seeds:
                out_dir = out_root / config / args.decoy_set / f"fold_{fold}_seed_{seed}"
                if (out_dir / "metrics.json").exists():
                    n_skipped += 1
                    continue
                t_run = time.perf_counter()
                try:
                    res = run_one(
                        config=config, fold=fold, seed=seed,
                        train_streams=train_streams, train_y=train_y,
                        train_targets=train_targets,
                        val_streams=val_streams, val_y=val_y,
                        val_targets_arr=val_targets_arr,
                        val_complex_ids=val_complex_ids,
                        out_dir=out_dir, device=device,
                    )
                    n_runs += 1
                    if seed == args.seeds[0]:
                        # Print compact result row on first seed only (others repeat)
                        lp = res["classifiers"]["linear_probe"]["aggregate"]
                        mp = res["classifiers"]["mlp_2layer"]["aggregate"]
                        tqdm.write(
                            f"    {config:<26s} fold={fold} seed={seed}  "
                            f"linear bedroc={lp.get('bedroc_mean',float('nan')):.3f}  "
                            f"mlp bedroc={mp.get('bedroc_mean',float('nan')):.3f}  "
                            f"({time.perf_counter()-t_run:.1f}s)"
                        )
                except Exception as exc:
                    tqdm.write(f"    FAILED {config}/fold_{fold}/seed_{seed}: {exc!r}")

        # Free
        del train_streams, val_streams

    elapsed_total = time.perf_counter() - t_total_start
    print(f"\n=== Phase 6 evaluation done in {elapsed_total/60:.1f} min ===")
    print(f"  runs completed: {n_runs}")
    print(f"  runs skipped (already had metrics.json): {n_skipped}")
    print(f"  output root: {out_root}")


if __name__ == "__main__":
    main()
