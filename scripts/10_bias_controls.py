"""
10_bias_controls.py — TIER3_EGNN_PLAN.md §4.3 bias-control panel.

Evaluates a trained M2 EGNN under two ablations, on the same per-fold val sets
the model was trained against:

  * Receptor ablation (§4.3.a / §5 gate 2):
      Replace every protein atom with a single dummy carbon at the pocket
      centroid. Keep ligand intact. If accuracy barely moves, the receptor
      isn't being used (DUD-E hidden-bias failure mode).

  * Ligand ablation (§4.3.b / §5 gate 3):
      Replace every ligand atom with a single dummy carbon at the ligand
      centroid. Keep pocket intact. If accuracy survives, the model is
      identifying the *target*, not the binding event.

§4.3.c (random vs. GroupKFold gap) requires retraining and lives elsewhere.

Reads from a 07_train_egnn run dir: each fold_K/ has a model.pt checkpoint and
the val set is recovered from index.csv via the manifest's groupkfold split.

Outputs to outputs/10_bias_controls/<run_id>/:
  per_fold_metrics.csv   one row per (fold, ablation) with full metrics.
  manifest.json
  config.yaml
  REPORT.md
  bias.log
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

from config import PROJECT_ROOT
from src.utils import load_canonical_groupkfold, setup_logging
from src.evaluation import compute_all_metrics

try:
    import torch
    from torch.utils.data import Dataset
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as PyGDataLoader
except ImportError as e:
    sys.stderr.write(f"Missing torch/torch_geometric: {e}\n"); sys.exit(1)

try:
    import lmdb
except ImportError:
    sys.stderr.write("Missing 'lmdb'\n"); sys.exit(1)

from src.models.egnn import EGNN

log = logging.getLogger("10_bias_controls")

# Element vocab must match 06_build_egnn_dataset.py.
ELEMENT_VOCAB = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
CARBON_IDX = ELEMENT_VOCAB.index("C")
NODE_FEATURE_DIM = len(ELEMENT_VOCAB) + 1
EDGE_ATTR_DIM = 3
EDGE_TYPE_INTRA_PROT = 0
EDGE_TYPE_INTRA_LIG = 1
EDGE_TYPE_CROSS = 2

_LMDB_ENV_CACHE: dict[str, "lmdb.Environment"] = {}


def _get_lmdb_env(path: str) -> "lmdb.Environment":
    env = _LMDB_ENV_CACHE.get(path)
    if env is None:
        env = lmdb.open(path, readonly=True, lock=False, subdir=False,
                        readahead=True, meminit=False, max_readers=512)
        _LMDB_ENV_CACHE[path] = env
    return env


# ── Ablation transforms ────────────────────────────────────────────────────

def _dummy_node_feature() -> np.ndarray:
    """Single-row node feature for a dummy carbon. is_protein bit is set
    contextually by the caller (set in dummy_protein vs dummy_ligand).
    """
    f = np.zeros(NODE_FEATURE_DIM, dtype=np.float32)
    f[CARBON_IDX] = 1.0
    return f


def _rebuild_edges(
    pos: torch.Tensor,
    is_protein: torch.Tensor,
    edge_cutoff: float = 4.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild edges + edge_attr after node-set surgery.

    Symmetric pairs i<j with distance < edge_cutoff. edge_attr is 3-d one-hot
    over {intra-prot, intra-lig, cross} matching the 06-time convention.
    """
    from scipy.spatial import cKDTree
    pos_np = pos.detach().cpu().numpy()
    pairs = cKDTree(pos_np).query_pairs(r=edge_cutoff, output_type="ndarray")
    if pairs.size == 0:
        return (torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, EDGE_ATTR_DIM), dtype=torch.float32))
    is_prot_np = is_protein.detach().cpu().numpy().astype(bool)
    i_arr = pairs[:, 0]; j_arr = pairs[:, 1]
    i_p = is_prot_np[i_arr]; j_p = is_prot_np[j_arr]
    types = np.where(i_p & j_p, EDGE_TYPE_INTRA_PROT,
             np.where(~i_p & ~j_p, EDGE_TYPE_INTRA_LIG, EDGE_TYPE_CROSS))
    edge_index = np.stack([
        np.concatenate([i_arr, j_arr]),
        np.concatenate([j_arr, i_arr]),
    ], axis=0).astype(np.int64)
    types_sym = np.concatenate([types, types])
    edge_attr = np.zeros((edge_index.shape[1], EDGE_ATTR_DIM), dtype=np.float32)
    edge_attr[np.arange(edge_attr.shape[0]), types_sym] = 1.0
    return (torch.from_numpy(edge_index), torch.from_numpy(edge_attr))


def receptor_ablation(data: Data) -> Data:
    """§4.3.a — replace all protein atoms with a single dummy carbon at the
    centroid of the pocket atoms. Keep ligand intact.
    """
    is_protein = data.x[:, -1] == 1
    is_ligand = ~is_protein
    if int(is_protein.sum().item()) == 0:
        return data  # nothing to ablate
    pocket_centroid = data.pos[is_protein].mean(dim=0, keepdim=True)  # (1, 3)

    lig_x = data.x[is_ligand]
    lig_pos = data.pos[is_ligand]
    lig_resid = data.residue_idx[is_ligand]

    dummy_x = torch.from_numpy(_dummy_node_feature()).unsqueeze(0)  # (1, F)
    dummy_x[0, -1] = 1.0  # is_protein bit
    dummy_pos = pocket_centroid
    dummy_resid = torch.tensor([-1], dtype=torch.long)  # mark as no-residue

    new_x = torch.cat([lig_x, dummy_x], dim=0)
    new_pos = torch.cat([lig_pos, dummy_pos], dim=0)
    new_resid = torch.cat([lig_resid, dummy_resid], dim=0)
    is_prot_new = torch.cat([torch.zeros(lig_x.size(0), dtype=torch.bool),
                              torch.ones(1, dtype=torch.bool)])
    edge_index, edge_attr = _rebuild_edges(new_pos, is_prot_new)

    return Data(
        x=new_x, pos=new_pos, edge_index=edge_index, edge_attr=edge_attr,
        residue_idx=new_resid, y=data.y,
        target=data.target, complex_id=data.complex_id,
    )


def ligand_ablation(data: Data) -> Data:
    """§4.3.b — replace all ligand atoms with a single dummy carbon at the
    centroid of the ligand atoms. Keep pocket intact.
    """
    is_protein = data.x[:, -1] == 1
    is_ligand = ~is_protein
    if int(is_ligand.sum().item()) == 0:
        return data
    lig_centroid = data.pos[is_ligand].mean(dim=0, keepdim=True)

    prot_x = data.x[is_protein]
    prot_pos = data.pos[is_protein]
    prot_resid = data.residue_idx[is_protein]

    dummy_x = torch.from_numpy(_dummy_node_feature()).unsqueeze(0)
    dummy_x[0, -1] = 0.0  # is_ligand
    dummy_pos = lig_centroid
    dummy_resid = torch.tensor([-1], dtype=torch.long)

    new_x = torch.cat([dummy_x, prot_x], dim=0)
    new_pos = torch.cat([dummy_pos, prot_pos], dim=0)
    new_resid = torch.cat([dummy_resid, prot_resid], dim=0)
    is_prot_new = torch.cat([torch.zeros(1, dtype=torch.bool),
                              torch.ones(prot_x.size(0), dtype=torch.bool)])
    edge_index, edge_attr = _rebuild_edges(new_pos, is_prot_new)

    return Data(
        x=new_x, pos=new_pos, edge_index=edge_index, edge_attr=edge_attr,
        residue_idx=new_resid, y=data.y,
        target=data.target, complex_id=data.complex_id,
    )


_ABLATIONS = {
    "none": None,
    "receptor": receptor_ablation,
    "ligand": ligand_ablation,
}


# ── Dataset wrapper that applies ablation eagerly + caches ────────────────

class AblationDataset(Dataset):
    def __init__(self, lmdb_path: str | Path, keys: list[str], ablation: str):
        self.lmdb_path = str(lmdb_path)
        self.keys = [k.encode("ascii") if isinstance(k, str) else k for k in keys]
        if ablation not in _ABLATIONS:
            raise ValueError(f"Unknown ablation {ablation!r}")
        self.ablation = ablation
        self.transform = _ABLATIONS[ablation]
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


# ── Evaluation ────────────────────────────────────────────────────────────

def _ligand_mask(batch_x: torch.Tensor) -> torch.Tensor:
    return batch_x[:, -1] == 0


def _evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, scores = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16) \
                 if device.type == "cuda" else _nullcm():
                logits = model(
                    batch.x, batch.pos, batch.edge_index,
                    batch.edge_attr, _ligand_mask(batch.x), batch.batch,
                )
            ys.append(batch.y.detach().float().cpu().numpy().ravel())
            scores.append(torch.sigmoid(logits.float()).cpu().numpy().ravel())
    return np.concatenate(ys), np.concatenate(scores)


import contextlib
@contextlib.contextmanager
def _nullcm():
    yield


# ── Main ──────────────────────────────────────────────────────────────────

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
    p.add_argument("--train-run", type=Path, required=True,
                   help="Path to a 07_train_egnn run dir (contains fold_K/model.pt + manifest.json).")
    p.add_argument("--eval-dataset-dir", type=Path, default=None,
                   help="Override the eval LMDB (default: train manifest's dataset_dir). "
                        "Use this to evaluate a model trained on DUDE-Z against Goldilocks decoys.")
    p.add_argument("--ablations", nargs="+",
                   default=["none", "receptor", "ligand"],
                   choices=list(_ABLATIONS),
                   help="Which ablations to evaluate (default: all three).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args(argv)


def _load_train_manifest(train_run: Path) -> dict:
    m = json.loads((train_run / "manifest.json").read_text())
    if m.get("variant", "full") != "full":
        log.warning("Training manifest variant=%r — bias controls assume full-complex M2.",
                    m.get("variant"))
    return m


def _resplit_keys(train_run: Path, train_manifest: dict,
                  eval_dataset_dir: Path) -> list[tuple[list[str], list[str], list[str]]]:
    """Recover per-fold (train_keys, val_keys, val_targets).

    The fold composition is derived from the *training* dataset (so the split
    matches what 07 used). Then, when eval_dataset_dir differs from the train
    dataset, val_keys are pulled from the eval LMDB's index.csv filtered to
    each fold's val_targets — this is how we evaluate a DUDE-Z-trained model
    against Goldilocks decoys.
    """
    train_dataset_dir = Path(train_manifest["dataset_dir"])
    train_idx = pd.read_csv(train_dataset_dir / "index.csv", dtype={"key": str})
    targets_in_train = set(train_manifest["targets"])
    train_idx = train_idx[train_idx["target"].isin(targets_in_train)].reset_index(drop=True)
    train_keys_arr = train_idx["key"].astype(str).values
    train_targets_arr = train_idx["target"].values

    fold_splits = load_canonical_groupkfold(train_run / "manifest.json")

    same_lmdb = eval_dataset_dir.resolve() == train_dataset_dir.resolve()
    if same_lmdb:
        eval_idx = train_idx
    else:
        eval_idx = pd.read_csv(eval_dataset_dir / "index.csv", dtype={"key": str})

    splits = []
    for train_i, val_i in fold_splits:
        val_targets = sorted(set(train_targets_arr[val_i]))
        if same_lmdb:
            val_keys = list(train_keys_arr[val_i])
        else:
            sub = eval_idx[eval_idx["target"].isin(val_targets)]
            val_keys = list(sub["key"].astype(str).values)
        splits.append((
            list(train_keys_arr[train_i]),  # train_keys (from train LMDB; rarely needed)
            val_keys,
            val_targets,
        ))
    return splits


def main(argv=None):
    args = parse_args(argv)

    args.train_run = args.train_run.resolve()
    if not (args.train_run / "manifest.json").exists():
        sys.stderr.write(f"No manifest.json in {args.train_run}\n")
        return 2

    run_id = _make_run_id(args.run_id)
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "outputs" / "10_bias_controls" / run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=logging.INFO, log_file=args.output_dir / "bias.log")
    log.setLevel(logging.INFO)

    train_manifest = _load_train_manifest(args.train_run)
    train_dataset_dir = Path(train_manifest["dataset_dir"])
    if args.eval_dataset_dir is not None:
        eval_dataset_dir = args.eval_dataset_dir.resolve()
    else:
        eval_dataset_dir = train_dataset_dir
    lmdb_path = eval_dataset_dir / "dataset.lmdb"
    # Keep `dataset_dir` name for backwards compat with the rest of main().
    dataset_dir = eval_dataset_dir

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    git_sha = _git_short_sha()

    log.info("=" * 70)
    log.info("10_bias_controls starting")
    log.info("  run_id: %s", run_id)
    log.info("  git_sha: %s", git_sha)
    log.info("  train_run: %s", args.train_run)
    log.info("  train_dataset_dir: %s", train_dataset_dir)
    log.info("  eval_dataset_dir: %s", eval_dataset_dir)
    log.info("  ablations: %s", args.ablations)
    log.info("  device: %s", device)

    splits = _resplit_keys(args.train_run, train_manifest, eval_dataset_dir)
    fold_paths = sorted(args.train_run.glob("fold_*/model.pt"))
    if len(fold_paths) != len(splits):
        log.warning("Found %d fold_K/model.pt files but %d split folds",
                    len(fold_paths), len(splits))

    rows = []
    t0 = time.perf_counter()
    for fold_idx, ((_train_keys, val_keys, val_targets), model_path) in enumerate(
        zip(splits, fold_paths)
    ):
        log.info("─" * 60)
        log.info("Fold %d  val=%s  n_val=%d  ckpt=%s",
                 fold_idx, val_targets, len(val_keys), model_path.relative_to(args.train_run))

        model = EGNN(
            in_node_dim=NODE_FEATURE_DIM, edge_attr_dim=EDGE_ATTR_DIM,
            h_dim=train_manifest["h_dim"], n_layers=train_manifest["n_layers"],
        ).to(device)
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)

        for ablation in args.ablations:
            t_a = time.perf_counter()
            ds = AblationDataset(lmdb_path, val_keys, ablation)
            loader = PyGDataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
            y_true, y_score = _evaluate(model, loader, device)
            metrics = compute_all_metrics(y_true.astype(int), y_score, threshold=0.5)
            wall = time.perf_counter() - t_a
            log.info("  ablation=%-8s  pr_auc=%.4f  roc_auc=%.4f  bedroc=%.4f  ef_1pct=%.2f  %.1fs",
                     ablation, metrics["pr_auc"], metrics["roc_auc"], metrics["bedroc"],
                     metrics["ef_1pct"], wall)
            rows.append({
                "fold": fold_idx,
                "held_out_targets": ",".join(val_targets),
                "n_val": len(val_keys),
                "ablation": ablation,
                "wall_s": wall,
                **{f"val_{k}": v for k, v in metrics.items()},
            })

    wall_total = time.perf_counter() - t0
    log.info("All folds × ablations done in %.1fs", wall_total)

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "per_fold_metrics.csv", index=False)

    # Aggregate: mean ± std across folds, per ablation
    agg = (df.groupby("ablation")[[c for c in df.columns if c.startswith("val_")]]
              .agg(["mean", "std"]))
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.to_csv(args.output_dir / "ablation_aggregate.csv")

    eval_manifest = json.loads((eval_dataset_dir / "manifest.json").read_text())
    manifest = {
        "run_id": run_id,
        "git_sha": git_sha,
        "venv": os.environ.get("VIRTUAL_ENV", "(none)"),
        "script": "scripts/10_bias_controls.py",
        "train_run": str(args.train_run),
        "train_dataset_dir": str(train_dataset_dir),
        "eval_dataset_dir": str(eval_dataset_dir),
        "eval_decoy_set": eval_manifest.get("decoy_set", "DUDE_Z"),
        "ablations": list(args.ablations),
        "n_folds": len(splits),
        "wall_seconds_total": wall_total,
        "rows": rows,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    cfg = {"args": {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}}
    cfg["args"]["ablations"] = list(cfg["args"]["ablations"])
    (args.output_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    write_report(args.output_dir, manifest, df)
    log.info("Outputs at %s", args.output_dir)
    return 0


def write_report(output_dir: Path, manifest: dict, df: pd.DataFrame) -> None:
    lines = [
        "# 10_bias_controls — Receptor & ligand ablation",
        "",
        f"- run_id: `{manifest['run_id']}`",
        f"- git_sha: `{manifest['git_sha']}`",
        f"- train_run: `{manifest['train_run']}`",
        f"- eval_dataset_dir: `{manifest['eval_dataset_dir']}`",
        f"- eval_decoy_set: **{manifest.get('eval_decoy_set', 'DUDE_Z')}**",
        f"- ablations: {', '.join(manifest['ablations'])}",
        f"- folds: {manifest['n_folds']}",
        f"- wall: **{manifest['wall_seconds_total']:.1f}s**",
        "",
        "## Per fold × ablation",
        "",
        "| fold | val | ablation | pr_auc | roc_auc | bedroc | ef_1pct |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['fold']} | {r['held_out_targets']} | {r['ablation']} | "
            f"{r['val_pr_auc']:.4f} | {r['val_roc_auc']:.4f} | "
            f"{r['val_bedroc']:.4f} | {r['val_ef_1pct']:.2f} |"
        )

    # Aggregate per ablation
    pivot = (df.groupby("ablation")[["val_pr_auc", "val_roc_auc", "val_bedroc", "val_ef_1pct"]]
                .agg(["mean", "std"]))
    lines += ["", "## Aggregate (mean ± std across folds)", ""]
    for ab in pivot.index:
        m = pivot.loc[ab]
        lines.append(
            f"- **{ab}**:  pr_auc {m[('val_pr_auc','mean')]:.4f}±{m[('val_pr_auc','std')]:.4f}  |  "
            f"roc_auc {m[('val_roc_auc','mean')]:.4f}±{m[('val_roc_auc','std')]:.4f}  |  "
            f"bedroc {m[('val_bedroc','mean')]:.4f}±{m[('val_bedroc','std')]:.4f}"
        )

    # §5 gate readouts
    if {"none", "receptor"}.issubset(set(df["ablation"].unique())):
        delta_bedroc = (
            df.loc[df["ablation"] == "none", "val_bedroc"].mean()
            - df.loc[df["ablation"] == "receptor", "val_bedroc"].mean()
        )
        lines += [
            "",
            "## §5 gate 2 — receptor matters",
            f"- Δ BEDROC (none − receptor-ablated): **{delta_bedroc:+.4f}**",
            f"- gate threshold: ≥ 0.05 (5 BEDROC points). "
            f"{'PASS' if delta_bedroc >= 0.05 else 'FAIL'}.",
        ]
    if "ligand" in df["ablation"].unique():
        m_lig_bedroc = df.loc[df["ablation"] == "ligand", "val_bedroc"].mean()
        lines += [
            "",
            "## §5 gate 3 — model not riding on target identity",
            f"- M2 ligand-ablated BEDROC (mean): **{m_lig_bedroc:.4f}**",
            "- gate threshold: ≤ ligand-only baseline BEDROC (M1 from `07_train_egnn --variant lig_only`). "
            "Compare manually vs M1 run.",
        ]

    lines += ["", "## Notes", "", "_Filled in after review by the operator._", ""]
    (output_dir / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
