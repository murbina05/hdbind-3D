"""
06_build_egnn_dataset.py — Tier 3 EGNN dataset builder.

Builds an LMDB of PyTorch-Geometric `Data` objects following
TIER3_EGNN_PLAN.md §3.1: 8 Å pocket extraction around any ligand heavy
atom, 4.5 Å edge cutoff, heavy-atom one-hot {C,N,O,F,P,S,Cl,Br,I} ⊕
protein/ligand indicator, edge-type one-hot {intra-prot, intra-lig, cross},
residue index per protein atom (ESM-C broadcast hook left as TODO stub).

Outputs to outputs/06_build_egnn_dataset/<run_id>/:
  dataset.lmdb    sequential int keys -> pickled Data (protocol 5)
  manifest.json   run config + per-target counts
  per_target.csv  flat per-target counts
  index.csv       key,target,complex_id,label
  config.yaml     resolved CLI args + relevant config.py constants
  build.log       full INFO log
  REPORT.md       short markdown summary

Usage:
  python scripts/06_build_egnn_dataset.py --targets AA2AR ADRB2 EGFR
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import (
    DOCKING_DIR, MOL2_PREFIX, MOL2_SUBDIR, MOL2_SUBDIRS, N_WORKERS, RANDOM_SEED,
    PROJECT_ROOT, TARGETS_DEBUG, get_all_targets,
)
from src.data_loading import (
    AD_EXCLUDED_TARGETS, AD_TRANSFORMED_ROOT,
    iter_ad_poses, iter_poses_from, load_protein,
)
from src.utils import set_thread_env, setup_logging

# Decoy-source enum (TIER3_DECOY_BIAS_PLAN Phase 3 §1).  Lowercase canonical
# names; map to (mol2_subdir | None, mol2_prefix | None) — None means the
# source uses a non-mol2 loader (currently only `ad`).
DECOY_SOURCES = {
    "dudez":      MOL2_SUBDIRS["DUDE_Z"],
    "goldilocks": MOL2_SUBDIRS["Goldilocks"],
    "extrema":    MOL2_SUBDIRS["Extrema"],
    "ad":         (None, None),
}
# Legacy --decoy-set values (CamelCase) → canonical --decoy-source values.
_DECOY_SET_TO_SOURCE = {
    "DUDE_Z": "dudez",
    "Goldilocks": "goldilocks",
    "Extrema": "extrema",
}
# Reverse map for manifest.decoy_set (preserves the CamelCase string that
# 11_eval_goldilocks / 12_verify_plec_groupkfold match against).
_SOURCE_TO_DECOY_SET = {v: k for k, v in _DECOY_SET_TO_SOURCE.items()}
_SOURCE_TO_DECOY_SET["ad"] = "AD"

try:
    import torch
    from torch_geometric.data import Data
except ImportError as e:
    sys.stderr.write(
        "Missing torch_geometric in venv. Install:\n"
        "    source /home/maurbina/.venvs/dc_featurizers/bin/activate\n"
        "    pip install torch_geometric\n"
        f"Original error: {e}\n"
    )
    sys.exit(1)

try:
    import lmdb
except ImportError:
    sys.stderr.write(
        "Missing 'lmdb' in venv. Install:\n"
        "    pip install lmdb\n"
    )
    sys.exit(1)

log = logging.getLogger("06_build_egnn_dataset")

ELEMENT_VOCAB = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
ELEMENT_TO_IDX = {e: i for i, e in enumerate(ELEMENT_VOCAB)}
NODE_FEATURE_DIM = len(ELEMENT_VOCAB) + 1
EDGE_ATTR_DIM = 3
EDGE_TYPE_INTRA_PROT = 0
EDGE_TYPE_INTRA_LIG = 1
EDGE_TYPE_CROSS = 2

LMDB_MAP_SIZE = 100 * 1024 ** 3   # 100 GB; AD's 40-target build uses ~22 GB


def attach_esm_c_residue_embedding(data):
    """TODO: broadcast ESM-C 600M residue embeddings onto protein atoms via
    data.residue_idx. §3.1 hook; not implemented in 06_*."""
    raise NotImplementedError("ESM-C broadcast lands in a later script.")


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


def _normalize_element(raw: str) -> str | None:
    """Normalize raw element to vocab key. Return None if outside vocab."""
    if not raw:
        return None
    s = raw.strip()
    cap = (s[0].upper() + s[1:].lower()) if len(s) > 1 else s.upper()
    return cap if cap in ELEMENT_TO_IDX else None


def _ligand_atoms(mol):
    """Extract heavy-atom coords + raw elements from an RDKit Mol with a conformer."""
    if mol.GetNumConformers() == 0:
        return None
    pos = mol.GetConformer().GetPositions().astype(np.float32)
    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    return pos, elements


def _build_data_object(
    lig_pos, lig_elem_idx,
    pocket_pos, pocket_elem_idx, pocket_resids,
    edge_cutoff, label, target, complex_id,
):
    n_lig = lig_pos.shape[0]
    n_prot = pocket_pos.shape[0]
    n_total = n_lig + n_prot

    pos = np.concatenate([lig_pos, pocket_pos], axis=0)
    elem_idx = np.concatenate([lig_elem_idx, pocket_elem_idx])

    x = np.zeros((n_total, NODE_FEATURE_DIM), dtype=np.float32)
    x[np.arange(n_total), elem_idx] = 1.0
    x[n_lig:, len(ELEMENT_VOCAB)] = 1.0

    residue_idx = np.full(n_total, -1, dtype=np.int64)
    residue_idx[n_lig:] = pocket_resids

    pairs = cKDTree(pos).query_pairs(r=edge_cutoff, output_type="ndarray")
    if pairs.size == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, EDGE_ATTR_DIM), dtype=np.float32)
    else:
        i_arr = pairs[:, 0]
        j_arr = pairs[:, 1]
        i_is_prot = i_arr >= n_lig
        j_is_prot = j_arr >= n_lig
        types = np.where(
            i_is_prot & j_is_prot, EDGE_TYPE_INTRA_PROT,
            np.where(~i_is_prot & ~j_is_prot, EDGE_TYPE_INTRA_LIG, EDGE_TYPE_CROSS),
        )
        edge_index = np.stack([
            np.concatenate([i_arr, j_arr]),
            np.concatenate([j_arr, i_arr]),
        ], axis=0).astype(np.int64)
        types_sym = np.concatenate([types, types])
        edge_attr = np.zeros((edge_index.shape[1], EDGE_ATTR_DIM), dtype=np.float32)
        edge_attr[np.arange(edge_attr.shape[0]), types_sym] = 1.0

    data = Data(
        x=torch.from_numpy(x),
        pos=torch.from_numpy(pos),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        residue_idx=torch.from_numpy(residue_idx),
        y=torch.tensor([label], dtype=torch.long),
    )
    data.target = target
    data.complex_id = complex_id
    return data


def _iter_decoys(target, decoy_source, decoy_subdir, decoy_prefix):
    """Source-aware decoy iterator. Returns (iterator, missing_input_msg|None)."""
    if decoy_source == "ad":
        sdf = AD_TRANSFORMED_ROOT / target.upper() / f"{target.lower()}_AD_aligned.sdf"
        if not sdf.exists():
            return iter([]), f"AD aligned SDF missing: {sdf}"
        return iter_ad_poses(target), None
    # mol2 sources: dudez | goldilocks | extrema
    check_path = (DOCKING_DIR / target / decoy_subdir
                  / f"{decoy_prefix}_decoy_poses.mol2")
    if not check_path.exists():
        return iter([]), f"missing {check_path}"
    return iter_poses_from(target, "decoy", subdir=decoy_subdir, prefix=decoy_prefix), None


def process_target(target, pocket_cutoff, edge_cutoff, limit_per_target,
                   decoy_source, decoy_subdir, decoy_prefix):
    set_thread_env()
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s | %(name)s | %(message)s")
    wlog = logging.getLogger(f"worker.{target}")
    t0 = time.perf_counter()

    protein = load_protein(target)
    heavy = protein.select_atoms("not name H*")
    prot_pos_all = heavy.positions.astype(np.float32)
    prot_elements_raw = list(heavy.elements)
    prot_resids_all = heavy.resids.astype(np.int64)

    prot_elem_idx_all = np.array(
        [ELEMENT_TO_IDX.get(_normalize_element(e), -1) for e in prot_elements_raw],
        dtype=np.int64,
    )
    keep_prot = prot_elem_idx_all >= 0
    prot_pos = prot_pos_all[keep_prot]
    prot_elem_idx = prot_elem_idx_all[keep_prot]
    prot_resids = prot_resids_all[keep_prot]

    if prot_pos.shape[0] == 0:
        wlog.warning("%s: no protein heavy atoms in vocab; skipping target", target)
        return {"target": target, "items": [], "counts": _empty_counts(time.perf_counter() - t0)}

    prot_tree = cKDTree(prot_pos)

    items = []
    n_actives = n_decoys = 0
    n_skipped_no_lig_atoms = 0
    n_skipped_empty_pocket = 0
    n_lig_atoms_list = []
    n_pocket_atoms_list = []
    n_edges_list = []
    n_cross_edges_list = []

    # Actives always come from DUDE_Z mol2; decoys vary across benchmarks.
    # AD branch reads decoys from per-target SDFs in AD_TRANSFORMED_ROOT
    # (Phase 3 of TIER3_DECOY_BIAS_PLAN — already in the DUD-Z receptor frame).
    active_check = (DOCKING_DIR / target / MOL2_SUBDIR / f"{MOL2_PREFIX}_ligand_poses.mol2")
    decoy_iter, decoy_missing_msg = _iter_decoys(
        target, decoy_source, decoy_subdir, decoy_prefix
    )
    sources = []
    if active_check.exists():
        sources.append(("ligand", 1,
                        iter_poses_from(target, "ligand",
                                        subdir=MOL2_SUBDIR, prefix=MOL2_PREFIX)))
    else:
        wlog.warning("%s: missing %s; skipping actives", target, active_check)
    if decoy_missing_msg is None:
        sources.append(("decoy", 0, decoy_iter))
    else:
        wlog.warning("%s: %s; skipping decoys", target, decoy_missing_msg)

    for kind, label, pose_iter in sources:
        count_for_kind = 0
        for mol_id, mol in pose_iter:
            if limit_per_target is not None and count_for_kind >= limit_per_target:
                break
            extracted = _ligand_atoms(mol)
            if extracted is None:
                n_skipped_no_lig_atoms += 1
                continue
            lig_pos, lig_elements_raw = extracted
            lig_idx = np.array(
                [ELEMENT_TO_IDX.get(_normalize_element(e), -1) for e in lig_elements_raw],
                dtype=np.int64,
            )
            keep_lig = lig_idx >= 0
            if keep_lig.sum() == 0:
                n_skipped_no_lig_atoms += 1
                continue
            lig_pos = lig_pos[keep_lig]
            lig_elem_idx = lig_idx[keep_lig]

            contact_lists = prot_tree.query_ball_point(lig_pos, r=pocket_cutoff)
            flat = [i for sub in contact_lists for i in sub]
            if not flat:
                n_skipped_empty_pocket += 1
                continue
            pocket_indices = np.unique(np.fromiter(flat, dtype=np.int64))

            pocket_pos = prot_pos[pocket_indices]
            pocket_elem = prot_elem_idx[pocket_indices]
            pocket_res = prot_resids[pocket_indices]

            data = _build_data_object(
                lig_pos, lig_elem_idx,
                pocket_pos, pocket_elem, pocket_res,
                edge_cutoff, label, target, mol_id,
            )

            n_lig_atoms_list.append(int(lig_pos.shape[0]))
            n_pocket_atoms_list.append(int(pocket_pos.shape[0]))
            n_edges_list.append(int(data.edge_index.shape[1]))
            if data.edge_attr.numel() > 0:
                n_cross = int((data.edge_attr.argmax(dim=1) == EDGE_TYPE_CROSS).sum().item())
            else:
                n_cross = 0
            n_cross_edges_list.append(n_cross)

            payload = pickle.dumps(data, protocol=5)
            items.append((payload, {"target": target, "complex_id": mol_id, "label": label}))
            if label == 1:
                n_actives += 1
            else:
                n_decoys += 1
            count_for_kind += 1

    wall = time.perf_counter() - t0
    counts = {
        "n_actives": n_actives,
        "n_decoys": n_decoys,
        "n_complexes": n_actives + n_decoys,
        "n_skipped_no_lig_atoms": n_skipped_no_lig_atoms,
        "n_skipped_empty_pocket": n_skipped_empty_pocket,
        "mean_n_lig_atoms": float(np.mean(n_lig_atoms_list)) if n_lig_atoms_list else 0.0,
        "mean_n_pocket_atoms": float(np.mean(n_pocket_atoms_list)) if n_pocket_atoms_list else 0.0,
        "mean_n_edges": float(np.mean(n_edges_list)) if n_edges_list else 0.0,
        "frac_cross_edges": (sum(n_cross_edges_list) / sum(n_edges_list))
                             if sum(n_edges_list) > 0 else 0.0,
        "wall_seconds": wall,
    }
    wlog.info("%s done: %d active, %d decoy, %.1fs", target, n_actives, n_decoys, wall)
    return {"target": target, "items": items, "counts": counts}


def _empty_counts(wall):
    return {
        "n_actives": 0, "n_decoys": 0, "n_complexes": 0,
        "n_skipped_no_lig_atoms": 0, "n_skipped_empty_pocket": 0,
        "mean_n_lig_atoms": 0.0, "mean_n_pocket_atoms": 0.0,
        "mean_n_edges": 0.0, "frac_cross_edges": 0.0, "wall_seconds": wall,
    }


def write_outputs(output_dir, target_results, args, run_id, git_sha, wall_total):
    lmdb_path = output_dir / "dataset.lmdb"
    log.info("Opening LMDB at %s (map_size=%d GB)", lmdb_path, LMDB_MAP_SIZE // 1024**3)
    env = lmdb.open(str(lmdb_path), map_size=LMDB_MAP_SIZE, subdir=False, lock=False)

    index_rows = []
    next_key = 0
    with env.begin(write=True) as txn:
        for r in target_results:
            for payload, meta in r["items"]:
                key = f"{next_key:09d}".encode("ascii")
                txn.put(key, payload)
                index_rows.append({
                    "key": key.decode("ascii"),
                    "target": meta["target"],
                    "complex_id": meta["complex_id"],
                    "label": meta["label"],
                })
                next_key += 1
    env.sync()
    env.close()
    log.info("Wrote %d entries to LMDB", next_key)

    pd.DataFrame(index_rows).to_csv(output_dir / "index.csv", index=False)

    rows = [{"target": r["target"], **r["counts"]} for r in target_results]
    pd.DataFrame(rows).sort_values("target").to_csv(
        output_dir / "per_target.csv", index=False
    )

    manifest = {
        "run_id": run_id,
        "git_sha": git_sha,
        "venv": os.environ.get("VIRTUAL_ENV", "(none)"),
        "script": "scripts/06_build_egnn_dataset.py",
        "targets": list(args.targets),
        "decoy_source": args.decoy_source,   # canonical
        "decoy_set": args.decoy_set,         # legacy alias for 11/12 string-match
        "pocket_cutoff_ang": args.pocket_cutoff,
        "edge_cutoff_ang": args.edge_cutoff,
        "limit_per_target": args.limit_per_target,
        "node_feature_dim": NODE_FEATURE_DIM,
        "edge_attr_dim": EDGE_ATTR_DIM,
        "element_vocab": ELEMENT_VOCAB,
        "lmdb_path": "dataset.lmdb",
        "n_complexes_total": next_key,
        "per_target": {r["target"]: r["counts"] for r in target_results},
        "wall_seconds_total": wall_total,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    cfg = {
        "args": {
            **{k: v for k, v in vars(args).items() if k != "output_dir"},
            "output_dir": str(args.output_dir),
        },
        "constants": {
            "MOL2_PREFIX": MOL2_PREFIX,
            "RANDOM_SEED": RANDOM_SEED,
            "DOCKING_DIR": str(DOCKING_DIR),
            "ELEMENT_VOCAB": ELEMENT_VOCAB,
            "NODE_FEATURE_DIM": NODE_FEATURE_DIM,
            "EDGE_ATTR_DIM": EDGE_ATTR_DIM,
        },
    }
    cfg["args"]["targets"] = list(cfg["args"]["targets"])
    (output_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return manifest


def write_report(output_dir, manifest):
    n = manifest["n_complexes_total"]
    n_targets = len(manifest["targets"])
    wall = manifest["wall_seconds_total"]
    extrapolated = wall * 43 / max(n_targets, 1)
    lines = [
        "# 06_build_egnn_dataset — Smoke run report",
        "",
        f"- run_id: `{manifest['run_id']}`",
        f"- git_sha: `{manifest['git_sha']}`",
        f"- targets: {', '.join(manifest['targets'])}",
        f"- pocket_cutoff: {manifest['pocket_cutoff_ang']} Å, edge_cutoff: {manifest['edge_cutoff_ang']} Å",
        f"- node_feature_dim: {manifest['node_feature_dim']}, edge_attr_dim: {manifest['edge_attr_dim']}",
        f"- total complexes: **{n}**",
        f"- wall (this run, {n_targets} targets): **{wall:.1f}s**",
        f"- extrapolated wall (43 targets, optimistic linear): **{extrapolated:.0f}s ≈ {extrapolated/60:.1f} min**",
        "",
        "## Per-target",
        "",
        "| target | actives | decoys | mean lig atoms | mean pocket atoms | mean edges | frac cross | wall (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tgt, c in manifest["per_target"].items():
        lines.append(
            f"| {tgt} | {c['n_actives']} | {c['n_decoys']} | "
            f"{c['mean_n_lig_atoms']:.1f} | {c['mean_n_pocket_atoms']:.1f} | "
            f"{c['mean_n_edges']:.0f} | {c['frac_cross_edges']:.3f} | "
            f"{c['wall_seconds']:.1f} |"
        )
    skipped_lines = []
    for tgt, c in manifest["per_target"].items():
        if c["n_skipped_no_lig_atoms"] or c["n_skipped_empty_pocket"]:
            skipped_lines.append(
                f"- {tgt}: skipped {c['n_skipped_no_lig_atoms']} (no lig atoms in vocab), "
                f"{c['n_skipped_empty_pocket']} (empty pocket)"
            )
    if skipped_lines:
        lines += ["", "## Skipped", ""] + skipped_lines
    lines += ["", "## Notes", "", "_Filled in after review by the operator._", ""]
    (output_dir / "REPORT.md").write_text("\n".join(lines))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", nargs="+", default=None,
                   help="Targets to process (default: smoke set AA2AR ADRB2 EGFR; pass --all-targets for all 43).")
    p.add_argument("--all-targets", action="store_true",
                   help="Process all 43 DUDE-Z targets (mutually exclusive with --targets).")
    p.add_argument("--n-workers", type=int, default=None,
                   help="CPU workers (default: min(len(targets), N_WORKERS)).")
    p.add_argument("--run-id", type=str, default=None,
                   help="Override run id (default: YYYYMMDD-HHMMSS-<git short>).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override output dir (default: outputs/06_build_egnn_dataset/<run_id>).")
    p.add_argument("--pocket-cutoff", type=float, default=8.0)
    p.add_argument("--edge-cutoff", type=float, default=4.5)
    p.add_argument("--limit-per-target", type=int, default=None,
                   help="Debug: at most N actives + N decoys per target.")
    p.add_argument("--decoy-source", choices=list(DECOY_SOURCES.keys()),
                   default=None,
                   help="Which decoy source to use (TIER3_DECOY_BIAS_PLAN Phase 3 §1). "
                        "Outputs to <run_id>/<decoy_source>/dataset.lmdb.")
    p.add_argument("--decoy-set", choices=list(MOL2_SUBDIRS.keys()),
                   default=None,
                   help="DEPRECATED: use --decoy-source. Kept for backward "
                        "compatibility with 11_eval_goldilocks / 12_verify_plec_groupkfold. "
                        "Mutually exclusive with --decoy-source.")
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
    # DUDE-Z target dirs are upper-case; tolerate lowercase CLI input.
    args.targets = [t.upper() for t in args.targets]

    # Resolve decoy source. --decoy-source is canonical; --decoy-set is a
    # deprecated alias preserved so 11/12 can keep grepping `decoy_set` in
    # the manifest.
    if args.decoy_source is not None and args.decoy_set is not None:
        sys.stderr.write("--decoy-source and --decoy-set are mutually exclusive\n")
        return 2
    if args.decoy_source is None and args.decoy_set is None:
        args.decoy_source = "dudez"
    elif args.decoy_source is None:
        args.decoy_source = _DECOY_SET_TO_SOURCE[args.decoy_set]
    args.decoy_set = _SOURCE_TO_DECOY_SET[args.decoy_source]

    # Defensive: drop ABL1 from any AD run (apo/holo conformational mismatch).
    args.targets = [t for t in args.targets
                    if not (args.decoy_source == "ad"
                            and t.upper() in AD_EXCLUDED_TARGETS)]

    run_id = _make_run_id(args.run_id)
    if args.output_dir is None:
        # Nested layout: <run_id>/<decoy_source>/dataset.lmdb (so a single
        # run_id can host multiple decoy sources side-by-side under Phase 5).
        args.output_dir = (PROJECT_ROOT / "outputs" / "06_build_egnn_dataset"
                           / run_id / args.decoy_source)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(level=logging.INFO, log_file=args.output_dir / "build.log")
    log.setLevel(logging.INFO)

    git_sha = _git_short_sha()
    log.info("=" * 70)
    log.info("06_build_egnn_dataset starting")
    log.info("  run_id: %s", run_id)
    log.info("  git_sha: %s", git_sha)
    log.info("  venv: %s", os.environ.get("VIRTUAL_ENV", "(none)"))
    log.info("  cuda_visible: %s (informational; this script is CPU-only)",
             torch.cuda.is_available())
    log.info("  targets: %s", args.targets)
    log.info("  output_dir: %s", args.output_dir)
    log.info("  pocket_cutoff: %.1f Å", args.pocket_cutoff)
    log.info("  edge_cutoff: %.1f Å", args.edge_cutoff)
    log.info("  limit_per_target: %s", args.limit_per_target)
    log.info("  decoy_source: %s   (legacy decoy_set: %s)",
             args.decoy_source, args.decoy_set)
    subdir, prefix = DECOY_SOURCES[args.decoy_source]
    log.info("  subdir/prefix: %s / %s", subdir, prefix)
    if args.decoy_source == "ad":
        log.info("  AD decoy SDF root: %s", AD_TRANSFORMED_ROOT)
        log.info("  AD excluded targets (always dropped): %s",
                 sorted(AD_EXCLUDED_TARGETS))

    n_workers = args.n_workers or min(len(args.targets), N_WORKERS)
    log.info("  n_workers: %d", n_workers)

    for t in args.targets:
        if not (DOCKING_DIR / t).is_dir():
            log.error("Target dir not found: %s", DOCKING_DIR / t)
            return 2

    # Drop targets that lack the requested decoy source's input file
    # (Goldilocks is missing AMPC; some Extrema targets may be missing too;
    # AD covers 41 of 43 by construction — ABL1 is excluded above and
    # 1 more target may not be in the AD dataset).
    if args.decoy_source != "dudez":
        kept = []
        skipped = []
        for t in args.targets:
            if args.decoy_source == "ad":
                decoy_path = (AD_TRANSFORMED_ROOT / t.upper()
                              / f"{t.lower()}_AD_aligned.sdf")
            else:
                decoy_path = (DOCKING_DIR / t / subdir
                              / f"{prefix}_decoy_poses.mol2")
            if decoy_path.exists():
                kept.append(t)
            else:
                skipped.append(t)
        if skipped:
            log.warning("Skipping %d target(s) missing %s decoys: %s",
                        len(skipped), args.decoy_source, skipped)
        args.targets = kept
        if not args.targets:
            log.error("No targets have %s decoys; aborting", args.decoy_source)
            return 2

    t0 = time.perf_counter()
    target_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(process_target, t, args.pocket_cutoff,
                        args.edge_cutoff, args.limit_per_target,
                        args.decoy_source, subdir, prefix): t
            for t in args.targets
        }
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="targets", unit="tgt"):
            t = futures[fut]
            try:
                r = fut.result()
                target_results.append(r)
                c = r["counts"]
                log.info("  %s: %d active + %d decoy = %d, "
                         "mean_lig=%.1f mean_pocket=%.1f mean_edges=%.0f cross=%.3f, %.1fs",
                         t, c["n_actives"], c["n_decoys"], c["n_complexes"],
                         c["mean_n_lig_atoms"], c["mean_n_pocket_atoms"],
                         c["mean_n_edges"], c["frac_cross_edges"], c["wall_seconds"])
            except Exception as e:
                log.exception("FAILED target %s: %s", t, e)

    wall_total = time.perf_counter() - t0
    log.info("All targets done in %.1fs", wall_total)

    order = {t: i for i, t in enumerate(args.targets)}
    target_results.sort(key=lambda r: order.get(r["target"], 1e9))

    manifest = write_outputs(args.output_dir, target_results, args,
                             run_id, git_sha, wall_total)
    write_report(args.output_dir, manifest)

    extrapolated = wall_total * 43 / max(len(args.targets), 1)
    log.info("Estimated 43-target wall: %.0fs ≈ %.1f min "
             "(linear extrapolation from %d-target run)",
             extrapolated, extrapolated / 60, len(args.targets))
    log.info("Outputs at %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
