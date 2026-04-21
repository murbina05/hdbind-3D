"""
Evaluate all featurizers on alternative decoy sets (Extrema, Goldilocks).
Actives always from DUDE_Z/dudez_1pt0LD_ligand_poses.mol2.
Decoys from variant-specific directories.

Outputs:
  results/decoy_sets/{decoy_set}/{featurizer}/{target}.pkl
  results/decoy_sets/per_target_metrics.csv
  results/logs/04_eval_decoy_sets.log

Usage:
    python scripts/04_eval_decoy_sets.py [--targets TARGET [...]]
                                          [--decoy-sets SET [...]]
                                          [--n-workers N]
"""
from __future__ import annotations
import argparse
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import get_all_targets, MOL2_SUBDIRS, N_WORKERS, RESULTS_DIR, DOCKING_DIR
from src.utils import setup_logging, set_thread_env, snap_mem, timer, save_pickle, load_pickle
from src.data_loading import iter_poses_from
from src.pocket_utils import load_pocket_residues
from src.featurizers.plif import PLIFFeaturizer
from src.featurizers.plec import PLECFeaturizer
from src.featurizers.pocket import PocketFeaturizer
from src.featurizers.distances import DistanceFeaturizer, _build_residue_property_vector
from src.gpu_utils import gpu_hd_project, log_gpu_status
from src.training import cross_validate

log = logging.getLogger(__name__)

DECOY_SETS_DIR = RESULTS_DIR / "decoy_sets"
_LOG_FILE = RESULTS_DIR / "logs" / f"{Path(__file__).stem}.log"
PRIMARY_METRICS = ["roc_auc", "pr_auc", "bedroc", "ef_1pct", "ef_5pct", "mcc", "f1"]


def _load_records(target: str, subdir: str, prefix: str) -> list[dict]:
    """Actives from DUDE_Z + decoys from subdir/prefix."""
    records = [
        {"id": mid, "mol": mol, "label": 1}
        for mid, mol in iter_poses_from(target, "ligand", "DUDE_Z", "dudez_1pt0LD")
    ]
    records += [
        {"id": mid, "mol": mol, "label": 0}
        for mid, mol in iter_poses_from(target, "decoy", subdir, prefix)
    ]
    return records


def _featurize_all(
    target: str,
    protein_path: str,
    records: list[dict],
    pocket_residues: list[str],
    decoy_set: str,
) -> dict[str, dict]:
    """Run all featurizers. Returns {feat_name: {mol_id: vec}}."""
    results: dict[str, dict] = {}

    # PLIF — must run first (GRIM reads its output)
    plif_feats = PLIFFeaturizer().featurize_target(
        target, protein_path, records, pocket_residues
    )
    results["plif"] = plif_feats

    # GRIM — apply saved DUDE_Z projection to these PLIF vectors
    proj_path = RESULTS_DIR / "tier1" / "grim" / f"{target}_projection.pkl"
    if proj_path.exists():
        proj = load_pickle(proj_path)
        ids = list(plif_feats.keys())
        if ids:
            X = np.stack([plif_feats[i] for i in ids]).astype(np.float32)
            hd_vecs = gpu_hd_project(X, proj)
            results["grim"] = {mid: hd_vecs[j] for j, mid in enumerate(ids)}
    else:
        log.warning("%s/%s: projection not found at %s — skipping GRIM", target, decoy_set, proj_path)

    # PLEC
    results["plec"] = PLECFeaturizer().featurize_target(
        target, protein_path, records, pocket_residues
    )

    # Pocket
    results["pocket_featurizer"] = PocketFeaturizer().featurize_target(
        target, protein_path, records, pocket_residues
    )

    # Distance matrix
    dist_results = DistanceFeaturizer().featurize_target(
        target, protein_path, records, pocket_residues
    )
    results["distance_matrix"] = dist_results

    # DistChem — derive from distance results to avoid recomputing distances
    res_props = _build_residue_property_vector(pocket_residues)
    results["distance_chemistry"] = {
        mid: np.concatenate([dist_vec, res_props]).astype(np.float32)
        for mid, dist_vec in dist_results.items()
    }

    return results


def _evaluate(
    all_features: dict[str, dict],
    label_map: dict[str, int],
) -> dict[str, dict]:
    """Run 5-fold CV per featurizer. Returns {feat_name: aggregated_metrics}."""
    metrics: dict[str, dict] = {}
    for feat_name, feat_dict in all_features.items():
        valid_ids = [i for i in feat_dict if i in label_map]
        if len(valid_ids) < 10:
            log.warning("  %s: only %d valid IDs — skipping CV", feat_name, len(valid_ids))
            continue
        y = np.array([label_map[i] for i in valid_ids], dtype=int)
        if y.sum() == 0 or (1 - y).sum() == 0:
            log.warning("  %s: single class — skipping CV", feat_name)
            continue
        X = np.stack([feat_dict[i] for i in valid_ids]).astype(np.float32)
        try:
            metrics[feat_name] = cross_validate(X, y)
        except Exception as e:
            log.error("  %s CV failed: %s", feat_name, e)
    return metrics


def process_target(
    target: str,
    decoy_set: str,
    subdir: str,
    prefix: str,
) -> tuple[str, str, dict | None, float, float]:
    set_thread_env()
    t0, m0 = time.perf_counter(), snap_mem()
    try:
        pocket_residues = load_pocket_residues(target)
        protein_path = str(DOCKING_DIR / target / "rec.crg.pdb")
        records = _load_records(target, subdir, prefix)
        log.info("%s/%s: loaded %d records (%d actives, %d decoys)",
                 decoy_set, target, len(records),
                 sum(r["label"] for r in records),
                 sum(1 - r["label"] for r in records))

        all_features = _featurize_all(target, protein_path, records, pocket_residues, decoy_set)

        # Save features to disk
        for feat_name, feat_dict in all_features.items():
            out = DECOY_SETS_DIR / decoy_set / feat_name / f"{target}.pkl"
            save_pickle(feat_dict, out)

        label_map = {r["id"]: r["label"] for r in records}
        metrics = _evaluate(all_features, label_map)

        return target, decoy_set, metrics, time.perf_counter() - t0, snap_mem() - m0
    except Exception as e:
        log.error("FAILED %s/%s: %s", decoy_set, target, e)
        return target, decoy_set, None, time.perf_counter() - t0, 0.0


def main(targets: list[str], decoy_sets: list[str], n_workers: int) -> None:
    setup_logging(log_file=_LOG_FILE)
    log_gpu_status()

    # Validate decoy set names
    work_items: list[tuple[str, str, str, str]] = []
    for ds in decoy_sets:
        if ds not in MOL2_SUBDIRS:
            log.error("Unknown decoy set '%s'. Known: %s", ds, list(MOL2_SUBDIRS))
            continue
        subdir, prefix = MOL2_SUBDIRS[ds]
        for target in targets:
            work_items.append((target, ds, subdir, prefix))

    log.info("Running %d jobs (%d targets × %d decoy sets) with %d workers",
             len(work_items), len(targets), len(decoy_sets), n_workers)

    rows: list[dict] = []
    with timer("Decoy set evaluation total"):
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            futures = {
                pool.submit(process_target, t, ds, sd, pf): (t, ds)
                for t, ds, sd, pf in work_items
            }
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="decoy eval", unit="job"):
                target, decoy_set, metrics, elapsed, rss_mb = future.result()
                log.info("[profile] %s/%s: %.2fs | ΔRSS %+.0f MB",
                         decoy_set, target, elapsed, rss_mb)
                if metrics is None:
                    continue
                for feat_name, agg in metrics.items():
                    row: dict = {"decoy_set": decoy_set, "featurizer": feat_name, "target": target}
                    for metric, vals in agg.items():
                        row[f"{metric}_mean"] = vals["mean"]
                        row[f"{metric}_std"] = vals["std"]
                    rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        out_path = DECOY_SETS_DIR / "per_target_metrics.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        log.info("Saved %d metric rows to %s", len(df), out_path)
    else:
        log.error("No metrics collected — check that decoy set mol2 files exist")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", default=get_all_targets())
    parser.add_argument(
        "--decoy-sets", nargs="+",
        default=[k for k in MOL2_SUBDIRS if k != "DUDE_Z"],
    )
    parser.add_argument("--n-workers", type=int, default=N_WORKERS)
    args = parser.parse_args()
    main(args.targets, args.decoy_sets, args.n_workers)
