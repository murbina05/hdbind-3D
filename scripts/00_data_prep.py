"""
Data preparation pipeline for DUDE-Z benchmark.
Validates all 43 targets, identifies pocket residues, generates 3D conformers,
and saves metadata.csv.

Usage:
    python scripts/00_data_prep.py [--targets TARGET [TARGET ...]] [--n-workers N]
"""
from __future__ import annotations
import argparse
import logging
import shutil
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from config import get_all_targets, N_WORKERS, DOCKING_DIR, DATA_DIR, RESULTS_DIR, PROJECT_ROOT
from src.utils import setup_logging, set_thread_env, timer, save_pickle
from src.data_loading import load_protein, load_crystal_ligand, iter_poses
from src.pocket_utils import find_pocket_residues, save_pocket_residues

log = logging.getLogger(__name__)


def prep_target(target: str) -> tuple[str, dict | None]:
    """Top-level worker — must be importable (no lambdas)."""
    set_thread_env()
    try:
        # 1. Validate files exist
        pdb_path = DOCKING_DIR / target / "rec.crg.pdb"
        xtal_path = DOCKING_DIR / target / "xtal-lig.pdb"
        if not pdb_path.exists():
            raise FileNotFoundError(f"rec.pdb missing: {pdb_path}")
        if not xtal_path.exists():
            raise FileNotFoundError(f"xtal-lig.pdb missing: {xtal_path}")

        # 2. Find pocket residues
        protein = load_protein(target)
        crystal_lig = load_crystal_ligand(target)
        if crystal_lig is None:
            raise ValueError(f"Crystal ligand parse failed for {target}")
        from rdkit.Chem import rdMolTransforms
        conf = crystal_lig.GetConformer()
        import numpy as np
        lig_coords = conf.GetPositions().astype(np.float32)
        pocket_residues = find_pocket_residues(protein, lig_coords)
        save_pocket_residues(target, pocket_residues)

        # 3. Count actives and decoys from docked pose mol2 files
        n_actives = sum(1 for _ in iter_poses(target, "ligand"))
        n_decoys  = sum(1 for _ in iter_poses(target, "decoy"))

        meta = {
            "target": target,
            "n_pocket_residues": len(pocket_residues),
            "n_actives": n_actives,
            "n_decoys": n_decoys,
        }
        log.info("%s: %d pocket residues, %d actives, %d decoys",
                 target, len(pocket_residues), n_actives, n_decoys)
        return target, meta

    except Exception as e:
        log.error("FAILED %s: %s", target, e)
        return target, None


def main(targets: list[str], n_workers: int) -> None:
    setup_logging()
    log.info("Data prep: %d targets, %d workers", len(targets), n_workers)

    # Copy config snapshot
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(PROJECT_ROOT / "config.py", RESULTS_DIR / "config_snapshot.py")

    metadata = []
    with timer("Data prep total"):
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(prep_target, t): t for t in targets}
            for future in as_completed(futures):
                target, meta = future.result()
                if meta is not None:
                    metadata.append(meta)

    # Write metadata CSV
    if metadata:
        meta_path = DATA_DIR / "metadata.csv"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["target", "n_pocket_residues", "n_actives", "n_decoys"])
            writer.writeheader()
            writer.writerows(sorted(metadata, key=lambda x: x["target"]))
        log.info("Metadata saved to %s (%d targets)", meta_path, len(metadata))

    failed = len(targets) - len(metadata)
    log.info("Done: %d/%d targets OK, %d failed", len(metadata), len(targets), failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", default=get_all_targets())
    parser.add_argument("--n-workers", type=int, default=N_WORKERS)
    args = parser.parse_args()
    main(args.targets, args.n_workers)
