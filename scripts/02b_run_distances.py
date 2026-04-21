"""
Compute GPU-accelerated pairwise distance features for all 43 DUDE-Z targets.
GPU parallelizes the distance computation across the full ligand batch per target;
targets are processed serially to avoid CUDA fork issues.

Usage:
    python scripts/02b_run_distances.py [--targets TARGET [TARGET ...]] [--n-workers N]
"""
from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path
from tqdm import tqdm

from config import get_all_targets, N_WORKERS, RESULTS_DIR, DOCKING_DIR
from src.utils import setup_logging, set_thread_env, snap_mem, timer, save_pickle
from src.data_loading import load_ligand_records
from src.pocket_utils import load_pocket_residues
from src.featurizers.distances import DistanceFeaturizer
from src.gpu_utils import log_gpu_status

log = logging.getLogger(__name__)

_LOG_FILE = RESULTS_DIR / "logs" / f"{Path(__file__).stem}.log"


def process_target(target: str) -> tuple[str, dict | None, float, float]:
    set_thread_env()
    t0, m0 = time.perf_counter(), snap_mem()
    try:
        pocket_residues = load_pocket_residues(target)
        protein_path = str(DOCKING_DIR / target / "rec.crg.pdb")
        ligand_records = load_ligand_records(target)
        featurizer = DistanceFeaturizer()
        features = featurizer.featurize_target(target, protein_path, ligand_records, pocket_residues)
        return target, features, time.perf_counter() - t0, snap_mem() - m0
    except Exception as e:
        log.error("FAILED %s: %s", target, e)
        return target, None, time.perf_counter() - t0, 0.0


def main(targets: list[str], n_workers: int) -> None:
    setup_logging(log_file=_LOG_FILE)
    log_gpu_status()
    log.info("Running DistanceFeaturizer on %d targets", len(targets))

    try:
        import torch
        _cuda = torch.cuda.is_available()
    except ImportError:
        _cuda = False

    featurizer = DistanceFeaturizer()
    with timer("Distance total"):
        for target in tqdm(targets, desc="Distances", unit="target"):
            if _cuda:
                torch.cuda.reset_peak_memory_stats()
            target_name, features, elapsed, rss_mb = process_target(target)
            gpu_mb = torch.cuda.max_memory_allocated() / 1_048_576 if _cuda else 0.0
            log.info("[profile] %s: %.2fs | ΔRSS %+.0f MB | GPU peak %.0f MB",
                     target_name, elapsed, rss_mb, gpu_mb)
            if features is not None:
                out = featurizer.output_path(RESULTS_DIR / "tier2", target_name)
                save_pickle(features, out)
                log.info("Saved %s (%d complexes)", target_name, len(features))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", default=get_all_targets())
    parser.add_argument("--n-workers", type=int, default=N_WORKERS)
    args = parser.parse_args()
    main(args.targets, args.n_workers)
