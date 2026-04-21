"""
Compute pocket+ligand features for all 43 DUDE-Z targets.

Usage:
    python scripts/02a_run_pocket_feat.py [--targets TARGET [TARGET ...]] [--n-workers N]
"""
from __future__ import annotations
import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

from config import get_all_targets, N_WORKERS, RESULTS_DIR, DOCKING_DIR
from src.utils import setup_logging, set_thread_env, snap_mem, timer, save_pickle
from src.data_loading import load_ligand_records
from src.pocket_utils import load_pocket_residues
from src.featurizers.pocket import PocketFeaturizer
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
        featurizer = PocketFeaturizer()
        features = featurizer.featurize_target(target, protein_path, ligand_records, pocket_residues)
        return target, features, time.perf_counter() - t0, snap_mem() - m0
    except Exception as e:
        log.error("FAILED %s: %s", target, e)
        return target, None, time.perf_counter() - t0, 0.0


def main(targets: list[str], n_workers: int) -> None:
    setup_logging(log_file=_LOG_FILE)
    log_gpu_status()
    log.info("Running PocketFeaturizer on %d targets with %d workers", len(targets), n_workers)

    featurizer = PocketFeaturizer()
    with timer("PocketFeaturizer total"):
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(process_target, t): t for t in targets}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Pocket", unit="target"):
                target, features, elapsed, rss_mb = future.result()
                log.info("[profile] %s: %.2fs | ΔRSS %+.0f MB", target, elapsed, rss_mb)
                if features is not None:
                    out = featurizer.output_path(RESULTS_DIR / "tier2", target)
                    save_pickle(features, out)
                    log.info("Saved %s (%d complexes)", target, len(features))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", default=get_all_targets())
    parser.add_argument("--n-workers", type=int, default=N_WORKERS)
    args = parser.parse_args()
    main(args.targets, args.n_workers)
