from __future__ import annotations
import json
import os
import time
import pickle
import logging
from pathlib import Path
from contextlib import contextmanager

import numpy as np

from config import OMP_THREAD_LIMIT, RANDOM_SEED


def setup_logging(level: int = logging.INFO, log_file: Path | str | None = None) -> None:
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    # Terminal: WARNING+ only so tqdm bars stay clean
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(ch)

    # File: full INFO+ log
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        root.addHandler(fh)

    logging.getLogger("MDAnalysis").setLevel(logging.WARNING)
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        pass


def snap_mem() -> float:
    """Current process RSS in MB. Returns 0 if psutil unavailable."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1_048_576
    except ImportError:
        return 0.0


def set_thread_env() -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(OMP_THREAD_LIMIT)


@contextmanager
def timer(label: str):
    log = logging.getLogger("timer")
    t0 = time.perf_counter()
    yield
    log.info("%s — %.1f s", label, time.perf_counter() - t0)


def save_pickle(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=5)


def load_pickle(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def canonical_split_indices(
    keys: np.ndarray,
    labels: np.ndarray,
    targets: np.ndarray,
    n_folds: int,
    cv: str = "groupkfold",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compute the per-fold (train_idx, val_idx) splits used by
    scripts/07_train_egnn.py. Single source of truth for split logic so that
    eval-time replay (10/11/12) cannot drift from training-time semantics
    (root cause of the prior Goldilocks fold-split inflation 0.062 → 0.192).

    cv='groupkfold' uses sklearn.GroupKFold over `targets` (deterministic, no
    seed). cv='stratified' uses StratifiedKFold(shuffle=True,
    random_state=config.RANDOM_SEED) over `labels`.
    """
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    if cv == "groupkfold":
        splitter = GroupKFold(n_splits=n_folds)
        split_iter = splitter.split(keys, labels, targets)
    elif cv == "stratified":
        splitter = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED,
        )
        split_iter = splitter.split(keys, labels)
    else:
        raise ValueError(f"Unknown cv mode: {cv!r}")
    return [(np.asarray(tr), np.asarray(va)) for tr, va in split_iter]


def load_canonical_groupkfold(
    manifest_path: str | Path,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Replay the canonical per-fold (train_idx, val_idx) splits recorded by
    scripts/07_train_egnn.py from its manifest.json. Indices apply to the
    `targets`-filtered `index.csv` from the manifest's `dataset_dir`, in row
    order after `.reset_index(drop=True)` — same array layout as 07's own
    keys_arr/targets_arr/labels_arr.

    Honours the manifest's `cv` field so the replay matches what training
    actually did (groupkfold or stratified). Defaults to 'groupkfold' for
    older manifests that predate the field.
    """
    import pandas as pd

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    full_index = pd.read_csv(
        Path(manifest["dataset_dir"]) / "index.csv", dtype={"key": str},
    )
    idx = full_index[
        full_index["target"].isin(set(manifest["targets"]))
    ].reset_index(drop=True)
    return canonical_split_indices(
        keys=idx["key"].astype(str).values,
        labels=idx["label"].values.astype(int),
        targets=idx["target"].values,
        n_folds=int(manifest["n_folds"]),
        cv=manifest.get("cv", "groupkfold"),
    )
