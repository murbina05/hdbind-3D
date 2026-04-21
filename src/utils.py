from __future__ import annotations
import os
import time
import pickle
import logging
from pathlib import Path
from contextlib import contextmanager
from config import OMP_THREAD_LIMIT


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
