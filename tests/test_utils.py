"""Smoke tests for src/utils.py."""
from __future__ import annotations
import os
import tempfile
from pathlib import Path
import pytest
from src.utils import save_pickle, load_pickle, timer, set_thread_env


def test_pickle_roundtrip():
    obj = {"key": [1, 2, 3], "val": "hello"}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sub" / "test.pkl"
        save_pickle(obj, path)
        loaded = load_pickle(path)
    assert loaded == obj


def test_pickle_creates_parent_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "a" / "b" / "c" / "test.pkl"
        save_pickle(42, path)
        assert path.exists()


def test_timer_runs():
    import time
    with timer("test_label"):
        time.sleep(0.01)


def test_set_thread_env():
    set_thread_env()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert os.environ.get(var) == "4"
