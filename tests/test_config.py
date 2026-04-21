"""Smoke tests for config.py."""
from __future__ import annotations
import pytest
from config import (
    get_all_targets, N_WORKERS, NESTED_WORKERS, OMP_THREAD_LIMIT,
    GPU_BATCH_SIZE, HD_DIM, PLEC_SIZE, POCKET_CUTOFF_ANG,
    CV_FOLDS, BEDROC_ALPHA, EF_FRACTIONS, THRESHOLD_SCAN,
    PROLIF_INTERACTIONS, RESIDUE_PROPERTIES,
    PROJECT_ROOT, DATA_DIR, RESULTS_DIR,
)


def test_n_workers_positive():
    assert N_WORKERS >= 1


def test_n_workers_capped():
    import os
    assert N_WORKERS <= min(32, os.cpu_count() or 1)


def test_nested_workers():
    assert NESTED_WORKERS == 4


def test_hd_dim():
    assert HD_DIM == 10000


def test_plec_size():
    assert PLEC_SIZE == 4096


def test_pocket_cutoff():
    assert POCKET_CUTOFF_ANG == 6.0


def test_prolif_interactions_count():
    assert len(PROLIF_INTERACTIONS) == 14


def test_residue_properties_keys():
    for key, val in RESIDUE_PROPERTIES.items():
        assert len(val) == 5, f"{key} property vector must have length 5"


def test_threshold_scan():
    assert len(THRESHOLD_SCAN) > 0
    assert all(0 < t < 1 for t in THRESHOLD_SCAN)


def test_ef_fractions():
    assert 0.01 in EF_FRACTIONS
    assert 0.05 in EF_FRACTIONS


def test_project_root_exists():
    assert PROJECT_ROOT.exists()


def test_get_all_targets_returns_list():
    # Requires DOCKING_DIR to exist; skip gracefully if not mounted
    try:
        targets = get_all_targets()
        assert isinstance(targets, list)
        assert len(targets) > 0
    except (FileNotFoundError, OSError):
        pytest.skip("DUDE-Z data not mounted — skipping target enumeration test")
