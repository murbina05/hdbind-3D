"""Smoke tests for src/pocket_utils.py."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import numpy as np
import pytest

SMOKE_TARGET = "AA2AR"


def test_save_load_pocket_residues():
    from src.pocket_utils import save_pocket_residues, load_pocket_residues
    from config import DATA_DIR

    residues = ["ALA_42_A", "TRP_246_A", "HIS_278_A"]
    # Temporarily point to a tmp dir
    tmp = tempfile.mkdtemp()
    orig_data_dir = None

    # Patch DATA_DIR for this test
    import src.pocket_utils as pu
    orig = pu.DATA_DIR
    pu.DATA_DIR = Path(tmp)
    try:
        save_pocket_residues(SMOKE_TARGET, residues)
        loaded = load_pocket_residues(SMOKE_TARGET)
        assert loaded == residues
    finally:
        pu.DATA_DIR = orig


def test_find_pocket_residues_requires_data():
    """Integration test — requires DUDE-Z data mounted."""
    from config import DOCKING_DIR
    if not (DOCKING_DIR / SMOKE_TARGET).exists():
        pytest.skip("DUDE-Z data not mounted")

    from src.data_loading import load_protein, load_crystal_ligand
    from src.pocket_utils import find_pocket_residues
    from rdkit.Chem import rdMolTransforms

    protein = load_protein(SMOKE_TARGET)
    lig = load_crystal_ligand(SMOKE_TARGET)
    conf = lig.GetConformer()
    coords = conf.GetPositions().astype(np.float32)

    pocket = find_pocket_residues(protein, coords)
    assert isinstance(pocket, list)
    assert len(pocket) > 0
    for r in pocket:
        assert "_" in r
