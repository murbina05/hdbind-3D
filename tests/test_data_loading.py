"""Smoke tests for src/data_loading.py — requires DUDE-Z data mounted."""
from __future__ import annotations
import pytest

SMOKE_TARGET = "AA2AR"


@pytest.fixture(scope="module")
def dude_available():
    from config import DOCKING_DIR
    if not (DOCKING_DIR / SMOKE_TARGET).exists():
        pytest.skip("DUDE-Z data not mounted — skipping data_loading tests")


def test_load_protein(dude_available):
    from src.data_loading import load_protein
    import MDAnalysis as mda
    prot = load_protein(SMOKE_TARGET)
    assert isinstance(prot, mda.Universe)
    assert prot.atoms.n_atoms > 0


def test_load_protein_missing_raises():
    from src.data_loading import load_protein
    with pytest.raises(FileNotFoundError):
        load_protein("__nonexistent_target__")


def test_load_crystal_ligand(dude_available):
    from src.data_loading import load_crystal_ligand
    mol = load_crystal_ligand(SMOKE_TARGET)
    assert mol is not None
    assert mol.GetNumAtoms() > 0


def test_iter_poses_actives(dude_available):
    from src.data_loading import iter_poses
    actives = list(iter_poses(SMOKE_TARGET, "ligand"))
    assert len(actives) > 0
    for mol_id, mol in actives:
        assert isinstance(mol_id, str)
        assert mol is not None
        assert mol.GetNumConformers() > 0


def test_iter_poses_decoys(dude_available):
    from src.data_loading import iter_poses
    decoys = list(iter_poses(SMOKE_TARGET, "decoy"))
    assert len(decoys) > 0


def test_load_labels(dude_available):
    from src.data_loading import load_labels
    ids, labels = load_labels(SMOKE_TARGET)
    assert len(ids) == len(labels)
    assert 1 in labels
    assert 0 in labels
