"""
Smoke tests for all featurizers — requires DUDE-Z data mounted for integration tests.
Unit-level tests use synthetic data where possible.
"""
from __future__ import annotations
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

SMOKE_TARGET = "AA2AR"


def _make_synthetic_mol(n_atoms: int = 10) -> Chem.Mol:
    """Build a minimal 3D mol for testing."""
    mol = Chem.MolFromSmiles("c1ccccc1")  # benzene
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    return Chem.RemoveHs(mol)


def _make_synthetic_records(n: int = 5) -> list[dict]:
    mol = _make_synthetic_mol()
    return [{"id": f"lig_{i}", "mol": mol, "label": int(i == 0)} for i in range(n)]


def _make_synthetic_pocket() -> list[str]:
    return ["ALA_10_A", "GLY_20_A", "PHE_30_A", "TRP_40_A"]


# ── BaseFeaturizer ────────────────────────────────────────────────────────

def test_base_featurizer_output_path(tmp_path):
    from src.featurizers.plif import PLIFFeaturizer
    feat = PLIFFeaturizer()
    path = feat.output_path(tmp_path, "AA2AR")
    assert path.endswith("AA2AR.pkl")
    assert "plif" in path


# ── DistanceFeaturizer (GPU-accelerated, no external data needed) ──────────

def test_distance_featurizer_synthetic(tmp_path):
    from src.featurizers.distances import DistanceFeaturizer
    import tempfile, os

    mol = _make_synthetic_mol()
    records = _make_synthetic_records(3)
    pocket = _make_synthetic_pocket()

    # Write a minimal PDB for the "protein"
    pdb_content = """\
ATOM      1  CA  ALA A  10       1.000   2.000   3.000  1.00  0.00           C
ATOM      2  CA  GLY A  20       4.000   5.000   6.000  1.00  0.00           C
ATOM      3  CA  PHE A  30       7.000   8.000   9.000  1.00  0.00           C
ATOM      4  CA  TRP A  40      10.000  11.000  12.000  1.00  0.00           C
END
"""
    pdb_path = str(tmp_path / "mock_protein.pdb")
    with open(pdb_path, "w") as f:
        f.write(pdb_content)

    feat = DistanceFeaturizer()
    results = feat.featurize_target("mock", pdb_path, records, pocket)
    assert len(results) > 0
    for mol_id, vec in results.items():
        assert vec.dtype == np.float32
        assert vec.ndim == 1
        assert len(vec) == 2 * len(pocket)  # 2 stats per residue


def test_distance_chem_featurizer_synthetic(tmp_path):
    from src.featurizers.distances import DistanceChemFeaturizer
    import tempfile

    records = _make_synthetic_records(2)
    pocket = _make_synthetic_pocket()

    pdb_content = """\
ATOM      1  CA  ALA A  10       1.000   2.000   3.000  1.00  0.00           C
ATOM      2  CA  GLY A  20       4.000   5.000   6.000  1.00  0.00           C
ATOM      3  CA  PHE A  30       7.000   8.000   9.000  1.00  0.00           C
ATOM      4  CA  TRP A  40      10.000  11.000  12.000  1.00  0.00           C
END
"""
    pdb_path = str(tmp_path / "mock_protein.pdb")
    with open(pdb_path, "w") as f:
        f.write(pdb_content)

    feat = DistanceChemFeaturizer()
    results = feat.featurize_target("mock", pdb_path, records, pocket)
    assert len(results) > 0
    M = len(pocket)
    for vec in results.values():
        # 2*M distance stats + 5*M residue properties
        assert len(vec) == 2 * M + 5 * M


# ── Integration tests requiring DUDE-Z data ────────────────────────────────

@pytest.fixture(scope="module")
def dude_available():
    from config import DOCKING_DIR
    if not (DOCKING_DIR / SMOKE_TARGET).exists():
        pytest.skip("DUDE-Z data not mounted")


def test_plec_featurizer_synthetic(tmp_path):
    from src.featurizers.plec import PLECFeaturizer
    from config import PLEC_SIZE

    records = _make_synthetic_records(3)
    pocket = _make_synthetic_pocket()

    # Pocket atoms placed close enough to benzene (origin ≈ 0,0,0) to fire contacts
    pdb_content = """\
ATOM      1  CA  ALA A  10       1.000   2.000   0.000  1.00  0.00           C
ATOM      2  CA  GLY A  20       4.000   0.000   0.000  1.00  0.00           C
ATOM      3  CA  PHE A  30      -1.000   2.000   0.000  1.00  0.00           C
ATOM      4  CA  TRP A  40       0.000  -2.000   0.000  1.00  0.00           C
END
"""
    pdb_path = str(tmp_path / "mock_protein.pdb")
    with open(pdb_path, "w") as f:
        f.write(pdb_content)

    feat = PLECFeaturizer()
    results = feat.featurize_target("mock", pdb_path, records, pocket)
    assert len(results) == len(records)
    for vec in results.values():
        assert vec.shape == (PLEC_SIZE,)
        assert vec.dtype == np.float32


def test_grim_featurizer_requires_plif(tmp_path):
    """GRIM must raise FileNotFoundError when PLIF results are absent."""
    from src.featurizers.grim import GRIMFeaturizer
    feat = GRIMFeaturizer()
    with pytest.raises(FileNotFoundError):
        feat.featurize_target("nonexistent_target", "dummy.pdb", [], [])
