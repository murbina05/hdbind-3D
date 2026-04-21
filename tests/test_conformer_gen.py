"""Smoke tests for src/conformer_gen.py."""
from __future__ import annotations
import pytest
from rdkit import Chem
from src.conformer_gen import generate_conformer, generate_conformers_batch


CAFFEINE_SMILES = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"
BAD_SMILES = "not_a_smiles"


def test_generate_conformer_success():
    mol_id, mol = generate_conformer(CAFFEINE_SMILES, "caffeine")
    assert mol_id == "caffeine"
    assert mol is not None
    assert isinstance(mol, Chem.Mol)
    assert mol.GetNumConformers() > 0


def test_generate_conformer_bad_smiles():
    mol_id, mol = generate_conformer(BAD_SMILES, "bad")
    assert mol_id == "bad"
    assert mol is None


def test_generate_conformers_batch():
    smiles_list = [
        ("caffeine", CAFFEINE_SMILES),
        ("bad", BAD_SMILES),
        ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ]
    results = generate_conformers_batch(smiles_list, n_workers=2)
    assert "caffeine" in results
    assert "aspirin" in results
    assert "bad" not in results


def test_conformer_has_3d_coords():
    _, mol = generate_conformer(CAFFEINE_SMILES, "caffeine")
    assert mol is not None
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    assert pos.shape[1] == 3
    assert pos.shape[0] == mol.GetNumAtoms()
