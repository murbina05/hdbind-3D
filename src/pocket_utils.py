from __future__ import annotations
"""
Pocket residue identification from crystal ligand proximity.
"""
import json
import numpy as np
import MDAnalysis as mda
from config import POCKET_CUTOFF_ANG, DATA_DIR


def find_pocket_residues(protein: mda.Universe, crystal_ligand_coords: np.ndarray) -> list[str]:
    """
    Return residue identifiers (resname+resid+segid) for protein residues
    with any heavy atom within POCKET_CUTOFF_ANG of any crystal ligand atom.
    """
    protein_heavy = protein.select_atoms("not name H*")
    pocket_resids = set()
    for atom in protein_heavy:
        dists = np.linalg.norm(crystal_ligand_coords - atom.position, axis=1)
        if dists.min() <= POCKET_CUTOFF_ANG:
            pocket_resids.add(f"{atom.resname}_{atom.resid}_{atom.segid}")
    return sorted(pocket_resids)


def save_pocket_residues(target: str, residues: list[str]) -> None:
    out = DATA_DIR / "pocket_residues" / f"{target}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(residues, indent=2))


def load_pocket_residues(target: str) -> list[str]:
    path = DATA_DIR / "pocket_residues" / f"{target}.json"
    return json.loads(path.read_text())
