from __future__ import annotations
"""
All I/O for DUDE-Z structures and docked poses.
Returns clean, validated Python objects — no downstream parsing elsewhere.
"""
from pathlib import Path
from typing import Iterator
import MDAnalysis as mda
from rdkit import Chem
from config import DOCKING_DIR, MOL2_SUBDIR, MOL2_PREFIX


def load_protein(target: str) -> mda.Universe:
    """Load rec.crg.pdb for a target. Raises FileNotFoundError if missing."""
    pdb = DOCKING_DIR / target / "rec.crg.pdb"
    if not pdb.exists():
        raise FileNotFoundError(f"Protein not found: {pdb}")
    return mda.Universe(str(pdb))


def load_crystal_ligand(target: str) -> Chem.Mol:
    """Load xtal-lig.pdb and return as RDKit Mol. Returns None if parse fails."""
    pdb = DOCKING_DIR / target / "xtal-lig.pdb"
    mol = Chem.MolFromPDBFile(str(pdb), removeHs=True, sanitize=True)
    return mol


def _parse_mol2(path: Path) -> Iterator[tuple[str, Chem.Mol]]:
    """
    Parse a multi-molecule DUDE-Z mol2 file.
    Yields (mol_id, mol) for each successfully parsed molecule.
    mol_id is taken from the '########## Name:' comment header.
    """
    current_block: list[str] = []
    current_id: str | None = None
    in_block = False

    def _emit(mol_id, lines):
        block = "".join(lines)
        mol = Chem.MolFromMol2Block(block, removeHs=True, sanitize=False)
        if mol is None:
            return
        try:
            Chem.SanitizeMol(mol)
            yield mol_id, mol
        except Exception:
            return

    with open(path) as f:
        for line in f:
            if line.startswith("##########"):
                if in_block and current_block:
                    yield from _emit(current_id, current_block)
                    current_block = []
                    in_block = False
                if "Name:" in line and "Long Name:" not in line:
                    current_id = line.split("Name:")[-1].strip()
            elif line.startswith("@<TRIPOS>MOLECULE"):
                in_block = True
                current_block.append(line)
            elif in_block:
                current_block.append(line)

    if in_block and current_block:
        yield from _emit(current_id, current_block)


def iter_poses(
    target: str,
    kind: str,
    prefix: str = MOL2_PREFIX,
) -> Iterator[tuple[str, Chem.Mol]]:
    """
    Yield (mol_id, mol) from a docked-pose mol2 file.
    kind: 'ligand' (actives) | 'decoy'
    prefix: mol2 filename prefix, default MOL2_PREFIX ('dudez_1pt0LD')
    """
    mol2 = DOCKING_DIR / target / MOL2_SUBDIR / f"{prefix}_{kind}_poses.mol2"
    if not mol2.exists():
        raise FileNotFoundError(f"Pose file not found: {mol2}")
    yield from _parse_mol2(mol2)


def iter_poses_from(
    target: str,
    kind: str,
    subdir: str,
    prefix: str,
) -> Iterator[tuple[str, Chem.Mol]]:
    """Like iter_poses but with explicit subdir/prefix — used for alt decoy sets."""
    mol2 = DOCKING_DIR / target / subdir / f"{prefix}_{kind}_poses.mol2"
    if not mol2.exists():
        raise FileNotFoundError(f"Pose file not found: {mol2}")
    yield from _parse_mol2(mol2)


def load_ligand_records(target: str) -> list[dict]:
    """Return [{id, mol, label}] for all actives (1) and decoys (0)."""
    records = []
    for mol_id, mol in iter_poses(target, "ligand"):
        records.append({"id": mol_id, "mol": mol, "label": 1})
    for mol_id, mol in iter_poses(target, "decoy"):
        records.append({"id": mol_id, "mol": mol, "label": 0})
    return records


def load_labels(target: str) -> tuple[list[str], list[int]]:
    """Return (ids, labels) for all actives (1) and decoys (0) of a target."""
    ids, labels = [], []
    for mol_id, _ in iter_poses(target, "ligand"):
        ids.append(mol_id); labels.append(1)
    for mol_id, _ in iter_poses(target, "decoy"):
        ids.append(mol_id); labels.append(0)
    return ids, labels
