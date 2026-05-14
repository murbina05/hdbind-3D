from __future__ import annotations
"""
All I/O for DUDE-Z structures and docked poses.
Returns clean, validated Python objects — no downstream parsing elsewhere.

Format asymmetry across decoy sources
─────────────────────────────────────
The four `--decoy-source` branches consumed by `scripts/06_build_egnn_dataset.py`
do NOT share a single on-disk format. Loaders differ by source:

  dudez / goldilocks / extrema  →  multi-record Tripos mol2 under
      DOCKING_DIR/<TARGET>/<subdir>/<prefix>_{ligand,decoy}_poses.mol2
      (parsed by `_parse_mol2` / `iter_poses_from`). Coordinates are already
      in the DUD-Z `rec.crg.pdb` receptor frame.

  ad                             →  one SDF per target at
      /home/maurbina/datasets/AD_decoys/transformed/<TARGET>/<target>_AD_aligned.sdf
      (parsed by `iter_ad_poses` via `Chem.SDMolSupplier`). Coordinates were
      originally in Chen et al.'s DUD-E receptor frame; Phase 3 of
      TIER3_DECOY_BIAS_PLAN.md (`scripts/16_align_ad_decoys.py`) applied a
      per-target Cα Kabsch transform onto the DUD-Z `rec.crg.pdb` frame, so
      the SDF coords are directly compatible with `load_protein(target)`.
      AD actives (label=1) are NOT used; AD only supplies decoys (off-target
      actives docked into a different DUD-Z target = label-0 negatives in this
      benchmark). Actives for the AD branch come from the DUD-Z mol2, same as
      every other source.

If you add a new decoy source, prefer the SDF/RDKit path — Tripos mol2 in this
codebase has historical landmines (NO_LONG_NAME collision in name parsing,
MDAnalysis bond-topology corruption on Universe reuse — see CLAUDE.md).
"""
from pathlib import Path
from typing import Iterator
import logging
import MDAnalysis as mda
from rdkit import Chem
from config import DOCKING_DIR, MOL2_SUBDIR, MOL2_PREFIX

# Phase 3 output root (TIER3_DECOY_BIAS_PLAN). Each target has one SDF here:
#   <AD_TRANSFORMED_ROOT>/<TARGET>/<target>_AD_aligned.sdf
AD_TRANSFORMED_ROOT = Path("/home/maurbina/datasets/AD_decoys/transformed")

# ABL1's apo (Chen) vs holo (DUD-Z) crystal structures diverge by 9.23 Å Cα
# RMSD post-Kabsch — the alignment is a conformational mismatch, not a frame
# mismatch. Excluded from the canonical 40-target AD list. See
# /home/maurbina/datasets/AD_decoys/transformed/CAVEATS.md.
AD_EXCLUDED_TARGETS = frozenset({"ABL1"})

_log = logging.getLogger(__name__)


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


def iter_ad_poses(target: str) -> Iterator[tuple[str, Chem.Mol]]:
    """
    Yield (mol_id, mol) for AD off-target-active decoys docked into `target`.

    Source: SDF written by `scripts/16_align_ad_decoys.py` after Cα Kabsch
    transform onto the DUD-Z receptor frame (TIER3_DECOY_BIAS_PLAN Phase 3).

    Schema-compatible with `iter_poses(target, kind='decoy')`: same (str, Mol)
    pair, mol has a single 3D conformer, hydrogens stripped, label-0 by
    convention. Mol `_Name` carries the upstream CHEMBL-id assigned by Chen
    et al.

    Caveats:
      - ABL1 is excluded (apo vs holo conformational mismatch — see
        CAVEATS.md). Calling with target='ABL1' raises ValueError. Pass
        through `AD_EXCLUDED_TARGETS` to filter callers.
    """
    if target.upper() in AD_EXCLUDED_TARGETS:
        raise ValueError(
            f"AD target {target!r} is excluded (see "
            f"{AD_TRANSFORMED_ROOT}/CAVEATS.md)"
        )
    sdf = AD_TRANSFORMED_ROOT / target.upper() / f"{target.lower()}_AD_aligned.sdf"
    if not sdf.exists():
        raise FileNotFoundError(f"AD aligned SDF not found: {sdf}")

    suppl = Chem.SDMolSupplier(str(sdf), removeHs=True, sanitize=False)
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"AD_{target}_{i}"
        yield mol_id, mol


def load_labels(target: str) -> tuple[list[str], list[int]]:
    """Return (ids, labels) for all actives (1) and decoys (0) of a target."""
    ids, labels = [], []
    for mol_id, _ in iter_poses(target, "ligand"):
        ids.append(mol_id); labels.append(1)
    for mol_id, _ in iter_poses(target, "decoy"):
        ids.append(mol_id); labels.append(0)
    return ids, labels
