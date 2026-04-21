from __future__ import annotations
import logging
import struct
import zlib
from collections import deque

import numpy as np
from scipy.spatial.distance import cdist
import MDAnalysis as mda
from rdkit import Chem
from src.featurizers.base import BaseFeaturizer
from config import PLEC_DEPTH_LIGAND, PLEC_DEPTH_PROTEIN, PLEC_SIZE, PLEC_CONTACT_CUTOFF

log = logging.getLogger(__name__)


class PLECFeaturizer(BaseFeaturizer):
    """
    Custom PLEC fingerprint — no ODDT dependency.
    For each close-contact ligand/protein atom pair, expands bond neighborhoods
    in both directions (depth_ligand into ligand, depth_protein into protein),
    hashes chemical environment tuples via RDKit atom invariants, and folds into
    a count vector of size PLEC_SIZE.
    Protein loading reuses the same ProLIF/MDAnalysis pattern as PLIFFeaturizer.
    CPU-only. Parallelism is at the target level in runner scripts.
    """

    @property
    def name(self) -> str:
        return "plec"

    def featurize_target(
        self,
        target: str,
        protein_path: str,
        ligand_records: list[dict],
        pocket_residues: list[str],
    ) -> dict[str, np.ndarray]:
        import prolif as plf

        if not ligand_records:
            raise ValueError(f"No ligand records provided for target {target}")

        u = mda.Universe(protein_path)

        pocket_res_mols: list[Chem.Mol | None] = []
        for res_id in pocket_residues:
            tokens = res_id.split("_")
            resname, resid = tokens[0], tokens[1]
            ag = u.select_atoms(f"resname {resname} and resid {resid}")
            if len(ag) == 0:
                log.warning("%s: residue %s not found in PDB", target, res_id)
                pocket_res_mols.append(None)
                continue
            try:
                prot_mol = plf.Molecule.from_mda(ag)
                pocket_res_mols.append(prot_mol[0])
            except Exception as e:
                log.warning("%s: failed to load residue %s — %s", target, res_id, e)
                pocket_res_mols.append(None)

        results: dict[str, np.ndarray] = {}

        for record in ligand_records:
            mol_id: str = record["id"]
            mol: Chem.Mol = record["mol"]
            if mol is None:
                log.warning("%s | %s: mol is None, skipping", target, mol_id)
                continue
            try:
                vec = _compute_plec_vector(
                    mol, pocket_res_mols,
                    PLEC_DEPTH_LIGAND, PLEC_DEPTH_PROTEIN,
                    PLEC_CONTACT_CUTOFF, PLEC_SIZE,
                )
                results[mol_id] = vec
            except Exception as e:
                log.error("%s | %s: PLEC failed — %s", target, mol_id, e)
                results[mol_id] = np.zeros(PLEC_SIZE, dtype=np.float32)

        log.info("%s: PLEC done — %d complexes", target, len(results))
        return results


def _atom_invariant(atom: Chem.Atom) -> tuple[int, int, int]:
    return (atom.GetAtomicNum(), atom.GetDegree(), int(atom.GetIsAromatic()))


def _bfs_neighbors(mol: Chem.Mol, start_idx: int, max_depth: int) -> list[tuple[Chem.Atom, int]]:
    """BFS from start_idx up to max_depth bonds. Returns [(atom, depth)] including the root at depth 0."""
    visited: set[int] = {start_idx}
    queue: deque[tuple[int, int]] = deque([(start_idx, 0)])
    result: list[tuple[Chem.Atom, int]] = []
    while queue:
        idx, depth = queue.popleft()
        result.append((mol.GetAtomWithIdx(idx), depth))
        if depth < max_depth:
            for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx not in visited:
                    visited.add(nidx)
                    queue.append((nidx, depth + 1))
    return result


def _hash_env(l_inv: tuple[int, int, int], p_inv: tuple[int, int, int], d_l: int, d_p: int, size: int) -> int:
    data = struct.pack(
        "!8B",
        min(l_inv[0], 255), min(l_inv[1], 255), l_inv[2],
        min(p_inv[0], 255), min(p_inv[1], 255), p_inv[2],
        d_l, d_p,
    )
    return zlib.crc32(data) % size


def _compute_plec_vector(
    mol: Chem.Mol,
    pocket_res_mols: list[Chem.Mol | None],
    d_lig: int,
    d_prot: int,
    cutoff: float,
    size: int,
) -> np.ndarray:
    vec = np.zeros(size, dtype=np.float32)
    lig_pos = mol.GetConformer().GetPositions()

    for rdmol in pocket_res_mols:
        if rdmol is None:
            continue
        prot_pos = rdmol.GetConformer().GetPositions()
        dists = cdist(lig_pos, prot_pos)
        contact_pairs = np.argwhere(dists <= cutoff)

        for la_idx, pa_idx in contact_pairs:
            lig_env = _bfs_neighbors(mol, int(la_idx), d_lig)
            prot_env = _bfs_neighbors(rdmol, int(pa_idx), d_prot)
            for l_atom, d_l in lig_env:
                l_inv = _atom_invariant(l_atom)
                for p_atom, d_p in prot_env:
                    p_inv = _atom_invariant(p_atom)
                    vec[_hash_env(l_inv, p_inv, d_l, d_p, size)] += 1.0

    return vec
