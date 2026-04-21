from __future__ import annotations
import logging
import numpy as np
from src.featurizers.base import BaseFeaturizer

log = logging.getLogger(__name__)


def _pocket_features(pocket_residues: list[str]) -> np.ndarray:
    """
    Fixed-size pocket composition vector from pocket residue list.
    Features: counts of [hydrophobic, polar, positive, negative, aromatic]
    residues + total pocket size. Shape: (6,) — constant per target.
    """
    from config import RESIDUE_PROPERTIES
    props = np.zeros(5, dtype=np.float32)
    for res_id in pocket_residues:
        resname = res_id.split('_')[0]
        props += RESIDUE_PROPERTIES.get(resname, RESIDUE_PROPERTIES['UNK'])
    return np.append(props, len(pocket_residues)).astype(np.float32)  # shape (6,)


def _ligand_features(mol) -> np.ndarray:
    """RDKit physicochemical descriptors for a single ligand. Shape: (9,)"""
    from rdkit.Chem import Descriptors, rdMolDescriptors
    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcFractionCSP3(mol),
    ], dtype=np.float32)


class PocketFeaturizer(BaseFeaturizer):
    """
    Lightweight pocket+ligand featurizer — no DeepChem required.
    Pocket: residue composition counts (6,).
    Ligand: RDKit physicochemical descriptors (9,).
    Combined: (15,) per complex.
    """

    @property
    def name(self) -> str:
        return "pocket_featurizer"

    def featurize_target(
        self,
        target: str,
        protein_path: str,
        ligand_records: list[dict],
        pocket_residues: list[str],
    ) -> dict[str, np.ndarray]:
        """
        Returns: {mol_id: np.ndarray (float32, shape=(15,))}
        """
        if not ligand_records:
            raise ValueError(f"No ligand records provided for target {target}")

        pocket_vec = _pocket_features(pocket_residues)  # (6,)

        results: dict[str, np.ndarray] = {}
        for record in ligand_records:
            mol_id: str = record["id"]
            mol = record.get("mol")
            if mol is None:
                log.warning("%s | %s: mol is None, skipping", target, mol_id)
                continue
            lig_vec = _ligand_features(mol)  # (9,)
            results[mol_id] = np.concatenate([pocket_vec, lig_vec])  # (15,)

        log.info("%s: PocketFeaturizer done — %d complexes, pocket dim=%d",
                 target, len(results), 15)
        return results
