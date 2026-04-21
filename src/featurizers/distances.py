from __future__ import annotations
import logging
import numpy as np
from rdkit.Chem import rdMolTransforms
from src.featurizers.base import BaseFeaturizer
from src.gpu_utils import gpu_batch_pairwise_distances
from config import GPU_BATCH_SIZE, RESIDUE_PROPERTIES

log = logging.getLogger(__name__)


def _get_mol_coords(mol) -> np.ndarray | None:
    """Extract heavy-atom 3D coordinates from RDKit Mol. Returns None if no conformer."""
    if mol is None or mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer()
    coords = conf.GetPositions()  # (N_atoms, 3)
    return coords.astype(np.float32)


def _get_residue_coords(protein, pocket_residues: list[str]) -> np.ndarray:
    """
    Extract alpha-carbon (CA) coordinates for pocket residues.
    Falls back to any heavy atom if CA is not found.
    """
    coords = []
    for res_id in pocket_residues:
        tokens = res_id.split("_")
        resname = tokens[0] if tokens else ""
        resid = int(tokens[1]) if len(tokens) > 1 else -1
        sel = protein.select_atoms(f"resid {resid} and name CA")
        if len(sel) == 0:
            sel = protein.select_atoms(f"resid {resid} and not name H*")
        if len(sel) > 0:
            coords.append(sel.positions.mean(axis=0))
        else:
            log.debug("No atoms found for residue %s", res_id)
    if not coords:
        raise ValueError("No residue coordinates extracted from pocket_residues")
    return np.array(coords, dtype=np.float32)  # (M, 3)


def _pad_ligand_coords(
    batch: list[dict],
) -> tuple[np.ndarray, list[int]]:
    """
    Pad ligand coordinates to max atom count in batch.
    Returns (B, N_max, 3) array and list of actual atom counts.
    """
    all_coords = []
    n_atoms_list = []
    for record in batch:
        coords = _get_mol_coords(record["mol"])
        if coords is None:
            coords = np.zeros((1, 3), dtype=np.float32)
        all_coords.append(coords)
        n_atoms_list.append(len(coords))

    n_max = max(n_atoms_list)
    padded = np.zeros((len(batch), n_max, 3), dtype=np.float32)
    for i, coords in enumerate(all_coords):
        padded[i, :len(coords)] = coords
    return padded, n_atoms_list


def _compute_distance_features(
    dist: np.ndarray,  # (N_atoms, M_residues)
    n_atoms: int,
) -> np.ndarray:
    """
    Compute per-residue distance statistics from a (N_atoms, M) distance matrix.
    Stats per residue: min distance, mean distance.
    Returns 1D float32 vector of length 2*M.
    """
    d = dist[:n_atoms, :]  # unpad
    per_res_min = d.min(axis=0)    # (M,)
    per_res_mean = d.mean(axis=0)  # (M,)
    return np.concatenate([per_res_min, per_res_mean]).astype(np.float32)


class DistanceFeaturizer(BaseFeaturizer):
    """
    GPU-accelerated pairwise ligand-pocket distance featurizer.
    Batches all ligands for a target into a single GPU call per batch.
    """

    @property
    def name(self) -> str:
        return "distance_matrix"

    def featurize_target(
        self,
        target: str,
        protein_path: str,
        ligand_records: list[dict],
        pocket_residues: list[str],
    ) -> dict[str, np.ndarray]:
        """
        Returns: {mol_id: np.ndarray (float32, shape=(2*M,))} where M = len(pocket_residues)
        """
        import MDAnalysis as mda

        if not ligand_records:
            raise ValueError(f"No ligand records provided for target {target}")
        if not pocket_residues:
            raise ValueError(f"No pocket residues for target {target}")

        protein = mda.Universe(protein_path)
        residue_coords = _get_residue_coords(protein, pocket_residues)  # (M, 3)

        results: dict[str, np.ndarray] = {}
        valid_records = [r for r in ligand_records if r.get("mol") is not None
                         and _get_mol_coords(r["mol"]) is not None]

        if not valid_records:
            log.warning("%s: no valid conformers for DistanceFeaturizer", target)
            return results

        for batch_start in range(0, len(valid_records), GPU_BATCH_SIZE):
            batch = valid_records[batch_start : batch_start + GPU_BATCH_SIZE]
            coords_batch, n_atoms_list = _pad_ligand_coords(batch)  # (B, N_max, 3)
            dist_batch = gpu_batch_pairwise_distances(coords_batch, residue_coords)  # (B, N_max, M)

            for i, record in enumerate(batch):
                feats = _compute_distance_features(dist_batch[i], n_atoms_list[i])
                results[record["id"]] = feats

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log.info("%s: DistanceFeaturizer done — %d complexes, dim=%d",
                 target, len(results), next(iter(results.values())).shape[0] if results else 0)
        return results


class DistanceChemFeaturizer(BaseFeaturizer):
    """
    Distance matrix features concatenated with residue physicochemical properties.
    Hybrid of geometry (DistanceFeaturizer) and chemistry (RESIDUE_PROPERTIES).
    """

    @property
    def name(self) -> str:
        return "distance_chemistry"

    def featurize_target(
        self,
        target: str,
        protein_path: str,
        ligand_records: list[dict],
        pocket_residues: list[str],
    ) -> dict[str, np.ndarray]:
        """
        Returns: {mol_id: np.ndarray (float32, shape=(2*M + 5*M,))}
        """
        # Get base distance features
        dist_feat = DistanceFeaturizer()
        dist_results = dist_feat.featurize_target(target, protein_path, ligand_records, pocket_residues)

        # Build residue property vector (5 properties per residue, stacked across M residues)
        res_props = _build_residue_property_vector(pocket_residues)  # (5*M,)

        results: dict[str, np.ndarray] = {}
        for mol_id, dist_vec in dist_results.items():
            results[mol_id] = np.concatenate([dist_vec, res_props]).astype(np.float32)

        log.info("%s: DistanceChemFeaturizer done — %d complexes, dim=%d",
                 target, len(results), next(iter(results.values())).shape[0] if results else 0)
        return results


def _build_residue_property_vector(pocket_residues: list[str]) -> np.ndarray:
    """
    Build a concatenated property vector for all pocket residues.
    Each residue gets a 5-dim vector: [hydrophobic, polar, positive, negative, aromatic].
    Returns (5 * M,) float32 array.
    """
    props = []
    for res_id in pocket_residues:
        resname = res_id.split("_")[0].upper()
        prop = RESIDUE_PROPERTIES.get(resname, RESIDUE_PROPERTIES["UNK"])
        props.append(prop)
    return np.array(props, dtype=np.float32).ravel()
