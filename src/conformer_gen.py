from __future__ import annotations
"""
RDKit 3D conformer generation. Parallelized across ligands within a target.
CPU-only: RDKit GPU build is not yet stable; MMFF minimization runs on CPU threads.
If a GPU-accelerated geometry optimization library becomes available
(e.g., torch-geometric force fields), add it behind a GPU_AVAILABLE guard here.
"""
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from rdkit import Chem
from rdkit.Chem import AllChem
from config import NESTED_WORKERS, RANDOM_SEED

log = logging.getLogger(__name__)


def generate_conformer(smiles: str, mol_id: str) -> tuple[str, Chem.Mol | None]:
    """
    Generate a single MMFF-optimized 3D conformer.
    Returns (mol_id, mol) or (mol_id, None) on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return mol_id, None
    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, randomSeed=RANDOM_SEED, maxAttempts=200)
    if result != 0:
        return mol_id, None
    ff_result = AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    if ff_result not in (0, 1):
        return mol_id, None
    mol = Chem.RemoveHs(mol)
    mol.SetProp("_Name", mol_id)
    return mol_id, mol


def generate_conformers_batch(
    smiles_list: list[tuple[str, str]],  # [(mol_id, smiles), ...]
    n_workers: int = NESTED_WORKERS,
) -> dict[str, Chem.Mol]:
    """
    Generate conformers in parallel. Returns {mol_id: mol} for successes.
    """
    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(generate_conformer, smi, mid): mid for mid, smi in smiles_list}
        for future in as_completed(futures):
            mol_id, mol = future.result()
            if mol is not None:
                results[mol_id] = mol
            else:
                log.warning("Conformer generation failed: %s", mol_id)
    return results
