from __future__ import annotations
import logging
import numpy as np
import MDAnalysis as mda
from rdkit import Chem
from src.featurizers.base import BaseFeaturizer
from config import PROLIF_INTERACTIONS

log = logging.getLogger(__name__)


class PLIFFeaturizer(BaseFeaturizer):
    """
    ProLIF 2.x PLIF featurizer.
    Output shape: (n_pocket_residues × n_interactions,) — one interaction block per residue.
    CPU-only. Parallelism is at the target level in runner scripts.
    """

    @property
    def name(self) -> str:
        return "plif"

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

        n_interactions = len(PROLIF_INTERACTIONS)
        zero_block = np.zeros(n_interactions, dtype=np.float32)

        # Each from_mda call modifies the Universe's bond topology in-place, corrupting
        # bond inference for subsequent aromatic residues (HIE, PHE, TRP…). Fix: load
        # each pocket residue from a fresh Universe so no state leaks between residues.
        fp = plf.Fingerprint(interactions=PROLIF_INTERACTIONS, count=False)

        pocket_residue_mols: list = []
        for res_id in pocket_residues:
            tokens = res_id.split("_")
            resname, resid = tokens[0], tokens[1]
            u = mda.Universe(protein_path)   # fresh Universe per residue
            ag = u.select_atoms(f"resname {resname} and resid {resid}")
            if len(ag) == 0:
                log.warning("%s: residue %s not found in PDB", target, res_id)
                pocket_residue_mols.append(None)
                continue
            try:
                prot_mol = plf.Molecule.from_mda(ag)
                pocket_residue_mols.append(prot_mol[0])
            except Exception as e:
                log.warning("%s: failed to load residue %s — %s", target, res_id, e)
                pocket_residue_mols.append(None)

        results: dict[str, np.ndarray] = {}

        for record in ligand_records:
            mol_id: str = record["id"]
            mol: Chem.Mol = record["mol"]
            if mol is None:
                continue

            try:
                lig_mol = plf.Molecule.from_rdkit(mol)
                lig_res = lig_mol[0]
            except Exception as e:
                log.error("%s | %s: ligand conversion failed — %s", target, mol_id, e)
                results[mol_id] = np.zeros(len(pocket_residues) * n_interactions, dtype=np.float32)
                continue

            blocks = []
            for prot_res in pocket_residue_mols:
                if prot_res is None:
                    blocks.append(zero_block)
                    continue
                try:
                    bv = fp.bitvector(lig_res, prot_res)
                    blocks.append(bv.astype(np.float32))
                except Exception:
                    blocks.append(zero_block)

            results[mol_id] = np.concatenate(blocks)

        if results:
            dim = next(iter(results.values())).shape[0]
            log.info("%s: PLIF done — %d complexes, dim=%d", target, len(results), dim)
        return results
