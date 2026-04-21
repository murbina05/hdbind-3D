from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseFeaturizer(ABC):
    """
    All featurizers must subclass this.
    Each featurizer is responsible for one target at a time.
    Parallelism across targets is handled externally by the runner scripts.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in results paths, e.g. 'plif', 'plec', 'distance_chemistry'."""
        ...

    @abstractmethod
    def featurize_target(
        self,
        target: str,
        protein_path: str,
        ligand_records: list[dict],  # [{"id": str, "mol": Mol, "label": int}, ...]
        pocket_residues: list[str],
    ) -> dict[str, np.ndarray]:
        """
        Compute features for all ligands/decoys of a single target.

        Returns:
            dict mapping complex_id -> feature vector (1D np.ndarray, float32)
        """
        ...

    def output_path(self, results_dir, target: str) -> str:
        from pathlib import Path
        p = Path(results_dir) / self.name / f"{target}.pkl"
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)
