from __future__ import annotations
import logging
import numpy as np
from src.featurizers.base import BaseFeaturizer
from src.gpu_utils import gpu_hd_project, get_device
from config import HD_DIM, RANDOM_SEED, RESULTS_DIR

log = logging.getLogger(__name__)


class GRIMFeaturizer(BaseFeaturizer):
    """
    Hyperdimensional random projection (GRIM) of PLIF vectors to HD_DIM-dimensional space.
    GPU-accelerated via gpu_hd_project (single large matrix multiply).
    Reads PLIF output from Tier 1a results — must run after PLIFFeaturizer.
    """

    def __init__(self) -> None:
        self._projection: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "grim"

    def _get_projection(self, input_dim: int) -> np.ndarray:
        if self._projection is None or self._projection.shape[0] != input_dim:
            rng = np.random.default_rng(RANDOM_SEED)
            proj = rng.standard_normal((input_dim, HD_DIM)).astype(np.float32)
            proj /= np.linalg.norm(proj, axis=0, keepdims=True)
            self._projection = proj
        return self._projection

    def featurize_target(
        self,
        target: str,
        protein_path: str,
        ligand_records: list[dict],
        pocket_residues: list[str],
    ) -> dict[str, np.ndarray]:
        """
        Load pre-computed PLIF vectors for this target and project to HD space.
        Returns: {mol_id: np.ndarray (float32, shape=(HD_DIM,))}
        """
        from src.utils import load_pickle

        plif_path = RESULTS_DIR / "tier1" / "plif" / f"{target}.pkl"
        if not plif_path.exists():
            raise FileNotFoundError(
                f"PLIF output not found for {target}: {plif_path}\n"
                "Run 01a_run_plif.py before 01c_run_grim.py."
            )

        plif_data: dict[str, np.ndarray] = load_pickle(plif_path)
        ids = list(plif_data.keys())

        if not ids:
            raise ValueError(f"PLIF data for {target} is empty")

        X = np.stack([plif_data[i] for i in ids]).astype(np.float32)  # (N, input_dim)
        proj = self._get_projection(X.shape[1])

        proj_path = RESULTS_DIR / "tier1" / "grim" / f"{target}_projection.pkl"
        if not proj_path.exists():
            from src.utils import save_pickle
            save_pickle(proj, proj_path)

        # Single GPU call — (N, input_dim) @ (input_dim, HD_DIM) → (N, HD_DIM)
        hd_vecs = gpu_hd_project(X, proj)

        log.info("%s: GRIM done — %d vectors projected to dim %d", target, len(ids), HD_DIM)
        return {mol_id: hd_vecs[j] for j, mol_id in enumerate(ids)}
