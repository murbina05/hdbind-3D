from __future__ import annotations
import logging
import torch
import numpy as np

log = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device() -> torch.device:
    return DEVICE


def to_tensor(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32)).to(DEVICE)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def gpu_pairwise_distances(
    ligand_coords: np.ndarray,   # (N_atoms, 3)
    residue_coords: np.ndarray,  # (M_atoms, 3)
) -> np.ndarray:
    """
    Compute pairwise Euclidean distances on GPU.
    Falls back to scipy on CPU if CUDA unavailable.
    Returns (N_atoms, M_atoms) float32 array on CPU.
    """
    if not torch.cuda.is_available():
        from scipy.spatial.distance import cdist
        return cdist(ligand_coords, residue_coords).astype(np.float32)
    lig_t = to_tensor(ligand_coords)    # (N, 3)
    res_t = to_tensor(residue_coords)   # (M, 3)
    dist_t = torch.cdist(lig_t.unsqueeze(0), res_t.unsqueeze(0)).squeeze(0)  # (N, M)
    return to_numpy(dist_t)


def gpu_batch_pairwise_distances(
    ligand_batch: np.ndarray,    # (B, N_atoms, 3)  — padded
    residue_coords: np.ndarray,  # (M_atoms, 3)
) -> np.ndarray:
    """
    Batched distance computation for B ligands at once.
    Returns (B, N_atoms, M_atoms) float32.
    """
    if not torch.cuda.is_available():
        from scipy.spatial.distance import cdist
        return np.stack([cdist(lig, residue_coords) for lig in ligand_batch]).astype(np.float32)
    lig_t = to_tensor(ligand_batch)     # (B, N, 3)
    res_t = to_tensor(residue_coords)   # (M, 3)
    res_t = res_t.unsqueeze(0).expand(lig_t.shape[0], -1, -1)  # (B, M, 3)
    dist_t = torch.cdist(lig_t, res_t)  # (B, N, M)
    return to_numpy(dist_t)


def gpu_hd_project(
    vectors: np.ndarray,             # (N, input_dim)
    projection_matrix: np.ndarray,   # (input_dim, hd_dim)
) -> np.ndarray:
    """
    Hyperdimensional random projection on GPU.
    Returns (N, hd_dim) float32.
    """
    if not torch.cuda.is_available():
        return (vectors @ projection_matrix).astype(np.float32)
    v_t = to_tensor(vectors)
    p_t = to_tensor(projection_matrix)
    result = torch.mm(v_t, p_t)
    return to_numpy(result)


def log_gpu_status() -> None:
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_properties(0)
        used = torch.cuda.memory_allocated(0) / 1e9
        total = dev.total_memory / 1e9
        log.info("GPU: %s | VRAM used: %.1f / %.1f GB", dev.name, used, total)
    else:
        log.warning("CUDA not available — running on CPU")
