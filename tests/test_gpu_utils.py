"""Smoke tests for src/gpu_utils.py."""
from __future__ import annotations
import numpy as np
import pytest
from src.gpu_utils import (
    gpu_pairwise_distances,
    gpu_batch_pairwise_distances,
    gpu_hd_project,
    get_device,
    log_gpu_status,
)


def test_get_device_returns_device():
    import torch
    dev = get_device()
    assert isinstance(dev, torch.device)


def test_gpu_pairwise_distances_shape():
    lig = np.random.rand(5, 3).astype(np.float32)
    res = np.random.rand(8, 3).astype(np.float32)
    dist = gpu_pairwise_distances(lig, res)
    assert dist.shape == (5, 8)
    assert dist.dtype == np.float32


def test_gpu_pairwise_distances_matches_scipy():
    from scipy.spatial.distance import cdist
    lig = np.random.rand(4, 3).astype(np.float32)
    res = np.random.rand(6, 3).astype(np.float32)
    dist_gpu = gpu_pairwise_distances(lig, res)
    dist_cpu = cdist(lig, res).astype(np.float32)
    np.testing.assert_allclose(dist_gpu, dist_cpu, atol=1e-4)


def test_gpu_batch_pairwise_distances_shape():
    batch = np.random.rand(3, 5, 3).astype(np.float32)
    res = np.random.rand(8, 3).astype(np.float32)
    dist = gpu_batch_pairwise_distances(batch, res)
    assert dist.shape == (3, 5, 8)


def test_gpu_hd_project_shape():
    N, input_dim, hd_dim = 10, 100, 1000
    vectors = np.random.rand(N, input_dim).astype(np.float32)
    proj = np.random.rand(input_dim, hd_dim).astype(np.float32)
    result = gpu_hd_project(vectors, proj)
    assert result.shape == (N, hd_dim)
    assert result.dtype == np.float32


def test_log_gpu_status_no_crash():
    log_gpu_status()
