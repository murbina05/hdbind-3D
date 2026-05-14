"""
Vanilla E(n)-equivariant graph neural network (Satorras et al., 2021),
following the formulation in TIER3_EGNN_PLAN.md §2.

Design choices implemented per plan:
  * 4 EGCL layers, hidden dim 128, edge attr dim 3 (intra-prot/intra-lig/cross).
  * Coordinate-update normalization: divide (x_i - x_j) by ||...|| + eps so a
    single far atom cannot dominate (lucidrains' norm_coors=True).
  * tanh-bounded phi_x output to keep per-step coordinate deltas finite —
    the plan §3.3 stability hooks for protein-scale graphs.
  * Ligand-atom-only mean-pool readout -> 2-layer MLP -> 1 logit.

The model is chirality-blind by construction (squared distances + (x_i - x_j)
displacements are reflection-invariant); this is a documented vanilla-EGNN
limitation flagged in plan §2 as the EGMN/GVP upgrade path.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


class EGCL(nn.Module):
    """Equivariant Graph Convolutional Layer (Satorras et al., eqs. 4-6).

        m_ij      = phi_e(h_i, h_j, ||x_i - x_j||^2, a_ij)
        x_i_new   = x_i + sum_{j!=i} (x_i - x_j)/(||...||+eps) * tanh(phi_x(m_ij))
        m_i       = sum_{j in N(i)} m_ij
        h_i_new   = phi_h(h_i, m_i)
    """

    def __init__(
        self,
        h_dim: int,
        edge_attr_dim: int,
        normalize_coords: bool = True,
        coord_tanh: bool = True,
        coord_eps: float = 1e-6,
    ):
        super().__init__()
        self.normalize_coords = normalize_coords
        self.coord_tanh = coord_tanh
        self.coord_eps = coord_eps

        # phi_e: (h_i, h_j, dist^2, edge_attr) -> message of size h_dim
        self.phi_e = nn.Sequential(
            nn.Linear(2 * h_dim + 1 + edge_attr_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
        )
        # phi_x: message -> 1 scalar gate per edge
        self.phi_x = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, 1),
        )
        # phi_h: (h_i, m_i) -> updated h
        self.phi_h = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
        )

        # Init phi_x final layer small so coord updates start near zero.
        nn.init.zeros_(self.phi_x[-1].weight)
        nn.init.zeros_(self.phi_x[-1].bias)

    def forward(
        self,
        h: torch.Tensor,           # (N, h_dim)
        x: torch.Tensor,           # (N, 3)
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: torch.Tensor,   # (E, edge_attr_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        i, j = edge_index[0], edge_index[1]
        diff = x[i] - x[j]                                    # (E, 3)
        d2 = (diff * diff).sum(dim=1, keepdim=True)           # (E, 1)

        m_ij = self.phi_e(torch.cat([h[i], h[j], d2, edge_attr], dim=1))   # (E, h_dim)

        coord_gate = self.phi_x(m_ij)                          # (E, 1)
        if self.coord_tanh:
            coord_gate = torch.tanh(coord_gate)

        if self.normalize_coords:
            diff = diff / (diff.norm(dim=1, keepdim=True) + self.coord_eps)

        coord_msg = diff * coord_gate                          # (E, 3)
        # Aggregate per source node i. Using mean keeps the magnitude
        # independent of node degree (recommended for protein graphs).
        x_update = scatter(coord_msg, i, dim=0, dim_size=x.size(0), reduce="mean")
        x_new = x + x_update

        m_i = scatter(m_ij, i, dim=0, dim_size=h.size(0), reduce="sum")
        h_new = self.phi_h(torch.cat([h, m_i], dim=1))
        return h_new, x_new


class EGNN(nn.Module):
    """4-layer EGNN for protein-ligand pocket graphs (plan §2)."""

    def __init__(
        self,
        in_node_dim: int = 10,
        edge_attr_dim: int = 3,
        h_dim: int = 128,
        n_layers: int = 4,
        head_dim: int | None = None,
    ):
        super().__init__()
        head_dim = head_dim or h_dim
        self.embed = nn.Linear(in_node_dim, h_dim)
        self.layers = nn.ModuleList([
            EGCL(h_dim, edge_attr_dim) for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(h_dim, head_dim),
            nn.SiLU(),
            nn.Linear(head_dim, 1),
        )

    def forward(
        self,
        x_node: torch.Tensor,       # (N, in_node_dim)
        pos: torch.Tensor,          # (N, 3)
        edge_index: torch.Tensor,   # (2, E)
        edge_attr: torch.Tensor,    # (E, edge_attr_dim)
        ligand_mask: torch.Tensor,  # (N,) bool — True for ligand atoms
        batch: torch.Tensor,        # (N,) long — graph index per node
    ) -> torch.Tensor:              # (B,) logits
        h = self.embed(x_node)
        x = pos
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)

        # Ligand-atom-only mean-pool, per graph.
        n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        lig_h = h[ligand_mask]
        lig_b = batch[ligand_mask]
        graph_emb = scatter(lig_h, lig_b, dim=0, dim_size=n_graphs, reduce="mean")
        logits = self.head(graph_emb).squeeze(-1)
        return logits
