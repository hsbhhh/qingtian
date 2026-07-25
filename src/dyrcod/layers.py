from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def add_self_loops(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
    fill_value: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = edge_index.device
    loop = torch.arange(num_nodes, device=device)
    loop_index = torch.stack([loop, loop], dim=0)
    loop_weight = torch.full((num_nodes,), fill_value, dtype=edge_weight.dtype, device=device)
    return torch.cat([edge_index, loop_index], dim=1), torch.cat([edge_weight, loop_weight], dim=0)


def normalize_edge_index(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    row, col = edge_index[0], edge_index[1]
    deg = torch.zeros(num_nodes, device=edge_weight.device, dtype=edge_weight.dtype)
    deg.scatter_add_(0, row, edge_weight)
    deg_inv_sqrt = deg.clamp(min=1e-12).pow(-0.5)
    return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def normalize_directional_edge_index(
    edge_index: torch.Tensor,
    directed_weight: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    row, col = edge_index[0], edge_index[1]
    deg_in = torch.zeros(num_nodes, device=directed_weight.device, dtype=directed_weight.dtype)
    deg_out = torch.zeros(num_nodes, device=directed_weight.device, dtype=directed_weight.dtype)
    deg_in.scatter_add_(0, row, directed_weight)
    deg_out.scatter_add_(0, col, directed_weight)
    deg_in_inv_sqrt = deg_in.clamp(min=1e-12).pow(-0.5)
    deg_out_inv_sqrt = deg_out.clamp(min=1e-12).pow(-0.5)
    norm_in = deg_in_inv_sqrt[row] * directed_weight * deg_out_inv_sqrt[col]
    norm_out = deg_out_inv_sqrt[col] * directed_weight * deg_in_inv_sqrt[row]
    return norm_in, norm_out


class SparseGCNLayer(nn.Module):
    """Sparse weighted GCN layer used by each static biological view."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes, fill_value=1.0)
        norm_weight = normalize_edge_index(edge_index, edge_weight, num_nodes)
        h = self.linear(x)
        row, col = edge_index[0], edge_index[1]
        out = torch.zeros_like(h)
        out.index_add_(0, row, h[col] * norm_weight.unsqueeze(-1))
        return out


class DirectionalSparseGCNLayer(nn.Module):
    """Sparse GCN layer with learned directional edge weights."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.in_linear = nn.Linear(in_dim, out_dim, bias=False)
        self.out_linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        directed_weight: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        row, col = edge_index[0], edge_index[1]
        norm_in, norm_out = normalize_directional_edge_index(edge_index, directed_weight, num_nodes)

        m_in = torch.zeros_like(x)
        m_out = torch.zeros_like(x)
        m_in.index_add_(0, row, x[col] * norm_in.unsqueeze(-1))
        m_out.index_add_(0, col, x[row] * norm_out.unsqueeze(-1))

        return self.self_linear(x) + self.in_linear(m_in) + self.out_linear(m_out)


class SingleViewEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        pos_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        current_dim = in_dim + pos_dim
        for layer_id in range(num_layers):
            next_dim = hidden_dim if layer_id < num_layers - 1 else out_dim
            self.layers.append(SparseGCNLayer(current_dim, next_dim))
            self.norms.append(nn.LayerNorm(next_dim))
            current_dim = next_dim

    def forward(
        self,
        x: torch.Tensor,
        pos_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        h = torch.cat([x, pos_feat], dim=1)
        num_nodes = h.size(0)
        for layer, norm in zip(self.layers, self.norms):
            h = layer(h, edge_index, edge_weight, num_nodes)
            h = F.dropout(F.relu(norm(h)), p=self.dropout, training=self.training)
        return h


class SingleViewDirectionalEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        pos_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        current_dim = in_dim + pos_dim
        for layer_id in range(num_layers):
            next_dim = hidden_dim if layer_id < num_layers - 1 else out_dim
            self.layers.append(DirectionalSparseGCNLayer(current_dim, next_dim))
            self.norms.append(nn.LayerNorm(next_dim))
            current_dim = next_dim

    def forward(
        self,
        x: torch.Tensor,
        pos_feat: torch.Tensor,
        edge_index: torch.Tensor,
        directed_weight: torch.Tensor,
    ) -> torch.Tensor:
        h = torch.cat([x, pos_feat], dim=1)
        num_nodes = h.size(0)
        for layer, norm in zip(self.layers, self.norms):
            h = layer(h, edge_index, directed_weight, num_nodes)
            h = F.dropout(F.relu(norm(h)), p=self.dropout, training=self.training)
        return h


class SafeGateFusion(nn.Module):
    """Node-wise gated fusion over biological network views."""

    def __init__(self, in_dim: int, num_items: int, dropout: float = 0.2):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(in_dim * num_items, in_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_items),
        )

    def forward(self, items: List[torch.Tensor], names: List[str]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cat = torch.cat(items, dim=-1)
        raw = torch.sigmoid(self.gate_net(cat))
        gate = raw / raw.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        stacked = torch.stack(items, dim=1)
        fused = torch.sum(stacked * gate.unsqueeze(-1), dim=1)
        stats = {name: gate[:, idx].mean() for idx, name in enumerate(names)}
        return fused, stats
