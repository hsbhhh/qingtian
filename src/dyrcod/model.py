from __future__ import annotations

import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SafeGateFusion, SingleViewDirectionalEncoder, SingleViewEncoder


class PairDirectionalReweighter(nn.Module):
    """Learns magnitude and direction adjustments for each undirected edge pair."""

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        view_names: List[str],
        hidden_dim: int = 64,
        mag_scale: float = 0.15,
        mag_min_scale: float = 0.90,
        mag_max_scale: float = 1.10,
        dir_range: float = 0.20,
        change_budget_target: float = 0.06,
        direction_budget_target: float = 0.06,
        warmup_epochs: int = 20,
    ):
        super().__init__()
        self.view_names = list(view_names)
        self.mag_scale = float(mag_scale)
        self.mag_min_scale = float(mag_min_scale)
        self.mag_max_scale = float(mag_max_scale)
        self.dir_range = float(dir_range)
        self.theta_center = math.pi / 4.0
        self.change_budget_target = float(change_budget_target)
        self.direction_budget_target = float(direction_budget_target)
        self.warmup_epochs = max(int(warmup_epochs), 1)
        self.current_epoch = 1

        self.pos_proj = nn.Sequential(
            nn.Linear(pos_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.view_mag_mlp = nn.ModuleDict(
            {
                view: nn.Sequential(
                    nn.Linear(embed_dim * 3, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, 1),
                )
                for view in self.view_names
            }
        )
        self.view_dir_mlp = nn.ModuleDict(
            {
                view: nn.Sequential(
                    nn.Linear(embed_dim * 3, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, 1),
                )
                for view in self.view_names
            }
        )
        self._pair_cache: Dict[tuple, Dict[str, torch.Tensor]] = {}

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = max(int(epoch), 1)

    def _warm_ratio(self) -> float:
        return min(float(self.current_epoch) / float(self.warmup_epochs), 1.0)

    def _get_pair_cache(self, view: str, edge_index: torch.Tensor, num_nodes: int) -> Dict[str, torch.Tensor]:
        cache_key = (view, int(num_nodes), int(edge_index.size(1)))
        if cache_key in self._pair_cache:
            return {key: value.to(edge_index.device) for key, value in self._pair_cache[cache_key].items()}

        row = edge_index[0].detach().cpu()
        col = edge_index[1].detach().cpu()
        low = torch.minimum(row, col)
        high = torch.maximum(row, col)
        pair_key = low * int(num_nodes) + high
        unique_key, pair_inv = torch.unique(pair_key, sorted=True, return_inverse=True)

        cached = {
            "pair_inv": pair_inv.long(),
            "pair_row": torch.div(unique_key, int(num_nodes), rounding_mode="floor").long(),
            "pair_col": torch.remainder(unique_key, int(num_nodes)).long(),
            "edge_is_canonical": ((row == low) & (col == high)).bool(),
        }
        self._pair_cache[cache_key] = cached
        return {key: value.to(edge_index.device) for key, value in cached.items()}

    @staticmethod
    def _pair_targets(
        pair_row: torch.Tensor,
        pair_col: torch.Tensor,
        y: torch.Tensor,
        supervised_mask: torch.Tensor,
    ):
        pair_mask = supervised_mask[pair_row] & supervised_mask[pair_col]
        if pair_mask.sum() == 0:
            return None
        y_row = y[pair_row[pair_mask]].float()
        y_col = y[pair_col[pair_mask]].float()
        pospos = (y_row > 0.5) & (y_col > 0.5)
        posneg = y_row != y_col
        return pair_mask, pospos, posneg

    def forward(
        self,
        base_embedding: torch.Tensor,
        pos_feat: torch.Tensor,
        edge_index_dict: Dict[str, torch.Tensor],
        edge_weight_dict: Dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        supervised_mask: Optional[torch.Tensor] = None,
        return_edges: bool = False,
    ):
        _ = self.pos_proj(pos_feat)
        base = F.layer_norm(base_embedding, (base_embedding.size(-1),))

        directed_weight_dict: Dict[str, torch.Tensor] = {}
        dynamic_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        edge_exports: Dict[str, Dict[str, torch.Tensor]] = {}
        edge_losses = []
        mag_budget_losses = []
        dir_budget_losses = []
        warm = self._warm_ratio()

        for view in self.view_names:
            edge_index = edge_index_dict[view]
            edge_weight = edge_weight_dict[view]
            cache = self._get_pair_cache(view, edge_index, num_nodes=base.size(0))
            pair_inv = cache["pair_inv"]
            pair_row = cache["pair_row"]
            pair_col = cache["pair_col"]
            edge_is_canonical = cache["edge_is_canonical"]

            num_pairs = int(pair_row.numel())
            pair_weight_sum = torch.zeros(num_pairs, device=edge_weight.device, dtype=edge_weight.dtype)
            pair_count = torch.zeros(num_pairs, device=edge_weight.device, dtype=edge_weight.dtype)
            pair_weight_sum.index_add_(0, pair_inv, edge_weight)
            pair_count.index_add_(0, pair_inv, torch.ones_like(edge_weight))
            pair_weight = pair_weight_sum / pair_count.clamp(min=1.0)

            za = base[pair_row]
            zb = base[pair_col]
            feat = torch.cat([za, zb, (za - zb).abs()], dim=-1)

            raw_mag = self.view_mag_mlp[view](feat).squeeze(-1)
            mag = 1.0 + warm * self.mag_scale * torch.tanh(raw_mag)
            mag = mag.clamp(min=self.mag_min_scale, max=self.mag_max_scale)

            raw_dir = self.view_dir_mlp[view](feat).squeeze(-1)
            theta = self.theta_center + warm * self.dir_range * torch.tanh(raw_dir)
            theta = theta.clamp(
                min=self.theta_center - self.dir_range,
                max=self.theta_center + self.dir_range,
            )

            pair_recv_from_high = pair_weight * mag * torch.cos(theta)
            pair_recv_from_low = pair_weight * mag * torch.sin(theta)

            directed_weight = torch.empty_like(edge_weight)
            directed_weight[edge_is_canonical] = pair_recv_from_high[pair_inv[edge_is_canonical]]
            directed_weight[~edge_is_canonical] = pair_recv_from_low[pair_inv[~edge_is_canonical]]
            directed_weight_dict[view] = directed_weight

            if return_edges:
                edge_exports[view] = {
                    "edge_index_directed": edge_index,
                    "edge_weight_directed": directed_weight,
                    "edge_weight_original": edge_weight,
                    "pair_edge_index": torch.stack([pair_row, pair_col], dim=0),
                    "pair_original_weight": pair_weight,
                    "theta": theta,
                    "w_pair_row_to_col": pair_recv_from_high,
                    "w_pair_col_to_row": pair_recv_from_low,
                }

            mag_budget_losses.append((torch.abs(mag - 1.0).mean() - self.change_budget_target).abs())
            dir_budget_losses.append((torch.abs(theta - self.theta_center).mean() - self.direction_budget_target).abs())

            if y is not None and supervised_mask is not None:
                pair_target = self._pair_targets(pair_row, pair_col, y, supervised_mask)
                if pair_target is not None:
                    pair_mask, pospos, posneg = pair_target
                    prob_keep = torch.sigmoid(raw_mag[pair_mask])
                    cur_losses = []
                    if pospos.any():
                        cur_losses.append(F.binary_cross_entropy(prob_keep[pospos], torch.full_like(prob_keep[pospos], 0.80)))
                    if posneg.any():
                        cur_losses.append(F.binary_cross_entropy(prob_keep[posneg], torch.full_like(prob_keep[posneg], 0.20)))
                    if cur_losses:
                        edge_losses.append(torch.stack(cur_losses).mean())

            cos_v = torch.cos(theta)
            sin_v = torch.sin(theta)
            dynamic_stats[view] = {
                "mean_mag": mag.mean(),
                "mean_theta_deg": theta.mean() * 180.0 / math.pi,
                "mean_dir_dev_deg": torch.abs(theta - self.theta_center).mean() * 180.0 / math.pi,
                "mean_pair_gap": torch.abs(cos_v - sin_v).mean(),
                "cos_mean": cos_v.mean(),
                "sin_mean": sin_v.mean(),
                "high_to_low_ratio": (cos_v > sin_v).float().mean(),
            }

        edge_loss = torch.stack(edge_losses).mean() if edge_losses else base_embedding.new_tensor(0.0)
        change_loss = (
            0.5 * torch.stack(mag_budget_losses).mean()
            + 0.5 * torch.stack(dir_budget_losses).mean()
            if mag_budget_losses
            else base_embedding.new_tensor(0.0)
        )

        if return_edges:
            return directed_weight_dict, dynamic_stats, edge_loss, change_loss, edge_exports
        return directed_weight_dict, dynamic_stats, edge_loss, change_loss


class OrthogonalRoleCommunityProjector(nn.Module):
    """Splits dynamic embeddings into role and community subspaces."""

    def __init__(self, embed_dim: int, role_dim: int, dropout: float = 0.2):
        super().__init__()
        self.role_proj = nn.Linear(embed_dim, role_dim, bias=False)
        self.comm_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.role_seed = nn.Sequential(
            nn.Linear(embed_dim * 2, role_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(role_dim),
        )
        self.comm_norm = nn.LayerNorm(embed_dim)

    def split(self, z: torch.Tensor, seed_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        role = self.role_proj(z) + 0.3 * self.role_seed(torch.cat([z, seed_feat], dim=-1))
        comm = self.comm_norm(self.comm_proj(z))
        return role, comm

    def orth_penalty(self) -> torch.Tensor:
        return ((self.role_proj.weight @ self.comm_proj.weight.t()) ** 2).mean()


class DynamicLIRSDriverGeneModel(nn.Module):
    """DyRCoD model for dynamic multi-view cancer driver gene prediction."""

    def __init__(
        self,
        in_dim: int,
        pos_dim: int,
        hidden_dim: int,
        embed_dim: int,
        view_names: List[str],
        num_layers: int = 2,
        dropout: float = 0.2,
        dynamic_hidden_dim: int = 64,
        change_budget_target: float = 0.06,
        warmup_epochs: int = 20,
        role_dim: int = 48,
        mag_scale: float = 0.15,
        mag_min_scale: float = 0.90,
        mag_max_scale: float = 1.10,
        dir_range: float = 0.20,
        direction_budget_target: float = 0.06,
        disable_projection_removal: bool = False,
        disable_role_community_decoupling: bool = False,
    ):
        super().__init__()
        self.view_names = list(view_names)
        self.disable_projection_removal = bool(disable_projection_removal)
        self.disable_role_community_decoupling = bool(disable_role_community_decoupling)

        self.static_view_encoders = nn.ModuleDict(
            {
                view: SingleViewEncoder(in_dim, pos_dim, hidden_dim, embed_dim, num_layers=num_layers, dropout=dropout)
                for view in self.view_names
            }
        )
        self.dynamic_view_encoders = nn.ModuleDict(
            {
                view: SingleViewDirectionalEncoder(
                    in_dim,
                    pos_dim,
                    hidden_dim,
                    embed_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                )
                for view in self.view_names
            }
        )

        self.static_fusion = SafeGateFusion(embed_dim, len(self.view_names), dropout=dropout)
        self.dynamic_fusion = SafeGateFusion(embed_dim, len(self.view_names), dropout=dropout)
        self.dynamic_edge_module = PairDirectionalReweighter(
            embed_dim=embed_dim,
            pos_dim=pos_dim,
            view_names=view_names,
            hidden_dim=dynamic_hidden_dim,
            mag_scale=mag_scale,
            mag_min_scale=mag_min_scale,
            mag_max_scale=mag_max_scale,
            dir_range=dir_range,
            change_budget_target=change_budget_target,
            direction_budget_target=direction_budget_target,
            warmup_epochs=warmup_epochs,
        )

        self.pos_seed_encoder = nn.Sequential(
            nn.Linear(pos_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )
        self.subspace_projector = OrthogonalRoleCommunityProjector(
            embed_dim=embed_dim,
            role_dim=role_dim,
            dropout=dropout,
        )
        self.role_to_comm = nn.Linear(role_dim, embed_dim, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        self.remove_strength = nn.Parameter(torch.tensor(0.30, dtype=torch.float32))

    def set_epoch(self, epoch: int) -> None:
        self.dynamic_edge_module.set_epoch(epoch)

    def _remove_role_projection(self, z_comm: torch.Tensor, z_role: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        role_in_comm = F.normalize(self.role_to_comm(z_role), dim=-1)
        proj_coeff = torch.sum(z_comm * role_in_comm, dim=-1, keepdim=True)
        proj = proj_coeff * role_in_comm
        strength = torch.sigmoid(self.remove_strength)
        if self.disable_projection_removal:
            return z_comm, strength.new_tensor(0.0)
        return z_comm - strength * proj, strength

    def forward(
        self,
        x: torch.Tensor,
        pos_feat: torch.Tensor,
        edge_index_dict: Dict[str, torch.Tensor],
        edge_weight_dict: Dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        supervised_mask: Optional[torch.Tensor] = None,
        return_edges: bool = False,
    ) -> dict[str, Union[torch.Tensor, dict]]:
        static_view_embeddings = {
            view: self.static_view_encoders[view](x, pos_feat, edge_index_dict[view], edge_weight_dict[view])
            for view in self.view_names
        }
        z_anchor, static_stats = self.static_fusion(
            [static_view_embeddings[view] for view in self.view_names],
            self.view_names,
        )

        dynamic_edge_result = self.dynamic_edge_module(
            base_embedding=z_anchor,
            pos_feat=pos_feat,
            edge_index_dict=edge_index_dict,
            edge_weight_dict=edge_weight_dict,
            y=y,
            supervised_mask=supervised_mask,
            return_edges=return_edges,
        )
        if return_edges:
            directed_weight_dict, dynamic_scale_stats, edge_loss, change_loss, edge_exports = dynamic_edge_result
        else:
            directed_weight_dict, dynamic_scale_stats, edge_loss, change_loss = dynamic_edge_result
            edge_exports = None

        dynamic_view_embeddings = {
            view: self.dynamic_view_encoders[view](
                x=x,
                pos_feat=pos_feat,
                edge_index=edge_index_dict[view],
                directed_weight=directed_weight_dict[view],
            )
            for view in self.view_names
        }
        z_dynamic, dynamic_stats = self.dynamic_fusion(
            [dynamic_view_embeddings[view] for view in self.view_names],
            self.view_names,
        )

        pos_seed = self.pos_seed_encoder(pos_feat)
        if self.disable_role_community_decoupling:
            role_embedding = z_dynamic.new_zeros(z_dynamic.size(0), self.subspace_projector.role_proj.out_features)
            comm_embedding = z_dynamic
        else:
            role_embedding, comm_embedding = self.subspace_projector.split(z_dynamic, pos_seed)

        comm_clean, remove_strength = self._remove_role_projection(comm_embedding, role_embedding)
        logits = self.classifier(comm_clean).squeeze(-1)

        if self.disable_role_community_decoupling:
            orthogonality_loss = logits.new_tensor(0.0)
        else:
            orthogonality_loss = self.subspace_projector.orth_penalty() + (
                F.cosine_similarity(
                    F.normalize(self.role_to_comm(role_embedding), dim=-1),
                    F.normalize(comm_embedding, dim=-1),
                    dim=-1,
                )
                ** 2
            ).mean()

        output = {
            "logits": logits,
            "prob": torch.sigmoid(logits),
            "embedding": comm_clean,
            "classifier_input_embedding": comm_clean,
            "role_embedding": role_embedding,
            "role_projected_embedding": self.role_to_comm(role_embedding),
            "community_embedding": comm_embedding,
            "clean_embedding": comm_clean,
            "remove_strength": remove_strength,
            "static_fusion_stats": static_stats,
            "dynamic_fusion_stats": dynamic_stats,
            "dynamic_scale_stats": dynamic_scale_stats,
            "dynamic_edge_loss": edge_loss,
            "dynamic_change_loss": change_loss,
            "orthogonality_loss": orthogonality_loss,
        }
        if return_edges and edge_exports is not None:
            output.update(
                {
                    "directed_edge_index_dict": {
                        view: edge_exports[view]["edge_index_directed"] for view in self.view_names
                    },
                    "directed_edge_weight_dict": {
                        view: edge_exports[view]["edge_weight_directed"] for view in self.view_names
                    },
                    "original_edge_weight_dict": {
                        view: edge_exports[view]["edge_weight_original"] for view in self.view_names
                    },
                    "dynamic_edge_export_dict": edge_exports,
                }
            )
        return output

    def forward_with_edges(self, *args, **kwargs):
        kwargs["return_edges"] = True
        return self.forward(*args, **kwargs)

    def get_dynamic_edges(self, *args, **kwargs):
        return self.forward_with_edges(*args, **kwargs)["dynamic_edge_export_dict"]
