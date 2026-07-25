from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicLIRSLoss(nn.Module):
    def __init__(
        self,
        cls_pos_weight: float = 1.0,
        lambda_orth: float = 0.15,
        lambda_edge: float = 0.10,
        lambda_change: float = 0.08,
    ):
        super().__init__()
        self.cls_pos_weight = float(cls_pos_weight)
        self.lambda_orth = float(lambda_orth)
        self.lambda_edge = float(lambda_edge)
        self.lambda_change = float(lambda_change)

    def forward(
        self,
        output_dict: Dict[str, torch.Tensor],
        y: torch.Tensor,
        supervised_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pos_weight = torch.tensor(self.cls_pos_weight, device=y.device)
        target = y[supervised_mask].float()
        logits = output_dict["logits"][supervised_mask]

        sup_loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        orth_loss = output_dict.get("orthogonality_loss", torch.tensor(0.0, device=y.device))
        edge_loss = output_dict.get("dynamic_edge_loss", torch.tensor(0.0, device=y.device))
        change_loss = output_dict.get("dynamic_change_loss", torch.tensor(0.0, device=y.device))

        total = (
            sup_loss
            + self.lambda_orth * orth_loss
            + self.lambda_edge * edge_loss
            + self.lambda_change * change_loss
        )
        return total, {
            "sup": sup_loss.detach(),
            "orth": orth_loss.detach(),
            "edge": edge_loss.detach(),
            "change": change_loss.detach(),
        }
