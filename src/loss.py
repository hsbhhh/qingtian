# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DriverGeneLoss(nn.Module):
    def __init__(
        self,
        cls_pos_weight=8.0,
        lambda_cls=1.0,
        lambda_proto=0.3,
        lambda_hard_neg=0.4,
        lambda_align=0.15,
        lambda_mixup=0.15,
        proto_margin=1.0,
        hard_neg_margin=0.5
    ):
        super().__init__()
        self.cls_pos_weight = cls_pos_weight

        self.lambda_cls = lambda_cls
        self.lambda_proto = lambda_proto
        self.lambda_hard_neg = lambda_hard_neg
        self.lambda_align = lambda_align
        self.lambda_mixup = lambda_mixup

        self.proto_margin = proto_margin
        self.hard_neg_margin = hard_neg_margin

    def classification_loss(self, logits, y, supervised_mask, sample_weight=None):
        if supervised_mask.sum() == 0:
            return logits.new_tensor(0.0)

        logits_sup = logits[supervised_mask]
        y_sup = y[supervised_mask].float()

        pos_weight = torch.tensor(self.cls_pos_weight, dtype=logits.dtype, device=logits.device)
        loss_vec = F.binary_cross_entropy_with_logits(
            logits_sup, y_sup, pos_weight=pos_weight, reduction='none'
        )

        if sample_weight is not None:
            w = sample_weight[supervised_mask].float()
            loss_vec = loss_vec * w

        return loss_vec.mean()

    def prototype_margin_loss(self, z, y, supervised_mask, p_pos, p_neg):
        if supervised_mask.sum() == 0:
            return z.new_tensor(0.0)

        z_sup = z[supervised_mask]
        y_sup = y[supervised_mask].float()

        d_pos = torch.sum((z_sup - p_pos.unsqueeze(0)) ** 2, dim=1)
        d_neg = torch.sum((z_sup - p_neg.unsqueeze(0)) ** 2, dim=1)

        pos_mask = (y_sup == 1)
        neg_mask = (y_sup == 0)

        loss_pos = z.new_tensor(0.0)
        loss_neg = z.new_tensor(0.0)

        if pos_mask.sum() > 0:
            loss_pos = F.relu(self.proto_margin + d_pos[pos_mask] - d_neg[pos_mask]).mean()
        if neg_mask.sum() > 0:
            loss_neg = F.relu(self.proto_margin + d_neg[neg_mask] - d_pos[neg_mask]).mean()

        return loss_pos + loss_neg

    def hard_negative_loss(self, logits, hard_neg_idx):
        if hard_neg_idx is None or len(hard_neg_idx) == 0:
            return logits.new_tensor(0.0)

        hard_logits = logits[hard_neg_idx]
        # 希望 hard negatives 更偏向负类
        return F.relu(self.hard_neg_margin + hard_logits).mean()

    def graph_structure_alignment_loss(self, view_embeddings, sampled_idx):
        """
        对齐视图间关系图，而不是直接对齐 embedding
        """
        if sampled_idx is None or len(sampled_idx) < 2:
            return next(iter(view_embeddings.values())).new_tensor(0.0)

        keys = list(view_embeddings.keys())
        loss = 0.0
        cnt = 0

        sims = {}
        for k in keys:
            z = view_embeddings[k][sampled_idx]
            z = F.normalize(z, p=2, dim=1)
            s = torch.matmul(z, z.t())        # [B,B]
            s = F.softmax(s, dim=1)
            sims[k] = s

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                loss = loss + F.mse_loss(sims[keys[i]], sims[keys[j]])
                cnt += 1

        return loss / max(cnt, 1)

    def positive_mixup_loss(self, mixed_logits):
        if mixed_logits is None or mixed_logits.numel() == 0:
            return mixed_logits.new_tensor(0.0) if mixed_logits is not None else torch.tensor(0.0)
        target = torch.ones_like(mixed_logits)
        return F.binary_cross_entropy_with_logits(mixed_logits, target)

    def forward(
        self,
        output_dict,
        y,
        supervised_mask,
        hard_neg_idx=None,
        align_sample_idx=None,
        sample_weight=None,
        mixed_logits=None
    ):
        logits = output_dict['logits']
        z = output_dict['embedding']
        p_pos = output_dict['p_pos']
        p_neg = output_dict['p_neg']
        view_embeddings = output_dict['view_embeddings']

        loss_cls = self.classification_loss(logits, y, supervised_mask, sample_weight=sample_weight)
        loss_proto = self.prototype_margin_loss(z, y, supervised_mask, p_pos, p_neg)
        loss_hard_neg = self.hard_negative_loss(logits, hard_neg_idx)
        loss_align = self.graph_structure_alignment_loss(view_embeddings, align_sample_idx)
        loss_mixup = self.positive_mixup_loss(mixed_logits)

        total = (
            self.lambda_cls * loss_cls
            + self.lambda_proto * loss_proto
            + self.lambda_hard_neg * loss_hard_neg
            + self.lambda_align * loss_align
            + self.lambda_mixup * loss_mixup
        )

        loss_dict = {
            'total': total.detach(),
            'cls': loss_cls.detach(),
            'proto': loss_proto.detach(),
            'hard_neg': loss_hard_neg.detach(),
            'align': loss_align.detach(),
            'mixup': loss_mixup.detach(),
        }
        return total, loss_dict
