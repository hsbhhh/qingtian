import torch
import torch.nn as nn
import torch.nn.functional as F


class DriverGeneLoss(nn.Module):
    def __init__(
        self,
        cls_pos_weight=8.0,
        lambda_cls=1.0,
        lambda_proto=0.10,
        lambda_shared=0.05,
        lambda_gate_div=0.03,
        lambda_task_aug=0.01,
        lambda_gate_sparse=1e-3,
        lambda_proto_balance=0.03,
    ):
        super().__init__()
        self.cls_pos_weight = cls_pos_weight
        self.lambda_cls = lambda_cls
        self.lambda_proto = lambda_proto
        self.lambda_shared = lambda_shared
        self.lambda_gate_div = lambda_gate_div
        self.lambda_task_aug = lambda_task_aug
        self.lambda_gate_sparse = lambda_gate_sparse
        self.lambda_proto_balance = lambda_proto_balance

    def classification_loss(self, logits, y, supervised_mask, sample_weight=None):
        if supervised_mask.sum() == 0:
            return logits.new_tensor(0.0)
        logits_sup = logits[supervised_mask]
        y_sup = y[supervised_mask].float()
        pos_weight = torch.tensor(self.cls_pos_weight, dtype=logits.dtype, device=logits.device)
        loss_vec = F.binary_cross_entropy_with_logits(logits_sup, y_sup, pos_weight=pos_weight, reduction='none')
        if sample_weight is not None:
            loss_vec = loss_vec * sample_weight[supervised_mask].float()
        return loss_vec.mean()

    def prototype_ce_loss(self, proto_logits, y, proto_mask):
        if proto_logits is None or proto_mask is None or proto_mask.sum() == 0:
            return y.new_tensor(0.0)
        logits_sup = proto_logits[proto_mask]
        y_sup = y[proto_mask].float()
        pos_weight = torch.tensor(self.cls_pos_weight, dtype=logits_sup.dtype, device=logits_sup.device)
        return F.binary_cross_entropy_with_logits(logits_sup, y_sup, pos_weight=pos_weight)

    def shared_consistency_loss(self, shared_embeddings):
        view_names = list(shared_embeddings.keys())
        if len(view_names) <= 1:
            return next(iter(shared_embeddings.values())).new_tensor(0.0)
        loss = 0.0
        pairs = 0
        for i in range(len(view_names)):
            for j in range(i + 1, len(view_names)):
                zi = F.normalize(shared_embeddings[view_names[i]], dim=1)
                zj = F.normalize(shared_embeddings[view_names[j]], dim=1)
                loss = loss + (1.0 - (zi * zj).sum(dim=1).mean())
                pairs += 1
        return loss / max(pairs, 1)

    def gate_diversity_loss(self, view_gates_cls, view_gates_proto):
        if view_gates_cls is None or view_gates_proto is None:
            return torch.tensor(0.0)
        cls_mean = view_gates_cls.mean(dim=0)
        proto_mean = view_gates_proto.mean(dim=0)
        target = torch.full_like(cls_mean, 1.0 / cls_mean.numel())
        loss_cls = ((cls_mean - target) ** 2).mean()
        loss_proto = ((proto_mean - target) ** 2).mean()
        return loss_cls + loss_proto

    def gate_sparsity_loss(self, feature_gate):
        if feature_gate is None:
            return torch.tensor(0.0)
        return feature_gate.mean()

    def proto_balance_loss(self, pos_assign, y, proto_mask, num_pos_prototypes):
        if pos_assign is None or proto_mask is None or proto_mask.sum() == 0:
            return y.new_tensor(0.0)
        pos_mask = proto_mask & (y > 0.5)
        if pos_mask.sum() == 0:
            return y.new_tensor(0.0)
        assign = pos_assign[pos_mask]
        counts = torch.bincount(assign, minlength=num_pos_prototypes).float()
        probs = counts / counts.sum().clamp(min=1.0)
        target = torch.full_like(probs, 1.0 / num_pos_prototypes)
        return ((probs - target) ** 2).mean()

    def forward(self, output_dict, y, supervised_mask, proto_mask=None, sample_weight=None):
        logits = output_dict['logits']
        if proto_mask is None:
            proto_mask = supervised_mask
        cls_loss = self.classification_loss(logits, y, supervised_mask, sample_weight=sample_weight)
        proto_loss = self.prototype_ce_loss(output_dict.get('proto_logits', None), y, proto_mask)
        shared_loss = self.shared_consistency_loss(output_dict['shared_embeddings'])
        gate_div = self.gate_diversity_loss(output_dict.get('view_gates_cls', None), output_dict.get('view_gates_proto', None)).to(logits.device)
        gate_sparse = self.gate_sparsity_loss(output_dict.get('feature_gate', None)).to(logits.device)
        task_aug = self.prototype_ce_loss(output_dict.get('aug_proto_logits', None), y, proto_mask)
        proto_balance = self.proto_balance_loss(
            output_dict.get('pos_assign', None),
            y,
            proto_mask,
            output_dict['pos_prototypes'].size(0),
        )
        total = (
            self.lambda_cls * cls_loss
            + self.lambda_proto * proto_loss
            + self.lambda_shared * shared_loss
            + self.lambda_gate_div * gate_div
            + self.lambda_task_aug * task_aug
            + self.lambda_gate_sparse * gate_sparse
            + self.lambda_proto_balance * proto_balance
        )
        return total, {
            'cls': cls_loss.detach(),
            'proto': proto_loss.detach(),
            'shared': shared_loss.detach(),
            'gate_div': gate_div.detach(),
            'task_aug': task_aug.detach(),
            'gate_sparse': gate_sparse.detach(),
            'proto_balance': proto_balance.detach(),
        }
