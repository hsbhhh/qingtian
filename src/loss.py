import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss_fun(nn.Module):
    def __init__(
        self,
        temperature=0.2,
        lambda_main=1.0,
        lambda_view=1.0,
        lambda_sup=1.0,
        lambda_unsup=0.2,
        max_unsup_samples=2048
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_main = lambda_main
        self.lambda_view = lambda_view
        self.lambda_sup = lambda_sup
        self.lambda_unsup = lambda_unsup
        self.max_unsup_samples = max_unsup_samples

    def masked_bce_loss(self, logits, labels, mask, pos_weight=None):
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        logits_m = logits[mask]
        labels_m = labels[mask]

        if pos_weight is not None:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            loss_fn = nn.BCEWithLogitsLoss()

        return loss_fn(logits_m, labels_m)

    def supervised_contrastive_loss(self, proj_embeddings, pos_idx, neg_idx):
        """
        proj_embeddings: dict, each tensor [N, D], already normalized
        pos_idx: tensor of strict positive node indices in training set
        neg_idx: tensor of strict negative node indices in training set
        """
        device = list(proj_embeddings.values())[0].device
        view_names = list(proj_embeddings.keys())
        num_views = len(view_names)

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            return torch.tensor(0.0, device=device)

        # 1) labeled nodes
        labeled_idx = torch.cat([pos_idx, neg_idx], dim=0)   # [L]
        node_labels = torch.cat([
            torch.ones(len(pos_idx), device=device),
            torch.zeros(len(neg_idx), device=device)
        ], dim=0)  # [L]

        L = labeled_idx.size(0)

        # 2) stack views: [L, V, D]
        z = torch.stack([proj_embeddings[v][labeled_idx] for v in view_names], dim=1)

        # 3) flatten -> [L*V, D]
        z_flat = z.reshape(L * num_views, -1)

        # 因为 projector 里已经 normalize 过，所以点积就是 cosine similarity
        sim_matrix = torch.matmul(z_flat, z_flat.t()) / self.temperature   # [LV, LV]

        # 数值稳定
        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach()
        exp_sim = torch.exp(sim_matrix)

        # 4) 构造 node_id / view_id / class_label
        node_ids = torch.arange(L, device=device).repeat_interleave(num_views)   # [LV]
        view_ids = torch.arange(num_views, device=device).repeat(L)              # [LV]
        flat_labels = node_labels.repeat_interleave(num_views)                   # [LV]

        # [LV, LV]
        same_node = node_ids.unsqueeze(1) == node_ids.unsqueeze(0)
        same_view = view_ids.unsqueeze(1) == view_ids.unsqueeze(0)
        same_label = flat_labels.unsqueeze(1) == flat_labels.unsqueeze(0)

        # 排除自己
        self_mask = torch.eye(L * num_views, dtype=torch.bool, device=device)

        # 正样本：
        # 1) 同一个节点不同视图
        # 2) 不同节点但同标签
        pos_mask = ((same_node & (~same_view)) | ((~same_node) & same_label)) & (~self_mask)

        # 负样本：不同节点且异标签
        neg_mask = ((~same_node) & (~same_label)) & (~self_mask)

        # 分母只包含正样本和负样本
        denom_mask = pos_mask | neg_mask

        # 5) 计算 loss
        denom = (exp_sim * denom_mask.float()).sum(dim=1) + 1e-12    # [LV]
        pos_exp = exp_sim * pos_mask.float()                         # [LV, LV]

        # 每个 anchor 的正样本个数
        pos_count = pos_mask.sum(dim=1)   # [LV]

        valid_anchor = (pos_count > 0) & (denom_mask.sum(dim=1) > 0)

        if valid_anchor.sum() == 0:
            return torch.tensor(0.0, device=device)

        # 对每个 anchor，把所有正样本的 -log(exp(sim_pos)/denom) 取平均
        log_prob = -torch.log((pos_exp + 1e-12) / denom.unsqueeze(1))
        loss_per_anchor = (log_prob * pos_mask.float()).sum(dim=1) / (pos_count.float() + 1e-12)

        loss = loss_per_anchor[valid_anchor].mean()
        return loss
    
    def unsupervised_consistency_loss(self, proj_embeddings, unlabeled_idx):
        """
        仅对无标签节点做“同一基因跨视图一致性”
        """
        device = list(proj_embeddings.values())[0].device
        view_names = list(proj_embeddings.keys())
        num_views = len(view_names)

        if len(unlabeled_idx) == 0:
            return torch.tensor(0.0, device=device)

        if len(unlabeled_idx) > self.max_unsup_samples:
            rand_perm = torch.randperm(len(unlabeled_idx), device=device)
            unlabeled_idx = unlabeled_idx[rand_perm[:self.max_unsup_samples]]

        # [U, V, D]
        z = torch.stack([proj_embeddings[v][unlabeled_idx] for v in view_names], dim=1)
        U = z.size(0)

        all_z = z.reshape(U * num_views, -1)  # [U*V, D]

        total_loss = 0.0
        valid_count = 0

        for i in range(U):
            for m in range(num_views):
                anchor = z[i, m]

                sim_all = F.cosine_similarity(
                    anchor.unsqueeze(0), all_z, dim=1
                ) / self.temperature
                exp_sim = torch.exp(sim_all)

                # self index in flattened all_z
                self_flat_idx = i * num_views + m

                # positive: same node, other views
                pos_indices = [
                    i * num_views + n for n in range(num_views) if n != m
                ]

                denom_mask = torch.ones(U * num_views, dtype=torch.bool, device=device)
                denom_mask[self_flat_idx] = False

                pos_mask = torch.zeros(U * num_views, dtype=torch.bool, device=device)
                pos_mask[pos_indices] = True

                denom = exp_sim[denom_mask].sum() + 1e-12
                pos_vals = exp_sim[pos_mask]

                if len(pos_vals) == 0:
                    continue

                loss_i = -torch.log(pos_vals / denom).mean()
                total_loss += loss_i
                valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=device)

        return total_loss / valid_count

    def forward(
        self,
        outputs,
        labels,
        train_mask,
        train_pos_idx,
        train_neg_idx,
        unlabeled_idx,
        pos_weight=None
    ):
        fused_logit = outputs["fused_logit"]
        view_logits = outputs["view_logits"]
        proj_embeddings = outputs["proj_embeddings"]

        main_loss = self.masked_bce_loss(
            fused_logit, labels, train_mask, pos_weight=pos_weight
        )

        view_loss = 0.0
        for _, logit in view_logits.items():
            view_loss += self.masked_bce_loss(
                logit, labels, train_mask, pos_weight=pos_weight
            )
        view_loss = view_loss / len(view_logits)

        sup_cl_loss = self.supervised_contrastive_loss(
            proj_embeddings, train_pos_idx, train_neg_idx
        )

        unsup_cl_loss = self.unsupervised_consistency_loss(
            proj_embeddings, unlabeled_idx
        )

        total_loss = (
            self.lambda_main * main_loss +
            self.lambda_view * view_loss +
            self.lambda_sup * sup_cl_loss +
            self.lambda_unsup * unsup_cl_loss
        )

        return {
            "total_loss": total_loss,
            "main_loss": main_loss,
            "view_loss": view_loss,
            "sup_cl_loss": sup_cl_loss,
            "unsup_cl_loss": unsup_cl_loss
        }