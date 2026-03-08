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
        proj_embeddings: dict of [N, D], each already normalized
        pos_idx: strict positive indices in training set
        neg_idx: strict negative indices in training set
        """
        device = list(proj_embeddings.values())[0].device
        view_names = list(proj_embeddings.keys())
        num_views = len(view_names)

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            return torch.tensor(0.0, device=device)

        labeled_idx = torch.cat([pos_idx, neg_idx], dim=0)
        labels = torch.cat([
            torch.ones(len(pos_idx), device=device),
            torch.zeros(len(neg_idx), device=device)
        ], dim=0)

        # [L, V, D]
        z = torch.stack([proj_embeddings[v][labeled_idx] for v in view_names], dim=1)

        total_loss = 0.0
        valid_count = 0

        L = z.size(0)

        for i in range(L):
            anchor_label = labels[i]

            for m in range(num_views):
                anchor = z[i, m]  # [D]

                sim_all = []
                pos_mask = []
                neg_mask = []

                for j in range(L):
                    for n in range(num_views):
                        # 排除自身同视图
                        if i == j and m == n:
                            continue

                        sim = F.cosine_similarity(
                            anchor.unsqueeze(0),
                            z[j, n].unsqueeze(0),
                            dim=1
                        )[0] / self.temperature

                        sim_all.append(sim)

                        same_label = (labels[j] == anchor_label)
                        same_node_other_view = (i == j and m != n)

                        if same_node_other_view or (same_label and i != j):
                            pos_mask.append(True)
                            neg_mask.append(False)
                        elif labels[j] != anchor_label:
                            pos_mask.append(False)
                            neg_mask.append(True)
                        else:
                            pos_mask.append(False)
                            neg_mask.append(False)

                if len(sim_all) == 0:
                    continue

                sim_all = torch.stack(sim_all)  # [K]
                exp_sim = torch.exp(sim_all)

                pos_mask = torch.tensor(pos_mask, dtype=torch.bool, device=device)
                neg_mask = torch.tensor(neg_mask, dtype=torch.bool, device=device)

                denom_mask = pos_mask | neg_mask

                if pos_mask.sum() == 0 or denom_mask.sum() == 0:
                    continue

                denom = exp_sim[denom_mask].sum() + 1e-12
                pos_vals = exp_sim[pos_mask]

                loss_i = -torch.log(pos_vals / denom).mean()
                total_loss += loss_i
                valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=device)

        return total_loss / valid_count

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