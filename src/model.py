import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def add_self_loops(edge_index, edge_weight, num_nodes, fill_value=1.0):
    device = edge_index.device
    loop = torch.arange(num_nodes, device=device)
    loop_index = torch.stack([loop, loop], dim=0)
    loop_weight = torch.full((num_nodes,), fill_value, dtype=edge_weight.dtype, device=device)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
    return edge_index, edge_weight


def normalize_edge_index(edge_index, edge_weight, num_nodes):
    row, col = edge_index[0], edge_index[1]
    deg = torch.zeros(num_nodes, device=edge_weight.device, dtype=edge_weight.dtype)
    deg.scatter_add_(0, row, edge_weight)
    deg_inv_sqrt = deg.clamp(min=1e-12).pow(-0.5)
    return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class SparseGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes, fill_value=1.0)
        norm_weight = normalize_edge_index(edge_index, edge_weight, num_nodes)
        h = self.linear(x)
        row, col = edge_index[0], edge_index[1]
        out = torch.zeros_like(h)
        out.index_add_(0, row, h[col] * norm_weight.unsqueeze(-1))
        return out


class ResidualGCNEncoder(nn.Module):
    def __init__(self, in_dim, pos_dim, hidden_dim, out_dim, num_layers=2, dropout=0.2, pos_hidden_dim=16):
        super().__init__()
        self.dropout = dropout
        self.pos_proj = nn.Sequential(
            nn.Linear(pos_dim, pos_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_hidden_dim, pos_hidden_dim),
        )
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_proj = nn.ModuleList()
        self.layer_dims = []
        current_dim = in_dim + pos_hidden_dim
        for layer_id in range(num_layers):
            next_dim = hidden_dim if layer_id < num_layers - 1 else out_dim
            self.layers.append(SparseGCNLayer(current_dim, next_dim))
            self.norms.append(nn.LayerNorm(next_dim))
            self.res_proj.append(nn.Linear(current_dim, next_dim, bias=False) if current_dim != next_dim else nn.Identity())
            self.layer_dims.append(next_dim)
            current_dim = next_dim
        self.jk_proj = nn.Sequential(
            nn.Linear(sum(self.layer_dims), out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x, pos_feat, edge_index, edge_weight):
        pe = self.pos_proj(pos_feat)
        h = torch.cat([x, pe], dim=1)
        num_nodes = h.size(0)
        layer_outs = []
        for layer, norm, res_proj in zip(self.layers, self.norms, self.res_proj):
            h_new = layer(h, edge_index, edge_weight, num_nodes)
            h_new = norm(h_new + 0.2 * res_proj(h))
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            layer_outs.append(h_new)
            h = h_new
        return self.jk_proj(torch.cat(layer_outs, dim=1))


class TopoPromptAdapter(nn.Module):
    def __init__(self, topo_dim, prompt_dim=16, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(topo_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, prompt_dim),
            nn.LayerNorm(prompt_dim),
        )

    def forward(self, topo_feat):
        return self.net(topo_feat)


class TopoConditionedGroupGate(nn.Module):
    def __init__(self, total_dim, topo_dim, base_dim=3, hidden_dim=64, prompt_dim=16, dropout=0.1):
        super().__init__()
        self.total_dim = total_dim
        self.base_dim = min(base_dim, total_dim)
        self.group_slices = {
            'base': slice(0, self.base_dim),
            'methy': slice(self.base_dim, min(self.base_dim + 1, total_dim)),
            'crispr': slice(min(self.base_dim + 1, total_dim), min(self.base_dim + 2, total_dim)),
            'hic': slice(min(self.base_dim + 2, total_dim), total_dim),
        }
        self.group_names = [k for k, s in self.group_slices.items() if s.start < s.stop]
        self.topo_prompt = TopoPromptAdapter(topo_dim, prompt_dim=prompt_dim, hidden_dim=hidden_dim, dropout=dropout)
        joint_dim = total_dim + prompt_dim
        self.group_gate = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.group_names)),
            nn.Sigmoid(),
        )
        self.feature_refine = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, total_dim),
            nn.Sigmoid(),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, topo_feat):
        topo_prompt = self.topo_prompt(topo_feat)
        joint = torch.cat([x, topo_prompt], dim=1)
        group_score = self.group_gate(joint)
        refine = self.feature_refine(joint)
        gate = torch.ones_like(x)
        gate_stats = {}
        for gid, name in enumerate(self.group_names):
            sl = self.group_slices[name]
            g = group_score[:, gid:gid + 1]
            gate[:, sl] = g * refine[:, sl]
            gate_stats[name] = g.mean().detach()
        x_out = x * (1.0 + self.res_scale * gate)
        return x_out, gate, gate_stats, topo_prompt


class ResidualSpecificAdapter(nn.Module):
    def __init__(self, in_dim, topo_prompt_dim, out_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + topo_prompt_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, shared_emb, topo_prompt):
        return self.scale * self.net(torch.cat([shared_emb, topo_prompt], dim=1))


class DualChannelViewEncoder(nn.Module):
    def __init__(self, in_dim, pos_dim, topo_prompt_dim, hidden_dim, shared_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.shared_encoder = ResidualGCNEncoder(in_dim, pos_dim, hidden_dim, shared_dim, num_layers=num_layers, dropout=dropout)
        self.specific_adapter = ResidualSpecificAdapter(shared_dim, topo_prompt_dim, shared_dim, hidden_dim=max(32, hidden_dim // 2), dropout=dropout)
        self.out_proj = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(shared_dim),
        )

    def forward(self, x, pos_feat, topo_prompt, edge_index, edge_weight):
        z_shared = self.shared_encoder(x, pos_feat, edge_index, edge_weight)
        r_specific = self.specific_adapter(z_shared, topo_prompt)
        z_view = self.out_proj(z_shared + r_specific)
        return z_view, z_shared, r_specific


class DualGateFusion(nn.Module):
    def __init__(self, view_names, embed_dim, topo_prompt_dim, gate_hidden_dim=32, dropout=0.1):
        super().__init__()
        self.view_names = list(view_names)
        self.cls_gate = nn.Sequential(
            nn.Linear(embed_dim + topo_prompt_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 1),
        )
        self.proto_gate = nn.Sequential(
            nn.Linear(embed_dim + topo_prompt_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 1),
        )
        self.cls_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )
        self.proto_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, view_embeddings, topo_prompt):
        cls_scores, proto_scores, stacked = [], [], []
        for v in self.view_names:
            z = view_embeddings[v]
            stacked.append(z.unsqueeze(1))
            inp = torch.cat([z, topo_prompt], dim=1)
            cls_scores.append(self.cls_gate(inp))
            proto_scores.append(self.proto_gate(inp))
        stacked = torch.cat(stacked, dim=1)
        cls_w = torch.softmax(torch.cat(cls_scores, dim=1), dim=1)
        proto_w = torch.softmax(torch.cat(proto_scores, dim=1), dim=1)
        z_cls = torch.sum(stacked * cls_w.unsqueeze(-1), dim=1)
        z_proto = torch.sum(stacked * proto_w.unsqueeze(-1), dim=1)
        gate_stats = {
            'cls': {v: cls_w[:, i].mean().detach() for i, v in enumerate(self.view_names)},
            'proto': {v: proto_w[:, i].mean().detach() for i, v in enumerate(self.view_names)},
        }
        return self.cls_proj(z_cls), self.proto_proj(z_proto), cls_w, proto_w, gate_stats


class SetContextEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, z_sub):
        if z_sub is None or z_sub.numel() == 0:
            return None
        mean_vec = z_sub.mean(dim=0)
        max_vec = z_sub.max(dim=0).values
        return self.net(torch.cat([mean_vec, max_vec], dim=0))


def _kmeans_torch(x, k, iters=5):
    n = x.size(0)
    if n == 0:
        return x.new_zeros((k, x.size(1)))
    k_eff = min(k, n)
    center_ids = torch.linspace(0, n - 1, steps=k_eff, device=x.device).long()
    centers = x[center_ids].clone()
    for _ in range(iters):
        dist = torch.cdist(x, centers)
        assign = dist.argmin(dim=1)
        new_centers = []
        for ci in range(k_eff):
            mask = assign == ci
            new_centers.append(x[mask].mean(dim=0) if mask.any() else centers[ci])
        centers = torch.stack(new_centers, dim=0)
    if k_eff < k:
        pad = centers[-1:].repeat(k - k_eff, 1)
        centers = torch.cat([centers, pad], dim=0)
    return centers


class MultiPositivePrototypeHead(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_pos_prototypes=3,
        hidden_dim=128,
        proto_alpha=0.25,
        ema_momentum=0.95,
        proto_shift_scale=0.15,
        task_aug_ratio=0.20,
        memory_size=32,
        proto_logit_scale=0.20,
        logit_temperature=2.5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_pos_prototypes = num_pos_prototypes
        self.proto_alpha = proto_alpha
        self.ema_momentum = ema_momentum
        self.proto_shift_scale = proto_shift_scale
        self.task_aug_ratio = task_aug_ratio
        self.memory_size = memory_size
        self.proto_logit_scale = proto_logit_scale
        self.logit_temperature = logit_temperature
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.pos_ctx_proj = nn.Linear(embed_dim, embed_dim)
        self.neg_ctx_proj = nn.Linear(embed_dim, embed_dim)
        self.register_buffer('ema_pos', torch.zeros(num_pos_prototypes, embed_dim))
        self.register_buffer('ema_neg', torch.zeros(embed_dim))
        self.register_buffer('ema_initialized', torch.tensor(False))
        self._proto_memory = []

    def _build_prototypes(self, z, pos_idx, neg_idx):
        pos_idx = list(pos_idx) if pos_idx is not None else []
        neg_idx = list(neg_idx) if neg_idx is not None else []
        if len(pos_idx) == 0:
            pos_proto = self.ema_pos if bool(self.ema_initialized.item()) else z.new_zeros((self.num_pos_prototypes, self.embed_dim))
        else:
            pos_proto = _kmeans_torch(z[pos_idx], self.num_pos_prototypes, iters=4)
        if len(neg_idx) == 0:
            neg_proto = self.ema_neg if bool(self.ema_initialized.item()) else z.new_zeros(self.embed_dim)
        else:
            neg_proto = z[neg_idx].mean(dim=0)
        return pos_proto, neg_proto

    def _calibrate(self, pos_proto, neg_proto, pos_ctx, neg_ctx):
        if pos_ctx is not None:
            pos_shift = self.pos_ctx_proj(pos_ctx).unsqueeze(0).expand_as(pos_proto)
            pos_proto = pos_proto + self.proto_shift_scale * pos_shift
        if neg_ctx is not None:
            neg_proto = neg_proto + self.proto_shift_scale * self.neg_ctx_proj(neg_ctx)
        if bool(self.ema_initialized.item()):
            pos_proto = (1.0 - self.proto_alpha) * pos_proto + self.proto_alpha * self.ema_pos
            neg_proto = (1.0 - self.proto_alpha) * neg_proto + self.proto_alpha * self.ema_neg
        return pos_proto, neg_proto

    def _update_ema(self, pos_proto, neg_proto):
        with torch.no_grad():
            if not bool(self.ema_initialized.item()):
                self.ema_pos.copy_(pos_proto.detach())
                self.ema_neg.copy_(neg_proto.detach())
                self.ema_initialized.fill_(True)
            else:
                self.ema_pos.mul_(self.ema_momentum).add_((1.0 - self.ema_momentum) * pos_proto.detach())
                self.ema_neg.mul_(self.ema_momentum).add_((1.0 - self.ema_momentum) * neg_proto.detach())
            self._proto_memory.append(pos_proto.detach().clone())
            if len(self._proto_memory) > self.memory_size:
                self._proto_memory.pop(0)

    def _augment_pos_proto(self, pos_proto):
        if len(self._proto_memory) < 4:
            return None
        if torch.rand(1, device=pos_proto.device).item() > self.task_aug_ratio:
            return None
        ref = self._proto_memory[torch.randint(0, len(self._proto_memory), (1,)).item()].to(pos_proto.device)
        lam = torch.empty(1, device=pos_proto.device).uniform_(0.45, 0.55).item()
        return lam * pos_proto + (1.0 - lam) * ref

    def forward(self, z_cls, z_proto, pos_idx, neg_idx, pos_ctx=None, neg_ctx=None, update_ema=False):
        pos_proto, neg_proto = self._build_prototypes(z_proto, pos_idx, neg_idx)
        pos_proto, neg_proto = self._calibrate(pos_proto, neg_proto, pos_ctx, neg_ctx)
        if update_ema:
            self._update_ema(pos_proto, neg_proto)

        d_pos_all = torch.cdist(z_proto, pos_proto, p=2).pow(2)
        d_pos, pos_assign = d_pos_all.min(dim=1)
        d_neg = ((z_proto - neg_proto.unsqueeze(0)) ** 2).mean(dim=1)

        proto_logit = (d_neg - d_pos) / math.sqrt(max(1, z_proto.size(1)))
        cls_logit = self.cls_head(z_cls).squeeze(-1)
        raw_logits = cls_logit + self.proto_logit_scale * proto_logit
        logits = raw_logits / self.logit_temperature
        prob = torch.sigmoid(logits)

        aug_pos_proto = self._augment_pos_proto(pos_proto)
        aug_proto_logits = None
        if aug_pos_proto is not None:
            d_pos_aug = torch.cdist(z_proto, aug_pos_proto, p=2).pow(2).min(dim=1).values
            aug_proto_logits = self.proto_logit_scale * (d_neg - d_pos_aug) / math.sqrt(max(1, z_proto.size(1)))

        return {
            'logits': logits,
            'prob': prob,
            'embedding': z_cls,
            'proto_embedding': z_proto,
            'pos_prototypes': pos_proto,
            'neg_prototype': neg_proto,
            'proto_logits': self.proto_logit_scale * proto_logit,
            'aug_proto_logits': aug_proto_logits,
            'pos_assign': pos_assign,
            'd_pos_all': d_pos_all,
        }


class DriverGeneFewShotModel(nn.Module):
    def __init__(
        self,
        in_dim,
        pos_dim,
        hidden_dim,
        embed_dim,
        num_layers,
        dropout,
        view_names,
        proto_alpha=0.25,
        base_feature_dim=3,
        feature_gate_hidden_dim=64,
        ema_momentum=0.95,
        proto_shift_scale=0.15,
        topo_prompt_dim=16,
        gate_hidden_dim=32,
        num_pos_prototypes=3,
        task_aug_ratio=0.3,
        proto_logit_scale=0.20,
        logit_temperature=2.5,
    ):
        super().__init__()
        self.view_names = list(view_names)
        self.feature_gate = TopoConditionedGroupGate(
            total_dim=in_dim,
            topo_dim=pos_dim,
            base_dim=base_feature_dim,
            hidden_dim=feature_gate_hidden_dim,
            prompt_dim=topo_prompt_dim,
            dropout=dropout,
        )
        self.view_encoders = nn.ModuleDict({
            v: DualChannelViewEncoder(
                in_dim=in_dim,
                pos_dim=pos_dim,
                topo_prompt_dim=topo_prompt_dim,
                hidden_dim=hidden_dim,
                shared_dim=embed_dim,
                num_layers=num_layers,
                dropout=dropout,
            ) for v in self.view_names
        })
        self.fusion = DualGateFusion(
            view_names=self.view_names,
            embed_dim=embed_dim,
            topo_prompt_dim=topo_prompt_dim,
            gate_hidden_dim=gate_hidden_dim,
            dropout=dropout,
        )
        self.set_context = SetContextEncoder(embed_dim=embed_dim, hidden_dim=hidden_dim)
        self.proto_head = MultiPositivePrototypeHead(
            embed_dim=embed_dim,
            num_pos_prototypes=num_pos_prototypes,
            hidden_dim=hidden_dim,
            proto_alpha=proto_alpha,
            ema_momentum=ema_momentum,
            proto_shift_scale=proto_shift_scale,
            task_aug_ratio=task_aug_ratio,
            proto_logit_scale=proto_logit_scale,
            logit_temperature=logit_temperature,
        )

    def forward(self, x, pos_feat, edge_index_dict, edge_weight_dict, pos_idx, neg_idx,  update_ema=False):
        x_gated, feature_gate, gate_stats, topo_prompt = self.feature_gate(x, pos_feat)
        view_embeddings, shared_embeddings, residual_embeddings = {}, {}, {}
        for v in self.view_names:
            z_v, z_shared, r_specific = self.view_encoders[v](x_gated, pos_feat, topo_prompt, edge_index_dict[v], edge_weight_dict[v])
            view_embeddings[v] = z_v
            shared_embeddings[v] = z_shared
            residual_embeddings[v] = r_specific

        z_cls, z_proto, view_gates_cls, view_gates_proto, fusion_stats = self.fusion(view_embeddings, topo_prompt)
        pos_ctx = self.set_context(z_proto[pos_idx] if pos_idx is not None and len(pos_idx) > 0 else None)
        neg_ctx = self.set_context(z_proto[neg_idx] if neg_idx is not None and len(neg_idx) > 0 else None)
        proto_out = self.proto_head(z_cls, z_proto, pos_idx, neg_idx, pos_ctx=pos_ctx, neg_ctx=neg_ctx, update_ema=update_ema)
        proto_out.update({
            'feature_gate': feature_gate,
            'feature_gate_stats': gate_stats,
            'topo_prompt': topo_prompt,
            'shared_embeddings': shared_embeddings,
            'residual_embeddings': residual_embeddings,
            'view_embeddings': view_embeddings,
            'view_gates_cls': view_gates_cls,
            'view_gates_proto': view_gates_proto,
            'fusion_stats': fusion_stats,
        })
        return proto_out
