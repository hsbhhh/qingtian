# model.py
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
    norm_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return norm_weight


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


class GCNEncoder(nn.Module):
    """
    每层都注入位置编码（结构先验），而不是只在输入层加一次
    """
    def __init__(self, in_dim, pos_dim, hidden_dim, out_dim, num_layers=2, dropout=0.2, pos_hidden_dim=16):
        super().__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        self.dropout = dropout

        self.pos_proj = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        current_dim = in_dim
        for layer_id in range(num_layers):
            self.pos_proj.append(
                nn.Sequential(
                    nn.Linear(pos_dim, pos_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(pos_hidden_dim, pos_hidden_dim)
                )
            )
            next_dim = out_dim if layer_id == num_layers - 1 else hidden_dim
            self.layers.append(SparseGCNLayer(current_dim + pos_hidden_dim, next_dim))
            if layer_id != num_layers - 1:
                self.norms.append(nn.LayerNorm(next_dim))
            current_dim = next_dim

    def forward(self, x, pos_feat, edge_index, edge_weight):
        h = x
        num_nodes = h.size(0)
        for i, layer in enumerate(self.layers):
            pe = self.pos_proj[i](pos_feat)
            h = torch.cat([h, pe], dim=1)
            h = layer(h, edge_index, edge_weight, num_nodes)
            if i < len(self.layers) - 1:
                h = self.norms[i](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class ViewFusionGate(nn.Module):
    """
    每个视图独立 gate，而不是共享一个 gate MLP
    """
    def __init__(self, view_names, in_dim, gate_hidden_dim=64):
        super().__init__()
        self.view_names = view_names
        self.gates = nn.ModuleDict({
            v: nn.Sequential(
                nn.Linear(in_dim, gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, 1)
            )
            for v in view_names
        })

    def forward(self, view_embeddings):
        # view_embeddings: dict {view_name: [N, D]}
        scores = []
        embeds = []
        for v in self.view_names:
            z = view_embeddings[v]
            scores.append(self.gates[v](z))   # [N,1]
            embeds.append(z)
        scores = torch.cat(scores, dim=1)     # [N,V]
        alpha = F.softmax(scores, dim=1)

        stacked = torch.stack(embeds, dim=1)  # [N,V,D]
        fused = torch.sum(alpha.unsqueeze(-1) * stacked, dim=1)
        return fused, alpha


class PrototypeHead(nn.Module):
    """
    用距离粗中心最近的 top-k 样本计算原型
    """
    def __init__(self, embed_dim, hidden_dim=128, proto_alpha=0.25, proto_topk_ratio=0.6, min_proto_k=8):
        super().__init__()
        self.proto_alpha = proto_alpha
        self.proto_topk_ratio = proto_topk_ratio
        self.min_proto_k = min_proto_k

        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def _topk_center_proto(self, z_sub):
        if z_sub.size(0) == 0:
            return torch.zeros(z_sub.size(1), device=z_sub.device)

        center = z_sub.mean(dim=0, keepdim=True)
        dist = torch.sum((z_sub - center) ** 2, dim=1)
        k = max(self.min_proto_k, int(z_sub.size(0) * self.proto_topk_ratio))
        k = min(k, z_sub.size(0))
        topk_idx = torch.topk(dist, k=k, largest=False).indices
        proto = z_sub[topk_idx].mean(dim=0)
        return proto

    def compute_prototypes(self, z, pos_idx, neg_idx):
        if len(pos_idx) == 0:
            p_pos = torch.zeros(z.size(1), device=z.device)
        else:
            p_pos = self._topk_center_proto(z[pos_idx])

        if len(neg_idx) == 0:
            p_neg = torch.zeros(z.size(1), device=z.device)
        else:
            p_neg = self._topk_center_proto(z[neg_idx])

        return p_pos, p_neg

    def classify_with_prototypes(self, z, p_pos, p_neg):
        logits_mlp = self.cls_head(z).squeeze(-1)

        d_pos = torch.sum((z - p_pos.unsqueeze(0)) ** 2, dim=1)
        d_neg = torch.sum((z - p_neg.unsqueeze(0)) ** 2, dim=1)

        proto_logit = (d_neg - d_pos) / (z.size(1) ** 0.5)
        logits = logits_mlp + self.proto_alpha * proto_logit
        return logits, logits_mlp, proto_logit

    def forward(self, z, pos_idx, neg_idx):
        p_pos, p_neg = self.compute_prototypes(z, pos_idx, neg_idx)
        logits, logits_mlp, proto_logit = self.classify_with_prototypes(z, p_pos, p_neg)

        return {
            'logits': logits,
            'logits_mlp': logits_mlp,
            'proto_logit': proto_logit,
            'p_pos': p_pos,
            'p_neg': p_neg
        }


class DriverGeneFewShotModel(nn.Module):
    def __init__(
        self,
        in_dim,
        pos_dim,
        hidden_dim=128,
        embed_dim=128,
        num_layers=2,
        dropout=0.2,
        view_names=('ppi', 'path', 'go'),
        proto_alpha=0.25,
        proto_topk_ratio=0.6,
        min_proto_k=8
    ):
        super().__init__()
        self.view_names = list(view_names)

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.spec_encoders = nn.ModuleDict({
            v: GCNEncoder(
                in_dim=hidden_dim,
                pos_dim=pos_dim,
                hidden_dim=hidden_dim,
                out_dim=embed_dim // 2,
                num_layers=num_layers,
                dropout=dropout
            ) for v in self.view_names
        })

        self.cons_encoder = GCNEncoder(
            in_dim=hidden_dim,
            pos_dim=pos_dim,
            hidden_dim=hidden_dim,
            out_dim=embed_dim // 2,
            num_layers=num_layers,
            dropout=dropout
        )

        self.post_view_proj = nn.ModuleDict({
            v: nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for v in self.view_names
        })

        self.fusion_gate = ViewFusionGate(
            view_names=self.view_names,
            in_dim=embed_dim,
            gate_hidden_dim=hidden_dim // 2
        )

        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim)
        )

        self.proto_head = PrototypeHead(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            proto_alpha=proto_alpha,
            proto_topk_ratio=proto_topk_ratio,
            min_proto_k=min_proto_k
        )

    def encode_one_view(self, x0, pos_feat, edge_index, edge_weight, view_name):
        z_spec = self.spec_encoders[view_name](x0, pos_feat, edge_index, edge_weight)
        z_cons = self.cons_encoder(x0, pos_feat, edge_index, edge_weight)
        z = torch.cat([z_spec, z_cons], dim=1)
        z = self.post_view_proj[view_name](z)
        return z, z_spec, z_cons

    def forward(self, x, pos_feat, edge_index_dict, edge_weight_dict, pos_idx, neg_idx):
        x0 = self.input_proj(x)

        view_embeddings = {}
        spec_embeddings = {}
        cons_embeddings = {}

        for v in self.view_names:
            z_v, z_spec, z_cons = self.encode_one_view(
                x0, pos_feat, edge_index_dict[v], edge_weight_dict[v], v
            )
            view_embeddings[v] = z_v
            spec_embeddings[v] = z_spec
            cons_embeddings[v] = z_cons

        fused_z, gate_weights = self.fusion_gate(view_embeddings)
        fused_z = self.fusion_proj(fused_z)

        proto_out = self.proto_head(fused_z, pos_idx=pos_idx, neg_idx=neg_idx)

        out = {
            'embedding': fused_z,
            'prob': torch.sigmoid(proto_out['logits']),
            'gate_weights': gate_weights,
            'view_embeddings': view_embeddings,
            'spec_embeddings': spec_embeddings,
            'cons_embeddings': cons_embeddings,
            'logits': proto_out['logits'],
            'logits_mlp': proto_out['logits_mlp'],
            'proto_logit': proto_out['proto_logit'],
            'p_pos': proto_out['p_pos'],
            'p_neg': proto_out['p_neg'],
        }
        return out

    def classify_embeddings(self, z, p_pos, p_neg):
        logits, logits_mlp, proto_logit = self.proto_head.classify_with_prototypes(z, p_pos, p_neg)
        return logits, logits_mlp, proto_logit
