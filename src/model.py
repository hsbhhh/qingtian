import torch
import torch.nn as nn
import torch.nn.functional as F


def build_sparse_adj(row, col, score, num_nodes, device, add_self_loop=True):
    row = row.to(device)
    col = col.to(device)
    score = score.to(device).float()

    if add_self_loop:
        self_loop = torch.arange(num_nodes, device=device)
        row = torch.cat([row, self_loop], dim=0)
        col = torch.cat([col, self_loop], dim=0)
        score = torch.cat([score, torch.ones(num_nodes, device=device)], dim=0)

    indices = torch.stack([row, col], dim=0)
    adj = torch.sparse_coo_tensor(indices, score, (num_nodes, num_nodes), device=device)
    adj = adj.coalesce()

    # D^{-1/2} A D^{-1/2}
    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-12), -0.5)

    row_idx, col_idx = adj.indices()
    val = adj.values() * deg_inv_sqrt[row_idx] * deg_inv_sqrt[col_idx]

    norm_adj = torch.sparse_coo_tensor(
        adj.indices(), val, adj.shape, device=device
    ).coalesce()

    return norm_adj


class GraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x, adj):
        x = self.linear(x)
        x = torch.sparse.mm(adj, x)
        return x


class ViewGCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim1,hidden_dim2, out_dim, dropout=0.2, negative_slope=0.2):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hidden_dim1)
        self.gc2 = GraphConvolution(hidden_dim1,hidden_dim2)
        self.gc3 = GraphConvolution(hidden_dim2, out_dim)

        self.dropout = dropout
        self.negative_slope = negative_slope

    def forward(self, x, adj):
        h = self.gc1(x, adj)
        h = F.leaky_relu(h, negative_slope=self.negative_slope)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.gc2(h, adj)
        h = F.leaky_relu(h, negative_slope=self.negative_slope)
        h = F.dropout(h, p=self.dropout, training=self.training)

        z = self.gc3(h, adj)
        return z


class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(in_dim // 2, 16)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # [N]


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=None):
        super().__init__()
        if proj_dim is None:
            proj_dim = in_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim)
        )

    def forward(self, x):
        z = self.net(x)
        z = F.normalize(z, p=2, dim=-1)
        return z


class MultiViewContrastiveModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim1,
        hidden_dim2,
        embed_dim,
        view_names=("ppi", "path", "go"),
        dropout=0.2
    ):
        super().__init__()
        self.view_names = list(view_names)
        self.num_views = len(self.view_names)

        self.encoders = nn.ModuleDict({
            view: ViewGCNEncoder(input_dim, hidden_dim1, hidden_dim2,embed_dim, dropout)
            for view in self.view_names
        })

        self.view_classifiers = nn.ModuleDict({
            view: MLPClassifier(embed_dim)
            for view in self.view_names
        })

        self.projectors = nn.ModuleDict({
            view: ProjectionHead(embed_dim, embed_dim)
            for view in self.view_names
        })

        # 注意力融合：输入是拼接后的所有视图表示
        self.att_mlp = nn.Sequential(
            nn.Linear(self.num_views * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, self.num_views)
        )

        self.fusion_classifier = MLPClassifier(embed_dim)

    def forward(self, node_features, edge_indices_with_score):
        device = node_features.device
        num_nodes = node_features.size(0)

        view_embeddings = {}
        view_logits = {}
        proj_embeddings = {}

        for view in self.view_names:
            row, col, score = edge_indices_with_score[view]
            adj = build_sparse_adj(row, col, score, num_nodes, device)

            z = self.encoders[view](node_features, adj)     # [N, D]
            logit = self.view_classifiers[view](z)          # [N]
            proj = self.projectors[view](z)                 # [N, D]

            view_embeddings[view] = z
            view_logits[view] = logit
            proj_embeddings[view] = proj

        # [N, V, D]
        stacked_views = torch.stack(
            [view_embeddings[v] for v in self.view_names], dim=1
        )

        # [N, V*D]
        concat_views = torch.cat(
            [view_embeddings[v] for v in self.view_names], dim=-1
        )

        # [N, V]
        att_logits = self.att_mlp(concat_views)
        att_weights = F.softmax(att_logits, dim=-1)

        # 加权融合 [N, D]
        fused_embedding = torch.sum(
            stacked_views * att_weights.unsqueeze(-1), dim=1
        )

        fused_logit = self.fusion_classifier(fused_embedding)  # [N]

        return {
            "view_embeddings": view_embeddings,
            "proj_embeddings": proj_embeddings,
            "view_logits": view_logits,
            "fused_embedding": fused_embedding,
            "fused_logit": fused_logit,
            "att_weights": att_weights
        }