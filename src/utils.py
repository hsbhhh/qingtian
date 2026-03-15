# utils.py
import os
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from collections import deque
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_edge_data_with_score(sparse_tensor):
    sparse_tensor = sparse_tensor.coalesce()
    row, col = sparse_tensor.indices()
    score = sparse_tensor.values().float()
    return row, col, score


def load_label_single(path, cancerType, device):
    label = np.loadtxt(os.path.join(path, f"label_file-P-{cancerType}.txt"))
    Y = torch.tensor(label, dtype=torch.float32, device=device)

    label_pos = np.loadtxt(os.path.join(path, f"pos-{cancerType}.txt"), dtype=int)
    label_neg = np.loadtxt(os.path.join(path, "pan-neg.txt"), dtype=int)

    label_pos = np.atleast_1d(label_pos).astype(int)
    label_neg = np.atleast_1d(label_neg).astype(int)
    return Y, label_pos, label_neg


def stratified_kfold_split(pos_label, neg_label, total_nodes, n_splits=5, seed=1234, val_ratio=0.125):
    pos_label = np.array(pos_label, dtype=int)
    neg_label = np.array(neg_label, dtype=int)

    labeled_idx = np.concatenate([pos_label, neg_label], axis=0)
    labeled_y = np.concatenate([
        np.ones(len(pos_label), dtype=int),
        np.zeros(len(neg_label), dtype=int)
    ], axis=0)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []

    for train_val_idx, test_idx_local in skf.split(labeled_idx, labeled_y):
        train_val_nodes = labeled_idx[train_val_idx]
        train_val_y = labeled_y[train_val_idx]
        test_nodes = labeled_idx[test_idx_local]

        inner_n_splits = max(int(round(1.0 / val_ratio)), 2)
        skf_inner = StratifiedKFold(n_splits=inner_n_splits, shuffle=True, random_state=seed)
        train_idx_local, val_idx_local = next(skf_inner.split(train_val_nodes, train_val_y))

        train_nodes = train_val_nodes[train_idx_local]
        val_nodes = train_val_nodes[val_idx_local]

        train_mask = torch.zeros(total_nodes, dtype=torch.bool)
        val_mask = torch.zeros(total_nodes, dtype=torch.bool)
        test_mask = torch.zeros(total_nodes, dtype=torch.bool)

        train_mask[torch.tensor(train_nodes, dtype=torch.long)] = True
        val_mask[torch.tensor(val_nodes, dtype=torch.long)] = True
        test_mask[torch.tensor(test_nodes, dtype=torch.long)] = True

        folds.append((
            sorted(train_nodes.tolist()),
            sorted(val_nodes.tolist()),
            sorted(test_nodes.tolist()),
            train_mask,
            val_mask,
            test_mask
        ))
    return folds


def build_edge_weight_dict(edge_indices_with_score, device):
    edge_index_dict = {}
    edge_weight_dict = {}

    for key, (row, col, score) in edge_indices_with_score.items():
        edge_index = torch.stack([row, col], dim=0).long().to(device)
        edge_weight = score.float()

        # 稍微稳一点，防止极端边权
        if edge_weight.numel() > 0:
            max_val = torch.quantile(edge_weight, 0.99).item()
            edge_weight = torch.clamp(edge_weight, min=0.0, max=max(1e-6, max_val))

        edge_index_dict[key] = edge_index
        edge_weight_dict[key] = edge_weight.to(device)

    return edge_index_dict, edge_weight_dict


def sparse_tensor_to_scipy(sparse_tensor):
    sparse_tensor = sparse_tensor.coalesce().cpu()
    row, col = sparse_tensor.indices().numpy()
    val = sparse_tensor.values().numpy()
    n = sparse_tensor.size(0)
    return sp.coo_matrix((val, (row, col)), shape=(n, n)).tocsr()


def normalize_adj_scipy(adj):
    row_sum = np.array(adj.sum(1)).flatten()
    row_sum[row_sum == 0] = 1.0
    inv = 1.0 / row_sum
    D_inv = sp.diags(inv)
    return D_inv @ adj


def compute_diffusion_matrix_from_ppi(ppi_sparse_tensor, alpha=0.15, topk=50):
    """
    PPR diffusion，作为额外 view（可选）
    """
    A = sparse_tensor_to_scipy(ppi_sparse_tensor).astype(np.float32)
    n = A.shape[0]

    A = ((A + A.T) > 0).astype(np.float32).multiply((A + A.T) / 2.0)
    A = A.tocsr()

    P = normalize_adj_scipy(A)
    I = sp.eye(n, dtype=np.float32, format='csr')
    M = I - (1.0 - alpha) * P
    S = alpha * sp.linalg.inv(M)
    S = S.tocsr()

    rows, cols, vals = [], [], []
    for i in range(n):
        row = S.getrow(i)
        if row.nnz == 0:
            continue
        data = row.data
        indices = row.indices
        if len(data) > topk:
            top_idx = np.argpartition(data, -topk)[-topk:]
            data = data[top_idx]
            indices = indices[top_idx]
        rows.extend([i] * len(indices))
        cols.extend(indices.tolist())
        vals.extend(data.tolist())

    S_topk = sp.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
    S_topk = ((S_topk + S_topk.T) / 2.0).tocsr()

    coo = S_topk.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def compute_graph_structural_features(ppi_sparse_tensor, pos_idx):
    """
    输出结构先验特征：
    degree, pagerank, min_dist_to_pos, mean_dist_to_pos
    """
    A = sparse_tensor_to_scipy(ppi_sparse_tensor).astype(np.float32)
    n = A.shape[0]

    G = ((A + A.T) > 0).astype(np.int32).tocsr()
    neighbors = [G[i].indices.tolist() for i in range(n)]

    # degree
    degree = np.array(G.sum(axis=1)).flatten().astype(np.float32)
    degree = np.log1p(degree)
    degree = (degree - degree.mean()) / (degree.std() + 1e-8)
    degree = degree.reshape(-1, 1)

    # pagerank
    P = normalize_adj_scipy(G.astype(np.float32))
    pr = np.ones(n, dtype=np.float32) / n
    beta = 0.85
    teleport = np.ones(n, dtype=np.float32) / n
    for _ in range(50):
        pr = beta * (P.T @ pr) + (1 - beta) * teleport
    pr = (pr - pr.mean()) / (pr.std() + 1e-8)
    pr = pr.reshape(-1, 1)

    # min shortest distance to positives
    pos_idx = list(map(int, np.atleast_1d(pos_idx).tolist()))
    INF = 10 ** 9
    min_dist = np.full(n, INF, dtype=np.float32)
    dq = deque()
    for s in pos_idx:
        min_dist[s] = 0
        dq.append(s)

    while dq:
        u = dq.popleft()
        for v in neighbors[u]:
            if min_dist[v] > min_dist[u] + 1:
                min_dist[v] = min_dist[u] + 1
                dq.append(v)

    reachable = min_dist < INF
    max_reach = min_dist[reachable].max() if np.any(reachable) else 10.0
    min_dist[~reachable] = max_reach + 1
    min_dist = min_dist / (min_dist.max() + 1e-8)

    # mean shortest distance to sampled positives
    sample_pos = pos_idx[: min(len(pos_idx), 32)]
    dist_list = []
    for s in sample_pos:
        dist = np.full(n, INF, dtype=np.float32)
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in neighbors[u]:
                if dist[v] > dist[u] + 1:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        dist[dist >= INF] = max_reach + 1
        dist_list.append(dist)

    if len(dist_list) > 0:
        mean_dist = np.mean(np.stack(dist_list, axis=1), axis=1)
    else:
        mean_dist = np.ones(n, dtype=np.float32) * (max_reach + 1)
    mean_dist = mean_dist / (mean_dist.max() + 1e-8)

    feats = np.concatenate([
        degree,
        pr,
        min_dist.reshape(-1, 1),
        mean_dist.reshape(-1, 1)
    ], axis=1)

    return torch.tensor(feats, dtype=torch.float32)


def mine_hard_negatives(embeddings, train_pos_idx, train_neg_idx, topk=64):
    if len(train_pos_idx) == 0 or len(train_neg_idx) == 0:
        return []

    with torch.no_grad():
        z_pos = embeddings[train_pos_idx]
        z_neg = embeddings[train_neg_idx]

        p_pos = z_pos.mean(dim=0, keepdim=True)
        dist = torch.sum((z_neg - p_pos) ** 2, dim=1)

        k = min(topk, z_neg.size(0))
        hard_local = torch.topk(dist, k=k, largest=False).indices
        hard_global = [train_neg_idx[i] for i in hard_local.cpu().numpy().tolist()]
    return hard_global


def sample_alignment_nodes(train_pos_idx, train_neg_idx, hard_neg_idx=None, max_nodes=256):
    pos_part = train_pos_idx[: min(len(train_pos_idx), max_nodes // 4)]
    neg_budget = max_nodes - len(pos_part)

    neg_candidates = list(train_neg_idx)
    if hard_neg_idx is not None:
        hard_part = hard_neg_idx[: min(len(hard_neg_idx), neg_budget // 2)]
    else:
        hard_part = []

    remain = neg_budget - len(hard_part)
    random.shuffle(neg_candidates)
    rand_neg = neg_candidates[: max(0, remain)]

    sampled = list(set(pos_part + hard_part + rand_neg))
    if len(sampled) > max_nodes:
        sampled = sampled[:max_nodes]
    return sampled


def generate_positive_mixup_embeddings(embeddings, pos_idx, alpha=0.4, num_mix=32):
    if len(pos_idx) < 2:
        return None

    pos_z = embeddings[pos_idx]
    n_pos = pos_z.size(0)
    device = pos_z.device

    idx1 = torch.randint(0, n_pos, (num_mix,), device=device)
    idx2 = torch.randint(0, n_pos, (num_mix,), device=device)

    lam = np.random.beta(alpha, alpha, size=(num_mix, 1)).astype(np.float32)
    lam = torch.tensor(lam, device=device)

    z1 = pos_z[idx1]
    z2 = pos_z[idx2]
    z_mix = lam * z1 + (1.0 - lam) * z2
    return z_mix


def build_hard_positive_weight(train_pos_idx, prob, total_nodes, low_prob_threshold=0.7, extra_weight=1.5):
    weights = torch.ones(total_nodes, dtype=torch.float32, device=prob.device)
    if len(train_pos_idx) == 0:
        return weights, []

    pos_prob = prob[train_pos_idx]
    hard_mask = pos_prob < low_prob_threshold
    hard_pos_idx = [train_pos_idx[i] for i in torch.where(hard_mask)[0].cpu().numpy().tolist()]
    if len(hard_pos_idx) > 0:
        weights[hard_pos_idx] = extra_weight
    return weights, hard_pos_idx


def find_best_f1_threshold(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    thresholds = np.unique(y_score)
    if thresholds.size == 0:
        return 0.5

    best_thr, best_f1 = 0.5, -1.0
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        try:
            f1 = f1_score(y_true, y_pred)
        except Exception:
            f1 = -1.0
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def compute_metrics(y_true, y_score, threshold=0.5, dynamic_f1=False):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    if dynamic_f1:
        threshold = find_best_f1_threshold(y_true, y_score)

    y_pred = (y_score >= threshold).astype(int)

    metrics = {}
    try:
        metrics['AUC'] = roc_auc_score(y_true, y_score)
    except Exception:
        metrics['AUC'] = np.nan

    try:
        metrics['AUPR'] = average_precision_score(y_true, y_score)
    except Exception:
        metrics['AUPR'] = np.nan

    try:
        metrics['F1'] = f1_score(y_true, y_pred)
    except Exception:
        metrics['F1'] = np.nan

    try:
        metrics['ACC'] = accuracy_score(y_true, y_pred)
    except Exception:
        metrics['ACC'] = np.nan

    try:
        metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    except Exception:
        metrics['Precision'] = np.nan

    try:
        metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
    except Exception:
        metrics['Recall'] = np.nan

    metrics['Threshold'] = float(threshold)
    return metrics


def summarize_metrics(metric_list):
    if len(metric_list) == 0:
        return {}
    keys = metric_list[0].keys()
    summary = {}
    for k in keys:
        vals = [m[k] for m in metric_list if k in m and not np.isnan(m[k])]
        if len(vals) == 0:
            summary[k] = (np.nan, np.nan)
        else:
            summary[k] = (float(np.mean(vals)), float(np.std(vals)))
    return summary


def format_metric_dict(metrics):
    parts = []
    for k, v in metrics.items():
        if isinstance(v, (float, int, np.floating, np.integer)):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


def format_metric_summary(summary):
    parts = []
    for k, value in summary.items():
        if isinstance(value, tuple) and len(value) == 2:
            mean_v, std_v = value
            if np.isnan(mean_v):
                parts.append(f"{k}=nan")
            else:
                parts.append(f"{k}={mean_v:.4f}±{std_v:.4f}")
        else:
            parts.append(f"{k}={value}")
    return " ".join(parts)


class EarlyStopping:
    def __init__(self, patience=30, mode='max'):
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_state = None
        self.should_stop = False

    def step(self, score, model):
        improved = False
        if self.best_score is None:
            improved = True
        elif self.mode == 'max' and score > self.best_score:
            improved = True
        elif self.mode == 'min' and score < self.best_score:
            improved = True

        if improved:
            self.best_score = score
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved

    def restore(self, model, device):
        if self.best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in self.best_state.items()})
        return model
