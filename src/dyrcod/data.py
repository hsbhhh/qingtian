from __future__ import annotations

import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse
import scipy.sparse as sp
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .config import resolve_path


POS_CONTEXT_FEATURE_NAMES = (
    "pagerank",
    "degree",
    "mean_distance_to_positive",
    "shortest_distance_to_positive",
)


@dataclass
class DyRCoDData:
    cancer: str
    genes: list[str]
    x: torch.Tensor
    y: torch.Tensor
    label_pos: np.ndarray
    label_neg: np.ndarray
    edge_index_dict: dict[str, torch.Tensor]
    edge_weight_dict: dict[str, torch.Tensor]
    adj_cpu_dict: dict[str, Any]
    view_names: list[str]
    feature_columns: list[str]


def _feature_path(config: Mapping[str, Any], key: str, cancer: str) -> Path:
    feature_dir = resolve_path(config["data"]["feature_dir"])
    pattern = config["data"][key].format(cancer=cancer.upper(), cancer_lower=cancer.lower())
    return feature_dir / pattern


def _label_dir(config: Mapping[str, Any]) -> Path:
    data_cfg = config["data"]
    return resolve_path(data_cfg["label_dir"]) / data_cfg.get("label_subdir", "")


def load_gene_table(config: Mapping[str, Any]) -> pd.DataFrame:
    table_path = resolve_path(config["data"]["feature_dir"]) / config["data"]["gene_table"]
    data_x_df = pd.read_csv(table_path, sep="\t", index_col=0).dropna()
    data_x_df.index = data_x_df.index.astype(str).str.strip().str.upper()
    return data_x_df


def _normalize_gene_df(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    if "gene" not in df.columns:
        raise ValueError(f"'gene' column not found in {df_name}")
    df = df.copy()
    df["gene"] = df["gene"].astype(str).str.strip().str.upper()
    return df.drop_duplicates(subset=["gene"]).set_index("gene")


def prepare_feature_matrix(
    config: Mapping[str, Any],
    cancer: str,
    device: torch.device,
) -> tuple[torch.Tensor, list[str], list[str]]:
    cancer = cancer.upper()
    data_x_df = load_gene_table(config)
    target_genes = pd.Index(data_x_df.index.astype(str).str.strip().str.upper(), name="gene")
    final_df = pd.DataFrame(index=target_genes)

    mutation_path = _feature_path(config, "mutation_pattern", cancer)
    cnv_path = _feature_path(config, "cnv_pattern", cancer)
    expression_path = _feature_path(config, "expression_pattern", cancer)
    crispr_path = _feature_path(config, "crispr_pattern", cancer)
    spatial_path = _feature_path(config, "spatial_file", cancer)

    for path in [mutation_path, cnv_path, expression_path, crispr_path, spatial_path]:
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")

    mut_df = _normalize_gene_df(pd.read_csv(mutation_path), "mutation_feature_file").add_prefix("mut_")
    cnv_df = _normalize_gene_df(pd.read_csv(cnv_path), "cnv_feature_file").add_prefix("cnv_")
    exp_df = _normalize_gene_df(pd.read_csv(expression_path), "expression_feature_file").add_prefix("exp_")

    final_df = final_df.join(mut_df, how="left")
    final_df = final_df.join(cnv_df, how="left")
    final_df = final_df.join(exp_df, how="left")

    crispr_df = pd.read_csv(crispr_path)
    crispr_df.columns = ["gene", "avg"]
    crispr_df["gene"] = crispr_df["gene"].astype(str).str.strip().str.upper()
    crispr_df = crispr_df.drop_duplicates(subset=["gene"]).set_index("gene")
    crispr_df.columns = ["crispr_avg"]
    crispr_df = crispr_df.reindex(final_df.index)

    spatial_df = pd.read_csv(spatial_path)
    if "chr_id" in spatial_df.columns:
        spatial_df = spatial_df.drop(columns=["chr_id"])
    spatial_df["gene"] = spatial_df["gene"].astype(str).str.strip().str.upper()
    spatial_df = spatial_df.drop_duplicates(subset=["gene"]).set_index("gene")
    spatial_df.columns = [f"spatial_{column}" for column in spatial_df.columns]
    spatial_df = spatial_df.reindex(final_df.index)

    final_df = pd.concat([final_df, crispr_df, spatial_df], axis=1)
    final_df = final_df.apply(pd.to_numeric, errors="coerce")
    final_df = final_df.fillna(final_df.median(numeric_only=True)).fillna(0.0)
    final_df = pd.DataFrame(
        StandardScaler().fit_transform(final_df.values),
        index=final_df.index,
        columns=final_df.columns,
    )
    return (
        torch.tensor(final_df.values, dtype=torch.float32, device=device),
        final_df.index.astype(str).tolist(),
        final_df.columns.astype(str).tolist(),
    )


def load_adjacency(path: Union[str, Path]):
    path = Path(path)
    try:
        try:
            return torch.load(path, map_location=torch.device("cpu"), weights_only=False)
        except TypeError:
            return torch.load(path, map_location=torch.device("cpu"))
    except Exception:
        with path.open("rb") as handle:
            return pickle.load(handle)


def adjacency_matrix(obj: Any):
    if isinstance(obj, dict):
        if "adj" not in obj:
            raise KeyError("Adjacency dictionary does not contain key 'adj'.")
        return obj["adj"]
    return obj


def extract_edge_data_with_score(adj: Any, device: Optional[torch.device] = None):
    adj = adjacency_matrix(adj)
    if sparse.issparse(adj):
        coo = adj.tocoo()
        row = torch.from_numpy(coo.row.astype(np.int64)).long()
        col = torch.from_numpy(coo.col.astype(np.int64)).long()
        score = torch.from_numpy(coo.data.astype(np.float32)).float()
    elif torch.is_tensor(adj) and adj.is_sparse:
        coalesced = adj.coalesce()
        indices = coalesced.indices()
        values = coalesced.values()
        row = indices[0].long()
        col = indices[1].long()
        score = values.float()
    elif torch.is_tensor(adj):
        row, col = torch.nonzero(adj, as_tuple=True)
        score = adj[row, col].float()
    else:
        raise TypeError(f"Unsupported adjacency type: {type(adj)}")

    if device is not None:
        row = row.to(device)
        col = col.to(device)
        score = score.to(device)
    return row, col, score


def build_edge_weight_dict(
    edge_indices_with_score: Mapping[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    edge_index_dict = {}
    edge_weight_dict = {}

    for key, (row, col, score) in edge_indices_with_score.items():
        edge_index_dict[key] = torch.stack([row, col], dim=0).long().to(device)
        edge_weight = score.float()
        if edge_weight.numel() > 0:
            max_val = torch.quantile(edge_weight, 0.99).item()
            edge_weight = torch.clamp(edge_weight, min=0.0, max=max(1e-6, max_val))
        edge_weight_dict[key] = edge_weight.to(device)
    return edge_index_dict, edge_weight_dict


def load_networks(
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any], list[str]]:
    data_cfg = config["data"]
    network_dir = resolve_path(data_cfg["network_dir"])
    view_names = list(config["model"]["views"])
    edge_data = {}
    adj_cpu_dict = {}
    for view in view_names:
        path = network_dir / data_cfg["networks"][view]
        adj = load_adjacency(path)
        row, col, score = extract_edge_data_with_score(adj)
        scale = float(data_cfg.get("network_weight_scale", {}).get(view, 1.0))
        edge_data[view] = (row, col, score * scale)
        adj_cpu_dict[view] = adj
    edge_index_dict, edge_weight_dict = build_edge_weight_dict(edge_data, device=device)
    return edge_index_dict, edge_weight_dict, adj_cpu_dict, view_names


def load_labels(
    config: Mapping[str, Any],
    cancer: str,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    cancer_lower = cancer.lower()
    data_cfg = config["data"]
    label_dir = _label_dir(config)
    label_path = label_dir / data_cfg["label_pattern"].format(cancer_lower=cancer_lower, cancer=cancer.upper())
    pos_path = label_dir / data_cfg["positive_pattern"].format(cancer_lower=cancer_lower, cancer=cancer.upper())
    neg_path = label_dir / data_cfg["negative_file"]
    label = np.loadtxt(label_path)
    y = torch.tensor(label, dtype=torch.float32, device=device)
    label_pos = np.atleast_1d(np.loadtxt(pos_path, dtype=int)).astype(int)
    label_neg = np.atleast_1d(np.loadtxt(neg_path, dtype=int)).astype(int)
    return y, label_pos, label_neg


def load_dataset(config: Mapping[str, Any], cancer: str, device: torch.device) -> DyRCoDData:
    x, genes, feature_columns = prepare_feature_matrix(config, cancer, device)
    y, label_pos, label_neg = load_labels(config, cancer, device)
    edge_index_dict, edge_weight_dict, adj_cpu_dict, view_names = load_networks(config, device)
    if len(y) != x.size(0):
        raise ValueError(f"Label length {len(y)} does not match feature rows {x.size(0)}")
    return DyRCoDData(
        cancer=cancer.upper(),
        genes=genes,
        x=x,
        y=y,
        label_pos=label_pos,
        label_neg=label_neg,
        edge_index_dict=edge_index_dict,
        edge_weight_dict=edge_weight_dict,
        adj_cpu_dict=adj_cpu_dict,
        view_names=view_names,
        feature_columns=feature_columns,
    )


def sparse_tensor_to_scipy(adj: Any) -> sp.csr_matrix:
    adj = adjacency_matrix(adj)
    if sparse.issparse(adj):
        return adj.tocsr()
    if torch.is_tensor(adj) and adj.is_sparse:
        adj = adj.coalesce().cpu()
        indices = adj.indices().numpy()
        values = adj.values().numpy()
        return sparse.coo_matrix((values, (indices[0], indices[1])), shape=adj.shape).tocsr()
    if torch.is_tensor(adj):
        return sparse.csr_matrix(adj.detach().cpu().numpy())
    if isinstance(adj, np.ndarray):
        return sparse.csr_matrix(adj)
    raise TypeError(f"Unsupported adjacency type: {type(adj)}")


def normalize_adj_scipy(adj: sp.csr_matrix) -> sp.csr_matrix:
    row_sum = np.array(adj.sum(1)).flatten()
    row_sum[row_sum == 0] = 1.0
    return sp.diags(1.0 / row_sum) @ adj


def _single_graph_structural_features(adj_tensor: Any, pos_idx: list[int]) -> np.ndarray:
    adj = sparse_tensor_to_scipy(adj_tensor).astype(np.float32)
    n_nodes = adj.shape[0]

    graph = ((adj + adj.T) > 0).astype(np.int32).tocsr()
    neighbors = [graph[i].indices.tolist() for i in range(n_nodes)]

    degree = np.array(graph.sum(axis=1)).flatten().astype(np.float32)
    degree = np.log1p(degree)
    degree = (degree - degree.mean()) / (degree.std() + 1e-8)

    transition = normalize_adj_scipy(graph.astype(np.float32))
    pagerank = np.ones(n_nodes, dtype=np.float32) / n_nodes
    teleport = np.ones(n_nodes, dtype=np.float32) / n_nodes
    beta = 0.85
    for _ in range(50):
        pagerank = beta * (transition.T @ pagerank) + (1 - beta) * teleport
    pagerank = (pagerank - pagerank.mean()) / (pagerank.std() + 1e-8)

    pos_idx = list(map(int, np.atleast_1d(pos_idx).tolist()))
    inf = 10**9
    min_dist = np.full(n_nodes, inf, dtype=np.float32)
    queue: deque[int] = deque()

    for src in pos_idx:
        min_dist[src] = 0
        queue.append(src)

    while queue:
        node = queue.popleft()
        for neighbor in neighbors[node]:
            if min_dist[neighbor] > min_dist[node] + 1:
                min_dist[neighbor] = min_dist[node] + 1
                queue.append(neighbor)

    reachable = min_dist < inf
    max_reach = min_dist[reachable].max() if np.any(reachable) else 10.0
    min_dist[~reachable] = max_reach + 1
    min_dist = min_dist / (min_dist.max() + 1e-8)

    sampled_pos = pos_idx[: min(len(pos_idx), 32)]
    dist_list = []
    for src in sampled_pos:
        dist = np.full(n_nodes, inf, dtype=np.float32)
        dist[src] = 0
        queue = deque([src])
        while queue:
            node = queue.popleft()
            for neighbor in neighbors[node]:
                if dist[neighbor] > dist[node] + 1:
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)
        dist[dist >= inf] = max_reach + 1
        dist_list.append(dist)

    if dist_list:
        mean_dist = np.mean(np.stack(dist_list, axis=1), axis=1)
    else:
        mean_dist = np.ones(n_nodes, dtype=np.float32) * (max_reach + 1)
    mean_dist = mean_dist / (mean_dist.max() + 1e-8)

    feature_values = {
        "pagerank": pagerank,
        "degree": degree,
        "mean_distance_to_positive": mean_dist,
        "shortest_distance_to_positive": min_dist,
    }
    feats = np.stack([feature_values[name] for name in POS_CONTEXT_FEATURE_NAMES], axis=1)
    return feats.astype(np.float32)


def compute_multiview_structural_features(
    adj_dict: Mapping[str, Any],
    pos_idx: list[int],
    view_names: list[str],
) -> torch.Tensor:
    ordered = [_single_graph_structural_features(adj_dict[view], pos_idx) for view in view_names if view in adj_dict]
    return torch.tensor(np.concatenate(ordered, axis=1), dtype=torch.float32)


def get_train_pos_idx(train_mask: torch.Tensor, y: torch.Tensor) -> list[int]:
    idx = torch.where(train_mask)[0].detach().cpu().numpy().tolist()
    return [int(i) for i in idx if float(y[i].item()) > 0.5]


def build_pos_feat(
    adj_cpu_dict: Mapping[str, Any],
    train_pos_idx: list[int],
    view_names: list[str],
    device: torch.device,
) -> torch.Tensor:
    pos_feat = compute_multiview_structural_features(adj_cpu_dict, train_pos_idx, view_names)
    return pos_feat.to(device=device, dtype=torch.float32)


def stratified_kfold_split(
    pos_label,
    neg_label,
    total_nodes: int,
    n_splits: int = 5,
    seed: int = 1234,
    val_ratio: float = 0.2,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    pos_label = np.array(pos_label, dtype=int)
    neg_label = np.array(neg_label, dtype=int)
    labeled_idx = np.concatenate([pos_label, neg_label], axis=0)
    labeled_y = np.concatenate([np.ones(len(pos_label), dtype=int), np.zeros(len(neg_label), dtype=int)], axis=0)

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
        folds.append((train_mask, val_mask, test_mask))
    return folds


def all_labeled_mask(label_pos: np.ndarray, label_neg: np.ndarray, total_nodes: int, device: torch.device) -> torch.Tensor:
    train_idx = sorted(set(map(int, label_pos.tolist())) | set(map(int, label_neg.tolist())))
    mask = torch.zeros(total_nodes, dtype=torch.bool, device=device)
    mask[torch.tensor(train_idx, dtype=torch.long, device=device)] = True
    return mask


def label_types(total_nodes: int, label_pos: np.ndarray, label_neg: np.ndarray) -> np.ndarray:
    labels = np.full(total_nodes, "Unlabeled", dtype=object)
    labels[np.asarray(label_pos, dtype=int)] = "Driver"
    labels[np.asarray(label_neg, dtype=int)] = "Negative"
    return labels
