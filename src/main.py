import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import argparse
import random
import gc
import copy

from model import MultiViewContrastiveModel
from loss import Loss_fun
from utils import (
    extract_edge_data_with_score,
    load_label_single,
    stratified_kfold_split,
    compute_metrics,
    set_seed
)

# =========================
# Hyperparameter and setting
# =========================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='STRING', help='dataset (CPDB, STRING)')
parser.add_argument('--cancerType', type=str, default='luad', help='Types of cancer (pan-cancer, kirc, luad...)')
parser.add_argument('--embed_dim', type=int, default=256, help='embedding dimension')
parser.add_argument('--hidden_dim1', type=int, default=128, help='hidden dimension')
parser.add_argument('--hidden_dim2', type=int, default=64, help='hidden dimension')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--device', type=int, default=0, help='GPU device ID (if available)')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
args = parser.parse_args()

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)

# =========================
# Data input
# =========================
dataPath = "./Data"

data_x_df = pd.read_csv(dataPath + f'/multiomics_features_{args.dataset}.tsv', sep='\t', index_col=0)
data_x_df = data_x_df.dropna()

scaler = StandardScaler()
features_scaled = scaler.fit_transform(data_x_df.values)
data_x = torch.tensor(features_scaled, dtype=torch.float32, device=device)

cancerType = args.cancerType.lower()

if cancerType == 'pan-cancer':
    data_x = data_x[:, :48]
    print("--- [INFO] Loading hyperparameters for Pan-Cancer ---")
    learning_rate = 0.001
    epochs = 30
    num_heads = 4   # 目前未使用，先保留
    num_layers = 3  # 目前未使用，先保留
    dropout = 0.1
else:
    cancerType_dict = {
        'kirc': [0, 16, 32],
        'brca': [1, 17, 33],
        'prad': [3, 19, 35],
        'stad': [4, 20, 36],
        'hnsc': [5, 21, 37],
        'luad': [6, 22, 38],
        'thca': [7, 23, 39],
        'blca': [8, 24, 40],
        'esca': [9, 25, 41],
        'lihc': [10, 26, 42],
        'ucec': [11, 27, 43],
        'coad': [12, 28, 44],
        'lusc': [13, 29, 45],
        'cesc': [14, 30, 46],
        'kirp': [15, 31, 47]
    }
    data_x = data_x[:, cancerType_dict[cancerType]]
    print(f"===== [INFO] Loading hyperparameters for Specific Cancer: {cancerType.upper()} =====")
    learning_rate = 1e-4
    epochs = 50
    num_heads = 2
    num_layers = 2
    dropout = 0.2

print(f"Applied Hyperparameters: Learning Rate={learning_rate}, Epochs={epochs}, Heads={num_heads}, Layers={num_layers}, Dropout={dropout}")
print(f"[INFO] node feature shape = {data_x.shape}")

node_features = data_x
num_nodes = node_features.size(0)
input_dim = node_features.size(1)

# =========================
# Load graph data
# =========================
ppiAdj = torch.load(dataPath + f'/{args.dataset}_ppi.pkl', map_location=torch.device("cpu"))
pathAdj = torch.load(dataPath + '/pathway_SimMatrix_filtered.pkl', map_location=torch.device("cpu"))
goAdj = torch.load(dataPath + '/GO_SimMatrix_filtered.pkl', map_location=torch.device("cpu"))

ppi_row, ppi_col, ppi_score = extract_edge_data_with_score(ppiAdj,device=device)
path_row, path_col, path_score = extract_edge_data_with_score(pathAdj,device=device)
go_row, go_col, go_score = extract_edge_data_with_score(goAdj,device=device)

edge_indices_with_score = {
    "ppi": (ppi_row.to(device), ppi_col.to(device), ppi_score.to(device)),
    "path": (path_row.to(device), path_col.to(device), path_score.to(device)),
    "go": (go_row.to(device), go_col.to(device), go_score.to(device)),
}

# =========================
# Load labels
# =========================
Y_raw, label_pos, label_neg = load_label_single(dataPath + "/dataset/specific-cancer/", cancerType, device)

# 构造严格二分类标签：正样本=1，严格负样本=0，其余默认0但不会进入mask计算
labels = torch.zeros(num_nodes, dtype=torch.float32, device=device)
labels[torch.tensor(label_pos, dtype=torch.long, device=device)] = 1.0
labels[torch.tensor(label_neg, dtype=torch.long, device=device)] = 0.0

all_idx = np.arange(num_nodes)
labeled_set = set(label_pos.tolist()) | set(label_neg.tolist())
unlabeled_idx_np = np.array(sorted(list(set(all_idx.tolist()) - labeled_set)), dtype=int)

# 打乱后再分折
rng = np.random.default_rng(args.seed)
label_pos = np.array(label_pos)
label_neg = np.array(label_neg)
rng.shuffle(label_pos)
rng.shuffle(label_neg)

l1 = len(label_pos) // 10
l2 = len(label_neg) // 10
folds = stratified_kfold_split(label_pos, label_neg, num_nodes, l1, l2)

# =========================
# Training / Evaluation
# =========================
all_fold_results = []

for fold, (train_idx, val_idx, test_idx, train_mask, val_mask, test_mask) in enumerate(folds):
    print(f"\n--------- Fold {fold + 1} Begin ---------")

    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    train_idx_tensor = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx_tensor = torch.tensor(val_idx, dtype=torch.long, device=device)
    test_idx_tensor = torch.tensor(test_idx, dtype=torch.long, device=device)

    train_set = set(train_idx)
    train_pos_idx = np.array([i for i in label_pos if i in train_set], dtype=int)
    train_neg_idx = np.array([i for i in label_neg if i in train_set], dtype=int)

    train_pos_idx_tensor = torch.tensor(train_pos_idx, dtype=torch.long, device=device)
    train_neg_idx_tensor = torch.tensor(train_neg_idx, dtype=torch.long, device=device)
    unlabeled_idx_tensor = torch.tensor(unlabeled_idx_np, dtype=torch.long, device=device)

    model = MultiViewContrastiveModel(
        input_dim=input_dim,
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        embed_dim=args.embed_dim,
        view_names=("ppi", "path", "go"),
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=args.weight_decay
    )

    # 类别不平衡权重
    if len(train_pos_idx) > 0:
        pos_weight = torch.tensor(
            [len(train_neg_idx) / max(len(train_pos_idx), 1)],
            dtype=torch.float32,
            device=device
        )
    else:
        pos_weight = None

    loss_fn = Loss_fun(
        temperature=0.2,
        lambda_main=1.0,
        lambda_view=1.0,
        lambda_sup=1.0,
        lambda_unsup=0.2,
        max_unsup_samples=2048
    )

    best_val_aupr = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(node_features, edge_indices_with_score)

        loss_dict = loss_fn(
            outputs=outputs,
            labels=labels,
            train_mask=train_mask,
            train_pos_idx=train_pos_idx_tensor,
            train_neg_idx=train_neg_idx_tensor,
            unlabeled_idx=unlabeled_idx_tensor,
            pos_weight=pos_weight
        )

        total_loss = loss_dict["total_loss"]
        total_loss.backward()
        optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(node_features, edge_indices_with_score)
            val_metrics = compute_metrics(
                val_outputs["fused_logit"], labels, val_mask
            )

        if val_metrics["aupr"] > best_val_aupr:
            best_val_aupr = val_metrics["aupr"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Fold {fold + 1} | Epoch {epoch + 1:03d} | "
                f"Total={loss_dict['total_loss'].item():.4f} | "
                f"Main={loss_dict['main_loss'].item():.4f} | "
                f"View={loss_dict['view_loss'].item():.4f} | "
                f"SupCL={loss_dict['sup_cl_loss'].item():.4f} | "
                f"UnsupCL={loss_dict['unsup_cl_loss'].item():.4f} | "
                f"Val AUC={val_metrics['auc']:.4f} | "
                f"Val AUPR={val_metrics['aupr']:.4f} | "
                f"Val F1={val_metrics['f1']:.4f}"
            )

        if patience_counter >= args.patience:
            print(f"[INFO] Early stopping at epoch {epoch + 1}")
            break

    # test
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_outputs = model(node_features, edge_indices_with_score)
        test_metrics = compute_metrics(
            test_outputs["fused_logit"], labels, test_mask
        )

    print(
        f"[Fold {fold + 1} Result] "
        f"AUC={test_metrics['auc']:.4f}, "
        f"AUPR={test_metrics['aupr']:.4f}, "
        f"F1={test_metrics['f1']:.4f}"
    )

    all_fold_results.append(test_metrics)

    del model, optimizer, best_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =========================
# Final summary
# =========================
mean_auc = np.mean([x["auc"] for x in all_fold_results])
mean_aupr = np.mean([x["aupr"] for x in all_fold_results])
mean_f1 = np.mean([x["f1"] for x in all_fold_results])

std_auc = np.std([x["auc"] for x in all_fold_results])
std_aupr = np.std([x["aupr"] for x in all_fold_results])
std_f1 = np.std([x["f1"] for x in all_fold_results])

print("\n========== Final 10-Fold Results ==========")
print(f"AUC  = {mean_auc:.4f} ± {std_auc:.4f}")
print(f"AUPR = {mean_aupr:.4f} ± {std_aupr:.4f}")
print(f"F1   = {mean_f1:.4f} ± {std_f1:.4f}")