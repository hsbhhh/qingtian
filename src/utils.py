import pandas as pd
import numpy as np
import torch
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def extract_edge_data_with_score(sparse_tensor,device):
    sparse_tensor = sparse_tensor.coalesce()
    row, col = sparse_tensor.indices()
    # score = sparse_tensor.values()
    score = torch.ones(len(row), device=device)
    return row, col, score


def load_label_single(path, cancerType, device):
    label = np.loadtxt(path + "label_file-P-" + cancerType + ".txt")
    Y = torch.tensor(label, dtype=torch.float32, device=device)

    label_pos = np.loadtxt(path + "pos-" + cancerType + ".txt", dtype=int)
    label_neg = np.loadtxt(path + "pan-neg.txt", dtype=int)

    if np.isscalar(label_pos):
        label_pos = np.array([int(label_pos)])
    if np.isscalar(label_neg):
        label_neg = np.array([int(label_neg)])

    return Y, label_pos, label_neg


def stratified_kfold_split(pos_label, neg_label, l, l1, l2):
    folds = []
    for i in range(10):
        pos_test = list(pos_label[i * l1:(i + 1) * l1])
        pos_train = list(set(pos_label) - set(pos_test))
        neg_test = list(neg_label[i * l2:(i + 1) * l2])
        neg_train = list(set(neg_label) - set(neg_test))

        val_size_pos = max(len(pos_train) // 8, 1)
        val_size_neg = max(len(neg_train) // 8, 1)

        pos_val = list(pos_train[:val_size_pos])
        pos_train_final = list(pos_train[val_size_pos:])
        neg_val = list(neg_train[:val_size_neg])
        neg_train_final = list(neg_train[val_size_neg:])

        train_idx = sorted(pos_train_final + neg_train_final)
        val_idx = sorted(pos_val + neg_val)
        test_idx = sorted(pos_test + neg_test)

        indexs1 = [False] * l
        indexs2 = [False] * l
        indexs3 = [False] * l

        for j in train_idx:
            indexs1[j] = True
        for j in val_idx:
            indexs2[j] = True
        for j in test_idx:
            indexs3[j] = True

        train_mask = torch.from_numpy(np.array(indexs1))
        val_mask = torch.from_numpy(np.array(indexs2))
        test_mask = torch.from_numpy(np.array(indexs3))

        folds.append((train_idx, val_idx, test_idx, train_mask, val_mask, test_mask))

    return folds


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(logits, labels, mask):
    """
    logits: [N]
    labels: [N]
    mask: [N] bool
    """
    mask = mask.bool()
    if mask.sum().item() == 0:
        return {"auc": 0.0, "aupr": 0.0, "f1": 0.0}

    y_true = labels[mask].detach().cpu().numpy()
    y_score = torch.sigmoid(logits[mask]).detach().cpu().numpy()
    y_pred = (y_score >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_true, y_score)
    except:
        auc = 0.0

    try:
        aupr = average_precision_score(y_true, y_score)
    except:
        aupr = 0.0

    try:
        f1 = f1_score(y_true, y_pred)
    except:
        f1 = 0.0

    return {"auc": auc, "aupr": aupr, "f1": f1}