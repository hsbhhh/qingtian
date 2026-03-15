# main.py
import os
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from model import DriverGeneFewShotModel
from loss import DriverGeneLoss
from utils import (
    set_seed,
    extract_edge_data_with_score,
    load_label_single,
    stratified_kfold_split,
    build_edge_weight_dict,
    compute_graph_structural_features,
    compute_diffusion_matrix_from_ppi,
    mine_hard_negatives,
    sample_alignment_nodes,
    generate_positive_mixup_embeddings,
    build_hard_positive_weight,
    compute_metrics,
    summarize_metrics,
    format_metric_dict,
    format_metric_summary,
    EarlyStopping,
)


def prepare_feature_matrix(data_x_df, cancer_type, device):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(data_x_df.values)
    data_x = torch.tensor(features_scaled, dtype=torch.float32, device=device)

    cancer_type = cancer_type.lower()
    if cancer_type == 'pan-cancer':
        data_x = data_x[:, :48]
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
        if cancer_type not in cancerType_dict:
            raise ValueError(f"Unsupported cancer type: {cancer_type}")
        data_x = data_x[:, cancerType_dict[cancer_type]]

    return data_x


def get_default_hparams(cancer_type):
    cancer_type = cancer_type.lower()
    if cancer_type == 'pan-cancer':
        return {
            'learning_rate': 1e-3,
            'epochs': 120,
            'num_layers': 3,
            'dropout': 0.15,
            'hidden_dim': 64,
            'embed_dim': 128,
        }
    else:
        return {
            'learning_rate': 5e-4,
            'epochs': 150,
            'num_layers': 2,
            'dropout': 0.2,
            'hidden_dim': 64,
            'embed_dim': 128,
        }


def build_model(args, in_dim, pos_dim, view_names):
    return DriverGeneFewShotModel(
        in_dim=in_dim,
        pos_dim=pos_dim,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        view_names=view_names,
        proto_alpha=args.proto_alpha,
        proto_topk_ratio=args.proto_topk_ratio,
        min_proto_k=args.min_proto_k
    )


def evaluate_split(model, x, pos_feat, edge_index_dict, edge_weight_dict, y, eval_mask, train_pos_idx, train_neg_idx):
    model.eval()
    with torch.no_grad():
        output = model(
            x=x,
            pos_feat=pos_feat,
            edge_index_dict=edge_index_dict,
            edge_weight_dict=edge_weight_dict,
            pos_idx=train_pos_idx,
            neg_idx=train_neg_idx
        )
        prob = output['prob']

    eval_idx = torch.where(eval_mask)[0]
    y_true = y[eval_idx].detach().cpu().numpy()
    y_score = prob[eval_idx].detach().cpu().numpy()
    metrics = compute_metrics(y_true, y_score, threshold=0.5, dynamic_f1=True)
    return metrics, prob.detach()


def train_one_fold(
    fold_id,
    model,
    criterion,
    optimizer,
    x,
    pos_feat,
    y,
    edge_index_dict,
    edge_weight_dict,
    train_mask,
    val_mask,
    test_mask,
    epochs=150,
    hard_neg_topk=64,
    patience=30,
    mixup_num=32,
    mixup_alpha=0.4,
    hard_pos_threshold=0.7,
    hard_pos_extra_weight=1.5,
    align_max_nodes=256,
    device='cpu'
):
    early_stopper = EarlyStopping(patience=patience, mode='max')

    train_idx_all = torch.where(train_mask)[0].cpu().numpy().tolist()
    train_pos_idx = [i for i in train_idx_all if y[i].item() == 1]
    train_neg_idx = [i for i in train_idx_all if y[i].item() == 0]

    prev_hard_pos_weight = torch.ones(y.size(0), dtype=torch.float32, device=device)

    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # warm forward for hard negative mining
        with torch.no_grad():
            warm_out = model(
                x=x,
                pos_feat=pos_feat,
                edge_index_dict=edge_index_dict,
                edge_weight_dict=edge_weight_dict,
                pos_idx=train_pos_idx,
                neg_idx=train_neg_idx
            )
            warm_emb = warm_out['embedding']
            hard_neg_idx = mine_hard_negatives(
                embeddings=warm_emb,
                train_pos_idx=train_pos_idx,
                train_neg_idx=train_neg_idx,
                topk=hard_neg_topk
            )
            align_sample_idx = sample_alignment_nodes(
                train_pos_idx=train_pos_idx,
                train_neg_idx=train_neg_idx,
                hard_neg_idx=hard_neg_idx,
                max_nodes=align_max_nodes
            )

        output = model(
            x=x,
            pos_feat=pos_feat,
            edge_index_dict=edge_index_dict,
            edge_weight_dict=edge_weight_dict,
            pos_idx=train_pos_idx,
            neg_idx=train_neg_idx
        )

        # positive mixup on embedding space
        mixed_z = generate_positive_mixup_embeddings(
            embeddings=output['embedding'],
            pos_idx=train_pos_idx,
            alpha=mixup_alpha,
            num_mix=mixup_num
        )
        if mixed_z is not None:
            mixed_logits, _, _ = model.classify_embeddings(
                mixed_z, output['p_pos'], output['p_neg']
            )
        else:
            mixed_logits = None

        total_loss, loss_dict = criterion(
            output_dict=output,
            y=y,
            supervised_mask=train_mask,
            hard_neg_idx=hard_neg_idx,
            align_sample_idx=align_sample_idx,
            sample_weight=prev_hard_pos_weight,
            mixed_logits=mixed_logits
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        with torch.no_grad():
            train_prob = output['prob'].detach()
            prev_hard_pos_weight, hard_pos_idx = build_hard_positive_weight(
                train_pos_idx=train_pos_idx,
                prob=train_prob,
                total_nodes=y.size(0),
                low_prob_threshold=hard_pos_threshold,
                extra_weight=hard_pos_extra_weight
            )

        train_y_true = y[train_mask].detach().cpu().numpy()
        train_y_score = train_prob[train_mask].detach().cpu().numpy()
        train_metrics = compute_metrics(train_y_true, train_y_score, threshold=0.5, dynamic_f1=True)

        val_metrics, _ = evaluate_split(
            model, x, pos_feat, edge_index_dict, edge_weight_dict, y, val_mask, train_pos_idx, train_neg_idx
        )

        val_key = val_metrics['AUPR']
        if np.isnan(val_key):
            val_key = val_metrics['AUC'] if not np.isnan(val_metrics['AUC']) else -1.0

        improved = early_stopper.step(val_key, model)
        if improved:
            best_epoch = epoch

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[Fold {fold_id}][Epoch {epoch}] "
                f"total={total_loss.item():.4f} "
                f"cls={loss_dict['cls'].item():.4f} "
                f"proto={loss_dict['proto'].item():.4f} "
                f"hardNeg={loss_dict['hard_neg'].item():.4f} "
                f"align={loss_dict['align'].item():.8f} "
                f"mixup={loss_dict['mixup'].item():.4f} "
                f"hardPos={len(hard_pos_idx)} "
                f"TrainAUC={train_metrics['AUC']:.4f} "
                f"TrainAUPR={train_metrics['AUPR']:.4f} "
                f"ValAUC={val_metrics['AUC']:.4f} "
                f"ValAUPR={val_metrics['AUPR']:.4f}"
            )

        if early_stopper.should_stop:
            print(f"[Fold {fold_id}] Early stopping at epoch {epoch}. Best epoch = {best_epoch}")
            break

    early_stopper.restore(model, device=device)

    train_metrics, _ = evaluate_split(
        model, x, pos_feat, edge_index_dict, edge_weight_dict, y, train_mask, train_pos_idx, train_neg_idx
    )
    val_metrics, _ = evaluate_split(
        model, x, pos_feat, edge_index_dict, edge_weight_dict, y, val_mask, train_pos_idx, train_neg_idx
    )
    test_metrics, _ = evaluate_split(
        model, x, pos_feat, edge_index_dict, edge_weight_dict, y, test_mask, train_pos_idx, train_neg_idx
    )

    return {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'best_epoch': best_epoch
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='STRING')
    parser.add_argument('--cancerType', type=str, default='luad')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--num_layers', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--patience', type=int, default=30)

    parser.add_argument('--use_diffusion_view', action='store_true')
    parser.add_argument('--diff_alpha', type=float, default=0.15)
    parser.add_argument('--diff_topk', type=int, default=50)

    parser.add_argument('--proto_alpha', type=float, default=0.25)
    parser.add_argument('--proto_topk_ratio', type=float, default=0.6)
    parser.add_argument('--min_proto_k', type=int, default=8)

    parser.add_argument('--cls_pos_weight', type=float, default=8.0)
    parser.add_argument('--lambda_cls', type=float, default=1.0)
    parser.add_argument('--lambda_proto', type=float, default=0.3)
    parser.add_argument('--lambda_hard_neg', type=float, default=0.4)
    parser.add_argument('--lambda_align', type=float, default=0.15)
    parser.add_argument('--lambda_mixup', type=float, default=0.15)

    parser.add_argument('--proto_margin', type=float, default=1.0)
    parser.add_argument('--hard_neg_margin', type=float, default=0.5)
    parser.add_argument('--hard_neg_topk', type=int, default=64)

    parser.add_argument('--mixup_num', type=int, default=32)
    parser.add_argument('--mixup_alpha', type=float, default=0.4)

    parser.add_argument('--hard_pos_threshold', type=float, default=0.7)
    parser.add_argument('--hard_pos_extra_weight', type=float, default=1.5)

    parser.add_argument('--align_max_nodes', type=int, default=256)

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    hparams = get_default_hparams(args.cancerType)
    if args.lr is None:
        args.lr = hparams['learning_rate']
    if args.epochs is None:
        args.epochs = hparams['epochs']
    if args.num_layers is None:
        args.num_layers = hparams['num_layers']
    if args.dropout is None:
        args.dropout = hparams['dropout']

    print(f"===== Device: {device} =====")
    print(
        f"Applied Hyperparameters: "
        f"LR={args.lr}, Epochs={args.epochs}, Layers={args.num_layers}, "
        f"Dropout={args.dropout}, Hidden={args.hidden_dim}, Embed={args.embed_dim}"
    )

    # data input
    dataPath = "./Data"

    data_x_df = pd.read_csv(
        os.path.join(dataPath, f'multiomics_features_{args.dataset}.tsv'),
        sep='\t', index_col=0
    )
    data_x_df = data_x_df.dropna()

    node_features = prepare_feature_matrix(data_x_df, args.cancerType, device)

    ppiAdj = torch.load(os.path.join(dataPath, f'{args.dataset}_ppi.pkl'), map_location=torch.device("cpu"))
    pathAdj = torch.load(os.path.join(dataPath, 'pathway_SimMatrix_filtered.pkl'), map_location=torch.device("cpu"))
    goAdj = torch.load(os.path.join(dataPath, 'GO_SimMatrix_filtered.pkl'), map_location=torch.device("cpu"))

    Y, label_pos, label_neg = load_label_single(dataPath + "/", args.cancerType.lower(), device)

    # 结构位置编码特征
    pos_feat = compute_graph_structural_features(ppiAdj, label_pos).to(device)

    ppi_row, ppi_col, ppi_score = extract_edge_data_with_score(ppiAdj)
    path_row, path_col, path_score = extract_edge_data_with_score(pathAdj)
    go_row, go_col, go_score = extract_edge_data_with_score(goAdj)

    edge_indices_with_score = {
        "ppi": (ppi_row, ppi_col, ppi_score),
        "path": (path_row, path_col, path_score),
        "go": (go_row, go_col, go_score),
    }
    view_names = ['ppi', 'path', 'go']

    if args.use_diffusion_view:
        diffAdj = compute_diffusion_matrix_from_ppi(
            ppi_sparse_tensor=ppiAdj,
            alpha=args.diff_alpha,
            topk=args.diff_topk
        )
        diff_row, diff_col, diff_score = extract_edge_data_with_score(diffAdj)
        edge_indices_with_score["diff"] = (diff_row, diff_col, diff_score)
        view_names.append('diff')

    edge_index_dict, edge_weight_dict = build_edge_weight_dict(edge_indices_with_score, device)

    print(f"Node features shape: {node_features.shape}")
    print(f"Position features shape: {pos_feat.shape}")
    print(f"Views: {view_names}")
    print(f"Positive samples: {len(label_pos)}, Negative samples: {len(label_neg)}")

    folds = stratified_kfold_split(
        pos_label=label_pos,
        neg_label=label_neg,
        total_nodes=node_features.size(0),
        n_splits=args.n_splits,
        seed=args.seed,
        val_ratio=0.125
    )

    all_train_metrics = []
    all_val_metrics = []
    all_test_metrics = []

    for fold_id, (train_idx, val_idx, test_idx, train_mask, val_mask, test_mask) in enumerate(folds, start=1):
        print(f"\n--------- Fold {fold_id} Begin ---------")

        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)

        model = build_model(
            args=args,
            in_dim=node_features.size(1),
            pos_dim=pos_feat.size(1),
            view_names=view_names
        ).to(device)

        criterion = DriverGeneLoss(
            cls_pos_weight=args.cls_pos_weight,
            lambda_cls=args.lambda_cls,
            lambda_proto=args.lambda_proto,
            lambda_hard_neg=args.lambda_hard_neg,
            lambda_align=args.lambda_align,
            lambda_mixup=args.lambda_mixup,
            proto_margin=args.proto_margin,
            hard_neg_margin=args.hard_neg_margin
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        fold_result = train_one_fold(
            fold_id=fold_id,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            x=node_features,
            pos_feat=pos_feat,
            y=Y,
            edge_index_dict=edge_index_dict,
            edge_weight_dict=edge_weight_dict,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            epochs=args.epochs,
            hard_neg_topk=args.hard_neg_topk,
            patience=args.patience,
            mixup_num=args.mixup_num,
            mixup_alpha=args.mixup_alpha,
            hard_pos_threshold=args.hard_pos_threshold,
            hard_pos_extra_weight=args.hard_pos_extra_weight,
            align_max_nodes=args.align_max_nodes,
            device=device
        )

        train_metrics = fold_result['train_metrics']
        val_metrics = fold_result['val_metrics']
        test_metrics = fold_result['test_metrics']

        all_train_metrics.append(train_metrics)
        all_val_metrics.append(val_metrics)
        all_test_metrics.append(test_metrics)

        print(f"[Fold {fold_id} Train] {format_metric_dict(train_metrics)}")
        print(f"[Fold {fold_id} Val]   {format_metric_dict(val_metrics)}")
        print(f"[Fold {fold_id} Test]  {format_metric_dict(test_metrics)}")

    train_summary = summarize_metrics(all_train_metrics)
    val_summary = summarize_metrics(all_val_metrics)
    test_summary = summarize_metrics(all_test_metrics)

    print("\n================ Final Cross-Validation Summary ================")
    print(f"[Train] {format_metric_summary(train_summary)}")
    print(f"[Val]   {format_metric_summary(val_summary)}")
    print(f"[Test]  {format_metric_summary(test_summary)}")


if __name__ == '__main__':
    main()
