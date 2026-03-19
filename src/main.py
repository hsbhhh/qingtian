import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from parsersed import args
from model import DriverGeneFewShotModel
from loss import DriverGeneLoss
from utils import (
    set_seed,
    extract_edge_data_with_score,
    load_label_single,
    stratified_kfold_split,
    build_edge_weight_dict,
    compute_diffusion_matrix_from_ppi,
    compute_graph_structural_features,
    build_hard_positive_weight,
    compute_metrics,
    summarize_metrics,
    format_metric_dict,
    format_metric_summary,
    EarlyStopping,
)





def prepare_feature_matrix(data_x_df, cancer_type, device):
    data_x_df = data_x_df.copy()
    data_x_df.index = data_x_df.index.astype(str).str.strip().str.upper()
    cancer_type = cancer_type.lower()

    if cancer_type == 'pan-cancer':
        base_df = data_x_df.iloc[:, :48].copy()
        base_scaler = StandardScaler()
        base_scaled = base_scaler.fit_transform(base_df.values)
        final_df = pd.DataFrame(base_scaled, index=base_df.index, columns=base_df.columns)

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

        base_df = data_x_df.iloc[:, cancerType_dict[cancer_type]].copy()
        base_scaler = StandardScaler()
        base_scaled = base_scaler.fit_transform(base_df.values)
        final_df = pd.DataFrame(base_scaled, index=base_df.index, columns=base_df.columns)

        if cancer_type == 'luad':
            dataPath = "./Data"

            gene_methy_path = dataPath + "/LUAD_methylation_features.csv"
            methy_file = pd.read_csv(gene_methy_path)
            methy_file.columns = ['gene', 'avg']
            methy_file['gene'] = methy_file['gene'].astype(str).str.strip().str.upper()
            methy_file = methy_file.drop_duplicates(subset=['gene']).set_index('gene')
            methy_file.columns = ['methy_avg']

            gene_crispr_path = dataPath + "/LUAD_crispr_avg_features.csv"
            crispr_file = pd.read_csv(gene_crispr_path)
            crispr_file.columns = ['gene', 'avg']
            crispr_file['gene'] = crispr_file['gene'].astype(str).str.strip().str.upper()
            crispr_file = crispr_file.drop_duplicates(subset=['gene']).set_index('gene')
            crispr_file.columns = ['crispr_avg']

            hic_feat_path = dataPath + "/my_gene_hic_5d_features.csv"
            hic_file = pd.read_csv(hic_feat_path)
            hic_file['gene'] = hic_file['gene'].astype(str).str.strip().str.upper()
            hic_file = hic_file.drop_duplicates(subset=['gene']).set_index('gene')
            hic_file.columns = ['hic_1', 'hic_2', 'hic_3', 'hic_4', 'hic_5']

            methy_df = methy_file.reindex(final_df.index)
            crispr_df = crispr_file.reindex(final_df.index)
            hic_df = hic_file.reindex(final_df.index)
            if 'methy_avg' in methy_df.columns:
                methy_df['methy_avg'] = methy_df['methy_avg'].fillna(methy_df['methy_avg'].median())

            if 'crispr_avg' in crispr_df.columns:
                crispr_df['crispr_avg'] = crispr_df['crispr_avg'].fillna(crispr_df['crispr_avg'].median())

            hic_cols = ['hic_1', 'hic_2', 'hic_3', 'hic_4', 'hic_5']
            for col in hic_cols:
                if col in hic_df.columns:
                    hic_df[col] = hic_df[col].fillna(hic_df[col].median())
            if 'crispr_avg' in crispr_df.columns:
                crispr_scaler = StandardScaler()
                crispr_df[['crispr_avg']] = crispr_scaler.fit_transform(crispr_df[['crispr_avg']].values)

            final_df = pd.concat(
                [
                    final_df,
                    methy_df,
                    crispr_df,
                    hic_df
                ],
                axis=1
            )

    print(f"[prepare_feature_matrix] cancer={cancer_type}, final shape={final_df.shape}")
    print(f"[prepare_feature_matrix] columns={list(final_df.columns)}")

    data_x = torch.tensor(final_df.values, dtype=torch.float32, device=device)
    return data_x


def get_split_indices(mask, y):
    idx = torch.where(mask)[0].detach().cpu().numpy().tolist()
    pos_idx = [i for i in idx if float(y[i].item()) > 0.5]
    neg_idx = [i for i in idx if float(y[i].item()) <= 0.5]
    return idx, pos_idx, neg_idx


def sample_episode(train_pos_idx, train_neg_idx, episode_pos, episode_neg):
    pos_k = min(len(train_pos_idx), episode_pos)
    neg_k = min(len(train_neg_idx), episode_neg)
    pos_support = random.sample(train_pos_idx, pos_k) if pos_k > 0 else []
    neg_support = random.sample(train_neg_idx, neg_k) if neg_k > 0 else []

    remain_pos = [i for i in train_pos_idx if i not in set(pos_support)]
    remain_neg = [i for i in train_neg_idx if i not in set(neg_support)]
    query_pos = random.sample(remain_pos, min(len(remain_pos), episode_pos)) if len(remain_pos) > 0 else []
    query_neg = random.sample(remain_neg, min(len(remain_neg), episode_neg)) if len(remain_neg) > 0 else []
    return pos_support, neg_support, query_pos, query_neg


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
        base_feature_dim=args.base_feature_dim,
        feature_gate_hidden_dim=args.feature_gate_hidden_dim,
        ema_momentum=args.ema_momentum,
        proto_shift_scale=args.proto_shift_scale,
        topo_prompt_dim=args.topo_prompt_dim,
        gate_hidden_dim=args.gate_hidden_dim,
        num_pos_prototypes=args.num_pos_prototypes,
        task_aug_ratio=args.task_aug_ratio,
        proto_logit_scale=args.proto_logit_scale,
        logit_temperature=args.logit_temperature,
    )


def evaluate_split(
    model,
    x,
    pos_feat,
    edge_index_dict,
    edge_weight_dict,
    y,
    mask,
    train_pos_idx,
    train_neg_idx,
    fixed_threshold=0.5,
    dynamic_f1=False,
):
    model.eval()
    with torch.no_grad():
        output = model(
            x=x,
            pos_feat=pos_feat,
            edge_index_dict=edge_index_dict,
            edge_weight_dict=edge_weight_dict,
            pos_idx=train_pos_idx,
            neg_idx=train_neg_idx,
            update_ema=False,
        )
        prob = output['prob']
        y_true = y[mask].detach().cpu().numpy()
        y_score = prob[mask].detach().cpu().numpy()

        metrics = compute_metrics(
            y_true,
            y_score,
            threshold=fixed_threshold,
            dynamic_f1=dynamic_f1
        )

        metrics_fixed05 = compute_metrics(
            y_true,
            y_score,
            threshold=0.5,
            dynamic_f1=False
        )
        metrics['F1@0.5'] = metrics_fixed05['F1']
        metrics['ACC@0.5'] = metrics_fixed05['ACC']
        metrics['Precision@0.5'] = metrics_fixed05['Precision']
        metrics['Recall@0.5'] = metrics_fixed05['Recall']

    return metrics, prob.detach(), output

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
    epochs,
    patience,
    episode_pos,
    episode_neg,
    ema_warmup_epochs,
    task_aug_start_epoch,
    device,
):
    stopper = EarlyStopping(patience=patience, mode='max')
    _, train_pos_idx, train_neg_idx = get_split_indices(train_mask, y)
    best_epoch = 0
    best_val_threshold = 0.5

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        support_pos_idx, support_neg_idx, query_pos_idx, query_neg_idx = sample_episode(train_pos_idx, train_neg_idx, episode_pos, episode_neg)
        use_ema = epoch > ema_warmup_epochs

        output = model(
            x=x,
            pos_feat=pos_feat,
            edge_index_dict=edge_index_dict,
            edge_weight_dict=edge_weight_dict,
            pos_idx=support_pos_idx,
            neg_idx=support_neg_idx,
            update_ema=use_ema,
        )

        if epoch < task_aug_start_epoch:
            output['aug_proto_logits'] = None

        sample_weight, hard_pos_nodes = build_hard_positive_weight(
            train_pos_idx=train_pos_idx,
            prob=output['prob'].detach(),
            total_nodes=x.size(0),
            low_prob_threshold=0.7,
            extra_weight=1.5,
        )
        sample_weight = sample_weight.to(device)

        proto_mask = torch.zeros_like(train_mask, dtype=torch.bool)
        episode_nodes = support_pos_idx + support_neg_idx + query_pos_idx + query_neg_idx
        if len(episode_nodes) > 0:
            proto_mask[torch.tensor(episode_nodes, device=device)] = True
        else:
            proto_mask = train_mask

        total_loss, loss_dict = criterion(
            output_dict=output,
            y=y,
            supervised_mask=train_mask,
            proto_mask=proto_mask,
            sample_weight=sample_weight,
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        train_metrics, _, _ = evaluate_split(
            model, x, pos_feat, edge_index_dict, edge_weight_dict,
            y, train_mask, train_pos_idx, train_neg_idx,
            fixed_threshold=0.5,
            dynamic_f1=True
        )

        val_metrics, _, _ = evaluate_split(
            model, x, pos_feat, edge_index_dict, edge_weight_dict,
            y, val_mask, train_pos_idx, train_neg_idx,
            fixed_threshold=0.5,
            dynamic_f1=True
        )

        if stopper.step(val_metrics['AUPR'], model):
            best_epoch = epoch
            best_val_threshold = float(val_metrics['Threshold'])

        if epoch % 10 == 0 or epoch == 1:
            gate_msg = {k: round(float(v.cpu().item()), 3) for k, v in output.get('feature_gate_stats', {}).items()}
            fusion_stats = output.get('fusion_stats', {})
            cls_gate_msg = {k: round(float(v.cpu().item()), 3) for k, v in fusion_stats.get('cls', {}).items()} if fusion_stats else {}
            proto_gate_msg = {k: round(float(v.cpu().item()), 3) for k, v in fusion_stats.get('proto', {}).items()} if fusion_stats else {}
            print(
                f'[Fold {fold_id}][Epoch {epoch}] '
                f'total={total_loss.item():.4f} '
                f'cls={loss_dict["cls"].item():.4f} '
                f'proto={loss_dict["proto"].item():.4f} '
                f'shared={loss_dict["shared"].item():.4f} '
                f'gateDiv={loss_dict["gate_div"].item():.4f} '
                f'protoBal={loss_dict["proto_balance"].item():.4f} '
                f'taskAug={loss_dict["task_aug"].item():.4f} '
                f'HardPos={len(hard_pos_nodes)} '
                f'EMA={int(use_ema)} '
                f'TrainAUC={train_metrics["AUC"]:.4f} '
                f'TrainAUPR={train_metrics["AUPR"]:.4f} '
                f'ValAUC={val_metrics["AUC"]:.4f} '
                f'ValAUPR={val_metrics["AUPR"]:.4f}'
            )
            print(f'[Fold {fold_id}][Epoch {epoch}] GroupGateMean={gate_msg}')
            print(f'[Fold {fold_id}][Epoch {epoch}] ViewGateCLS={cls_gate_msg} ViewGatePROTO={proto_gate_msg}')

        if stopper.should_stop:
            print(f'[Fold {fold_id}] Early stopping at epoch {epoch}. Best epoch={best_epoch}, Best ValAUPR={stopper.best_score:.4f}')
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    train_metrics, _, _ = evaluate_split(
        model, x, pos_feat, edge_index_dict, edge_weight_dict,
        y, train_mask, train_pos_idx, train_neg_idx,
        fixed_threshold=best_val_threshold,
        dynamic_f1=False
    )

    val_metrics, _, _ = evaluate_split(
        model, x, pos_feat, edge_index_dict, edge_weight_dict,
        y, val_mask, train_pos_idx, train_neg_idx,
        fixed_threshold=best_val_threshold,
        dynamic_f1=False
    )

    test_metrics, _, _ = evaluate_split(
        model, x, pos_feat, edge_index_dict, edge_weight_dict,
        y, test_mask, train_pos_idx, train_neg_idx,
        fixed_threshold=best_val_threshold,
        dynamic_f1=False
    )
    return {
        'best_epoch': best_epoch,
        'best_val_aupr': stopper.best_score,
        'best_val_threshold': best_val_threshold,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
    }


def main():
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print(f'===== Device: {device} =====')
    print(f'Applied Hyperparameters: LR={args.lr}, Epochs={args.epochs}, Layers={args.num_layers}, Dropout={args.dropout}, Hidden={args.hidden_dim}, Embed={args.embed_dim}')

    data_path = './Data'
    data_x_df = pd.read_csv(os.path.join(data_path, f'multiomics_features_{args.dataset}.tsv'), sep='\t', index_col=0)
    data_x_df = data_x_df.dropna()
    data_x_df.index = data_x_df.index.astype(str).str.strip().str.upper()

    node_features = prepare_feature_matrix(data_x_df, args.cancerType, device)
    ppi_adj = torch.load(os.path.join(data_path, f'{args.dataset}_ppi.pkl'), map_location=torch.device('cpu'))
    path_adj = torch.load(os.path.join(data_path, 'pathway_SimMatrix_filtered.pkl'), map_location=torch.device('cpu'))
    go_adj = torch.load(os.path.join(data_path, 'GO_SimMatrix_filtered.pkl'), map_location=torch.device('cpu'))

    Y, label_pos, label_neg = load_label_single(data_path + '/', args.cancerType.lower(), device)
    pos_feat = compute_graph_structural_features(ppi_adj, label_pos).to(device)

    ppi_row, ppi_col, ppi_score = extract_edge_data_with_score(ppi_adj)
    path_row, path_col, path_score = extract_edge_data_with_score(path_adj)
    go_row, go_col, go_score = extract_edge_data_with_score(go_adj)

    edge_indices_with_score = {
        'ppi': (ppi_row, ppi_col, ppi_score),
        'path': (path_row, path_col, path_score),
        'go': (go_row, go_col, go_score),
    }
    view_names = ['ppi', 'path', 'go']



    edge_index_dict, edge_weight_dict = build_edge_weight_dict(edge_indices_with_score, device)

    print(f'Node features shape: {node_features.shape}')
    print(f'Position features shape: {pos_feat.shape}')
    print(f'Views: {view_names}')
    print(f'Positive samples: {len(label_pos)}, Negative samples: {len(label_neg)}')

    folds = stratified_kfold_split(
        pos_label=label_pos,
        neg_label=label_neg,
        total_nodes=node_features.size(0),
        n_splits=args.n_splits,
        seed=args.seed,
        val_ratio=0.125,
    )

    all_train_metrics, all_val_metrics, all_test_metrics = [], [], []
    for fold_id, (_, _, _, train_mask, val_mask, test_mask) in enumerate(folds, start=1):
        print(f'\n--------- Fold {fold_id} Begin ---------')
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)

        model = build_model(args, in_dim=node_features.size(1), pos_dim=pos_feat.size(1), view_names=view_names).to(device)
        criterion = DriverGeneLoss(
            cls_pos_weight=args.cls_pos_weight,
            lambda_cls=args.lambda_cls,
            lambda_proto=args.lambda_proto,
            lambda_shared=args.lambda_shared,
            lambda_gate_div=args.lambda_gate_div,
            lambda_task_aug=args.lambda_task_aug,
            lambda_gate_sparse=args.lambda_gate_sparse,
            lambda_proto_balance=args.lambda_proto_balance,
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
            patience=args.patience,
            episode_pos=args.episode_pos,
            episode_neg=args.episode_neg,
            ema_warmup_epochs=args.ema_warmup_epochs,
            task_aug_start_epoch=args.task_aug_start_epoch,
            device=device,
        )

        print(
            f'[Fold {fold_id}] '
            f'BestEpoch={fold_result["best_epoch"]} '
            f'BestValAUPR={fold_result["best_val_aupr"]:.4f} '
            f'BestValThr={fold_result["best_val_threshold"]:.4f}'
        )
        print(f'[Fold {fold_id} Train] {format_metric_dict(fold_result["train_metrics"])}')
        print(f'[Fold {fold_id} Val]   {format_metric_dict(fold_result["val_metrics"])}')
        print(f'[Fold {fold_id} Test]  {format_metric_dict(fold_result["test_metrics"])}')

        all_train_metrics.append(fold_result['train_metrics'])
        all_val_metrics.append(fold_result['val_metrics'])
        all_test_metrics.append(fold_result['test_metrics'])

    train_summary = summarize_metrics(all_train_metrics)
    val_summary = summarize_metrics(all_val_metrics)
    test_summary = summarize_metrics(all_test_metrics)
    print(f'[Train] {format_metric_summary(train_summary)}')
    print(f'[Val]   {format_metric_summary(val_summary)}')
    print(f'[Test]  {format_metric_summary(test_summary)}')


if __name__ == '__main__':
    main()
