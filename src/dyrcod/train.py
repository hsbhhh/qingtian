from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch

from .config import apply_cli_overrides, load_config, resolve_path
from .data import DyRCoDData, build_pos_feat, get_train_pos_idx, load_dataset, stratified_kfold_split
from .loss import DynamicLIRSLoss
from .metrics import compute_metrics, format_metric_dict, format_metric_summary, summarize_metrics
from .model import DynamicLIRSDriverGeneModel
from .utils import EarlyStopping, choose_device, ensure_dir, set_seed, write_json


def build_model(config: Mapping[str, Any], data: DyRCoDData, pos_dim: int) -> DynamicLIRSDriverGeneModel:
    model_cfg = config["model"]
    return DynamicLIRSDriverGeneModel(
        in_dim=data.x.size(1),
        pos_dim=pos_dim,
        hidden_dim=int(model_cfg["hidden_dim"]),
        embed_dim=int(model_cfg["embed_dim"]),
        view_names=data.view_names,
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
        dynamic_hidden_dim=int(model_cfg["dynamic_hidden_dim"]),
        change_budget_target=float(model_cfg["change_budget_target"]),
        warmup_epochs=int(model_cfg["warmup_epochs"]),
        role_dim=int(model_cfg["role_dim"]),
        mag_scale=float(model_cfg["mag_scale"]),
        mag_min_scale=float(model_cfg["mag_min_scale"]),
        mag_max_scale=float(model_cfg["mag_max_scale"]),
        dir_range=float(model_cfg["dir_range"]),
        direction_budget_target=float(model_cfg["direction_budget_target"]),
    )


def evaluate_split(
    model: DynamicLIRSDriverGeneModel,
    data: DyRCoDData,
    pos_feat: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
    dynamic_f1: bool = False,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(
            x=data.x,
            pos_feat=pos_feat,
            edge_index_dict=data.edge_index_dict,
            edge_weight_dict=data.edge_weight_dict,
        )["logits"]
        prob = torch.sigmoid(logits)
        y_true = data.y[mask].detach().cpu().numpy()
        y_score = prob[mask].detach().cpu().numpy()

    metrics = compute_metrics(y_true, y_score, threshold=threshold, dynamic_f1=dynamic_f1)
    fixed_metrics = compute_metrics(y_true, y_score, threshold=0.5, dynamic_f1=False)
    metrics["F1@0.5"] = fixed_metrics["F1"]
    metrics["ACC@0.5"] = fixed_metrics["ACC"]
    metrics["Precision@0.5"] = fixed_metrics["Precision"]
    metrics["Recall@0.5"] = fixed_metrics["Recall"]
    return metrics


def predict_all_nodes(model: DynamicLIRSDriverGeneModel, data: DyRCoDData, pos_feat: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(
            x=data.x,
            pos_feat=pos_feat,
            edge_index_dict=data.edge_index_dict,
            edge_weight_dict=data.edge_weight_dict,
        )["logits"]
    return torch.sigmoid(logits).detach().float().cpu()


def checkpoint_score(config: Mapping[str, Any], metrics: Mapping[str, float]) -> float:
    train_cfg = config["training"]
    return float(
        float(train_cfg["ckpt_w_aupr"]) * metrics["AUPR"]
        + float(train_cfg["ckpt_w_f1"]) * metrics["F1@0.5"]
        + float(train_cfg["ckpt_w_recall"]) * metrics["Recall@0.5"]
    )


def train_one_fold(
    config: Mapping[str, Any],
    fold_id: int,
    data: DyRCoDData,
    pos_feat: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> tuple[DynamicLIRSDriverGeneModel, dict[str, Any]]:
    train_cfg = config["training"]
    model = build_model(config, data, pos_dim=pos_feat.size(1)).to(data.x.device)
    criterion = DynamicLIRSLoss(
        cls_pos_weight=float(train_cfg["cls_pos_weight"]),
        lambda_orth=float(train_cfg["lambda_orth"]),
        lambda_edge=float(train_cfg["lambda_edge"]),
        lambda_change=float(train_cfg["lambda_change"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    stopper = EarlyStopping(patience=int(train_cfg["patience"]))
    best_epoch = 0
    best_val_aupr_at_ckpt = float("-inf")
    best_val_f1_fixed05_at_ckpt = float("-inf")
    best_val_recall_fixed05_at_ckpt = float("-inf")

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        model.set_epoch(epoch)
        optimizer.zero_grad()

        output = model(
            x=data.x,
            pos_feat=pos_feat,
            edge_index_dict=data.edge_index_dict,
            edge_weight_dict=data.edge_weight_dict,
            y=data.y,
            supervised_mask=train_mask,
        )
        total_loss, loss_dict = criterion(output_dict=output, y=data.y, supervised_mask=train_mask)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        train_metrics = evaluate_split(model, data, pos_feat, train_mask, threshold=0.5, dynamic_f1=True)
        val_metrics = evaluate_split(model, data, pos_feat, val_mask, threshold=0.5, dynamic_f1=True)
        ckpt_score = checkpoint_score(config, val_metrics)

        if stopper.step(ckpt_score, model):
            best_epoch = epoch
            best_val_aupr_at_ckpt = float(val_metrics["AUPR"])
            best_val_f1_fixed05_at_ckpt = float(val_metrics["F1@0.5"])
            best_val_recall_fixed05_at_ckpt = float(val_metrics["Recall@0.5"])

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[Fold {fold_id}][Epoch {epoch}] "
                f"total={total_loss.item():.4f} "
                f"sup={loss_dict['sup'].item():.4f} "
                f"orth={loss_dict['orth'].item():.4f} "
                f"edge={loss_dict['edge'].item():.4f} "
                f"change={loss_dict['change'].item():.4f} "
                f"ValAUC={val_metrics['AUC']:.4f} "
                f"ValAUPR={val_metrics['AUPR']:.4f} "
                f"ValF1@0.5={val_metrics['F1@0.5']:.4f} "
                f"CkptScore={ckpt_score:.4f}",
                flush=True,
            )

        if stopper.should_stop:
            print(
                f"[Fold {fold_id}] Early stopping at epoch {epoch}. "
                f"Best epoch={best_epoch}, Best score={stopper.best_score:.4f}",
                flush=True,
            )
            break

    if stopper.best_state is not None:
        model.load_state_dict({key: value.to(data.x.device) for key, value in stopper.best_state.items()})

    val_metrics_dyn = evaluate_split(model, data, pos_feat, val_mask, threshold=0.5, dynamic_f1=True)
    best_val_threshold = float(val_metrics_dyn["Threshold"])
    train_metrics = evaluate_split(model, data, pos_feat, train_mask, threshold=best_val_threshold, dynamic_f1=False)
    val_metrics = evaluate_split(model, data, pos_feat, val_mask, threshold=best_val_threshold, dynamic_f1=False)
    test_metrics = evaluate_split(model, data, pos_feat, test_mask, threshold=best_val_threshold, dynamic_f1=False)

    result = {
        "fold": int(fold_id),
        "best_epoch": int(best_epoch),
        "best_checkpoint_score": float(stopper.best_score if stopper.best_score is not None else float("nan")),
        "best_val_aupr_at_ckpt": best_val_aupr_at_ckpt,
        "best_val_f1_fixed05_at_ckpt": best_val_f1_fixed05_at_ckpt,
        "best_val_recall_fixed05_at_ckpt": best_val_recall_fixed05_at_ckpt,
        "best_val_threshold": best_val_threshold,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    return model, result


def run_cross_validation(config: Mapping[str, Any]) -> dict[str, Any]:
    seed = int(config["project"]["seed"])
    set_seed(seed)
    num_threads = int(config.get("runtime", {}).get("num_threads", 1))
    torch.set_num_threads(max(num_threads, 1))

    device = choose_device(config["runtime"]["device"], int(config["runtime"].get("gpu", 0)))
    cancer = str(config["training"]["cancer"]).upper()
    output_dir = ensure_dir(resolve_path(config["project"]["output_dir"]) / cancer)
    print(f"[DyRCoD] Device: {device}", flush=True)
    print(f"[DyRCoD] Cancer: {cancer}", flush=True)

    data = load_dataset(config, cancer, device)
    print(f"[DyRCoD] Node features: {tuple(data.x.shape)}", flush=True)
    print(f"[DyRCoD] Views: {data.view_names}", flush=True)
    print(
        f"[DyRCoD] Positive={len(data.label_pos)} Negative={len(data.label_neg)} "
        f"Unlabeled={data.x.size(0) - len(set(data.label_pos.tolist()) | set(data.label_neg.tolist()))}",
        flush=True,
    )

    folds = stratified_kfold_split(
        pos_label=data.label_pos,
        neg_label=data.label_neg,
        total_nodes=data.x.size(0),
        n_splits=int(config["training"]["folds"]),
        seed=seed,
        val_ratio=float(config["training"]["val_ratio"]),
    )

    all_train_metrics: list[dict[str, float]] = []
    all_val_metrics: list[dict[str, float]] = []
    all_test_metrics: list[dict[str, float]] = []
    fold_results: list[dict[str, Any]] = []

    for fold_id, (train_mask, val_mask, test_mask) in enumerate(folds, start=1):
        print(f"\n--------- Fold {fold_id} Begin ---------", flush=True)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)
        train_pos_idx = get_train_pos_idx(train_mask, data.y)
        pos_feat = build_pos_feat(data.adj_cpu_dict, train_pos_idx, data.view_names, device)
        model, fold_result = train_one_fold(config, fold_id, data, pos_feat, train_mask, val_mask, test_mask)
        fold_result["prediction_scores"] = predict_all_nodes(model, data, pos_feat).numpy()

        print(f"[Fold {fold_id} Train] {format_metric_dict(fold_result['train_metrics'])}", flush=True)
        print(f"[Fold {fold_id} Val]   {format_metric_dict(fold_result['val_metrics'])}", flush=True)
        print(f"[Fold {fold_id} Test]  {format_metric_dict(fold_result['test_metrics'])}", flush=True)

        all_train_metrics.append(fold_result["train_metrics"])
        all_val_metrics.append(fold_result["val_metrics"])
        all_test_metrics.append(fold_result["test_metrics"])
        fold_results.append(fold_result)

    summary = {
        "model": "DyRCoD",
        "cancer": cancer,
        "seed": seed,
        "folds": int(config["training"]["folds"]),
        "feature_dim": int(data.x.size(1)),
        "views": data.view_names,
        "num_genes": int(data.x.size(0)),
        "num_positive": int(len(data.label_pos)),
        "num_negative": int(len(data.label_neg)),
        "train_summary": summarize_metrics(all_train_metrics),
        "val_summary": summarize_metrics(all_val_metrics),
        "test_summary": summarize_metrics(all_test_metrics),
        "fold_results": [
            {key: value for key, value in row.items() if key != "prediction_scores"}
            for row in fold_results
        ],
    }
    write_json(output_dir / "cross_validation_summary.json", summary)

    fold_metric_rows = []
    for result in fold_results:
        row = {"fold": result["fold"], "best_epoch": result["best_epoch"], "best_val_threshold": result["best_val_threshold"]}
        row.update({f"test_{key}": value for key, value in result["test_metrics"].items()})
        fold_metric_rows.append(row)
    pd.DataFrame(fold_metric_rows).to_csv(output_dir / "cross_validation_metrics.csv", index=False)

    print(f"[Train] {format_metric_summary(summary['train_summary'])}", flush=True)
    print(f"[Val]   {format_metric_summary(summary['val_summary'])}", flush=True)
    print(f"[Test]  {format_metric_summary(summary['test_summary'])}", flush=True)
    print(f"[DyRCoD] Summary written to {output_dir / 'cross_validation_summary.json'}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DyRCoD with stratified cross-validation.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cancer", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--gpu", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    run_cross_validation(config)


if __name__ == "__main__":
    main()
