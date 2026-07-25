from __future__ import annotations

import argparse
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import torch

from .config import apply_cli_overrides, load_config, resolve_path
from .data import all_labeled_mask, build_pos_feat, label_types, load_dataset
from .loss import DynamicLIRSLoss
from .train import build_model
from .utils import choose_device, ensure_dir, set_seed, write_json


def train_final_model(
    config: Mapping[str, Any],
    epochs: Optional[int] = None,
) -> tuple[Any, Any, torch.Tensor]:
    seed = int(config["project"]["seed"])
    set_seed(seed)
    torch.set_num_threads(max(int(config.get("runtime", {}).get("num_threads", 1)), 1))
    device = choose_device(config["runtime"]["device"], int(config["runtime"].get("gpu", 0)))
    cancer = str(config["training"]["cancer"]).upper()
    data = load_dataset(config, cancer, device)

    train_mask = all_labeled_mask(data.label_pos, data.label_neg, data.x.size(0), device)
    train_pos_idx = [int(idx) for idx in data.label_pos.tolist()]
    pos_feat = build_pos_feat(data.adj_cpu_dict, train_pos_idx, data.view_names, device)
    model = build_model(config, data, pos_dim=pos_feat.size(1)).to(device)

    train_cfg = config["training"]
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

    final_epochs = int(epochs if epochs is not None else config.get("prediction", {}).get("epochs", train_cfg["epochs"]))
    for epoch in range(1, final_epochs + 1):
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
        if epoch == 1 or epoch % 10 == 0 or epoch == final_epochs:
            print(
                f"[Final][Epoch {epoch}] "
                f"total={total_loss.item():.4f} "
                f"sup={loss_dict['sup'].item():.4f} "
                f"orth={loss_dict['orth'].item():.4f} "
                f"edge={loss_dict['edge'].item():.4f} "
                f"change={loss_dict['change'].item():.4f}",
                flush=True,
            )
    return model, data, pos_feat


def run_final_prediction(
    config: Mapping[str, Any],
    epochs: Optional[int] = None,
    threshold: Optional[float] = None,
):
    model, data, pos_feat = train_final_model(config, epochs=epochs)
    model.eval()
    with torch.no_grad():
        output = model(
            x=data.x,
            pos_feat=pos_feat,
            edge_index_dict=data.edge_index_dict,
            edge_weight_dict=data.edge_weight_dict,
        )
        probabilities = torch.sigmoid(output["logits"]).detach().float().cpu().numpy()

    cutoff = float(threshold if threshold is not None else config.get("prediction", {}).get("threshold", 0.5))
    labels = label_types(data.x.size(0), data.label_pos, data.label_neg)
    true_label = np.full(data.x.size(0), "NA", dtype=object)
    true_label[labels == "Driver"] = "1"
    true_label[labels == "Negative"] = "0"

    predictions = pd.DataFrame(
        {
            "gene_symbol": np.asarray(data.genes, dtype=object).astype(str),
            "prediction_score": probabilities.reshape(-1),
            "pred_label": (probabilities.reshape(-1) >= cutoff).astype(int),
            "true_label": true_label,
            "label_type": labels.astype(str),
        }
    )
    predictions = predictions.sort_values(
        ["prediction_score", "gene_symbol"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    predictions["rank"] = np.arange(1, len(predictions) + 1, dtype=int)

    cancer = data.cancer
    output_dir = ensure_dir(resolve_path(config["project"]["output_dir"]) / cancer)
    prediction_path = output_dir / f"{cancer.lower()}_all_gene_predictions.csv"
    predictions.to_csv(prediction_path, index=False)

    manifest = {
        "model": "DyRCoD",
        "cancer": cancer,
        "seed": int(config["project"]["seed"]),
        "threshold": cutoff,
        "num_genes": int(data.x.size(0)),
        "num_positive": int(len(data.label_pos)),
        "num_negative": int(len(data.label_neg)),
        "num_unlabeled": int(data.x.size(0) - len(set(data.label_pos.tolist()) | set(data.label_neg.tolist()))),
        "prediction_file": prediction_path,
    }
    write_json(output_dir / f"{cancer.lower()}_prediction_manifest.json", manifest)

    if bool(config.get("prediction", {}).get("save_model", False)):
        torch.save(
            {
                "model": "DyRCoD",
                "cancer": cancer,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "feature_columns": data.feature_columns,
                "view_names": data.view_names,
            },
            output_dir / f"{cancer.lower()}_final_model.pt",
        )

    print(f"[DyRCoD] Prediction file written to {prediction_path}", flush=True)
    return prediction_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a final DyRCoD model and predict all genes.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cancer", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--gpu", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    run_final_prediction(config, epochs=args.epochs, threshold=args.threshold)


if __name__ == "__main__":
    main()
