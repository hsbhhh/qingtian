from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "seed": 1234,
        "output_dir": "outputs",
    },
    "runtime": {
        "device": "auto",
        "gpu": 0,
        "num_threads": 1,
    },
    "data": {
        "root": "data/processed",
        "feature_dir": "data/processed/features",
        "network_dir": "data/processed/networks",
        "label_dir": "data/processed/labels",
        "gene_table": "multiomics_features_STRING.tsv",
        "mutation_pattern": "mutation/{cancer}_mutation_subtype_features.csv",
        "cnv_pattern": "CNV/{cancer}_CNV_features.csv",
        "expression_pattern": "Expression/{cancer}_expression_features.csv",
        "crispr_pattern": "CRISPR/{cancer}_crispr_avg_features.csv",
        "spatial_file": "Spatial/spatial_features.csv",
        "label_subdir": "specific-cancer",
        "label_pattern": "label_file-P-{cancer_lower}.txt",
        "positive_pattern": "pos-{cancer_lower}.txt",
        "negative_file": "pan-neg.txt",
        "networks": {
            "PPI": "STRING_ppi.pkl",
            "GO": "GO_SimMatrix_filtered_fixed.pkl",
            "Pathway": "KEGG_IDF_Cosine_threshold_0.6.pkl",
        },
        "network_weight_scale": {
            "PPI": 0.001,
            "GO": 1.0,
            "Pathway": 1.0,
        },
    },
    "model": {
        "hidden_dim": 64,
        "embed_dim": 128,
        "dropout": 0.1,
        "num_layers": 2,
        "role_dim": 48,
        "dynamic_hidden_dim": 64,
        "warmup_epochs": 20,
        "mag_scale": 0.15,
        "mag_min_scale": 0.60,
        "mag_max_scale": 1.40,
        "dir_range": 0.30,
        "change_budget_target": 0.06,
        "direction_budget_target": 0.06,
        "views": ["PPI", "GO", "Pathway"],
    },
    "training": {
        "cancer": "LUAD",
        "folds": 10,
        "epochs": 80,
        "patience": 20,
        "val_ratio": 0.125,
        "lr": 0.0005,
        "weight_decay": 0.0001,
        "cls_pos_weight": 5.0,
        "lambda_orth": 0.15,
        "lambda_edge": 0.10,
        "lambda_change": 0.12,
        "ckpt_w_aupr": 1.0,
        "ckpt_w_f1": 0.25,
        "ckpt_w_recall": 0.15,
    },
    "prediction": {
        "epochs": 50,
        "threshold": 0.5,
        "save_model": False,
    },
}


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config

    config_path = resolve_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return deep_update(config, loaded)


def resolve_path(path: Union[str, Path], root: Optional[Union[str, Path]] = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    base = PROJECT_ROOT if root is None else Path(root)
    return (base / candidate).resolve()


def config_path(config: Mapping[str, Any], *keys: str, root_key: Optional[str] = None) -> Path:
    value: Any = config
    for key in keys:
        value = value[key]
    root = config[root_key] if root_key else None
    return resolve_path(value, root=root)


def apply_cli_overrides(config: dict[str, Any], args: Any) -> dict[str, Any]:
    if getattr(args, "cancer", None):
        config["training"]["cancer"] = str(args.cancer).upper()
    if getattr(args, "epochs", None) is not None:
        config["training"]["epochs"] = int(args.epochs)
    if getattr(args, "folds", None) is not None:
        config["training"]["folds"] = int(args.folds)
    if getattr(args, "patience", None) is not None:
        config["training"]["patience"] = int(args.patience)
    if getattr(args, "lr", None) is not None:
        config["training"]["lr"] = float(args.lr)
    if getattr(args, "weight_decay", None) is not None:
        config["training"]["weight_decay"] = float(args.weight_decay)
    if getattr(args, "output_dir", None):
        config["project"]["output_dir"] = str(args.output_dir)
    if getattr(args, "device", None):
        config["runtime"]["device"] = str(args.device)
    if getattr(args, "gpu", None) is not None:
        config["runtime"]["gpu"] = int(args.gpu)
    if getattr(args, "seed", None) is not None:
        config["project"]["seed"] = int(args.seed)
    return config
