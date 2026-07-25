from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_best_f1_threshold(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    thresholds = np.unique(y_score)
    if thresholds.size == 0:
        return 0.5

    best_thr, best_f1 = 0.5, -1.0
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        try:
            cur_f1 = f1_score(y_true, y_pred)
        except Exception:
            cur_f1 = -1.0
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_thr = float(threshold)
    return best_thr


def compute_metrics(y_true, y_score, threshold: float = 0.5, dynamic_f1: bool = False) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    if dynamic_f1:
        threshold = find_best_f1_threshold(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)

    metrics: dict[str, float] = {}
    try:
        metrics["AUC"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        metrics["AUC"] = float("nan")
    try:
        metrics["AUPR"] = float(average_precision_score(y_true, y_score))
    except Exception:
        metrics["AUPR"] = float("nan")
    try:
        metrics["F1"] = float(f1_score(y_true, y_pred))
    except Exception:
        metrics["F1"] = float("nan")
    try:
        metrics["ACC"] = float(accuracy_score(y_true, y_pred))
    except Exception:
        metrics["ACC"] = float("nan")
    try:
        metrics["Precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    except Exception:
        metrics["Precision"] = float("nan")
    try:
        metrics["Recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    except Exception:
        metrics["Recall"] = float("nan")

    metrics["Threshold"] = float(threshold)
    return metrics


def summarize_metrics(metric_list: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    if not metric_list:
        return {}

    summary: dict[str, tuple[float, float]] = {}
    for key in metric_list[0].keys():
        values = [m[key] for m in metric_list if key in m and not np.isnan(m[key])]
        summary[key] = (float("nan"), float("nan")) if not values else (float(np.mean(values)), float(np.std(values)))
    return summary


def format_metric_dict(metrics: dict[str, float]) -> str:
    return " ".join(f"{key}={value:.4f}" for key, value in metrics.items())


def format_metric_summary(summary: dict[str, tuple[float, float]]) -> str:
    parts = []
    for key, (mean_v, std_v) in summary.items():
        parts.append(f"{key}=nan" if np.isnan(mean_v) else f"{key}={mean_v:.4f}+/-{std_v:.4f}")
    return " ".join(parts)
