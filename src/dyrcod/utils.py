from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
import torch


def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device_name: str = "auto", gpu: int = 0) -> torch.device:
    requested = str(device_name).lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda") and torch.cuda.is_available():
        if requested == "cuda":
            return torch.device(f"cuda:{int(gpu)}")
        return torch.device(requested)
    if requested == "auto" and torch.cuda.is_available():
        return torch.device(f"cuda:{int(gpu)}")
    return torch.device("cpu")


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_json_ready(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return make_json_ready(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): make_json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_ready(item) for item in value]
    return value


def write_json(path: Union[str, Path], payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


class EarlyStopping:
    def __init__(self, patience: int = 30):
        self.patience = int(patience)
        self.best_score: Optional[float] = None
        self.counter = 0
        self.best_state: Optional[dict[str, torch.Tensor]] = None
        self.should_stop = False

    def step(self, score: float, model: torch.nn.Module) -> bool:
        improved = self.best_score is None or score > self.best_score
        if improved:
            self.best_score = float(score)
            self.counter = 0
            self.best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved
