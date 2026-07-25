from __future__ import annotations

import argparse

from .config import apply_cli_overrides, load_config
from .train import run_cross_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DyRCoD by running cross-validation.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cancer", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--gpu", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    summary = run_cross_validation(config)
    print("[DyRCoD] Evaluation complete.")
    print(summary["test_summary"])


if __name__ == "__main__":
    main()
