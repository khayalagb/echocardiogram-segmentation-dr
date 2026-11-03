#!/usr/bin/env python
"""Train a U-Net with physics-inspired domain randomisation augmentation.

Same architecture, seed and budget as the baseline (default.yaml).
Only the augmentation pipeline differs.

Example:
    python scripts/08_train_dr.py --config configs/dr.yaml --run-name dr

Runtime: ~45–60 min on Mac MPS (early stopping typically at epoch 35–60).
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from echo_seg.training import train_dr
from echo_seg.utils import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Train U-Net with DR augmentation")
    parser.add_argument("--config", default="configs/dr.yaml")
    parser.add_argument("--run-name", default="dr", help="Subdirectory name under models_dir")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger = get_logger("train_dr_main", log_dir=cfg["output"]["logs_dir"])
    logger.info("Starting DR training run: %s (seed=%d)", args.run_name, cfg["training"]["seed"])

    ckpt_path = train_dr(cfg, run_name=args.run_name)
    logger.info("Best checkpoint: %s", ckpt_path)


if __name__ == "__main__":
    main()
