#!/usr/bin/env python
"""Train a U-Net segmentation model on the CAMUS training split."""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from echo_seg.training import train
from echo_seg.utils import get_logger


def main():
    parser = argparse.ArgumentParser(description="Train U-Net on CAMUS")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override training seed")
    parser.add_argument("--run-name", default="baseline", help="Subdirectory name under models/")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["training"]["seed"] = args.seed

    logger = get_logger("train_main", log_dir=cfg["output"]["logs_dir"])
    logger.info("Starting training run: %s (seed=%d)", args.run_name, cfg["training"]["seed"])

    ckpt_path = train(cfg, run_name=args.run_name)
    logger.info("Best checkpoint: %s", ckpt_path)


if __name__ == "__main__":
    main()
