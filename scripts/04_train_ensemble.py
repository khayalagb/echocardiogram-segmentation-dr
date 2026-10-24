#!/usr/bin/env python
"""Train remaining ensemble members (seeds 1–4); seed 0 is the baseline from 02_train.py."""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from echo_seg.training import train
from echo_seg.utils import get_logger


def main():
    parser = argparse.ArgumentParser(description="Train ensemble members 1-4")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--members",
        nargs="+",
        type=int,
        help="Indices into cfg.uncertainty.seeds to train (default: 1 2 3 4)",
        default=[1, 2, 3, 4],
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger = get_logger("train_ensemble", log_dir=cfg["output"]["logs_dir"])
    seeds = cfg["uncertainty"]["seeds"]

    for i in args.members:
        seed = seeds[i]
        run_name = f"ensemble_{i:02d}"
        cfg["training"]["seed"] = seed
        logger.info("Training ensemble member %d/%d (seed=%d, run=%s)",
                    i, len(seeds) - 1, seed, run_name)
        ckpt = train(cfg, run_name=run_name)
        logger.info("Member %d saved: %s", i, ckpt)

    logger.info("Ensemble training complete. Members: %s", args.members)


if __name__ == "__main__":
    main()
