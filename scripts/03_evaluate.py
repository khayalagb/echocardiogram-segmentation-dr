#!/usr/bin/env python
"""Evaluate a trained checkpoint on the held-out test split (use once, after config is frozen)."""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from echo_seg.evaluation import run_evaluation
from echo_seg.utils import get_logger


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on test split")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth")
    parser.add_argument("--split", default="test", choices=["val", "test"],
                        help="Use 'val' for development; open test split only once")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger = get_logger("evaluate_main", log_dir=cfg["output"]["logs_dir"])
    logger.info("Evaluating on split=%s with checkpoint=%s", args.split, args.checkpoint)

    if args.split == "test":
        logger.warning("Opening test split. Use this only after the final configuration is locked.")

    run_evaluation(cfg, args.checkpoint, split=args.split)
    logger.info("Per-case results saved to %s/%s_per_case.csv", cfg["output"]["results_dir"], args.split)


if __name__ == "__main__":
    main()
