#!/usr/bin/env python
"""Evaluate a checkpoint across all corruption conditions.

Runs six conditions (clean, gain_high, speckle, blur, attenuation, low_res)
on the test split (or --split val for development). Saves per-condition CSVs
and a combined shift_summary.json under results/. Prints a summary table.

Runtime: ~15 min on Mac MPS (inference only, no training).
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from echo_seg.evaluation import run_all_shifts
from echo_seg.utils import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Shifted test-set robustness evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger = get_logger("robustness", log_dir=cfg["output"]["logs_dir"])
    logger.info("Robustness eval | checkpoint=%s | split=%s", args.checkpoint, args.split)

    if args.split == "test":
        logger.warning("Opening test split. Use only after configuration is locked.")

    combined = run_all_shifts(cfg, args.checkpoint, split=args.split)

    # Print summary table
    structures = ("lv_cavity", "myocardium", "left_atrium")
    header = f"{'Condition':<14}" + "".join(f"  {s[:8]:>10}" for s in structures)
    print("\n" + header)
    print("-" * len(header))
    for condition, cdf in combined.groupby("condition", sort=False):
        row = f"{condition:<14}"
        for s in structures:
            row += f"  {cdf[f'dice_{s}'].mean():>10.3f}"
        print(row)

    logger.info(
        "Results saved to %s/shift_*.csv and shift_summary.json",
        cfg["output"]["results_dir"],
    )


if __name__ == "__main__":
    main()
