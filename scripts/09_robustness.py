#!/usr/bin/env python
"""Evaluate a checkpoint across all corruption conditions.

Runs seven conditions (clean, gain_high, speckle, blur, attenuation,
low_res, speckle_corr) on the chosen split. Output files are named
shift_{model_name}_{split}_{condition}_per_case.csv so multiple arms
never overwrite each other.

Saves per-condition CSVs, a combined CSV, and a summary JSON under results/.
Prints a summary table to stdout.

Runtime: ~20 min on Mac MPS (inference only, no training).

Example:
    python scripts/09_robustness.py \\
        --checkpoint models.nosync/baseline/best.pth \\
        --model-name baseline --split val
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
    parser.add_argument("--model-name", required=True,
                        help="Short label for this arm (e.g. baseline, dr). "
                             "Used in output file names.")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger = get_logger("robustness", log_dir=cfg["output"]["logs_dir"])
    logger.info(
        "Robustness eval | model=%s | checkpoint=%s | split=%s",
        args.model_name, args.checkpoint, args.split,
    )
    if args.split == "test":
        logger.warning("Opening test split. Use only after configuration is locked.")

    combined = run_all_shifts(
        cfg, args.checkpoint, split=args.split, model_name=args.model_name,
    )

    # Print summary table
    structures = ("lv_cavity", "myocardium", "left_atrium")
    header = f"{'Condition':<16}" + "".join(f"  {s[:8]:>10}" for s in structures)
    print(f"\n[{args.model_name} / {args.split}]")
    print(header)
    print("-" * len(header))
    condition_order = list(combined["condition"].unique())  # insertion order via dict
    for condition in condition_order:
        cdf = combined[combined["condition"] == condition]
        row = f"{condition:<16}"
        for s in structures:
            row += f"  {cdf[f'dice_{s}'].mean():>10.3f}"
        print(row)

    prefix = f"shift_{args.model_name}_{args.split}"
    logger.info(
        "Results saved to results/%s_combined_per_case.csv and results/%s_summary.json",
        prefix, prefix,
    )


if __name__ == "__main__":
    main()
