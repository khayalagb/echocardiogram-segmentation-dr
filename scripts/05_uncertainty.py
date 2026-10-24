#!/usr/bin/env python
"""Ensemble uncertainty analysis and risk-coverage curve generation."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from echo_seg.uncertainty import run_ensemble_evaluation, risk_coverage_curve
from echo_seg.utils import get_logger


def _find_checkpoints(models_dir: Path, run_names: list[str]) -> list[str]:
    paths = []
    for name in run_names:
        ckpt = models_dir / name / "best.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        paths.append(str(ckpt))
    return paths


def main():
    parser = argparse.ArgumentParser(description="Ensemble uncertainty and risk-coverage analysis")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--run-names",
        nargs="+",
        default=["baseline", "ensemble_01", "ensemble_02", "ensemble_03", "ensemble_04"],
        help="Model run names under models/ dir",
    )
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger = get_logger("uncertainty", log_dir=cfg["output"]["logs_dir"])
    models_dir = Path(cfg["output"]["models_dir"])
    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = _find_checkpoints(models_dir, args.run_names)
    logger.info("Using %d ensemble members", len(checkpoints))

    df = run_ensemble_evaluation(cfg, checkpoints, split=args.split)

    # Risk-coverage for each structure (error = 1 - Dice)
    coverage_results = {}
    structures = ["lv_cavity", "myocardium", "left_atrium"]
    n_steps = cfg["uncertainty"]["coverage_steps"]

    fig, axes = plt.subplots(1, len(structures), figsize=(15, 4))
    for ax, name in zip(axes, structures):
        errors = 1 - df[f"dice_{name}"].values
        uncertainties = df["uncertainty"].values
        cov, risk = risk_coverage_curve(uncertainties, errors, n_steps=n_steps)

        from scipy.stats import spearmanr
        rho, p = spearmanr(uncertainties, errors)
        coverage_results[name] = {
            "spearman_rho": float(rho),
            "spearman_p": float(p),
            "coverage": cov.tolist(),
            "risk": risk.tolist(),
        }
        ax.plot(cov, risk, marker="o", linewidth=2)
        ax.set_xlabel("Coverage (fraction retained)")
        ax.set_ylabel("Mean error (1 - Dice)")
        ax.set_title(f"{name}\nSpearman ρ={rho:.3f} (p={p:.3f})")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)

    fig.suptitle("Risk-Coverage Curves (ensemble)", fontsize=13)
    fig.tight_layout()
    out_fig = results_dir / f"{args.split}_risk_coverage.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Risk-coverage figure saved: %s", out_fig)

    with open(results_dir / f"{args.split}_uncertainty_summary.json", "w") as f:
        json.dump(coverage_results, f, indent=2)

    for name, res in coverage_results.items():
        logger.info(
            "%s | Spearman ρ=%.3f (p=%.3f) | risk at full coverage=%.3f",
            name, res["spearman_rho"], res["spearman_p"], res["risk"][0],
        )


if __name__ == "__main__":
    main()
