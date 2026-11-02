"""Test-time corruption functions and shifted robustness evaluation loop.

Six conditions:
  clean        — no modification (baseline reference)
  gain_high    — uniform over-gain (+50 %)
  speckle      — multiplicative Rayleigh noise (physics-inspired)
  blur         — anisotropic Gaussian PSF (axial σ=2, lateral σ=1)
  attenuation  — depth-linear gain ramp (1.2 → 0.5)
  low_res      — ×4 downsample/upsample  [held-out: NOT used in DR training]

All functions operate on (B, C, H, W) float32 tensors in [0, 1].
"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from echo_seg.evaluation.analysis import run_evaluation
from echo_seg.evaluation.metrics import bootstrap_ci

STRUCTURES = ("lv_cavity", "myocardium", "left_atrium")


# ---------------------------------------------------------------------------
# Corruption functions
# ---------------------------------------------------------------------------

def _gain_high(img: torch.Tensor) -> torch.Tensor:
    return (img * 1.5).clamp(0, 1)


def _speckle(img: torch.Tensor) -> torch.Tensor:
    # Rayleigh scale=0.798 → mean≈1.0; np global RNG must be seeded by caller
    noise = np.random.rayleigh(scale=0.798, size=img.shape).astype(np.float32)
    return (img * torch.from_numpy(noise).to(img.device)).clamp(0, 1)


def _blur(img: torch.Tensor) -> torch.Tensor:
    arr = img.cpu().numpy()
    blurred = gaussian_filter(arr, sigma=[0, 0, 2, 1])
    return torch.from_numpy(blurred).to(img.device)


def _attenuation(img: torch.Tensor) -> torch.Tensor:
    H = img.shape[-2]
    ramp = torch.linspace(1.2, 0.5, H, dtype=img.dtype, device=img.device).view(1, 1, H, 1)
    return (img * ramp).clamp(0, 1)


def _low_res(img: torch.Tensor) -> torch.Tensor:
    # shortcut: factor=4 chosen empirically; increase if a coarser shift is needed
    h, w = img.shape[-2], img.shape[-1]
    down = F.avg_pool2d(img, 4)
    return F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)


CORRUPTIONS: dict[str, Callable | None] = {
    "clean": None,
    "gain_high": _gain_high,
    "speckle": _speckle,
    "blur": _blur,
    "attenuation": _attenuation,
    "low_res": _low_res,
}


# ---------------------------------------------------------------------------
# Shifted evaluation loop
# ---------------------------------------------------------------------------

def run_all_shifts(
    cfg: dict,
    checkpoint_path: str,
    split: str = "test",
    seed: int = 42,
) -> pd.DataFrame:
    """Run run_evaluation for every corruption condition.

    Saves results/shift_{condition}_per_case.csv per condition and
    results/shift_combined_per_case.csv + results/shift_summary.json overall.
    Returns the combined per-case DataFrame (with a 'condition' column).
    """
    frames = []
    for name, fn in CORRUPTIONS.items():
        np.random.seed(seed)  # reproducible per-condition noise
        df = run_evaluation(
            cfg,
            checkpoint_path,
            split=split,
            corrupt_fn=fn,
            output_label=f"shift_{name}",
        )
        df.insert(0, "condition", name)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    results_dir = Path(cfg["output"]["results_dir"])
    combined.to_csv(results_dir / "shift_combined_per_case.csv", index=False)

    summary = _build_shift_summary(combined, cfg)
    with open(results_dir / "shift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return combined


def _build_shift_summary(df: pd.DataFrame, cfg: dict) -> dict:
    """Per-condition × structure mean Dice + CI; also stratified by quality."""
    bs_n = cfg["evaluation"]["bootstrap_n"]
    bs_seed = cfg["evaluation"]["bootstrap_seed"]
    out: dict = {"by_condition": {}, "by_condition_quality": {}}

    for condition, cdf in df.groupby("condition"):
        out["by_condition"][condition] = {}
        for name in STRUCTURES:
            vals = cdf[f"dice_{name}"].dropna().tolist()
            mean, lo, hi = bootstrap_ci(vals, n=bs_n, seed=bs_seed)
            out["by_condition"][condition][name] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}

        out["by_condition_quality"][condition] = {}
        for quality, qdf in cdf.groupby("quality"):
            out["by_condition_quality"][condition][quality] = {}
            for name in STRUCTURES:
                vals = qdf[f"dice_{name}"].dropna().tolist()
                if len(vals) >= 2:
                    mean, lo, hi = bootstrap_ci(vals, n=bs_n, seed=bs_seed)
                else:
                    mean, lo, hi = float(np.mean(vals)), float("nan"), float("nan")
                out["by_condition_quality"][condition][quality][name] = {
                    "mean": mean, "ci_lo": lo, "ci_hi": hi,
                }

    return out
