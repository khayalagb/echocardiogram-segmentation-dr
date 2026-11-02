"""Test-time corruption functions and shifted robustness evaluation loop.

Seven conditions:
  clean         — no modification (baseline reference)
  gain_high     — uniform over-gain (+50 %)
  speckle       — multiplicative Rayleigh noise; Rayleigh(σ=0.798) has mean≈1.0,
                  std≈0.52 — severe shift; spatially uncorrelated (i.i.d. per pixel).
  blur          — anisotropic Gaussian PSF (axial σ=2, lateral σ=1)
  attenuation   — depth-linear gain ramp (1.2 → 0.5)
  low_res       — ×4 downsample/upsample           [HELD-OUT: not in DR training]
  speckle_corr  — spatially correlated speckle      [HELD-OUT: not in DR training]
                  Complex Gaussian envelope (σ=1.5 px), renormalized, then scaled to
                  multiplier std≈0.30. Redesigned after seeing baseline-only val results
                  (before any DR results): original σ=4 smoothing reduced std to 0.04,
                  making it a near-no-op.

Overlap with DR training augmentations: gain_high, speckle, blur, attenuation.
Held-out (unplanned shifts): low_res, speckle_corr, CAMUS quality tiers.

Multiplier statistics (measured on 256×256, seed 42):
  speckle:      mean=1.001, std=0.523
  speckle_corr: mean=1.000, std=0.300 (calibrated)

Severities are fixed after seeing baseline-only val results, before any DR results.
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
    # Rayleigh(0.798) → mean≈1.0, std≈0.47; np global RNG seeded by caller
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
    # factor fixed at 4 (64×64 effective resolution on 256×256 input); held-out condition
    h, w = img.shape[-2], img.shape[-1]
    down = F.avg_pool2d(img, 4)
    return F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)


def _speckle_corr(img: torch.Tensor) -> torch.Tensor:
    # Held-out: spatially correlated speckle (NOT used in DR training).
    # Complex Gaussian envelope with σ=1.5 px spatial correlation, scaled so
    # multiplier std≈0.30 and mean≈1.0. Always positive (min ~0.43 empirically).
    shape = img.shape
    real = np.random.standard_normal(shape).astype(np.float32)
    imag = np.random.standard_normal(shape).astype(np.float32)
    real = gaussian_filter(real, sigma=[0, 0, 1.5, 1.5])
    imag = gaussian_filter(imag, sigma=[0, 0, 1.5, 1.5])
    envelope = np.sqrt(real ** 2 + imag ** 2)
    envelope /= (envelope.mean() + 1e-8)          # mean ≈ 1
    s = 0.30 / (envelope.std() + 1e-8)            # scale to std ≈ 0.30
    multiplier = (1.0 + s * (envelope - 1.0)).astype(np.float32)
    return (img * torch.from_numpy(multiplier).to(img.device)).clamp(0, 1)


# Keys with [HELD-OUT] suffix in value are NOT used in DR training augmentation.
CORRUPTIONS: dict[str, Callable | None] = {
    "clean": None,
    "gain_high": _gain_high,
    "speckle": _speckle,
    "blur": _blur,
    "attenuation": _attenuation,
    "low_res": _low_res,           # held-out
    "speckle_corr": _speckle_corr, # held-out
}

# Explicit sets for DR training overlap documentation
CORRUPTIONS_IN_DR_TRAINING = frozenset({"gain_high", "speckle", "blur", "attenuation"})
CORRUPTIONS_HELD_OUT = frozenset({"low_res", "speckle_corr"})


# ---------------------------------------------------------------------------
# Shifted evaluation loop
# ---------------------------------------------------------------------------

def run_all_shifts(
    cfg: dict,
    checkpoint_path: str,
    split: str = "test",
    model_name: str = "baseline",
    seed: int = 42,
) -> pd.DataFrame:
    """Run run_evaluation for every corruption condition.

    Output files are named shift_{model_name}_{split}_{condition}_per_case.csv
    so multiple arms never overwrite each other.
    Also saves shift_{model_name}_{split}_combined_per_case.csv and
    shift_{model_name}_{split}_summary.json.
    Returns the combined per-case DataFrame (with 'condition' and 'model' columns).
    """
    prefix = f"shift_{model_name}_{split}"
    frames = []
    for name, fn in CORRUPTIONS.items():
        np.random.seed(seed)  # reproducible per-condition noise
        df = run_evaluation(
            cfg,
            checkpoint_path,
            split=split,
            corrupt_fn=fn,
            output_label=f"{prefix}_{name}",
        )
        df.insert(0, "model", model_name)
        df.insert(1, "condition", name)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    results_dir = Path(cfg["output"]["results_dir"])
    combined.to_csv(results_dir / f"{prefix}_combined_per_case.csv", index=False)

    summary = _build_shift_summary(combined, cfg)
    with open(results_dir / f"{prefix}_summary.json", "w") as f:
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
