"""Test-time corruption functions for robustness evaluation.

All functions operate on (B, C, H, W) float32 tensors in [0, 1].
Caller must set np.random.seed before calling stochastic functions.

Overlap with DR training augmentations: gain_high, speckle, blur, attenuation.
Held-out (not in any training): low_res, speckle_corr.
"""

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


def _gain_high(img: torch.Tensor) -> torch.Tensor:
    return (img * 1.5).clamp(0, 1)


def _speckle(img: torch.Tensor) -> torch.Tensor:
    # Rayleigh(0.798) → mean≈1.0, std≈0.47; spatially i.i.d.
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
    # factor fixed at 4 (64×64 effective resolution on 256×256 input); held-out
    h, w = img.shape[-2], img.shape[-1]
    down = F.avg_pool2d(img, 4)
    return F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)


def _speckle_corr(img: torch.Tensor) -> torch.Tensor:
    # Held-out: spatially correlated speckle (NOT used in DR training).
    # Complex Gaussian envelope σ=1.5 px, scaled so multiplier std≈0.30, mean≈1.0.
    shape = img.shape
    real = np.random.standard_normal(shape).astype(np.float32)
    imag = np.random.standard_normal(shape).astype(np.float32)
    real = gaussian_filter(real, sigma=[0, 0, 1.5, 1.5])
    imag = gaussian_filter(imag, sigma=[0, 0, 1.5, 1.5])
    envelope = np.sqrt(real ** 2 + imag ** 2)
    envelope /= (envelope.mean() + 1e-8)
    s = 0.30 / (envelope.std() + 1e-8)
    multiplier = (1.0 + s * (envelope - 1.0)).astype(np.float32)
    return (img * torch.from_numpy(multiplier).to(img.device)).clamp(0, 1)


CORRUPTIONS: dict[str, Callable | None] = {
    "clean": None,
    "gain_high": _gain_high,
    "speckle": _speckle,
    "blur": _blur,
    "attenuation": _attenuation,
    "low_res": _low_res,
    "speckle_corr": _speckle_corr,
}

CORRUPTIONS_IN_DR_TRAINING = frozenset({"gain_high", "speckle", "blur", "attenuation"})
CORRUPTIONS_HELD_OUT = frozenset({"low_res", "speckle_corr"})
