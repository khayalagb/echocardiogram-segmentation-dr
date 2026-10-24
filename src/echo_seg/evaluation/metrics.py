"""Per-case segmentation metrics with patient-level bootstrap confidence intervals."""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice similarity coefficient for binary masks."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = (pred_b & gt_b).sum()
    denom = pred_b.sum() + gt_b.sum()
    if denom == 0:
        return 1.0  # both empty → perfect agreement
    return float(2 * inter / denom)


def hd95(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, float] = (1.0, 1.0),
) -> float:
    """95th-percentile Hausdorff distance in physical units (mm).

    Returns np.inf when either mask is empty (no structure predicted or annotated).
    """
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    if not pred_b.any() or not gt_b.any():
        return float("inf")

    # Surface = mask XOR eroded mask
    pred_surf = pred_b ^ binary_erosion(pred_b)
    gt_surf = gt_b ^ binary_erosion(gt_b)

    dt_pred = distance_transform_edt(~pred_b, sampling=spacing)
    dt_gt = distance_transform_edt(~gt_b, sampling=spacing)

    d1 = dt_gt[pred_surf]
    d2 = dt_pred[gt_surf]
    return float(np.percentile(np.concatenate([d1, d2]), 95))


def bootstrap_ci(
    values: list[float],
    n: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap CI by resampling over cases (call once per patient group).

    Returns (mean, lower, upper).
    """
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)]
    )
    alpha = (1 - confidence) / 2
    return (
        float(arr.mean()),
        float(np.percentile(means, 100 * alpha)),
        float(np.percentile(means, 100 * (1 - alpha))),
    )
