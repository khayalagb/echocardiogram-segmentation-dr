"""Deep ensemble inference, uncertainty scoring, and risk-coverage analysis."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from echo_seg.data import CamusDataset, build_val_transform
from echo_seg.evaluation.metrics import dice_score
from echo_seg.models import build_model
from echo_seg.utils import get_logger

STRUCTURES = {1: "lv_cavity", 2: "myocardium", 3: "left_atrium"}


class EnsemblePredictor:
    """Loads multiple checkpoints and averages softmax probabilities."""

    def __init__(self, checkpoint_paths: list[str], device: str | None = None) -> None:
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.models = []
        for path in checkpoint_paths:
            ckpt = torch.load(path, map_location=self.device)
            cfg = ckpt["cfg"]
            m = build_model(
                encoder=cfg["model"]["encoder"],
                encoder_weights=None,
                in_channels=cfg["model"]["in_channels"],
                num_classes=cfg["model"]["num_classes"],
            ).to(self.device)
            m.load_state_dict(ckpt["model_state"])
            m.eval()
            self.models.append(m)

    @torch.no_grad()
    def predict_proba(self, img: torch.Tensor) -> np.ndarray:
        """Return mean softmax probabilities (C, H, W) averaged over ensemble members."""
        probs = []
        for m in self.models:
            logits = m(img.to(self.device))
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
        return np.mean(probs, axis=0)[0]  # (C, H, W)


def mean_pixel_entropy(probs: np.ndarray) -> float:
    """Mean Shannon entropy over all pixels; higher = more uncertain.

    probs: (C, H, W) softmax probabilities summing to 1 along C.
    """
    eps = 1e-8
    entropy_map = -(probs * np.log(probs + eps)).sum(axis=0)  # (H, W)
    return float(entropy_map.mean())


def run_ensemble_evaluation(cfg: dict, checkpoint_paths: list[str], split: str = "test") -> dict:
    """Run ensemble on *split*; return per-case dict with uncertainty and Dice.

    Saves results/{split}_uncertainty.csv.
    """
    import json
    import pandas as pd
    from pathlib import Path as P

    logger = get_logger("ensemble", log_dir=cfg["output"]["logs_dir"])
    logger.info("Ensemble of %d members on split=%s", len(checkpoint_paths), split)

    ensemble = EnsemblePredictor(checkpoint_paths)
    ds = CamusDataset(
        cfg["output"]["manifest_path"],
        split,
        transform=build_val_transform(tuple(cfg["data"]["image_size"])),
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    rows = []
    for batch in dl:
        img = batch["image"]
        mask_np = batch["mask"][0].numpy()
        meta = {k: v[0] if isinstance(v, (list, torch.Tensor)) else v for k, v in batch["meta"].items()}

        probs = ensemble.predict_proba(img)  # (C, H, W)
        pred_np = probs.argmax(axis=0)       # (H, W)
        uncertainty = mean_pixel_entropy(probs)

        row = {
            "patient_id": meta["patient_id"],
            "view": meta["view"],
            "phase": meta["phase"],
            "quality": meta["quality"],
            "uncertainty": uncertainty,
        }
        for label, name in STRUCTURES.items():
            row[f"dice_{name}"] = dice_score(pred_np == label, mask_np == label)
        rows.append(row)
        logger.debug("Patient %s %s %s | uncertainty=%.4f", meta["patient_id"], meta["view"], meta["phase"], uncertainty)

    df = pd.DataFrame(rows)
    results_dir = P(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / f"{split}_uncertainty.csv", index=False)
    logger.info("Saved uncertainty results to %s", results_dir / f"{split}_uncertainty.csv")
    return df


def risk_coverage_curve(
    uncertainties: np.ndarray,
    errors: np.ndarray,
    n_steps: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute risk-coverage pairs by rejecting the most uncertain cases.

    Args:
        uncertainties: per-case uncertainty scores (higher = less confident)
        errors: per-case error values (e.g. 1 - Dice; higher = worse)
        n_steps: number of coverage thresholds

    Returns:
        coverage: fraction of cases retained (1.0 → keep all, 0.0 → keep none)
        risk: mean error of retained cases at each threshold
    """
    order = np.argsort(uncertainties)  # ascending: keep most certain first
    coverages = np.linspace(1.0, 1.0 / n_steps, n_steps)
    risks = []
    for cov in coverages:
        k = max(1, int(round(cov * len(order))))
        kept = order[:k]
        risks.append(errors[kept].mean())
    return coverages, np.array(risks)
