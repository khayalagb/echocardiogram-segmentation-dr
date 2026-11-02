#!/usr/bin/env python
"""Visualise domain-randomisation augmentation on real CAMUS images.

Produces a grid: each row is one original image; columns show the original
plus four independent DR-augmented versions with the predicted sector mask
overlaid (from the baseline checkpoint).

Saves: results/dr_preview.png

Example:
    python scripts/07_dr_preview.py --config configs/dr.yaml --n-images 4
"""

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from echo_seg.data import CamusDataset, build_dr_transform, build_val_transform
from echo_seg.models import build_model
from echo_seg.utils import get_logger

_COLOURS = {
    1: (0.20, 0.60, 1.00, 0.45),  # LV cavity — blue
    2: (1.00, 0.55, 0.10, 0.45),  # Myocardium — orange
    3: (0.20, 0.80, 0.20, 0.45),  # Left atrium — green
}
_LABELS = {1: "LV cavity", 2: "Myocardium", 3: "Left atrium"}
_N_AUGMENTED = 4


def _overlay(ax, img_np: np.ndarray, mask_np: np.ndarray) -> None:
    ax.imshow(img_np, cmap="gray", vmin=0, vmax=1)
    for cls, colour in _COLOURS.items():
        overlay = np.zeros((*img_np.shape, 4), dtype=np.float32)
        overlay[mask_np == cls] = colour
        ax.imshow(overlay)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dr.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to baseline best.pth; if omitted, no mask overlay")
    parser.add_argument("--n-images", type=int, default=4,
                        help="Number of images to show (rows)")
    parser.add_argument("--split", default="val",
                        help="Dataset split to sample from (train/val)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger = get_logger("dr_preview", log_dir=cfg["output"]["logs_dir"])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    aug_cfg = cfg["augmentation"]
    img_size = tuple(cfg["data"]["image_size"])

    val_ds = CamusDataset(cfg["output"]["manifest_path"], args.split,
                          transform=build_val_transform(img_size))
    dr_transform = build_dr_transform(
        image_size=img_size,
        rotation_limit=aug_cfg["rotation_limit"],
        shift_limit=aug_cfg["shift_limit"],
        scale_limit=aug_cfg["scale_limit"],
        gamma_limit=tuple(aug_cfg["gamma_limit"]),
        brightness_limit=aug_cfg["brightness_limit"],
        speckle_s_range=tuple(aug_cfg["speckle_s_range"]),
        blur_sigma_axial=tuple(aug_cfg["blur_sigma_axial"]),
        blur_sigma_lateral=tuple(aug_cfg["blur_sigma_lateral"]),
        attenuation_top=tuple(aug_cfg["attenuation_top"]),
        attenuation_bot=tuple(aug_cfg["attenuation_bot"]),
    )

    model = None
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model_cfg = cfg["model"]
        model = build_model(
            encoder=model_cfg["encoder"],
            encoder_weights=None,
            in_channels=model_cfg["in_channels"],
            num_classes=model_cfg["num_classes"],
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        logger.info("Loaded checkpoint: %s", args.checkpoint)
    else:
        logger.info("No checkpoint given — showing ground-truth mask instead")

    n_rows = min(args.n_images, len(val_ds))
    n_cols = 1 + _N_AUGMENTED  # original + augmented versions
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    indices = np.linspace(0, len(val_ds) - 1, n_rows, dtype=int)

    for row, idx in enumerate(indices):
        sample = val_ds[idx]
        img_tensor = sample["image"]       # (1, H, W)
        gt_mask = sample["mask"].numpy()   # (H, W)

        # raw numpy image for re-augmenting
        img_np = img_tensor.numpy()[0]     # (H, W), float32

        # original column
        if model is not None:
            with torch.no_grad():
                logits = model(img_tensor.unsqueeze(0).to(device))
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        else:
            pred = gt_mask

        _overlay(axes[row, 0], img_np, pred)
        if row == 0:
            axes[row, 0].set_title("Original", fontsize=9)

        # augmented columns
        for col in range(1, n_cols):
            aug = dr_transform(image=img_np.copy(), mask=gt_mask.copy())
            aug_img = aug["image"].numpy()[0]
            aug_mask = aug["mask"].numpy()

            if model is not None:
                aug_tensor = aug["image"].unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(aug_tensor)
                aug_pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            else:
                aug_pred = aug_mask

            _overlay(axes[row, col], aug_img, aug_pred)
            if row == 0:
                axes[row, col].set_title(f"DR aug {col}", fontsize=9)

    legend_patches = [
        mpatches.Patch(color=_COLOURS[c][:3], label=_LABELS[c], alpha=0.7)
        for c in _COLOURS
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Domain-randomisation augmentation preview\n"
        f"(mask = {'model prediction' if model else 'ground truth'})",
        fontsize=10,
    )
    fig.tight_layout()

    out_path = Path(cfg["output"]["results_dir"]) / "dr_preview.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Saved grid: %s", out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
