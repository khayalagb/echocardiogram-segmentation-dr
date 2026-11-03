"""PyTorch Dataset for CAMUS echocardiography images."""

import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
import albumentations as A


class CamusDataset(Dataset):
    """Loads CAMUS images and segmentation masks from a manifest CSV.

    Returns dicts with keys:
        image   – float32 tensor (1, H, W) in [0, 1]
        mask    – int64 tensor (H, W) with values 0..num_classes-1
        meta    – dict with patient_id, view, phase, quality, spacing
    """

    def __init__(
        self,
        manifest_path: str,
        split: str,
        transform: A.Compose | None = None,
    ) -> None:
        self.transform = transform
        self.rows = []
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        img_arr, spacing = _load_image(row["image"])
        mask_arr = _load_mask(row["mask"])

        # albumentations expects HWC uint8/float32; mask as HW int
        if self.transform is not None:
            aug = self.transform(image=img_arr, mask=mask_arr)
            img_t = aug["image"].float()          # (1, H, W) from ToTensorV2
            mask_t = aug["mask"].long()           # (H, W)
        else:
            img_t = torch.from_numpy(img_arr).unsqueeze(0).float()
            mask_t = torch.from_numpy(mask_arr).long()

        return {
            "image": img_t,
            "mask": mask_t,
            "meta": {
                "patient_id": row["patient_id"],
                "view": row["view"],
                "phase": row["phase"],
                "quality": row["quality"],
                "spacing": spacing,
                "image_path": row["image"],
            },
        }


def _load_image(path: str) -> tuple[np.ndarray, tuple[float, float]]:
    """Load a 2D grayscale image; return (H, W) float32 array in [0,1] and (y,x) spacing in mm."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # shape: (slices, H, W) or (H, W)
    if arr.ndim == 3:
        arr = arr[0]  # CAMUS MHD stores 2D frames as (1, H, W)
    arr = arr.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    spacing_raw = img.GetSpacing()  # (x, y) in mm, possibly (x, y, z)
    spacing = (float(spacing_raw[1]), float(spacing_raw[0]))  # (y, x)
    return arr, spacing


def _load_mask(path: str) -> np.ndarray:
    """Load a 2D label mask; return (H, W) int32 array."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim == 3:
        arr = arr[0]
    return arr.astype(np.int32)
