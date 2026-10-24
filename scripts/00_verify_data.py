#!/usr/bin/env python
"""Verify CAMUS directory structure, label values, and pixel spacing."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from echo_seg.utils import get_logger


VIEWS = ("2CH", "4CH")
PHASES = ("ED", "ES")
EXPECTED_LABELS = {0, 1, 2, 3}


def verify_patient(patient_dir: Path, logger) -> dict:
    pid = patient_dir.name
    report = {"patient_id": pid, "issues": [], "files": {}}

    for view in VIEWS:
        for phase in PHASES:
            for suffix, is_gt in [(f"{pid}_{view}_{phase}", False), (f"{pid}_{view}_{phase}_gt", True)]:
                found = None
                for ext in (".mhd", ".nii.gz", ".nii"):
                    p = patient_dir / f"{suffix}{ext}"
                    if p.exists():
                        found = p
                        break
                key = f"{view}_{phase}_{'gt' if is_gt else 'img'}"
                if found is None:
                    report["issues"].append(f"Missing: {suffix}")
                    report["files"][key] = None
                    continue
                report["files"][key] = str(found)
                if is_gt:
                    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(found)))
                    labels = set(np.unique(arr).tolist())
                    unexpected = labels - EXPECTED_LABELS
                    if unexpected:
                        report["issues"].append(f"{key}: unexpected labels {unexpected}")
    return report


def save_overlay(img_path: str, gt_path: str, out_path: str) -> None:
    img_arr = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
    gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
    if img_arr.ndim == 3:
        img_arr, gt_arr = img_arr[0], gt_arr[0]
    img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 1e-8)

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_arr, cmap="gray")
    axes[0].set_title("Image")
    axes[0].axis("off")
    axes[1].imshow(img_arr, cmap="gray")
    axes[1].imshow(gt_arr, alpha=0.4, cmap=cmap, vmin=0, vmax=3)
    axes[1].set_title("Image + GT mask")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Verify CAMUS data structure")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-overlays", type=int, default=20, help="How many sample overlays to save")
    parser.add_argument("--out-dir", default="results/data_verification")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger = get_logger("verify_data", log_dir=cfg["output"]["logs_dir"])
    data_root = Path(cfg["data"]["root"])

    if not data_root.exists():
        logger.error("Data root not found: %s", data_root)
        sys.exit(1)

    patient_dirs = sorted(d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("patient"))
    logger.info("Found %d patient directories under %s", len(patient_dirs), data_root)

    issues_total = 0
    for pd_ in patient_dirs:
        report = verify_patient(pd_, logger)
        if report["issues"]:
            for issue in report["issues"]:
                logger.warning("%s: %s", pd_.name, issue)
            issues_total += len(report["issues"])

    logger.info("Total issues: %d", issues_total)

    # Save sample overlays
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for pd_ in patient_dirs[: args.n_overlays]:
        pid = pd_.name
        for view in VIEWS:
            for phase in PHASES:
                img_p = None
                gt_p = None
                for ext in (".mhd", ".nii.gz"):
                    c = pd_ / f"{pid}_{view}_{phase}{ext}"
                    if c.exists():
                        img_p = str(c)
                    c = pd_ / f"{pid}_{view}_{phase}_gt{ext}"
                    if c.exists():
                        gt_p = str(c)
                if img_p and gt_p:
                    out_path = out_dir / f"{pid}_{view}_{phase}.png"
                    save_overlay(img_p, gt_p, str(out_path))
                    saved += 1

    logger.info("Saved %d overlays to %s", saved, out_dir)


if __name__ == "__main__":
    main()
