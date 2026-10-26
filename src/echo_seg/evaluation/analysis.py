"""Stratified evaluation over the test set; saves per-case CSV and summary tables."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from echo_seg.data import CamusDataset, build_val_transform
from echo_seg.evaluation.metrics import bootstrap_ci, dice_score, hd95
from echo_seg.models import build_model
from echo_seg.utils import get_logger

STRUCTURES = {1: "lv_cavity", 2: "myocardium", 3: "left_atrium"}


def run_evaluation(cfg: dict, checkpoint_path: str, split: str = "test") -> pd.DataFrame:
    """Evaluate a checkpoint on *split*; return per-case metrics DataFrame.

    Saves results/{split}_per_case.csv and results/{split}_summary.json.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger = get_logger("evaluate", log_dir=cfg["output"]["logs_dir"])
    logger.info("Evaluating checkpoint %s on split=%s", checkpoint_path, split)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_cfg = cfg["model"]
    model = build_model(
        encoder=model_cfg["encoder"],
        encoder_weights=None,  # weights come from checkpoint
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = CamusDataset(
        cfg["output"]["manifest_path"],
        split,
        transform=build_val_transform(tuple(cfg["data"]["image_size"])),
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    logger.info("Evaluating %d samples", len(ds))

    rows = []
    latencies = []
    for batch in dl:
        img = batch["image"].to(device)
        mask_np = batch["mask"][0].numpy()
        meta = {k: v[0] if isinstance(v, (list, torch.Tensor)) else v for k, v in batch["meta"].items()}

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(img)
        latencies.append(time.perf_counter() - t0)

        pred_np = logits.argmax(dim=1)[0].cpu().numpy()
        sp = meta["spacing"]
        spacing = (float(sp[0]), float(sp[1])) if len(sp) >= 2 else (float(sp[0]), float(sp[0]))

        row = {
            "patient_id": meta["patient_id"],
            "view": meta["view"],
            "phase": meta["phase"],
            "quality": meta["quality"],
        }
        for label, name in STRUCTURES.items():
            row[f"dice_{name}"] = dice_score(pred_np == label, mask_np == label)
            row[f"hd95_{name}"] = hd95(pred_np == label, mask_np == label, spacing)
        rows.append(row)

    df = pd.DataFrame(rows)
    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / f"{split}_per_case.csv", index=False)

    summary = _build_summary(df, cfg)
    with open(results_dir / f"{split}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Mean Dice (LV/MYO/LA): %.3f / %.3f / %.3f",
        df["dice_lv_cavity"].mean(),
        df["dice_myocardium"].mean(),
        df["dice_left_atrium"].mean(),
    )
    logger.info(
        "Median latency: %.1f ms | p95: %.1f ms",
        np.median(latencies) * 1000,
        np.percentile(latencies, 95) * 1000,
    )
    return df


def _build_summary(df: pd.DataFrame, cfg: dict) -> dict:
    bs_cfg = cfg["evaluation"]
    summary: dict = {"overall": {}, "stratified": {}}

    for name in STRUCTURES.values():
        col = f"dice_{name}"
        mean, lo, hi = bootstrap_ci(
            df[col].dropna().tolist(),
            n=bs_cfg["bootstrap_n"],
            seed=bs_cfg["bootstrap_seed"],
        )
        hd_vals = df[f"hd95_{name}"].replace(float("inf"), np.nan).dropna().tolist()
        hd_mean, hd_lo, hd_hi = bootstrap_ci(hd_vals, n=bs_cfg["bootstrap_n"], seed=bs_cfg["bootstrap_seed"])
        summary["overall"][name] = {
            "dice": {"mean": mean, "ci_lo": lo, "ci_hi": hi},
            "hd95": {"mean": hd_mean, "ci_lo": hd_lo, "ci_hi": hd_hi},
        }

    # Stratify by view, phase, quality
    for strat_col in ("view", "phase", "quality"):
        summary["stratified"][strat_col] = {}
        for group_val, grp in df.groupby(strat_col):
            summary["stratified"][strat_col][group_val] = {}
            for name in STRUCTURES.values():
                vals = grp[f"dice_{name}"].dropna().tolist()
                if len(vals) >= 2:
                    m, lo, hi = bootstrap_ci(vals, n=bs_cfg["bootstrap_n"], seed=bs_cfg["bootstrap_seed"])
                else:
                    m, lo, hi = (float(np.mean(vals)), float("nan"), float("nan"))
                summary["stratified"][strat_col][group_val][name] = {
                    "dice_mean": m, "ci_lo": lo, "ci_hi": hi
                }

    # Worst cases per structure (bottom 10 by Dice)
    summary["worst_cases"] = {}
    for name in STRUCTURES.values():
        worst = df.nsmallest(10, f"dice_{name}")[
            ["patient_id", "view", "phase", "quality", f"dice_{name}", f"hd95_{name}"]
        ].to_dict(orient="records")
        summary["worst_cases"][name] = worst

    return summary
