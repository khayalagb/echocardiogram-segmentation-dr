"""Shifted robustness evaluation loop.

Corruption functions live in echo_seg.corruptions; imported here for the
evaluation loop and re-exported for callers that previously imported from here.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from echo_seg.corruptions import (  # noqa: F401 — re-export
    CORRUPTIONS,
    CORRUPTIONS_HELD_OUT,
    CORRUPTIONS_IN_DR_TRAINING,
)
from echo_seg.evaluation.analysis import run_evaluation
from echo_seg.evaluation.metrics import bootstrap_ci

STRUCTURES = ("lv_cavity", "myocardium", "left_atrium")


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
