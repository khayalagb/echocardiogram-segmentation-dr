#!/usr/bin/env python
"""Create patient-level data split and verify no cross-split leak."""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from echo_seg.data import create_manifest, verify_no_leak
from echo_seg.utils import get_logger


def main():
    parser = argparse.ArgumentParser(description="Build manifest CSV with patient-level splits")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger = get_logger("prepare_split", log_dir=cfg["output"]["logs_dir"])

    data_cfg = cfg["data"]
    manifest_path = cfg["output"]["manifest_path"]

    logger.info("Scanning data root: %s", data_cfg["root"])
    counts = create_manifest(
        data_root=data_cfg["root"],
        output_path=manifest_path,
        seed=data_cfg["split_seed"],
        train_frac=data_cfg["train_fraction"],
        val_frac=data_cfg["val_fraction"],
    )
    logger.info("Manifest written to %s", manifest_path)
    logger.info("Split counts: train=%d | val=%d | test=%d | skipped=%d",
                counts["train"], counts["val"], counts["test"], counts.get("skipped", 0))

    logger.info("Running leak test...")
    verify_no_leak(manifest_path)
    logger.info("Leak test PASSED — no patient appears in multiple splits")


if __name__ == "__main__":
    main()
