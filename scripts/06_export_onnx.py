#!/usr/bin/env python
"""Export the baseline model to ONNX and run a parity check against PyTorch output."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from echo_seg.models import build_model
from echo_seg.utils import get_logger


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX with parity check")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth")
    parser.add_argument("--output", default=None, help="Output .onnx path (default: next to checkpoint)")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of random inputs for parity check")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger = get_logger("export_onnx", log_dir=cfg["output"]["logs_dir"])
    ckpt_path = Path(args.checkpoint)
    onnx_path = Path(args.output) if args.output else ckpt_path.with_suffix(".onnx")

    device = torch.device("cpu")  # export on CPU for portability
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
    logger.info("Loaded checkpoint from epoch %d (val_dice=%.4f)", ckpt.get("epoch", -1), ckpt.get("val_dice", float("nan")))

    h, w = cfg["data"]["image_size"]
    dummy = torch.randn(1, model_cfg["in_channels"], h, w)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version=17,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    logger.info("Exported ONNX model to %s", onnx_path)

    # Parity check
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rtol = cfg["evaluation"]["onnx_rtol"]
    atol = cfg["evaluation"]["onnx_atol"]

    max_abs_err = 0.0
    pt_latencies, ort_latencies = [], []

    for _ in range(args.n_samples):
        x = torch.randn(1, model_cfg["in_channels"], h, w)
        t0 = time.perf_counter()
        with torch.no_grad():
            pt_out = model(x).numpy()
        pt_latencies.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        ort_out = sess.run(None, {"image": x.numpy()})[0]
        ort_latencies.append(time.perf_counter() - t0)

        max_abs_err = max(max_abs_err, float(np.abs(pt_out - ort_out).max()))
        np.testing.assert_allclose(pt_out, ort_out, rtol=rtol, atol=atol)

    report = {
        "checkpoint": str(ckpt_path),
        "onnx_path": str(onnx_path),
        "n_parity_samples": args.n_samples,
        "max_abs_error": max_abs_err,
        "rtol": rtol,
        "atol": atol,
        "parity_passed": True,
        "pytorch_latency_ms": {"mean": np.mean(pt_latencies) * 1000, "p95": np.percentile(pt_latencies, 95) * 1000},
        "onnxruntime_latency_ms": {"mean": np.mean(ort_latencies) * 1000, "p95": np.percentile(ort_latencies, 95) * 1000},
    }
    report_path = ckpt_path.parent / "onnx_parity_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Parity PASSED | max_abs_err=%.2e | rtol=%.1e atol=%.1e", max_abs_err, rtol, atol)
    logger.info("PyTorch latency: mean=%.1f ms p95=%.1f ms", report["pytorch_latency_ms"]["mean"], report["pytorch_latency_ms"]["p95"])
    logger.info("ORT latency:     mean=%.1f ms p95=%.1f ms", report["onnxruntime_latency_ms"]["mean"], report["onnxruntime_latency_ms"]["p95"])
    logger.info("Report saved: %s", report_path)


if __name__ == "__main__":
    main()
