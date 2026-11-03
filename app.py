"""Streamlit demo: echocardiogram segmentation via ONNX inference."""

import io
from pathlib import Path

import numpy as np
import onnxruntime as ort
import streamlit as st
import torch
from PIL import Image

from echo_seg.corruptions import CORRUPTIONS, CORRUPTIONS_HELD_OUT, CORRUPTIONS_IN_DR_TRAINING

# ── constants ────────────────────────────────────────────────────────────────
MODELS = {
    "Baseline": Path("models/baseline/best.onnx"),
    "DR": Path("models/dr/best.onnx"),
}
IMAGE_SIZE = (256, 256)

STRUCTURES = {
    1: ("LV cavity",   (220,  50,  50)),
    2: ("Myocardium",  ( 50, 200,  80)),
    3: ("Left atrium", ( 50, 120, 220)),
}

COLOURS = np.array(
    [
        [  0,   0,   0,   0],
        [220,  50,  50, 180],
        [ 50, 200,  80, 180],
        [ 50, 120, 220, 180],
    ],
    dtype=np.uint8,
)

CORRUPTION_LABELS = {
    "clean":        "Clean (no corruption)",
    "gain_high":    "Gain high (+50 %)",
    "speckle":      "Speckle (Rayleigh, i.i.d.)",
    "blur":         "Blur (anisotropic PSF)",
    "attenuation":  "Attenuation (depth ramp)",
    "low_res":      "Low resolution ×4  [held-out]",
    "speckle_corr": "Speckle corr. (correlated)  [held-out]",
}


# ── inference helpers ─────────────────────────────────────────────────────────
@st.cache_resource
def load_session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def preprocess(gray: np.ndarray) -> np.ndarray:
    # Normalise at original resolution first, then resize — matches dataset.py order.
    arr = gray.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    pil = Image.fromarray(arr, mode="F").resize(IMAGE_SIZE, Image.BILINEAR)
    return np.array(pil, dtype=np.float32)[np.newaxis, np.newaxis]  # (1,1,256,256)


def apply_corruption(arr_01: np.ndarray, condition: str) -> np.ndarray:
    """Apply named corruption to a (1,1,H,W) float32 array in [0,1]."""
    fn = CORRUPTIONS[condition]
    if fn is None:
        return arr_01
    np.random.seed(42)
    tensor = torch.from_numpy(arr_01)
    return fn(tensor).numpy()


def run_inference(sess: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    logits = sess.run(None, {"image": x})[0]  # (1,4,H,W)
    return logits[0].argmax(axis=0).astype(np.uint8)  # (H,W)


def make_overlay(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = np.stack([gray, gray, gray], axis=-1)
    colour_layer = COLOURS[mask, :3]
    alpha = COLOURS[mask, 3:4] / 255.0
    return (rgb * (1 - alpha) + colour_layer * alpha).clip(0, 255).astype(np.uint8)


def load_gt_mask(uploaded_file) -> np.ndarray | None:
    raw = Image.open(io.BytesIO(uploaded_file.read())).convert("L")
    arr = np.array(raw.resize(IMAGE_SIZE, Image.NEAREST), dtype=np.uint8)
    if arr.max() > 3:
        return None
    return arr


def dice_score(pred: np.ndarray, gt: np.ndarray, label: int) -> float | None:
    p = pred == label
    g = gt == label
    if g.sum() == 0 and p.sum() == 0:
        return None
    denom = p.sum() + g.sum()
    if denom == 0:
        return None
    return float(2 * (p & g).sum() / denom)


# ── page ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Echo segmentation", layout="wide")
st.title("Echocardiogram segmentation")
st.caption("ResNet34 U-Net · CAMUS dataset · ONNX Runtime inference")
st.warning(
    "Research demo only. Not for clinical use. Do not upload identifiable patient data.",
    icon=":material/warning:",
)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    corruption_key = st.selectbox(
        "Corruption condition",
        options=list(CORRUPTION_LABELS.keys()),
        format_func=lambda k: CORRUPTION_LABELS[k],
    )
    if corruption_key in CORRUPTIONS_HELD_OUT:
        st.info("Held-out condition: absent from both training augmentation sets.")
    elif corruption_key in CORRUPTIONS_IN_DR_TRAINING:
        st.info("In DR training augmentation set.")

    st.header("Colour legend")
    for _, (name, rgb) in STRUCTURES.items():
        swatch = np.full((20, 20, 3), rgb, dtype=np.uint8)
        c1, c2 = st.columns([1, 4])
        c1.image(swatch, width=20)
        c2.caption(name)

# ── check models available ───────────────────────────────────────────────────
missing = [name for name, path in MODELS.items() if not path.exists()]
if missing:
    st.error(
        f"ONNX model(s) not found: {', '.join(missing)}. "
        "Run `scripts/06_export_onnx.py --checkpoint models/<name>/best.pth` for each.",
        icon=":material/error:",
    )
    st.stop()

sess_baseline = load_session(MODELS["Baseline"])
sess_dr = load_session(MODELS["DR"])

# ── uploaders ─────────────────────────────────────────────────────────────────
up_col, gt_col = st.columns(2)
with up_col:
    uploaded = st.file_uploader(
        "Echo frame",
        type=["png", "jpg", "jpeg"],
        help="Grayscale PNG or JPEG — apical 2CH or 4CH view.",
    )
with gt_col:
    uploaded_gt = st.file_uploader(
        "Ground-truth mask (optional)",
        type=["png"],
        help=(
            "Single-channel PNG: pixel values 0–3 "
            "(0 background, 1 LV cavity, 2 myocardium, 3 left atrium). "
            "Enables Dice score."
        ),
    )

if uploaded is None:
    st.info("Upload an echo frame to run segmentation.", icon=":material/upload:")
    st.stop()

# ── preprocessing + corruption ────────────────────────────────────────────────
raw_img = Image.open(io.BytesIO(uploaded.read())).convert("L")
gray_256 = np.array(raw_img.resize(IMAGE_SIZE, Image.BILINEAR), dtype=np.uint8)
x_clean = preprocess(gray_256)          # (1,1,256,256) float32 in [0,1]
x_corrupt = apply_corruption(x_clean, corruption_key)

# Convert corrupted tensor back to uint8 for display
gray_corrupt = (x_corrupt[0, 0] * 255).clip(0, 255).astype(np.uint8)

# ── inference ─────────────────────────────────────────────────────────────────
mask_baseline = run_inference(sess_baseline, x_corrupt)
mask_dr = run_inference(sess_dr, x_corrupt)

gt_mask: np.ndarray | None = None
if uploaded_gt is not None:
    gt_mask = load_gt_mask(uploaded_gt)
    if gt_mask is None:
        st.warning(
            "Ground-truth PNG pixel values exceed 3 — does not look like a label mask. "
            "Dice scores will not be shown.",
            icon=":material/warning:",
        )

# ── visualisation ─────────────────────────────────────────────────────────────
col_input, col_baseline, col_dr = st.columns(3)
with col_input:
    label = f"Input ({CORRUPTION_LABELS[corruption_key]})"
    st.subheader(label)
    st.image(gray_corrupt, width="stretch")

with col_baseline:
    st.subheader("Baseline")
    st.image(make_overlay(gray_corrupt, mask_baseline), width="stretch")

with col_dr:
    st.subheader("DR")
    st.image(make_overlay(gray_corrupt, mask_dr), width="stretch")

# ── metrics ───────────────────────────────────────────────────────────────────
st.subheader("Per-structure metrics")
header_cols = st.columns([2, 1, 1])
header_cols[0].markdown("**Structure**")
header_cols[1].markdown("**Baseline**")
header_cols[2].markdown("**DR**")

for label_idx, (name, _) in STRUCTURES.items():
    row = st.columns([2, 1, 1])
    row[0].markdown(name)
    for col, mask in zip(row[1:], [mask_baseline, mask_dr]):
        if gt_mask is not None:
            d = dice_score(mask, gt_mask, label_idx)
            col.metric("Dice", f"{d:.3f}" if d is not None else "N/A")
        else:
            frac = float((mask == label_idx).sum() / mask.size)
            col.metric("Coverage", f"{frac * 100:.1f} %")

if gt_mask is None:
    st.caption("Showing pixel coverage. Upload a ground-truth mask PNG to compute Dice.")
