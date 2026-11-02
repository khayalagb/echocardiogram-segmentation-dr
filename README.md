# Echocardiogram Segmentation on CAMUS

2D segmentation of LV cavity, myocardium, and left atrium on apical echocardiograms from the [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/) dataset. Emphasis on leak-free evaluation, patient-level confidence intervals, stratified failure analysis, and ensemble-based uncertainty.

> Research and demonstration only. Not a medical device.

## Results

Bootstrap 95% CIs over 1 000 resamples (n=1 000 random seeds, patient-level test split).

| Structure | Dice (95% CI) | HD95 mm (95% CI) |
|-----------|--------------|-----------------|
| LV cavity | 0.933 (0.928–0.937) | 1.62 (1.49–1.73) |
| Myocardium | 0.866 (0.860–0.872) | 1.78 (1.67–1.91) |
| Left atrium | 0.886 (0.872–0.897) | 2.48 (2.21–2.78) |

Risk-coverage curves and per-structure analysis: `results/test_risk_coverage.png`

## Setup

```bash
# 1. Create environment
python -m venv .venv && source .venv/bin/activate

# 2. Install (CPU-only torch for dev; use GPU wheel for training)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # GPU
pip install -e ".[dev]"
```

## Data

Download CAMUS from the [Human Heart Project](https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html). Place under `data/CAMUS_public/` so the structure is:

```
data/CAMUS_public/
  patient0001/
    patient0001_2CH_ED.mhd
    patient0001_2CH_ED_gt.mhd
    ...
```

Update `configs/default.yaml → data.root` if your path differs.

## Reproduce

```bash
# Check data integrity
python scripts/00_verify_data.py

# Build patient-level split (runs leak test automatically)
python scripts/01_prepare_split.py

# Train baseline
python scripts/02_train.py --run-name baseline

# Evaluate on validation split (development)
python scripts/03_evaluate.py --checkpoint models/baseline/best.pth --split val

# Train remaining ensemble members
python scripts/04_train_ensemble.py

# Uncertainty analysis
python scripts/05_uncertainty.py

# Evaluate on test split — open once, after configuration is locked
python scripts/03_evaluate.py --checkpoint models/baseline/best.pth --split test

# Export to ONNX + parity check
python scripts/06_export_onnx.py --checkpoint models/baseline/best.pth
```

See `learning/RUNBOOK.md` for a step-by-step guide with expected outputs.

## Tests

```bash
pytest
```

## Robustness to acquisition variability

### Motivation

Ultrasound images vary substantially across acquisitions: gain settings, depth, scanner frequency,
and tissue properties all affect the intensity distribution. A model trained on a single centre may
fail silently when any of these change. Standard augmentation (affine, gamma, Gaussian noise) does
not systematically cover physics-driven shifts such as speckle, PSF blur, and depth attenuation.

This section asks whether augmenting with physics-inspired appearance randomisation improves
robustness to such shifts.

### Method

The baseline (trained with standard augmentation) is compared against a DR arm (trained with
physics-inspired augmentation: Rayleigh speckle, anisotropic blur, depth-attenuation ramp,
wider gain range). Both use the same architecture, seed, and 100-epoch budget.

Seven test-time conditions are applied to the **held-out test split**:

| Condition | In DR training | Description |
|-----------|---------------|-------------|
| clean | — | unmodified images |
| gain\_high | yes | uniform ×1.5 gain, clipped |
| speckle | yes | multiplicative Rayleigh noise (mean≈1, std≈0.47); spatially i.i.d. |
| blur | yes | anisotropic Gaussian PSF (axial σ=2 px, lateral σ=1 px) |
| attenuation | yes | depth-linear gain ramp (1.2 → 0.5) |
| low\_res | **no** | ×4 downsample/upsample (64 px effective) |
| speckle\_corr | **no** | Rayleigh noise smoothed (σ=4 px), renorm. to mean≈1 |

`low_res` and `speckle_corr` are held out: they appear in neither arm's training augmentation.
Any improvement there is not by construction. The CAMUS image-quality tiers (good/medium/poor)
provide a further unplanned natural shift.

### Results

*Placeholder — fill in after runs complete.*

| Condition | Baseline LV | DR LV | Baseline Myo | DR Myo | Baseline LA | DR LA |
|-----------|-------------|-------|--------------|--------|-------------|-------|
| clean | — | — | — | — | — | — |
| gain\_high | — | — | — | — | — | — |
| speckle | — | — | — | — | — | — |
| blur | — | — | — | — | — | — |
| attenuation | — | — | — | — | — | — |
| **low\_res** | — | — | — | — | — | — |
| **speckle\_corr** | — | — | — | — | — | — |

Bootstrap 95% CIs are reported in `results/shift_baseline_test_summary.json` and
`results/shift_dr_test_summary.json`.

**Note on interpretation:** the baseline is already robust to most conditions. Blur, attenuation,
and correlated speckle cause only minor degradation. Uncorrelated speckle (`speckle`) is the main
weakness (Myo Dice drops ~0.10 on the val set). DR training is expected to reduce that gap for
conditions it covers directly; `low_res` and `speckle_corr` are the honest generalization tests.

### Reproducing the robustness study

```bash
# 1. Evaluate baseline across all shift conditions (val, then test once locked)
python scripts/09_robustness.py \
    --checkpoint models.nosync/baseline/best.pth \
    --model-name baseline --split val

# 2. Train DR arm (same seed and budget as baseline)
python scripts/08_train_dr.py --config configs/dr.yaml --run-name dr

# 3. Evaluate DR arm
python scripts/09_robustness.py \
    --checkpoint models.nosync/dr/best.pth \
    --model-name dr --split val

# 4. Open test split only once, after configuration is locked
python scripts/09_robustness.py \
    --checkpoint models.nosync/baseline/best.pth \
    --model-name baseline --split test
python scripts/09_robustness.py \
    --checkpoint models.nosync/dr/best.pth \
    --model-name dr --split test

# 5. Explore results
jupyter notebook notebooks/04_robustness_analysis.ipynb
```

---

## Limitations

- Single dataset (one centre, one scanner vendor). Generalisation to other devices is not demonstrated.
- Test set of ~75 patients yields wide confidence intervals; thresholds are stated post-hoc.
- Augmentation diversity is limited; speckle simulation uses additive Gaussian noise, not a physical model.
- Ensemble members share the same architecture and data; they underestimate epistemic uncertainty relative to architecture-diverse ensembles.

## Structure

```
configs/          training and evaluation hyperparameters
src/echo_seg/     installable package
  data/           split, dataset, augmentation
  models/         U-Net factory
  training/       training loop
  evaluation/     metrics, bootstrap CIs, stratified analysis
  uncertainty/    ensemble inference, risk-coverage
scripts/          00–06 execution scripts (in order)
tests/            pytest tests (split leak, metrics, augmentation)
```

## Citation

```
Leclerc S. et al. Deep learning for segmentation using an open large-scale dataset
in 2D echocardiography. IEEE TMI, 2019.
```
