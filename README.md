# Echocardiogram Segmentation on CAMUS

2D segmentation of LV cavity, myocardium, and left atrium on apical echocardiograms from the [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/) dataset. Emphasis on leak-free evaluation, patient-level confidence intervals, stratified failure analysis, and ensemble-based uncertainty.

> Research and demonstration only. Not a medical device.

## Results

*(Populated after running `scripts/03_evaluate.py` on the test split.)*

| Structure | Dice (95% CI) | HD95 mm (95% CI) |
|-----------|--------------|-----------------|
| LV cavity | — | — |
| Myocardium | — | — |
| Left atrium | — | — |

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
