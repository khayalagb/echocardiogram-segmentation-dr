# Model Card

## Model details

| Field | Value |
|-------|-------|
| Architecture | U-Net with ResNet34 encoder (ImageNet pretrained) |
| Framework | PyTorch via segmentation-models-pytorch |
| Input | Grayscale 2D echocardiogram, resized to 256×256 |
| Output | 4-class logit map: background / LV cavity / myocardium / left atrium |
| Export | ONNX opset 17 |

## Intended use

Research and portfolio demonstration. Evaluates rigorous evaluation methodology on a public benchmark. **Not validated for clinical use. Not a medical device.**

## Training data

CAMUS dataset — apical 2CH and 4CH views, end-diastolic and end-systolic frames. Patient-level split 70/15/15. Full data provenance in `DATA_CARD.md`.

## Evaluation

Dice and HD95 per structure with 95% bootstrap confidence intervals resampled over patients. Results stratified by view, phase, and image quality. Test split opened once after configuration was locked.

## Out-of-scope uses

- Clinical decision support
- Any application outside the CAMUS domain without further validation
- Any use implying CE or FDA clearance

## Known limitations

- Trained and tested on a single-centre, single-vendor dataset
- Speckle noise augmentation is Gaussian, not a physics-based US model
- Ensemble members share architecture and training data
