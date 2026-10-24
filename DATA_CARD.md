# Data Card

## Dataset

**CAMUS** (Cardiac Acquisitions for Multi-structure Ultrasound Segmentation)  
Leclerc S. et al., IEEE TMI 2019. DOI: 10.1109/TMI.2019.2900516

## Source

Human Heart Project, CREATIS, INSA Lyon.  
Download: https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html  
Licence: verify at download time. Do not redistribute images. Do not commit images to this repository.

## Contents

| Field | Value |
|-------|-------|
| Patients | ~450 with ground truth annotations |
| Views | Apical 2-chamber (2CH), apical 4-chamber (4CH) |
| Phases | End-diastole (ED), end-systole (ES) |
| Images per patient | 4 (2 views × 2 phases) |
| Mask labels | 0 background, 1 LV cavity, 2 myocardium, 3 left atrium |
| Format | MetaImage (.mhd + .raw) or NIfTI |
| Image quality labels | good / medium / poor (in Info_*CH.cfg) |

## Split (this project)

| Split | Patients (approx.) | Images |
|-------|--------------------|--------|
| Train | 315 (70%) | 1260 |
| Val   | 67 (15%)  |  268 |
| Test  | 68 (15%)  |  272 |

Split is patient-level: all images of a patient go to the same subset. Verified by automated test (`tests/test_split.py`, `scripts/01_prepare_split.py`). Seed: 42. Manifest: `data/manifest.csv` (gitignored).

## Known issues

- Pixel spacing varies across patients; HD95 is computed in physical units (mm) using per-image spacing from the file header.
- Image quality labels originate from the dataset providers and are self-reported per view.
- The dataset represents a single clinical centre; models trained on it may not generalise to other scanners or populations.
