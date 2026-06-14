# Dataset Card — MVTec AD

## Summary
**MVTec Anomaly Detection (MVTec AD)** is the standard benchmark for unsupervised
industrial defect detection and localization. It contains **15 categories** (5
textures: carpet, grid, leather, tile, wood; 10 objects: bottle, cable, capsule,
hazelnut, metal_nut, pill, screw, toothbrush, transistor, zipper).

- **Train:** defect-free ("good") images only — the one-class setting.
- **Test:** "good" images plus multiple defect types per category, each with a
  **pixel-accurate ground-truth mask** for localization scoring.
- **Resolution:** high-resolution RGB (≈700–1024 px); this project resizes to 256×256
  with aspect-preserving pad-to-square.

## Why one-class
No anomalies are available at training time — the model learns only what "normal"
looks like and flags deviations. This mirrors real production lines, where defects
are rare, diverse, and unknown in advance, so collecting a labelled defect set is
impractical.

## Layout (as consumed here)
```
datasets/mvtec_ad/<category>/
  train/good/*.png            # normal only
  test/good/*.png             # normal test
  test/<defect_type>/*.png    # anomalous test, one folder per defect
  ground_truth/<defect_type>/<stem>_mask.png   # pixel mask for each defect image
```
`good` test images have an all-zero (empty) mask by construction.

## Evaluation protocol
- **Image-level:** AUROC over test images (anomaly vs. normal), threshold-free.
- **Pixel-level:** pixel-AUROC and pixel-AP over every pixel against the masks.
- **Region-level:** **AUPRO@30%** — the official MVTec localization metric — weights
  every connected defect *region* equally and integrates per-region overlap up to a
  30% false-positive rate, so small defects are not drowned out by large ones.

Masks are pushed through the **same** resize/pad transform as the input image so the
anomaly map and the mask are pixel-aligned (`pixel_metrics.py`).

## Provenance & license
MVTec AD is released by MVTec Software GmbH for **non-commercial research/education**
under the CC BY-NC-SA 4.0 license. It is **not redistributed** in this repository;
download it from the official source and place it at `datasets/mvtec_ad/`.
See: https://www.mvtec.com/company/research/datasets/mvtec-ad

## Class balance note
Defect pixels are a small fraction of each image (often <3%, e.g. screw 0.25%,
leather 0.65%). This extreme imbalance is why **pixel-AP** sits far below
**pixel-AUROC** for the same model, and why AUPRO (region-weighted) is the metric the
benchmark is built around.
