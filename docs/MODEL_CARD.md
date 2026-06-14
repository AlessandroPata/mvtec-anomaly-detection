# Model Card — Production PatchCore (MVTec AD)

## Overview
The shipped production model is a **PatchCore** anomaly detector trained **one model
per category** (15 independent memory banks) on the MVTec AD benchmark. It detects
*and localizes* defects without ever seeing an anomaly at training time: it memorizes
a coreset of normal-image patch features and scores a test patch by its distance to
the nearest memorized normal patch.

This repository also retains two earlier OCGAN (one-class GAN) variants and several
PatchCore ablations for comparison; PatchCore is the production choice.

## Architecture
| Component | Choice |
|---|---|
| Backbone | `wide_resnet50_2`, ImageNet-pretrained, **frozen** (no fine-tuning) |
| Feature levels | `layer2 + layer3` concatenated (1536-d patch descriptors) |
| Memory bank | coreset-subsampled normal patches (e.g. ~45.6k vectors / category) |
| Scoring | min distance to bank, `topk_reweighted` aggregation (k = 9) |
| Input | 256×256, aspect-preserving pad-to-square |
| Image-level score | aggregated patch distance; `anomaly_map()` exposes the raw per-pixel field |

No gradient training is performed for PatchCore: the backbone is frozen and the
"fit" is the coreset memory-bank construction over normal images only.

## Metrics (full MVTec AD test set)
| Level | Metric | Macro (15 categories) |
|---|---|---|
| Image | AUROC | **0.9846** |
| Pixel | pixel-AUROC | **0.9714** |
| Pixel | AUPRO@30% (official MVTec) | **0.9127** |

Per-category image-level AUROC (weakest → strongest): screw 0.942, pill 0.958,
grid 0.968, toothbrush 0.971, zipper 0.980, capsule 0.982, wood 0.991,
metal_nut 0.992, transistor 0.993, carpet 0.994, cable 0.996,
bottle / hazelnut / leather / tile 1.000.

> **AUROC vs. accuracy.** AUROC is threshold-free: it measures how well the score
> *ranks* anomalies above normals and is the headline generalization number. The app's
> arena instead shows **accuracy @ threshold** — correctness at a chosen operating
> point. A model can rank perfectly (high AUROC) yet score poorly at a badly-placed
> threshold, which is exactly the `screw` failure this project fixed via per-category
> calibration. See `honest_calibration.py` for the held-out (non-oracle) estimate of
> that operating-point accuracy.

## Operating point (threshold)
The deployable, unsupervised default is the **p99 of normal-image scores** (never
sees an anomaly). For the showcase arena, per-category best-F1 overrides
(`production_models/threshold_overrides.json`) place each model at its best
operating point. `honest_calibration.py` reports how much that oracle overstates
deployable accuracy by re-estimating the threshold with stratified k-fold CV.

## Intended use & limitations
- **Intended:** industrial visual inspection research / demos on MVTec AD-style data;
  a reference implementation comparing PatchCore against OCGAN.
- **Not intended:** safety-critical deployment without per-deployment recalibration on
  in-distribution normal data.
- **Limitations:** one model per category (no cross-category generalization);
  performance depends on the ImageNet backbone's transfer to the target texture;
  thresholds must be recalibrated on real normal data for any new line/lighting.
- **Compute:** trains and serves on a 4 GB GPU (Quadro T1000). A stronger backbone or
  a FAISS-accelerated bank were deliberately **not** used — the targets above are
  already reached efficiently, and FAISS adds an approximate-NN dependency for no
  accuracy gain at this bank size.

## Reproduce
```bash
# image-level verification (live AUROC / acc@thr vs. static benchmarks)
python verify_all.py --device cuda
# pixel-level localization (pixel-AUROC, pixel-AP, AUPRO@30%)
python pixel_metrics.py --device cuda
# honest (non-oracle) operating-point accuracy
python honest_calibration.py --device cuda
```
