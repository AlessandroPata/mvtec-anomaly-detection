<h1 align="center">MVTec Anomaly Detection</h1>
<p align="center">
  <em>Several one-class anomaly-detection models, refined and compared on MVTec AD —<br/>
  from a CVPR-2019 OCGAN reproduction to a production PatchCore system, served live.</em>
</p>

<p align="center">
  <img alt="image AUROC" src="https://img.shields.io/badge/image%20AUROC-0.9846-2d6a8f">
  <img alt="pixel AUROC" src="https://img.shields.io/badge/pixel%20AUROC-0.9714-2d6a8f">
  <img alt="AUPRO@30%" src="https://img.shields.io/badge/AUPRO%4030%25-0.9127-2d6a8f">
  <img alt="categories" src="https://img.shields.io/badge/MVTec%20categories-15-444">
  <img alt="python" src="https://img.shields.io/badge/python-3.13-3776ab">
  <!-- After pushing, set <OWNER>/<REPO>: -->
  <!-- <img alt="ci" src="https://github.com/<OWNER>/<REPO>/actions/workflows/ci.yml/badge.svg"> -->
</p>

---

## Overview

This project follows a complete one-class anomaly-detection journey on the
**MVTec AD** industrial benchmark: it starts by reproducing and modernizing the
**OCGAN** generative approach (CVPR 2019), then pivots to a memory-based
**PatchCore** system that reaches a **macro image-level AUROC of 0.9846** across all
15 categories. Every model is trained *one per category* in the realistic setting
where **no defects are seen during training** — only normal images.

Rather than shipping a single model, the repo **refines and compares a family of
detectors** — two OCGAN variants and four PatchCore iterations — under one rigorous
4-split, multi-seed protocol, and exposes all of it through a **live webapp** where
you can run any model on any test image and watch it decide in real time.

| Level | Metric | Macro (15 categories) |
|---|---|---:|
| Image | AUROC (threshold-free ranking) | **0.9846** |
| Pixel | pixel-AUROC | **0.9714** |
| Region | AUPRO@30% *(official MVTec localization metric)* | **0.9127** |

> **Why two numbers per model?** *AUROC* measures how well scores **rank** anomalies
> above normals — it is threshold-free and is the headline generalization metric.
> *Accuracy@threshold* (shown in the live arena) measures correctness at a chosen
> operating point. A model can rank perfectly yet decide poorly at a bad threshold —
> exactly the `screw` failure this project diagnosed and fixed with per-category
> calibration (see [Honest evaluation](#honest-evaluation)).

## Models compared

| Family | Variant | Macro AUROC | Idea |
|---|---|---:|---|
| OCGAN | `ocgan_final` | 0.8276 | Modernized one-class GAN, 7 fused scoring heads |
| OCGAN | `ocgan_optv2` | 0.8378 | Re-tuned GAN retrain |
| PatchCore | `v1` | 0.9051 | Frozen features + memory bank, first cut |
| PatchCore | `v2` | 0.9397 | top-k reweighted + multi-scale features |
| PatchCore | `v3` | 0.9828 | Full bank (the coreset paradox) |
| **PatchCore** | **`production`** | **0.9846** | + per-category feature levels (layer1 for screw) |

The GAN and PatchCore paradigms **fail on different categories** (the GAN is best
exactly where PatchCore is weakest, on `screw`), because they measure different
things — *what I can regenerate* vs. *what I have already seen*. That complementarity
is studied honestly in [Honest evaluation](#honest-evaluation), and is why the webapp
serves **both** families live.

## Production model

```
input image 256×256 (ImageNet-norm)
        │
        ▼  wide_resnet50_2 backbone (ImageNet, FROZEN — no fine-tuning)
        │
        ▼  layer2 + layer3 patch features, concatenated (1536-d)
        │  (layer1+layer2+layer3 for screw)
        ▼  memory bank of normal patches (k-center coreset only when > 70k)
        │
        ▼  min distance to bank · top-k reweighted (k=9) → anomaly score + heatmap
```

There is **no gradient training** for PatchCore: the backbone is frozen and "fitting"
is building the memory bank from normal images. See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)
and [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md) for full details.

## Quickstart

**One command (Windows):**
```bat
start-webapp.bat
```
This launches the FastAPI backend, which also serves the pre-built React frontend at
**http://localhost:8000**.

**First-time setup:**
```bash
cd app/ocgan-modernized
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-webapp.txt
# frontend (only if you change it): cd ../frontend && npm install && npm run build
```

**Dev mode (hot reload):**
```bash
# backend
cd app/ocgan-modernized && .venv\Scripts\python.exe server.py --port 8000 --device auto
# frontend (second shell)
cd app/frontend && npm run dev      # http://localhost:5173, proxies /api → :8000
```

The dataset is **not** redistributed; download MVTec AD from the
[official source](https://www.mvtec.com/company/research/datasets/mvtec-ad) and place
it at `datasets/mvtec_ad/`.

## The webapp

- **Evaluation Lab** — leaderboard, 7-model × 15-category AUROC heatmap, evolution
  curve, and the pixel-level localization panel (pixel-AUROC, pixel-AP, AUPRO).
- **Test Arena** — pick a category, an image set, and one or more models; watch them
  classify live (SSE-streamed) with score, threshold, verdict, an interactive
  **threshold slider**, and a **per-defect breakdown**.
- **Dataset Explorer** — browse MVTec AD with the ground-truth defect masks overlaid.
- **Models / Methodology** — per-model architecture and the experimental protocol.

## Honest evaluation

The whole point of this repo is that the numbers are **measured, not asserted**. The
offline scripts under `app/ocgan-modernized/` regenerate every figure:

| Script | What it produces |
|---|---|
| `verify_all.py` | Re-runs every model on every category through the exact server path; flags AUROC drift and the threshold bug. |
| `pixel_metrics.py` | Pixel-AUROC, pixel-AP and **AUPRO@30%** vs. the ground-truth masks. |
| `honest_calibration.py` | Oracle vs. **held-out (cross-validated)** vs. unsupervised-p99 operating-point accuracy — quantifies how much "best-F1 on the test set" overstates deployable accuracy. |
| `ensemble_experiment.py` | A GAN + PatchCore late-fusion study with the fusion weight chosen on calibration folds and scored on held-out folds — does the ensemble actually beat PatchCore alone? |
| `recalibrate_thresholds.py` | Per-category operating points (`threshold_overrides.json`). |

All consolidated live at `GET /api/evaluation`. Fast regression tests
(`tests/test_metrics_regression.py`) run in CI on CPU.

> **What we deliberately did *not* do.** No FAISS-accelerated nearest-neighbour and no
> heavier backbone: the targets above are already reached **efficiently** (the whole
> system trains and serves on a 4 GB GPU), and at this bank size FAISS would add an
> approximate-NN dependency for no measurable accuracy gain.

## Repository layout

```
app/ocgan-modernized/   models, memory banks, eval scripts, FastAPI server
app/frontend/           React + Vite + Tailwind webapp (served from /dist)
datasets/mvtec_ad/      MVTec AD (download separately)
docs/                   MODEL_CARD.md, DATASET_CARD.md, planning
relazione/              full project report (Italian) + slide generator
outputs/ , logs/        training-run artefacts and aggregated CSVs
```

## License & attribution

Code is provided for research and educational use. **MVTec AD** is © MVTec Software
GmbH, released under CC BY-NC-SA 4.0 for non-commercial research — see the
[dataset card](docs/DATASET_CARD.md). The OCGAN approach is from Perera et al.,
*OCGAN: One-class Novelty Detection Using GANs with Constrained Latent Representations*,
CVPR 2019; PatchCore from Roth et al., CVPR 2022.
