# ocgan-modernized — MVTec AD Anomaly Detection

PatchCore-based anomaly detection on the MVTec AD dataset, with a FastAPI inference server for production deployment.

## Final results

**Macro AUROC: 0.9846** across 15 MVTec categories (previously 0.7866 with the GAN-based pipeline → +19.8pp).

| Category | AUROC | Config |
|----------|------:|--------|
| bottle      | 1.0000 | layer2+layer3 |
| cable       | 0.9960 | layer2+layer3 |
| capsule     | 0.9824 | layer2+layer3 |
| carpet      | 0.9943 | layer2+layer3 |
| grid        | 0.9680 | layer2+layer3 |
| hazelnut    | 1.0000 | layer2+layer3 |
| leather     | 1.0000 | layer2+layer3 |
| metal_nut   | 0.9924 | layer2+layer3 |
| pill        | 0.9580 | layer2+layer3 |
| screw       | 0.9419 | **layer1+layer2+layer3** |
| tile        | 1.0000 | layer2+layer3 |
| toothbrush  | 0.9710 | layer2+layer3 |
| transistor  | 0.9933 | layer2+layer3 |
| wood        | 0.9911 | layer2+layer3 |
| zipper      | 0.9801 | layer2+layer3 |
| **Macro**   | **0.9846** | |

Backbone: `wide_resnet50_2` (ImageNet, frozen). Aggregation: `topk_reweighted` k=9. Memory bank: full training set, max 70k patches, k-center coreset only when needed (hazelnut).

## Architecture

```
input image (256×256, ImageNet-norm)
        │
        ▼
wide_resnet50_2 backbone (frozen)
        │
        ▼  layer2 + layer3 features pooled to layer3 spatial size, concatenated
        │  (or layer1+layer2+layer3 for screw)
        ▼
patch embeddings (L2-normalized)
        │
        ▼  per patch: min L2 distance to memory bank
        ▼
top-k=9 reweighted aggregation → image-level anomaly score
        │
        ▼
threshold (99th percentile of held-out training scores)
        │
        ▼
{anomaly_score, is_anomalous, anomaly_probability, heatmap}
```

## Repository structure

```
project/ocgan-modernized/
├── server.py                       # FastAPI inference server (PatchCore v2.0)
├── models/
│   ├── patchcore_inference.py      # PatchCoreInference class for production
│   └── backbones/                  # wide_resnet50_2 / resnet50 builders
├── scripts/
│   ├── patchcore_pure.py           # standalone bank-build + eval (Hydra)
│   ├── export_patchcore_banks.py   # export 15 production banks
│   ├── run_patchcore_v3.sh         # full 15-cat × 3-seed eval
│   └── run_p1_tuning.sh            # screw/grid layer ablation
├── production_models/{cat}/
│   ├── patchcore_bank.pt           # ~280–500 MB per category
│   ├── config.yaml                 # legacy GAN config (unused)
│   └── manifest.json
├── logs/
│   ├── patchcore_v3.csv            # final per-cat AUROC
│   └── patchcore_p1.csv            # P1 ablation results
└── configs/                        # Hydra configs (per-category)
```

## Quick start

### Run inference server

```bash
cd project/ocgan-modernized
python server.py --port 8000 --device cuda
# then: POST /api/predict with file=image, category=bottle
```

Endpoints:
- `GET /api/health` — server status, loaded models
- `GET /api/categories` — list of 15 supported categories
- `POST /api/predict` — multipart with `file` (image) + `category` (form field) → JSON with `anomaly_score`, `anomaly_probability`, `is_anomalous`, `threshold`, `heatmap_base64`

### Re-export memory banks

```bash
python scripts/export_patchcore_banks.py --device cuda            # all 15 cats
python scripts/export_patchcore_banks.py --device cuda --category screw  # one cat
```

Builds bank from `train_normal` (85% split), calibrates threshold on `val_normal` (15% held-out). Per-category feature-level overrides defined in `FEATURE_LEVEL_OVERRIDES` in the script.

### Reproduce evaluation

```bash
bash scripts/run_patchcore_v3.sh        # 15 cats × 3 seeds, ~10 min
bash scripts/run_p1_tuning.sh           # grid/screw layer ablation, ~3 min
```

## Methodology

### Why PatchCore wins (vs the original GAN-based pipeline)

The repo originally pursued a one-class GAN with reconstruction + multiple auxiliary scorers (teacher-student, latent compactness, memory bank, learned fusion). After the Sprint 1 v3 fixes (wiring `use_skip_connections`, `unfreeze_from`, `scoring_topk`), the macro AUROC was 0.7866.

Sprint 4 stripped everything except the memory bank and used **frozen ImageNet features directly** — no training, no fusion. This jumped to 0.9051 (+12pp) on the first attempt, eventually reaching 0.9846 after tuning. Three ingredients drove the gain:

1. **No bank pruning when it fits** — the original used `coreset=10000` with k-center greedy, dropping 80% of patches. Setting `max_patches=70000` keeps the full bank for 14/15 categories. Zipper alone went 0.7184 → 0.9801, capsule 0.7724 → 0.9824.
2. **`topk_reweighted` aggregation** — a softmax-weighted top-k mean instead of plain top-k mean, down-weighting redundant top distances. Beats `topk_mean/k=3` on every weak category.
3. **Multi-scale features (layer2+layer3)** — concat `layer2` (pooled to layer3 spatial size) with `layer3` for richer per-patch embeddings. For `screw`, adding `layer1` was a further +2.7pp (fine thread detail).

The trained-GAN component never produced features competitive with frozen ImageNet, so it was dropped entirely.

### Threshold calibration

Important detail: the threshold cannot be calibrated on the same training images that built the bank — every patch is in the bank → distance ≈ 0 → threshold ≈ 0 → everything flagged anomalous. The export script holds out 15% of training (`val_normal`) and sets threshold = 99th percentile of those scores. For categories with very low intra-class variance (e.g. bottle), this gives a clean separation; for hard categories (screw, pill), `anomaly_probability` (sigmoid of the z-score against val_normal) is a more robust signal than the binary flag.

### Per-category overrides

`screw` is the only category that benefits from `layer1+layer2+layer3` (+2.7pp). All others are best with `layer2+layer3`. Tested but unimproved: grid, pill, toothbrush.

## Environment

- Python: `/usr/local/bin/python` (PyTorch 2.1.1+cu121)
- GPU: Quadro RTX 5000 (16 GB) — works on any CUDA GPU with ≥8 GB
- Pinned: `numpy<2` (wandb compat), `opencv-python-headless<4.12` (numpy compat)
- Server deps: `fastapi`, `uvicorn`, `python-multipart`, `hydra-core`

## Webapp

Showcase webapp: frontend in `../frontend` (React/Vite), served by this API in production.

    # one-time setup (Windows PowerShell)
    python -m venv .venv ; .\.venv\Scripts\Activate.ps1
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # or plain `pip install torch torchvision` for CPU
    pip install -r requirements-webapp.txt
    python scripts/calibrate_variant_thresholds.py        # thresholds for reconstructed v1/v2
    python scripts/build_webapp_data.py                   # regenerate frontend benchmark JSON

    # run (also serves ../frontend/dist if built)
    python server.py --port 8000 --device auto

New API: `/api/meta`, `/api/dataset/thumb`, `/api/dataset/mask`, `/api/arena/start`,
`/api/arena/jobs/{id}` (poll), `/api/arena/jobs/{id}/stream` (SSE), `/api/arena/jobs/{id}/cancel`.
Both predict endpoints accept `model_variant`:
- `production` | `patchcore_v2` | `patchcore_v1` — PatchCore; v1/v2 reconstructed from the
  production banks (k-center coreset 10k + that era's aggregation; thresholds recalibrated on
  the original val_normal split, seed 43, p99).
- `ocgan_final` | `ocgan_optv2` — the GAN, run live through `webapp/gan_engine.py` from the
  original training checkpoints (`production_models/{cat}/model.pt` and the optv2 seed-43 run
  dirs). Weights, MAD normalization stats, logistic fusion and val_mixed threshold come from
  the checkpoint; only the memory bank is rebuilt at load (seeded, <1 min/category). Exception:
  optv2 refits its calibration on the val splits at load — its runs trained on the committed
  (training-era) code state, whose scoring differs from the current tree, and their fp16
  numerics overflow on this GPU (engine runs fp32 and detects the reconstructor architecture
  from the checkpoint keys, since the era's builder ignored `use_skip_connections`).
  The server keeps one GAN resident at a time. Smoke test: `python scripts/smoke_gan_predict.py
  cuda ocgan_final bottle`.

Tests: `python -m pytest -q` — `test_smoke_train.py` and `test_aggregate_runs.py` require the
original training environment (hydra config tree + Paperspace output paths) and are expected
to fail on a webapp-only setup; everything else must pass.

## Open items

- Bank files are 280–500 MB each. Quantization to int8 (~4× smaller) is feasible but unnecessary at 15-cat scale.
