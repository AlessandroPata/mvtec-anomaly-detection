# MVTec AD Lab — Model Showcase Webapp

Showcase of the project's anomaly-detection journey on MVTec AD (OCGAN → PatchCore,
0.7866 → **0.9846** macro AUROC) with a live Test Arena.

## Layout
- `ocgan-modernized/` — models, eval scripts, FastAPI inference server
- `frontend/` — React/Vite webapp
- `../datasets/mvtec_ad/` — dataset (15 categories)
- `docs/superpowers/` — design spec + implementation plans

## Run (development)
    # 1. backend
    cd ocgan-modernized
    .\.venv\Scripts\Activate.ps1
    python server.py --port 8000 --device auto

    # 2. frontend (second shell)
    cd frontend
    npm run dev          # http://localhost:5173 (proxies /api → :8000)

## Run (single process)
    cd frontend ; npm run build
    cd ..\ocgan-modernized ; python server.py --port 8000 --device auto
    # → http://localhost:8000 serves the built app

## First-time setup
See `ocgan-modernized/README.md` (venv, torch, calibrate_variant_thresholds.py,
build_webapp_data.py) and run `npm install` in `frontend/`.
