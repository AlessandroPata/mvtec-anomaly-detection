# MVTec AD Model Showcase — Webapp Design

**Date:** 2026-06-10
**Status:** Approved by user (Arena with reconstructed variants ✓, English UI ✓, 6-page layout ✓)

## Goal

A polished web application that tells the story of the project's anomaly-detection models on MVTec AD: showcase every architecture iteration (OCGAN v1 → PatchCore Production), compare their evaluation results with rich charts, and let the user run a **live batch test** (default 100 random test images) against a chosen model, watching results stream in.

## Context & constraints

- **Backend assets:** `ocgan-modernized/` contains a working FastAPI server (`server.py`, PatchCore-only), 15 production memory banks (`production_models/{cat}/patchcore_bank.pt`), the full MVTec AD dataset at `../../datasets/mvtec_ad` (15 categories, `test/{defect}/*.png`, plus `ground_truth/` masks), and result CSVs/logs for 7 model iterations.
- **Curated data already exists** in the frontend draft: `frontend/src/data/architectures.ts` (7 model cards: ocgan_v1, ocgan_v3, patchcore_v1, patchcore_v2, patchcore_v3, patchcore_p1, production_final) and `frontend/src/data/benchmarks.json` (per-category metrics for 6 models + macro AUROC for all 7).
- **GAN models cannot run live:** checkpoints exist (`model.pt`, `best_checkpoint.pt` in outputs) but the GAN inference pipeline was removed and is not in git history. GAN appears in static results only.
- **Historical per-image GAN scores** exist in `outputs/.../test_blind_component_scores.csv` but are keyed by loader index, not filename — too fragile to replay. Not used.
- **Local machine:** Windows 11, Python 3.13 (torch NOT installed yet), Node 24, Quadro T1000 4 GB (CUDA possible, CPU fallback mandatory).
- **Frontend stack (kept from draft):** React 19, Vite 8, TypeScript, Tailwind 4, Recharts, Framer Motion, Zustand, react-router 7. UI is rebuilt from scratch on this stack; curated data files are reused.
- **UI language:** English.

## Architecture

Two processes in dev; one in production:

```
frontend (Vite dev :5173) ──proxy /api──▶ server.py (FastAPI :8000)
                                              │ PatchCoreInference + variant engine
                                              │ MVTec dataset (images, thumbs)
                                              └ serves frontend dist/ statically in prod
```

- Static pages (Home, Models, Evaluation, Methodology) work entirely from bundled JSON — graceful degradation if the backend is down. Arena and Dataset Explorer require the backend and show a friendly offline state otherwise.
- `vite.config.ts` proxies `/api` → `http://localhost:8000`. `server.py` additionally mounts `frontend/dist` so `python server.py` alone can serve the whole app.

## Components

### 1. Data build script — `ocgan-modernized/scripts/build_webapp_data.py`
Parses the repo's CSVs (`logs/patchcore_v3.csv`, `logs/patchcore_p1.csv`, `logs/patchcore_tuning.csv`, `final_per_category_multiseed_aggregated.csv`, `optv2_multiseed_aggregated.csv`, …) and regenerates:
- `frontend/src/data/benchmarks.json` — per-category metrics (AUROC, AUPRC, best-F1, FPR@95, seeds, config) for each of the 7 models + macro table. Fills the gaps in the existing draft file (production_final per-category rows).
- `frontend/src/data/insights.json` — curated ablation findings with the small datasets behind them (aggregation comparison, coreset 10k vs full bank per category, layer ablation for screw/grid, seed stability std).
Numbers are cross-checked against the README table (macro 0.9846).

### 2. Variant engine — backend
`models/patchcore_variants.py`: wraps a loaded production bank and emulates historical PatchCore configs at inference time. A variant = `{bank_subset, aggregation, topk}`:
- `production` — full bank, `topk_reweighted` k=9 (as shipped; per-category feature levels incl. screw l1+l2+l3).
- `patchcore_v3` — same as production (alias, shown for the story).
- `patchcore_v1` — k-center greedy coreset 10 000 (same algorithm as `scripts/patchcore_pure.py`), `topk_mean` k=3.
- `patchcore_v2` — exact config extracted from `logs/patchcore_v2*` during implementation (coreset/agg per log).
Coreset selection runs once per category on first use and caches **indices** to `production_models/{cat}/variants/{variant}_idx.pt` (small files). Variants are labeled `reconstructed` in API responses; production is labeled `production`. Screw's v1/v2 emulation uses the production l1+l2+l3 bank (original was l2+l3) — flagged `approximate: true` in the API and UI tooltip.

**Variant thresholds** — recalibrated faithfully to the original methodology (99th percentile of held-out `val_normal` scores): offline script `scripts/calibrate_variant_thresholds.py` computes val-image patch features once per category (expensive backbone pass), then scores them against each variant bank (cheap), writing `production_models/variant_thresholds.json`. Runs once; committed/cached. ~10–15 min CPU total, minutes on GPU.

### 3. Server extensions — `server.py` (+ `arena.py` module)
Existing endpoints unchanged. New:
- `GET  /api/meta` — categories, available variants (with labels/approx flags), device, dataset stats summary.
- `GET  /api/dataset/thumb?cat&defect&filename&size=128` — downscaled JPEG, disk-cached under `.thumb_cache/`.
- `POST /api/predict` and `/api/predict/from-dataset` — accept optional `model_variant` (default `production`); heatmap computed here (on-demand only).
- `POST /api/arena/start` `{category, variant, n_images∈[10..150] (default 100), seed?}` → `{job_id, images:[…]}`. Sampling is **stratified** across defect folders (proportional, ≥1 per defect type, always includes goods) and **deterministic given seed** (seed shown in UI, shareable via URL).
- `GET  /api/arena/jobs/{id}/stream` — **SSE**: one `result` event per image `{idx, filename, defect, gt, score, prob, verdict, correct, ms}`, then a `summary` event `{accuracy, precision, recall, f1, auroc, confusion{tp,tn,fp,fn}, mean_ms, p95_ms, device}`. Heatmaps are *not* in the stream (fetched on click).
- `GET  /api/arena/jobs/{id}?since=N` — polling fallback returning results N…end + status.
- `POST /api/arena/jobs/{id}/cancel`.
Job execution: single background worker thread, one job at a time (`409` if busy). Per-image errors yield an `error` verdict and the batch continues. AUROC on the sample computed with a small self-contained implementation (no sklearn dependency).
Device: auto (`cuda` if available else `cpu`), `--device` flag still wins; per-category OOM falls back to CPU with a logged warning and `device` reported per job.

### 4. Frontend — `frontend/src` (rebuilt)
**Visual direction:** dark "industrial control room / QC instrument" theme to match MVTec's factory-inspection domain. Near-black background, steel grays, blueprint-style grid accents; signal palette: emerald = normal, red = anomaly, amber = threshold; one interactive accent (cyan). Monospace tabular numerals for metrics, count-up animations, staggered card entrances, pulsing cells while the batch streams. Distinctive, not a generic dashboard. (frontend-design skill engaged at implementation.)

**Pages (react-router):**
- `/` **Home** — hero with project arc ("from a one-class GAN to frozen-feature PatchCore, +19.8 pp"), animated headline stats (macro AUROC 0.9846, 15 categories, 4 perfect 1.0000 categories, ~10 min eval), journey timeline of the 7 iterations with a macro-AUROC sparkline, CTAs → Arena / Evaluation.
- `/models` **Model Gallery** — 7 cards (status badge production/deprecated, family, macro AUROC, per-category mini bars) → `/models/:id` detail: pipeline diagram (custom blocks from existing `pipeline` data), core idea, strengths/weaknesses, hyperparameter table, "what changed vs previous" chips, per-category results table.
- `/evaluation` **Evaluation Lab** — the comparison hub: sortable leaderboard with metric switcher (AUROC/AUPRC/best-F1/FPR@95); macro-AUROC evolution line; **model × category heatmap** (7×15 colored grid, click cell → drill-down); per-category grouped bars; per-model radar; seed-stability error bars (where multiseed data exists); 3–4 ablation insight cards with mini-charts (aggregation, coreset size, layer choice).
- `/arena` **Test Arena** — config panel (category picker with thumbnails, variant cards labeled Production/Reconstructed with AUROC + approx tooltip, n images 25/50/**100**/custom, seed with shuffle, Run) → live phase: thumbnail grid where each cell fills in as its result streams (border green=correct / red=wrong, icon for FP/FN), live counters (done/correct/accuracy), cancel button → summary: 2×2 confusion matrix, metric chips, score-distribution strip with threshold line, worst-misses callouts. Click any image → modal: full image, **heatmap overlay toggle** (fetched on demand), score-vs-threshold gauge, metadata. Second tab **Single test**: pick any test image or upload your own → result card with heatmap.
- `/dataset` **Dataset Explorer** — 15 category cards (thumb, train/test counts, #defect types) → `/dataset/:category`: defect-type tabs, lazy image grid, ground-truth mask overlay toggle where masks exist.
- `/methodology` **Methodology** — prose from the README: why PatchCore won (3 ingredients), threshold calibration story, honest notes on reconstructed variants, hardware/env, reproduce commands.

**Shared:** sidebar AppShell with health indicator (polls `/api/health`; offline banner disables live features only), URL-encoded arena state (`/arena?cat=screw&variant=production&n=100&seed=42`), keyboard-friendly modals, loading/empty/error states everywhere.

## Data flow (arena happy path)

1. UI `POST /api/arena/start` → backend samples images (stratified, seeded), returns job id + image list → grid renders placeholders with thumbs.
2. UI opens SSE stream; backend worker scores image-by-image with the selected variant; each `result` event fills a cell and updates counters.
3. `summary` event renders the final panel. SSE drop → reconnect with `?since=`, else polling fallback.
4. Heatmap requested only when a result is clicked (`/api/predict/from-dataset` + `model_variant`).

## Error handling

- Backend down → static pages unaffected; Arena/Dataset render an offline card with retry.
- torch missing / banks missing → server exits at startup with an explicit message; `/api/health` exposes device + loaded models.
- GPU OOM at bank load → per-category CPU fallback, logged, surfaced in `device` field.
- Busy worker → `409` with current job id; UI offers to watch the running job.
- Per-image inference failure → `verdict: "error"`, excluded from metrics, shown hatched in the grid.
- SSE unsupported/dropped → polling fallback.

## Testing

- **Backend (pytest):** sampler (stratification, determinism, edge n > available), variant engine math (aggregation values vs hand-computed fixtures on a tiny synthetic bank), AUROC implementation vs known values, arena endpoints via `TestClient` with a temp mini-dataset and fake banks (no real .pt in tests).
- **Frontend:** `tsc -b` and eslint clean; vitest + RTL for arena store reducer (streaming updates, summary math), leaderboard sorting, heatmap grid color mapping; `vite build` passes.
- **Manual E2E checklist** against the real server (one GPU and one CPU run, offline-mode check, 100-image batch).

## Out of scope

- Live GAN inference (no inference code; static results only).
- Replays of historical per-image GAN scores (index→filename mapping too fragile).
- Pixel-level segmentation metrics (image-level only, as in the project's evaluation).
- Authentication, deployment beyond `python server.py` + built dist.

## Implementation order

1. Environment: install torch (CUDA if the T1000 wheel works, else CPU), fastapi deps; smoke-test one real prediction.
2. `build_webapp_data.py` + regenerated benchmarks/insights JSON (verified vs README).
3. Backend: variant engine + threshold calibration + arena jobs/SSE + thumbs + static mount (+ tests).
4. Frontend foundation: theme, shell, router, api client, health store.
5. Pages: Home → Evaluation → Models → Arena → Dataset → Methodology.
6. Polish pass: motion, states, README/run docs.
