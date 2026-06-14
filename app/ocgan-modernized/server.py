"""
OCGAN2026 Inference API Server — PatchCore edition

FastAPI backend serving anomaly detection via PatchCore memory banks.
Each MVTec AD category has its own pre-built bank loaded on demand and cached.

Usage:
    python server.py                     # default: port 8000, cpu
    python server.py --port 8080         # custom port
    python server.py --device cuda       # GPU inference
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.patchcore_inference import PatchCoreInference  # noqa: E402
from models.patchcore_variants import (  # noqa: E402
    VARIANT_SPECS, available_variants, build_variant_model, load_calibration,
)
from webapp.gan_engine import GAN_SPECS, GanInference, gan_variant_entries  # noqa: E402
from webapp.jobs import JobBusyError, JobManager  # noqa: E402
from webapp.metrics import summarize, verdict_of  # noqa: E402
from webapp.sampler import list_test_images, sample_test_images  # noqa: E402
from webapp.thumbs import get_thumb, safe_name  # noqa: E402

import json  # noqa: E402
import random as _random  # noqa: E402
import threading  # noqa: E402
from functools import lru_cache  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PRODUCTION_MODELS_DIR = PROJECT_ROOT / "production_models"
DATASET_ROOT = PROJECT_ROOT.parent.parent / "datasets" / "mvtec_ad"
THUMB_CACHE_DIR = PROJECT_ROOT / ".thumb_cache"
FRONTEND_DIST = PROJECT_ROOT.parent / "frontend" / "dist"

CATEGORIES = sorted(
    [d.name for d in PRODUCTION_MODELS_DIR.iterdir()
     if d.is_dir() and (d / "patchcore_bank.pt").exists()]
)

# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

_model_cache: dict[str, PatchCoreInference] = {}
_variant_cache: dict[tuple[str, str], object] = {}
_device = "cpu"

job_manager = JobManager()


def get_model(category: str) -> PatchCoreInference:
    if category not in _model_cache:
        print(f"[server] Loading PatchCore model for: {category}")
        bank_path = PRODUCTION_MODELS_DIR / category / "patchcore_bank.pt"
        _model_cache[category] = PatchCoreInference(category, bank_path, device=_device)
        print(f"[server] '{category}' loaded on {_device}")
    return _model_cache[category]


# GAN trainers hold a full backbone + teacher + decoder, so only one stays
# resident; loading one evicts the previous (~1 min rebuild per category).
_gan_cache: dict[tuple[str, str], GanInference] = {}
_gan_lock = threading.Lock()


def get_gan_model(category: str, variant: str) -> GanInference:
    key = (category, variant)
    with _gan_lock:
        if key not in _gan_cache:
            print(f"[server] Rebuilding GAN {variant} for: {category} (bank + fusion fit)")
            _gan_cache.clear()
            if _device == "cuda":
                torch.cuda.empty_cache()
            try:
                _gan_cache[key] = GanInference(category, variant, device=_device)
            except FileNotFoundError as e:
                raise HTTPException(status_code=400, detail=str(e))
            print(f"[server] GAN {variant}/{category} ready on {_device}")
        return _gan_cache[key]


_threshold_overrides: dict | None = None


def _load_threshold_overrides() -> dict:
    """Per-(variant, category) best-F1 thresholds from recalibrate_thresholds.py.

    Applied on top of the bank/checkpoint threshold so the Arena operating point
    reflects each model's best achievable accuracy. Absent file → no-op.
    """
    global _threshold_overrides
    if _threshold_overrides is None:
        p = PRODUCTION_MODELS_DIR / "threshold_overrides.json"
        try:
            _threshold_overrides = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except Exception:
            _threshold_overrides = {}
    return _threshold_overrides


def get_variant_model(category: str, variant: str = "production"):
    model = _resolve_variant_model(category, variant)
    ov = _load_threshold_overrides().get(variant, {}).get(category)
    if ov is not None:
        try:
            model.threshold = float(ov)
        except Exception:
            pass
    return model


def _resolve_variant_model(category: str, variant: str = "production"):
    if variant in GAN_SPECS:
        return get_gan_model(category, variant)
    if variant not in VARIANT_SPECS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown variant '{variant}'. "
                                   f"Available: {list(VARIANT_SPECS) + list(GAN_SPECS)}")
    key = (category, variant)
    if key not in _variant_cache:
        base = get_model(category)
        spec = VARIANT_SPECS[variant]
        cal = load_calibration(PRODUCTION_MODELS_DIR).get(category, {}).get(variant)
        try:
            _variant_cache[key] = build_variant_model(base, spec, cal, PRODUCTION_MODELS_DIR)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return _variant_cache[key]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

app = FastAPI(
    title="OCGAN2026 Anomaly Detection API",
    version="2.0.0",
    description="PatchCore-based anomaly detection for MVTec AD categories",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def numpy_to_base64_png(arr: np.ndarray) -> str:
    import base64
    arr_uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(arr_uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Health / categories
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models_loaded": list(_model_cache.keys()),
        "backend": "patchcore",
    }


@app.get("/api/categories")
def categories():
    return {"categories": CATEGORIES}


@lru_cache(maxsize=1)
def _meta_payload() -> str:
    calibration = load_calibration(PRODUCTION_MODELS_DIR)
    cats = []
    for c in CATEGORIES:
        entry = {"name": c,
                 "variants": available_variants(c, calibration) + gan_variant_entries(c)}
        try:
            groups = list_test_images(DATASET_ROOT, c)
            entry["test_total"] = sum(len(v) for v in groups.values())
            entry["defect_types"] = [
                {"name": g, "count": len(v), "is_anomaly": g != "good"}
                for g, v in groups.items()
            ]
        except FileNotFoundError:
            entry["test_total"] = 0
            entry["defect_types"] = []
        cats.append(entry)
    return json.dumps({
        "categories": cats,
        "device": _device,
        "dataset_available": DATASET_ROOT.exists(),
    })


def meta_cache_clear() -> None:
    _meta_payload.cache_clear()


@app.get("/api/meta")
def meta_endpoint():
    return JSONResponse(content=json.loads(_meta_payload()))


BENCHMARKS_JSON = PROJECT_ROOT.parent / "frontend" / "src" / "data" / "benchmarks.json"


def _read_json(path: Path):
    """Best-effort JSON load; None if absent or unreadable (artefact not yet built)."""
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except Exception:
        return None


@app.get("/api/evaluation")
def evaluation_endpoint():
    """Consolidated evaluation artefacts produced by the offline scripts:
      - benchmarks      : image-level AUROC per variant/category (verify_all gate)
      - pixel_metrics   : pixel-AUROC / pixel-AP / AUPRO@30% (pixel_metrics.py)
      - honest_calibration : oracle vs held-out vs p99 accuracy (honest_calibration.py)
      - ensemble        : GAN+PatchCore late-fusion verdict (ensemble_experiment.py)
      - threshold_overrides : applied per-category operating points
    Each block is null when its script has not been run yet, so the page degrades
    gracefully instead of 500-ing."""
    benchmarks = _read_json(BENCHMARKS_JSON) or {}
    return JSONResponse(content={
        "device": _device,
        "categories": CATEGORIES,
        "benchmarks": benchmarks,
        "pixel_metrics": _read_json(PRODUCTION_MODELS_DIR / "pixel_metrics.json"),
        "honest_calibration": _read_json(PRODUCTION_MODELS_DIR / "honest_calibration.json"),
        "ensemble": _read_json(PRODUCTION_MODELS_DIR / "ensemble_experiment.json"),
        "threshold_overrides": _read_json(PRODUCTION_MODELS_DIR / "threshold_overrides.json"),
    })


# ---------------------------------------------------------------------------
# Dataset browsing
# ---------------------------------------------------------------------------

@app.get("/api/dataset/test-images")
def dataset_test_images(cat: str):
    """Return all test images for a category, grouped by defect type."""
    test_dir = DATASET_ROOT / cat / "test"
    if not test_dir.exists():
        raise HTTPException(status_code=404, detail=f"Test dir not found: {test_dir}")

    result = []
    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue
        images = sorted(f.name for f in defect_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"))
        result.append({
            "defect_type": defect_dir.name,
            "is_anomaly": defect_dir.name != "good",
            "count": len(images),
            "images": images,
        })
    return {"category": cat, "defect_types": result}


@app.get("/api/dataset/sample")
def dataset_sample(cat: str, defect: str, filename: str):
    """Serve a single test image by filename."""
    if not all(safe_name(v) for v in (cat, defect, filename)):
        raise HTTPException(status_code=400, detail="Invalid path component")
    img_path = DATASET_ROOT / cat / "test" / defect / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {img_path}")
    return FileResponse(str(img_path), media_type="image/png")


@app.get("/api/dataset/thumb")
def dataset_thumb(cat: str, defect: str, filename: str, size: int = 128):
    try:
        path = get_thumb(DATASET_ROOT, THUMB_CACHE_DIR, cat, defect, filename, size)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(path), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/dataset/mask")
def dataset_mask(cat: str, defect: str, filename: str):
    if not all(safe_name(v) for v in (cat, defect, filename)):
        raise HTTPException(status_code=400, detail="Invalid path component")
    stem = Path(filename).stem
    mask = DATASET_ROOT / cat / "ground_truth" / defect / f"{stem}_mask.png"
    if not mask.is_file():
        raise HTTPException(status_code=404, detail="No ground-truth mask")
    return FileResponse(str(mask), media_type="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})


# ---------------------------------------------------------------------------
# Predict (upload OR test-set image)
# ---------------------------------------------------------------------------

@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    category: str = Form(...),
    model_variant: str = Form("production"),
):
    if category not in CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown category '{category}'. Available: {CATEGORIES}",
        )

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    model = get_variant_model(category, model_variant)
    result = model.predict(pil_image)

    return JSONResponse(content={
        "anomaly_score": result["anomaly_score"],
        "anomaly_probability": result["anomaly_probability"],
        "is_anomaly": result["is_anomalous"],
        "threshold": result["threshold"],
        "category": result["category"],
        "inference_ms": result["inference_time_ms"],
        "score_components": result.get("score_components", {}),
        "heatmap_base64": numpy_to_base64_png(result["heatmap"]),
        "model_variant": model_variant,
    })


@app.post("/api/predict/from-dataset")
async def predict_from_dataset(
    category: str = Form(...),
    defect: str = Form(...),
    filename: str = Form(...),
    model_variant: str = Form("production"),
):
    """Run inference on a test-set image identified by category/defect/filename."""
    if category not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Unknown category '{category}'")
    if not all(safe_name(v) for v in (defect, filename)):
        raise HTTPException(status_code=400, detail="Invalid path component")

    img_path = DATASET_ROOT / category / "test" / defect / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {img_path}")

    pil_image = Image.open(img_path).convert("RGB")
    model = get_variant_model(category, model_variant)
    result = model.predict(pil_image)

    return JSONResponse(content={
        "anomaly_score": result["anomaly_score"],
        "anomaly_probability": result["anomaly_probability"],
        "is_anomaly": result["is_anomalous"],
        "threshold": result["threshold"],
        "category": result["category"],
        "inference_ms": result["inference_time_ms"],
        "score_components": result.get("score_components", {}),
        "heatmap_base64": numpy_to_base64_png(result["heatmap"]),
        "defect_type": defect,
        "filename": filename,
        "ground_truth_anomaly": defect != "good",
        "model_variant": model_variant,
    })


# ---------------------------------------------------------------------------
# Test Arena (batch runs over sampled test images)
# ---------------------------------------------------------------------------

class ArenaStartRequest(BaseModel):
    category: str
    variant: str = "production"
    n_images: int = Field(default=100, ge=5, le=150)
    seed: int | None = None


def _make_runner(model, category: str):
    def runner(job):
        try:
            for i, img in enumerate(job.images):
                if job.cancel_requested:
                    job.finish("cancelled", summary=summarize(job.results))
                    return
                path = DATASET_ROOT / category / "test" / img.defect / img.filename
                try:
                    pil = Image.open(path).convert("RGB")
                    r = model.predict(pil)
                    res = {
                        "idx": i, "defect_type": img.defect, "filename": img.filename,
                        "ground_truth_anomaly": img.is_anomaly,
                        "anomaly_score": r["anomaly_score"],
                        "anomaly_probability": r["anomaly_probability"],
                        "is_anomaly": r["is_anomalous"], "threshold": r["threshold"],
                        "inference_ms": r["inference_time_ms"],
                        "verdict": verdict_of(img.is_anomaly, r["is_anomalous"]),
                        "correct": img.is_anomaly == r["is_anomalous"],
                    }
                except Exception as e:  # noqa: BLE001 — keep the batch alive
                    res = {"idx": i, "defect_type": img.defect, "filename": img.filename,
                           "ground_truth_anomaly": img.is_anomaly,
                           "verdict": "error", "error": str(e)}
                job.add_result(res)
            job.finish("done", summary=summarize(job.results))
        except Exception as e:  # noqa: BLE001
            job.finish("error", error=str(e))
    return runner


@app.post("/api/arena/start")
def arena_start(payload: ArenaStartRequest):
    if payload.category not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Unknown category '{payload.category}'")
    model = get_variant_model(payload.category, payload.variant)  # 400 if unknown/uncalibrated
    seed = payload.seed if payload.seed is not None else _random.randint(0, 999_999)
    try:
        images = sample_test_images(DATASET_ROOT, payload.category, payload.n_images, seed)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        job = job_manager.start(payload.category, payload.variant, images,
                                _make_runner(model, payload.category), seed=seed)
    except JobBusyError as e:
        raise HTTPException(status_code=409, detail={"message": str(e), "job_id": e.current_id})
    return {
        "job_id": job.id, "seed": seed, "n": len(images),
        "category": payload.category, "variant": payload.variant,
        "images": [{"idx": i, "defect_type": im.defect, "filename": im.filename,
                    "ground_truth_anomaly": im.is_anomaly} for i, im in enumerate(images)],
    }


@app.get("/api/arena/jobs/{job_id}")
def arena_poll(job_id: str, since: int = 0):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    results, status, summary = job.wait_results(since, timeout=0.0)
    return {"job_id": job.id, "status": status, "results": results,
            "summary": summary, "error": job.error,
            "total": len(job.images), "done": since + len(results)}


@app.get("/api/arena/jobs/{job_id}/stream")
def arena_stream(job_id: str, since: int = 0):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    def gen():
        cursor = since
        while True:
            batch, status, summary = job.wait_results(cursor, timeout=15.0)
            for r in batch:
                yield f"event: result\ndata: {json.dumps(r)}\n\n"
            cursor += len(batch)
            if status != "running":
                yield ("event: summary\ndata: "
                       + json.dumps({"status": status, "summary": summary, "error": job.error})
                       + "\n\n")
                return
            if not batch:
                yield ": heartbeat\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/arena/jobs/{job_id}/cancel")
def arena_cancel(job_id: str):
    job = job_manager.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job.id, "status": job.status, "cancel_requested": job.cancel_requested}


# ---------------------------------------------------------------------------
# Static frontend (serves ../frontend/dist when built) + SPA fallback
# ---------------------------------------------------------------------------

if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")


@app.exception_handler(404)
async def spa_fallback(request, exc):
    if (request.method == "GET" and not request.url.path.startswith("/api")
            and FRONTEND_DIST.is_dir()):
        return FileResponse(FRONTEND_DIST / "index.html")
    return JSONResponse({"detail": getattr(exc, "detail", "Not found")}, status_code=404)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="OCGAN2026 Inference Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--preload", action="store_true",
        help="Preload all category models at startup instead of on first request",
    )
    args = parser.parse_args()

    if args.device == "auto":
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        _device = args.device
    print(f"[server] Starting on {args.host}:{args.port}  device={_device}")
    print(f"[server] Available categories: {CATEGORIES}")
    print(f"[server] Dataset root: {DATASET_ROOT}")

    if args.preload:
        print("[server] Preloading all models...")
        t0 = time.time()
        for cat in CATEGORIES:
            get_model(cat)
        print(f"[server] All models loaded in {time.time()-t0:.0f}s")

    uvicorn.run(app, host=args.host, port=args.port)
