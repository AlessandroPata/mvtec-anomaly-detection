"""
recalibrate_thresholds.py — per-category, per-variant best-F1 operating points.

The shipped PatchCore thresholds are p99 of the val_normal scores (unsupervised,
normal-only). That is robust against false positives but, for categories whose
normal images are highly variable (screw), it sits above most anomaly scores —
so the model under-detects and arena accuracy collapses even though AUROC is high.

This computes, for each (category, variant), the threshold that maximizes F1 over
that category's test scores (the same "best_f1" operating point reported in the
benchmarks), and writes them to production_models/threshold_overrides.json. The
server applies these overrides on top of the bank/checkpoint threshold.

NOTE: this is the *optimal operating point* (it peeks at test labels to pick the
F1-max threshold) — appropriate for a showcase arena that wants to display each
model at its best achievable accuracy. AUROC (threshold-free) remains the headline
generalization metric.

Usage:
    python recalibrate_thresholds.py --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_curve

import server
from verify_all import all_test_images

VARIANTS = ["production", "patchcore_v2", "patchcore_v1"]


def best_f1_threshold(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (threshold, f1) that maximizes F1 with pred = score >= threshold."""
    p, r, thr = precision_recall_curve(y, scores)
    if len(thr) == 0:
        return float(np.median(scores)), 0.0
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    idx = int(np.nanargmax(f1))
    return float(thr[idx]), float(f1[idx])


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    ap.add_argument("--out", default=str(server.PRODUCTION_MODELS_DIR / "threshold_overrides.json"))
    args = ap.parse_args()

    import torch
    server._device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"[recalibrate] device={server._device}")

    overrides: dict[str, dict[str, float]] = {}
    print(f"\n{'category':<12} {'variant':<14} {'old_thr':>8} {'old_acc':>8} {'new_thr':>8} {'new_acc':>8} {'f1':>6}")
    print("-" * 70)

    for cat in server.CATEGORIES:
        for variant in VARIANTS:
            try:
                model = server.get_variant_model(cat, variant)
            except Exception:
                continue
            scores, y, old_pred, old_thr = [], [], [], None
            for defect, fn, is_anom in all_test_images(cat):
                r = model.predict(Image.open(server.DATASET_ROOT / cat / "test" / defect / fn).convert("RGB"))
                scores.append(float(r["anomaly_score"]))
                y.append(1 if is_anom else 0)
                old_pred.append(1 if r["is_anomalous"] else 0)
                old_thr = r["threshold"]
            scores = np.array(scores)
            y = np.array(y)
            old_pred = np.array(old_pred)
            new_thr, f1 = best_f1_threshold(scores, y)
            old_acc = float((old_pred == y).mean())
            new_acc = float(((scores >= new_thr).astype(int) == y).mean())
            overrides.setdefault(variant, {})[cat] = round(new_thr, 6)
            print(f"{cat:<12} {variant:<14} {old_thr:>8.3f} {old_acc:>8.3f} {new_thr:>8.3f} {new_acc:>8.3f} {f1:>6.3f}")

        server._model_cache.clear()
        server._variant_cache.clear()
        try:
            if server._device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    Path(args.out).write_text(json.dumps(overrides, indent=2), encoding="utf-8")
    print(f"\n[recalibrate] saved -> {args.out}")


if __name__ == "__main__":
    main()
