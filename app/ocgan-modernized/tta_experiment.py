"""
tta_experiment.py — does rotation test-time augmentation help the weak categories?

screw is rotation-variant (screws appear at arbitrary angles), so averaging the
anomaly score over rotated copies of the query *might* lift its AUROC. This only
measures the effect (image-level AUROC, base vs TTA) — we integrate TTA only if it
genuinely helps. If kept, thresholds MUST be recalibrated afterwards (scores change).

Usage:
    python tta_experiment.py --device cuda --categories screw,metal_nut,grid
"""
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

import server
from verify_all import all_test_images

ANGLES = [0, 90, 180, 270]


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    ap.add_argument("--categories", default="screw,metal_nut,grid")
    args = ap.parse_args()

    import torch
    server._device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    print(f"[tta] device={server._device}  angles={ANGLES}")
    print(f"\n{'category':<12} {'AUROC_base':>10} {'AUROC_TTA':>10} {'delta':>8}")
    print("-" * 44)

    for cat in cats:
        model = server.get_variant_model(cat, "production")
        y, base, tta = [], [], []
        for defect, fn, is_anom in all_test_images(cat):
            img = Image.open(server.DATASET_ROOT / cat / "test" / defect / fn).convert("RGB")
            base.append(float(model.predict(img)["anomaly_score"]))
            rot_scores = [float(model.predict(img.rotate(a, expand=False))["anomaly_score"]) for a in ANGLES]
            tta.append(float(np.mean(rot_scores)))
            y.append(1 if is_anom else 0)
        y = np.array(y)
        a_base = roc_auc_score(y, base)
        a_tta = roc_auc_score(y, tta)
        print(f"{cat:<12} {a_base:>10.4f} {a_tta:>10.4f} {a_tta - a_base:>+8.4f}")
        server._model_cache.clear()
        server._variant_cache.clear()
        try:
            if server._device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
