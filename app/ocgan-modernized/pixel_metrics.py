"""
pixel_metrics.py — pixel-level localization metrics for the production PatchCore.

MVTec AD is a *localization* benchmark and ships pixel-accurate masks, but the
project only scored image-level AUROC. This runs the production model over the
full test set, builds the raw anomaly map (PatchCoreInference.anomaly_map) and
compares it pixel-by-pixel against the ground-truth masks:

  - pixel-AUROC : how well anomaly-map values separate defect pixels from normal
  - pixel-AP    : average precision on the (heavily imbalanced) pixel labels
  - AUPRO@30%   : area under the per-region-overlap vs FPR curve, integrated up to
                  FPR=0.30 (the official MVTec AD localization metric). Unlike
                  pixel-AUROC it weights every connected defect region equally, so
                  a few huge defects can't dominate many small ones.

Good images contribute all-normal pixels. Masks are pushed through the same
ResizePadToSquare transform as the input image so they align with the map.

Output: production_models/pixel_metrics.json  +  pixel_metrics_results.md

Usage:
    python pixel_metrics.py --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score

import server
from verify_all import all_test_images
from utils.transforms import ResizePadToSquare


def compute_aupro(region_scores: list[np.ndarray], normal_scores: np.ndarray,
                  max_fpr: float = 0.30, n_thresh: int = 300) -> float:
    """AUPRO@max_fpr — the official MVTec AD localization score.

    Sweeps a threshold over the anomaly map and, at each level, measures
      - PRO : mean over GT connected regions of the fraction of that region's
              pixels flagged anomalous (every region weighted equally), and
      - FPR : fraction of normal pixels flagged anomalous,
    then integrates PRO over FPR in [0, max_fpr] and normalises by max_fpr.

    region_scores : one 1-D array of anomaly scores per GT connected region.
    normal_scores : anomaly scores over all normal (mask==0) pixels.
    """
    if not region_scores or normal_scores.size == 0:
        return float("nan")
    lo = min(float(normal_scores.min()), min(float(r.min()) for r in region_scores))
    hi = max(float(normal_scores.max()), max(float(r.max()) for r in region_scores))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return float("nan")
    thresholds = np.linspace(lo, hi, n_thresh)

    normal_sorted = np.sort(normal_scores)
    n_norm = normal_sorted.size
    fpr = 1.0 - np.searchsorted(normal_sorted, thresholds, side="left") / n_norm

    pro = np.zeros(n_thresh)
    for r in region_scores:
        rs = np.sort(r)
        pro += 1.0 - np.searchsorted(rs, thresholds, side="left") / rs.size
    pro /= len(region_scores)

    # sort by FPR, anchor at (0,0), collapse duplicate FPRs (upper envelope), integrate
    order = np.argsort(fpr)
    fpr_s = np.concatenate([[0.0], fpr[order]])
    pro_s = np.concatenate([[0.0], pro[order]])
    uniq, inv = np.unique(fpr_s, return_inverse=True)
    pro_u = np.zeros_like(uniq)
    np.maximum.at(pro_u, inv, pro_s)
    grid = np.linspace(0.0, max_fpr, 200)
    pro_i = np.interp(grid, uniq, pro_u)
    trapz = getattr(np, "trapezoid", None) or np.trapz  # np.trapz removed in numpy 2.x
    return float(trapz(pro_i, grid) / max_fpr)


def load_mask_aligned(category: str, defect: str, filename: str, size: int, resizer) -> np.ndarray:
    """Ground-truth mask at (size, size), aligned to the model's padded-square input."""
    if defect == "good":
        return np.zeros((size, size), dtype=np.uint8)
    stem = Path(filename).stem
    mpath = server.DATASET_ROOT / category / "ground_truth" / defect / f"{stem}_mask.png"
    if not mpath.exists():
        return np.zeros((size, size), dtype=np.uint8)
    m = Image.open(mpath).convert("L")
    t = torch.from_numpy(np.array(m)).float().unsqueeze(0) / 255.0  # (1,H,W)
    t = resizer(t)  # (1,size,size)
    return (t.squeeze(0).numpy() > 0.5).astype(np.uint8)


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    ap.add_argument("--out", default=str(server.PRODUCTION_MODELS_DIR / "pixel_metrics.json"))
    args = ap.parse_args()

    server._device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"[pixel] device={server._device}")
    print(f"\n{'category':<12} {'pixel_AUROC':>11} {'pixel_AP':>9} {'AUPRO@30%':>10} {'defect_px%':>10} {'n_img':>6}")
    print("-" * 64)

    results: dict[str, dict] = {}
    for cat in server.CATEGORIES:
        model = server.get_model(cat)  # production PatchCore
        size = model.image_size
        resizer = ResizePadToSquare(size)
        scores_all, labels_all = [], []
        region_scores: list[np.ndarray] = []  # one array per GT connected region (for AUPRO)
        for defect, fn, _is_anom in all_test_images(cat):
            pil = Image.open(server.DATASET_ROOT / cat / "test" / defect / fn).convert("RGB")
            amap = model.anomaly_map(pil).astype(np.float32)
            mask = load_mask_aligned(cat, defect, fn, size, resizer)
            if mask.any():  # split each defect into connected regions, weighted equally by AUPRO
                labeled, n_reg = ndimage.label(mask)
                for rid in range(1, n_reg + 1):
                    region_scores.append(amap[labeled == rid])
            scores_all.append(amap.ravel())
            labels_all.append(mask.ravel())
        scores = np.concatenate(scores_all)
        labels = np.concatenate(labels_all)
        n_def = int(labels.sum())
        if 0 < n_def < labels.size:
            pauroc = float(roc_auc_score(labels, scores))
            pap = float(average_precision_score(labels, scores))
            aupro = compute_aupro(region_scores, scores[labels == 0])
        else:
            pauroc = pap = aupro = float("nan")
        results[cat] = {
            "pixel_auroc": round(pauroc, 4), "pixel_ap": round(pap, 4),
            "aupro": round(aupro, 4), "n_regions": len(region_scores),
            "n_images": len(scores_all), "n_pixels": int(labels.size),
            "defect_pixel_frac": round(float(labels.mean()), 5),
        }
        print(f"{cat:<12} {pauroc:>11.4f} {pap:>9.4f} {aupro:>10.4f} {labels.mean()*100:>9.3f}% {len(scores_all):>6}")
        server._model_cache.clear()
        try:
            if server._device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    valid = [v["pixel_auroc"] for v in results.values() if not np.isnan(v["pixel_auroc"])]
    macro = float(np.mean(valid)) if valid else float("nan")
    valid_pro = [v["aupro"] for v in results.values() if not np.isnan(v["aupro"])]
    macro_pro = float(np.mean(valid_pro)) if valid_pro else float("nan")
    print(f"\n[pixel] macro pixel-AUROC = {macro:.4f}  |  macro AUPRO@30% = {macro_pro:.4f}  ({len(valid)} categories)")

    Path(args.out).write_text(json.dumps(
        {"macro_pixel_auroc": round(macro, 4), "macro_aupro": round(macro_pro, 4), "per_category": results},
        indent=2), encoding="utf-8")
    md = ["# Pixel-level localization metrics (production PatchCore)\n",
          f"Macro pixel-AUROC: **{macro:.4f}**  ·  Macro AUPRO@30%: **{macro_pro:.4f}**\n",
          "| category | pixel-AUROC | pixel-AP | AUPRO@30% | defect pixel % |",
          "|---|---|---|---|---|"]
    for c, v in results.items():
        md.append(f"| {c} | {v['pixel_auroc']:.4f} | {v['pixel_ap']:.4f} | {v['aupro']:.4f} | {v['defect_pixel_frac']*100:.3f}% |")
    Path(server.PROJECT_ROOT / "pixel_metrics_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[pixel] saved -> {args.out}")


if __name__ == "__main__":
    main()
