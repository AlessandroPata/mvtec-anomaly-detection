"""
honest_calibration.py — does the per-category threshold generalize, or is the
arena accuracy a self-fulfilling artefact of peeking at the test labels?

`recalibrate_thresholds.py` picks each category's operating point as the best-F1
threshold over the *whole* test set, then reports accuracy on that same set. That
is an oracle: it can only overstate the deployable accuracy. AUROC (threshold-free)
is unaffected, but the arena's acc@thr deserves an honest estimate.

MVTec AD has no labelled validation anomalies, so we estimate honestly with
stratified k-fold cross-validation: pick the best-F1 threshold on the (k-1)/k
calibration folds, score accuracy/F1 on the held-out fold, average over folds.
We report, per category:

  - oracle_acc  : best-F1 threshold fit and scored on the full test set (what the
                  shipped override and the arena display — the optimistic bound)
  - honest_acc  : mean held-out accuracy under cross-validated thresholds (±std)
  - p99_acc     : accuracy of the unsupervised p99-of-normal-scores threshold
                  (never sees an anomaly — the only truly deployable rule here)
  - gap         : oracle_acc - honest_acc (how much the oracle overstates)

A small gap is the good outcome: it means the operating point is stable, not an
artefact. Side effect: caches raw test scores to score_cache_<variant>.json so the
analysis can be re-run without the GPU.

Usage:
    python honest_calibration.py --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import server
from recalibrate_thresholds import best_f1_threshold
from verify_all import all_test_images


def collect_scores(category: str, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """Anomaly scores + binary labels over the full test set (server predict path)."""
    model = server.get_variant_model(category, variant)
    scores, y = [], []
    for defect, fn, is_anom in all_test_images(category):
        r = model.predict(Image.open(server.DATASET_ROOT / category / "test" / defect / fn).convert("RGB"))
        scores.append(float(r["anomaly_score"]))
        y.append(1 if is_anom else 0)
    return np.asarray(scores, dtype=np.float64), np.asarray(y, dtype=int)


def honest_cv_accuracy(scores: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 43):
    """Stratified k-fold: threshold chosen on calibration folds, scored on held-out fold."""
    if y.sum() < k or (len(y) - y.sum()) < k:  # too few of a class to stratify
        k = max(2, int(min(y.sum(), len(y) - y.sum())))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    accs, f1s = [], []
    for calib_idx, eval_idx in skf.split(scores.reshape(-1, 1), y):
        thr, _ = best_f1_threshold(scores[calib_idx], y[calib_idx])
        pred = (scores[eval_idx] >= thr).astype(int)
        accs.append(float((pred == y[eval_idx]).mean()))
        f1s.append(float(f1_score(y[eval_idx], pred, zero_division=0)))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s))


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    ap.add_argument("--variant", default="production")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=str(server.PRODUCTION_MODELS_DIR / "honest_calibration.json"))
    args = ap.parse_args()

    import torch
    server._device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"[honest] device={server._device} variant={args.variant} folds={args.folds}")
    print(f"\n{'category':<12} {'oracle_acc':>10} {'honest_acc':>12} {'p99_acc':>8} {'gap':>7}")
    print("-" * 54)

    results: dict[str, dict] = {}
    score_cache: dict[str, dict] = {}
    for cat in server.CATEGORIES:
        scores, y = collect_scores(cat, args.variant)
        score_cache[cat] = {"scores": [round(float(s), 6) for s in scores], "y": [int(v) for v in y]}

        thr_oracle, _ = best_f1_threshold(scores, y)
        oracle_acc = float(((scores >= thr_oracle).astype(int) == y).mean())
        honest_acc, honest_std, honest_f1 = honest_cv_accuracy(scores, y, k=args.folds)
        thr_p99 = float(np.percentile(scores[y == 0], 99)) if (y == 0).any() else thr_oracle
        p99_acc = float(((scores >= thr_p99).astype(int) == y).mean())
        gap = oracle_acc - honest_acc

        results[cat] = {
            "oracle_thr": round(thr_oracle, 6), "oracle_acc": round(oracle_acc, 4),
            "honest_acc": round(honest_acc, 4), "honest_acc_std": round(honest_std, 4),
            "honest_f1": round(honest_f1, 4),
            "p99_thr": round(thr_p99, 6), "p99_acc": round(p99_acc, 4),
            "gap": round(gap, 4), "n": int(len(y)), "n_anom": int(y.sum()),
        }
        print(f"{cat:<12} {oracle_acc:>10.4f} {honest_acc:>8.4f}±{honest_std:.3f} {p99_acc:>8.4f} {gap:>+7.4f}")

        server._model_cache.clear()
        server._variant_cache.clear()
        try:
            if server._device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    macro = {
        "oracle_acc": round(float(np.mean([r["oracle_acc"] for r in results.values()])), 4),
        "honest_acc": round(float(np.mean([r["honest_acc"] for r in results.values()])), 4),
        "p99_acc": round(float(np.mean([r["p99_acc"] for r in results.values()])), 4),
        "gap": round(float(np.mean([r["gap"] for r in results.values()])), 4),
    }
    print(f"\n[honest] macro  oracle={macro['oracle_acc']:.4f}  honest={macro['honest_acc']:.4f}  "
          f"p99={macro['p99_acc']:.4f}  mean gap={macro['gap']:+.4f}")

    Path(args.out).write_text(json.dumps({"variant": args.variant, "folds": args.folds,
                                          "macro": macro, "per_category": results}, indent=2), encoding="utf-8")
    cache_path = server.PRODUCTION_MODELS_DIR / f"score_cache_{args.variant}.json"
    cache_path.write_text(json.dumps(score_cache), encoding="utf-8")
    print(f"[honest] saved -> {args.out}")
    print(f"[honest] score cache -> {cache_path}")

    md = ["# Honest threshold calibration (is the arena accuracy an oracle artefact?)\n",
          f"Variant: **{args.variant}** · {args.folds}-fold stratified CV\n",
          f"Macro: oracle acc **{macro['oracle_acc']:.4f}**, honest (held-out) acc **{macro['honest_acc']:.4f}**, "
          f"unsupervised p99-normal acc **{macro['p99_acc']:.4f}**, mean gap **{macro['gap']:+.4f}**\n",
          "| category | oracle acc | honest acc (±std) | p99-normal acc | gap |",
          "|---|---|---|---|---|"]
    for c, v in results.items():
        md.append(f"| {c} | {v['oracle_acc']:.4f} | {v['honest_acc']:.4f} ±{v['honest_acc_std']:.3f} "
                  f"| {v['p99_acc']:.4f} | {v['gap']:+.4f} |")
    Path(server.PROJECT_ROOT / "honest_calibration_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
