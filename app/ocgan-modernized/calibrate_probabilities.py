"""
calibrate_probabilities.py — turn raw anomaly scores into *calibrated* probabilities.

AUROC and the operating-point threshold (honest_calibration.py) say how well the
model ranks and decides, but they say nothing about whether a score of 0.8 means
"80% likely anomalous". A raw PatchCore score is monotonic with anomaly-ness but
is not a probability. We fix that with post-hoc calibration:

  - Platt / sigmoid : p = sigmoid(a·score + b)        (2 params, robust on little data)
  - Isotonic        : monotonic step fit              (non-parametric, needs more data)

Quality is measured with the Brier score (mean squared error of the probability)
and ECE (expected calibration error, 10 bins) — both vs. the uncalibrated baseline
(the raw score clipped to [0,1] used as a probability). To avoid an oracle, the
calibrator is fit on stratified CV calibration folds and scored on the held-out
folds; the deployed calibrator is then refit on all data and its parameters saved
in a version-independent form (np.interp breakpoints / sigmoid coeffs), so the
server can apply it without depending on the exact sklearn build.

Runs entirely from production_models/score_cache_production.json (no GPU).

Output: production_models/probability_calibration.json + probability_calibration_results.md

Usage:
    python calibrate_probabilities.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import server


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: weighted |confidence - accuracy| over equal-width bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def fit_platt(s: np.ndarray, y: np.ndarray):
    lr = LogisticRegression(C=1e6, solver="lbfgs").fit(s.reshape(-1, 1), y)
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    return {"method": "platt", "a": a, "b": b}


def fit_isotonic(s: np.ndarray, y: np.ndarray):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s, y)
    return {"method": "isotonic",
            "x": [float(v) for v in iso.X_thresholds_],
            "y": [float(v) for v in iso.y_thresholds_]}


def apply_calibrator(cal: dict, s: np.ndarray) -> np.ndarray:
    """Version-independent application (no sklearn needed at inference)."""
    s = np.asarray(s, dtype=np.float64)
    if cal["method"] == "platt":
        z = cal["a"] * s + cal["b"]
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
    if cal["method"] == "isotonic":
        return np.interp(s, np.asarray(cal["x"]), np.asarray(cal["y"]))
    return np.clip(s, 0.0, 1.0)


def cv_calibrate(s: np.ndarray, y: np.ndarray, fitter, k: int = 5, seed: int = 43):
    """Pool held-out calibrated predictions over stratified k folds."""
    kk = max(2, min(k, int(y.sum()), int(len(y) - y.sum())))
    skf = StratifiedKFold(n_splits=kk, shuffle=True, random_state=seed)
    p_held = np.empty_like(s, dtype=np.float64)
    for cal_idx, ev_idx in skf.split(s.reshape(-1, 1), y):
        cal = fitter(s[cal_idx], y[cal_idx])
        p_held[ev_idx] = apply_calibrator(cal, s[ev_idx])
    return p_held


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="production")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=str(server.PRODUCTION_MODELS_DIR / "probability_calibration.json"))
    args = ap.parse_args()

    cache_path = server.PRODUCTION_MODELS_DIR / f"score_cache_{args.variant}.json"
    if not cache_path.exists():
        raise SystemExit(f"score cache not found: {cache_path} — run honest_calibration.py first")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))

    print(f"[calib] variant={args.variant} folds={args.folds} (from cached scores, no GPU)")
    print(f"\n{'category':<12} {'method':<9} {'Brier_raw':>9} {'Brier_cal':>9} {'ECE_raw':>8} {'ECE_cal':>8}")
    print("-" * 60)

    results: dict[str, dict] = {}
    for cat in server.CATEGORIES:
        if cat not in cache:
            continue
        s = np.asarray(cache[cat]["scores"], dtype=np.float64)
        y = np.asarray(cache[cat]["y"], dtype=int)
        if not (0 < y.sum() < len(y)):
            continue
        p_raw = np.clip(s, 0.0, 1.0)  # uncalibrated baseline: score-as-probability
        # honest held-out quality for each method, pick the better by CV Brier
        cands = {}
        for name, fitter in (("platt", fit_platt), ("isotonic", fit_isotonic)):
            try:
                p_cv = cv_calibrate(s, y, fitter, k=args.folds)
                cands[name] = (brier(p_cv, y), ece(p_cv, y))
            except Exception:
                continue
        best = min(cands, key=lambda m: cands[m][0])
        brier_cal, ece_cal = cands[best]
        # deployed calibrator: refit on all data
        deployed = (fit_platt if best == "platt" else fit_isotonic)(s, y)
        results[cat] = {
            "method": best, "calibrator": deployed,
            "brier_raw": round(brier(p_raw, y), 5), "brier_cal": round(brier_cal, 5),
            "ece_raw": round(ece(p_raw, y), 5), "ece_cal": round(ece_cal, 5),
            "n": int(len(y)), "n_anom": int(y.sum()),
            "candidates": {m: {"brier": round(v[0], 5), "ece": round(v[1], 5)} for m, v in cands.items()},
        }
        print(f"{cat:<12} {best:<9} {results[cat]['brier_raw']:>9.5f} {brier_cal:>9.5f} "
              f"{results[cat]['ece_raw']:>8.5f} {ece_cal:>8.5f}")

    macro = {
        "brier_raw": round(float(np.mean([r["brier_raw"] for r in results.values()])), 5),
        "brier_cal": round(float(np.mean([r["brier_cal"] for r in results.values()])), 5),
        "ece_raw": round(float(np.mean([r["ece_raw"] for r in results.values()])), 5),
        "ece_cal": round(float(np.mean([r["ece_cal"] for r in results.values()])), 5),
    }
    print(f"\n[calib] macro Brier {macro['brier_raw']:.5f} -> {macro['brier_cal']:.5f}  |  "
          f"ECE {macro['ece_raw']:.5f} -> {macro['ece_cal']:.5f}")

    Path(args.out).write_text(json.dumps(
        {"variant": args.variant, "folds": args.folds, "macro": macro, "per_category": results}, indent=2),
        encoding="utf-8")
    print(f"[calib] saved -> {args.out}")

    md = ["# Probability calibration (raw score → calibrated probability)\n",
          f"Variant **{args.variant}** · {args.folds}-fold CV · Platt vs. isotonic, better-by-Brier per category.\n",
          f"Macro Brier **{macro['brier_raw']:.4f} → {macro['brier_cal']:.4f}**, "
          f"ECE **{macro['ece_raw']:.4f} → {macro['ece_cal']:.4f}** (lower is better).\n",
          "| category | method | Brier raw | Brier cal | ECE raw | ECE cal |",
          "|---|---|---|---|---|---|"]
    for c, v in results.items():
        md.append(f"| {c} | {v['method']} | {v['brier_raw']:.4f} | {v['brier_cal']:.4f} "
                  f"| {v['ece_raw']:.4f} | {v['ece_cal']:.4f} |")
    Path(server.PROJECT_ROOT / "probability_calibration_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
