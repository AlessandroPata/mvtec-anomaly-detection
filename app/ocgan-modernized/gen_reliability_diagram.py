"""
gen_reliability_diagram.py — visualize the probability calibration of section 8.9.

A reliability diagram bins predictions by their predicted probability and plots, per
bin, the predicted confidence (x) against the observed fraction of true anomalies (y).
Perfect calibration lies on the diagonal; a curve below it means over-confidence. We
pool all 15 categories and draw the raw score-as-probability against the post-hoc
calibrated probability (calibrate_probabilities.py), with the expected calibration
error (ECE) in the legend.

Runs from production_models/score_cache_production.json + probability_calibration.json
(no GPU).

Output: relazione/figures/fig_reliability.png

Usage:
    python gen_reliability_diagram.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import server
from calibrate_probabilities import apply_calibrator, ece


def reliability_curve(p: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """Per-bin (mean confidence, observed accuracy, weight)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    xs, ys, ws = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            xs.append(float(p[m].mean()))
            ys.append(float(y[m].mean()))
            ws.append(float(m.mean()))
    return np.array(xs), np.array(ys), np.array(ws)


def main():
    cache = json.loads((server.PRODUCTION_MODELS_DIR / "score_cache_production.json").read_text(encoding="utf-8"))
    cal = json.loads((server.PRODUCTION_MODELS_DIR / "probability_calibration.json").read_text(encoding="utf-8"))["per_category"]

    p_raw, p_cal, y_all = [], [], []
    for c, d in cache.items():
        s = np.asarray(d["scores"], dtype=np.float64)
        y = np.asarray(d["y"], dtype=int)
        p_raw.append(np.clip(s, 0.0, 1.0))
        calib = cal.get(c, {}).get("calibrator")
        p_cal.append(apply_calibrator(calib, s) if calib else np.clip(s, 0.0, 1.0))
        y_all.append(y)
    p_raw = np.concatenate(p_raw); p_cal = np.concatenate(p_cal); y_all = np.concatenate(y_all)
    ece_raw, ece_cal = ece(p_raw, y_all), ece(p_cal, y_all)
    print(f"[reliability] pooled ECE raw={ece_raw:.4f} -> calibrated={ece_cal:.4f}  (n={len(y_all)})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#888", label="perfect calibration")
    for p, color, name, e in ((p_raw, "#c2553a", "raw score", ece_raw), (p_cal, "#1f9d55", "calibrated", ece_cal)):
        xs, ys, _ = reliability_curve(p, y_all)
        ax.plot(xs, ys, marker="o", color=color, label=f"{name} (ECE {e:.3f})")
    ax.set_xlabel("predicted probability (confidence)")
    ax.set_ylabel("observed fraction of anomalies")
    ax.set_title("Reliability diagram — production PatchCore (pooled)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)
    out = server.PROJECT_ROOT.parent.parent / "relazione" / "figures" / "fig_reliability.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[reliability] figure -> {out}")


if __name__ == "__main__":
    main()
