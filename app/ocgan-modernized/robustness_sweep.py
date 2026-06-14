"""
robustness_sweep.py — how much does production AUROC degrade under input corruption?

A frozen-backbone PatchCore is only as robust as ImageNet features are to the shift.
We probe that directly: each test image is corrupted at increasing severities and
re-scored, and we watch image-level AUROC fall away from the clean baseline. This is
a standard out-of-distribution robustness check for anomaly detectors and answers a
question the clean benchmark cannot: would this survive a dimmer lamp or a noisier
sensor on the line?

Corruptions (severity 1→3): additive Gaussian noise, Gaussian blur, progressive
darkening (brightness↓) and contrast loss. Labels are unchanged, so only the score
distribution moves; we report macro and per-category AUROC per condition and plot
the degradation curves.

Output: production_models/robustness_sweep.json + robustness_sweep_results.md
        + relazione/figures/fig_robustness.png

Usage:
    python robustness_sweep.py --device cuda            # full test set
    python robustness_sweep.py --device cuda --sample 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import roc_auc_score

import server
from verify_all import all_test_images

CORRUPTIONS = {
    "noise": {1: 0.04, 2: 0.08, 3: 0.16},
    "blur": {1: 1.0, 2: 2.0, 3: 3.0},
    "brightness": {1: 0.8, 2: 0.6, 3: 0.4},   # progressive darkening
    "contrast": {1: 0.7, 2: 0.5, 3: 0.3},     # progressive contrast loss
}


def corrupt(img: Image.Image, kind: str, level, rng: np.random.Generator) -> Image.Image:
    if kind == "noise":
        a = np.asarray(img, np.float32) / 255.0
        a = a + rng.normal(0.0, float(level), a.shape)
        return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(float(level)))
    if kind == "brightness":
        return ImageEnhance.Brightness(img).enhance(float(level))
    if kind == "contrast":
        return ImageEnhance.Contrast(img).enhance(float(level))
    return img


def safe_auroc(y, s):
    return float(roc_auc_score(y, s)) if 0 < int(np.sum(y)) < len(y) else float("nan")


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    ap.add_argument("--sample", type=int, default=0, help="images/category (0 = full test set)")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--out", default=str(server.PRODUCTION_MODELS_DIR / "robustness_sweep.json"))
    args = ap.parse_args()

    server._device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    rng = np.random.default_rng(args.seed)
    conditions = ["clean"] + [f"{k}_{lvl}" for k in CORRUPTIONS for lvl in CORRUPTIONS[k]]
    print(f"[robust] device={server._device} sample={args.sample or 'full'} conditions={len(conditions)}")

    per_cat: dict[str, dict] = {}
    for cat in server.CATEGORIES:
        model = server.get_model(cat)
        items = all_test_images(cat)
        if args.sample:
            imgs = server.sample_test_images(server.DATASET_ROOT, cat, args.sample, args.seed)
            items = [(im.defect, im.filename, im.is_anomaly) for im in imgs]
        scores = {c: [] for c in conditions}
        y = []
        for defect, fn, is_anom in items:
            base = Image.open(server.DATASET_ROOT / cat / "test" / defect / fn).convert("RGB")
            y.append(1 if is_anom else 0)
            scores["clean"].append(float(model.predict(base)["anomaly_score"]))
            for kind, levels in CORRUPTIONS.items():
                for lvl, val in levels.items():
                    img = corrupt(base, kind, val, rng)
                    scores[f"{kind}_{lvl}"].append(float(model.predict(img)["anomaly_score"]))
        y = np.asarray(y)
        per_cat[cat] = {c: round(safe_auroc(y, np.asarray(scores[c])), 4) for c in conditions}
        print(f"  {cat:<12} clean={per_cat[cat]['clean']:.4f} "
              f"noise3={per_cat[cat]['noise_3']:.4f} blur3={per_cat[cat]['blur_3']:.4f} "
              f"bright3={per_cat[cat]['brightness_3']:.4f} contrast3={per_cat[cat]['contrast_3']:.4f}")
        server._model_cache.clear()
        try:
            if server._device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    macro = {c: round(float(np.nanmean([per_cat[cat][c] for cat in per_cat])), 4) for c in conditions}
    print("\n[robust] macro AUROC per condition:")
    print(f"  clean = {macro['clean']:.4f}")
    for k in CORRUPTIONS:
        row = "  ".join(f"sev{lvl} {macro[f'{k}_{lvl}']:.4f}" for lvl in CORRUPTIONS[k])
        print(f"  {k:<11} {row}")

    Path(args.out).write_text(json.dumps(
        {"sample": args.sample, "corruptions": {k: CORRUPTIONS[k] for k in CORRUPTIONS},
         "macro": macro, "per_category": per_cat}, indent=2), encoding="utf-8")
    print(f"[robust] saved -> {args.out}")

    # markdown
    md = ["# Robustness to input corruption (production PatchCore, image-level AUROC)\n",
          f"Clean macro AUROC **{macro['clean']:.4f}**. Each corruption shown at severity 1→3.\n",
          "| corruption | sev 1 | sev 2 | sev 3 | drop @ sev3 |", "|---|---|---|---|---|"]
    for k in CORRUPTIONS:
        drop = macro["clean"] - macro[f"{k}_3"]
        md.append(f"| {k} | {macro[f'{k}_1']:.4f} | {macro[f'{k}_2']:.4f} | {macro[f'{k}_3']:.4f} | −{drop:.4f} |")
    Path(server.PROJECT_ROOT / "robustness_sweep_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # degradation-curve figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.2))
        xs = [0, 1, 2, 3]
        for k, color in zip(CORRUPTIONS, ["#c2553a", "#2d6a8f", "#1f9d55", "#8a6d3b"]):
            ys = [macro["clean"]] + [macro[f"{k}_{lvl}"] for lvl in (1, 2, 3)]
            ax.plot(xs, ys, marker="o", label=k, color=color)
        ax.axhline(macro["clean"], ls="--", lw=0.8, color="#888", label="clean")
        ax.set_xlabel("corruption severity"); ax.set_ylabel("macro image-AUROC")
        ax.set_xticks(xs); ax.set_ylim(min(0.5, min(macro.values()) - 0.03), 1.005)
        ax.set_title("PatchCore robustness to input corruption (15 categories)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig_path = server.PROJECT_ROOT.parent.parent / "relazione" / "figures" / "fig_robustness.png"
        fig.tight_layout(); fig.savefig(fig_path, dpi=140); plt.close(fig)
        print(f"[robust] figure -> {fig_path}")
    except Exception as e:
        print(f"[robust] figure skipped: {e}")


if __name__ == "__main__":
    main()
