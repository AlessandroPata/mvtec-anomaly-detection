"""
verify_all.py — Cross-check every model variant on every MVTec category.

For each (category, variant) it runs the EXACT same inference path the Test
Arena uses (server.get_variant_model + model.predict) over the test set, then
computes:
  - AUROC      (threshold-independent ranking; comparable to the Evaluation page)
  - acc@thr    (accuracy at the model's calibrated threshold; what the Arena shows)
and prints them next to the static Evaluation AUROC from frontend benchmarks.json.

It flags:
  - AUROC drift   |AUROC_live - AUROC_eval| > 0.05
  - threshold bug AUROC_live >= 0.85 but acc@thr < 0.60  (ranking ok, operating point wrong)

PatchCore variants run over the FULL test set; GAN variants over a capped,
seeded sample (they rebuild ~1 min/category and are much slower).

Usage:
    python verify_all.py --device cuda
    python verify_all.py --device cpu --variants production,patchcore_v2,patchcore_v1
    python verify_all.py --gan-cap 40
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

import server  # reuse the server's exact model-loading + predict path


ALL_VARIANTS = ["production", "patchcore_v2", "patchcore_v1", "ocgan_final", "ocgan_optv2"]
GAN_VARIANTS = {"ocgan_final", "ocgan_optv2"}

ROOT = Path(__file__).resolve().parent
BENCHMARKS = ROOT.parent / "frontend" / "src" / "data" / "benchmarks.json"


def load_eval_auroc() -> dict:
    """eval[variant][category] = auroc (from frontend static benchmarks)."""
    out: dict[str, dict[str, float]] = {}
    try:
        data = json.loads(BENCHMARKS.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] could not read benchmarks.json: {e}")
        return out
    for variant, rows in data.get("per_category", {}).items():
        out[variant] = {}
        for r in rows:
            if r.get("auroc") is not None:
                out[variant][r["category"]] = float(r["auroc"])
    return out


def all_test_images(category: str):
    """Every test image as (defect, filename, is_anomaly)."""
    test_dir = server.DATASET_ROOT / category / "test"
    items = []
    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue
        is_anom = defect_dir.name != "good"
        for f in sorted(defect_dir.iterdir()):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
                items.append((defect_dir.name, f.name, is_anom))
    return items


def sampled_test_images(category: str, n: int, seed: int = 43):
    imgs = server.sample_test_images(server.DATASET_ROOT, category, n, seed)
    return [(im.defect, im.filename, im.is_anomaly) for im in imgs]


def evaluate(category: str, variant: str, gan_cap: int):
    """Run predict over the test set; return metrics dict or {'skip': reason}."""
    try:
        model = server.get_variant_model(category, variant)
    except Exception as e:  # uncalibrated / unavailable / missing checkpoint
        return {"skip": f"{type(e).__name__}: {getattr(e, 'detail', e)}"}

    is_gan = variant in GAN_VARIANTS
    images = sampled_test_images(category, gan_cap) if is_gan else all_test_images(category)

    y_true, scores, preds = [], [], []
    thr = None
    t0 = time.time()
    for defect, filename, is_anom in images:
        path = server.DATASET_ROOT / category / "test" / defect / filename
        try:
            pil = Image.open(path).convert("RGB")
            r = model.predict(pil)
        except Exception as e:
            return {"skip": f"predict error: {e}"}
        y_true.append(1 if is_anom else 0)
        scores.append(float(r["anomaly_score"]))
        preds.append(1 if r["is_anomalous"] else 0)
        thr = r["threshold"]
    elapsed = time.time() - t0

    y_true = np.array(y_true)
    scores = np.array(scores, dtype=np.float64)
    preds = np.array(preds)
    n = len(y_true)
    n_anom = int(y_true.sum())
    n_norm = n - n_anom

    try:
        auroc = float(roc_auc_score(y_true, scores)) if 0 < n_anom < n else float("nan")
    except Exception:
        auroc = float("nan")
    acc = float((preds == y_true).mean()) if n else float("nan")
    pred_anom_rate = float(preds.mean()) if n else float("nan")

    return {
        "n": n, "n_anom": n_anom, "n_norm": n_norm,
        "auroc": auroc, "acc": acc, "thr": float(thr) if thr is not None else float("nan"),
        "pred_anom_rate": pred_anom_rate, "elapsed": elapsed, "is_gan": is_gan,
    }


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS),
                    help="comma list from: " + ",".join(ALL_VARIANTS))
    ap.add_argument("--categories", default="", help="comma list; empty = all")
    ap.add_argument("--gan-cap", type=int, default=50, help="images per GAN run")
    ap.add_argument("--out", default=str(ROOT / "verify_results.md"))
    args = ap.parse_args()

    import torch
    if args.device == "auto":
        server._device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        server._device = args.device
    print(f"[verify] device={server._device}")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    cats = [c.strip() for c in args.categories.split(",") if c.strip()] or list(server.CATEGORIES)
    eval_auroc = load_eval_auroc()

    rows = []
    header = (f"{'category':<12} {'variant':<14} {'n':>4} {'AUROC_live':>10} {'acc@thr':>8} "
              f"{'thr':>9} {'pred_an%':>8} {'AUROC_eval':>10} {'flag'}")
    print("\n" + header)
    print("-" * len(header))

    for cat in cats:
        for variant in variants:
            m = evaluate(cat, variant, args.gan_cap)
            ev = eval_auroc.get(variant, {}).get(cat)
            if "skip" in m:
                ev_s = f"{ev:.3f}" if ev is not None else "—"
                print(f"{cat:<12} {variant:<14} {'—':>4} {'—':>10} {'—':>8} {'—':>9} {'—':>8} "
                      f"{ev_s:>10} SKIP ({m['skip'][:40]})")
                rows.append({"category": cat, "variant": variant, "skip": m["skip"], "eval": ev})
                continue

            flags = []
            if ev is not None and not np.isnan(m["auroc"]) and abs(m["auroc"] - ev) > 0.05:
                flags.append("AUROC_DRIFT")
            if not np.isnan(m["auroc"]) and m["auroc"] >= 0.85 and m["acc"] < 0.60:
                flags.append("THRESH_BUG")
            flag = ",".join(flags) if flags else "ok"

            ev_s = f"{ev:.3f}" if ev is not None else "—"
            print(f"{cat:<12} {variant:<14} {m['n']:>4} {m['auroc']:>10.4f} {m['acc']:>8.3f} "
                  f"{m['thr']:>9.3f} {m['pred_anom_rate']*100:>7.1f}% {ev_s:>10} {flag}")
            rows.append({"category": cat, "variant": variant, "eval": ev, **m, "flag": flag})

        # free memory between categories (a 4GB GPU can't hold 15 resident backbones)
        server._model_cache.clear()
        server._variant_cache.clear()
        server._gan_cache.clear()
        try:
            if server._device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    # save markdown
    lines = ["# Verification: live AUROC / acc@thr vs Evaluation AUROC\n",
             f"device={server._device}\n",
             "| category | variant | n | AUROC_live | acc@thr | thr | pred_an% | AUROC_eval | flag |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ev_s = f"{r['eval']:.3f}" if r.get("eval") is not None else "—"
        if "skip" in r:
            lines.append(f"| {r['category']} | {r['variant']} | — | — | — | — | — | {ev_s} | SKIP: {r['skip'][:60]} |")
        else:
            lines.append(f"| {r['category']} | {r['variant']} | {r['n']} | {r['auroc']:.4f} | "
                         f"{r['acc']:.3f} | {r['thr']:.3f} | {r['pred_anom_rate']*100:.1f}% | {ev_s} | {r['flag']} |")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[verify] saved → {args.out}")

    # summary
    drift = [r for r in rows if "AUROC_DRIFT" in r.get("flag", "")]
    thr_bug = [r for r in rows if "THRESH_BUG" in r.get("flag", "")]
    print(f"\n[summary] {len(rows)} combos | AUROC_DRIFT: {len(drift)} | THRESH_BUG: {len(thr_bug)}")
    for r in thr_bug:
        print(f"  THRESH_BUG  {r['category']}/{r['variant']}: AUROC={r['auroc']:.3f} but acc@thr={r['acc']:.3f} "
              f"(predicts anomalous {r['pred_anom_rate']*100:.0f}% of the time, thr={r['thr']:.3f})")


if __name__ == "__main__":
    main()
