"""
ensemble_experiment.py — does fusing the GAN with PatchCore beat PatchCore alone?

PatchCore (production) is strong (macro image-AUROC ~0.985); the retired GAN is
much weaker (~0.83). A late-fusion ensemble only helps if the GAN contributes
*decorrelated* signal on the categories where PatchCore is weakest. We test this
honestly instead of asserting it.

For each category we score the SAME seeded, stratified sample of test images with
both the production PatchCore and the stronger GAN (ocgan_optv2). Each model's
scores are z-normalised, fused as  (1-w)·z_pc + w·z_gan, and the weight w is chosen
by stratified k-fold CV (best AUROC on the calibration folds) and then scored on
the held-out fold. We compare the held-out ensemble AUROC against PatchCore alone.

Honest model selection (w picked on calibration data, scored on held-out data)
means a reported gain is real, not an artefact of fitting w on the test labels.

Output: production_models/ensemble_experiment.json + ensemble_experiment_results.md

Usage:
    python ensemble_experiment.py --device cuda --sample 60
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import server

WEIGHTS = np.linspace(0.0, 1.0, 21)  # fusion grid: 0 = PatchCore only, 1 = GAN only


def zscore(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return (x - mu) / (sd if sd > 1e-9 else 1.0)


def score_models(category: str, sample: int, seed: int, gan_variant: str):
    """Paired (pc_scores, gan_scores, y) over the same seeded test sample."""
    imgs = server.sample_test_images(server.DATASET_ROOT, category, sample, seed)
    pc = server.get_variant_model(category, "production")
    pc_scores, y = [], []
    for im in imgs:
        path = server.DATASET_ROOT / category / "test" / im.defect / im.filename
        r = pc.predict(Image.open(path).convert("RGB"))
        pc_scores.append(float(r["anomaly_score"]))
        y.append(1 if im.is_anomaly else 0)
    gan = server.get_variant_model(category, gan_variant)
    gan_scores = []
    for im in imgs:
        path = server.DATASET_ROOT / category / "test" / im.defect / im.filename
        r = gan.predict(Image.open(path).convert("RGB"))
        gan_scores.append(float(r["anomaly_score"]))
    return (np.asarray(pc_scores, float), np.asarray(gan_scores, float), np.asarray(y, int))


def safe_auroc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def honest_ensemble(pc, gan, y, k=5, seed=43):
    """CV: pick fusion weight on calibration folds, score AUROC on held-out fold.
    Returns (pc_auroc_full, gan_auroc_full, ensemble_heldout_auroc, mean_best_w)."""
    pc_au, gan_au = safe_auroc(y, pc), safe_auroc(y, gan)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    kk = max(2, min(k, n_pos, n_neg))
    skf = StratifiedKFold(n_splits=kk, shuffle=True, random_state=seed)
    ens_aurocs, best_ws = [], []
    for calib_idx, eval_idx in skf.split(pc.reshape(-1, 1), y):
        # normalisation fit on calibration data only
        mu_pc, sd_pc = pc[calib_idx].mean(), pc[calib_idx].std()
        mu_gn, sd_gn = gan[calib_idx].mean(), gan[calib_idx].std()
        zc_pc, zc_gn = zscore(pc[calib_idx], mu_pc, sd_pc), zscore(gan[calib_idx], mu_gn, sd_gn)
        ze_pc, ze_gn = zscore(pc[eval_idx], mu_pc, sd_pc), zscore(gan[eval_idx], mu_gn, sd_gn)
        yc, ye = y[calib_idx], y[eval_idx]
        best_w, best_au = 0.0, -1.0
        for w in WEIGHTS:
            au = safe_auroc(yc, (1 - w) * zc_pc + w * zc_gn)
            if not np.isnan(au) and au > best_au:
                best_au, best_w = au, float(w)
        ens_aurocs.append(safe_auroc(ye, (1 - best_w) * ze_pc + best_w * ze_gn))
        best_ws.append(best_w)
    ens = float(np.nanmean(ens_aurocs)) if ens_aurocs else float("nan")
    return pc_au, gan_au, ens, float(np.mean(best_ws))


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    ap.add_argument("--sample", type=int, default=60, help="test images per category")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--gan", default="ocgan_optv2")
    ap.add_argument("--out", default=str(server.PRODUCTION_MODELS_DIR / "ensemble_experiment.json"))
    args = ap.parse_args()

    import torch
    server._device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"[ensemble] device={server._device} sample={args.sample} gan={args.gan}")
    print(f"\n{'category':<12} {'AUROC_pc':>9} {'AUROC_gan':>10} {'ens(heldout)':>12} {'mean_w':>7} {'delta':>8}")
    print("-" * 62)

    results: dict[str, dict] = {}
    for cat in server.CATEGORIES:
        try:
            pc, gan, y = score_models(cat, args.sample, args.seed, args.gan)
        except Exception as e:
            print(f"{cat:<12} SKIP ({type(e).__name__}: {str(e)[:40]})")
            results[cat] = {"skip": f"{type(e).__name__}: {e}"}
            server._model_cache.clear(); server._variant_cache.clear(); server._gan_cache.clear()
            continue
        pc_au, gan_au, ens, mean_w = honest_ensemble(pc, gan, y, seed=args.seed)
        delta = ens - pc_au
        results[cat] = {
            "auroc_pc": round(pc_au, 4), "auroc_gan": round(gan_au, 4),
            "auroc_ensemble_heldout": round(ens, 4), "mean_fusion_weight": round(mean_w, 3),
            "delta_vs_patchcore": round(delta, 4), "n": int(len(y)), "n_anom": int(y.sum()),
        }
        print(f"{cat:<12} {pc_au:>9.4f} {gan_au:>10.4f} {ens:>12.4f} {mean_w:>7.3f} {delta:>+8.4f}")
        server._model_cache.clear(); server._variant_cache.clear(); server._gan_cache.clear()
        try:
            if server._device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    ok = [v for v in results.values() if "skip" not in v]
    macro_pc = float(np.mean([v["auroc_pc"] for v in ok])) if ok else float("nan")
    macro_ens = float(np.nanmean([v["auroc_ensemble_heldout"] for v in ok])) if ok else float("nan")
    helped = [c for c, v in results.items() if "skip" not in v and v["delta_vs_patchcore"] > 0.002]
    macro = {"macro_auroc_pc": round(macro_pc, 4), "macro_auroc_ensemble": round(macro_ens, 4),
             "macro_delta": round(macro_ens - macro_pc, 4), "categories_helped": helped}
    verdict = ("ensemble materially beats PatchCore" if macro_ens - macro_pc > 0.002
               else "no material gain — ship PatchCore alone")
    print(f"\n[ensemble] macro PatchCore={macro_pc:.4f}  ensemble={macro_ens:.4f}  "
          f"delta={macro_ens - macro_pc:+.4f}")
    print(f"[ensemble] categories helped (>+0.002): {helped or 'none'}")
    print(f"[ensemble] VERDICT: {verdict}")

    Path(args.out).write_text(json.dumps(
        {"sample": args.sample, "gan": args.gan, "macro": macro, "verdict": verdict,
         "per_category": results}, indent=2), encoding="utf-8")

    md = ["# GAN + PatchCore late-fusion ensemble — honest held-out test\n",
          f"Sample {args.sample}/category · fusion weight chosen by stratified CV on calibration folds, "
          f"AUROC scored on held-out folds.\n",
          f"**Macro:** PatchCore alone {macro_pc:.4f} · honest ensemble {macro_ens:.4f} "
          f"(Δ {macro_ens - macro_pc:+.4f}). **Verdict: {verdict}.**\n",
          "| category | AUROC PatchCore | AUROC GAN | ensemble (held-out) | mean weight | Δ |",
          "|---|---|---|---|---|---|"]
    for c, v in results.items():
        if "skip" in v:
            md.append(f"| {c} | — | — | — | — | SKIP |")
        else:
            md.append(f"| {c} | {v['auroc_pc']:.4f} | {v['auroc_gan']:.4f} | {v['auroc_ensemble_heldout']:.4f} "
                      f"| {v['mean_fusion_weight']:.2f} | {v['delta_vs_patchcore']:+.4f} |")
    Path(server.PROJECT_ROOT / "ensemble_experiment_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[ensemble] saved -> {args.out}")


if __name__ == "__main__":
    main()
