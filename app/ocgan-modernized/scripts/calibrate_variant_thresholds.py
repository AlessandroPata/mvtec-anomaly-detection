"""Recalibrate anomaly thresholds for reconstructed PatchCore variants.

Method (identical to scripts/export_patchcore_banks.py): hold out the same 15%
val_normal split (seed 43), score each held-out normal image with the variant,
threshold = 99th percentile of those scores. The expensive backbone pass runs
once per batch per bank; variants share the coreset bank.

Usage:
    python scripts/calibrate_variant_thresholds.py --device cuda
    python scripts/calibrate_variant_thresholds.py --device cpu --categories bottle screw
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.patchcore_inference import PatchCoreInference          # noqa: E402
from models.patchcore_variants import (                            # noqa: E402
    VARIANT_SPECS, get_coreset_indices, variant_stats,
)
from models.patchcore_common import aggregate_image_score          # noqa: E402
# Same dataset + transform stack as the export script:
from datasets.mvtec_ad import MVTecADDataset                       # noqa: E402
from utils.transforms import Compose, NormalizeTensor, ResizePadToSquare  # noqa: E402

SEED = 43
VAL_NORMAL_RATIO = 0.15
IMAGE_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATASET_ROOT = ROOT.parent.parent / "datasets" / "mvtec_ad"
PRODUCTION_DIR = ROOT / "production_models"

ALL_CATEGORIES = sorted(
    d.name for d in PRODUCTION_DIR.iterdir()
    if d.is_dir() and (d / "patchcore_bank.pt").exists()
)


def val_loader(category: str):
    ds = MVTecADDataset(
        root=str(DATASET_ROOT),
        category=category,
        split="val_normal",
        image_size=IMAGE_SIZE,
        image_transform=Compose([
            ResizePadToSquare(IMAGE_SIZE),
            NormalizeTensor(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        val_normal_ratio=VAL_NORMAL_RATIO,
        val_mixed_ratio=0.0,
        seed=SEED,
    )
    return torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)


@torch.no_grad()
def patch_min_dists(model: PatchCoreInference, batch: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    """batch: [B,3,H,W] already normalized → [B, P] min distance per patch."""
    outs = model.backbone(batch.to(model.device))
    fmap = model._get_feature_map(outs)
    b, c, h, w = fmap.shape
    fmap = torch.nan_to_num(fmap, nan=0.0, posinf=0.0, neginf=0.0)
    patches = fmap.permute(0, 2, 3, 1).reshape(b, h * w, c)
    patches = F.normalize(patches, p=2, dim=2, eps=1e-8)
    flat = patches.reshape(b * h * w, c)
    dists = torch.cdist(flat, bank)
    dists = torch.nan_to_num(dists, nan=1e6, posinf=1e6, neginf=1e6)
    return dists.reshape(b, h * w, bank.shape[0]).min(dim=2).values


@torch.no_grad()
def calibrate_category(category: str, device: str, variants: list[str]) -> dict:
    t0 = time.time()
    model = PatchCoreInference(category, PRODUCTION_DIR / category / "patchcore_bank.pt", device=device)
    coreset_idx = get_coreset_indices(PRODUCTION_DIR, category, model.bank)
    coreset_bank = model.bank[coreset_idx.to(model.bank.device)]

    scores: dict[str, list[float]] = {vid: [] for vid in variants}
    n_imgs = 0
    for batch in val_loader(category):
        x = batch["image"]  # MVTecADDataset.__getitem__ returns a dict (mvtec_ad.py)
        n_imgs += x.shape[0]
        md = patch_min_dists(model, x, coreset_bank)  # both v1/v2 use the coreset bank
        for vid in variants:
            spec = VARIANT_SPECS[vid]
            s = aggregate_image_score(md, spec.aggregation, spec.topk)
            scores[vid].extend(float(v) for v in s)

    out = {vid: variant_stats(scores[vid]) for vid in variants}
    for vid, st in out.items():
        print(f"  [{category}] {vid}: n={st['n_val']} thr(p99)={st['threshold']:.4f} "
              f"mean={st['score_mean']:.4f} std={st['score_std']:.4f}")
    print(f"  [{category}] done in {time.time() - t0:.0f}s ({n_imgs} val images)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--categories", nargs="*", default=ALL_CATEGORIES)
    args = ap.parse_args()

    variants = [vid for vid, s in VARIANT_SPECS.items() if s.kind == "reconstructed"]
    out_path = PRODUCTION_DIR / "variant_thresholds.json"
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}

    for cat in args.categories:
        print(f"[calibrate] {cat} on {args.device} …")
        existing[cat] = {**existing.get(cat, {}), **calibrate_category(cat, args.device, variants)}
        out_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")  # save incrementally

    print(f"[calibrate] wrote {out_path}")


if __name__ == "__main__":
    main()
