"""
Export PatchCore memory banks for all 15 MVTec categories to production_models/.

Builds wide_resnet50_2 / layer2+layer3 / max_patches=70000 bank from full training set,
calibrates threshold at 99th-percentile of training scores, saves everything.

Usage:
    python scripts/export_patchcore_banks.py [--device cuda] [--category bottle]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
sys.path.insert(0, str(ROOT))

from datasets.mvtec_ad import MVTecADDataset  # noqa: E402
from models.backbones.build import build_backbone  # noqa: E402
from utils.transforms import Compose, NormalizeTensor, ResizePadToSquare  # noqa: E402

# ---------- constants ----------

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

BACKBONE_NAME = "wide_resnet50_2"
FEATURE_LEVEL = "layer2+layer3"
AGGREGATION = "topk_reweighted"

# Per-category overrides (P1 tuning results)
FEATURE_LEVEL_OVERRIDES = {
    "screw": "layer1+layer2+layer3",  # +2.7pp vs layer2+layer3
}
TOPK = 9
MAX_PATCHES = 70000
CANDIDATE_POOL_SIZE = 20000
SEED = 43
IMAGE_SIZE = 256
BATCH_SIZE = 16
NUM_WORKERS = 4
VAL_NORMAL_RATIO = 0.15  # held-out for threshold calibration

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DATASET_ROOT = ROOT.parent.parent / "datasets" / "mvtec_ad"
PRODUCTION_DIR = ROOT / "production_models"


# ---------- backbone config ----------

def make_backbone_cfg() -> object:
    return OmegaConf.create({
        "model": {
            "backbone": {
                "name": BACKBONE_NAME,
                "pretrained": True,
                "frozen": True,
                "unfreeze_from": "none",
            },
        },
    })


# ---------- data helpers ----------

def _make_transform() -> Compose:
    return Compose([
        ResizePadToSquare(IMAGE_SIZE),
        NormalizeTensor(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_train_loader(category: str) -> DataLoader:
    ds = MVTecADDataset(
        root=str(DATASET_ROOT),
        category=category,
        split="train_normal",
        image_size=IMAGE_SIZE,
        image_transform=_make_transform(),
        val_normal_ratio=VAL_NORMAL_RATIO,
        val_mixed_ratio=0.0,
        seed=SEED,
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)


def build_valnormal_loader(category: str) -> DataLoader:
    """Held-out normal images for threshold calibration."""
    ds = MVTecADDataset(
        root=str(DATASET_ROOT),
        category=category,
        split="val_normal",
        image_size=IMAGE_SIZE,
        image_transform=_make_transform(),
        val_normal_ratio=VAL_NORMAL_RATIO,
        val_mixed_ratio=0.0,
        seed=SEED,
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)


# ---------- PatchCore helpers (same as patchcore_pure.py) ----------

@torch.no_grad()
def extract_patch_embeddings(feature_map: torch.Tensor) -> torch.Tensor:
    feature_map = torch.nan_to_num(feature_map, nan=0.0, posinf=0.0, neginf=0.0)
    patches = feature_map.permute(0, 2, 3, 1).reshape(-1, feature_map.shape[1])
    finite_rows = torch.isfinite(patches).all(dim=1)
    patches = patches[finite_rows]
    if patches.numel() == 0:
        return torch.empty((0, feature_map.shape[1]), device=feature_map.device, dtype=feature_map.dtype)
    return F.normalize(patches, p=2, dim=1, eps=1e-8)


@torch.no_grad()
def kcenter_greedy_select(features: torch.Tensor, k: int,
                          candidate_pool_size: int | None = None) -> torch.Tensor:
    n = features.shape[0]
    if k >= n:
        return torch.arange(n, device=features.device)

    x = features
    if candidate_pool_size is not None and n > candidate_pool_size:
        step = max(n // candidate_pool_size, 1)
        base_idx = torch.arange(0, n, step, device=features.device)[:candidate_pool_size]
        x = x[base_idx]
    else:
        base_idx = None

    n_work = x.shape[0]
    if k >= n_work:
        selected = torch.arange(n_work, device=x.device)
        return base_idx[selected] if base_idx is not None else selected

    center = x.mean(dim=0, keepdim=True)
    min_dists = torch.cdist(x, center).squeeze(1)
    first_idx = torch.argmax(min_dists)

    selected = [first_idx]
    min_dists = torch.cdist(x, x[first_idx:first_idx + 1]).squeeze(1)
    for _ in range(1, k):
        next_idx = torch.argmax(min_dists)
        selected.append(next_idx)
        new_dists = torch.cdist(x, x[next_idx:next_idx + 1]).squeeze(1)
        min_dists = torch.minimum(min_dists, new_dists)

    selected_idx = torch.stack(selected)
    if base_idx is not None:
        selected_idx = base_idx[selected_idx]
    return selected_idx


@torch.no_grad()
def get_feature_map(outputs: dict, level: str) -> torch.Tensor:
    if level == "layer2+layer3":
        l2 = torch.nan_to_num(outputs["layer2"], nan=0.0, posinf=0.0, neginf=0.0)
        l3 = torch.nan_to_num(outputs["layer3"], nan=0.0, posinf=0.0, neginf=0.0)
        l2_pooled = F.adaptive_avg_pool2d(l2, l3.shape[2:])
        return torch.cat([l2_pooled, l3], dim=1)
    if level == "layer1+layer2+layer3":
        l1 = torch.nan_to_num(outputs["layer1"], nan=0.0, posinf=0.0, neginf=0.0)
        l2 = torch.nan_to_num(outputs["layer2"], nan=0.0, posinf=0.0, neginf=0.0)
        l3 = torch.nan_to_num(outputs["layer3"], nan=0.0, posinf=0.0, neginf=0.0)
        l1_pooled = F.adaptive_avg_pool2d(l1, l3.shape[2:])
        l2_pooled = F.adaptive_avg_pool2d(l2, l3.shape[2:])
        return torch.cat([l1_pooled, l2_pooled, l3], dim=1)
    return outputs[level]


@torch.no_grad()
def score_loader(bank: torch.Tensor, loader: DataLoader,
                 backbone, device: str, feature_level: str = FEATURE_LEVEL) -> list[float]:
    scores = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        outs = backbone(images)
        fmap = get_feature_map(outs, feature_level)
        b, c, h, w = fmap.shape
        fmap = torch.nan_to_num(fmap, nan=0.0, posinf=0.0, neginf=0.0)
        patches = fmap.permute(0, 2, 3, 1).reshape(b, h * w, c)
        patches = F.normalize(patches, p=2, dim=2, eps=1e-8)
        flat = patches.reshape(b * h * w, c)
        dists = torch.cdist(flat, bank)
        dists = torch.nan_to_num(dists, nan=1e6, posinf=1e6, neginf=1e6)
        dists = dists.reshape(b, h * w, bank.shape[0])
        min_d = dists.min(dim=2).values  # [B, P]
        k = min(TOPK, min_d.shape[1])
        topk_d, _ = min_d.topk(k, dim=1)
        weights = 1.0 - torch.softmax(1.0 / topk_d.clamp(min=1e-6), dim=1)
        img_scores = (weights * topk_d).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-6)
        scores.extend(img_scores.cpu().numpy().tolist())
    return scores


# ---------- main ----------

def export_category(category: str, device: str) -> None:
    t0 = time.time()
    feature_level = FEATURE_LEVEL_OVERRIDES.get(category, FEATURE_LEVEL)
    print(f"\n{'='*60}")
    print(f"[export] category={category}  device={device}  feature_level={feature_level}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cfg = make_backbone_cfg()
    backbone = build_backbone(cfg).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    loader = build_train_loader(category)

    # ---- 1. Build raw bank ----
    bank_chunks = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        outs = backbone(images)
        fmap = get_feature_map(outs, feature_level)
        patches = extract_patch_embeddings(fmap).detach().cpu()
        if patches.numel():
            bank_chunks.append(patches)
    bank = torch.cat(bank_chunks, dim=0)
    print(f"[export] raw patches={bank.shape[0]} dim={bank.shape[1]} ({time.time()-t0:.0f}s)")

    bank = bank.to(device)
    if bank.shape[0] > MAX_PATCHES:
        idx = kcenter_greedy_select(bank, k=MAX_PATCHES,
                                    candidate_pool_size=CANDIDATE_POOL_SIZE)
        bank = bank[idx]
    bank = F.normalize(bank, p=2, dim=1, eps=1e-8).contiguous()
    print(f"[export] coreset={bank.shape[0]} ({time.time()-t0:.0f}s)")

    # ---- 2. Calibrate threshold on held-out val_normal (not in the bank) ----
    val_loader = build_valnormal_loader(category)
    val_scores = score_loader(bank, val_loader, backbone, device, feature_level)
    if len(val_scores) == 0:
        val_scores = score_loader(bank, build_train_loader(category), backbone, device, feature_level)
    threshold = float(np.percentile(val_scores, 99))
    train_mean = float(np.mean(val_scores))
    train_std = float(np.std(val_scores))
    print(f"[export] val_normal n={len(val_scores)} mean={train_mean:.4f} std={train_std:.4f} threshold(p99)={threshold:.4f}")

    # ---- 3. Save ----
    out_dir = PRODUCTION_DIR / category
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "patchcore_bank.pt"
    torch.save({
        "bank": bank.cpu(),
        "backbone": BACKBONE_NAME,
        "feature_level": feature_level,
        "aggregation": AGGREGATION,
        "topk": TOPK,
        "image_size": IMAGE_SIZE,
        "threshold": threshold,
        "train_score_mean": train_mean,
        "train_score_std": train_std,
        "category": category,
    }, save_path)
    size_mb = save_path.stat().st_size / 1e6
    print(f"[export] saved {save_path} ({size_mb:.1f} MB)  total={time.time()-t0:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--category", default=None, help="Single category; omit for all 15")
    args = parser.parse_args()

    cats = [args.category] if args.category else CATEGORIES
    for cat in cats:
        export_category(cat, args.device)

    print("\n[export] All done.")


if __name__ == "__main__":
    main()
