"""
Sprint 4 — PatchCore-pure (no GAN, no reconstruction, no fusion).

Usage:
    python scripts/patchcore_pure.py \\
        --config-path ../configs \\
        --config-name experiments/final_per_category/bottle \\
        model.backbone.name=wide_resnet50_2 \\
        memory_bank.aggregation=topk_mean \\
        memory_bank.topk=3

Reads only what it needs from cfg (dataset + memory_bank); ignores all
training/loss/fusion machinery.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
sys.path.insert(0, str(ROOT))

from datasets.build import build_loader  # noqa: E402
from models.backbones.build import build_backbone  # noqa: E402


# ---------- helpers (lifted from base_trainer.py, kept verbatim) ----------

@torch.no_grad()
def extract_patch_embeddings(feature_map: torch.Tensor) -> torch.Tensor:
    if feature_map.ndim != 4:
        raise ValueError(f"Expected 4D feature map, got shape {feature_map.shape}")
    feature_map = torch.nan_to_num(feature_map, nan=0.0, posinf=0.0, neginf=0.0)
    patches = feature_map.permute(0, 2, 3, 1).reshape(-1, feature_map.shape[1])
    finite_rows = torch.isfinite(patches).all(dim=1)
    patches = patches[finite_rows]
    if patches.numel() == 0:
        return torch.empty((0, feature_map.shape[1]),
                           device=feature_map.device, dtype=feature_map.dtype)
    return F.normalize(patches, p=2, dim=1, eps=1e-8)


@torch.no_grad()
def kcenter_greedy_select(features: torch.Tensor, k: int,
                          init: str = "mean",
                          candidate_pool_size: int | None = None) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"Expected [N, D], got shape={tuple(features.shape)}")
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

    if init == "mean":
        center = x.mean(dim=0, keepdim=True)
        min_dists = torch.cdist(x, center).squeeze(1)
        first_idx = torch.argmax(min_dists)
    else:
        first_idx = torch.randint(0, n_work, (1,), device=x.device).squeeze(0)

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
def aggregate_image_score(min_dists: torch.Tensor, aggregation: str, topk: int) -> torch.Tensor:
    """min_dists: [B, P]  ->  [B] image-level score."""
    b, p = min_dists.shape
    if aggregation == "topk_mean":
        k = min(topk, p)
        return min_dists.topk(k, dim=1).values.mean(dim=1)
    if aggregation == "topk_reweighted":
        k = min(topk, p)
        topk_d, _ = min_dists.topk(k, dim=1)
        weights = 1.0 - torch.softmax(1.0 / topk_d.clamp(min=1e-6), dim=1)
        return (weights * topk_d).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-6)
    if aggregation == "mean":
        return min_dists.mean(dim=1)
    if aggregation == "max":
        return min_dists.max(dim=1).values
    raise ValueError(f"Unsupported aggregation: {aggregation}")


# ---------- main ----------

@hydra.main(version_base=None, config_path="../configs", config_name="default_mvtec")
def main(cfg: DictConfig) -> None:
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Force frozen backbone, deterministic dataset split
    seed = int(cfg.project.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Override backbone to forced-frozen
    backbone = build_backbone(cfg).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    train_loader = build_loader(cfg, "train_normal")
    test_loader = build_loader(cfg, "test_blind")

    feature_level = str(cfg.memory_bank.feature_level)
    aggregation = str(cfg.memory_bank.aggregation)
    topk = int(getattr(cfg.memory_bank, "topk", 3))
    max_patches = int(cfg.memory_bank.max_patches)
    candidate_pool_size = int(getattr(cfg.memory_bank, "candidate_pool_size", 20000))

    print(f"[patchcore-pure] backbone={cfg.model.backbone.name} "
          f"level={feature_level} agg={aggregation} k={topk} "
          f"coreset={max_patches} pool={candidate_pool_size}")

    # ---- 1. Build memory bank ----
    bank_chunks = []
    for batch in train_loader:
        images = batch["image"].to(device, non_blocking=True)
        outs = backbone(images)
        fmap = get_feature_map(outs, feature_level)
        patches = extract_patch_embeddings(fmap).detach()
        if patches.numel():
            bank_chunks.append(patches)
    bank = torch.cat(bank_chunks, dim=0)
    print(f"[bank] raw patches={bank.shape[0]} dim={bank.shape[1]}")

    if bank.shape[0] > max_patches:
        idx = kcenter_greedy_select(bank, k=max_patches, init="mean",
                                    candidate_pool_size=candidate_pool_size)
        bank = bank[idx]
    bank = F.normalize(bank, p=2, dim=1, eps=1e-8).contiguous()
    print(f"[bank] coreset={bank.shape[0]} (build {time.time()-t0:.1f}s)")

    # ---- 2. Score test set ----
    y_true, y_score = [], []
    for batch in test_loader:
        images = batch["image"].to(device, non_blocking=True)
        outs = backbone(images)
        fmap = get_feature_map(outs, feature_level)
        b = fmap.shape[0]
        # extract patches without flattening across batch
        fmap = torch.nan_to_num(fmap, nan=0.0, posinf=0.0, neginf=0.0)
        patches = fmap.permute(0, 2, 3, 1).reshape(b, -1, fmap.shape[1])  # [B, P, C]
        patches = F.normalize(patches, p=2, dim=2, eps=1e-8)
        flat = patches.reshape(b * patches.shape[1], patches.shape[2])
        dists = torch.cdist(flat, bank)
        dists = torch.nan_to_num(dists, nan=1e6, posinf=1e6, neginf=1e6)
        dists = dists.reshape(b, patches.shape[1], bank.shape[0])
        min_d = dists.min(dim=2).values  # [B, P]
        scores = aggregate_image_score(min_d, aggregation, topk)
        y_score.extend(scores.cpu().numpy().tolist())
        y_true.extend(batch["label"].cpu().numpy().tolist())

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fpr95 = float(fpr[np.argmin(np.abs(tpr - 0.95))])
    # best F1
    p_grid = np.linspace(0.0, 1.0, 101)
    bf1 = 0.0
    s_min, s_max = float(y_score.min()), float(y_score.max())
    for q in p_grid:
        thr_q = s_min + q * (s_max - s_min)
        f1 = f1_score(y_true, (y_score >= thr_q).astype(int), zero_division=0)
        if f1 > bf1:
            bf1 = f1

    cat = str(cfg.dataset.category)
    elapsed = time.time() - t0
    print(f"[Test] cat={cat} seed={seed} AUROC={auroc:.4f} AUPRC={auprc:.4f} "
          f"best_F1={bf1:.4f} FPR@95={fpr95:.4f} elapsed={elapsed:.1f}s")
    print("Training finished.")  # sentinel for run-state tracking

    # CSV append for aggregator
    out_csv = Path(os.environ.get("PATCHCORE_CSV", "logs/patchcore_pure.csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not out_csv.exists()
    with open(out_csv, "a") as f:
        if new:
            f.write("category,seed,backbone,feature_level,aggregation,topk,coreset,auroc,auprc,best_f1,fpr95,elapsed_s\n")
        f.write(f"{cat},{seed},{cfg.model.backbone.name},{feature_level},{aggregation},{topk},"
                f"{max_patches},{auroc:.4f},{auprc:.4f},{bf1:.4f},{fpr95:.4f},{elapsed:.1f}\n")


if __name__ == "__main__":
    main()
