"""Shared PatchCore scoring utilities.

These are the reference implementations from scripts/patchcore_pure.py (the
evaluation script). Inference, variant reconstruction, and threshold
calibration import from here so live scores match the published eval exactly.
"""
from __future__ import annotations

import torch


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
