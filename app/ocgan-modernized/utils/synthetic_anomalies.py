from __future__ import annotations

import random

import numpy as np
import torch

from utils.perlin import create_perlin_anomaly


def _rand_int(low: int, high: int, rng: random.Random | None = None) -> int:
    r = rng if rng is not None else random
    return r.randint(low, high)


def apply_cutpaste_anomaly(
    image: torch.Tensor,
    min_patch_ratio: float = 0.1,
    max_patch_ratio: float = 0.3,
    py_rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    image: Tensor [C, H, W] in float
    returns:
      anomalous_image: [C, H, W]
      anomaly_mask: [1, H, W] with 0/1
    """
    c, h, w = image.shape
    r = py_rng if py_rng is not None else random
    out = image.clone()
    mask = torch.zeros((1, h, w), dtype=image.dtype, device=image.device)

    patch_h = max(1, int(h * r.uniform(min_patch_ratio, max_patch_ratio)))
    patch_w = max(1, int(w * r.uniform(min_patch_ratio, max_patch_ratio)))

    if patch_h >= h:
        patch_h = h - 1
    if patch_w >= w:
        patch_w = w - 1

    src_top = _rand_int(0, h - patch_h, r)
    src_left = _rand_int(0, w - patch_w, r)

    dst_top = _rand_int(0, h - patch_h, r)
    dst_left = _rand_int(0, w - patch_w, r)

    patch = image[:, src_top:src_top + patch_h, src_left:src_left + patch_w].clone()
    out[:, dst_top:dst_top + patch_h, dst_left:dst_left + patch_w] = patch
    mask[:, dst_top:dst_top + patch_h, dst_left:dst_left + patch_w] = 1.0

    return out, mask


def maybe_apply_synthetic_anomaly(
    image: torch.Tensor,
    enabled: bool,
    probability: float,
    mode: str,
    min_patch_ratio: float,
    max_patch_ratio: float,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if not enabled:
        h, w = image.shape[-2:]
        return image, torch.zeros((1, h, w), dtype=image.dtype, device=image.device), 0

    # Local RNGs isolate this call from global random state, enabling
    # deterministic (epoch, idx)-conditioned synthetic anomalies.
    if seed is not None:
        py_rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
    else:
        py_rng = None
        np_rng = np.random.default_rng()

    draw = py_rng.random() if py_rng is not None else random.random()
    if draw > probability:
        h, w = image.shape[-2:]
        return image, torch.zeros((1, h, w), dtype=image.dtype, device=image.device), 0

    if mode == "cutpaste":
        anomalous, mask = apply_cutpaste_anomaly(
            image=image,
            min_patch_ratio=min_patch_ratio,
            max_patch_ratio=max_patch_ratio,
            py_rng=py_rng,
        )
        return anomalous, mask, 1

    if mode == "perlin":
        # Convert tensor [C, H, W] to numpy [H, W, C]
        img_np = image.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        anomalous_np, mask_np = create_perlin_anomaly(img_np, np_rng)
        # Convert back to tensors
        anomalous = torch.from_numpy(anomalous_np).permute(2, 0, 1).to(image.device)
        mask = torch.from_numpy(mask_np).unsqueeze(0).to(image.device)
        return anomalous, mask, 1

    raise ValueError(f"Synthetic anomaly mode non supportata: {mode}")
