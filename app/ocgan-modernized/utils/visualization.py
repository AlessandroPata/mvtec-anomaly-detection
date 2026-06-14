from __future__ import annotations

from pathlib import Path

import torch
from torchvision.utils import save_image


def save_debug_images(
    images: torch.Tensor,
    output_dir: str | Path,
    prefix: str,
    max_images: int = 4,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = images.detach().cpu()[:max_images]
    for i, img in enumerate(images):
        save_image(img, output_dir / f"{prefix}_{i}.png")
