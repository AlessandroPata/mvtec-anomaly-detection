from __future__ import annotations

import torch
import torch.nn.functional as F


class NormalizeTensor:
    def __init__(self, mean: list[float], std: list[float]) -> None:
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(dtype=x.dtype, device=x.device)
        std = self.std.to(dtype=x.dtype, device=x.device)
        return (x - mean) / std


class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


class ResizePadToSquare:
    def __init__(
        self,
        size: int,
        pad_value: float = 0.0,
        mode: str = "bilinear",
        antialias: bool = True,
    ) -> None:
        self.size = size
        self.pad_value = pad_value
        self.mode = mode
        self.antialias = antialias

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        c, h, w = x.shape

        scale = min(self.size / h, self.size / w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        x = F.interpolate(
            x.unsqueeze(0),
            size=(new_h, new_w),
            mode=self.mode,
            align_corners=False if self.mode in {"bilinear", "bicubic"} else None,
            antialias=self.antialias if self.mode in {"bilinear", "bicubic"} else False,
        ).squeeze(0)

        pad_h = self.size - new_h
        pad_w = self.size - new_w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = F.pad(
            x,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=self.pad_value,
        )
        return x


class DirectResize:
    def __init__(
        self,
        size: int,
        mode: str = "bilinear",
        antialias: bool = True,
    ) -> None:
        self.size = size
        self.mode = mode
        self.antialias = antialias

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x.unsqueeze(0),
            size=(self.size, self.size),
            mode=self.mode,
            align_corners=False if self.mode in {"bilinear", "bicubic"} else None,
            antialias=self.antialias if self.mode in {"bilinear", "bicubic"} else False,
        ).squeeze(0)


class CenterCropTensor:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, h, w = x.shape
        crop_h = min(self.size, h)
        crop_w = min(self.size, w)

        top = max((h - crop_h) // 2, 0)
        left = max((w - crop_w) // 2, 0)

        return x[:, top:top + crop_h, left:left + crop_w]
