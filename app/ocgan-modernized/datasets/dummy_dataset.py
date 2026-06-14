from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from utils.synthetic_anomalies import maybe_apply_synthetic_anomaly


class DummyAnomalyDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        image_size: int = 256,
        length: int = 32,
        transform=None,
        synthetic_anomalies_cfg=None,
        apply_synthetic_anomalies: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.length = length
        self.transform = transform
        self.synthetic_anomalies_cfg = synthetic_anomalies_cfg
        self.apply_synthetic_anomalies = apply_synthetic_anomalies

    def __len__(self) -> int:
        return self.length

    def _make_image(self, idx: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(idx + hash(self.split) % 100000)
        return torch.rand((3, self.image_size, self.image_size), generator=g)

    def _make_label(self, idx: int) -> int:
        if self.split == "train_normal":
            return 0
        if self.split == "val_normal":
            return 0
        if self.split in {"val_mixed", "test_blind"}:
            return idx % 2
        return 0

    def __getitem__(self, idx: int) -> dict:
        image = self._make_image(idx)
        label = self._make_label(idx)

        if self.transform is not None:
            image = self.transform(image)

        anomaly_mask = torch.zeros((1, image.shape[-2], image.shape[-1]), dtype=image.dtype)
        if label == 1:
            anomaly_mask[:, image.shape[-2] // 4:image.shape[-2] // 2, image.shape[-1] // 4:image.shape[-1] // 2] = 1.0

        synthetic_image = image.clone()
        synthetic_mask = torch.zeros_like(anomaly_mask)
        synthetic_label = 0

        if self.apply_synthetic_anomalies and self.synthetic_anomalies_cfg is not None:
            synthetic_image, synthetic_mask, synthetic_label = maybe_apply_synthetic_anomaly(
                image=image,
                enabled=bool(self.synthetic_anomalies_cfg.enabled),
                probability=float(self.synthetic_anomalies_cfg.probability),
                mode=str(self.synthetic_anomalies_cfg.mode),
                min_patch_ratio=float(self.synthetic_anomalies_cfg.min_patch_ratio),
                max_patch_ratio=float(self.synthetic_anomalies_cfg.max_patch_ratio),
            )

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "anomaly_mask": anomaly_mask,
            "synthetic_image": synthetic_image,
            "synthetic_mask": synthetic_mask,
            "synthetic_label": torch.tensor(synthetic_label, dtype=torch.long),
            "split": self.split,
        }
"""Stub for DummyAnomalyDataset — only needed by training, not inference."""
