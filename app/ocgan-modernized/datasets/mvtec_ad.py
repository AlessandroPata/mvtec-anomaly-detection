from __future__ import annotations

from pathlib import Path
from typing import Any
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.synthetic_anomalies import maybe_apply_synthetic_anomaly


class MVTecADDataset(Dataset):
    def __init__(
        self,
        root: str,
        category: str,
        split: str,
        image_size: int = 256,
        image_transform=None,
        mask_transform=None,
        synthetic_anomalies_cfg=None,
        apply_synthetic_anomalies: bool = False,
        val_normal_ratio: float = 0.2,
        val_mixed_ratio: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.category = category
        self.split = split
        self.image_size = image_size
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.synthetic_anomalies_cfg = synthetic_anomalies_cfg
        self.apply_synthetic_anomalies = apply_synthetic_anomalies
        self.val_normal_ratio = float(val_normal_ratio)
        self.val_mixed_ratio = float(val_mixed_ratio)
        self.seed = int(seed)
        self._epoch = 0

        self.items = self._build_split_items()

    def set_epoch(self, epoch: int) -> None:
        """Called by the trainer so per-item synthetic anomaly RNG depends on epoch."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def _category_root(self) -> Path:
        return self.root / self.category

    def _scan_category_files(self):
        category_root = self._category_root()

        train_good_dir = category_root / "train" / "good"
        test_dir = category_root / "test"
        gt_dir = category_root / "ground_truth"

        if not train_good_dir.exists():
            raise FileNotFoundError(f"Directory non trovata: {train_good_dir}")
        if not test_dir.exists():
            raise FileNotFoundError(f"Directory non trovata: {test_dir}")

        train_good_items = []
        for p in sorted(train_good_dir.iterdir()):
            if p.is_file():
                train_good_items.append(
                    {
                        "image_path": p,
                        "mask_path": None,
                        "label": 0,
                        "is_anomaly": 0,
                        "defect_type": "good",
                    }
                )

        test_good_items = []
        test_anomaly_items = []

        for defect_dir in sorted(test_dir.iterdir()):
            if not defect_dir.is_dir():
                continue

            defect_type = defect_dir.name
            for p in sorted(defect_dir.iterdir()):
                if not p.is_file():
                    continue

                if defect_type == "good":
                    test_good_items.append(
                        {
                            "image_path": p,
                            "mask_path": None,
                            "label": 0,
                            "is_anomaly": 0,
                            "defect_type": "good",
                        }
                    )
                else:
                    stem = p.stem
                    mask_path = gt_dir / defect_type / f"{stem}_mask.png"
                    test_anomaly_items.append(
                        {
                            "image_path": p,
                            "mask_path": mask_path,
                            "label": 1,
                            "is_anomaly": 1,
                            "defect_type": defect_type,
                        }
                    )

        return train_good_items, test_good_items, test_anomaly_items

    def _build_split_items(self):
        train_good_items, test_good_items, test_anomaly_items = self._scan_category_files()

        rng = random.Random(self.seed)
        rng.shuffle(train_good_items)
        rng.shuffle(test_good_items)
        rng.shuffle(test_anomaly_items)

        n_val_normal = int(len(train_good_items) * self.val_normal_ratio)
        n_val_good = int(len(test_good_items) * self.val_mixed_ratio)
        n_val_anom = int(len(test_anomaly_items) * self.val_mixed_ratio)

        val_normal_items = train_good_items[:n_val_normal]
        train_normal_items = train_good_items[n_val_normal:]

        val_mixed_items = test_good_items[:n_val_good] + test_anomaly_items[:n_val_anom]
        test_blind_items = test_good_items[n_val_good:] + test_anomaly_items[n_val_anom:]

        rng.shuffle(val_mixed_items)
        rng.shuffle(test_blind_items)

        if self.split == "train_normal":
            return train_normal_items
        if self.split == "val_normal":
            return val_normal_items
        if self.split == "val_mixed":
            return val_mixed_items
        if self.split == "test_blind":
            return test_blind_items

        raise ValueError(f"Split non supportato: {self.split}")

    def _load_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        image_np = np.array(image, dtype=np.float32) / 255.0
        image_t = torch.from_numpy(image_np).permute(2, 0, 1)
        return image_t

    def _load_mask_for_sample(self, item: dict[str, Any], image_shape_hw: tuple[int, int]) -> torch.Tensor:
        h, w = image_shape_hw

        if item["mask_path"] is not None and Path(item["mask_path"]).exists():
            mask = Image.open(item["mask_path"]).convert("L")
            mask_np = (np.array(mask, dtype=np.float32) > 0).astype(np.float32)
            mask_t = torch.from_numpy(mask_np).unsqueeze(0)
            return mask_t

        return torch.zeros((1, h, w), dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]

        image = self._load_image(item["image_path"])
        anomaly_mask = self._load_mask_for_sample(item, (image.shape[-2], image.shape[-1]))

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.mask_transform is not None:
            anomaly_mask = self.mask_transform(anomaly_mask)

        anomaly_mask = (anomaly_mask > 0.5).to(dtype=image.dtype)

        synthetic_image = image.clone()
        synthetic_mask = torch.zeros_like(anomaly_mask)
        synthetic_label = 0

        if self.apply_synthetic_anomalies and self.synthetic_anomalies_cfg is not None:
            item_seed = (self.seed * 1_000_003
                         + self._epoch * 100_003
                         + idx) & 0x7FFFFFFF
            synthetic_image, synthetic_mask, synthetic_label = maybe_apply_synthetic_anomaly(
                image=image,
                enabled=bool(self.synthetic_anomalies_cfg.enabled),
                probability=float(self.synthetic_anomalies_cfg.probability),
                mode=str(self.synthetic_anomalies_cfg.mode),
                min_patch_ratio=float(self.synthetic_anomalies_cfg.min_patch_ratio),
                max_patch_ratio=float(self.synthetic_anomalies_cfg.max_patch_ratio),
                seed=item_seed,
            )

        return {
            "image": image,
            "label": torch.tensor(int(item["label"]), dtype=torch.long),
            "is_anomaly": torch.tensor(int(item["is_anomaly"]), dtype=torch.long),
            "anomaly_mask": anomaly_mask,
            "synthetic_image": synthetic_image,
            "synthetic_mask": synthetic_mask,
            "synthetic_label": torch.tensor(int(synthetic_label), dtype=torch.long),
            "split": self.split,
            "path": str(item["image_path"]),
            "category": self.category,
            "defect_type": item["defect_type"],
        }