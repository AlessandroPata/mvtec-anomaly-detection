"""MVTec AD dataset for anomaly detection training and evaluation."""
from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.synthetic_anomalies import maybe_apply_synthetic_anomaly


class MVTecADDataset(Dataset):
    """
    MVTec AD dataset.

    Splits:
      - train_normal: normal training images
      - val_normal: held-out normal images for score normalization
      - val_mixed: normal + anomalous images for fusion fitting / model selection
      - test_blind: full test set (normal + all defect types)
    """

    def __init__(
        self,
        root: str,
        category: str,
        split: str,
        image_size: int = 256,
        val_normal_ratio: float = 0.2,
        val_mixed_ratio: float = 0.5,
        seed: int = 42,
        synthetic_cfg=None,
        transform=None,
        length: int | None = None,
    ) -> None:
        self.root = Path(root) / category
        self.category = category
        self.split = split
        self.image_size = image_size
        self.synthetic_cfg = synthetic_cfg
        self.length = length
        self.seed = int(seed)
        self._epoch = 0

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        self.samples: list[dict] = []
        self._build_split(val_normal_ratio, val_mixed_ratio, seed)

    def set_epoch(self, epoch: int) -> None:
        """Called by the trainer so per-item synthetic anomaly RNG depends on epoch."""
        self._epoch = int(epoch)

    def _build_split(
        self, val_normal_ratio: float, val_mixed_ratio: float, seed: int
    ) -> None:
        train_good_dir = self.root / "train" / "good"
        test_dir = self.root / "test"

        # Collect all normal training images
        all_train_normal = sorted(train_good_dir.glob("*.png"))
        if not all_train_normal:
            all_train_normal = sorted(train_good_dir.glob("*.jpg"))

        # Split train normals into train_normal and val_normal
        rng = random.Random(seed)
        indices = list(range(len(all_train_normal)))
        rng.shuffle(indices)
        n_val = max(1, int(len(all_train_normal) * val_normal_ratio))
        val_indices = set(indices[:n_val])

        train_normal_paths = [
            all_train_normal[i] for i in range(len(all_train_normal))
            if i not in val_indices
        ]
        val_normal_paths = [
            all_train_normal[i] for i in range(len(all_train_normal))
            if i in val_indices
        ]

        # Collect test images (normal + defects)
        test_samples = []
        for subdir in sorted(test_dir.iterdir()):
            if not subdir.is_dir():
                continue
            label = 0 if subdir.name == "good" else 1
            for img_path in sorted(subdir.glob("*.png")):
                test_samples.append({"path": img_path, "label": label, "defect": subdir.name})
            for img_path in sorted(subdir.glob("*.jpg")):
                test_samples.append({"path": img_path, "label": label, "defect": subdir.name})

        # Split test into val_mixed and test_blind
        rng2 = random.Random(seed + 1)
        test_indices = list(range(len(test_samples)))
        rng2.shuffle(test_indices)
        n_val_mixed = max(1, int(len(test_samples) * val_mixed_ratio))
        val_mixed_indices = set(test_indices[:n_val_mixed])

        if self.split == "train_normal":
            for p in train_normal_paths:
                self.samples.append({"path": p, "label": 0, "split": "train_normal"})
        elif self.split == "val_normal":
            for p in val_normal_paths:
                self.samples.append({"path": p, "label": 0, "split": "val_normal"})
        elif self.split == "val_mixed":
            for i in test_indices[:n_val_mixed]:
                s = test_samples[i]
                self.samples.append({
                    "path": s["path"], "label": s["label"], "split": "val_mixed"
                })
        elif self.split == "test_blind":
            for i in test_indices[n_val_mixed:]:
                s = test_samples[i]
                self.samples.append({
                    "path": s["path"], "label": s["label"], "split": "test_blind"
                })
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def __len__(self) -> int:
        if self.length is not None:
            return self.length
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        real_idx = idx % len(self.samples)
        sample = self.samples[real_idx]

        image = Image.open(sample["path"]).convert("RGB")
        image = self.transform(image)  # [C, H, W] in [0, 1]

        # Apply synthetic anomalies if configured
        if self.synthetic_cfg is not None and self.synthetic_cfg.enabled:
            train_only = getattr(self.synthetic_cfg, "train_only", True)
            apply = (self.split == "train_normal") if train_only else True

            if apply:
                item_seed = (self.seed * 1_000_003
                             + self._epoch * 100_003
                             + idx) & 0x7FFFFFFF
                synth_image, synth_mask, synth_label = maybe_apply_synthetic_anomaly(
                    image=image,
                    enabled=True,
                    probability=float(self.synthetic_cfg.probability),
                    mode=str(self.synthetic_cfg.mode),
                    min_patch_ratio=float(getattr(self.synthetic_cfg, "min_patch_ratio", 0.1)),
                    max_patch_ratio=float(getattr(self.synthetic_cfg, "max_patch_ratio", 0.3)),
                    seed=item_seed,
                )
            else:
                synth_image = image.clone()
                synth_mask = torch.zeros((1, image.shape[1], image.shape[2]))
                synth_label = 0
        else:
            synth_image = image.clone()
            synth_mask = torch.zeros((1, image.shape[1], image.shape[2]))
            synth_label = 0

        return {
            "image": image,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "split": sample["split"],
            "synthetic_image": synth_image,
            "synthetic_mask": synth_mask,
            "synthetic_label": torch.tensor(synth_label, dtype=torch.float32),
        }
