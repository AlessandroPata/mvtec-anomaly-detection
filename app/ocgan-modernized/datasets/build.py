from __future__ import annotations

from torch.utils.data import DataLoader
from datasets.mvtec_ad import MVTecADDataset
from datasets import DummyAnomalyDataset
from utils.transforms import Compose, DirectResize, NormalizeTensor, ResizePadToSquare, CenterCropTensor


def build_base_resize_transform(cfg):
    image_size = int(cfg.dataset.image_size)

    if cfg.preprocessing.resize_mode == "pad":
        resize_t = ResizePadToSquare(
            size=image_size,
            pad_value=float(cfg.preprocessing.pad_value),
            mode=str(cfg.preprocessing.interpolation),
            antialias=bool(cfg.preprocessing.antialias),
        )
    else:
        resize_t = DirectResize(
            size=image_size,
            mode=str(cfg.preprocessing.interpolation),
            antialias=bool(cfg.preprocessing.antialias),
        )

    transforms = [resize_t]

    if bool(cfg.preprocessing.center_crop):
        transforms.append(CenterCropTensor(image_size))

    return transforms


def build_normalization_transform(cfg):
    if not cfg.normalization.enabled:
        return None

    if cfg.normalization.mode == "imagenet":
        return NormalizeTensor(cfg.normalization.mean, cfg.normalization.std)

    if cfg.normalization.mode in {"none", "train_only_stats"}:
        return None

    raise ValueError(f"Normalization mode non supportata: {cfg.normalization.mode}")


def build_transform(cfg, split_name: str):
    transforms = build_base_resize_transform(cfg)

    norm_t = build_normalization_transform(cfg)
    if norm_t is not None:
        transforms.append(norm_t)

    return Compose(transforms)

def build_mask_transform(cfg):
    transforms = build_base_resize_transform(cfg)
    return Compose(transforms)

def build_dataset(cfg, split_cfg, split_name: str):
    image_transform = build_transform(cfg, split_name)
    mask_transform = build_mask_transform(cfg)
    transform = build_transform(cfg, split_name)
    apply_synthetic_anomalies = bool(
        cfg.synthetic_anomalies.enabled and split_name == "train_normal"
    )
    
    if cfg.dataset.name == "dummy":
        dataset = DummyAnomalyDataset(
            split=split_cfg.split,
            image_size=cfg.dataset.image_size,
            length=getattr(split_cfg, "length", 128),
            transform=transform,
            synthetic_anomalies_cfg=cfg.synthetic_anomalies,
            apply_synthetic_anomalies=apply_synthetic_anomalies,
        )
    elif cfg.dataset.name == "mvtec_ad":
        dataset = MVTecADDataset(
            root=cfg.dataset.root,
            category=cfg.dataset.category,
            split=split_cfg.split,
            image_size=cfg.dataset.image_size,
            image_transform=image_transform,
            mask_transform=mask_transform,
            synthetic_anomalies_cfg=cfg.synthetic_anomalies,
            apply_synthetic_anomalies=apply_synthetic_anomalies,
            val_normal_ratio=float(cfg.dataset.val_normal_ratio),
            val_mixed_ratio=float(cfg.dataset.val_mixed_ratio),
            seed=int(cfg.dataset.seed),
        )
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")
        
    return dataset


def build_loader(cfg, split_name: str):
    split_cfg = getattr(cfg.dataset, split_name)
    dataset = build_dataset(cfg, split_cfg, split_name)

    loader = DataLoader(
        dataset,
        batch_size=split_cfg.batch_size,
        shuffle=split_cfg.shuffle,
        num_workers=split_cfg.num_workers,
        pin_memory=split_cfg.pin_memory,
    )
    return loader


def build_all_dataloaders(cfg) -> dict[str, DataLoader]:
    return {
        "train_normal": build_loader(cfg, "train_normal"),
        "val_normal": build_loader(cfg, "val_normal"),
        "val_mixed": build_loader(cfg, "val_mixed"),
        "test_blind": build_loader(cfg, "test_blind"),
    }
