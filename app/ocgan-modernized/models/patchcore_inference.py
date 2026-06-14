"""
PatchCoreInference — production inference using a pre-built memory bank.

Loads backbone + saved bank from production_models/{cat}/patchcore_bank.pt.
Returns anomaly score, heatmap (from patch distances), and is_anomalous flag.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.backbones.build import build_backbone  # noqa: E402
from models.patchcore_common import aggregate_image_score  # noqa: E402
from utils.transforms import NormalizeTensor, ResizePadToSquare  # noqa: E402


class PatchCoreInference:
    """
    Production inference model backed by a PatchCore memory bank.

    The bank is built once at training time and saved to disk.
    At inference time: load backbone + bank → extract patches → min-dist → aggregate.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, category: str, bank_path: Path, device: str = "cpu") -> None:
        self.category = category
        self.device = device

        ckpt = torch.load(bank_path, map_location="cpu", weights_only=False)
        self.bank: torch.Tensor = ckpt["bank"].to(device)
        self.feature_level: str = ckpt["feature_level"]
        self.aggregation: str = ckpt["aggregation"]
        self.topk: int = int(ckpt["topk"])
        self.image_size: int = int(ckpt["image_size"])
        self.threshold: float = float(ckpt["threshold"])
        self.train_score_mean: float = float(ckpt.get("train_score_mean", 0.0))
        self.train_score_std: float = float(ckpt.get("train_score_std", 1.0))

        # Build backbone (same config used during export)
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({
            "model": {
                "backbone": {
                    "name": ckpt["backbone"],
                    "pretrained": True,
                    "frozen": True,
                    "unfreeze_from": "none",
                    "output_layers": ["layer1", "layer2", "layer3", "layer4"],
                },
            },
        })
        self.backbone = build_backbone(cfg).to(device).eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self._preprocess = [
            ResizePadToSquare(self.image_size),
            NormalizeTensor(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ]

    def preprocess_image(self, pil_image: Image.Image) -> torch.Tensor:
        img = pil_image.convert("RGB")
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        for t in self._preprocess:
            tensor = t(tensor)
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def _get_feature_map(self, outputs: dict) -> torch.Tensor:
        if self.feature_level == "layer2+layer3":
            l2 = torch.nan_to_num(outputs["layer2"], nan=0.0, posinf=0.0, neginf=0.0)
            l3 = torch.nan_to_num(outputs["layer3"], nan=0.0, posinf=0.0, neginf=0.0)
            l2_pooled = F.adaptive_avg_pool2d(l2, l3.shape[2:])
            return torch.cat([l2_pooled, l3], dim=1)
        if self.feature_level == "layer1+layer2+layer3":
            l1 = torch.nan_to_num(outputs["layer1"], nan=0.0, posinf=0.0, neginf=0.0)
            l2 = torch.nan_to_num(outputs["layer2"], nan=0.0, posinf=0.0, neginf=0.0)
            l3 = torch.nan_to_num(outputs["layer3"], nan=0.0, posinf=0.0, neginf=0.0)
            l1_pooled = F.adaptive_avg_pool2d(l1, l3.shape[2:])
            l2_pooled = F.adaptive_avg_pool2d(l2, l3.shape[2:])
            return torch.cat([l1_pooled, l2_pooled, l3], dim=1)
        return outputs[self.feature_level]

    @torch.no_grad()
    def _aggregate(self, min_dists: torch.Tensor) -> torch.Tensor:
        """min_dists: [B, P] → [B] image-level score."""
        return aggregate_image_score(min_dists, self.aggregation, self.topk)

    @torch.no_grad()
    def anomaly_map(self, pil_image: Image.Image) -> np.ndarray:
        """Raw (un-normalized) per-pixel anomaly map at image_size resolution.

        Same min-distance field as predict()'s heatmap but WITHOUT the per-image
        min/max normalization, so values are comparable across images — required
        for pixel-level AUROC / AUPRO against the ground-truth masks.
        """
        tensor = self.preprocess_image(pil_image)
        outs = self.backbone(tensor)
        fmap = self._get_feature_map(outs)
        b, c, h, w = fmap.shape
        fmap_clean = torch.nan_to_num(fmap, nan=0.0, posinf=0.0, neginf=0.0)
        patches = fmap_clean.permute(0, 2, 3, 1).reshape(b, h * w, c)
        patches = F.normalize(patches, p=2, dim=2, eps=1e-8)
        flat = patches.reshape(b * h * w, c)
        dists = torch.cdist(flat, self.bank)
        dists = torch.nan_to_num(dists, nan=1e6, posinf=1e6, neginf=1e6)
        min_d = dists.reshape(b, h * w, self.bank.shape[0]).min(dim=2).values
        heat = F.interpolate(min_d.reshape(b, 1, h, w),
                             size=(self.image_size, self.image_size),
                             mode="bilinear", align_corners=False)
        return heat.squeeze().cpu().numpy()

    @torch.no_grad()
    def predict(self, pil_image: Image.Image) -> dict:
        t0 = time.perf_counter()

        tensor = self.preprocess_image(pil_image)
        outs = self.backbone(tensor)
        fmap = self._get_feature_map(outs)
        b, c, h, w = fmap.shape

        # per-patch distances
        fmap_clean = torch.nan_to_num(fmap, nan=0.0, posinf=0.0, neginf=0.0)
        patches = fmap_clean.permute(0, 2, 3, 1).reshape(b, h * w, c)
        patches = F.normalize(patches, p=2, dim=2, eps=1e-8)
        flat = patches.reshape(b * h * w, c)
        dists = torch.cdist(flat, self.bank)
        dists = torch.nan_to_num(dists, nan=1e6, posinf=1e6, neginf=1e6)
        dists = dists.reshape(b, h * w, self.bank.shape[0])
        min_d = dists.min(dim=2).values  # [B, P]

        # image-level score
        score = float(self._aggregate(min_d).item())

        # spatial heatmap: reshape min_d to (H, W), upsample to image_size
        heat = min_d.reshape(b, 1, h, w)
        heat = F.interpolate(heat, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        heat_np = heat.squeeze().cpu().numpy()
        hmin, hmax = heat_np.min(), heat_np.max()
        if hmax - hmin > 1e-8:
            heat_np = (heat_np - hmin) / (hmax - hmin)
        else:
            heat_np = np.zeros_like(heat_np)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        is_anomalous = score >= self.threshold

        # Normalize score to rough [0,1] for anomaly_probability
        if self.train_score_std > 1e-8:
            z = (score - self.train_score_mean) / self.train_score_std
            # sigmoid to squash to (0,1); z=0 → 0.5, z>>0 → anomalous
            anomaly_probability = float(1.0 / (1.0 + np.exp(-z)))
        else:
            anomaly_probability = None

        return {
            "anomaly_score": round(score, 6),
            "anomaly_probability": round(anomaly_probability, 4) if anomaly_probability is not None else None,
            "is_anomalous": bool(is_anomalous),
            "threshold": round(self.threshold, 6),
            "category": self.category,
            "inference_time_ms": round(elapsed_ms, 1),
            "score_components": {"patchcore_score": round(score, 6)},
            "heatmap": heat_np,
            "reconstruction": None,
        }
