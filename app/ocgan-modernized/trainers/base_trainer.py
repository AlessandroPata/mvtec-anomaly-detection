from __future__ import annotations
from utils.data_audit import (
    audit_split_contamination,
    compute_channel_stats,
    find_exact_duplicates,
    summarize_labels,
)
import time
import json
import csv
from torchvision.utils import save_image
from models.discriminative_head import SimpleDiscriminativeHead
import torch.nn.functional as F
from utils.mining import pgd_mine_latent
from losses import build_reconstruction_loss
from utils.model_selection import compute_selection_score
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from utils.histograms import save_score_histogram
import numpy as np
from metrics import compute_f1_at_threshold, compute_image_level_metrics
import torch
from models.backbones import build_backbone
from torch import nn
from torch.optim import AdamW
from utils.latent import LatentCenter
from datasets.build import build_all_dataloaders
from models.components import ReconstructionModel
from utils.logger import RunLogger
from utils.profiler import build_profiler, profiler_context
from utils.training import EMA, check_tensor_finite, clip_gradients
from utils.visualization import save_debug_images
from sklearn.linear_model import LogisticRegression
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


@dataclass
class BaseTrainer:
    cfg: Any
    run_dir: str

    def _extract_multiscale_patches(
        self, outputs: dict[str, torch.Tensor], feature_level: str
    ) -> torch.Tensor:
        """Return [B, H*W, C] patch tensor. Handles 'layer2+layer3' multi-scale concat."""
        if feature_level == "layer2+layer3":
            l2 = torch.nan_to_num(outputs["layer2"], nan=0.0, posinf=0.0, neginf=0.0)
            l3 = torch.nan_to_num(outputs["layer3"], nan=0.0, posinf=0.0, neginf=0.0)
            # Pool layer2 (32x32) down to layer3 spatial (16x16)
            l2_pooled = F.adaptive_avg_pool2d(l2, l3.shape[2:])
            feature_map = torch.cat([l2_pooled, l3], dim=1)  # [B, 1536, H, W]
        else:
            feature_map = torch.nan_to_num(
                self.get_feature_tensor_by_level(outputs, feature_level),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
        b, c, h, w = feature_map.shape
        patches = feature_map.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return torch.nan_to_num(patches, nan=0.0, posinf=0.0, neginf=0.0)

    def compute_memory_score(
        self,
        outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not self.cfg.memory_bank.enabled or self.memory_bank is None:
            batch_size = outputs["global"].shape[0]
            return torch.zeros(batch_size, device=self.device)

        feature_level = str(self.cfg.memory_bank.feature_level)
        aggregation = str(self.cfg.memory_bank.aggregation)
        topk = int(getattr(self.cfg.memory_bank, "topk", 1))

        patches = self._extract_multiscale_patches(outputs, feature_level)  # [B, P, C]
        b, p, c = patches.shape

        memory = torch.nan_to_num(self.memory_bank, nan=0.0, posinf=0.0, neginf=0.0)
        finite_rows = torch.isfinite(memory).all(dim=1)
        memory = memory[finite_rows]

        if memory.numel() == 0:
            return torch.zeros(b, device=self.device, dtype=patches.dtype)

        memory = F.normalize(memory, p=2, dim=1, eps=1e-8)  # [M, C]
        patches_norm = F.normalize(patches.reshape(b * p, c), p=2, dim=1, eps=1e-8)
        patches_norm = torch.nan_to_num(patches_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # Vectorized cdist: [B*P, M]
        try:
            dists = torch.cdist(patches_norm, memory)  # [B*P, M]
        except Exception as e:
            print(f"[memory_bank] cdist failed: {e}")
            return torch.zeros(b, device=self.device, dtype=patches.dtype)

        dists = torch.nan_to_num(dists, nan=1e6, posinf=1e6, neginf=1e6)
        dists = dists.reshape(b, p, memory.shape[0])  # [B, P, M]

        # Nearest-neighbour distance per patch
        min_dists = dists.min(dim=2).values  # [B, P]
        min_dists = torch.nan_to_num(min_dists, nan=0.0, posinf=0.0, neginf=0.0)

        if aggregation == "topk_reweighted":
            # PatchCore image-level score: top-k patch distances, reweighted
            # w_i = 1 - softmax(1/d_{i,1}) to down-weight redundant patches
            k = min(topk, p)
            topk_dists, topk_idx = min_dists.topk(k, dim=1)  # [B, k]
            # For each top-k patch, find its nearest-neighbour dist to the other top-k patches
            # Simplified: use 1 - softmax over topk_dists as reweighting (avoid O(k²) cdist)
            weights = 1.0 - torch.softmax(1.0 / (topk_dists.clamp(min=1e-6)), dim=1)  # [B, k]
            img_scores = (weights * topk_dists).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-6)
        elif aggregation == "topk_mean":
            k = min(topk, p)
            topk_dists, _ = min_dists.topk(k, dim=1)
            img_scores = topk_dists.mean(dim=1)
        elif aggregation == "mean":
            img_scores = min_dists.mean(dim=1)
        elif aggregation == "max":
            img_scores = min_dists.max(dim=1).values
        else:
            raise ValueError(f"Unsupported memory aggregation: {aggregation}")

        result = torch.nan_to_num(img_scores, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    def get_fusion_feature_names(self) -> list[str]:
        if bool(self.cfg.score_fusion_learned.use_normalized_components):
            return [
                "norm_recon_score",
                "norm_perceptual_score",
                "norm_feature_score",
                "norm_latent_score",
                "norm_memory_score",
                "norm_discriminative_score",
                "norm_teacher_student_score",
            ]
        return [
            "recon_score",
            "perceptual_score",
            "feature_score",
            "latent_score",
            "memory_score",
            "discriminative_score",
            "teacher_student_score",
        ]

    @torch.no_grad()
    
    def _sanitize_numpy_1d(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        return arr[np.isfinite(arr)]
    
    def _tensor_finite_mask(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 0:
            return torch.isfinite(x).reshape(1)
        if x.ndim == 1:
            return torch.isfinite(x)
        # Flatten all dims except batch, then check all finite
        flat = x.reshape(x.shape[0], -1)
        return torch.isfinite(flat).all(dim=1)
    
    @torch.no_grad()
    def kcenter_greedy_select(
        self,
        features: torch.Tensor,
        k: int,
        init: str = "mean",
        candidate_pool_size: int | None = None,
    ) -> torch.Tensor:
        """
        Select k rows from `features` using k-center greedy.

        Args:
            features: [N, D] normalized feature matrix
            k: number of points to keep
            init: initialization strategy, one of {"mean", "random"}
            candidate_pool_size: optional pre-subsample size for speed.
                If set and N > candidate_pool_size, first reduce deterministically.

        Returns:
            indices: [k] indices into the (possibly reduced) feature matrix passed
                     to the greedy routine.
        """
        if features.ndim != 2:
            raise ValueError(f"Expected [N, D], got shape={tuple(features.shape)}")

        n = features.shape[0]
        if k >= n:
            return torch.arange(n, device=features.device)

        x = features

        # Optional deterministic pre-subsample for speed
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

        # Initialization
        if init == "mean":
            center = x.mean(dim=0, keepdim=True)                       # [1, D]
            min_dists = torch.cdist(x, center).squeeze(1)             # [N]
            first_idx = torch.argmax(min_dists)
        elif init == "random":
            first_idx = torch.randint(0, n_work, (1,), device=x.device).squeeze(0)
            min_dists = torch.cdist(x, x[first_idx:first_idx+1]).squeeze(1)
        else:
            raise ValueError(f"Unsupported init: {init}")

        selected = [first_idx]
        min_dists = torch.cdist(x, x[first_idx:first_idx+1]).squeeze(1)

        for _ in range(1, k):
            next_idx = torch.argmax(min_dists)
            selected.append(next_idx)

            new_dists = torch.cdist(x, x[next_idx:next_idx+1]).squeeze(1)
            min_dists = torch.minimum(min_dists, new_dists)

        selected_idx = torch.stack(selected)

        if base_idx is not None:
            selected_idx = base_idx[selected_idx]

        return selected_idx

    def _log_non_finite_score(
        self,
        score_name: str,
        score: torch.Tensor,
        batch: dict[str, Any] | None = None,
    ) -> None:
        mask = self._tensor_finite_mask(score)
        if bool(mask.all()):
            return

        bad_idx = torch.where(~mask)[0].detach().cpu().tolist()
        values = score.detach().cpu().numpy()

        print(f"[non_finite_score] {score_name}: bad_idx={bad_idx}")

        if batch is not None:
            labels = batch.get("label")
            splits = batch.get("split")

            for i in bad_idx[:8]:
                label_i = int(labels[i]) if labels is not None else None
                split_i = splits[i] if splits is not None else None
                print(
                    f"[non_finite_score] {score_name}: "
                    f"sample_idx={i} label={label_i} split={split_i} value={values[i]}"
                )

    def _disable_score_component(self, score_name: str, score: torch.Tensor) -> torch.Tensor:
        print(f"[score_component_disabled] {score_name} -> replaced with zeros")
        return torch.zeros_like(score)
    
    def _sanitize_numpy_2d(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _sanitize_torch_score(self, score: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    def _sanitize_bn_running_stats(self) -> None:
        # AMP fp16 forward pass can corrupt BN running stats (running_mean/var updated
        # during forward, before GradScaler can block the step). Reset any non-finite
        # values to neutral (mean=0, var=1) to prevent NaN propagation at eval time.
        for m in self.model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                if m.running_mean is not None and not torch.isfinite(m.running_mean).all():
                    m.running_mean.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                if m.running_var is not None and not torch.isfinite(m.running_var).all():
                    m.running_var.nan_to_num_(nan=1.0, posinf=1.0, neginf=1.0)
                    m.running_var.clamp_(min=0.0)
    
    def collect_fusion_training_data(self, loader) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()

        feature_names = self.get_fusion_feature_names()
        features = []
        labels = []

        for batch in loader:
            images = batch["image"].to(self.device)

            with torch.cuda.amp.autocast(
                enabled=bool(self.cfg.runtime.amp and self.device == "cuda"),
            ):
                outputs = self.model(images)
                components = self.compute_score_components(batch, outputs)

            batch_features = []
            for name in feature_names:
                batch_features.append(components[name].detach().cpu().numpy())

            batch_features = np.stack(batch_features, axis=1)
            features.append(batch_features)

            labels.append(batch["label"].cpu().numpy())

        x = np.concatenate(features, axis=0).astype(np.float64)
        y = np.concatenate(labels, axis=0).astype(np.int64)

        x = self._sanitize_numpy_2d(x)

        valid_rows = np.isfinite(x).all(axis=1)
        x = x[valid_rows]
        y = y[valid_rows]

        return x, y

    def fit_learned_score_fusion(self) -> None:
        if not self.cfg.score_fusion_learned.enabled:
            self.learned_score_fusion = None
            self.learned_score_fusion_feature_names = []
            return

        fit_split = str(self.cfg.score_fusion_learned.fit_split)

        if fit_split == "val_mixed":
            loader = self.val_mixed_loader
        else:
            raise ValueError(f"Unsupported learned fusion fit split: {fit_split}")

        x, y = self.collect_fusion_training_data(loader)

        if x.shape[0] == 0:
            print("[score_fusion_learned] skipped: empty fusion training data")
            self.learned_score_fusion = None
            self.learned_score_fusion_feature_names = []
            return

        if len(np.unique(y)) < 2:
            print("[score_fusion_learned] skipped: fusion training data has a single class")
            self.learned_score_fusion = None
            self.learned_score_fusion_feature_names = []
            return

        x = self._sanitize_numpy_2d(x)

        model = LogisticRegression(
            max_iter=int(self.cfg.score_fusion_learned.max_iter),
            C=float(self.cfg.score_fusion_learned.C),
        )

        try:
            model.fit(x, y)
        except Exception as e:
            print(f"[score_fusion_learned] disabled due to fit failure: {e}")
            self.learned_score_fusion = None
            self.learned_score_fusion_feature_names = []
            return

        self.learned_score_fusion = model
        self.learned_score_fusion_feature_names = self.get_fusion_feature_names()

        print(f"[score_fusion_learned] fitted {self.cfg.score_fusion_learned.method} on {fit_split}")
        print(f"[score_fusion_learned] features={self.learned_score_fusion_feature_names}")
        print(f"[score_fusion_learned] coef={model.coef_.tolist()} intercept={model.intercept_.tolist()}")

    def apply_learned_score_fusion(self, components: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.learned_score_fusion is None:
            return components["final_score"]

        feature_names = self.learned_score_fusion_feature_names
        device = components["final_score"].device

        batch_features = []
        for name in feature_names:
            batch_features.append(
                self._sanitize_torch_score(components[name]).detach().cpu().numpy()
            )

        x = np.stack(batch_features, axis=1).astype(np.float64)
        x = self._sanitize_numpy_2d(x)

        try:
            probs = self.learned_score_fusion.predict_proba(x)[:, 1]
        except Exception as e:
            print(f"[score_fusion_learned] predict fallback to linear score due to: {e}")
            return components["final_score"]

        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)

        return torch.tensor(probs, device=device, dtype=components["final_score"].dtype)

    @torch.no_grad()
    def build_memory_bank(self) -> None:
        if not self.cfg.memory_bank.enabled:
            self.memory_bank = None
            return

        self.model.eval()

        feature_level = str(self.cfg.memory_bank.feature_level)
        max_train_batches = int(self.cfg.memory_bank.max_train_batches)
        max_patches = int(self.cfg.memory_bank.max_patches)

        collected = []

        for batch_idx, batch in enumerate(self.train_loader):
            # max_train_batches == -1 means use all batches
            if max_train_batches != -1 and batch_idx >= max_train_batches:
                break

            images = batch["image"].to(self.device)

            with torch.cuda.amp.autocast(
                enabled=bool(self.cfg.runtime.amp and self.device == "cuda"),
            ):
                outputs = self.model(images)

            if feature_level == "layer2+layer3":
                # Multi-scale: pool layer2 to layer3 spatial, concat → [B, 1536, H, W]
                l2 = torch.nan_to_num(outputs["layer2"], nan=0.0, posinf=0.0, neginf=0.0)
                l3 = torch.nan_to_num(outputs["layer3"], nan=0.0, posinf=0.0, neginf=0.0)
                l2_pooled = F.adaptive_avg_pool2d(l2, l3.shape[2:])
                feature_map = torch.cat([l2_pooled, l3], dim=1)
            else:
                feature_map = self.get_feature_tensor_by_level(outputs, feature_level)
            patch_embeddings = self.extract_patch_embeddings(feature_map).detach()

            if patch_embeddings.numel() == 0:
                continue

            collected.append(patch_embeddings)

        if not collected:
            self.memory_bank = None
            return

        memory_bank = torch.cat(collected, dim=0)

        if memory_bank.shape[0] > max_patches:
            selection_method = "kcenter_greedy"
            candidate_pool_size = None
            init_method = "mean"

            if self._cfg_has("memory_bank") and "selection_method" in self.cfg.memory_bank:
                selection_method = str(self.cfg.memory_bank.selection_method)

            if self._cfg_has("memory_bank") and "candidate_pool_size" in self.cfg.memory_bank:
                candidate_pool_size = int(self.cfg.memory_bank.candidate_pool_size)

            if self._cfg_has("memory_bank") and "kcenter_init" in self.cfg.memory_bank:
                init_method = str(self.cfg.memory_bank.kcenter_init)

            if selection_method == "random":
                perm = torch.randperm(memory_bank.shape[0], device=memory_bank.device)[:max_patches]
                memory_bank = memory_bank[perm]
                print(f"[memory_bank] random subsample -> kept {memory_bank.shape[0]} patches")
            elif selection_method == "kcenter_greedy":
                selected_idx = self.kcenter_greedy_select(
                    memory_bank,
                    k=max_patches,
                    init=init_method,
                    candidate_pool_size=candidate_pool_size,
                )
                memory_bank = memory_bank[selected_idx]
                print(
                    f"[memory_bank] kcenter_greedy subsample -> kept {memory_bank.shape[0]} patches "
                    f"(candidate_pool_size={candidate_pool_size}, init={init_method})"
                )
            else:
                raise ValueError(f"Unsupported memory_bank.selection_method: {selection_method}")

        memory_bank = torch.nan_to_num(memory_bank, nan=0.0, posinf=0.0, neginf=0.0)
        finite_rows = torch.isfinite(memory_bank).all(dim=1)
        memory_bank = memory_bank[finite_rows]

        if memory_bank.numel() == 0:
            self.memory_bank = None
            print("[memory_bank] empty after finite filtering")
            return

        memory_bank = F.normalize(memory_bank, p=2, dim=1, eps=1e-8)
        finite_rows = torch.isfinite(memory_bank).all(dim=1)
        memory_bank = memory_bank[finite_rows]

        if memory_bank.numel() == 0:
            self.memory_bank = None
            print("[memory_bank] empty after normalization")
            return

        self.memory_bank = memory_bank.contiguous()

        if not torch.isfinite(self.memory_bank).all():
            print("[memory_bank][debug] built memory bank still contains non-finite values")

        print(
            f"[memory_bank] level={feature_level} "
            f"patches={self.memory_bank.shape[0]} "
            f"dim={self.memory_bank.shape[1]}"
        )

    def extract_patch_embeddings(self, feature_map: torch.Tensor) -> torch.Tensor:
        if feature_map.ndim != 4:
            raise ValueError(f"Expected 4D feature map, got shape {feature_map.shape}")

        feature_map = torch.nan_to_num(feature_map, nan=0.0, posinf=0.0, neginf=0.0)
        patches = feature_map.permute(0, 2, 3, 1).reshape(-1, feature_map.shape[1])
        finite_rows = torch.isfinite(patches).all(dim=1)
        patches = patches[finite_rows]

        if patches.numel() == 0:
            return torch.empty((0, feature_map.shape[1]), device=feature_map.device, dtype=feature_map.dtype)

        patches = F.normalize(patches, p=2, dim=1, eps=1e-8)
        return patches

    def compute_feature_discriminative_loss(
        self,
        clean_features: torch.Tensor,
        synthetic_features: torch.Tensor,
        synthetic_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.feature_discriminator is None:
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero

        clean_pooled = self.pool_feature_tensor(clean_features)
        synthetic_pooled = self.pool_feature_tensor(synthetic_features)

        clean_logits = self.feature_discriminator(clean_pooled).squeeze(-1)
        synthetic_logits = self.feature_discriminator(synthetic_pooled).squeeze(-1)

        clean_targets = torch.zeros_like(clean_logits)
        synthetic_targets = synthetic_labels.float()

        loss_clean = F.binary_cross_entropy_with_logits(clean_logits, clean_targets)
        loss_synth = F.binary_cross_entropy_with_logits(synthetic_logits, synthetic_targets)

        total_loss = 0.5 * (loss_clean + loss_synth)

        with torch.no_grad():
            synth_score = torch.sigmoid(synthetic_logits).mean()

        return total_loss, synth_score

    def pool_feature_tensor(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 2:
            return features
        if features.ndim == 4:
            return torch.mean(features, dim=(2, 3))
        raise ValueError(f"Unsupported feature tensor shape: {features.shape}")

    def get_feature_tensor_by_level(
        self,
        outputs: dict[str, torch.Tensor],
        level: str,
    ) -> torch.Tensor:
        if level not in outputs:
            raise KeyError(f"Feature level '{level}' not found in model outputs. Available keys: {list(outputs.keys())}")
        return outputs[level]

    def build_feature_discriminator(self, in_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 2, 1),
        )

    def generate_feature_space_anomalies(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg.feature_synthetic_anomalies

        batch_size = features.shape[0]
        device = features.device

        synthetic_features = features.clone()
        synthetic_labels = torch.zeros(batch_size, device=device, dtype=torch.float32)

        if not cfg.enabled:
            return synthetic_features, synthetic_labels

        prob = float(cfg.probability)
        apply_mask = torch.rand(batch_size, device=device) < prob

        if not apply_mask.any():
            return synthetic_features, synthetic_labels

        selected_idx = torch.where(apply_mask)[0]
        synthetic_labels[selected_idx] = 1.0

        if cfg.mode == "gaussian":
            noise = torch.randn_like(synthetic_features[selected_idx]) * float(cfg.gaussian_std)
            synthetic_features[selected_idx] = synthetic_features[selected_idx] + noise

        elif cfg.mode == "mix":
            shuffled_idx = selected_idx[torch.randperm(len(selected_idx), device=device)]
            alpha = float(cfg.mix_alpha)
            synthetic_features[selected_idx] = (
                alpha * synthetic_features[selected_idx]
                + (1.0 - alpha) * synthetic_features[shuffled_idx]
            )

        else:
            raise ValueError(f"Unsupported feature_synthetic_anomalies mode: {cfg.mode}")

        return synthetic_features, synthetic_labels

    def compute_roc_curve_points(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        order = np.argsort(-y_score)
        y_true_sorted = y_true[order]
        y_score_sorted = y_score[order]

        positives = max(int((y_true == 1).sum()), 1)
        negatives = max(int((y_true == 0).sum()), 1)

        tps = np.cumsum(y_true_sorted == 1)
        fps = np.cumsum(y_true_sorted == 0)

        tpr = tps / positives
        fpr = fps / negatives
        thresholds = y_score_sorted
        return fpr, tpr, thresholds

    def compute_fpr_at_target_tpr(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        target_tpr: float = 0.95,
    ) -> float:
        if len(np.unique(y_true)) < 2:
            return float("nan")

        fpr, tpr, _ = self.compute_roc_curve_points(y_true, y_score)

        valid_idx = np.where(tpr >= target_tpr)[0]
        if len(valid_idx) == 0:
            return 1.0

        return float(fpr[valid_idx[0]])

    def save_score_histograms(
        self,
        split_name: str,
        labels: list[int],
        score_components: dict[str, list[float]],
        epoch: int | None = None,
    ) -> None:
        if not self.cfg.logging.save_score_histograms:
            return

        labels_arr = np.asarray(labels, dtype=int)
        hist_dir = self.run_dir / "score_histograms"
        if epoch is not None:
            hist_dir = hist_dir / f"epoch_{epoch:03d}"
        hist_dir.mkdir(parents=True, exist_ok=True)

        for score_name, values in score_components.items():
            values_arr = np.asarray(values, dtype=float)
            normal_scores = values_arr[labels_arr == 0]
            anomaly_scores = values_arr[labels_arr == 1]

            save_score_histogram(
                normal_scores=normal_scores,
                anomaly_scores=anomaly_scores,
                output_path=hist_dir / f"{split_name}_{score_name}.png",
                title=f"{split_name} - {score_name}",
                bins=int(self.cfg.logging.histogram_bins),
            )

    def build_discriminative_input(
        self,
        image: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> torch.Tensor:
        diff = torch.abs(image - reconstruction)
        return torch.cat([image, reconstruction, diff], dim=1)

    def compute_score_components(
        self,
        batch: dict[str, Any],
        outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        image = batch["image"].to(self.device)

        for key, value in outputs.items():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                print(f"[non_finite_outputs] key={key} shape={tuple(value.shape)}")

        safe_outputs = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                safe_outputs[key] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                safe_outputs[key] = value

        outputs = safe_outputs
        reconstruction = outputs["reconstruction"]

        raw_scores = {
            "memory_score": self.compute_memory_score(outputs),
            "recon_score": self.compute_reconstruction_score(batch, outputs),
            "perceptual_score": self.compute_perceptual_score(outputs),
            "feature_score": self.compute_feature_discrepancy_score(outputs),
            "latent_score": self.compute_latent_one_class_score(outputs),
            "discriminative_score": self.compute_discriminative_score(image, reconstruction),
            "teacher_student_score": self.compute_teacher_student_score(batch, outputs),
        }

        sanitized_scores = {}
        for score_name, score_value in raw_scores.items():
            self._log_non_finite_score(score_name, score_value, batch=batch)

            finite_mask = self._tensor_finite_mask(score_value)
            if not bool(finite_mask.all()):
                score_value = self._disable_score_component(score_name, score_value)

            score_value = self._sanitize_torch_score(score_value)
            sanitized_scores[score_name] = score_value

        memory_score = sanitized_scores["memory_score"]
        recon_score = sanitized_scores["recon_score"]
        perceptual_score = sanitized_scores["perceptual_score"]
        feature_score = sanitized_scores["feature_score"]
        latent_score = sanitized_scores["latent_score"]
        discriminative_score = sanitized_scores["discriminative_score"]
        teacher_student_score = sanitized_scores["teacher_student_score"]

        norm_recon_score = self.normalize_score_tensor(recon_score, "recon_score")
        norm_perceptual_score = self.normalize_score_tensor(perceptual_score, "perceptual_score")
        norm_feature_score = self.normalize_score_tensor(feature_score, "feature_score")
        norm_latent_score = self.normalize_score_tensor(latent_score, "latent_score")
        norm_memory_score = self.normalize_score_tensor(memory_score, "memory_score")
        norm_discriminative_score = self.normalize_score_tensor(discriminative_score, "discriminative_score")
        norm_teacher_student_score = self.normalize_score_tensor(teacher_student_score, "teacher_student_score")

        final_score = self.compute_final_score(
            recon_score=norm_recon_score,
            perceptual_score=norm_perceptual_score,
            feature_score=norm_feature_score,
            latent_score=norm_latent_score,
            memory_score=norm_memory_score,
            discriminative_score=norm_discriminative_score,
            teacher_student_score=norm_teacher_student_score,
        )

        components = {
            "final_score": final_score,
            "recon_score": recon_score,
            "perceptual_score": perceptual_score,
            "feature_score": feature_score,
            "latent_score": latent_score,
            "memory_score": memory_score,
            "discriminative_score": discriminative_score,
            "teacher_student_score": teacher_student_score,
            "norm_recon_score": norm_recon_score,
            "norm_perceptual_score": norm_perceptual_score,
            "norm_feature_score": norm_feature_score,
            "norm_latent_score": norm_latent_score,
            "norm_memory_score": norm_memory_score,
            "norm_discriminative_score": norm_discriminative_score,
            "norm_teacher_student_score": norm_teacher_student_score,
        }

        if self.cfg.score_fusion_learned.enabled and self.learned_score_fusion is not None:
            components["linear_final_score"] = components["final_score"]
            components["final_score"] = self.apply_learned_score_fusion(components)

        return components

    def forward_discriminative(
        self,
        image: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.discriminative_head is None:
            b, _, h, w = image.shape
            zero_map_logits = torch.zeros((b, 1, h, w), device=image.device, dtype=image.dtype)
            zero_score_logits = torch.zeros((b,), device=image.device, dtype=image.dtype)
            return zero_map_logits, zero_score_logits

        disc_input = self.build_discriminative_input(image, reconstruction)
        logits_map = self.discriminative_head(disc_input)              # logits, non sigmoid
        image_logits = torch.mean(logits_map, dim=(1, 2, 3))          # logits immagine
        return logits_map, image_logits

    def compute_discriminative_score(
        self,
        image: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> torch.Tensor:
        logits_map, image_logits = self.forward_discriminative(image, reconstruction)
        return torch.sigmoid(image_logits)

    def compute_discriminative_losses(
        self,
        synthetic_image: torch.Tensor,
        synthetic_mask: torch.Tensor,
        synthetic_label: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            synth_outputs = self.model(synthetic_image)

        reconstruction = synth_outputs["reconstruction"]
        pred_map_logits, pred_image_logits = self.forward_discriminative(
            synthetic_image,
            reconstruction,
        )

        target_mask = synthetic_mask.float()
        target_label = synthetic_label.float()

        map_loss = F.binary_cross_entropy_with_logits(pred_map_logits, target_mask)
        image_loss = F.binary_cross_entropy_with_logits(pred_image_logits, target_label)

        total_disc_loss = (
            float(self.cfg.discriminative.map_loss_weight) * map_loss
            + float(self.cfg.discriminative.image_loss_weight) * image_loss
        )

        pred_map_prob = torch.sigmoid(pred_map_logits)
        return total_disc_loss, map_loss, image_loss, pred_map_prob

    def inspect_train_synthetic_samples(self) -> None:
        batch = next(iter(self.train_loader))
        synth_count = int(batch["synthetic_label"].sum().item())
        print(
            f"[train_loader] synthetic_batch_shape={tuple(batch['synthetic_image'].shape)} "
            f"synthetic_count={synth_count}"
        )

    def run_latent_mining(self, outputs: dict[str, torch.Tensor], epoch: int) -> dict[str, torch.Tensor] | None:
        if not self.cfg.mining_runtime.enabled:
            return None

        if epoch <= int(self.cfg.mining_runtime.warmup_epochs):
            return None

        latent = outputs["latent"]
        center = self.latent_center.center.detach()

        z_start, z_hard, score_start, score_final = pgd_mine_latent(
            latent=latent,
            center=center,
            steps=int(self.cfg.mining_runtime.steps),
            step_size=float(self.cfg.mining_runtime.step_size),
            noise_std=float(self.cfg.mining_runtime.noise_std),
            clamp_value=float(self.cfg.mining_runtime.clamp_value),
        )

        target_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        hard_reconstruction = target_model.reconstructor(z_hard)

        return {
            "z_start": z_start.detach(),
            "z_hard": z_hard.detach(),
            "score_start": score_start.detach(),
            "score_final": score_final.detach(),
            "hard_reconstruction": hard_reconstruction.detach(),
        }

    def _scoring_topk(self) -> int:
        return int(getattr(self.cfg, "scoring_topk", 0) or 0)

    @staticmethod
    def _aggregate_err(err: torch.Tensor, topk: int) -> torch.Tensor:
        # err: [B, ...] per-sample error map. If topk>0, average top-k pixel errors
        # per sample (PatchCore-style patch pooling); otherwise global mean.
        if topk and topk > 0:
            flat = err.flatten(1)
            k = min(topk, flat.shape[1])
            return flat.topk(k, dim=1).values.mean(dim=1)
        return err.mean(dim=tuple(range(1, err.ndim)))

    def compute_reconstruction_score(self, batch: dict[str, Any], outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        target = batch["image"].to(self.device)
        reconstruction = outputs["reconstruction"]
        err = (reconstruction - target) ** 2  # [B, C, H, W]
        # Per-pixel error across channels, then top-k pool over spatial pixels.
        err_map = err.mean(dim=1)  # [B, H, W]
        return self._aggregate_err(err_map, self._scoring_topk())

    def compute_feature_discrepancy_score(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.mean((outputs["global"] - outputs["recon_global"]) ** 2, dim=1)

    def compute_perceptual_score(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        score = torch.zeros(outputs["global"].shape[0], device=self.device)
        topk = self._scoring_topk()

        for level in self.cfg.losses.perceptual.levels:
            orig = torch.nan_to_num(outputs[level], nan=0.0, posinf=0.0, neginf=0.0)
            recon = torch.nan_to_num(outputs[f"recon_{level}"], nan=0.0, posinf=0.0, neginf=0.0)

            if orig.ndim >= 3:
                # Feature map [B, C, H, W]: mean over channels, then top-k over spatial.
                diff_map = torch.abs(orig - recon).mean(dim=1)  # [B, H, W]
                diff = self._aggregate_err(diff_map, topk)
            else:
                # Global feature vector [B, C]: only a mean makes sense.
                diff = torch.mean(torch.abs(orig - recon), dim=tuple(range(1, orig.ndim)))

            if not torch.isfinite(diff).all():
                print(f"[perceptual][debug] non-finite diff at level={level}")
                diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)

            score = score + diff

        return score

    def compute_final_score(
        self,
        recon_score: torch.Tensor,
        perceptual_score: torch.Tensor,
        feature_score: torch.Tensor,
        latent_score: torch.Tensor,
        memory_score: torch.Tensor,
        discriminative_score: torch.Tensor,
        teacher_student_score: torch.Tensor,
    ) -> torch.Tensor:
        if not self.cfg.score_fusion.enabled:
            return (
                recon_score
                + perceptual_score
                + feature_score
                + latent_score
                + memory_score
                + discriminative_score
                + teacher_student_score
            )

        return (
            float(self.cfg.score_fusion.recon_weight) * recon_score
            + float(self.cfg.score_fusion.perceptual_weight) * perceptual_score
            + float(self.cfg.score_fusion.feature_weight) * feature_score
            + float(self.cfg.score_fusion.latent_weight) * latent_score
            + float(self.cfg.score_fusion.memory_weight) * memory_score
            + float(self.cfg.score_fusion.discriminative_weight) * discriminative_score
            + float(self.cfg.score_fusion.teacher_student_weight) * teacher_student_score
        )

    def normalize_score_tensor(self, score: torch.Tensor, score_name: str) -> torch.Tensor:
        score = self._sanitize_torch_score(score)

        if not self.cfg.score_normalization.enabled:
            return score

        if score_name not in self.score_norm_stats:
            return score

        stats = self.score_norm_stats[score_name]

        median_value = stats.get("median", 0.0)
        scale_value = stats.get("scale", 1.0)

        if not np.isfinite(median_value):
            median_value = 0.0
        if not np.isfinite(scale_value) or scale_value <= 1e-12:
            scale_value = 1.0

        median = torch.tensor(median_value, device=score.device, dtype=score.dtype)
        scale = torch.tensor(scale_value, device=score.device, dtype=score.dtype)

        normalized = (score - median) / scale
        normalized = self._sanitize_torch_score(normalized)
        return normalized

    @torch.no_grad()
    def collect_score_components_on_loader(self, loader) -> dict[str, list[float]]:
        self.model.eval()

        collected = {
            "recon_score": [],
            "perceptual_score": [],
            "feature_score": [],
            "latent_score": [],
            "memory_score": [],
            "discriminative_score": [],
            "teacher_student_score": [],
        }

        for batch in loader:
            images = batch["image"].to(self.device)

            with torch.cuda.amp.autocast(
                enabled=bool(self.cfg.runtime.amp and self.device == "cuda"),
            ):
                outputs = self.model(images)

                image = batch["image"].to(self.device)
                reconstruction = outputs["reconstruction"]

                recon_score = self.compute_reconstruction_score(batch, outputs)
                perceptual_score = self.compute_perceptual_score(outputs)
                feature_score = self.compute_feature_discrepancy_score(outputs)
                latent_score = self.compute_latent_one_class_score(outputs)
                memory_score = self.compute_memory_score(outputs)
                discriminative_score = self.compute_discriminative_score(image, reconstruction)
                teacher_student_score = self.compute_teacher_student_score(batch, outputs)
            
            debug_scores = {
                "recon_score": recon_score,
                "perceptual_score": perceptual_score,
                "feature_score": feature_score,
                "latent_score": latent_score,
                "memory_score": memory_score,
                "discriminative_score": discriminative_score,
                "teacher_student_score": teacher_student_score,
            }

            for score_name, score_tensor in debug_scores.items():
                self._log_non_finite_score(score_name, score_tensor, batch=batch)
                    
            collected["recon_score"].extend(recon_score.detach().cpu().numpy().tolist())
            collected["perceptual_score"].extend(perceptual_score.detach().cpu().numpy().tolist())
            collected["feature_score"].extend(feature_score.detach().cpu().numpy().tolist())
            collected["latent_score"].extend(latent_score.detach().cpu().numpy().tolist())
            collected["memory_score"].extend(memory_score.detach().cpu().numpy().tolist())
            collected["discriminative_score"].extend(discriminative_score.detach().cpu().numpy().tolist())
            collected["teacher_student_score"].extend(teacher_student_score.detach().cpu().numpy().tolist())

        return collected

    def fit_score_normalization(self) -> None:
        if not self.cfg.score_normalization.enabled:
            self.score_norm_stats = {}
            return

        reference_split = str(self.cfg.score_normalization.reference_split)
        method = str(self.cfg.score_normalization.method)
        eps = float(self.cfg.score_normalization.eps)

        if reference_split == "val_normal":
            loader = self.val_normal_loader
        elif reference_split == "train_normal":
            loader = self.train_loader
        else:
            raise ValueError(f"Unsupported score normalization reference split: {reference_split}")

        collected = self.collect_score_components_on_loader(loader)

        self.score_norm_stats = {}
        for score_name, values in collected.items():
            arr = np.asarray(values, dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            finite_ratio = float(np.isfinite(arr).mean()) if arr.size > 0 else 0.0

            if finite.size > 0:
                min_val = float(finite.min())
                max_val = float(finite.max())
                mean_val = float(finite.mean())
            else:
                min_val = float("nan")
                max_val = float("nan")
                mean_val = float("nan")

            print(
                f"[score_normalization][debug] {score_name}: "
                f"count={arr.size} finite_ratio={finite_ratio:.4f} "
                f"min={min_val:.6f} max={max_val:.6f} mean={mean_val:.6f}"
            )

            self.score_norm_stats[score_name] = self.compute_robust_stats(
                arr,
                method=method,
                eps=eps,
            )

        print(f"[score_normalization] fitted on {reference_split} with method={method}")
        for score_name, stats in self.score_norm_stats.items():
            print(
                f"[score_normalization] {score_name}: "
                f"median={stats['median']:.6f} scale={stats['scale']:.6f}"
            )

    def compute_robust_stats(self, values: np.ndarray, method: str = "mad", eps: float = 1e-6) -> dict[str, float]:
        values = self._sanitize_numpy_1d(values)

        if values.size == 0:
            return {
                "median": 0.0,
                "scale": 1.0,
            }

        median = float(np.median(values))

        if not np.isfinite(median):
            median = 0.0

        if method == "mad":
            abs_dev = np.abs(values - median)
            abs_dev = abs_dev[np.isfinite(abs_dev)]
            if abs_dev.size == 0:
                scale = 1.0
            else:
                scale = float(np.median(abs_dev))
        elif method == "iqr":
            q25 = float(np.percentile(values, 25))
            q75 = float(np.percentile(values, 75))
            scale = q75 - q25
        else:
            raise ValueError(f"Unsupported score normalization method: {method}")

        if not np.isfinite(scale) or scale < eps:
            scale = 1.0

        return {
            "median": float(median),
            "scale": float(scale),
        }

    def compute_latent_compactness_loss(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.cfg.latent_compactness.enabled:
            return torch.tensor(0.0, device=self.device)
        return self.latent_center.loss(outputs["latent"])

    def compute_latent_one_class_score(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.latent_center.score(outputs["latent"])

    def compute_latent_one_class_loss(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.cfg.one_class.enabled:
            return torch.tensor(0.0, device=self.device)
        return torch.mean(self.compute_latent_one_class_score(outputs))

    def compute_perceptual_loss(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.cfg.losses.perceptual.enabled:
            return torch.tensor(0.0, device=self.device)

        losses = []
        for level in self.cfg.losses.perceptual.levels:
            orig = torch.nan_to_num(outputs[level], nan=0.0, posinf=0.0, neginf=0.0)
            recon = torch.nan_to_num(outputs[f"recon_{level}"], nan=0.0, posinf=0.0, neginf=0.0)
            losses.append(torch.mean(torch.abs(orig - recon)))

        return sum(losses) / max(len(losses), 1)

    def _cfg_has(self, key: str) -> bool:
        try:
            return key in self.cfg
        except Exception:
            return hasattr(self.cfg, key)

    def _build_scheduler(self) -> None:
        self.scheduler = None

        if not self._cfg_has("scheduler"):
            print("Scheduler: none (no scheduler section in cfg)")
            return

        if not bool(self.cfg.scheduler.enabled):
            print("Scheduler: none (disabled)")
            return

        sched_name = str(self.cfg.scheduler.name).lower()
        if sched_name != "cosine":
            raise ValueError(f"Unsupported scheduler: {sched_name}")

        total_epochs = int(self.cfg.training.epochs)
        warmup_epochs = int(self.cfg.scheduler.warmup_epochs)
        warmup_start_factor = float(self.cfg.scheduler.warmup_start_factor)
        min_lr = float(self.cfg.scheduler.min_lr)

        if warmup_epochs > 0:
            warmup = LinearLR(
                self.optimizer,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                self.optimizer,
                T_max=max(total_epochs - warmup_epochs, 1),
                eta_min=min_lr,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(total_epochs, 1),
                eta_min=min_lr,
            )

        print(
            f"Scheduler: {sched_name} | warmup_epochs={warmup_epochs} "
            f"| warmup_start_factor={warmup_start_factor} | min_lr={min_lr}"
        )

    def __post_init__(self) -> None:
        self.discriminative_head: nn.Module | None = None
        self.device = self.cfg.project.device
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.cfg, "early_stopping") and self.cfg.early_stopping.mode == "min":
            self.early_stopping_best = float("inf")
        else:
            self.early_stopping_best = float("-inf")
        self.early_stopping_counter = 0
        self.logger = RunLogger(self.run_dir)
        self.best_val_threshold = None
        self.start_epoch = 0
        self.best_metric = float("-inf")
        self.loaders = {}
        self._loader = None
        self.val_normal_loader = None
        self.val_mixed_loader = None
        self.test_loader = None
        self.scheduler = None
        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scaler = None
        self.ema: EMA | None = None
        self.profiler = None
        self.fixed_batch = None

    def _build_model(self) -> nn.Module:
        model = ReconstructionModel(self.cfg)

        if (
            self.cfg.distributed.enabled
            and self.cfg.distributed.strategy == "dp"
            and self.device == "cuda"
            and torch.cuda.device_count() > 1
        ):
            print(f"Wrapping model with DataParallel on {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)

        model = model.to(self.device)
        return model
    
    def is_early_stopping_improvement(self, value: float) -> bool:
        mode = str(self.cfg.early_stopping.mode)
        min_delta = float(self.cfg.early_stopping.min_delta)

        if self.early_stopping_best is None:
            return True

        if mode == "max":
            return value >= (self.early_stopping_best + min_delta)
        elif mode == "min":
            return value <= (self.early_stopping_best - min_delta)
        else:
            raise ValueError(f"Unsupported early stopping mode: {mode}")


    def update_early_stopping(self, value: float) -> bool:
        if not self.cfg.early_stopping.enabled:
            return False

        if self.is_early_stopping_improvement(value):
            self.early_stopping_best = float(value)
            self.early_stopping_counter = 0
            print(
                f"[EarlyStopping] improvement detected: "
                f"best={self.early_stopping_best:.4f}"
            )
            return False

        self.early_stopping_counter += 1
        patience = int(self.cfg.early_stopping.patience)

        print(
            f"[EarlyStopping] no improvement: "
            f"counter={self.early_stopping_counter}/{patience} "
            f"best={self.early_stopping_best:.4f} current={value:.4f}"
        )

        if self.early_stopping_counter >= patience:
            print(f"[EarlyStopping] stopping training.")
            return True

        return False

    @torch.no_grad()
    def extract_teacher_features(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.teacher_backbone is None:
            raise RuntimeError("teacher_backbone is not initialized")

        self.teacher_backbone.eval()
        return self.teacher_backbone(image)

    def compute_teacher_student_score(
        self,
        batch: dict[str, Any],
        outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not self.cfg.teacher_student.enabled or self.teacher_backbone is None:
            batch_size = outputs["global"].shape[0]
            return torch.zeros(batch_size, device=self.device)

        image = batch["image"].to(self.device)
        reconstruction = outputs["reconstruction"]

        teacher_real = self.extract_teacher_features(image)
        teacher_recon = self.extract_teacher_features(reconstruction)

        per_level_scores = []
        for level in self.cfg.teacher_student.levels:
            real_feat = torch.nan_to_num(teacher_real[level], nan=0.0, posinf=0.0, neginf=0.0)
            recon_feat = torch.nan_to_num(teacher_recon[level], nan=0.0, posinf=0.0, neginf=0.0)
            diff = torch.abs(real_feat - recon_feat)

            if diff.ndim == 4:
                diff = diff.reshape(diff.shape[0], -1).mean(dim=1)
            elif diff.ndim == 2:
                diff = diff.mean(dim=1)
            else:
                raise ValueError(f"Unsupported feature shape for level {level}: {diff.shape}")

            per_level_scores.append(diff)
        
        result = torch.stack(per_level_scores, dim=0).mean(dim=0)
        if not torch.isfinite(result).all():
            print("[teacher_student][debug] non-finite final score")
        result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        
        return result

    def compute_teacher_student_loss(
        self,
        batch: dict[str, Any],
        outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not self.cfg.teacher_student.enabled or self.teacher_backbone is None:
            return torch.tensor(0.0, device=self.device)

        image = batch["image"].to(self.device)
        reconstruction = outputs["reconstruction"]

        teacher_real = self.extract_teacher_features(image)
        teacher_recon = self.extract_teacher_features(reconstruction)

        losses = []
        for level in self.cfg.teacher_student.levels:
            real_feat = torch.nan_to_num(teacher_real[level], nan=0.0, posinf=0.0, neginf=0.0)
            recon_feat = torch.nan_to_num(teacher_recon[level], nan=0.0, posinf=0.0, neginf=0.0)
            losses.append(torch.mean(torch.abs(real_feat - recon_feat)))

        return torch.stack(losses).mean()

    def compute_feature_discrepancy_loss(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.cfg.feature_discrepancy.enabled:
            return torch.tensor(0.0, device=self.device)

        losses = []
        levels = list(self.cfg.feature_discrepancy.levels)

        for level in levels:
            if level == "global":
                orig = outputs["global"]
                recon = outputs["recon_global"]
                losses.append(torch.mean((orig - recon) ** 2))
            else:
                orig = outputs[level]
                recon = outputs[f"recon_{level}"]
                losses.append(torch.mean((orig - recon) ** 2))

        return sum(losses) / max(len(losses), 1)

    def setup(self) -> None:
        print("=== Trainer setup ===")
        print(f"Project: {self.cfg.project.name}")
        print(f"Run dir: {self.run_dir}")
        print(f"Device: {self.device}")
        print(f"Visible GPU count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"Visible GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
        print(f"Dataset: {self.cfg.dataset.name}")
        print(f"Backbone: {self.cfg.model.backbone.name}")
        print(f"Pretrained: {self.cfg.model.backbone.pretrained}")
        print(f"Frozen backbone: {self.cfg.model.backbone.frozen}")
        print(f"Reconstruction enabled: {self.cfg.model.reconstruction.enabled}")
        print(f"Latent enabled: {self.cfg.model.latent.enabled}")
        print(f"Head enabled: {self.cfg.model.head.enabled}")
        print(f"Mining enabled: {self.cfg.model.mining.enabled}")

        self.learned_score_fusion = None
        self.learned_score_fusion_feature_names = []
        self.score_norm_stats = {}
        self.teacher_backbone = None
        self.memory_bank = None
        self.loaders = build_all_dataloaders(self.cfg)
        self.train_loader = self.loaders["train_normal"]
        self.val_normal_loader = self.loaders["val_normal"]
        self.val_mixed_loader = self.loaders["val_mixed"]
        self.test_loader = self.loaders["test_blind"]

        print(f"Train loader built with {len(self.train_loader)} batches")
        print(f"Val-normal loader built with {len(self.val_normal_loader)} batches")
        print(f"Val-mixed loader built with {len(self.val_mixed_loader)} batches")
        print(f"Test-blind loader built with {len(self.test_loader)} batches")
        self.model = self._build_model()
        self.build_memory_bank()

        if self.cfg.teacher_student.enabled:
            self.teacher_backbone = build_backbone(self.cfg).to(self.device)
            self.teacher_backbone.eval()
            for p in self.teacher_backbone.parameters():
                p.requires_grad = False

        if self.cfg.feature_discriminative.enabled:
            target_level = str(self.cfg.feature_synthetic_anomalies.target_level)
            if target_level == "global":
                in_dim = 2048
            elif target_level == "layer4":
                in_dim = 2048
            elif target_level == "layer3":
                in_dim = 1024
            elif target_level == "layer2":
                in_dim = 512
            else:
                raise ValueError(f"Unsupported feature level for feature discriminator: {target_level}")

            self.feature_discriminator = self.build_feature_discriminator(in_dim).to(self.device)
        else:
            self.feature_discriminator = None

        if self.cfg.discriminative.enabled:
            self.discriminative_head = SimpleDiscriminativeHead(in_channels=9).to(self.device)
        else:
            self.discriminative_head = None

        if self._cfg_has("optimizer"):
            lr = float(self.cfg.optimizer.lr)
            weight_decay = float(self.cfg.optimizer.weight_decay)
        else:
            lr = float(self.cfg.training.lr)
            weight_decay = float(self.cfg.training.weight_decay)

        # Separate backbone (partially unfrozen) params from the rest so they
        # can use a reduced learning rate. Identify by id membership to avoid
        # double-counting when the reconstruction model also references them.
        backbone = getattr(self.model, "backbone", None)
        backbone_param_ids: set[int] = set()
        backbone_params: list = []
        if backbone is not None and hasattr(backbone, "trainable_parameters"):
            for p in backbone.trainable_parameters():
                if p.requires_grad and id(p) not in backbone_param_ids:
                    backbone_param_ids.add(id(p))
                    backbone_params.append(p)

        other_params = [
            p for p in self.model.parameters()
            if p.requires_grad and id(p) not in backbone_param_ids
        ]

        if self.discriminative_head is not None:
            other_params += [p for p in self.discriminative_head.parameters() if p.requires_grad]

        if self.feature_discriminator is not None:
            other_params += [p for p in self.feature_discriminator.parameters() if p.requires_grad]

        unfreeze_lr_factor = float(getattr(self.cfg.model.backbone, "unfreeze_lr_factor", 0.1))
        param_groups = [{"params": other_params, "lr": lr}]
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": lr * unfreeze_lr_factor,
            })

        self.optimizer = AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
        )

        if backbone_params:
            print(
                f"Optimizer: AdamW | lr={lr} | weight_decay={weight_decay} | "
                f"backbone_unfrozen_params={len(backbone_params)} "
                f"@ lr={lr * unfreeze_lr_factor:.2e}"
            )
        else:
            print(f"Optimizer: AdamW | lr={lr} | weight_decay={weight_decay}")

        self._build_scheduler()

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=bool(self.cfg.runtime.amp and self.device == "cuda"),
        )

        if self.cfg.ema.enabled:
            self.ema = EMA(self.model, decay=float(self.cfg.ema.decay))
            print(f"EMA: enabled | decay={float(self.cfg.ema.decay)}")
        else:
            print("EMA: disabled")

        self.profiler = build_profiler(self.cfg, self.run_dir)
        if self.profiler is not None:
            print(f"Profiler enabled. Traces will be saved to: {self.run_dir / 'profiler'}")
        self.inspect_eval_loaders()
        self.run_anti_leakage_audit()
        self.reconstruction_loss_fn = build_reconstruction_loss(self.cfg)
        self.latent_center = LatentCenter(
            dim=int(self.cfg.model.latent.dim),
            device=self.device,
            momentum=float(self.cfg.latent_compactness.momentum),
            normalize_latent=bool(self.cfg.latent_compactness.normalize_latent),
        )
        self.inspect_train_synthetic_samples()

    def log_epoch_metrics(self, epoch: int, metrics: dict[str, Any]) -> None:
        payload = {"epoch": epoch, **metrics}
        self.logger.log_metrics(payload)
        self.logger.save_epoch_summary(payload)

    def get_checkpoint_state(self, epoch: int, metric_value: float) -> dict:
        model_state = (
            self.model.module.state_dict()
            if isinstance(self.model, nn.DataParallel)
            else self.model.state_dict()
        )

        state = {
            "epoch": int(epoch),
            "best_metric": float(self.best_metric),
            "metric_value": float(metric_value),
            "monitor_name": str(self.cfg.checkpoint.monitor),
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
            "score_norm_stats": self.score_norm_stats,
            "best_val_threshold": self.best_val_threshold,
            "learned_score_fusion_feature_names": self.learned_score_fusion_feature_names,
            "latent_center_center": self.latent_center.center.detach().cpu()
            if hasattr(self, "latent_center") and self.latent_center is not None
            else None,
            "discriminative_head_state_dict": (
                self.discriminative_head.state_dict()
                if self.discriminative_head is not None
                else None
            ),
            "feature_discriminator_state_dict": (
                self.feature_discriminator.state_dict()
                if self.feature_discriminator is not None
                else None
            ),
        }

        if self.learned_score_fusion is not None:
            state["learned_score_fusion"] = self.learned_score_fusion
        else:
            state["learned_score_fusion"] = None

        return state

    def save_last_checkpoint(self, epoch: int, metric_value: float) -> None:
        if not self.cfg.checkpoint.save_last:
            return
        ckpt_path = self.run_dir / "last_checkpoint.pt"
        torch.save(self.get_checkpoint_state(epoch, metric_value), ckpt_path)

    def save_best_checkpoint(self, epoch: int, metric_value: float, monitor_value: float | None = None) -> None:
        if not self.cfg.checkpoint.save_best:
            return

        if monitor_value is None:
            monitor_value = metric_value

        mode = self.cfg.checkpoint.mode
        is_better = (
            monitor_value > self.best_metric if mode == "max" else monitor_value < self.best_metric
        )

        if is_better:
            self.best_metric = float(monitor_value)
            ckpt_path = self.run_dir / "best_checkpoint.pt"
            torch.save(self.get_checkpoint_state(epoch, metric_value), ckpt_path)
            print(
                f"New best checkpoint saved at epoch {epoch} "
                f"with monitor={monitor_value:.4f}"
            )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.start_epoch = int(ckpt["epoch"])
        self.best_metric = float(ckpt.get("best_metric", float("-inf")))

        target_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        if "model_state_dict" in ckpt:
            target_model.load_state_dict(ckpt["model_state_dict"])

        if (
            self.discriminative_head is not None
            and ckpt.get("discriminative_head_state_dict") is not None
        ):
            self.discriminative_head.load_state_dict(ckpt["discriminative_head_state_dict"])

        if (
            self.feature_discriminator is not None
            and ckpt.get("feature_discriminator_state_dict") is not None
        ):
            self.feature_discriminator.load_state_dict(ckpt["feature_discriminator_state_dict"])

        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if self.scaler is not None and ckpt.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        self.score_norm_stats = ckpt.get("score_norm_stats", {})
        self.best_val_threshold = ckpt.get("best_val_threshold", None)
        self.learned_score_fusion_feature_names = ckpt.get("learned_score_fusion_feature_names", [])
        self.learned_score_fusion = ckpt.get("learned_score_fusion", None)

        if (
            ckpt.get("latent_center_center") is not None
            and hasattr(self, "latent_center")
            and self.latent_center is not None
        ):
            self.latent_center.center.data.copy_(ckpt["latent_center_center"].to(self.device))

        print(f"Resumed from checkpoint: {checkpoint_path}")
        print(f"Start epoch set to: {self.start_epoch}")
        print(f"Best metric restored to: {self.best_metric:.4f}")


    def maybe_resume(self) -> None:
        if not self.cfg.training.resume:
            return

        checkpoint_path = self.cfg.training.resume_path
        if checkpoint_path is None:
            checkpoint_path = str(self.run_dir / "last_checkpoint.pt")

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")

        self.load_checkpoint(str(checkpoint_path))

    def compute_loss(self, batch: dict[str, Any], outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        target = batch["image"].to(self.device)
        reconstruction = outputs["reconstruction"]
        reconstruction = torch.nan_to_num(reconstruction, nan=0.0, posinf=0.0, neginf=0.0)
        outputs["reconstruction"] = reconstruction

        recon_total, _ = self.reconstruction_loss_fn(reconstruction, target)
        latent_reg = torch.mean(outputs["latent"] ** 2)
        feat_disc_loss = self.compute_feature_discrepancy_loss(outputs)
        perceptual_loss = self.compute_perceptual_loss(outputs)
        latent_compactness_loss = self.compute_latent_compactness_loss(outputs)
        one_class_loss = self.compute_latent_one_class_loss(outputs)

        total_loss = (
            recon_total
            + float(self.cfg.losses.latent_regularization.weight) * latent_reg
            + float(self.cfg.feature_discrepancy.weight) * feat_disc_loss
            + float(self.cfg.losses.perceptual.weight) * perceptual_loss
            + float(self.cfg.latent_compactness.weight) * latent_compactness_loss
            + float(self.cfg.one_class.loss_weight) * one_class_loss
        )
        return total_loss

    def compute_image_scores(self, batch: dict[str, Any], outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        components = self.compute_score_components(batch, outputs)
        return components["final_score"]
    
    def _save_single_image(self, tensor: torch.Tensor, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        x = tensor.detach().cpu()
        if x.ndim == 4:
            x = x[0]

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if x.ndim == 2:
            x = x.unsqueeze(0)

        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)

        x_min = float(x.min())
        x_max = float(x.max())

        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)
        else:
            x = torch.zeros_like(x)

        save_image(x, str(path))


    @torch.no_grad()
    def dump_failure_cases(
        self,
        loader,
        split_name: str,
        threshold: float,
        max_per_group: int = 8,
    ) -> None:
        self.model.eval()

        rows = []

        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(self.device)

            with torch.cuda.amp.autocast(
                enabled=bool(self.cfg.runtime.amp and self.device == "cuda"),
            ):
                outputs = self.model(images)
                components = self.compute_score_components(batch, outputs)

            reconstruction = outputs["reconstruction"]
            abs_diff = torch.mean(torch.abs(images - reconstruction), dim=1, keepdim=True)

            final_score = components["final_score"]
            pred = (final_score >= threshold).long()
            label = batch["label"].to(self.device).long()

            for i in range(images.shape[0]):
                item = {
                    "batch_idx": int(batch_idx),
                    "sample_idx": int(i),
                    "label": int(label[i].item()),
                    "pred": int(pred[i].item()),
                    "final_score": float(components["final_score"][i].detach().cpu().item()),
                    "recon_score": float(components["recon_score"][i].detach().cpu().item()),
                    "perceptual_score": float(components["perceptual_score"][i].detach().cpu().item()),
                    "feature_score": float(components["feature_score"][i].detach().cpu().item()),
                    "latent_score": float(components["latent_score"][i].detach().cpu().item()),
                    "memory_score": float(components["memory_score"][i].detach().cpu().item()),
                    "discriminative_score": float(components["discriminative_score"][i].detach().cpu().item()),
                    "teacher_student_score": float(components["teacher_student_score"][i].detach().cpu().item()),
                    "input": images[i].detach().cpu(),
                    "recon": reconstruction[i].detach().cpu(),
                    "abs_diff": abs_diff[i].detach().cpu(),
                }

                if "linear_final_score" in components:
                    item["linear_final_score"] = float(
                        components["linear_final_score"][i].detach().cpu().item()
                    )

                if item["label"] == 1 and item["pred"] == 0:
                    item["group"] = "false_negative"
                    item["rank_score"] = item["final_score"]          # più basso = peggiore FN
                elif item["label"] == 0 and item["pred"] == 1:
                    item["group"] = "false_positive"
                    item["rank_score"] = -item["final_score"]         # più alto = peggiore FP
                elif item["label"] == 1 and item["pred"] == 1:
                    item["group"] = "true_positive"
                    item["rank_score"] = -item["final_score"]         # più alto = TP forte
                else:
                    item["group"] = "true_negative"
                    item["rank_score"] = item["final_score"]          # più basso = TN sicuro

                rows.append(item)

        dump_dir = self.run_dir / "failure_analysis" / split_name
        dump_dir.mkdir(parents=True, exist_ok=True)

        for group_name in ["false_negative", "false_positive", "true_positive", "true_negative"]:
            group_items = [r for r in rows if r["group"] == group_name]
            group_items = sorted(group_items, key=lambda x: x["rank_score"])[:max_per_group]

            group_dir = dump_dir / group_name
            group_dir.mkdir(parents=True, exist_ok=True)

            summary = []

            for k, item in enumerate(group_items):
                case_id = f"{k:02d}_b{item['batch_idx']:03d}_i{item['sample_idx']:02d}"

                self._save_single_image(item["input"], group_dir / f"{case_id}_input.png")
                self._save_single_image(item["recon"], group_dir / f"{case_id}_recon.png")
                self._save_single_image(item["abs_diff"], group_dir / f"{case_id}_absdiff.png")

                meta = {
                    "group": item["group"],
                    "label": item["label"],
                    "pred": item["pred"],
                    "final_score": item["final_score"],
                    "recon_score": item["recon_score"],
                    "perceptual_score": item["perceptual_score"],
                    "feature_score": item["feature_score"],
                    "latent_score": item["latent_score"],
                    "memory_score": item["memory_score"],
                    "discriminative_score": item["discriminative_score"],
                    "teacher_student_score": item["teacher_student_score"],
                }

                if "linear_final_score" in item:
                    meta["linear_final_score"] = item["linear_final_score"]

                with open(group_dir / f"{case_id}_scores.json", "w") as f:
                    json.dump(meta, f, indent=2)

                summary.append({"case_id": case_id, **meta})

            with open(group_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

        print(f"[failure_analysis] saved to {dump_dir}")
    
    @torch.no_grad()
    def evaluate_loader(self, loader, split_name: str, threshold: float | None = None) -> dict[str, float]:
        self.model.eval()

        component_scores = {
            "final_score": [],
            "recon_score": [],
            "perceptual_score": [],
            "feature_score": [],
            "latent_score": [],
            "memory_score": [],
            "discriminative_score": [],
            "teacher_student_score": [],
        }

        if bool(self.cfg.score_fusion_learned.enabled):
            component_scores["linear_final_score"] = []

        y_true = []
        y_score = []
        rows = []
        global_idx = 0

        for batch in loader:
            images = batch["image"].to(self.device)

            with torch.cuda.amp.autocast(
                enabled=bool(self.cfg.runtime.amp and self.device == "cuda"),
            ):
                outputs = self.model(images)
                components = self.compute_score_components(batch, outputs)
                scores = components["final_score"]

                batch_size = len(batch["label"])
                for i in range(batch_size):
                    row = {
                        "split": split_name,
                        "sample_idx_in_loader": global_idx,
                        "label": int(batch["label"][i].cpu().item()),
                        "final_score": float(components["final_score"][i].detach().cpu().item()),
                        "linear_final_score": float(components["linear_final_score"][i].detach().cpu().item())
                            if "linear_final_score" in components else None,
                        "recon_score": float(components["recon_score"][i].detach().cpu().item()),
                        "perceptual_score": float(components["perceptual_score"][i].detach().cpu().item()),
                        "feature_score": float(components["feature_score"][i].detach().cpu().item()),
                        "latent_score": float(components["latent_score"][i].detach().cpu().item()),
                        "memory_score": float(components["memory_score"][i].detach().cpu().item()),
                        "discriminative_score": float(components["discriminative_score"][i].detach().cpu().item()),
                        "teacher_student_score": float(components["teacher_student_score"][i].detach().cpu().item()),
                    }
                    rows.append(row)
                    global_idx += 1

                for key in component_scores:
                    if key not in components:
                        continue
                    component_scores[key].extend(
                        components[key].detach().cpu().numpy().tolist()
                    )

            y_true.extend(batch["label"].cpu().numpy().tolist())
            y_score.extend(scores.detach().cpu().numpy().tolist())

        y_true_np = np.asarray(y_true, dtype=int)
        y_score_np = np.asarray(y_score, dtype=float)

        fpr_at_95_tpr = self.compute_fpr_at_target_tpr(
            y_true=y_true_np,
            y_score=y_score_np,
            target_tpr=0.95,
        )

        metrics = compute_image_level_metrics(y_true_np, y_score_np)

        result = {
            f"{split_name}_auroc": metrics.auroc,
            f"{split_name}_auprc": metrics.auprc,
            f"{split_name}_best_f1": metrics.best_f1,
            f"{split_name}_best_threshold": metrics.best_threshold,
            f"{split_name}_fpr_at_95_tpr": fpr_at_95_tpr,
        }

        if threshold is not None:
            result[f"{split_name}_f1_at_given_threshold"] = compute_f1_at_threshold(
                y_true_np, y_score_np, threshold
            )

        if split_name == "val_mixed":
            self.save_score_histograms(
                split_name=split_name,
                labels=y_true,
                score_components=component_scores,
                epoch=getattr(self, "_current_epoch_for_eval", None),
            )

        if len(rows) > 0:
            csv_path = self.run_dir / f"{split_name}_component_scores.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        return result

    def train_one_epoch(self, epoch: int) -> float:
        first_batch_mining_outputs = None
        first_batch_images = None
        first_batch_synthetic_images = None
        first_batch_synthetic_masks = None
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        running_teacher_student_loss = 0.0
        running_teacher_student_score = 0.0
        running_feature_disc_loss = 0.0
        running_feature_disc_score = 0.0
        running_feature_synth_score = 0.0
        running_feature_synth_ratio = 0.0
        running_real_disc_score = 0.0
        running_disc_total_loss = 0.0
        running_disc_map_loss = 0.0
        running_disc_image_loss = 0.0
        running_disc_score = 0.0
        first_batch_disc_map = None
        running_mining_start_score = 0.0
        running_mining_final_score = 0.0
        num_mining_batches = 0
        running_recon_score = 0.0
        running_perceptual_score = 0.0
        running_feature_score = 0.0
        running_latent_compactness_loss = 0.0
        running_recon_loss = 0.0
        running_perceptual_loss = 0.0
        running_feat_disc_loss = 0.0
        running_one_class_loss = 0.0
        running_latent_score = 0.0
        running_loss = 0.0
        running_l1_loss = 0.0
        running_ms_ssim_loss = 0.0
        running_mse_loss = 0.0
        num_steps = 0
        num_samples = 0
        last_batch_shape = None
        first_batch_images = None

        epoch_start = time.perf_counter()

        if self.cfg.debug.overfit_one_batch and self.fixed_batch is None:
            self.fixed_batch = next(iter(self.train_loader))

        iterable = (
            [self.fixed_batch] * len(self.train_loader)
            if self.cfg.debug.overfit_one_batch
            else self.train_loader
        )

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        with profiler_context(self.profiler):
            for batch_idx, batch in enumerate(iterable):
                images = batch["image"].to(self.device)
                synthetic_image = batch["synthetic_image"].to(self.device)
                synthetic_mask = batch["synthetic_mask"].to(self.device)
                synthetic_label = batch["synthetic_label"].to(self.device)

                disc_total_loss = torch.tensor(0.0, device=self.device)
                disc_map_loss_value = torch.tensor(0.0, device=self.device)
                disc_image_loss_value = torch.tensor(0.0, device=self.device)
                disc_pred_map = None

                if batch_idx == 0:
                    first_batch_images = images
                    first_batch_synthetic_images = batch["synthetic_image"].to(self.device)
                    first_batch_synthetic_masks = batch["synthetic_mask"].to(self.device)

                if batch_idx == 0 and disc_pred_map is not None:
                    first_batch_disc_map = disc_pred_map.detach()

                with torch.amp.autocast(
                    "cuda",
                    enabled=bool(self.cfg.runtime.amp and self.device == "cuda"),
                ):
                    outputs = self.model(images)
                    loss = self.compute_loss(batch, outputs)

                    teacher_student_loss_value = torch.tensor(0.0, device=self.device)
                    if self.cfg.teacher_student.enabled:
                        teacher_student_loss_value = self.compute_teacher_student_loss(batch, outputs)
                        loss = loss + float(self.cfg.teacher_student.loss_weight) * teacher_student_loss_value

                    teacher_student_score = self.compute_teacher_student_score(batch, outputs)

                    feature_synth_score_value = torch.tensor(0.0, device=self.device)
                    feature_synth_ratio_value = torch.tensor(0.0, device=self.device)
                    feature_disc_loss_value = torch.tensor(0.0, device=self.device)
                    feature_disc_score_value = torch.tensor(0.0, device=self.device)

                    if self.cfg.feature_synthetic_anomalies.enabled:
                        target_level = str(self.cfg.feature_synthetic_anomalies.target_level)
                        clean_features = self.get_feature_tensor_by_level(outputs, target_level)

                        synthetic_features, synthetic_feature_labels = self.generate_feature_space_anomalies(clean_features)

                        if self.cfg.feature_discriminative.enabled:
                            feature_disc_loss_value, feature_disc_score_value = self.compute_feature_discriminative_loss(
                                clean_features=clean_features,
                                synthetic_features=synthetic_features,
                                synthetic_labels=synthetic_feature_labels,
                            )
                            loss = loss + float(self.cfg.feature_discriminative.weight) * feature_disc_loss_value

                        feature_delta = torch.mean(
                            torch.abs(synthetic_features - clean_features).reshape(clean_features.shape[0], -1),
                            dim=1,
                        )

                        if synthetic_feature_labels.sum() > 0:
                            feature_synth_score_value = feature_delta[synthetic_feature_labels > 0].mean().detach()
                        else:
                            feature_synth_score_value = torch.tensor(0.0, device=self.device)

                        feature_synth_ratio_value = synthetic_feature_labels.mean().detach()

                    if self.cfg.discriminative.enabled:
                        disc_total_loss, disc_map_loss_value, disc_image_loss_value, disc_pred_map = (
                            self.compute_discriminative_losses(
                                synthetic_image=synthetic_image,
                                synthetic_mask=synthetic_mask,
                                synthetic_label=synthetic_label,
                            )
                        )
                        loss = loss + disc_total_loss

                    loss = loss / self.cfg.training.grad_accum_steps
                    recon_total, recon_parts = self.reconstruction_loss_fn(outputs["reconstruction"], images)
                    recon_loss_value = recon_total.detach()
                    feat_disc_value = self.compute_feature_discrepancy_loss(outputs).detach()
                    l1_loss_value = recon_parts.get("l1_loss", torch.tensor(0.0, device=self.device)).detach()
                    ms_ssim_loss_value = recon_parts.get("ms_ssim_loss", torch.tensor(0.0, device=self.device)).detach()
                    mse_loss_value = recon_parts.get("mse_loss", torch.tensor(0.0, device=self.device)).detach()
                    perceptual_loss_value = self.compute_perceptual_loss(outputs).detach()
                    latent_compactness_value = self.compute_latent_compactness_loss(outputs).detach()
                    one_class_loss_value = self.compute_latent_one_class_loss(outputs).detach()
                    latent_score_value = self.compute_latent_one_class_score(outputs).mean().detach()
                    recon_score_value = self.compute_reconstruction_score(batch, outputs).mean().detach()
                    perceptual_score_value = self.compute_perceptual_score(outputs).mean().detach()
                    feature_score_value = self.compute_feature_discrepancy_score(outputs).mean().detach()
                    mining_outputs = self.run_latent_mining(outputs, epoch)
                    disc_score_value = torch.tensor(0.0, device=self.device)
                    real_disc_score_value = self.compute_discriminative_score(images, outputs["reconstruction"]).mean().detach()

                    if self.cfg.discriminative.enabled and disc_pred_map is not None:
                        disc_score_value = disc_pred_map.mean().detach()

                    if batch_idx == 0 and mining_outputs is not None:
                        first_batch_mining_outputs = mining_outputs

                    running_recon_loss += float(recon_loss_value.item())
                    running_feat_disc_loss += float(feat_disc_value.item())
                    running_l1_loss += float(l1_loss_value.item())
                    running_ms_ssim_loss += float(ms_ssim_loss_value.item())
                    running_mse_loss += float(mse_loss_value.item())
                    running_perceptual_loss += float(perceptual_loss_value.item())
                    running_latent_compactness_loss += float(latent_compactness_value.item())
                    running_one_class_loss += float(one_class_loss_value.item())
                    running_latent_score += float(latent_score_value.item())
                    running_recon_score += float(recon_score_value.item())
                    running_perceptual_score += float(perceptual_score_value.item())
                    running_feature_score += float(feature_score_value.item())
                    running_disc_total_loss += float(disc_total_loss.detach().item())
                    running_disc_map_loss += float(disc_map_loss_value.detach().item())
                    running_disc_image_loss += float(disc_image_loss_value.detach().item())
                    running_disc_score += float(disc_score_value.item())
                    running_real_disc_score += float(real_disc_score_value.item())
                    running_feature_synth_score += float(feature_synth_score_value.item())
                    running_feature_synth_ratio += float(feature_synth_ratio_value.item())
                    running_feature_disc_loss += float(feature_disc_loss_value.detach().item())
                    running_feature_disc_score += float(feature_disc_score_value.detach().item())
                    running_teacher_student_loss += float(teacher_student_loss_value.detach().item())
                    running_teacher_student_score += float(teacher_student_score.mean().detach().item())

                    if mining_outputs is not None:
                        mining_start_value = mining_outputs["score_start"].mean().detach()
                        mining_final_value = mining_outputs["score_final"].mean().detach()

                        running_mining_start_score += float(mining_start_value.item())
                        running_mining_final_score += float(mining_final_value.item())
                        num_mining_batches += 1
                    else:
                        mining_start_value = torch.tensor(0.0, device=self.device)
                        mining_final_value = torch.tensor(0.0, device=self.device)

                if self.cfg.latent_compactness.enabled:
                    self.latent_center.update(outputs["latent"])

                if self.cfg.runtime.detect_nan:
                    check_tensor_finite(images, "images")
                    for feat_name, feat_tensor in outputs.items():
                        check_tensor_finite(feat_tensor, f"outputs.{feat_name}")
                    check_tensor_finite(loss, "loss")

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.cfg.training.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)

                    if self.cfg.training.grad_clip_norm is not None:
                        clip_gradients(
                            self.model.parameters(),
                            float(self.cfg.training.grad_clip_norm),
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                    if self.ema is not None:
                        self.ema.update(self.model)

                    self._sanitize_bn_running_stats()

                if self.profiler is not None:
                    self.profiler.step()

                last_batch_shape = tuple(outputs["global"].shape)
                batch_size = int(images.shape[0])
                num_samples += batch_size
                running_loss += float(loss.item()) * self.cfg.training.grad_accum_steps
                num_steps += 1
                memory_score = self.compute_memory_score(outputs)

                if batch_idx == 0 and self.cfg.debug.save_batch_shapes:
                    print(
                        f"[Epoch {epoch}] first batch: "
                        f"input_shape={tuple(batch['image'].shape)} "
                        f"global_feat_shape={tuple(outputs['global'].shape)} "
                        f"latent_shape={tuple(outputs['latent'].shape)} "
                        f"recon_shape={tuple(outputs['reconstruction'].shape)} "
                        f"layer1_shape={tuple(outputs['layer1'].shape)} "
                        f"layer2_shape={tuple(outputs['layer2'].shape)} "
                        f"layer3_shape={tuple(outputs['layer3'].shape)} "
                        f"layer4_shape={tuple(outputs['layer4'].shape)} "
                        f"l1_loss={float(l1_loss_value.item()):.6f} "
                        f"ms_ssim_loss={float(ms_ssim_loss_value.item()):.6f} "
                        f"feat_disc_loss={float(feat_disc_value.item()):.6f} "
                        f"perceptual_loss={float(perceptual_loss_value.item()):.6f} "
                        f"latent_compactness_loss={float(latent_compactness_value.item()):.6f} "
                        f"one_class_loss={float(one_class_loss_value.item()):.6f} "
                        f"recon_score={float(recon_score_value.item()):.6f} "
                        f"perceptual_score={float(perceptual_score_value.item()):.6f} "
                        f"feature_score={float(feature_score_value.item()):.6f} "
                        f"latent_score={float(latent_score_value.item()):.6f} "
                        f"mining_start_score={float(mining_start_value.item()):.6f} "
                        f"mining_final_score={float(mining_final_value.item()):.6f} "
                        f"real_disc_score={float(real_disc_score_value.item()):.6f} "
                        f"disc_total_loss={float(disc_total_loss.detach().item()):.6f} "
                        f"disc_map_loss={float(disc_map_loss_value.detach().item()):.6f} "
                        f"disc_image_loss={float(disc_image_loss_value.detach().item()):.6f} "
                        f"disc_score={float(disc_score_value.item()):.6f} "
                        f"feature_synth_score={float(feature_synth_score_value.item()):.6f} "
                        f"feature_synth_ratio={float(feature_synth_ratio_value.item()):.6f} "
                        f"feature_disc_loss={float(feature_disc_loss_value.detach().item()):.6f} "
                        f"feature_disc_score={float(feature_disc_score_value.item()):.6f} "
                        f"memory_score={float(memory_score.mean().item()):.6f} "
                        f"teacher_student_loss={float(teacher_student_loss_value.detach().item()):.6f} "
                        f"teacher_student_score={float(teacher_student_score.mean().item()):.6f} "
                        f"total_loss={float(loss.item()) * self.cfg.training.grad_accum_steps:.6f}"
                    )

                if self.profiler is not None and batch_idx + 1 >= int(self.cfg.profiler.profile_batches):
                    break

        epoch_time = time.perf_counter() - epoch_start
        avg_loss = running_loss / max(num_steps, 1)
        metric_value = -avg_loss
        throughput = num_samples / max(epoch_time, 1e-8)
        vram_peak_mb = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
            if self.device == "cuda"
            else 0.0
        )

        metrics = {
            "avg_latent_compactness_loss": running_latent_compactness_loss / max(num_steps, 1),
            "avg_loss": avg_loss,
            "metric": metric_value,
            "epoch_time_sec": epoch_time,
            "throughput_samples_per_sec": throughput,
            "vram_peak_mb": vram_peak_mb,
            "num_steps": num_steps,
            "num_samples": num_samples,
            "last_batch_shape": str(last_batch_shape),
            "avg_l1_loss": running_l1_loss / max(num_steps, 1),
            "avg_ms_ssim_loss": running_ms_ssim_loss / max(num_steps, 1),
            "avg_mse_loss": running_mse_loss / max(num_steps, 1),
            "avg_recon_loss": running_recon_loss / max(num_steps, 1),
            "avg_feature_discrepancy_loss": running_feat_disc_loss / max(num_steps, 1),
            "avg_one_class_loss": running_one_class_loss / max(num_steps, 1),
            "avg_latent_score": running_latent_score / max(num_steps, 1),
            "avg_perceptual_loss": running_perceptual_loss / max(num_steps, 1),
            "avg_recon_score": running_recon_score / max(num_steps, 1),
            "avg_perceptual_score": running_perceptual_score / max(num_steps, 1),
            "avg_feature_score": running_feature_score / max(num_steps, 1),
            "avg_mining_start_score": running_mining_start_score / max(num_mining_batches, 1),
            "avg_mining_final_score": running_mining_final_score / max(num_mining_batches, 1),
            "avg_disc_total_loss": running_disc_total_loss / max(num_steps, 1),
            "avg_disc_map_loss": running_disc_map_loss / max(num_steps, 1),
            "avg_disc_image_loss": running_disc_image_loss / max(num_steps, 1),
            "avg_disc_score": running_disc_score / max(num_steps, 1),
            "avg_real_disc_score": running_real_disc_score / max(num_steps, 1),
            "avg_feature_synth_score": running_feature_synth_score / max(num_steps, 1),
            "avg_feature_synth_ratio": running_feature_synth_ratio / max(num_steps, 1),
            "avg_feature_disc_loss": running_feature_disc_loss / max(num_steps, 1),
            "avg_feature_disc_score": running_feature_disc_score / max(num_steps, 1),
            "avg_teacher_student_loss": running_teacher_student_loss / max(num_steps, 1),
            "avg_teacher_student_score": running_teacher_student_score / max(num_steps, 1),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }

        print(
            f"Epoch {epoch}/{self.cfg.training.epochs} - "
            f"lr={float(self.optimizer.param_groups[0]['lr']):.8f} - "
            f"avg_loss={avg_loss:.6f} - "
            f"avg_recon_loss={running_recon_loss / max(num_steps, 1):.6f} - "
            f"avg_l1_loss={running_l1_loss / max(num_steps, 1):.6f} - "
            f"avg_ms_ssim_loss={running_ms_ssim_loss / max(num_steps, 1):.6f} - "
            f"avg_feat_disc_loss={running_feat_disc_loss / max(num_steps, 1):.6f} - "
            f"avg_perceptual_loss={running_perceptual_loss / max(num_steps, 1):.6f} - "
            f"avg_latent_compactness_loss={running_latent_compactness_loss / max(num_steps, 1):.6f} - "
            f"avg_one_class_loss={running_one_class_loss / max(num_steps, 1):.6f} - "
            f"avg_recon_score={running_recon_score / max(num_steps, 1):.6f} - "
            f"avg_perceptual_score={running_perceptual_score / max(num_steps, 1):.6f} - "
            f"avg_feature_score={running_feature_score / max(num_steps, 1):.6f} - "
            f"avg_latent_score={running_latent_score / max(num_steps, 1):.6f} - "
            f"avg_mining_start_score={running_mining_start_score / max(num_mining_batches, 1):.6f} - "
            f"avg_mining_final_score={running_mining_final_score / max(num_mining_batches, 1):.6f} - "
            f"avg_disc_total_loss={running_disc_total_loss / max(num_steps, 1):.6f} - "
            f"avg_disc_map_loss={running_disc_map_loss / max(num_steps, 1):.6f} - "
            f"avg_disc_image_loss={running_disc_image_loss / max(num_steps, 1):.6f} - "
            f"avg_disc_score={running_disc_score / max(num_steps, 1):.6f} - "
            f"avg_real_disc_score={running_real_disc_score / max(num_steps, 1):.6f} - "
            f"metric={metric_value:.6f} - "
            f"time={epoch_time:.3f}s - "
            f"throughput={throughput:.2f} samples/s - "
            f"vram_peak_mb={vram_peak_mb:.2f} - "
            f"avg_feature_synth_score={running_feature_synth_score / max(num_steps, 1):.6f} - "
            f"avg_feature_synth_ratio={running_feature_synth_ratio / max(num_steps, 1):.6f} - "
            f"avg_feature_disc_loss={running_feature_disc_loss / max(num_steps, 1):.6f} - "
            f"avg_feature_disc_score={running_feature_disc_score / max(num_steps, 1):.6f} - "
            f"avg_teacher_student_loss={running_teacher_student_loss / max(num_steps, 1):.6f} - "
            f"avg_teacher_student_score={running_teacher_student_score / max(num_steps, 1):.6f}"
        )

        if self.cfg.logging.save_debug_images and first_batch_images is not None:
            save_debug_images(
                first_batch_images,
                self.run_dir / "debug_images",
                prefix=f"epoch_{epoch:03d}_input",
                max_images=int(self.cfg.logging.debug_image_count),
            )
            save_debug_images(
                outputs["reconstruction"],
                self.run_dir / "debug_images",
                prefix=f"epoch_{epoch:03d}_recon",
                max_images=int(self.cfg.logging.debug_image_count),
            )
        if self.cfg.logging.save_debug_images and first_batch_mining_outputs is not None:
            save_debug_images(
                first_batch_mining_outputs["hard_reconstruction"],
                self.run_dir / "debug_images",
                prefix=f"epoch_{epoch:03d}_hard_negative",
                max_images=int(self.cfg.logging.debug_image_count),
            )
        if self.cfg.logging.save_debug_images and first_batch_synthetic_images is not None:
            save_debug_images(
                first_batch_synthetic_images,
                self.run_dir / "debug_images",
                prefix=f"epoch_{epoch:03d}_synthetic",
                max_images=int(self.cfg.logging.debug_image_count),
            )
        if self.cfg.logging.save_debug_images and first_batch_disc_map is not None:
            disc_map_to_save = first_batch_disc_map.repeat(1, 3, 1, 1)
            save_debug_images(
                disc_map_to_save,
                self.run_dir / "debug_images",
                prefix=f"epoch_{epoch:03d}_disc_map",
                max_images=int(self.cfg.logging.debug_image_count),
            )

        return metric_value

    def train(self) -> None:
        print("=== Training loop ===")
        self.maybe_resume()

        selected_threshold = None

        for epoch in range(self.start_epoch + 1, self.cfg.training.epochs + 1):
            # Propagate epoch to dataset so per-item synthetic anomaly RNG changes
            # deterministically across epochs (see datasets/mvtec_*.py).
            train_ds = getattr(self.train_loader, "dataset", None)
            if train_ds is not None and hasattr(train_ds, "set_epoch"):
                train_ds.set_epoch(epoch)
            train_metric = self.train_one_epoch(epoch)

            self.fit_score_normalization()
            self.fit_learned_score_fusion()

            self._current_epoch_for_eval = epoch
            val_metrics = self.evaluate_loader(self.val_mixed_loader, split_name="val_mixed")
            selected_threshold = val_metrics["val_mixed_best_threshold"]

            selection_score = compute_selection_score(self.cfg, val_metrics)

            print(
                f"[Validation] "
                f"AUROC={val_metrics['val_mixed_auroc']:.4f} "
                f"AUPRC={val_metrics['val_mixed_auprc']:.4f} "
                f"best_F1={val_metrics['val_mixed_best_f1']:.4f} "
                f"best_threshold={val_metrics['val_mixed_best_threshold']:.6f} "
                f"FPR@95TPR={val_metrics['val_mixed_fpr_at_95_tpr']:.4f} "
                f"selection_score={selection_score:.4f}"
            )

            merged_metrics = {
                "train_metric": train_metric,
                "val_selection_score": selection_score,
                **val_metrics,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
            }

            self.log_epoch_metrics(epoch, merged_metrics)

            prev_best = self.best_metric
            self.save_last_checkpoint(epoch, train_metric)
            self.save_best_checkpoint(epoch, train_metric, monitor_value=selection_score)

            if self.best_metric != prev_best:
                self.best_val_threshold = val_metrics["val_mixed_best_threshold"]

            should_stop = self.update_early_stopping(selection_score)

            if should_stop:
                break

            if self.scheduler is not None:
                self.scheduler.step()
                print(f"[Scheduler] epoch={epoch} next_lr={float(self.optimizer.param_groups[0]['lr']):.8f}")

        if selected_threshold is not None:
            final_threshold = (
                self.best_val_threshold
                if self.best_val_threshold is not None
                else selected_threshold
            )

            with open(self.run_dir / "selected_threshold.txt", "w") as f:
                f.write(f"split=val_mixed\n")
                f.write(f"threshold={final_threshold}\n")

            best_ckpt = self.run_dir / "best_checkpoint.pt"
            if best_ckpt.exists():
                self.load_checkpoint(str(best_ckpt))
                print(f"[Final] loaded best checkpoint from {best_ckpt}")

            test_metrics = self.evaluate_loader(
                self.test_loader,
                split_name="test_blind",
                threshold=final_threshold,
            )

            print(
                f"[Test] "
                f"AUROC={test_metrics['test_blind_auroc']:.4f} "
                f"AUPRC={test_metrics['test_blind_auprc']:.4f} "
                f"best_F1={test_metrics['test_blind_best_f1']:.4f} "
                f"F1@val_threshold={test_metrics['test_blind_f1_at_given_threshold']:.4f} "
                f"FPR@95TPR={test_metrics['test_blind_fpr_at_95_tpr']:.4f}"
            )

            self.logger.log_metrics({"epoch": "final_test", **test_metrics})

            if bool(self.cfg.analysis.save_failure_analysis):
                self.dump_failure_cases(
                    self.test_loader,
                    split_name="test_blind",
                    threshold=final_threshold,
                    max_per_group=8,
                )

            print("Training finished.")

    def inspect_eval_loaders(self) -> None:
        for loader_name in ["val_normal_loader", "val_mixed_loader", "test_loader"]:
            loader = getattr(self, loader_name)
            batch = next(iter(loader))
            anomaly_sum = int(batch["label"].sum().item())

            print(
                f"[{loader_name}] "
                f"batch_shape={tuple(batch['image'].shape)} "
                f"anomaly_count={anomaly_sum} "
                f"split={batch['split'][0]}"
            )

    def _collect_small_sample(self, loader, max_items: int = 16) -> list[dict[str, Any]]:
        samples = []
        for batch in loader:
            batch_size = batch["image"].shape[0]
            for i in range(batch_size):
                sample = {
                    "image": batch["image"][i].detach().cpu(),
                    "label": int(batch["label"][i]),
                    "is_anomaly": int(batch["label"][i]),
                    "split": batch["split"][i],
                }
                samples.append(sample)
                if len(samples) >= max_items:
                    return samples
        return samples

    def run_anti_leakage_audit(self) -> None:
        print("=== Anti-leakage audit ===")

        split_samples = {
            "train_normal": self._collect_small_sample(self.train_loader, max_items=16),
            "val_normal": self._collect_small_sample(self.val_normal_loader, max_items=16),
            "val_mixed": self._collect_small_sample(self.val_mixed_loader, max_items=16),
            "test_blind": self._collect_small_sample(self.test_loader, max_items=16),
        }

        for split_name, samples in split_samples.items():
            summary = summarize_labels(samples)
            print(f"[{split_name}] summary: {summary}")

            if self.cfg.anti_leakage.exact_dedup:
                duplicates = find_exact_duplicates(samples)
                print(f"[{split_name}] exact_duplicates={len(duplicates)}")

        if self.cfg.anti_leakage.contamination_check:
            overlaps = audit_split_contamination(split_samples)
            print(f"[cross-split overlaps] {overlaps}")

        if self.cfg.normalization.enabled and self.cfg.normalization.mode == "train_only_stats":
            train_mean, train_std = compute_channel_stats(split_samples["train_normal"])
            print(f"[train-only normalization stats] mean={train_mean}, std={train_std}")
        elif self.cfg.normalization.enabled and self.cfg.normalization.mode == "imagenet":
            print(
                f"[normalization] mode=imagenet "
                f"mean={list(self.cfg.normalization.mean)} "
                f"std={list(self.cfg.normalization.std)}"
            )
        else:
            print("[normalization] disabled or mode=none")
