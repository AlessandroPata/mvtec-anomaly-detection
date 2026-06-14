"""Live inference for the retired OCGAN models, rebuilt from training checkpoints.

The GAN was always scored through the BaseTrainer pipeline: memory bank built
from train features, MAD score normalization fitted on val_normal, and a
logistic-regression fusion fitted on val_mixed. The best_checkpoint.pt embeds
the fitted normalization stats, the pickled fusion model and the val_mixed
best-F1 threshold the original run used for its final test eval — so loading
restores them verbatim instead of refitting. The only state rebuilt at load is
the memory bank (frozen-backbone features of train images, seeded like the
original run), because it was never serialized.

First load per category costs well under a minute on GPU; the server keeps a
single GAN model resident (they hold a full backbone + teacher + decoder).

Checkpoint sources:
  ocgan_final  -> production_models/{cat}/model.pt (best seed of the final
                  GAN config, collected by collect_production_models.sh)
  ocgan_optv2  -> ../outputs/ocgan-modernized/{cat}_optv2_s43_seed43_*/best_checkpoint.pt
                  (seed 43 of the optv2 multiseed eval)
"""
from __future__ import annotations

import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class RawLogRegFusion:
    """Version-independent applier for the GAN's 7-head logistic score fusion.

    The retired-GAN checkpoints pickle a fitted sklearn ``LogisticRegression``.
    Unpickling it couples inference to the exact sklearn build it was trained
    with — it raises ``InconsistentVersionWarning`` across releases and can
    silently drift if the internal layout changes. We keep only the learned
    weights and reimplement the binary ``predict_proba`` as a plain
    ``sigmoid(x·coef + intercept)`` in NumPy, so the fusion is exactly
    reproducible regardless of the installed sklearn. Drop-in for the single
    ``estimator.predict_proba(x)[:, 1]`` call in BaseTrainer.
    """

    __slots__ = ("coef_", "intercept_")

    def __init__(self, coef, intercept):
        self.coef_ = np.asarray(coef, dtype=np.float64).reshape(-1)
        self.intercept_ = float(np.asarray(intercept, dtype=np.float64).reshape(-1)[0])

    @classmethod
    def from_estimator(cls, est) -> "RawLogRegFusion":
        return cls(est.coef_, est.intercept_)

    def predict_proba(self, x):
        x = np.asarray(x, dtype=np.float64)
        z = x @ self.coef_ + self.intercept_
        p1 = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
        return np.stack([1.0 - p1, p1], axis=1)


def _as_raw_fusion(fusion):
    """Convert a pickled sklearn estimator to the NumPy applier; pass through
    None or an already-raw applier unchanged."""
    if fusion is None or isinstance(fusion, RawLogRegFusion):
        return fusion
    if hasattr(fusion, "coef_") and hasattr(fusion, "intercept_"):
        return RawLogRegFusion.from_estimator(fusion)
    return fusion

ROOT = Path(__file__).resolve().parent.parent
STORAGE_ROOT = ROOT.parent.parent
GAN_OUTPUTS_DIR = STORAGE_ROOT / "outputs" / "ocgan-modernized"
DATASET_ROOT = STORAGE_ROOT / "datasets" / "mvtec_ad"
CACHE_DIR = ROOT / ".gan_cache"

SCORE_COMPONENT_KEYS = (
    "norm_recon_score", "norm_perceptual_score", "norm_feature_score",
    "norm_memory_score", "norm_teacher_student_score",
)


@dataclass(frozen=True)
class GanSpec:
    id: str
    label: str
    kind: str
    description: str
    recalibrate: bool = False


GAN_SPECS: dict[str, GanSpec] = {
    "ocgan_final": GanSpec(
        "ocgan_final", "OCGAN final — live checkpoint", "gan",
        "The retired GAN (7 fused scoring heads), best seed per category. "
        "First run per category rebuilds bank + fusion (~1 min).",
    ),
    "ocgan_optv2": GanSpec(
        "ocgan_optv2", "OCGAN optv2 — live checkpoint", "gan",
        "Optimized GAN retrain, seed 43 of the multiseed eval. Original "
        "weights; score calibration refit on the validation splits at first "
        "load (the archived calibration came from a training-era code state "
        "and fp16 numerics that don't reproduce here).",
        recalibrate=True,
    ),
}


def load_run_config(cfg_path: Path):
    """The archived config.yaml files contain the config dumped twice
    (train.py saved it, then the resume path appended it again). Keep the
    first copy: cut at the second top-level `project:` line."""
    from omegaconf import OmegaConf

    text = cfg_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.rstrip("\r\n") == "project:"]
    if len(starts) > 1:
        text = "".join(lines[: starts[1]])
    return OmegaConf.create(text)


def gan_checkpoint_paths(variant_id: str, category: str) -> tuple[Path, Path] | None:
    """Return (checkpoint, config) for a category, or None if not on disk."""
    if variant_id == "ocgan_final":
        ckpt = ROOT / "production_models" / category / "model.pt"
        cfg = ROOT / "production_models" / category / "config.yaml"
        return (ckpt, cfg) if ckpt.exists() and cfg.exists() else None
    if variant_id == "ocgan_optv2":
        if not GAN_OUTPUTS_DIR.exists():
            return None
        runs = sorted(GAN_OUTPUTS_DIR.glob(f"{category}_optv2_s43_seed43_*"))
        for run in reversed(runs):  # latest run that actually saved a checkpoint
            ckpt = run / "best_checkpoint.pt"
            cfg = run / "config.yaml"
            if ckpt.exists() and cfg.exists():
                return ckpt, cfg
        return None
    return None


def gan_variant_entries(category: str) -> list[dict]:
    """Meta entries shaped like patchcore_variants.available_variants rows."""
    out = []
    for spec in GAN_SPECS.values():
        out.append({
            **asdict(spec),
            "aggregation": None, "topk": None, "coreset": None,
            "available": gan_checkpoint_paths(spec.id, category) is not None,
            "approximate": False,
        })
    return out


class GanInference:
    """PatchCoreInference-compatible wrapper around a BaseTrainer in eval mode."""

    def __init__(self, category: str, variant_id: str, device: str = "cpu"):
        from datasets.build import build_transform
        from trainers.base_trainer import BaseTrainer
        from utils.repro import set_seed

        paths = gan_checkpoint_paths(variant_id, category)
        if paths is None:
            raise FileNotFoundError(f"No {variant_id} checkpoint for {category}")
        ckpt_path, cfg_path = paths

        with warnings.catch_warnings():
            # legacy checkpoints embed a sklearn-pickled fusion model; we convert
            # it to RawLogRegFusion right after, so its version warning is noise.
            try:
                from sklearn.exceptions import InconsistentVersionWarning
                warnings.simplefilter("ignore", InconsistentVersionWarning)
            except Exception:
                pass
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

        cfg = load_run_config(cfg_path)
        cfg.project.device = device
        cfg.dataset.root = str(DATASET_ROOT)
        # The optv2 configs say use_skip_connections=true but their checkpoints
        # hold a plain BaseReconstructor (the builder ignored the flag at the
        # time). Trust the weights, not the config.
        cfg.model.reconstruction.use_skip_connections = any(
            k.startswith("reconstructor.fuse") for k in ckpt["model_state_dict"]
        )
        for split in ("train_normal", "val_normal", "val_mixed", "test_blind"):
            split_cfg = getattr(cfg.dataset, split)
            split_cfg.num_workers = 0  # in-process: no worker spawn on Windows
            split_cfg.pin_memory = False
        # The original runs used amp, but the optv2 backbones (87 unfrozen
        # params) overflow fp16 on this GPU: NaN activations from layer2 on,
        # nan_to_num then flattens every reconstruction score. fp32 is cheap
        # enough at batch size 1.
        cfg.runtime.amp = False
        cfg.runtime.detect_nan = False
        cfg.logging.save_debug_images = False
        cfg.logging.save_score_histograms = False
        cfg.logging.use_wandb = False
        cfg.analysis.save_failure_analysis = False

        set_seed(int(cfg.project.seed), deterministic=False)
        run_dir = CACHE_DIR / f"{variant_id}_{category}"
        run_dir.mkdir(parents=True, exist_ok=True)

        trainer = BaseTrainer(cfg=cfg, run_dir=str(run_dir))
        trainer.setup()

        # Weights + calibration only — trainer.load_checkpoint() also restores
        # the optimizer state, whose param groups no longer match in eval mode.
        trainer.model.load_state_dict(ckpt["model_state_dict"])
        if trainer.discriminative_head is not None and ckpt.get("discriminative_head_state_dict") is not None:
            trainer.discriminative_head.load_state_dict(ckpt["discriminative_head_state_dict"])
        if trainer.feature_discriminator is not None and ckpt.get("feature_discriminator_state_dict") is not None:
            trainer.feature_discriminator.load_state_dict(ckpt["feature_discriminator_state_dict"])
        trainer.score_norm_stats = ckpt.get("score_norm_stats", {})
        # Apply the fusion through a version-independent NumPy applier instead of
        # the sklearn estimator pickled in the checkpoint (see RawLogRegFusion).
        trainer.learned_score_fusion = _as_raw_fusion(ckpt.get("learned_score_fusion", None))
        trainer.learned_score_fusion_feature_names = ckpt.get("learned_score_fusion_feature_names", [])
        if ckpt.get("latent_center_center") is not None and trainer.latent_center is not None:
            trainer.latent_center.center.data.copy_(ckpt["latent_center_center"].to(device))

        threshold = ckpt.get("best_val_threshold")
        if GAN_SPECS[variant_id].recalibrate or threshold is None:
            # Refit normalization + fusion + threshold on the val splits,
            # exactly like train.py did before its final test eval.
            trainer.fit_score_normalization()
            trainer.fit_learned_score_fusion()
            val_metrics = trainer.evaluate_loader(trainer.val_mixed_loader, split_name="val_mixed")
            threshold = val_metrics["val_mixed_best_threshold"]

        self.trainer = trainer
        self.threshold = float(threshold)
        self.transform = build_transform(cfg, "test_blind")
        self.is_probability = trainer.learned_score_fusion is not None
        self.amp = bool(cfg.runtime.amp)
        self.device = device
        self.category = category
        self.variant_id = variant_id

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        t0 = time.perf_counter()
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)
        x = self.transform(x).unsqueeze(0).to(self.device)
        batch = {"image": x, "label": torch.zeros(1, dtype=torch.long)}

        self.trainer.model.eval()
        with torch.cuda.amp.autocast(enabled=self.amp):
            outputs = self.trainer.model(x)
            components = self.trainer.compute_score_components(batch, outputs)

        score = float(components["final_score"][0].detach().cpu().item())
        recon = torch.nan_to_num(outputs["reconstruction"]).float()
        err = (x.float() - recon).abs().mean(dim=1)[0].detach().cpu().numpy()
        lo, hi = float(err.min()), float(err.max())
        heatmap = (err - lo) / (hi - lo) if hi > lo else np.zeros_like(err)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "anomaly_score": score,
            "anomaly_probability": float(np.clip(score, 0.0, 1.0)) if self.is_probability else None,
            "is_anomalous": bool(score >= self.threshold),
            "threshold": self.threshold,
            "category": self.category,
            "inference_time_ms": elapsed_ms,
            "heatmap": heatmap,
            "score_components": {
                k: float(components[k][0].detach().cpu().item())
                for k in SCORE_COMPONENT_KEYS if k in components
            },
        }
