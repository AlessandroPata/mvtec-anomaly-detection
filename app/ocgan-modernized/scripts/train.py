from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from trainers.base_trainer import BaseTrainer
from utils.env import get_env_info, save_env_info
from utils.repro import set_seed


def make_run_dir(cfg: DictConfig) -> Path:
    if cfg.training.resume and cfg.training.resume_path is not None:
        resume_path = Path(cfg.training.resume_path).resolve()
        return resume_path.parent

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.project.experiment_name}_seed{cfg.project.seed}_{timestamp}"
    base_dir = Path(cfg.project.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    run_dir = base_dir / cfg.project.name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    if cfg.project.device == "cuda" and not torch.cuda.is_available():
        print("CUDA non disponibile, passo a CPU.")
        cfg.project.device = "cpu"

    set_seed(cfg.project.seed, deterministic=cfg.project.deterministic)

    run_dir = make_run_dir(cfg)

    print("=== Loaded config ===")
    print(OmegaConf.to_yaml(cfg))
    print(f"Run directory: {run_dir}")

    if cfg.logging.save_config and not (cfg.training.resume and cfg.training.resume_path is not None):
        OmegaConf.save(cfg, run_dir / "config.yaml")

    if cfg.logging.save_env_info and not (cfg.training.resume and cfg.training.resume_path is not None):
        env_info = get_env_info()
        save_env_info(env_info, run_dir)

    trainer = BaseTrainer(cfg=cfg, run_dir=str(run_dir))
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
