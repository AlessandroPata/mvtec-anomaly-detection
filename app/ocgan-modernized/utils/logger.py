from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


class RunLogger:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.epoch_summary_path = self.run_dir / "epoch_summary.yaml"

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

    def save_epoch_summary(self, summary: dict[str, Any]) -> None:
        OmegaConf.save(config=OmegaConf.create(summary), f=str(self.epoch_summary_path))
