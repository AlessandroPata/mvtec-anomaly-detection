from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_smoke_train_runs() -> None:
    project_root = Path(__file__).resolve().parents[1]

    cmd = [
        sys.executable,
        "-m",
        "scripts.train",
        "training.epochs=1",
        "dataset.train_normal.length=8",
        "dataset.train_normal.batch_size=4",
        "dataset.train_normal.num_workers=0",
        "dataset.val_normal.length=8",
        "dataset.val_normal.batch_size=4",
        "dataset.val_normal.num_workers=0",
        "dataset.val_mixed.length=8",
        "dataset.val_mixed.batch_size=4",
        "dataset.val_mixed.num_workers=0",
        "dataset.test_blind.length=8",
        "dataset.test_blind.batch_size=4",
        "dataset.test_blind.num_workers=0",
        "project.experiment_name=smoke_test",
        "runtime.amp=false",
        "ema.enabled=false",
        "debug.save_batch_shapes=false",
    ]

    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"Smoke train failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "Training finished." in result.stdout
