from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


def get_env_info() -> dict:
    info = {
        "python_version": str(sys.version).replace("\n", " "),
        "platform": str(platform.platform()),
        "hostname": str(socket.gethostname()),
        "cwd": str(os.getcwd()),
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": str(torch.backends.cudnn.version()),
        "gpu_count": int(torch.cuda.device_count()),
        "gpu_names": [],
    }

    if torch.cuda.is_available():
        info["gpu_names"] = [
            str(torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())
        ]

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        info["git_commit"] = result.stdout.strip()
    except Exception:
        info["git_commit"] = None

    return info


def save_env_info(env_info: dict, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_path = output_dir / "env_info.yaml"
    OmegaConf.save(config=OmegaConf.create(env_info), f=str(env_path))
