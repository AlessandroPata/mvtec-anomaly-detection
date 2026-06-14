from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class BaseAnomalyDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Any = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError