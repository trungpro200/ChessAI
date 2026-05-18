from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ShardDataset(Dataset):
    """Index over npz shards; loads one shard at a time to limit RAM."""

    def __init__(self, shard_dir: Path, shard_names: list[str]) -> None:
        self.shard_dir = Path(shard_dir)
        self.shard_paths = [self.shard_dir / name for name in shard_names]
        self.cumsum: list[int] = [0]
        for path in self.shard_paths:
            with np.load(path) as data:
                n = len(data["z"])
            self.cumsum.append(self.cumsum[-1] + n)

        self._cache_idx = -1
        self._cache: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return self.cumsum[-1]

    def _locate(self, index: int) -> tuple[int, int]:
        lo, hi = 0, len(self.shard_paths) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.cumsum[mid] <= index:
                lo = mid
            else:
                hi = mid - 1
        return lo, index - self.cumsum[lo]

    def _get_shard(self, shard_idx: int) -> dict[str, np.ndarray]:
        if shard_idx != self._cache_idx:
            self._cache = dict(np.load(self.shard_paths[shard_idx]))
            self._cache_idx = shard_idx
        assert self._cache is not None
        return self._cache

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard_idx, local = self._locate(index)
        shard = self._get_shard(shard_idx)
        features = torch.from_numpy(shard["features"][local].astype(np.float32))
        return {
            "features": features,
            "from_sq": int(shard["from_sq"][local]),
            "plane": int(shard["plane"][local]),
            "z": torch.tensor(shard["z"][local], dtype=torch.float32),
            "z_eval": torch.tensor(shard["z_eval"][local], dtype=torch.float32),
            "has_eval": bool(shard["has_eval"][local]),
        }


def load_manifest(data_dir: Path) -> dict:
    with open(data_dir / "manifest.json", encoding="utf-8") as f:
        return json.load(f)


def build_datasets(data_dir: Path) -> tuple[ShardDataset, ShardDataset]:
    manifest = load_manifest(data_dir)
    shard_dir = data_dir / "shards"
    train = ShardDataset(shard_dir, manifest["train_shards"])
    val = ShardDataset(shard_dir, manifest["val_shards"])
    return train, val
