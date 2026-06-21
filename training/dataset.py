from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset


import random
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path

class ShardDataset(IterableDataset):
    def __init__(self, shard_dir: Path, shard_names: list[str]):
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.shard_paths = [self.shard_dir / name for name in shard_names]
        
        # Hardcode or estimate this so __init__ doesn't touch the disk
        # 300 shards * 80,000 positions
        self.total_positions = len(self.shard_paths) * 80000 

    def __len__(self):
        return self.total_positions

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # 1. Split shard list so workers never touch the same file
        if worker_info is None:
            my_shards = self.shard_paths
        else:
            my_shards = [s for i, s in enumerate(self.shard_paths) 
                         if i % worker_info.num_workers == worker_info.id]
        
        random.shuffle(my_shards)

        for path in my_shards:
            # 2. LOAD ONE SHARD
            # We use 'with' so the file handle and memory are released after the shard is exhausted
            with np.load(path) as data:
                # We pull them into local variables. 
                # This is the ONLY time RAM should increase.
                features = data["features"]
                from_sq = data["from_sq"]
                plane = data["plane"]
                z = data["z"]
                z_eval = data["z_eval"]
                has_eval = data["has_eval"]
                
                n = len(z)
                indices = np.arange(n)
                np.random.shuffle(indices)

                # 3. USE ALL DATA IN SHARD
                for idx in indices:
                    yield {
                        "features": torch.from_numpy(features[idx].astype(np.float32)),
                        "from_sq": int(from_sq[idx]),
                        "plane": int(plane[idx]),
                        "z": torch.tensor(z[idx], dtype=torch.float32),
                        "z_eval": torch.tensor(z_eval[idx], dtype=torch.float32),
                        "has_eval": bool(has_eval[idx]),
                    }
            
            # 4. SHARD IS DROPPED HERE
            # Once we exit the 'with' block, the local variables are eligible for Garbage Collection.
            # RAM should drop back down before the next 'path' in 'my_shards' is loaded.

    def _get_sample(self, shard, local):
        # Single-slice access is fast with mmap
        return {
            "features": torch.from_numpy(shard["features"][local].astype(np.float32)),
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
