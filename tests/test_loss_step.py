import tempfile
from pathlib import Path

import numpy as np
import torch

from Model.chess_model import ChessModel
from Model.device import device
from training.dataset import ShardDataset
from training.losses import compute_loss


def test_loss_backward():
    model = ChessModel()
    model.train()
    b = 4
    x = torch.randn(b, 64, 103, device=device)
    batch = {
        "from_sq": torch.randint(0, 64, (b,), device=device),
        "plane": torch.randint(0, 73, (b,), device=device),
        "z": torch.randn(b, device=device),
        "z_eval": torch.randn(b, device=device),
        "has_eval": torch.tensor([True, False, True, False], device=device),
    }
    policy, value = model(x)
    loss, metrics = compute_loss(policy, value, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert "policy_acc" in metrics


def test_shard_dataset_roundtrip():
    n = 16
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "train_shard_00000.npz"
        np.savez(
            path,
            features=np.random.randn(n, 64, 103).astype(np.float16),
            from_sq=np.zeros(n, dtype=np.uint8),
            plane=np.zeros(n, dtype=np.uint8),
            z=np.zeros(n, dtype=np.float32),
            z_eval=np.full(n, np.nan, dtype=np.float32),
            has_eval=np.zeros(n, dtype=bool),
        )
        ds = ShardDataset(Path(tmp), [path.name])
        item = ds[0]
        assert item["features"].shape == (64, 103)

if __name__ == "__main__":
    test_loss_backward()