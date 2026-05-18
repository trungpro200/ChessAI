from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    data_dir: Path = Path("data")
    pgn_path: Path = Path("high_quality_games_2026-01.pgn")
    shard_dir: Path = Path("data/shards")
    checkpoint_dir: Path = Path("checkpoints")

    shard_size: int = 320_000
    min_plies: int = 8
    val_fraction: float = 0.01

    batch_size: int = 512
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    w_policy: float = 1.0
    w_value: float = 1.0
    w_eval: float = 0.25
    eval_cp_scale: float = 4.0

    num_workers: int = 0
    preprocess_workers: int = 8
    amp: bool = True

    log_every: int = 100
    val_every: int = 1000
    checkpoint_every: int = 5000
    logdir: Path | None = None

    seed: int = 42

    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"
