"""PGN → sharded npz datasets. Run: python -m training.preprocess"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from tqdm import tqdm

from training.config import TrainConfig
from training.encode_game import EncodedSample, encode_game
from training.pgn_parser import game_split_key, is_val_split, stream_games


def _samples_to_arrays(samples: list[EncodedSample]) -> dict[str, np.ndarray]:
    return {
        "features": np.stack([s.features for s in samples]).astype(np.float16),
        "from_sq": np.array([s.from_sq for s in samples], dtype=np.uint8),
        "plane": np.array([s.plane for s in samples], dtype=np.uint8),
        "z": np.array([s.z for s in samples], dtype=np.float32),
        "z_eval": np.array([s.z_eval for s in samples], dtype=np.float32),
        "has_eval": np.array([s.has_eval for s in samples], dtype=bool),
    }


def _process_game(
    game,
    game_index: int,
    val_fraction: float,
    eval_cp_scale: float,
    min_plies: int,
) -> tuple[str, list[EncodedSample]]:
    key = game_split_key(game.headers, game_index)
    split = "val" if is_val_split(key, val_fraction) else "train"
    samples = encode_game(game, eval_cp_scale=eval_cp_scale, min_plies=min_plies)
    return split, samples


class ShardWriter:
    def __init__(self, out_dir: Path, shard_size: int, split: str) -> None:
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.split = split
        self.buffer: list[EncodedSample] = []
        self.shard_index = 0
        self.paths: list[str] = []
        self.total = 0

    def add(self, samples: list[EncodedSample]) -> None:
        if not samples:
            return
        self.buffer.extend(samples)
        while len(self.buffer) >= self.shard_size:
            self._write_chunk(self.buffer[: self.shard_size])
            self.buffer = self.buffer[self.shard_size :]

    def flush(self) -> None:
        if self.buffer:
            self._write_chunk(self.buffer)
            self.buffer = []

    def _write_chunk(self, chunk: list[EncodedSample]) -> None:
        arr = _samples_to_arrays(chunk)
        path = self.out_dir / f"{self.split}_shard_{self.shard_index:05d}.npz"
        np.savez_compressed(path, **arr)
        self.paths.append(path.name)
        self.total += len(chunk)
        self.shard_index += 1


def _iter_game_chunks(
    pgn_path: Path, max_games: int | None, chunk_size: int = 1000
):
    chunk: list[tuple] = []
    with open(pgn_path, encoding="utf-8") as f:
        for game, game_index in stream_games(f):
            if max_games is not None and game_index >= max_games:
                break
            chunk.append((game, game_index))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

class PreprocessTask:
    def __init__(self, val_fraction: float, eval_cp_scale: float, min_plies: int):
        self.val_fraction = val_fraction
        self.eval_cp_scale = eval_cp_scale
        self.min_plies = min_plies

    def __call__(self, item: tuple) -> tuple[str, list[EncodedSample]]:
        game, game_index = item
        return _process_game(
            game, game_index, self.val_fraction, self.eval_cp_scale, self.min_plies
        )

def preprocess(
    pgn_path: Path,
    out_dir: Path,
    shard_size: int = 80_000,
    min_plies: int = 8,
    val_fraction: float = 0.01,
    workers: int = 8,
    eval_cp_scale: float = 4.0,
    max_games: int | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        "train": ShardWriter(out_dir, shard_size, "train"),
        "val": ShardWriter(out_dir, shard_size, "val"),
    }
    games_skipped = 0
    games_processed = 0

    # Instantiate the picklable task class here
    _task = PreprocessTask(val_fraction, eval_cp_scale, min_plies)
    

    chunks = _iter_game_chunks(pgn_path, max_games)
    pool: mp.pool.Pool | None = None
    if workers > 1:
        pool = mp.Pool(workers)

    pbar = tqdm(desc="games", unit="game")
    try:
        for games_chunk in chunks:
            if pool is not None:
                for split, samples in pool.imap_unordered(_task, games_chunk, chunksize=16):
                    games_processed += 1
                    pbar.update(1)
                    if not samples:
                        games_skipped += 1
                    else:
                        writers[split].add(samples)
            else:
                for item in games_chunk:
                    split, samples = _task(item)
                    games_processed += 1
                    pbar.update(1)
                    if not samples:
                        games_skipped += 1
                    else:
                        writers[split].add(samples)
    finally:
        pbar.close()
        if pool is not None:
            pool.close()
            pool.join()

    for w in writers.values():
        w.flush()

    manifest = {
        "pgn": str(pgn_path),
        "shard_size": shard_size,
        "min_plies": min_plies,
        "val_fraction": val_fraction,
        "eval_cp_scale": eval_cp_scale,
        "games_processed": games_processed,
        "games_skipped": games_skipped,
        "train_shards": writers["train"].paths,
        "val_shards": writers["val"].paths,
        "train_positions": writers["train"].total,
        "val_positions": writers["val"].total,
    }

    manifest_path = out_dir.parent / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    print(f"Wrote manifest to {manifest_path}")
    print(f"Train: {writers['train'].total:,} positions, {len(writers['train'].paths)} shards")
    print(f"Val: {writers['val'].total:,} positions, {len(writers['val'].paths)} shards")


def main() -> None:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="Preprocess PGN into training shards")
    parser.add_argument("--pgn", type=Path, default=cfg.pgn_path)
    parser.add_argument("--out", type=Path, default=cfg.shard_dir)
    parser.add_argument("--shard-size", type=int, default=cfg.shard_size)
    parser.add_argument("--min-plies", type=int, default=cfg.min_plies)
    parser.add_argument("--val-fraction", type=float, default=cfg.val_fraction)
    parser.add_argument("--workers", type=int, default=cfg.preprocess_workers)
    parser.add_argument("--eval-cp-scale", type=float, default=cfg.eval_cp_scale)
    parser.add_argument("--max-games", type=int, default=None)
    args = parser.parse_args()

    preprocess(
        pgn_path=args.pgn,
        out_dir=args.out,
        shard_size=args.shard_size,
        min_plies=args.min_plies,
        val_fraction=args.val_fraction,
        workers=args.workers,
        eval_cp_scale=args.eval_cp_scale,
        max_games=args.max_games,
    )


if __name__ == "__main__":
    main()
