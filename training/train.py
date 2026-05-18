"""Supervised training. Run: python -m training.train --data data"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from Model.chess_model import ChessModel
from Model.device import device
from training.checkpoint import load_checkpoint, save_checkpoint
from training.config import TrainConfig
from training.dataset import build_datasets
from training.losses import compute_loss


def _collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "from_sq": torch.tensor([b["from_sq"] for b in batch], dtype=torch.long),
        "plane": torch.tensor([b["plane"] for b in batch], dtype=torch.long),
        "z": torch.stack([b["z"] for b in batch]),
        "z_eval": torch.stack([b["z_eval"] for b in batch]),
        "has_eval": torch.tensor([b["has_eval"] for b in batch], dtype=torch.bool),
    }


def _run_epoch(
    model: ChessModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    cfg: TrainConfig,
    train: bool = True,
) -> dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "eval_loss": 0.0, "policy_acc": 0.0}
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            if scaler is not None and train:
                with torch.amp.autocast("cuda"):
                    policy, value = model(batch["features"])
                    loss, metrics = compute_loss(
                        policy,
                        value,
                        batch,
                        w_policy=cfg.w_policy,
                        w_value=cfg.w_value,
                        w_eval=cfg.w_eval,
                    )
            else:
                policy, value = model(batch["features"])
                loss, metrics = compute_loss(
                    policy,
                    value,
                    batch,
                    w_policy=cfg.w_policy,
                    w_value=cfg.w_value,
                    w_eval=cfg.w_eval,
                )

        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

        for k in totals:
            totals[k] += metrics.get(k, 0.0)
        n_batches += 1

    if n_batches == 0:
        return totals
    return {k: v / n_batches for k, v in totals.items()}


def train(cfg: TrainConfig, resume: Path | None = None) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_ds, val_ds = build_datasets(cfg.data_dir)
    print(f"Train positions: {len(train_ds):,}, val: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=_collate,
        pin_memory=device == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=_collate,
        pin_memory=device == "cuda",
    )

    model = ChessModel()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.epochs * len(train_loader))
    )

    step = 0
    epoch_start = 0
    best_val_acc = 0.0

    if resume is not None and resume.exists():
        ckpt = load_checkpoint(resume, model, optimizer, map_location=device)
        step = ckpt.get("step", 0)
        epoch_start = ckpt.get("epoch", 0)
        print(f"Resumed from {resume} at step {step}")

    use_amp = cfg.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epoch_start, cfg.epochs):
        t0 = time.time()
        train_metrics = _run_epoch(model, train_loader, optimizer, scaler, cfg, train=True)
        scheduler.step()
        val_metrics = _run_epoch(model, val_loader, optimizer, scaler, cfg, train=False)
        step += len(train_loader)

        print(
            f"Epoch {epoch + 1}/{cfg.epochs} ({time.time() - t0:.0f}s) | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['policy_acc']:.3f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['policy_acc']:.3f}"
        )

        save_checkpoint(
            cfg.checkpoint_dir / "latest.pt",
            model,
            optimizer,
            step=step,
            epoch=epoch + 1,
            metrics={"train": train_metrics, "val": val_metrics},
        )

        if val_metrics["policy_acc"] >= best_val_acc:
            best_val_acc = val_metrics["policy_acc"]
            save_checkpoint(
                cfg.checkpoint_dir / "best.pt",
                model,
                optimizer,
                step=step,
                epoch=epoch + 1,
                metrics={"train": train_metrics, "val": val_metrics},
            )
            print(f"  New best val policy acc: {best_val_acc:.4f}")


def main() -> None:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="Train ChessModel on PGN shards")
    parser.add_argument("--data", type=Path, default=cfg.data_dir)
    parser.add_argument("--checkpoint-dir", type=Path, default=cfg.checkpoint_dir)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--w-eval", type=float, default=cfg.w_eval)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    cfg.data_dir = args.data
    cfg.checkpoint_dir = args.checkpoint_dir
    cfg.batch_size = args.batch_size
    cfg.epochs = args.epochs
    cfg.lr = args.lr
    cfg.w_eval = args.w_eval
    cfg.amp = not args.no_amp

    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
