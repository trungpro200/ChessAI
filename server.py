import argparse
import warnings

import torch
import zmq

from Model import ChessModel
from Model.device import device
from training.checkpoint import load_checkpoint

PORT = 3636


def parse_args():
    parser = argparse.ArgumentParser(description="ZMQ inference server for ChessModel")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pt checkpoint (model weights)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(f"tcp://*:{PORT}")
    socket.setsockopt(zmq.RCVTIMEO, 1000)

    model = ChessModel()
    model.eval()

    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location=device)
        print(f"Loaded checkpoint: {args.checkpoint}")

    print(f"Python server online on port {PORT} (device={device})")

    try:
        while True:
            try:
                frames = socket.recv_multipart()
            except zmq.Again:
                continue

            batch_size = int.from_bytes(frames[0], "little")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                batch = (
                    torch.frombuffer(frames[1], dtype=torch.float32)
                    .view(-1, 64, 103)
                    .to(device=device)
                )

            with torch.no_grad():
                policy, value = model(batch)

            _ = batch_size, policy.shape, value.shape
            socket.send(b"Ok")
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        socket.close()


if __name__ == "__main__":
    main()
