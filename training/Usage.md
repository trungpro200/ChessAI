# 1) Preprocess full PGN (~4–8 GB shards)
`python -m training.preprocess --pgn high_quality_games_2026-01.pgn --out data/shards --workers 8`

# 2) Train
`python -m training.train --data data --epochs 1 --batch-size 512`

# 3) Inference server with weights
`python server.py --checkpoint checkpoints/best.pt`