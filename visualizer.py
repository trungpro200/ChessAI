import torch
from Model.chess_model import ChessModel, verify_model, TARGET_PARAMS, MAX_PARAMS

md = ChessModel()
info = verify_model(md)
print(f"Total parameters: {info['params']:,}")
print(f"Budget: {TARGET_PARAMS:,} (legacy) .. {MAX_PARAMS:,} (max)")
print(f"Policy shape: {info['policy_shape']}, Value shape: {info['value_shape']}")
assert info["within_budget"], f"Parameter count {info['params']} exceeds {MAX_PARAMS}"
print(md)
