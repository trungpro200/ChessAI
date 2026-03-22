import bulletchess as chess
import Model
import torch
import timeit 
from Model.device import device
import copy
from collections import deque


board = chess.Board()
model = Model.ChessModel(token_dim=103)

buffer = deque()

# 1.601748300017789s -> 0.357 -> 0.5
# print(Model.encode_board(board))
tokens = Model.State(board)
moves = [
    Model.encode_move(chess.Move.from_uci('g1f3')),
    Model.encode_move(chess.Move.from_uci('g8f6')),
    Model.encode_move(chess.Move.from_uci('f3g1')),
    Model.encode_move(chess.Move.from_uci('f6g8')),
]

sp = Model.SelfPlay(model)
sp.play_game(tokens)

print(Model.GLOBAL_BUFFER)

# tokens = tokens.tokens.unsqueeze(0).to('cuda')