import bulletchess as chess
import Model
import torch
import timeit 
from Model.device import device
import copy
from collections import deque


board = chess.Board()
model = Model.ChessModel()

buffer = deque()
print(board.pretty())
# 1.601748300017789s -> 0.357 -> 0.5
# print(Model.encode_board(board))
state = Model.State(board)

sp = Model.SelfPlay(model, batch_size=32)

sp.play_game(state)
print(Model.GLOBAL_BUFFER)

# moves = board.legal_moves()
# score = {}
# for move in moves:
#     score[move] = Model.mvv_lva_score(board, move)

# moves.sort(key=lambda x: score[x], reverse=True)

# for move in moves:
#     print(f"{move.san(board)} - {score[move]}")