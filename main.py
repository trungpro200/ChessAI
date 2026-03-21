import bulletchess as chess
import Model
import torch
import timeit 
from Model.device import device
import copy

board = chess.Board()
model = Model.ChessModel(token_dim=103)

# 1.601748300017789s -> 0.357 -> 0.5
# print(Model.encode_board(board))
tokens = Model.Tokens(board)

moves = [
    Model.encode_move(chess.Move.from_uci('g1f3')),
    Model.encode_move(chess.Move.from_uci('g8f6')),
    Model.encode_move(chess.Move.from_uci('f3g1')),
    Model.encode_move(chess.Move.from_uci('f6g8')),
]


def history_test():
    for move in moves:
        tokens.encode_board_propagate(move)

# timer = timeit.Timer(history_test)

# t = timer.timeit(2500)
# print(t)

history_test()
# history_test()

# for plane in tokens.history_planes:
#     print(plane[:, 7].view(8,8))

torch.set_printoptions(profile='full')
print(tokens.tokens)
# print(tokens.tokens[:, 0:96])
# print(e[:, 12:24])
