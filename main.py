import bulletchess as chess
import Model
import torch
import timeit 
from Model.device import device

board = chess.Board.from_fen('rnbqkbnr/1pppp1pp/p7/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3')
model = Model.ChessModel(token_dim=103)

# 1.601748300017789s
# print(Model.encode_board(board))

e = Model.encode_board_init(board).to(device).unsqueeze(0)
res = model(e)

print(res)
print(res[0].shape)
