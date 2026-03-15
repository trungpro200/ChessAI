import bulletchess as chess
import Model
import torch
import timeit 

board = chess.Board()

bit = board[chess.Piece(chess.WHITE, chess.PAWN)]

#0.3784495000145398s
# print(Model.encode_board(board))

model = Model.ChessModel()

test = torch.ones(1,64,101, device='cuda')
v = model(test)

print(v)