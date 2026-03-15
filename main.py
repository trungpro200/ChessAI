import bulletchess as chess
import Model
import torch
import timeit 

board = chess.Board()

bit = board[chess.Piece(chess.WHITE, chess.PAWN)]

#0.3784495000145398s
# print(Model.encode_board(board))

model = Model.ShawRelativeAttention()

test = torch.ones(1,64,112, device='cuda')
v = model(test)


board.__hash__()