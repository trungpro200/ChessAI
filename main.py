import bulletchess as chess
import Model
import torch
import timeit 

board = chess.Board()

bit = board[chess.Piece(chess.WHITE, chess.PAWN)]

#0.3784495000145398s
# print(Model.encode_board(board))
t = timeit.Timer(lambda: Model.encode_board(board))

print(t.timeit(10000))