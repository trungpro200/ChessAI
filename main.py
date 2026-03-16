import bulletchess as chess
import Model
import torch
import timeit 
from Model.device import device

board = chess.Board.from_fen('rnbqkbnr/1pppp1pp/p7/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3')
model = Model.ChessModel(token_dim=103)

# 1.601748300017789s
# print(Model.encode_board(board))

e = Model.encode_board_init(board)
d = torch.Tensor(e)

def test():
    d.copy_(e)
    Model.encode_board_propagate(d, board, (1,1))
    # print(d[:, 0:12])
    # print(d[:, 12:24])

torch.set_printoptions(profile='full')
test()

# print(e[:, 12:24])

timer = timeit.Timer(test)

t = timer.timeit(10000)

print(t)
