import bulletchess as chess
import Model
import torch
import timeit 
from Model.device import device
import copy
from collections import deque


board = chess.Board.from_fen("rnbqkbnr/pppp1ppp/8/4p3/5PP1/8/PPPPP2P/RNBQKBNR b KQkq f3 0 1")
model = Model.ChessModel(token_dim=103)

buffer = deque()

# 1.601748300017789s -> 0.357 -> 0.5
# print(Model.encode_board(board))
state = Model.State(board)
moves = [
    Model.encode_move(chess.Move.from_uci('g1f3'), board),
    Model.encode_move(chess.Move.from_uci('g8f6'), board),
    Model.encode_move(chess.Move.from_uci('f3g1'), board),
    Model.encode_move(chess.Move.from_uci('f6g8'), board),
]

sp = Model.SelfPlay(model, 1)
# sp.play_game(tokens)

# print(Model.GLOBAL_BUFFER)

sp.run_mcts(state)
# sp.simulate(state, state.board.__hash__(), deque())
# print(Model.encode_move(chess.Move.from_uci('d8h4'), state.board))
print(sp.sample_move(sp.TT[board.__hash__()]))